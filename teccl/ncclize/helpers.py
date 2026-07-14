"""Shared helpers for inspecting a parsed TE-CCL schedule.

Two concerns live here, both built from the per-epoch flow lists that
teccl_ncclize.parse_flows() already produces (so they add no new parsing):

  * a per-GPU, per-epoch human-readable view of the schedule, and
  * a realizability feasibility check over that view.

They are kept in separate functions on purpose. check_epoch_ordering_
feasibility() is cheap and side-effect-free, so it can always be run; writing
the human-readable dump (write_gpu_epoch_debug) is optional and independent.
See the cross-GPU realizability gap note: a send whose GPU has no op (send or
recv arrival) in the immediately preceding epoch has nothing for ncclize /
enforce_send_epoch_ordering to gate it against, so it fires early on real
hardware. Recvs are counted at their arrival epoch, since a send can be gated
behind a recv that lands in the previous epoch.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, TextIO, Tuple


@dataclass(frozen=True)
class EpochStep:
    """One transfer a single GPU takes part in during one epoch.

    peer is the other endpoint; is_send is True when this GPU is the source
    (this GPU -> peer) and False when it is the destination (peer -> this GPU).

    A recv is bucketed at the epoch the chunk *arrives* (its completion epoch),
    which for a switch-relayed flow is later than start_epoch (the epoch the
    flow was sent). start_epoch is None for sends.
    """
    chunk_id: int
    peer: int
    is_send: bool
    start_epoch: Optional[int] = None

    def render(self, epoch: Optional[int] = None) -> str:
        if self.is_send:
            return f'send c{self.chunk_id} -> g{self.peer}'
        text = f'recv c{self.chunk_id} <- g{self.peer}'
        # Flag a recv that was in flight across epochs (arrived after it was
        # sent), so a reader can see why an otherwise send-free epoch is active.
        if (epoch is not None and self.start_epoch is not None
                and self.start_epoch != epoch):
            text += f' (sent epoch {self.start_epoch})'
        return text


@dataclass
class GpuEpochView:
    """Per-GPU, per-epoch schedule, indexed by raw solver epoch number.

    epochs is the *contiguous* range of raw epoch numbers [min..max], where max
    covers the latest recv arrival, which can be later than any flow-start
    epoch. An epoch with no send *and* no recv arrival still appears explicitly,
    as NONE for every GPU -- which matters for the feasibility check, since such
    an epoch is exactly a place the generated schedule loses an epoch of pacing.
    """
    num_nodes: int
    epochs: List[int]
    per_gpu: Dict[int, Dict[int, List[EpochStep]]]  # gpu -> epoch -> steps

    def steps(self, gpu: int, epoch: int) -> List[EpochStep]:
        return self.per_gpu[gpu].get(epoch, [])

    def has_send(self, gpu: int, epoch: int) -> bool:
        return any(s.is_send for s in self.steps(gpu, epoch))


@dataclass(frozen=True)
class OrderingViolation:
    """A send that cannot be paced to its intended epoch on real hardware."""
    gpu: int
    epoch: int
    prev_epoch: int
    sends: Tuple[EpochStep, ...]


def build_gpu_epoch_view(steps_in_order, sorted_epochs, num_nodes,
                         flow_completion_epochs=None) -> GpuEpochView:
    """Turn parse_flows() output into a per-GPU, per-epoch view.

    steps_in_order[i] is the flow list for raw epoch sorted_epochs[i]; each
    flow is a (chunk_id, src, dst) tuple in the dense 0-indexed GPU numbering.

    A send is placed at its flow-start epoch (when the source initiates it). A
    recv is placed at its *arrival* epoch, looked up in flow_completion_epochs
    (keyed (step_idx, chunk_id, src, dst)); for a switch-relayed flow this is
    later than the start epoch. Placing recvs at arrival is what lets the
    feasibility check treat a send as realizable when the same GPU has a recv
    landing in the immediately preceding epoch to gate it against (ncclize can
    add that dependency). When a flow is absent from flow_completion_epochs the
    start epoch is used as a safe fallback.
    """
    flow_completion_epochs = flow_completion_epochs or {}

    per_gpu: Dict[int, Dict[int, List[EpochStep]]] = {
        g: {} for g in range(num_nodes)}
    max_epoch = sorted_epochs[-1] if sorted_epochs else None
    for step_idx, (epoch, sends) in enumerate(zip(sorted_epochs, steps_in_order)):
        for chunk_id, src, dst in sends:
            per_gpu[src].setdefault(epoch, []).append(
                EpochStep(chunk_id, dst, is_send=True))
            recv_epoch = flow_completion_epochs.get(
                (step_idx, chunk_id, src, dst), epoch)
            per_gpu[dst].setdefault(recv_epoch, []).append(
                EpochStep(chunk_id, src, is_send=False, start_epoch=epoch))
            max_epoch = max(max_epoch, recv_epoch)

    if sorted_epochs:
        epochs = list(range(sorted_epochs[0], max_epoch + 1))
    else:
        epochs = []

    # Deterministic ordering within each epoch: sends first, then by peer/chunk.
    for gpu_epochs in per_gpu.values():
        for step_list in gpu_epochs.values():
            step_list.sort(key=lambda s: (not s.is_send, s.peer, s.chunk_id))

    return GpuEpochView(num_nodes, epochs, per_gpu)


def check_epoch_ordering_feasibility(view: GpuEpochView) -> List[OrderingViolation]:
    """Flag every send with no same-GPU predecessor to be paced against.

    For each GPU, a send in epoch N is only realizable at its intended start
    time if that GPU also has an op in epoch N-1 to gate it against -- either a
    send, or a recv arriving in N-1 (ncclize can make the send depend on that
    recv). Recvs are placed at their arrival epoch by build_gpu_epoch_view
    precisely so this check sees them. A send whose immediately-preceding epoch
    is idle (NONE) for that GPU has no such gate and will instead fire as early
    as the runtime allows, so it is reported here (see
    enforce_send_epoch_ordering and the cross-GPU realizability gap).

    The first epoch in the range is never a violation: own-chunk sends there
    legitimately start at t=0.
    """
    violations: List[OrderingViolation] = []
    for gpu in range(view.num_nodes):
        for i, epoch in enumerate(view.epochs):
            if i == 0:
                continue
            sends = tuple(s for s in view.steps(gpu, epoch) if s.is_send)
            if not sends:
                continue
            prev_epoch = view.epochs[i - 1]
            if not view.steps(gpu, prev_epoch):
                violations.append(
                    OrderingViolation(gpu, epoch, prev_epoch, sends))
    return violations


def warn_epoch_ordering_violations(
        violations: List[OrderingViolation],
        stream: Optional[TextIO] = None) -> None:
    """Print a warning for every feasibility violation (no-op if none)."""
    if stream is None:
        stream = sys.stderr
    if not violations:
        return
    print(f'WARNING: {len(violations)} send(s) in this schedule cannot be '
          f'realized as-is -- each starts in an epoch whose immediately '
          f'preceding epoch has no send and no recv arrival for the same GPU, '
          f'so the generated MSCCL schedule has no same-GPU op to gate it '
          f'against and it will fire early on real hardware:', file=stream)
    for v in violations:
        detail = ', '.join(s.render() for s in v.sends)
        print(f'  - GPU {v.gpu}: epoch {v.epoch} [{detail}] '
              f'preceded by NONE at epoch {v.prev_epoch}', file=stream)


def format_gpu_epoch_view(
        view: GpuEpochView,
        violations: Optional[List[OrderingViolation]] = None,
        source: Optional[str] = None) -> str:
    """Render the per-GPU epoch schedule as human-readable text.

    Sends flagged by check_epoch_ordering_feasibility() are marked inline with
    [!]; a trailing summary lists every violation. Pass the violations you
    already computed (so the check runs once); pass None to render without any
    feasibility annotations.
    """
    flagged = {(v.gpu, v.epoch): v for v in (violations or [])}
    lines: List[str] = []
    lines.append('# Per-GPU epoch schedule (human-readable debug view)')
    if source:
        lines.append(f'# Source: {source}')
    if view.epochs:
        lines.append(f'# GPUs: {view.num_nodes}   '
                     f'Epochs: {view.epochs[0]}..{view.epochs[-1]} '
                     f'(raw solver epoch numbers)')
    else:
        lines.append(f'# GPUs: {view.num_nodes}   Epochs: (none)')
    lines.append('# Epochs with no send and no recv arrival are shown as NONE.')
    lines.append('# Steps: "send cC -> gD" (this GPU sends chunk C to GPU D),')
    lines.append('#        "recv cC <- gS" (chunk C from GPU S *arrives* here;')
    lines.append('#        a relayed recv is shown at its arrival epoch, tagged')
    lines.append('#        "(sent epoch N)" when it was in flight from earlier).')
    lines.append('# [!] marks a send whose immediately-preceding epoch is NONE')
    lines.append('#     for this GPU: no same-GPU op (send or recv arrival) exists')
    lines.append('#     to gate it, so it cannot be held to its intended epoch on')
    lines.append('#     real hardware.')
    lines.append('')

    for gpu in range(view.num_nodes):
        lines.append(f'GPU {gpu}:')
        for epoch in view.epochs:
            steps = view.steps(gpu, epoch)
            if not steps:
                lines.append(f'  epoch {epoch}: NONE')
                continue
            body = '; '.join(s.render(epoch) for s in steps)
            v = flagged.get((gpu, epoch))
            marker = (f'   [!] no pacing predecessor '
                      f'(epoch {v.prev_epoch} is NONE)') if v else ''
            lines.append(f'  epoch {epoch}: {body}{marker}')
        lines.append('')

    n = len(violations) if violations is not None else 0
    lines.append(f'## Feasibility check: {n} violation(s)')
    if violations:
        for v in violations:
            detail = ', '.join(s.render() for s in v.sends)
            lines.append(f'  - GPU {v.gpu}, epoch {v.epoch} [{detail}] '
                         f'preceded by NONE at epoch {v.prev_epoch}')
    elif violations is not None:
        lines.append('  (schedule is realizable under the same-GPU pacing model)')
    lines.append('')
    return '\n'.join(lines)


def write_gpu_epoch_debug(
        path: str,
        view: GpuEpochView,
        violations: Optional[List[OrderingViolation]] = None,
        source: Optional[str] = None) -> None:
    """Write the human-readable per-GPU epoch schedule to path."""
    with open(path, 'w') as f:
        f.write(format_gpu_epoch_view(view, violations, source))
