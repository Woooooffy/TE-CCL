"""
The FINAL post-process: the assembled hierarchical solution -> ONE flat schedule on the fine
topology. Runs exactly once, after the recursion has returned.

Inputs are the two halves of the ROOT level:
  * `IdentityResolution.pieces` -- inter-cell flows, already pinned to real fine GPUs and fine
    switch ids, carrying the root level's own pacing (`ResolvedPiece.rate`);
  * the `IntraFlow`s of everything beneath it, which `teccl.hierarchy.flatten.rebase` has ALREADY
    folded into the root's `(band, local_round)` -- one fold per level on the way up, so however
    deep the recursion went, what arrives here is two-level.

Output is the `flow_str_info` dict every solver emits, so `teccl/ncclize/teccl_ncclize.py` consumes
it unchanged and MSCCL XML comes out the far end.


THIS IS ONLY A FLATTENING
-------------------------
Each solver step owns its own fine schedule; this module translates the root's onto one ABSOLUTE
axis and does nothing else. There is no placement policy here, no dependency analysis, no
re-derivation of an ordering some level already decided. The per-layer half of the job --
each level's time grid and the fold that produced these flows -- lives in
`teccl.hierarchy.flatten`; the two were one module only while the solver had two levels, at which
point they were indistinguishable.

delta and m are the ROOT level's grid, from `flatten.derive_grid`, so a coarse epoch is
exactly m fine epochs. The intra level emits `(band, local_round)`, where band is the coarse epoch a
transfer runs concurrently with and local_round is a fine-epoch offset -- the units already line up,
because a round is "one chunk across one NVSwitch port" = bytes_per_chunk / nvlink_bw = delta. So
the whole translation is:

    W = width of the prologue band (its round count; 0 if empty)

    prologue (band -1)     fine epoch = local_round                 in [0, W)
    band k >= 0            fine epoch = W + m*k + local_round
    network send, epoch k  fine epoch = W + m*k                     (the band's leading edge)
    piece held at dst      fine epoch = W + m*(arrival_epoch + 1)   (end of its arrival epoch)

Band k is CONCURRENT WITH coarse epoch k, not a seam between epochs: the uplink and the NVSwitch are
different links, so intra-cell work hides under network time instead of serializing against it.
(Measured, charging intra work as a gap between coarse epochs costs 0.2306 s / 0.8311 s for
allgather / alltoall versus ~0.20 / ~0.80 for the concurrent reading -- it turns a win into a loss
against a 0.22 s flat baseline.)

Bands 0..K-1 are exactly m fine epochs wide because their leading edges are the network sends, which
are Delta apart by construction. `rounds <= m` per (cell, band) is therefore a real feasibility
certificate -- it says the intra work genuinely fits inside the coarse epoch it hides under -- and it
is asserted, not assumed. The PROLOGUE and the EPILOGUE (band K, fanning out the final arrivals) have
no network send to align to, so they are as wide as their own schedules need, and they are the only
intra work charged directly to the makespan. That is the honest accounting: they are the two bands
with no network time to hide under.

Why the round index is used directly rather than collapsed to a precedence depth: `_schedule_band`
already places a tree child in a strictly later round than its parent, so the round index ALREADY
carries the dependency order. Re-deriving a "level" from the flows would recompute that, and would
additionally discard the port-contention the scheduler resolved -- understating prologue and epilogue
time by the ratio of rounds to depth (5-6x on these schedules).
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from teccl.hierarchy.crossbar_solve import PROLOGUE_BAND, IntraFlow, rounds_in
from teccl.hierarchy.flatten import aligned_band, assert_bands_fit, derive_grid
from teccl.hierarchy.reconstruct import Identity, IdentityResolution
from teccl.hierarchy.scale import ChunkScale
from teccl.topologies.topology import Topology

# Record phases. These are ANNOTATION only -- placement is decided by the level that produced the
# flow, and the stitch reads (band, local_round). NETWORK is the one the code branches on, because
# only inter-cell sends carry a rate and occupy a coarse-epoch-aligned link.
PROLOGUE = "prologue"          # ran before coarse epoch 0's first network send
NETWORK = "network"            # the inter-cell send itself
INGRESS = "ingress"            # fallback label for an intra flow with no recorded kind


@dataclass(frozen=True)
class DeliveryRecord:
    """One physical GPU->GPU transfer on the fine topology, at an absolute fine epoch.

    A record is emitted MERGED (`via_switches` annotates the route) rather than as separate
    GPU->switch and switch->GPU hops: the ncclize parsers derive the GPU id universe from the set
    of ids appearing as flow endpoints, so a switch id appearing there would be mistaken for a GPU.
    """
    identity: Identity                 # sub-identity (s, ci) at the REFINED scale
    sender: int
    receiver: int
    via_switches: Tuple[int, ...]
    volume: float
    epoch: int                         # absolute fine epoch the transfer starts in
    completion: int                    # absolute fine epoch by which the receiver holds it
    rate: Optional[float]              # GB/s, or None for a deliberately unpaced flow
    phase: str
    cell: Optional[int] = None         # provenance, for the auxiliary annotation only
    level: int = 0


# ----------------------------------------------------------------------------------------------
# S0. Grid and preconditions
# ----------------------------------------------------------------------------------------------
def assert_native_ownership(res: IdentityResolution) -> None:
    """Every staged identity must be staged BY ITS OWN SOURCE.

    This is the host-transit guard. A coarse path that store-and-forwards through an intermediate
    CELL would need a chained ingress->egress relay whose deadline is a downstream send, which
    identity resolution does not model (it files each leg by physical endpoints, and the transit
    cell does not want the data, so the identity has no piece there). It fails loud upstream; this
    keeps it failing loud here rather than silently emitting a staging relay from a GPU that never
    receives the data.
    """
    bad = [(d.cell, d.identity, d.src_gpu) for d in res.intra_demands
           if d.kind == "egress_stage" and d.src_gpu != d.identity[0]]
    if bad:
        raise AssertionError(
            f"{len(bad)} egress_stage demands are staged by a GPU that is not the identity's "
            f"source {bad[:5]} -- that is host transit (a coarse path relayed through an "
            f"intermediate cell), which is unmodelled.")


# ----------------------------------------------------------------------------------------------
# Lay the assembled schedule on the absolute fine-epoch axis
# ----------------------------------------------------------------------------------------------
def prologue_width(intra_flows: Sequence[IntraFlow]) -> int:
    """How many fine epochs the pre-epoch-0 band needs.

    Cells schedule independently on their own NVSwitch, so the prologue is as long as the busiest
    cell's prologue, not the sum. This is the one intra band whose length is charged directly to
    the makespan (it has no network send to hide under), which is why it is measured in ROUNDS --
    the schedule the level actually produced -- rather than estimated from a dependency depth.
    """
    return rounds_in([f for f in intra_flows if f.band == PROLOGUE_BAND])


def build_records(res: IdentityResolution, intra_flows: Sequence[IntraFlow],
                  m: int) -> Tuple[List[DeliveryRecord], int]:
    """Place every inter-cell piece and intra-cell flow on the absolute fine epoch axis.

    Pure translation -- see the module docstring for the four-line mapping. Returns the records and
    the prologue width W.
    """
    W = prologue_width(intra_flows)

    records: List[DeliveryRecord] = []
    for f in intra_flows:
        epoch = f.local_round if f.band == PROLOGUE_BAND else W + m * f.band + f.local_round
        # One record per sub-chunk the transfer carries, ALL AT THE SAME EPOCH. A coalesced flow is
        # one physical transfer of contiguous bytes, so its sub-chunks have consecutive labels and
        # ncclize merges them back into a single cnt=Q operation -- which it can only do for
        # addresses that share a step.
        for identity in f.identities:
            records.append(DeliveryRecord(
                identity=identity, sender=f.sender, receiver=f.receiver,
                # A hop over a DIRECT link has no switch to name (a physical ring cell), and
                # `_segment` omits the "via switches" clause for an empty tuple. `(None,)` would
                # instead emit the literal text "via switches None", which ncclize's
                # PATH_SEGMENT_RE would then fail to parse.
                via_switches=() if f.via_switch is None else (f.via_switch,),
                volume=f.volume / len(f.identities), epoch=epoch,
                # The transfer occupies its whole span and the receiver holds it once it finishes,
                # which is what lets a flow ending at round r feed one starting at r+1 -- the
                # precedence _schedule_band already enforced.
                completion=epoch + f.span, rate=None,
                phase=PROLOGUE if f.band == PROLOGUE_BAND else (f.kind or INGRESS),
                cell=f.cell, level=f.local_round))

    for p in res.pieces:
        records.append(DeliveryRecord(
            identity=p.identity, sender=p.egress_gpu, receiver=p.ingress_gpu,
            via_switches=tuple(p.via_switches), volume=p.volume,
            epoch=W + m * p.send_epoch,
            # The piece is held at the END of its arrival epoch, i.e. the leading edge of the next
            # band -- which is where the fan-out that consumes it was scheduled.
            completion=W + m * (p.arrival_epoch + 1), rate=p.rate, phase=NETWORK,
            cell=p.src_cell))

    return records, W


# ----------------------------------------------------------------------------------------------
# S4. Back-trace: chunk paths, causality, coverage
# ----------------------------------------------------------------------------------------------
def back_trace(records: Sequence[DeliveryRecord], fine_demand, subdivision: int
               ) -> Dict[Tuple[int, Identity], List[DeliveryRecord]]:
    """For every fine demand, the chain of records that delivers it, verified as it is built.

    The emitter and the verifier are one traversal on purpose: the path list serialized into
    "8-Chunk paths" IS the evidence that each demand is met, so a demand that cannot be traced back
    to its source is a hard error rather than a missing line in the output.
    """
    # (identity, gpu) -> earliest epoch that gpu holds it
    holds: Dict[Tuple[Identity, int], int] = {}
    # (identity, gpu) -> the record that got it there earliest
    via: Dict[Tuple[Identity, int], DeliveryRecord] = {}
    for r in sorted(records, key=lambda r: (r.completion, r.epoch)):
        key = (r.identity, r.receiver)
        if key not in holds or r.completion < holds[key]:
            holds[key] = r.completion
            via[key] = r
    # A source owns its own identities from the start.
    sources = {(r.identity, r.identity[0]) for r in records}
    for key in sources:
        holds[key] = 0

    paths: Dict[Tuple[int, Identity], List[DeliveryRecord]] = {}
    n = len(fine_demand)
    chunks = len(fine_demand[0][0]) if n else 0
    missing: List[Tuple] = []
    for s in range(n):
        for t in range(n):
            for ci in range(chunks):
                if not fine_demand[s][t][ci] or s == t:
                    continue
                for j in range(subdivision):
                    sub = (s, ci * subdivision + j)
                    chain: List[DeliveryRecord] = []
                    cur = t
                    seen = set()
                    while cur != s:
                        rec = via.get((sub, cur))
                        if rec is None:
                            missing.append((t, sub, cur))
                            break
                        if (sub, cur) in seen:
                            raise AssertionError(
                                f"cycle while tracing demand ({t}, {sub}) back to its source")
                        seen.add((sub, cur))
                        # Causality: the sender must already hold the data when it sends.
                        held = holds.get((sub, rec.sender))
                        if held is None or held > rec.epoch:
                            raise AssertionError(
                                f"causality violation delivering {sub} to {t}: {rec.sender} sends "
                                f"in fine epoch {rec.epoch} but holds the data only from "
                                f"{held!r} ({rec.phase}, via {rec.via_switches})")
                        chain.append(rec)
                        cur = rec.sender
                    else:
                        paths[(t, sub)] = list(reversed(chain))
    if missing:
        raise AssertionError(
            f"{len(missing)} demanded (dest, sub-identity) pairs are never delivered "
            f"[(dest, identity, stuck at)]: {missing[:6]}")
    return paths


# ----------------------------------------------------------------------------------------------
# S5. Serialize
# ----------------------------------------------------------------------------------------------
def _segment(r: DeliveryRecord) -> str:
    """One "8-Chunk paths" / "7-Flows" hop, in the grammar teccl_ncclize's PATH_SEGMENT_RE reads.

    The rate token is emitted only when the producing level paced the flow. Its absence is
    meaningful, not a default: an intra-cell hop carries ordering but is deliberately not pinned to
    the epoch grid, so throttling it would be inventing a constraint the solve never imposed.
    """
    out = f"{r.sender}->{r.receiver} with volume {r.volume:g} in epoch {r.epoch}"
    if r.rate is not None:
        out += f" at rate {r.rate:g}"
    if r.via_switches:
        out += " via switches " + "->".join(str(x) for x in r.via_switches)
    return out


def _chunk_label(ci: int, subdivision: int, dst_major: bool, num_gpus: int,
                 dst_dense: int) -> int:
    """The refined sub-chunk index, re-expressed in the COLLECTIVE's own chunk addressing.

    Sub-chunk refinement is collective-agnostic on purpose -- it only subdivides bytes, so it names
    the j-th piece of chunk ci as `ci * Q + j` (reconstruct._emit_refined). But the label written
    into the schedule is read back by ncclize through the collective's addressing convention, and
    for a DST-MAJOR collective that convention already packs the destination into the low digits:
    demand.py lays alltoall down as `chunk = dst_dense + c * num_gpus`, and ncclize decodes
    `chunk % num_gpus` as the destination. Naively refining that index to `chunk * Q + j` moves the
    destination into a different digit and the decode yields a different GPU.

    So the refinement has to be applied to the SUB-CHUNK component, keeping the destination where
    the convention expects it:

        src-major (allgather/gather/broadcast):  label = ci * Q + j     (already the refined index)
        dst-major (alltoall):                    label = dst + (c * Q + j) * num_gpus

    This belongs here rather than in the refinement itself: the recursion layer's job is to name
    bytes consistently, and the label convention is a property of the output format.
    """
    if not dst_major:
        return ci
    original, j = divmod(ci, subdivision)
    dst, c = original % num_gpus, original // num_gpus
    if dst != dst_dense:
        raise AssertionError(
            f"alltoall chunk {original} encodes destination {dst} but the demand is at dense rank "
            f"{dst_dense}; the demand array and the label convention disagree.")
    return dst + (c * subdivision + j) * num_gpus


def serialize(records: Sequence[DeliveryRecord],
              paths: Dict[Tuple[int, Identity], List[DeliveryRecord]],
              scale: ChunkScale, delta: float, collective: str,
              num_sources: int, subdivision: int = 1) -> Dict:
    """Build the flow_str_info dict, in the nested (LP) format.

    Chunk labels are PER-SOURCE: a demand key names the source separately ("for chunk {ci} from
    {s}"), and ncclize composes the global id itself as src_dense * num_subchunks + ci. Writing a
    pre-composed global id here would inflate num_subchunks by the GPU count and check_implements
    would reject the schedule.
    """
    # Dense ranks exactly as ncclize derives them: the sorted set of ids appearing as a flow
    # endpoint is the GPU universe (switch ids only ever appear in a "via switches" annotation).
    gpus = sorted({r.sender for r in records} | {r.receiver for r in records})
    dense = {g: i for i, g in enumerate(gpus)}
    dst_major = collective == "alltoall"

    label_of: Dict[Identity, int] = {}

    def label(identity: Identity, dst: int) -> int:
        s, ci = identity
        out = _chunk_label(ci, subdivision, dst_major, len(gpus), dense[dst])
        prev = label_of.setdefault(identity, out)
        if prev != out:
            raise AssertionError(
                f"identity {identity} would be labelled {prev} and {out}; a chunk's label must "
                f"not depend on which demand is being served.")
        return out

    chunk_paths: Dict[str, List] = {}
    for (t, identity), chain in sorted(paths.items()):
        met = max(r.completion for r in chain)
        key = f"Demand at {t} for chunk {label(identity, t)} from {identity[0]} met by epoch {met}"
        chunk_paths[key] = [[[r.epoch, _segment(r)] for r in chain]]

    # "7-Flows" is human-readable only under the LP format (the ops come from "8-Chunk paths"), so
    # a record that serves no demand still deserves a line; label it via any destination it has.
    flows_str = sorted({(r.epoch, f"Chunk {label_of.get(r.identity, r.identity[1])} from "
                                 f"{r.identity[0]} traveled over {_segment(r)}")
                        for r in records})

    epochs_required = max((r.completion for r in records), default=0) + 1
    finish = delta * epochs_required
    info: Dict = {
        "0-Collective": collective,
        # The parser axis: this is an LP-formulation schedule (nested per-demand paths), whatever
        # collective it implements. Without it, is_lp_format falls back to sniffing the structure.
        "0-Formulation": "LP",
        "1-Epoch_Duration": delta,
        "2-Expected_Epoch_Duration": delta,
        "3-Epochs_Required": epochs_required,
        "4-Collective_Finish_Time": finish,
        "5-Algo_Bandwidth": num_sources * scale.payload_per_gpu / finish if finish else 0.0,
        "7-Flows": [x[1] for x in flows_str],
        "8-Chunk paths": chunk_paths,
        # Derived from the LIVE scale, not the root topology: the schedule describes chunks at the
        # refined granularity, and ncclize sizes every op from this.
        "9-Chunk_Size": scale.bytes_per_chunk,
    }
    return info


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------
def assert_link_capacity(records: Sequence[DeliveryRecord], fine_topology: Topology,
                         coarse_epoch: float, scale: ChunkScale) -> None:
    """Per fine link per COARSE epoch, the network traffic must fit the link.

    Only network records are checked: they are the ones the coarse solve made a capacity promise
    about, and the intra flows are certified separately by the phase-3 round bound (`rounds <= m`),
    which is the same statement in the NVSwitch's own units.
    """
    load: Dict[Tuple[int, int, int], float] = defaultdict(float)
    for r in records:
        if r.phase != NETWORK:
            continue
        gb = r.volume * scale.bytes_per_chunk
        load[(r.sender, r.via_switches[0], r.epoch)] += gb
        load[(r.via_switches[-1], r.receiver, r.completion)] += gb
    over = [(a, b, k, round(v, 4), round(fine_topology.capacity[a][b] * coarse_epoch, 4))
            for (a, b, k), v in sorted(load.items())
            if v > fine_topology.capacity[a][b] * coarse_epoch + 1e-9]
    if over:
        raise AssertionError(
            f"fine link oversubscribed within a coarse epoch on {len(over)} (link, epoch) pairs "
            f"[(src, dst, epoch, GB, capacity)]: {over[:6]}")


def _piece_key(identity, egress_gpu: int, ingress_gpu: int, send_epoch: int):
    """The identity of a network send, shared by a ResolvedPiece and its DeliveryRecord.

    `build_records` copies exactly these four fields across (record.epoch is the piece's
    send_epoch scaled onto the fine axis), so this is what lets the caller intersect the surviving
    records with `res.pieces` without carrying record objects into the check.
    """
    return (tuple(identity), egress_gpu, ingress_gpu, send_epoch)


def emitted_network_sends(paths: Dict[Tuple[int, Identity], List[DeliveryRecord]],
                          W: int, m: int) -> set:
    """The network sends that survive `back_trace`, keyed by `_piece_key`.

    A record reachable from some demand's chain is one ncclize will emit; everything else is
    pruned and exists only as a "7-Flows" line. `m` un-scales record.epoch back to the coarse
    epoch the piece was filed at (record.epoch = W + m * send_epoch, see build_records).
    """
    return {_piece_key(r.identity, r.sender, r.receiver, (r.epoch - W) // m)
            for chain in paths.values() for r in chain if r.phase == NETWORK}


def _piece_is_emitted(p, emitted: set) -> bool:
    return _piece_key(p.identity, p.egress_gpu, p.ingress_gpu, p.send_epoch) in emitted


def check_network_pacing(res: IdentityResolution, coarse_epoch: float,
                         emitted: Optional[set] = None) -> List[Tuple[int, int]]:
    """Per-uplink (network-layer) realizability: which paced sends still fire early.

    This is the network layer's own pacing check, run in its own units (coarse epochs) -- the
    counterpart to the flat check in teccl.ncclize.helpers, which cannot reason correctly about a
    flattened multi-level fine axis. The intra layer needs no such check: its NVLink hops are
    deliberately unpaced and follow data dependencies only. It mirrors the rule the ncclize gate
    uses (teccl_ncclize._finish_before_start_gates) so that what is reported here is exactly what
    that manifest cannot pin: both ask "did some paced event land on THIS GPU exactly when this
    send is due", not "did the same GPU start a send in the previous epoch".

    `emitted` is what makes that mirroring real, and it is REQUIRED for the answer to mean
    anything. `res.pieces` is a strict SUPERSET of what ships: ncclize builds its ops (and hence
    its gate manifest) from the demand chains in "8-Chunk paths", so a piece that `back_trace`
    pruned -- because that identity reached that receiver earlier by another route -- never
    becomes an XML send. Those pruned pieces are not idle bookkeeping: counted here they
    MANUFACTURE P2/P3 clock ticks at epochs where the emitted schedule has none, and every real
    residual they cover is reported as pinned. Measured on the 96-GPU dual-plane clustered
    allgather: 768 of 16896 pieces are pruned, and the phantom ticks they contribute hide all 16
    genuine residuals (GPUs 0-15 at coarse epoch 4), whose sends then fire an epoch early and
    drive 16 receivers to 150% of their ingress capacity.

    So pass the emitted set built by `emitted_network_sends` from the NETWORK records that survive
    `back_trace`. Pieces outside it are skipped on both sides of the rule -- they neither need
    pinning nor may serve as anyone's clock. `None` means "no pruning information", which is only
    correct for a caller that has not run `back_trace` yet.

    Why the pruned pieces exist at all is a separate, deferred problem -- see the note on
    `assert_native_ownership` / lp_formulation.node_constraint_helper: the coarse LP has no copy
    semantics, so a cell that both wants an identity and relays it onward pulls it across the
    fabric twice.

    A send occupies its uplink -- (egress_gpu, first switch on its route) -- from its send epoch for
    volume/rate coarse epochs, i.e. it FINISHES serializing at send_epoch + duration (duration = 1
    under the fill-one-epoch rate rule, more for a sub-rate multi-epoch send). A send at epoch k is
    held to k if either clock ticks at k:

      P2 -- some paced send FROM THE SAME GPU finishes at k (it occupied one of that GPU's uplinks
        through k-1 and completed at the boundary), so the GPU has an op completion to gate on; or
      P3 -- some paced delivery ARRIVES at the sending GPU at k. The GPU may be sending nothing
        into k while it is still receiving, and a recv landing at k is then the only rate-paced
        tick available. ncclize realizes this one as a depid/deps edge on the send, which the
        kernel discharges when the proxy marks the recv slot filled.

    At k=0 the send is held by the egress staging it depends on (P1). If NEITHER clock ticks at k
    (k>0) nothing pins the send and it fires as early as its data dependency allows; those
    residuals are returned as (gpu, epoch) pairs. Closing them needs a clock this layer does not
    have -- the remaining cases are GPUs that neither send nor receive anything paced in the
    preceding epoch.

    BOTH pools are per GPU, not per uplink, and that is deliberate. A pacing clock only has to be
    a rate-paced event the sending GPU's proxy observes; it does not have to free the particular
    link this send needs. Restricting P2 to the same uplink is the right rule for link CONTENTION
    and the wrong one for pacing -- on a multi-uplink GPU (dual-plane, multi-rail) it discards
    every cross-uplink completion and reports residuals the gate manifest can in fact pin, since a
    netdep is an op-completion edge between two threadblocks of one GPU either way. Using FINISH
    rather than "starts at k-1" is what makes a multi-epoch send correctly pin a later send that
    begins as it completes.
    """
    chunk = res.scale.bytes_per_chunk
    # gpu -> the coarse epochs at which one of its paced sends finishes, at which it starts a send
    # (the sends we must pin), and at which a paced delivery lands on it.
    finishes: Dict[int, set] = defaultdict(set)
    starts: Dict[int, set] = defaultdict(set)
    arrivals: Dict[int, set] = defaultdict(set)
    for p in res.pieces:
        if p.rate is None:
            continue  # an unpaced network flow imposes no pacing (and none exists today)
        if emitted is not None and not _piece_is_emitted(p, emitted):
            continue  # pruned by back_trace: never an XML send, so never anyone's clock either
        duration = max(1, round(p.volume * chunk / (p.rate * coarse_epoch)))
        finishes[p.egress_gpu].add(p.send_epoch + duration)
        starts[p.egress_gpu].add(p.send_epoch)
        arrivals[p.ingress_gpu].add(p.send_epoch + duration)
    residual: List[Tuple[int, int]] = []
    for gpu, epochs in sorted(starts.items()):
        for k in sorted(epochs):
            if k == 0 or k in finishes[gpu] or k in arrivals[gpu]:
                continue
            residual.append((gpu, k))
    return residual


def build_flat_schedule(res: IdentityResolution, intra_flows: Sequence[IntraFlow],
                        fine_topology: Topology, fine_demand, coarse_epoch: float,
                        collective: str, grid: Optional[Tuple[float, int]] = None
                        ) -> Tuple[Dict, List[DeliveryRecord]]:
    """The assembled hierarchical solution -> one flat schedule dict on the fine topology.

    Runs ONCE, at the end, on flows that `flatten.rebase` has already folded into the ROOT's
    `(band, local_round)`. Everything here is therefore two-level whatever the recursion's depth
    was, which is the whole point of flattening on return.

    `grid` is the root level's own `(delta, m)`. Pass it -- `solve_hierarchical` has it on the root
    LevelSolution -- so the axis laid out here is the same one the recursion folded onto rather than
    an independent recomputation. It defaults to re-deriving for callers replaying a schedule from
    JSON, who have no LevelSolution to take it from.

    Returns (flow_str_info, records).
    """
    delta, m = grid if grid is not None else derive_grid(res.scale, fine_topology, coarse_epoch)
    assert_native_ownership(res)

    # Bands 0..K-1 are pinned to the network sends, so they must fit inside a coarse epoch.
    num_coarse_epochs = max((p.send_epoch for p in res.pieces), default=-1) + 1
    assert_bands_fit(intra_flows, m, num_coarse_epochs)

    records, W = build_records(res, intra_flows, m)
    assert_link_capacity(records, fine_topology, coarse_epoch, scale=res.scale)
    paths = back_trace(records, fine_demand, res.subdivision)

    # Network-layer pacing residuals (paced sends neither clock can pin). Reported, not fatal.
    # Runs AFTER back_trace and on its surviving records only: ncclize derives its gate manifest
    # from the demand chains, so a check over all of res.pieces would count pruned pieces as
    # clock ticks the emitted schedule does not have and report residuals as pinned. See
    # check_network_pacing's docstring for the measured case that motivated this.
    emitted = emitted_network_sends(paths, W, m)
    pruned = sum(1 for p in res.pieces
                 if p.rate is not None and not _piece_is_emitted(p, emitted))
    residual = check_network_pacing(res, coarse_epoch, emitted)
    if pruned:
        print(f"[flat] network pacing: {pruned} paced piece(s) are pruned by back_trace and never "
              f"emitted; they are excluded from the pacing clocks below (see the deferred coarse "
              f"no-copy relay gap in check_network_pacing's docstring)")
    if residual:
        print(f"[flat] network pacing: {len(residual)} (gpu, epoch) send group(s) fire early -- at "
              f"epoch k no send from that GPU finishes serializing (P2) and no paced delivery "
              f"arrives at it (P3), so nothing pins them to their coarse epoch [(gpu, epoch)]: "
              f"{residual[:8]}")
    else:
        print("[flat] network pacing: every emitted paced send is pinned to its coarse epoch, by "
              "one of the GPU's own sends completing (P2) or by a delivery arriving at it (P3)")

    # Algorithmic bandwidth counts the demand-bearing sources, matching the convention in
    # lp_formulation.get_flow_schedule (len(self.sources)); payload_per_gpu comes from the scale,
    # so the figure is invariant under refinement.
    num_sources = sum(1 for s in range(len(fine_demand))
                      if any(fine_demand[s][t][ci]
                             for t in range(len(fine_demand))
                             for ci in range(len(fine_demand[s][t]))))
    info = serialize(records, paths, res.scale, delta, collective, num_sources,
                     subdivision=res.subdivision)
    return info, records
