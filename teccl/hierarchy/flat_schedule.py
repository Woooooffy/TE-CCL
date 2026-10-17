"""
The FINAL post-process: the assembled hierarchical solution -> ONE flat schedule on the fine
topology. Runs exactly once, after the recursion has returned.

Inputs are the two halves of the ROOT level:
  * `IdentityResolution.pieces` -- inter-cell flows, already pinned to real fine GPUs and fine
    switch ids, one Send per transmission (a switch copy keeps its branches), paced at the root
    level's own rate (teccl.sends.rate);
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
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from teccl.hierarchy.crossbar_solve import PROLOGUE_BAND, IntraFlow, rounds_in
from teccl.hierarchy.flatten import aligned_band, assert_bands_fit, derive_grid
from teccl.hierarchy.reconstruct import Identity, IdentityResolution
from teccl.hierarchy.scale import ChunkScale
from teccl.sends import Branch, Send, delivery_index, link_windows, rate
from teccl.topologies.topology import Topology

# Record phases. These are ANNOTATION only -- placement is decided by the level that produced the
# flow, and the stitch reads (band, local_round). NETWORK is the one the code branches on, because
# only inter-cell sends carry a rate and occupy a coarse-epoch-aligned link.
PROLOGUE = "prologue"          # ran before coarse epoch 0's first network send
NETWORK = "network"            # the inter-cell send itself
INGRESS = "ingress"            # fallback label for an intra flow with no recorded kind


# The stitch's records are Sends (teccl/sends.py) on the ABSOLUTE fine-epoch axis: a network send
# keeps its fan-out (a switch copy is one record with several branches), and an intra flow is a
# one-branch send whose `kind` is its phase. A delivery is a (send, branch) pair.
Delivery = Tuple[Send, Branch]


# ----------------------------------------------------------------------------------------------
# S0. Grid and preconditions
# ----------------------------------------------------------------------------------------------
def assert_native_ownership(res: IdentityResolution) -> None:
    """Every staged identity must be staged BY ITS OWN SOURCE -- unless it is a modelled transit.

    This is the host-transit guard. A staging relay from a GPU other than the identity's source is
    legitimate in exactly one case: a cell FORWARDING an identity it received, which _emit_refined
    emits as an egress_stage from the landing GPU WITH an explicit `release_band` (the epoch after
    the inbound leg lands). That is reachable only when the coarse solve budgeted the relay's
    epoch (the MILP's forward-dwell constraint), and back_trace below independently checks its
    causality. Anything else -- a non-source staging relay with no release -- is a GPU being asked
    to send data nothing ever delivered to it, so it still fails loud here.
    """
    bad = [(d.cell, d.identity, d.src_gpu) for d in res.intra_demands
           if d.kind == "egress_stage" and d.src_gpu != d.identity[0] and d.release_band is None]
    if bad:
        raise AssertionError(
            f"{len(bad)} egress_stage demands are staged by a GPU that is not the identity's "
            f"source {bad[:5]} and carry no release -- host transit (a coarse path relayed "
            f"through an intermediate cell) that no coarse solve budgeted, which is unmodelled.")


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
                  m: int) -> Tuple[List[Send], int]:
    """Place every inter-cell send and intra-cell flow on the absolute fine epoch axis.

    Pure translation -- see the module docstring for the four-line mapping. Returns the records and
    the prologue width W.
    """
    W = prologue_width(intra_flows)

    records: List[Send] = []
    for f in intra_flows:
        epoch = f.local_round if f.band == PROLOGUE_BAND else W + m * f.band + f.local_round
        # One record per sub-chunk the transfer carries, ALL AT THE SAME EPOCH. A coalesced flow is
        # one physical transfer of contiguous bytes, so its sub-chunks have consecutive labels and
        # ncclize merges them back into a single cnt=Q operation -- which it can only do for
        # addresses that share a step.
        for identity in f.identities:
            records.append(Send(
                identity=identity, sender=f.sender, epoch=epoch, span=f.span,
                volume=f.volume / len(f.identities),
                # Every receiver holds the data once the transfer finishes, which is what lets a
                # flow ending at round r feed one starting at r+1 -- the precedence _schedule_band
                # already enforced. (A hop over a DIRECT link has no switch to name: a physical
                # ring cell.)
                fanout=tuple(replace(b, ready=epoch + f.span) for b in f.fanout),
                kind=PROLOGUE if f.band == PROLOGUE_BAND else (f.kind or INGRESS)))

    for p in res.pieces:
        # A send leaves at its band's leading edge, holds its links for its whole span of coarse
        # epochs, and each branch is held at the END of its arrival epoch -- the leading edge of
        # the band where the fan-out that consumes it was scheduled.
        records.append(replace(p, epoch=W + m * p.epoch, span=m * p.span, kind=NETWORK,
                               fanout=tuple(replace(b, ready=W + m * b.ready) for b in p.fanout)))

    return records, W


# ----------------------------------------------------------------------------------------------
# S4. Back-trace: chunk paths, causality, coverage
# ----------------------------------------------------------------------------------------------
def back_trace(records: Sequence[Send], fine_demand, subdivision: int
               ) -> Dict[Tuple[int, Identity], List[Delivery]]:
    """For every fine demand, the chain of deliveries that brings it from its source, verified as
    it is built.

    The emitter and the verifier are one traversal on purpose: the path list serialized into
    "8-Chunk paths" IS the evidence that each demand is met, so a demand that cannot be traced back
    to its source is a hard error rather than a missing line in the output.
    """
    # (identity, gpu) -> the delivery that got it there earliest
    via = delivery_index(records)
    # (identity, gpu) -> earliest epoch that gpu holds it; a source owns its identities from 0
    holds = {key: branch.ready for key, (_send, branch) in via.items()}
    for r in records:
        holds[(r.identity, r.identity[0])] = 0

    paths: Dict[Tuple[int, Identity], List[Delivery]] = {}
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
                    chain: List[Delivery] = []
                    cur = t
                    seen = set()
                    while cur != s:
                        hop = via.get((sub, cur))
                        if hop is None:
                            missing.append((t, sub, cur))
                            break
                        if (sub, cur) in seen:
                            raise AssertionError(
                                f"cycle while tracing demand ({t}, {sub}) back to its source")
                        seen.add((sub, cur))
                        send, branch = hop
                        # Causality: the sender must already hold the data when it sends.
                        held = holds.get((sub, send.sender))
                        if held is None or held > send.epoch:
                            raise AssertionError(
                                f"causality violation delivering {sub} to {t}: {send.sender} sends "
                                f"in fine epoch {send.epoch} but holds the data only from "
                                f"{held!r} ({send.kind}, via {branch.via})")
                        chain.append(hop)
                        cur = send.sender
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
def _segment(send: Send, branch: Branch, bytes_per_chunk: float, delta: float) -> str:
    """One "8-Chunk paths" / "7-Flows" hop, in the grammar teccl_ncclize's PATH_SEGMENT_RE reads.

    One line per BRANCH: the string grammar names one receiver, so a switch copy is written as
    one line per branch with the same sender and epoch. This projection is the one place the stitch
    drops the fan-out; the records themselves keep it.

    The rate token is emitted only for a network send, the one kind its level paced. Its absence
    is meaningful, not a default: an intra-cell hop carries ordering but is deliberately not pinned
    to the epoch grid, so throttling it would be inventing a constraint the solve never imposed.
    """
    out = f"{send.sender}->{branch.receiver} with volume {send.volume:g} in epoch {send.epoch}"
    if send.kind == NETWORK:
        out += f" at rate {rate(send, bytes_per_chunk, delta):g}"
    if branch.via:
        out += " via switches " + "->".join(str(x) for x in branch.via)
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


def serialize(records: Sequence[Send],
              paths: Dict[Tuple[int, Identity], List[Delivery]],
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
    gpus = sorted({r.sender for r in records} | {b.receiver for r in records for b in r.fanout})
    dense = {g: i for i, g in enumerate(gpus)}
    dst_major = collective == "alltoall"
    bpc = scale.bytes_per_chunk

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
        met = max(b.ready for _r, b in chain)
        key = f"Demand at {t} for chunk {label(identity, t)} from {identity[0]} met by epoch {met}"
        chunk_paths[key] = [[[r.epoch, _segment(r, b, bpc, delta)] for r, b in chain]]

    # "7-Flows" is human-readable only under the LP format (the ops come from "8-Chunk paths"), so
    # a record that serves no demand still deserves a line; label it via any destination it has.
    flows_str = sorted({(r.epoch, f"Chunk {label_of.get(r.identity, r.identity[1])} from "
                                 f"{r.identity[0]} traveled over {_segment(r, b, bpc, delta)}")
                        for r in records for b in r.fanout})

    # "10-Sends": the transmissions themselves, with every receiver each one reaches -- what ncclize
    # builds its ops from (teccl_ncclize.parse_sends), and the one section that keeps a switch copy
    # as ONE send. Only what the demand chains use: a send with only the branches some chain takes,
    # in first-use order, so a schedule without copy expands to exactly the hops, in exactly the
    # order, the legacy parser reads off "8-Chunk paths".
    # The label is the DEMAND's (a dst-major label encodes the demand's final destination, not a
    # relay hop's receiver), exactly as "8-Chunk paths" writes it on every hop of the chain.
    used: Dict[Send, List[Branch]] = {}
    demand_dst: Dict[Send, int] = {}
    for (t, _identity), chain in sorted(paths.items()):
        for r, b in chain:
            branches = used.setdefault(r, [])
            demand_dst.setdefault(r, t)
            if b not in branches:
                branches.append(b)
    sends = [{"chunk": label(r.identity, demand_dst[r]), "from": r.identity[0],
              "sender": r.sender, "epoch": r.epoch, "span": r.span,
              "rate": rate(r, bpc, delta) if r.kind == NETWORK else None, "volume": r.volume,
              "branches": [{"to": b.receiver, "via": list(b.via), "ready": b.ready} for b in bs]}
             for r, bs in used.items()]

    epochs_required = max((b.ready for r in records for b in r.fanout), default=0) + 1
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
        "10-Sends": sends,
    }
    return info


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------
def assert_link_capacity(records: Sequence[Send], fine_topology: Topology, coarse_epoch: float,
                         scale: ChunkScale, m: int) -> None:
    """Per fine link per COARSE epoch, the network traffic must fit the link.

    Only network records are checked: they are the ones the coarse solve made a capacity promise
    about, and the intra flows are certified separately by the phase-3 round bound (`rounds <= m`),
    which is the same statement in the NVSwitch's own units.

    A send charges every link of its tree once (teccl.sends.link_windows), spread evenly over the
    coarse epochs it holds the link: a switch copy's shared links count one transmission.
    """
    load: Dict[Tuple[int, int, int], float] = defaultdict(float)
    for r in records:
        if r.kind != NETWORK:
            continue
        per_epoch = r.volume * scale.bytes_per_chunk * m / r.span
        for (a, b), window in link_windows(r).items():
            for e in range(window.start, window.stop, m):
                load[(a, b, e)] += per_epoch
    over = [(a, b, k, round(v, 4), round(fine_topology.capacity[a][b] * coarse_epoch, 4))
            for (a, b, k), v in sorted(load.items())
            if v > fine_topology.capacity[a][b] * coarse_epoch + 1e-9]
    if over:
        raise AssertionError(
            f"fine link oversubscribed within a coarse epoch on {len(over)} (link, epoch) pairs "
            f"[(src, dst, epoch, GB, capacity)]: {over[:6]}")


def emitted_network_sends(paths: Dict[Tuple[int, Identity], List[Delivery]],
                          W: int, m: int) -> set:
    """The network deliveries that survive `back_trace`, as (identity, sender, coarse epoch,
    receiver).

    A delivery reachable from some demand's chain is one ncclize will emit; everything else is
    pruned and exists only as a "7-Flows" line. `m` un-scales the record epoch back to the coarse
    epoch the send was filed at (record epoch = W + m * epoch, see build_records).
    """
    return {(r.identity, r.sender, (r.epoch - W) // m, b.receiver)
            for chain in paths.values() for r, b in chain if r.kind == NETWORK}


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
    its gate manifest) from the demand chains in "8-Chunk paths", so a delivery that `back_trace`
    pruned -- because that identity reached that receiver earlier by another route -- never
    becomes an XML send. Those pruned deliveries are not idle bookkeeping: counted here they
    MANUFACTURE P2/P3 clock ticks at epochs where the emitted schedule has none, and every real
    residual they cover is reported as pinned. Measured on the 96-GPU dual-plane clustered
    allgather: 768 of 16896 pieces are pruned, and the phantom ticks they contribute hide all 16
    genuine residuals (GPUs 0-15 at coarse epoch 4), whose sends then fire an epoch early and
    drive 16 receivers to 150% of their ingress capacity.

    So pass the emitted set built by `emitted_network_sends` from the NETWORK records that survive
    `back_trace`. A send none of whose branches survives is skipped on both sides of the rule --
    it neither needs pinning nor may serve as anyone's clock. `None` means "no pruning
    information", which is only correct for a caller that has not run `back_trace` yet.

    Why the pruned pieces exist at all is a separate, deferred problem -- see the note on
    `assert_native_ownership` / lp_formulation.node_constraint_helper: the coarse LP has no copy
    semantics, so a cell that both wants an identity and relays it onward pulls it across the
    fabric twice.

    A send occupies its uplink from its epoch for its `span` coarse epochs, i.e. it FINISHES
    serializing at epoch + span (the self-clocking pacing rule, teccl.sends.rate). One send is one
    transmission however many branches it has: it is one clock tick at its sender, and an arrival
    tick at each surviving receiver. A send at epoch k is held to k if either clock ticks at k:

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
    # gpu -> the coarse epochs at which one of its paced sends finishes, at which it starts a send
    # (the sends we must pin), and at which a paced delivery lands on it.
    finishes: Dict[int, set] = defaultdict(set)
    starts: Dict[int, set] = defaultdict(set)
    arrivals: Dict[int, set] = defaultdict(set)
    for p in res.pieces:
        branches = [b for b in p.fanout
                    if emitted is None or (p.identity, p.sender, p.epoch, b.receiver) in emitted]
        if not branches:
            continue  # pruned by back_trace: never an XML send, so never anyone's clock either
        finishes[p.sender].add(p.epoch + p.span)
        starts[p.sender].add(p.epoch)
        for b in branches:
            arrivals[b.receiver].add(p.epoch + p.span)
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
                        ) -> Tuple[Dict, List[Send]]:
    """The assembled hierarchical solution -> one flat schedule dict on the fine topology.

    Runs ONCE, at the end, on flows that `flatten.rebase` has already folded into the ROOT's
    `(band, local_round)`. Everything here is therefore two-level whatever the recursion's depth
    was, which is the whole point of flattening on return.

    `grid` is the root level's own `(delta, m)`. Pass it -- `solve_hierarchical` has it on the root
    LevelSolution -- so the axis laid out here is the same one the recursion folded onto rather than
    an independent recomputation. It defaults to re-deriving for callers replaying a schedule from
    JSON, who have no LevelSolution to take it from.

    Returns (flow_str_info, records). The records keep every send's fan-out; the info dict is the
    per-branch projection ncclize reads today.
    """
    delta, m = grid if grid is not None else derive_grid(res.scale, fine_topology, coarse_epoch)
    assert_native_ownership(res)

    # Bands 0..K-1 are pinned to the network sends, so they must fit inside a coarse epoch.
    num_coarse_epochs = max((p.epoch for p in res.pieces), default=-1) + 1
    assert_bands_fit(intra_flows, m, num_coarse_epochs)

    records, W = build_records(res, intra_flows, m)
    assert_link_capacity(records, fine_topology, coarse_epoch, res.scale, m)
    paths = back_trace(records, fine_demand, res.subdivision)

    # Network-layer pacing residuals (paced sends neither clock can pin). Reported, not fatal.
    # Runs AFTER back_trace and on its surviving deliveries only: ncclize derives its gate manifest
    # from the demand chains, so a check over all of res.pieces would count pruned deliveries as
    # clock ticks the emitted schedule does not have and report residuals as pinned. See
    # check_network_pacing's docstring for the measured case that motivated this.
    emitted = emitted_network_sends(paths, W, m)
    pruned = sum(1 for p in res.pieces
                 if not any((p.identity, p.sender, p.epoch, b.receiver) in emitted for b in p.fanout))
    residual = check_network_pacing(res, coarse_epoch, emitted)
    if pruned:
        print(f"[flat] network pacing: {pruned} paced send(s) are pruned by back_trace and never "
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
