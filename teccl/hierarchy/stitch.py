"""
Phase 4: stitch the hierarchical solution into ONE flat schedule on the fine topology.

Inputs are the two halves of the solve below the coarse LP:
  * `IdentityResolution.pieces` -- inter-cell flows, already pinned to real fine GPUs and fine
    switch ids, carrying the coarse level's own pacing (`ResolvedPiece.rate`);
  * the per-cell `IntraFlow`s from the memoized NVSwitch schedule (teccl.hierarchy.intra_solve).

Output is the `flow_str_info` dict every solver emits, so `teccl/ncclize/teccl_ncclize.py` consumes
it unchanged and MSCCL XML comes out the far end.


THE EPOCH AXIS IS A NUMBERING CONVENTION, NOT A SIMULATION
---------------------------------------------------------
delta = scale.bytes_per_chunk / (fastest fine link) and m = Delta/delta, so a coarse epoch is
exactly m fine epochs. Band k -- the m fine epochs starting at `P + m*k` -- is CONCURRENT WITH
coarse epoch k, not a seam between epochs: the uplink and the NVSwitch are different links, so
intra-cell work hides under network time instead of serializing against it. (Measured on these
schedules, charging intra work as a gap between coarse epochs costs 0.2306 s / 0.8311 s for
allgather / alltoall versus 0.2022 / 0.8006 for the concurrent reading -- it turns a win into a
loss against a 0.22 s flat baseline.)

    band k == fine epochs [P + m*k, P + m*(k+1))

    P + m*k                          coarse epoch k's network sends              (offset 0)
    P + m*(k+1) + 1 + level          fan-out of pieces ARRIVING in epoch k        (low offsets)
    P + m*k - 1 - (Lmax - level)     staging for sends leaving in epoch k         (top of band k-1)
    [0, P), at fine epoch = level    epoch-0 staging + self_distribution          (prologue)

Two placements deserve their reasons written down:

  * Staging for a send in coarse epoch k sits at the TOP OF BAND k-1, not in the prologue. It has
    to precede the send it feeds, and putting it one band earlier keeps it in the band the phase-3
    scheduler actually costed it in (`_group_by_gap` pins a hard job to its deadline gap, and the
    `rounds <= m` certificate is computed on that placement). Collapsing every staging relay into
    the prologue would emit a placement nothing ever checked, and would concentrate it in the one
    band that has no network time to hide under. Only epoch-0 staging has no earlier band, so only
    it is genuinely prologue work.

  * Fan-out of a piece keys off `arrival_epoch + 1`, not `arrival_epoch`: `_extract_pieces` records
    the arrival as the LAST HOP's epoch, so the data lands at the END of that coarse epoch and the
    first band that can consume it is the next one.

Because a precedence level is at most a small tree depth while m is tens of epochs, NO fine-grained
tracking is needed here -- no per-fine-epoch capacity accounting, no spans, no packing or spill.
The grid exists to give intra levels somewhere to sit between network epochs and to make
`Epochs_Required * delta` land near the truth. Note m must NOT be shrunk to just fit the levels:
that would compact the epoch count but charge each intra level a full Delta/m. Epoch count is
nearly free (ncclize drops globally empty epochs); reported time is not.


PRECEDENCE LEVEL, NOT ROUND
---------------------------
`IntraFlow.local_round` is a TIME index; what the schedule needs is a DEPENDENCY level, and the two
differ by 5-6x (depth 3 vs 17 rounds for allgather). Rounds beyond the depth carry information
ncclize cannot use, because intra flows are not rate-limited. So the stitch discards `local_round`
and keeps `level` = longest-path depth in the same-identity sender->receiver DAG within a phase.
This is sound because the only same-buffer recv-then-send chains in the intra layer come from
broadcast-tree edges (`_Job.predecessor` is set only there); direct and dedup-merged deliveries are
terminal, and a cross-phase chain cannot exist while egress staging is native-only (asserted).
`_schedule_gap` still runs -- it is demoted from "produces the schedule axis" to "produces the
feasibility certificate", `rounds <= m`.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from teccl.hierarchy.intra_solve import IntraFlow
from teccl.hierarchy.reconstruct import Identity, IdentityResolution
from teccl.hierarchy.scale import ChunkScale
from teccl.topologies.topology import Topology

# Phases, in the order they occupy the axis within a band.
PROLOGUE = "prologue"          # available from t=0: self_distribution + epoch-0 staging
STAGE = "stage"                # native -> gateway, must precede the network send it feeds
NETWORK = "network"            # the inter-cell send itself
INGRESS = "ingress"            # fan-out of an arrival, inside the destination cell


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
def derive_grid(scale: ChunkScale, fine_topology: Topology, coarse_epoch: float,
                max_refinement: int = 128) -> Tuple[float, int]:
    """The fine epoch duration and how many of them a coarse epoch holds.

    Both are DERIVED from the live ChunkScale, never written down: refinement changes the chunk
    size, and delta and m must move with it or the round count and the epoch grid silently desync.
    """
    if scale is None:
        raise ValueError("IdentityResolution has no ChunkScale; cannot derive the fine epoch grid")
    if scale.refinement_from_root > max_refinement:
        raise ValueError(
            f"cumulative chunk refinement {scale.refinement_from_root} exceeds the budget "
            f"{max_refinement}: ncclize's chunk_up() would have to expand every chunk that finely. "
            f"The budget is shared by every level of the recursion.")
    max_bw = max(max(row) for row in fine_topology.capacity)
    delta = scale.epoch_duration(max_bw)
    m_float = coarse_epoch / delta
    m = int(round(m_float))
    if abs(m_float - m) > 1e-6 or m < 1:
        raise AssertionError(
            f"a coarse epoch is not a whole number of fine epochs: Delta={coarse_epoch} / "
            f"delta={delta} = {m_float}. Rounding it would corrupt every deadline on the axis.")
    return delta, m


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
# S1. Phases and precedence levels
# ----------------------------------------------------------------------------------------------
def flow_phase(flow: IntraFlow) -> Tuple[str, int]:
    """(phase, coarse epoch it is anchored to) for one intra flow.

    `gap` alone cannot say this: _group_by_gap pins a HARD job to its deadline gap and a soft one
    to its release gap, so a staging relay for coarse epoch 0, a self_distribution, and a fan-out
    of an epoch-0 arrival all land in gap 0 while belonging to three different places on the axis.
    The job's kind and hardness disambiguate them.
    """
    if flow.kind == "egress_stage" or flow.hard:
        # gap == the send_epoch of the piece this relay feeds.
        return (PROLOGUE, 0) if flow.gap == 0 else (STAGE, flow.gap)
    if flow.kind == "ingress_distribution":
        # gap == arrival_epoch; the data lands at the END of that coarse epoch.
        return INGRESS, flow.gap
    return PROLOGUE, 0


def levels(flows: Sequence[IntraFlow]) -> Dict[int, int]:
    """Longest-path depth of each flow in the same-identity sender->receiver DAG of its phase.

    Returns a map keyed by `id(flow)` (IntraFlow is frozen but not hashable-by-value in a way that
    distinguishes two identical rounds of one transfer). Direct deliveries get 0; only broadcast
    tree edges produce a positive depth, because they are the only place a receiver forwards what
    it just received.
    """
    by_group: Dict[Tuple, List[IntraFlow]] = defaultdict(list)
    for f in flows:
        phase, anchor = flow_phase(f)
        by_group[(f.cell, phase, anchor)].append(f)

    out: Dict[int, int] = {}
    for key, group in by_group.items():
        # producers[(identity, gpu)] = flows that put `identity` on `gpu`
        producers: Dict[Tuple[Identity, int], List[IntraFlow]] = defaultdict(list)
        for f in group:
            producers[(f.identity, f.receiver)].append(f)

        memo: Dict[int, int] = {}
        visiting: set = set()

        def depth(f: IntraFlow) -> int:
            fid = id(f)
            if fid in memo:
                return memo[fid]
            if fid in visiting:
                raise AssertionError(
                    f"cycle in the intra-cell precedence DAG at {key}: identity {f.identity} "
                    f"{f.sender}->{f.receiver}. A forwarding chain must be a tree.")
            visiting.add(fid)
            # The sender forwards only what someone delivered to it within this same phase; if
            # nothing did, it owned the data already and this flow is at depth 0.
            d = 0
            for p in producers.get((f.identity, f.sender), ()):
                d = max(d, depth(p) + 1)
            visiting.discard(fid)
            memo[fid] = d
            return d

        for f in group:
            out[id(f)] = depth(f)
    return out


# ----------------------------------------------------------------------------------------------
# S2/S3. Absolute fine epochs -> delivery records
# ----------------------------------------------------------------------------------------------
def build_records(res: IdentityResolution, intra_flows: Sequence[IntraFlow],
                  m: int) -> Tuple[List[DeliveryRecord], int]:
    """Place every inter-cell piece and intra-cell flow on the absolute fine epoch axis.

    Returns the records and P (the prologue width).
    """
    lvl = levels(intra_flows)

    grouped: Dict[Tuple[str, int], List[IntraFlow]] = defaultdict(list)
    for f in intra_flows:
        grouped[flow_phase(f)].append(f)

    # The prologue occupies fine epochs [0, P), one per precedence level, and the first network
    # send sits at P. At least 1 so epoch 0 is never a network send with nothing before it.
    prologue = grouped.get((PROLOGUE, 0), [])
    P = max(1, max((lvl[id(f)] for f in prologue), default=-1) + 1)

    records: List[DeliveryRecord] = []

    def emit_intra(f: IntraFlow, epoch: int, phase: str) -> None:
        records.append(DeliveryRecord(
            identity=f.identity, sender=f.sender, receiver=f.receiver,
            via_switches=(f.via_switch,), volume=f.volume, epoch=epoch,
            # An intra hop is not paced, so it is modelled as occupying its epoch and being
            # available at the next one -- which is exactly what makes a level-l flow able to feed
            # a level-(l+1) flow placed one epoch later.
            completion=epoch + 1, rate=None, phase=phase, cell=f.cell, level=lvl[id(f)]))

    for f in prologue:
        emit_intra(f, lvl[id(f)], PROLOGUE)

    # Staging for coarse epoch k: the top of band k-1, deepest level last so it lands at
    # P + m*k - 1, immediately before the send it feeds.
    for (phase, k), group in sorted(grouped.items()):
        if phase != STAGE:
            continue
        lmax = max(lvl[id(f)] for f in group)
        if lmax + 1 > m:
            raise AssertionError(
                f"staging for coarse epoch {k} has precedence depth {lmax + 1} > m={m}; it cannot "
                f"fit in the band before the send it feeds.")
        for f in group:
            emit_intra(f, P + m * k - 1 - (lmax - lvl[id(f)]), STAGE)

    # Fan-out of the pieces that arrived in coarse epoch k: the low offsets of band k+1, after
    # offset 0 which belongs to that band's own network sends.
    for (phase, k), group in sorted(grouped.items()):
        if phase != INGRESS:
            continue
        lmax = max(lvl[id(f)] for f in group)
        if lmax + 2 > m:
            raise AssertionError(
                f"fan-out of coarse epoch {k}'s arrivals has depth {lmax + 1}, which does not fit "
                f"in a band of m={m} fine epochs after the network send at offset 0.")
        for f in group:
            emit_intra(f, P + m * (k + 1) + 1 + lvl[id(f)], INGRESS)

    for p in res.pieces:
        records.append(DeliveryRecord(
            identity=p.identity, sender=p.egress_gpu, receiver=p.ingress_gpu,
            via_switches=tuple(p.via_switches), volume=p.volume, epoch=P + m * p.send_epoch,
            # The piece occupies coarse epoch(s) up to its arrival epoch and is held at the start
            # of the next band -- the same convention the fan-out placement above reads.
            completion=P + m * (p.arrival_epoch + 1), rate=p.rate, phase=NETWORK,
            cell=p.src_cell))

    return records, P


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


def stitch(res: IdentityResolution, intra_flows: Sequence[IntraFlow], fine_topology: Topology,
           fine_demand, coarse_epoch: float, collective: str) -> Tuple[Dict, List[DeliveryRecord]]:
    """Hierarchical solution -> flat schedule dict. Returns (flow_str_info, records)."""
    delta, m = derive_grid(res.scale, fine_topology, coarse_epoch)
    assert_native_ownership(res)

    records, _P = build_records(res, intra_flows, m)
    assert_link_capacity(records, fine_topology, coarse_epoch, scale=res.scale)
    paths = back_trace(records, fine_demand, res.subdivision)

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
