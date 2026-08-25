"""
The CROSSBAR level solver: a closed-form (Gurobi-free) solve for a cell whose internal fabric is a
single non-blocking switch.

This is one row in the recursion's base-case dispatch table (teccl.hierarchy.solve.solve_flat), not
a distinct layer of the design. A level is "solve a demand set on a topology"; when that topology
happens to be a crossbar the answer is known in closed form, so the level is MEMOIZED rather than
handed to a real formulation. Every current topology's innermost level (an 8-GPU + NVSwitch host) is
that case, which is why this module carried the whole intra-cell phase before the recursion existed.

Step B of the level above (teccl.hierarchy.reconstruct.build_child_problems) hands each cell a list
of IntraCellDemand descriptors that must be satisfied INSIDE it before/after the inter-cell sends.
This module schedules those onto the cell's internal switch and emits fine flows.

The core observation (see the design note hierarchical_phase3_forward_plan): a single NVSwitch is a
non-blocking crossbar, so scheduling a set of point-to-point transfers on it is exactly a
BIRKHOFF-VON NEUMANN / bipartite edge-coloring / preemptive open-shop problem. The only contention
is each GPU's one egress link and one ingress link (both uniform on an NVSwitch); a "round" is a
bipartite b-matching whose row/column sums are bounded by port capacity. The optimal makespan is the
max port load, and for a symmetric all-to-all (allgather/alltoall) demand the edge-coloring IS the
ring -- it falls out with no special-casing.

Two demand classes, with a hard/soft split driven by network impact:
  * egress_stage        HARD  -- a native GPU must relay an identity to a gateway GPU before that
                                 gateway egresses it onto the (slow) network. Missing this deadline
                                 slips the internode schedule, so it is a hard constraint.
  * ingress_distribution / self_distribution   SOFT -- fan-out delivery whose finishing time does
                                 not gate the network (unless an ingress feeds a downstream transit
                                 egress, in which case it is promoted to hard).

The scheduler is EDF list-scheduling: at each round it runs a priority-weighted greedy b-matching
(hard first, then earliest deadline, then a ring-distance tiebreak), which meets every hard deadline
given the large NVLink:network slack and recovers the ring for the symmetric case.

THIS MODULE OWNS ITS OWN FINE SCHEDULE, COMPLETELY. That is the recursion contract: a solver step
is responsible for scheduling the flows it produces, and the stitch above it only flattens. Here the
schedule is (band, local_round), where `band` is the coarse epoch a transfer runs concurrently with
and `local_round` is its fine-epoch offset inside that band -- and no rescaling is needed between
the two, because a round IS a fine epoch: a round is "one chunk across one NVSwitch port", which
takes bytes_per_chunk / nvlink_bw, and that is precisely how delta is defined.

Two pieces do all the placement work:
  * `_assign_bands` picks the band from READINESS (as early as the data exists), with the prologue
    band -1 as the escape for work that must precede coarse epoch 0's sends.
  * `_schedule_band` orders the rounds inside a band, already respecting precedence and port
    capacity -- so the round index carries the dependency order and nothing downstream has to
    re-derive it.

What the memoized NVSwitch case deliberately does NOT do is pin its transfers to a timeline: the
inner fabric is far faster than the outer, so intra-cell work hides under network time and only DATA
DEPENDENCIES matter. That is why this simple readiness+greedy-rounds scheme suffices. A future level
with real ordering requirements would enforce them in its own solve, not by asking the stitch to
reconstruct them.
"""
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from math import inf
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from teccl.hierarchy.bands import PROLOGUE_BAND, band_of
from teccl.hierarchy.cell import Cell
from teccl.hierarchy.reconstruct import Identity, IntraCellDemand

EPS = 1e-9

# Debug prints are gated on the TECCL_INTRA_DEBUG env var (or the `debug=` param on schedule_cell).
# They narrate each phase-3 step -- job building + fan-out density decisions + dedup, then the
# per-round matching, then a per-cell optimality summary -- so the schedule can be eyeballed for
# correctness against the max-port-load lower bound.
_ENV_DEBUG = os.environ.get("TECCL_INTRA_DEBUG", "").lower() in ("1", "true", "yes", "on")

def _p(debug: bool, msg: str = "") -> None:
    if debug:
        print(msg)


@dataclass(frozen=True)
class IntraFlow:
    """One lowered intra-cell hop, logical GPU->GPU (annotated with the switch at stitch time).

    (band, local_round) IS this level's fine schedule, and it is complete: `band` is the coarse
    epoch this transfer runs concurrently with, and `local_round` is its offset within that band.
    One round is exactly one fine epoch -- a round is defined as "one chunk across one NVSwitch
    port", which takes bytes_per_chunk / nvlink_bw, and that is the definition of delta. So the
    stitch needs no rescaling: absolute fine epoch = (where band starts) + local_round.

    band == PROLOGUE_BAND (-1) is the work that must precede coarse epoch 0's network sends. It is
    empty on a topology whose gateways own their own data (rail-optimized), and non-empty only
    where the boundary forces a relay before the first send (the hetero cluster).
    band == (number of coarse epochs) is the epilogue: redistributing the final arrivals.

    kind is provenance for debugging and the serialized schedule; hard records that a network send
    waited on this transfer. Neither is load-bearing downstream -- placement is fully decided here.
    """
    cell: int
    identity: Identity
    sender: int                 # fine GPU id
    receiver: int               # fine GPU id
    via_switch: int             # the cell's internal NVSwitch id
    volume: float
    band: int                   # coarse epoch this runs concurrently with (-1 == prologue)
    local_round: int            # fine-epoch offset within the band (its START)
    span: int = 1               # fine epochs it occupies: ceil(volume / port_cap)
    kind: str = ""              # egress_stage | ingress_distribution | self_distribution
    hard: bool = False          # a network send waited on this transfer
    # The sub-chunks this transfer carries, in label order. More than one when co-travelling
    # sub-chunks of a refined chunk were coalesced: they are contiguous bytes over one edge, so
    # they are one transfer and are emitted into one fine epoch, where ncclize can merge them into
    # a single cnt=Q operation. Defaults to (identity,).
    identities: Tuple[Identity, ...] = ()

    def __post_init__(self):
        if not self.identities:
            object.__setattr__(self, "identities", (self.identity,))


def rounds_in(flows: Sequence[IntraFlow]) -> int:
    """How many rounds a set of flows occupies: the exclusive end of the last one.

    `local_round + span` is where a flow finishes, so the max over a set is exactly its round count.
    This one line was written out at seven call sites for four different meanings -- a band's
    makespan, a schedule's peak, the prologue's width, the fold stride in `solve.rebase` -- which is
    how two of them drifted apart. It lives next to IntraFlow because that is the type it reads, and
    because `bands.py` cannot host it without an import cycle.
    """
    return max((f.local_round + f.span for f in flows), default=0)


def band_rounds(flows: Sequence[IntraFlow]) -> Dict[Tuple[int, int], int]:
    """Rounds occupied per (cell, band). Cells schedule independently on their own switch, so the
    cost of a band is the busiest cell's, never the sum -- which is why the key is the pair."""
    used: Dict[Tuple[int, int], int] = defaultdict(int)
    for f in flows:
        used[(f.cell, f.band)] = max(used[(f.cell, f.band)], f.local_round + f.span)
    return dict(used)


@dataclass
class _Job:
    """A single point-to-point transfer the scheduler must place into rounds.

    remaining/ completion tracks mutate during scheduling. `predecessor` references another _Job
    that must complete in a strictly earlier round before this job becomes ready (used only by
    broadcast-tree lowering, where a child forwards only after it has received). Referencing the
    parent job by object (not list index) keeps precedence intact when jobs are regrouped by band."""
    identity: Identity           # representative; `identities` is the full set this job carries
    src: int
    dst: int
    volume: float
    # The FIRST band this job could possibly run in. PROLOGUE_BAND (-1) means "the data exists
    # before the collective starts" (anything sourced from a GPU's native chunks); a fan-out of a
    # network arrival is arrival_epoch + 1, since a piece lands at the END of its arrival epoch.
    # Note this is the earliest POSSIBLE band, not the preferred one -- _assign_bands will not use
    # the prologue unless a deadline forces it.
    release_gap: int
    deadline_gap: float          # inf for soft
    hard: bool
    kind: str                    # provenance, for debugging
    predecessor: "Optional[_Job]" = None
    remaining: float = field(default=None)   # type: ignore[assignment]
    completion_round: Optional[int] = None
    # Every sub-chunk this job moves, in label order. Normally just (identity,); after
    # _coalesce_subchunks it is the Q co-travelling sub-chunks of one refined chunk, which are one
    # physical transfer and have to be emitted as one.
    identities: Tuple[Identity, ...] = ()

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.volume
        if not self.identities:
            self.identities = (self.identity,)


# --------------------------------------------------------------------------------------------
# Step 1: IntraCellDemand -> jobs (with fan-out lowering)
# --------------------------------------------------------------------------------------------
def _ring_distance(src: int, dst: int, gpus: Sequence[int]) -> int:
    """Position distance from src to dst along the cell's GPU ordering, used as the default ring
    tiebreak so a symmetric all-to-all edge-colors into the canonical ring."""
    order = {g: i for i, g in enumerate(gpus)}
    n = len(gpus)
    return (order[dst] - order[src]) % n


def _binomial_tree_edges(root: int, wanters: Sequence[int]
                         ) -> List[Tuple[int, int, int]]:
    """Recursive-doubling broadcast tree over {root} U wanters. Returns (parent, child, birth_round)
    edges: at round t every GPU that already holds the data forwards to one that does not, so after
    ceil(log2(n)) rounds all hold it. birth_round is the round the edge fires (0-indexed), which the
    scheduler turns into a precedence (a child cannot forward before it received)."""
    holders = [root]
    remaining = list(wanters)
    edges: List[Tuple[int, int, int]] = []
    rnd = 0
    while remaining:
        newly = []
        for h in list(holders):
            if not remaining:
                break
            child = remaining.pop(0)
            edges.append((h, child, rnd))
            newly.append(child)
        holders.extend(newly)
        rnd += 1
    return edges


def _intra_send_load(gpu: int, demands: Sequence[IntraCellDemand]) -> float:
    """Total volume `gpu` must push INTO THE NVSWITCH (its intra-cell send-link load) across all
    intra demands. This is a CLOSED intra-cell quantity -- it counts only GPU->switch hops on this
    cell's crossbar, never the network egress (gateway->leaf->other cell), which is a separate link
    owned by the coarse solve. A GPU that plays a network-gateway role shows up here as a big intra
    sender only because arrived/departing data generates intra fan-out and staging that funnel
    through it -- the number measured is still purely NVSwitch send load."""
    tot = 0.0
    for d in demands:
        if d.src_gpu == gpu:
            tot += d.volume * max(1, len(d.dst_gpus))
    return tot


def _max_intra_recv_load(demands: Sequence[IntraCellDemand]) -> float:
    """Max volume any single GPU must PULL OUT OF THE NVSWITCH (its intra-cell recv-link load).
    Also a closed intra-cell quantity (switch->GPU hops only). On a non-blocking crossbar the
    makespan floor is the busiest port, so this is the intra recv-bound lower bound the fan-out
    density test measures a source's send load against."""
    recv: Dict[int, float] = defaultdict(float)
    for d in demands:
        for t in d.dst_gpus:
            recv[t] += d.volume
    return max(recv.values(), default=0.0)


def _coalesce_subchunks(jobs: List[_Job], subdivision: int, debug: bool = False) -> List[_Job]:
    """Merge the sub-chunks of one refined chunk that travel the same edge under the same
    constraints into ONE job.

    Sub-chunk refinement (reconstruct._emit_refined) splits chunk (s, ci) into Q commodities
    (s, ci*Q+j) so that two shares taking DIFFERENT routes stay distinguishable. Where they take
    the SAME route they are not two transfers -- they are contiguous bytes of one chunk moving
    between one pair of GPUs -- and keeping them apart costs real work downstream: port_cap is one
    sub-chunk per port per round, so the scheduler is forced to put them in consecutive rounds,
    hence consecutive fine epochs, hence different ncclize steps, where `make_intervals` can no
    longer merge them into a single cnt=Q operation. Measured on the hetero allgather that was
    154/154 pairs split and roughly double the XML op count.

    Merging is only sound when the two are indistinguishable to the scheduler, so the key includes
    release, deadline and hardness: sub-chunks that stage for different network epochs, or arrive
    in different epochs, stay separate and simply do not benefit. Tree jobs are excluded -- they
    carry a precedence chain, and their labels are not contiguous in general.
    """
    if subdivision <= 1:
        return jobs
    # Precedence is held by OBJECT REFERENCE, so a job another job waits on must survive merging
    # intact -- replacing it with a merged job would leave its children pointing at an object the
    # scheduler never runs, and they would never become ready.
    referenced = {id(j.predecessor) for j in jobs if j.predecessor is not None}
    groups: Dict[Tuple, List[_Job]] = defaultdict(list)
    passthrough: List[_Job] = []
    for j in jobs:
        if j.predecessor is not None or id(j) in referenced or len(j.identities) != 1:
            passthrough.append(j)
            continue
        s, ci = j.identity
        groups[(s, ci // subdivision, j.src, j.dst, j.release_gap, j.deadline_gap, j.hard)].append(j)

    out: List[_Job] = list(passthrough)
    merged = 0
    for key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(group) == 1:
            out.append(group[0])
            continue
        group.sort(key=lambda j: j.identity[1])
        first = group[0]
        out.append(_Job(
            identity=first.identity, src=first.src, dst=first.dst,
            volume=sum(j.volume for j in group),
            release_gap=first.release_gap, deadline_gap=first.deadline_gap, hard=first.hard,
            kind=first.kind, predecessor=None,
            identities=tuple(j.identity for j in group)))
        merged += 1
        _p(debug, f"      coalesce {first.src}->{first.dst} chunk {key[0]},{key[1]}: "
                  f"{len(group)} sub-chunks {[j.identity[1] for j in group]} -> one transfer "
                  f"of volume {out[-1].volume:g}")
    if debug and merged:
        _p(debug, f"  [_coalesce_subchunks] {merged} co-travelling sub-chunk group(s) merged; "
                  f"{len(jobs)} -> {len(out)} jobs")
    return out


def _to_jobs(demands: Sequence[IntraCellDemand], cell: Cell,
             switch_copy: bool = False, debug: bool = False,
             subdivision: int = 1) -> List[_Job]:
    """Convert a cell's IntraCellDemand list into scheduler jobs.

    egress_stage        -> a HARD point-to-point delivery (native -> gateway), deadline = its epoch.
    ingress_distribution / self_distribution -> a fan-out, lowered by a density test that compares
    the source GPU's intra-cell SEND load to the cell's max intra-cell RECV load (both NVSwitch
    port loads -- a closed intra-cell comparison, nothing to do with the network link):
        * recv-bound (send_load <= max_recv) or switch multicast -> N DIRECT edges (the dense
          allgather/alltoall case; the scheduler edge-colors them into the ring). switch_copy keeps
          it a single logical send only when we later model multicast at emit time; for now the
          unicast direct edges are emitted and the ring absorbs them at no makespan cost.
        * send-bound isolated fan-out (source is an intra send hotspot) -> a binomial broadcast
          TREE (spreads that source's send load), edges carry precedence so a child forwards only
          after it received.

    DIRECT deliveries of the same (identity, src, dst) are DEDUPED: one physical transfer of an
    identity to a GPU satisfies every demand wanting it there (an egress_stage relay 5->4 and the
    internal-allgather self_distribution 5->4 are the same send). The merged delivery takes the max
    volume, the earliest release, and -- if any contributor is hard -- the tightest hard deadline.
    Tree-edge jobs are not deduped (they carry a precedence chain and only arise for isolated
    fan-outs where no overlapping direct delivery exists)."""
    max_recv = _max_intra_recv_load(demands)
    if debug:
        by_kind = defaultdict(int)
        for d in demands:
            by_kind[d.kind] += 1
        _p(debug, f"  [_to_jobs] {len(demands)} demands "
                  f"({dict(by_kind)}), max intra-recv load = {max_recv:g}")
    # merged direct deliveries keyed by (identity, src, dst)
    direct: Dict[Tuple[Identity, int, int], Dict] = {}
    tree_jobs: List[_Job] = []

    def _add_direct(identity, src, dst, volume, release, deadline, hard, kind):
        key = (identity, src, dst)
        e = direct.get(key)
        if e is None:
            direct[key] = dict(volume=volume, release=release,
                               deadline=(deadline if hard else inf), hard=hard, kind=kind)
        else:
            _p(debug, f"      dedup delivery {src}->{dst} id={identity}: "
                      f"{e['kind']} + {kind} (one send satisfies both)")
            e["volume"] = max(e["volume"], volume)
            e["release"] = min(e["release"], release)
            e["hard"] = e["hard"] or hard
            if hard:
                e["deadline"] = min(e["deadline"], deadline)

    for d in demands:
        if d.kind == "egress_stage":
            (gw,) = d.dst_gpus
            _add_direct(d.identity, d.src_gpu, gw, d.volume, PROLOGUE_BAND, d.deadline_epoch,
                        True, d.kind)
            continue

        # fan-out demand (ingress_distribution or self_distribution)
        wanters = [t for t in d.dst_gpus if t != d.src_gpu]
        if not wanters:
            continue
        # A network piece lands at the END of its arrival epoch, so the first band that can fan it
        # out is the next one. Anything sourced from native data exists before the collective even
        # starts, which is what makes the prologue available to it.
        release = d.deadline_epoch + 1 if d.kind == "ingress_distribution" else PROLOGUE_BAND
        # ingress demand may be promoted to hard by a caller (transit cell); default soft.
        hard = getattr(d, "hard", False)

        send_load = _intra_send_load(d.src_gpu, demands)
        dense = switch_copy or send_load <= max_recv + EPS
        if debug:
            why = "switch_copy" if switch_copy else f"send={send_load:g} {'<=' if dense else '>'} max_recv={max_recv:g}"
            _p(debug, f"    fan-out id={d.identity} {d.kind} src={d.src_gpu} -> "
                      f"{len(wanters)} wanters: {'DIRECT/ring' if dense else 'binomial TREE'} ({why})")
        if dense:
            for w in wanters:
                _add_direct(d.identity, d.src_gpu, w, d.volume, release, d.deadline_epoch,
                            hard, d.kind)
        else:
            deadline = d.deadline_epoch if hard else inf
            # tree edges reference the delivering job for precedence (a child forwards only after
            # it received).
            deliver_job: Dict[int, _Job] = {}   # child gpu -> the job that delivered to it
            for (parent, child, birth) in _binomial_tree_edges(d.src_gpu, wanters):
                pred = deliver_job.get(parent)   # None if parent is the root (has data at start)
                job = _Job(identity=d.identity, src=parent, dst=child, volume=d.volume,
                           release_gap=release, deadline_gap=deadline, hard=hard,
                           kind=d.kind, predecessor=pred)
                tree_jobs.append(job)
                deliver_job[child] = job
                _p(debug, f"        tree edge r{birth}: {parent}->{child}"
                          f"{'' if pred is None else f' (after {parent} receives)'}")

    jobs = [_Job(identity=k[0], src=k[1], dst=k[2], volume=e["volume"],
                 release_gap=e["release"], deadline_gap=e["deadline"], hard=e["hard"],
                 kind=e["kind"])
            for k, e in direct.items()]
    jobs += tree_jobs
    jobs = _coalesce_subchunks(jobs, subdivision, debug=debug)
    if debug:
        nh = sum(1 for j in jobs if j.hard)
        _p(debug, f"  [_to_jobs] -> {len(jobs)} jobs ({nh} hard, {len(jobs) - nh} soft; "
                  f"{len(tree_jobs)} tree)")
    return jobs


# --------------------------------------------------------------------------------------------
# Step 2: the EDF-weighted greedy b-matching scheduler (one band's worth of jobs)
# --------------------------------------------------------------------------------------------
def _max_port_load(jobs: Sequence[_Job]) -> float:
    """The Birkhoff-von Neumann / open-shop makespan lower bound: the busiest GPU egress or ingress
    total. A port-bound schedule needs exactly ceil(this / port_cap) rounds."""
    send: Dict[int, float] = defaultdict(float)
    recv: Dict[int, float] = defaultdict(float)
    for j in jobs:
        send[j.src] += j.volume
        recv[j.dst] += j.volume
    return max([*send.values(), *recv.values()], default=0.0)


def _schedule_band(jobs: List[_Job], gpus: Sequence[int], switch: int, band: int,
                   port_cap: float = 1.0, ring_hint: Optional[Callable] = None,
                   max_rounds: int = 10_000, cell_id: Optional[int] = None,
                   debug: bool = False) -> List[IntraFlow]:
    """Schedule one band's jobs into rounds on a non-blocking crossbar. Each round runs a
    priority-weighted greedy b-matching bounded by per-GPU egress/ingress capacity (`port_cap`,
    uniform on an NVSwitch). Priority key: hard first, then earliest deadline, then a ring-distance
    tiebreak (so a symmetric all-to-all edge-colors into the canonical ring), then (src, dst).

    A job is scheduled ATOMICALLY: it starts at the first round where both its ports have room and
    holds them for `span = ceil(volume / port_cap)` consecutive rounds. That matters for a
    coalesced multi-sub-chunk transfer -- splitting it back across non-adjacent rounds would defeat
    the coalescing, since the sub-chunks then land in different fine epochs again. Jobs smaller
    than a port still share a round (two 0.5s pack together), so the packing behaviour is
    unchanged for everything that is not a multi-round transfer.

    Mutates each job's `completion_round`. Returns the emitted IntraFlows (logical GPU->GPU with
    the cell's switch recorded), one per job."""
    if ring_hint is None:
        ring_hint = lambda s, d: _ring_distance(s, d, gpus)

    def _key(j: _Job):
        return (not j.hard, j.deadline_gap, ring_hint(j.src, j.dst), j.src, j.dst)

    lb = math.ceil(_max_port_load(jobs) / port_cap - EPS) if jobs else 0
    if debug:
        nh = sum(1 for j in jobs if j.hard)
        _p(debug, f"    [band {band}] {len(jobs)} jobs ({nh} hard), "
                  f"max-port-load lower bound = {lb} round(s)")

    def _span(volume: float) -> int:
        return max(1, math.ceil(volume / port_cap - EPS))

    def _needs(volume: float, span: int) -> List[float]:
        """Port capacity this job consumes in each round of its span (full rounds, then the tail)."""
        return [min(port_cap, volume - i * port_cap) for i in range(span)]

    # (gpu, round) -> capacity already committed. Sparse: only touched rounds appear.
    egress_used: Dict[Tuple[int, int], float] = defaultdict(float)
    ingress_used: Dict[Tuple[int, int], float] = defaultdict(float)

    flows: List[IntraFlow] = []
    pending = list(jobs)
    for r in range(max_rounds):
        if not pending:
            break
        ready = [j for j in pending
                 if j.predecessor is None
                 or (j.predecessor.completion_round is not None
                     and j.predecessor.completion_round < r)]
        if not ready:
            # Every remaining job waits on a predecessor that will never complete -- only possible
            # if a precedence chain is cyclic (never emitted here). Surface it loudly.
            raise RuntimeError(
                f"intra-cell schedule stalled at round {r}, band {band}: "
                f"{[(j.src, j.dst, j.kind) for j in pending]}")
        ready.sort(key=_key)
        matched: List[str] = []
        for j in ready:
            span, = (_span(j.volume),)
            needs = _needs(j.volume, span)
            if not all(egress_used[(j.src, r + i)] + needs[i] <= port_cap + EPS
                       and ingress_used[(j.dst, r + i)] + needs[i] <= port_cap + EPS
                       for i in range(span)):
                continue      # a port is busy in one of the rounds this job would span
            for i in range(span):
                egress_used[(j.src, r + i)] += needs[i]
                ingress_used[(j.dst, r + i)] += needs[i]
            flows.append(IntraFlow(cell=cell_id, identity=j.identity, sender=j.src, receiver=j.dst,
                                   via_switch=switch, volume=j.volume, band=band, local_round=r,
                                   span=span, kind=j.kind, hard=j.hard, identities=j.identities))
            j.remaining = 0.0
            j.completion_round = r + span - 1
            pending.remove(j)
            if debug:
                tag = "H" if j.hard else " "
                label = (f"id{j.identity[0]}x{len(j.identities)}" if len(j.identities) > 1
                         else f"id{j.identity[0]}")
                matched.append(f"{j.src}->{j.dst}{tag}({label},{j.volume:g}"
                               f"{f',{span}rnds' if span > 1 else ''})")
        if debug and matched:
            _p(debug, f"      round {r}: " + "  ".join(matched))
    else:
        raise RuntimeError(f"intra-cell schedule exceeded {max_rounds} rounds in band {band}")
    if debug:
        used = rounds_in(flows)
        verdict = "OPTIMAL (= port bound)" if used == lb else (
            f"above bound (tree/precedence depth)" if used > lb else "below bound?!")
        _p(debug, f"    [band {band}] used {used} round(s) vs bound {lb} -> {verdict}")
    return flows


def _assert_ports(flows: Sequence[IntraFlow], port_cap: float = 1.0) -> None:
    """No GPU egress or ingress link carries more than port_cap in any (band, round).

    A flow occupies its ports for its whole span, so a multi-round transfer is charged to each
    round it covers -- port_cap per full round and the remainder in the tail."""
    eg: Dict[Tuple[int, int, int], float] = defaultdict(float)
    ing: Dict[Tuple[int, int, int], float] = defaultdict(float)
    for f in flows:
        for i in range(f.span):
            share = min(port_cap, f.volume - i * port_cap)
            eg[(f.band, f.local_round + i, f.sender)] += share
            ing[(f.band, f.local_round + i, f.receiver)] += share
    for k, v in eg.items():
        assert v <= port_cap + 1e-6, f"egress over-subscribed {k}: {v}"
    for k, v in ing.items():
        assert v <= port_cap + 1e-6, f"ingress over-subscribed {k}: {v}"


def _assert_deadlines(jobs: Sequence[_Job]) -> None:
    """Every hard job finished (and, once absolute epochs exist, before its deadline). At the
    round level we can only check completion here; the band-level deadline is enforced by
    _assign_bands, which never places a job in a band at or after its deadline."""
    for j in jobs:
        if j.hard:
            assert j.completion_round is not None, f"hard job never completed: {j.src}->{j.dst}"


# --------------------------------------------------------------------------------------------
# Step 3: per-cell orchestration
# --------------------------------------------------------------------------------------------
def _assign_bands(jobs: Sequence[_Job]) -> Dict[int, List[_Job]]:
    """Group this cell's jobs by band, applying the shared policy (teccl.hierarchy.bands.band_of).

    Applied at the JOB level rather than the demand level because `_to_jobs` has to run first: its
    dedup merges deliveries that several demands share (taking the earliest release and tightest
    deadline, so the merged job's band is the right one for all of them) and its density test needs
    the whole cell's load. The level boundary applies the same rule one step earlier, to demands --
    see the bands module docstring for why both call sites exist.
    """
    by_band: Dict[int, List[_Job]] = defaultdict(list)
    for j in jobs:
        what = f"job {j.src}->{j.dst} ({j.kind}, identity {j.identity})"
        by_band[band_of(j.release_gap, j.deadline_gap, j.hard, what)].append(j)
    return dict(by_band)


def schedule_cell(cell_id: int, cell: Cell, demands: Sequence[IntraCellDemand],
                  switch_copy: bool = False, ring_hint: Optional[Callable] = None,
                  port_cap: float = 1.0, debug: Optional[bool] = None,
                  subdivision: int = 1, topology=None) -> List[IntraFlow]:
    """Schedule every intra-cell demand of one cell onto its internal NVSwitch and return the fine
    IntraFlows. The switch is the cell's single internal switch (the memoized full-mesh case).

    ring_hint(src, dst) -> comparable, optional override of the default ring-distance tiebreak (the
    home for a phase-2-supplied ordering knob). Flows carry (band, local_round) -- this level's
    complete fine schedule, in units the stitch can place without rescaling.

    `topology` is accepted and unused: this row reads everything it needs off the `Cell` (its gpu
    order and its one internal switch). It is in the signature so the closed-form rows are
    interchangeable at the dispatch point -- the ring row does need the capacity view, to tell a
    physical ring from a logical one.

    debug: None -> honor the TECCL_INTRA_DEBUG env var; True/False -> force. Debug narrates every
    step (job build, fan-out density, dedup, per-round matching, optimality-vs-bound)."""
    debug = _ENV_DEBUG if debug is None else debug
    assert len(cell.internal_switches) == 1, (
        f"cell {cell_id} has {len(cell.internal_switches)} internal switches; the memoized "
        f"generator handles the single-NVSwitch case only")
    switch = cell.internal_switches[0]
    _p(debug, f"\n=== schedule_cell {cell_id}: gpus={list(cell.gpus)} nvswitch={switch}, "
              f"{len(demands)} intra demands ===")
    jobs = _to_jobs(demands, cell, switch_copy, debug=debug, subdivision=subdivision)
    flows: List[IntraFlow] = []
    for band, bjobs in sorted(_assign_bands(jobs).items()):
        flows += _schedule_band(bjobs, cell.gpus, switch, band, port_cap=port_cap,
                                ring_hint=ring_hint, cell_id=cell_id, debug=debug)
    _assert_deadlines(jobs)
    _assert_ports(flows, port_cap=port_cap)
    if debug:
        bands = sorted({f.band for f in flows})
        peak = max(band_rounds(flows).values(), default=0)
        _p(debug, f"=== cell {cell_id}: {len(flows)} flows across bands {bands}, "
                  f"peak {peak} rounds/band; all hard deadlines met, ports within cap ===")
    return flows


# --------------------------------------------------------------------------------------------
# Dispatch: is this level a crossbar, and if so, what does its solved routing look like?
# --------------------------------------------------------------------------------------------
def is_crossbar(topology: "Topology") -> bool:
    """Does this level's graph consist of data nodes hanging off exactly one shared switch?

    That is the shape whose optimal schedule is known in closed form, so it is the shape the
    base-case dispatcher may route here instead of to a real formulation. Three conditions, all
    necessary: exactly one switch (two switches means a routing choice), every data node bidirection-
    ally attached to it (otherwise some pair has no path and the closed form is simply wrong), and no
    direct data-to-data link (a link the crossbar schedule would leave unused, so the closed form
    would no longer be optimal -- it would still be CORRECT, but silently pessimistic, and a level
    that quietly gives up bandwidth is worse than one that admits it needs a solver).
    """
    n = len(topology.capacity)
    switches = list(topology.switch_indices)
    if len(switches) != 1:
        return False
    sw = switches[0]
    passive = set(getattr(topology, "passive_indices", []))
    data = [i for i in range(n) if i != sw and i not in passive]
    if not data:
        return False
    for i in data:
        if topology.capacity[i][sw] <= 0 or topology.capacity[sw][i] <= 0:
            return False
        for j in data:
            if i != j and topology.capacity[i][j] > 0:
                return False
    return True


def crossbar_routing(coarse, mapping,
                     id_sets: Dict[Tuple[int, int], List[Identity]]
                     ) -> List[Tuple[object, Identity]]:
    """This level's ROUTING decision, as (coarse piece, identity) pairs for step A.

    On a crossbar there is nothing to route: every delivery is `U -> switch -> V`. So unlike a
    formulation level there is no model to solve and no solved model to walk paths back out of --
    this just restates the level's own demand in the piece vocabulary
    `reconstruct.assign_identities_preserving` consumes, keeping each identity attached so the level
    spends nothing from the refinement budget (see that function's docstring).

    Driven off `id_sets` rather than off IntraCellDemands so that it reads the same at every depth:
    the ROOT has no IntraCellDemands at all (nothing above it has resolved anything yet, so its
    demand is still a tensor), and `identity_sets` is the one description both the root and a child
    level always have. Same-cell deliveries never appear in `id_sets` by construction, which is
    correct -- they do not cross this level's fabric, and `build_child_problems` re-emits them one
    level down as self_distribution.

    Epochs ARE decided here, by the same edge-colouring the round scheduler uses one granularity
    down (`_colour_epochs`). A level owns its own timing -- that is the recursion contract -- and
    leaving every piece in epoch 0 is not "deferring the decision", it is asserting that a coarse
    node can transmit its entire payload in one epoch, which immediately trips the level's own
    capacity check (`_assert_rate_within_capacity`) because each flow is paced to fill exactly one
    epoch.
    """
    from teccl.hierarchy.reconstruct import make_piece
    switches = list(coarse.switch_indices)
    assert len(switches) == 1, (
        f"crossbar_routing needs exactly one switch on the level's graph, got {switches}")
    coarse_sw = switches[0]
    fine_sw = mapping.coarse_passthrough[coarse_sw]

    pairs: List[Tuple[int, int]] = []
    carried: List[Identity] = []
    for (u, v), identities in sorted(id_sets.items()):
        for ident in identities:
            pairs.append((u, v))
            carried.append(ident)

    epochs = _colour_epochs(pairs)
    return [(make_piece(src_cell=u, dst_cell=v,
                        egress_neighbor=coarse_sw, ingress_neighbor=coarse_sw,
                        via_switches=(fine_sw,), volume=1.0,
                        send_epoch=k, arrival_epoch=k), ident)
            for (u, v), ident, k in zip(pairs, carried, epochs)]


def _colour_epochs(pairs: Sequence[Tuple[int, int]]) -> List[int]:
    """Assign each (sender, receiver) transfer an epoch, one transfer per port per epoch.

    This is the SAME Birkhoff-von Neumann edge-colouring the round scheduler does, applied one
    granularity up: on a crossbar the only contention is each node's single egress port and single
    ingress port, so a set of transfers is schedulable in epoch k exactly when it is a matching.
    First-fit on the smallest epoch free at BOTH endpoints is the standard greedy; on the symmetric
    all-to-all demand this level actually sees, it recovers the optimal max-port-load makespan, the
    same way the round scheduler recovers the ring.

    A crossbar level has no reason to reach for anything cleverer: the optimum here is the busiest
    port, and the greedy meets it whenever the demand is regular.
    """
    send_used: Dict[int, set] = defaultdict(set)
    recv_used: Dict[int, set] = defaultdict(set)
    out: List[int] = []
    for (u, v) in pairs:
        k = 0
        while k in send_used[u] or k in recv_used[v]:
            k += 1
        send_used[u].add(k)
        recv_used[v].add(k)
        out.append(k)
    return out
