"""
Phase-3 intra-cell scheduler for the hierarchical solver.

Identity resolution (teccl.hierarchy.reconstruct) turns the identity-free coarse LP solution into
concrete inter-cell pieces plus a list of IntraCellDemand descriptors that must be satisfied INSIDE
each cell before/after the inter-cell sends. This module schedules those intra-cell demands onto the
cell's internal fabric (a single non-blocking NVSwitch for the memoized case) and emits fine flows.

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

Timeline/banding (which absolute fine epoch a round becomes, band width, empty-epoch compaction) is
NOT decided here -- the scheduler emits (gap, local_round) and the downstream stitch step maps that
to absolute fine epochs. `_group_by_gap` is the single seam that encodes the (v1: per-gap) timeline
policy; swapping it for a full-timeline policy later leaves the scheduler and its tests untouched.
"""
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from math import inf
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    """One lowered intra-cell hop, logical GPU->GPU (expanded to GPU->switch->GPU at stitch time).

    gap / local_round are relative coordinates: `gap` is the coarse-epoch band this transfer was
    scheduled into and `local_round` is its round index within that band. Absolute fine-epoch
    numbering is a stitch concern, deliberately not decided here.
    """
    cell: int
    identity: Identity
    sender: int                 # fine GPU id
    receiver: int               # fine GPU id
    via_switch: int             # the cell's internal NVSwitch id
    volume: float
    gap: int
    local_round: int


@dataclass
class _Job:
    """A single point-to-point transfer the scheduler must place into rounds.

    remaining/ completion tracks mutate during scheduling. `predecessor` references another _Job
    that must complete in a strictly earlier round before this job becomes ready (used only by
    broadcast-tree lowering, where a child forwards only after it has received). Referencing the
    parent job by object (not list index) keeps precedence intact when jobs are regrouped by gap."""
    identity: Identity
    src: int
    dst: int
    volume: float
    release_gap: int
    deadline_gap: float          # inf for soft
    hard: bool
    kind: str                    # provenance, for debugging
    predecessor: "Optional[_Job]" = None
    remaining: float = field(default=None)   # type: ignore[assignment]
    completion_round: Optional[int] = None

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.volume


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


def _egress_load(gateway: int, demands: Sequence[IntraCellDemand]) -> float:
    """Total volume gateway must SEND across all intra demands (for the fan-out density test)."""
    tot = 0.0
    for d in demands:
        if d.src_gpu == gateway:
            tot += d.volume * max(1, len(d.dst_gpus))
        # egress_stage relays land ON the gateway (it later egresses them to the network), so they
        # add to its send pressure inside the cell only indirectly; the dominant term is fan-out.
    return tot


def _max_ingress_load(demands: Sequence[IntraCellDemand]) -> float:
    """Max volume any single GPU must RECEIVE across all fan-out demands (the ingress bound)."""
    recv: Dict[int, float] = defaultdict(float)
    for d in demands:
        for t in d.dst_gpus:
            recv[t] += d.volume
    return max(recv.values(), default=0.0)


def _to_jobs(demands: Sequence[IntraCellDemand], cell: Cell,
             switch_copy: bool = False, debug: bool = False) -> List[_Job]:
    """Convert a cell's IntraCellDemand list into scheduler jobs.

    egress_stage        -> a HARD point-to-point delivery (native -> gateway), deadline = its epoch.
    ingress_distribution / self_distribution -> a fan-out, lowered by a density test:
        * ingress-bound (E_gateway <= I_max) or switch multicast -> N DIRECT edges (the dense
          allgather/alltoall case; the scheduler edge-colors them into the ring). switch_copy keeps
          it a single logical send only when we later model multicast at emit time; for now the
          unicast direct edges are emitted and the ring absorbs them at no makespan cost.
        * egress-bound isolated fan-out -> a binomial broadcast TREE (spreads the single-gateway
          egress), edges carry precedence so a child forwards only after it received.

    DIRECT deliveries of the same (identity, src, dst) are DEDUPED: one physical transfer of an
    identity to a GPU satisfies every demand wanting it there (an egress_stage relay 5->4 and the
    internal-allgather self_distribution 5->4 are the same send). The merged delivery takes the max
    volume, the earliest release, and -- if any contributor is hard -- the tightest hard deadline.
    Tree-edge jobs are not deduped (they carry a precedence chain and only arise for isolated
    fan-outs where no overlapping direct delivery exists)."""
    i_max = _max_ingress_load(demands)
    if debug:
        by_kind = defaultdict(int)
        for d in demands:
            by_kind[d.kind] += 1
        _p(debug, f"  [_to_jobs] {len(demands)} demands "
                  f"({dict(by_kind)}), I_max ingress load = {i_max:g}")
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
            _add_direct(d.identity, d.src_gpu, gw, d.volume, 0, d.deadline_epoch, True, d.kind)
            continue

        # fan-out demand (ingress_distribution or self_distribution)
        wanters = [t for t in d.dst_gpus if t != d.src_gpu]
        if not wanters:
            continue
        release = d.deadline_epoch if d.kind == "ingress_distribution" else 0
        # ingress demand may be promoted to hard by a caller (transit cell); default soft.
        hard = getattr(d, "hard", False)

        e_gateway = _egress_load(d.src_gpu, demands)
        dense = switch_copy or e_gateway <= i_max + EPS
        if debug:
            why = "switch_copy" if switch_copy else f"E_gw={e_gateway:g} {'<=' if dense else '>'} I_max={i_max:g}"
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
    if debug:
        nh = sum(1 for j in jobs if j.hard)
        _p(debug, f"  [_to_jobs] -> {len(jobs)} jobs ({nh} hard, {len(jobs) - nh} soft; "
                  f"{len(direct)} direct, {len(tree_jobs)} tree)")
    return jobs


# --------------------------------------------------------------------------------------------
# Step 2: the EDF-weighted greedy b-matching scheduler (one gap's worth of jobs)
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


def _schedule_gap(jobs: List[_Job], gpus: Sequence[int], switch: int, gap: int,
                  port_cap: float = 1.0, ring_hint: Optional[Callable] = None,
                  max_rounds: int = 10_000, cell_id: Optional[int] = None,
                  debug: bool = False) -> List[IntraFlow]:
    """Schedule one gap's jobs into rounds on a non-blocking crossbar. Each round runs a
    priority-weighted greedy b-matching bounded by per-GPU egress/ingress capacity (`port_cap`,
    uniform on an NVSwitch). Priority key: hard first, then earliest deadline, then a ring-distance
    tiebreak (so a symmetric all-to-all edge-colors into the canonical ring), then (src, dst).

    Mutates each job's `remaining` / `completion_round`. Returns the emitted IntraFlows (logical
    GPU->GPU with the cell's switch recorded)."""
    if ring_hint is None:
        ring_hint = lambda s, d: _ring_distance(s, d, gpus)

    def _key(j: _Job):
        return (not j.hard, j.deadline_gap, ring_hint(j.src, j.dst), j.src, j.dst)

    lb = math.ceil(_max_port_load(jobs) / port_cap - EPS) if jobs else 0
    if debug:
        nh = sum(1 for j in jobs if j.hard)
        _p(debug, f"    [gap {gap}] {len(jobs)} jobs ({nh} hard), "
                  f"max-port-load lower bound = {lb} round(s)")

    flows: List[IntraFlow] = []
    for r in range(max_rounds):
        ready = [j for j in jobs
                 if j.remaining > EPS
                 and (j.predecessor is None
                      or (j.predecessor.completion_round is not None
                          and j.predecessor.completion_round < r))]
        if not ready:
            break
        ready.sort(key=_key)
        egress_free: Dict[int, float] = {g: port_cap for g in gpus}
        ingress_free: Dict[int, float] = {g: port_cap for g in gpus}
        progressed = False
        matched: List[str] = []
        for j in ready:
            a = min(j.remaining, egress_free[j.src], ingress_free[j.dst])
            if a <= EPS:
                continue
            flows.append(IntraFlow(cell=cell_id, identity=j.identity, sender=j.src, receiver=j.dst,
                                   via_switch=switch, volume=a, gap=gap, local_round=r))
            j.remaining -= a
            egress_free[j.src] -= a
            ingress_free[j.dst] -= a
            progressed = True
            if j.remaining <= EPS:
                j.completion_round = r
            if debug:
                tag = "H" if j.hard else " "
                done = "done" if j.remaining <= EPS else f"rem {j.remaining:g}"
                matched.append(f"{j.src}->{j.dst}{tag}(id{j.identity[0]},{a:g},{done})")
        if debug and matched:
            _p(debug, f"      round {r}: " + "  ".join(matched))
        if not progressed:
            # No ready job could claim any port -- only possible if a predecessor chain is
            # unsatisfiable (a cyclic tree, never emitted here). Surface it loudly.
            unresolved = [(j.src, j.dst, j.kind) for j in jobs if j.remaining > EPS]
            raise RuntimeError(f"intra-cell schedule stalled at round {r}, gap {gap}: {unresolved}")
    else:
        raise RuntimeError(f"intra-cell schedule exceeded {max_rounds} rounds in gap {gap}")
    if debug:
        used = (max((f.local_round for f in flows), default=-1) + 1)
        verdict = "OPTIMAL (= port bound)" if used == lb else (
            f"above bound (tree/precedence depth)" if used > lb else "below bound?!")
        _p(debug, f"    [gap {gap}] used {used} round(s) vs bound {lb} -> {verdict}")
    return flows


def _assert_ports(flows: Sequence[IntraFlow], port_cap: float = 1.0) -> None:
    """No GPU egress or ingress link carries more than port_cap in any (gap, round)."""
    eg: Dict[Tuple[int, int, int], float] = defaultdict(float)
    ing: Dict[Tuple[int, int, int], float] = defaultdict(float)
    for f in flows:
        eg[(f.gap, f.local_round, f.sender)] += f.volume
        ing[(f.gap, f.local_round, f.receiver)] += f.volume
    for k, v in eg.items():
        assert v <= port_cap + 1e-6, f"egress over-subscribed {k}: {v}"
    for k, v in ing.items():
        assert v <= port_cap + 1e-6, f"ingress over-subscribed {k}: {v}"


def _assert_deadlines(jobs: Sequence[_Job]) -> None:
    """Every hard job finished (and, once absolute epochs exist, before its deadline). At the
    round level we can only check completion here; the gap-level deadline is enforced by grouping."""
    for j in jobs:
        if j.hard:
            assert j.completion_round is not None, f"hard job never completed: {j.src}->{j.dst}"


# --------------------------------------------------------------------------------------------
# Step 3: per-cell orchestration
# --------------------------------------------------------------------------------------------
def _group_by_gap(jobs: Sequence[_Job]) -> Dict[int, List[_Job]]:
    """v1 timeline policy (the deferred seam): pin each job to a SINGLE coarse-epoch gap -- hard
    jobs to their deadline gap (they must complete before that egress), soft jobs to their release
    gap. A full-timeline policy (spread a job's window across gaps) would replace only this
    function; the scheduler and its tests are unaffected. A fan-out's tree jobs share a
    release/deadline, so a precedence chain never straddles gaps."""
    by_gap: Dict[int, List[_Job]] = defaultdict(list)
    for j in jobs:
        if j.hard and j.deadline_gap != inf:
            gap = int(j.deadline_gap)
        else:
            gap = int(j.release_gap)
        by_gap[gap].append(j)
    return dict(by_gap)


def schedule_cell(cell_id: int, cell: Cell, demands: Sequence[IntraCellDemand],
                  switch_copy: bool = False, ring_hint: Optional[Callable] = None,
                  port_cap: float = 1.0, debug: Optional[bool] = None) -> List[IntraFlow]:
    """Schedule every intra-cell demand of one cell onto its internal NVSwitch and return the fine
    IntraFlows. The switch is the cell's single internal switch (the memoized full-mesh case).

    ring_hint(src, dst) -> comparable, optional override of the default ring-distance tiebreak (the
    home for a phase-2-supplied ordering knob). Timeline/banding is deferred: flows carry
    (gap, local_round), not absolute fine epochs.

    debug: None -> honor the TECCL_INTRA_DEBUG env var; True/False -> force. Debug narrates every
    step (job build, fan-out density, dedup, per-round matching, optimality-vs-bound)."""
    debug = _ENV_DEBUG if debug is None else debug
    assert len(cell.internal_switches) == 1, (
        f"cell {cell_id} has {len(cell.internal_switches)} internal switches; the memoized "
        f"generator handles the single-NVSwitch case only")
    switch = cell.internal_switches[0]
    _p(debug, f"\n=== schedule_cell {cell_id}: gpus={list(cell.gpus)} nvswitch={switch}, "
              f"{len(demands)} intra demands ===")
    jobs = _to_jobs(demands, cell, switch_copy, debug=debug)
    flows: List[IntraFlow] = []
    for gap, gjobs in sorted(_group_by_gap(jobs).items()):
        flows += _schedule_gap(gjobs, cell.gpus, switch, gap, port_cap=port_cap,
                               ring_hint=ring_hint, cell_id=cell_id, debug=debug)
    _assert_deadlines(jobs)
    _assert_ports(flows, port_cap=port_cap)
    if debug:
        gaps = sorted({f.gap for f in flows})
        peak = max((max((f.local_round for f in flows if f.gap == g), default=-1) + 1
                    for g in gaps), default=0)
        _p(debug, f"=== cell {cell_id}: {len(flows)} flows across {len(gaps)} gap(s), "
                  f"peak {peak} rounds/gap; all hard deadlines met, ports within cap ===")
    return flows
