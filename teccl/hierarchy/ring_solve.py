"""
The RING level solver: a closed-form (Gurobi-free) solve for a cell scheduled as a ring.

This is a SECOND MEMOIZED ROW in the recursion's base-case dispatch table
(teccl.hierarchy.solve.solve_flat / _solve_base), exactly parallel to
teccl.hierarchy.crossbar_solve. Both answer the same question -- "schedule this set of
IntraCellDemands on this cell's internal fabric and emit fine IntraFlows in (band, local_round)" --
and both own their own fine schedule completely, which is the recursion contract. They differ only
in the communication STRUCTURE they impose, and therefore in what contention they have to model:

    crossbar_solve   every GPU can reach every other in one hop through the switch. Contention is
                     each GPU's single egress and single ingress PORT, so scheduling is a
                     Birkhoff-von Neumann edge-colouring and a fan-out is lowered per demand by a
                     density test (direct star vs binomial tree).
    ring_solve       a GPU talks only to its ring NEIGHBOURS. Contention is each directed ring
                     EDGE, a delivery is a store-and-forward hop chain, and a fan-out is lowered
                     uniformly as a ring multicast -- no per-demand choice at all.

WHY A RING AT ALL, WHEN THE CROSSBAR CASE IS ALREADY OPTIMAL ON MAKESPAN. Because makespan is not
the only thing the schedule is judged on. The crossbar's star fan-out gives every GPU 7 distinct
inbound peers on an 8-GPU host, and although the scheduler's per-round matching means only one of
them is meant to be sending at a time, that matching is a COSTING model: `local_round` sets the
absolute fine epoch and the within-connection order, and then it is gone -- the emitted XML carries
no cross-threadblock ordering between a receiver's inbound connections, so nothing stops all 7
firing at once. A ring makes the property structural instead of scheduled: one inbound and one
outbound connection per GPU, so N:1 fan-in cannot be expressed, whatever the runtime does with the
ordering. That is precisely the reason NCCL's bandwidth-optimal intra-node path is a ring (per
channel, each GPU has exactly one predecessor and one successor, and parallelism comes from laying
down several independent channels rather than from fanning in).

WHAT IT COSTS, MEASURED IN THE MODEL RATHER THAN ASSERTED. For a full fan-out (the allgather shape)
the ring is FREE: a star from one source is N-1 sends off one egress port, a ring broadcast is N-1
sends spread one per port, and with several identities in flight the two have the same busiest-link
load and therefore the same round count. For a point-to-point shape (the alltoall shape) the ring is
strictly WORSE on a crossbar, because a direct hop exists and the ring pays `distance(src, dst)`
hops instead of one -- on 8 GPUs that is ~4x the traffic. Both effects fall out of the link bound
below and are visible in the debug summary; this module does not try to hide the second one, because
the point of having both rows in the table is to be able to measure the trade.

SELECTION. Two ways in, mirroring the crossbar's `is_crossbar`:
  * `is_ring(topology)` -- the graph IS physically a ring (no switch, every data node with exactly
    two undirected neighbours forming one cycle). Then there is no choice to make, and the claim is
    made at EVERY dispatch point: a ring is a ring whether it is a level's graph (`ring_routing`
    supplies that level's pieces and epochs) or a bottom cell's interior (`schedule_cell` supplies
    its fine schedule). Neither row is restricted to a depth.
  * the ALGORITHM FLAG -- `TECCL_INTRA_ALGO=ring` (or setting the module-level `INTRA_ALGO`) forces
    a single-switch crossbar NODE onto a LOGICAL ring over `cell.gpus`, so the NVSwitch case can be
    A/B'd against the crossbar path without changing any topology. It is scoped to a node's fabric
    because that is what it selects; `should_use_ring` documents why a crossbar LEVEL is left alone.

The logical ring is deliberately UNIDIRECTIONAL. On a crossbar a GPU has one egress port into the
switch, so giving it two ring out-edges would let two logical edges oversubscribe one physical port;
with one out-edge the ring edge and the port are the same resource and tracking either is exact. A
PHYSICAL bidirectional ring is different -- the two out-edges are distinct links -- so there both
directions are used and a delivery takes the shorter way round.
"""
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from math import inf
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from teccl.hierarchy.bands import PROLOGUE_BAND
from teccl.hierarchy.cell import Cell
# _Job and _assign_bands are shared with the crossbar row ON PURPOSE. A job is "one point-to-point
# transfer with a release, a deadline and an optional predecessor", which is the same object
# whatever fabric places it, and the band POLICY has to be identical across rows or the two
# base-case solvers would put the same work in different coarse epochs -- the exact drift
# teccl/hierarchy/bands.py exists to prevent. IntraFlow is likewise the one output type the stitch
# reads, so re-declaring it here would fork the schedule format.
from teccl.hierarchy.crossbar_solve import (IntraFlow, _Job, _assign_bands, band_rounds,
                                            is_crossbar, rounds_in)
from teccl.hierarchy.reconstruct import Identity, IntraCellDemand

EPS = 1e-9

# --------------------------------------------------------------------------------------------
# The algorithm flag
# --------------------------------------------------------------------------------------------
ALGO_CROSSBAR = "crossbar"
ALGO_RING = "ring"
_ALGOS = (ALGO_CROSSBAR, ALGO_RING)

# Which base-case solver a SINGLE-SWITCH (crossbar) cell is routed to. A physical ring always uses
# the ring row regardless of this; the flag exists only to force the NVSwitch case over so the two
# lowerings can be compared on identical input.
#
# Read from the environment at import, and left as a module global so a test (or a driver) can flip
# it in-process: `ring_solve.INTRA_ALGO = ring_solve.ALGO_RING`. Every read goes through
# `intra_algo()` rather than capturing the value, so a later assignment takes effect.
INTRA_ALGO = (os.environ.get("TECCL_INTRA_ALGO", "").strip().lower() or ALGO_CROSSBAR)

_ENV_DEBUG = os.environ.get("TECCL_INTRA_DEBUG", "").lower() in ("1", "true", "yes", "on")


def _p(debug: bool, msg: str = "") -> None:
    if debug:
        print(msg)


def intra_algo() -> str:
    """The configured base-case algorithm for a crossbar cell, validated on every read."""
    algo = (INTRA_ALGO or ALGO_CROSSBAR).strip().lower()
    if algo not in _ALGOS:
        raise ValueError(
            f"INTRA_ALGO / TECCL_INTRA_ALGO is {INTRA_ALGO!r}; expected one of {_ALGOS}")
    return algo


def ring_forced() -> bool:
    """Is the flag asking for a crossbar cell to be scheduled as a ring?"""
    return intra_algo() == ALGO_RING


def should_use_ring(topology, cell_fabric: bool = False) -> bool:
    """The dispatch predicate: does this graph belong to the ring row of the table?

    A physically-ring fabric has no alternative -- the crossbar closed form is simply wrong there,
    because it would emit one-hop transfers over links that do not exist -- so `is_ring` claims it
    at EVERY dispatch point, exactly as `is_crossbar` claims a switched one. Neither row is
    depth-restricted.

    `cell_fabric` says which dispatch point is asking: True for a bottom CELL's interior
    (`solve._solve_base`), False for a LEVEL's graph (`solve.solve_flat`). It exists only to scope
    the algorithm FLAG, which by its own definition selects the fabric of a single-NVSwitch NODE --
    the thing the flag is for is comparing the two lowerings of one host's interior. Shape-based
    selection ignores it entirely.

    A crossbar LEVEL is therefore left on the crossbar row even with the flag set. That is not a
    ranking of the rows: routing a level as a ring means multi-hop, which means an intermediate CELL
    receiving data and forwarding it onward, and that transit case is unimplemented across the whole
    lowering half (see `bands.band_of` and `reconstruct._build_slots`) rather than in this row. When
    a level's graph genuinely IS a ring, `ring_routing` routes it and that shared limitation is what
    raises -- identically to how it would raise for any other row that produced a multi-hop route.
    """
    return is_ring(topology) or (cell_fabric and ring_forced() and is_crossbar(topology))


# --------------------------------------------------------------------------------------------
# The ring order
# --------------------------------------------------------------------------------------------
CW, CCW = 1, -1


@dataclass(frozen=True)
class RingOrder:
    """The cyclic order transfers are scheduled around, plus how a hop is realized.

    `gpus` is the cycle: `gpus[i]`'s clockwise successor is `gpus[(i + 1) % n]`. `bidirectional`
    says whether the counter-clockwise edges are usable as INDEPENDENT links -- true only for a
    physical bidirectional ring, where they are distinct physical links, and false for a logical
    ring over a crossbar, where both directions would share one egress port (see the module
    docstring).

    `via_switch` is the cell's internal switch when the ring is LOGICAL (each hop is really
    gpu -> switch -> gpu, and the stitch has to record that switch so the emitted segment names the
    route the hardware takes), and None on a physical ring, where a hop is a direct link.
    """
    gpus: Tuple[int, ...]
    bidirectional: bool
    via_switch: Optional[int]
    # gpu -> its index in `gpus`. Derived, not input: cached because `position` is on the hot path
    # of every distance/routing query and a linear `.index()` there makes lowering quadratic in the
    # cell size. Excluded from equality and repr so the order still compares by its real content.
    _pos: Dict[int, int] = field(default=None, compare=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_pos", {g: i for i, g in enumerate(self.gpus)})

    @property
    def n(self) -> int:
        return len(self.gpus)

    def position(self, gpu: int) -> int:
        try:
            return self._pos[gpu]
        except KeyError:
            raise KeyError(
                f"gpu {gpu} is not on the ring {list(self.gpus)}") from None

    def node_at(self, src: int, offset: int, direction: int) -> int:
        """The node `offset` hops from `src` in `direction`."""
        return self.gpus[(self.position(src) + direction * offset) % self.n]

    def distance(self, src: int, dst: int, direction: int) -> int:
        """Hops from src to dst travelling in `direction` (0 when they are the same node)."""
        raw = (self.position(dst) - self.position(src)) % self.n
        return raw if direction == CW else (-raw) % self.n

    def directions(self) -> Tuple[int, ...]:
        return (CW, CCW) if self.bidirectional else (CW,)

    def best_direction(self, src: int, dst: int) -> int:
        """The direction that reaches dst in fewest hops (clockwise wins an exact tie)."""
        if not self.bidirectional:
            return CW
        return CW if self.distance(src, dst, CW) <= self.distance(src, dst, CCW) else CCW

    def edges(self) -> List[Tuple[int, int]]:
        """Every directed ring edge that may carry traffic."""
        out = [(self.gpus[i], self.gpus[(i + 1) % self.n]) for i in range(self.n)]
        if self.bidirectional:
            out += [(self.gpus[(i + 1) % self.n], self.gpus[i]) for i in range(self.n)]
        return out

    def is_edge(self, u: int, v: int) -> bool:
        return any(self.distance(u, v, d) == 1 for d in self.directions())


def _data_nodes(topology) -> List[int]:
    """The cell's data-bearing nodes: everything that is neither a switch nor masked out.

    `passive_indices` is how `solve._CellView` marks the parent's other nodes, which are present in
    the capacity matrix but are not part of this cell.
    """
    n = len(topology.capacity)
    switches = set(getattr(topology, "switch_indices", []) or [])
    passive = set(getattr(topology, "passive_indices", []) or [])
    return [i for i in range(n) if i not in switches and i not in passive]


def ring_topology_order(topology) -> Optional[RingOrder]:
    """Is this level's graph a PHYSICAL ring, and if so in what order?

    Necessary conditions, all of them: no switch at all (a switch means the crossbar row, or a
    routing choice this closed form does not make), at least three data nodes, and a data-node graph
    that is exactly one cycle -- every node with two undirected neighbours, all reachable in one
    walk. A "ring" with a chord, or two disjoint cycles, is not a shape whose optimal schedule is
    the busiest-link bound, so it is rejected rather than silently scheduled pessimistically.

    Bidirectional when every ring edge carries capacity BOTH ways; unidirectional when the cycle is
    consistently oriented. Anything in between (some edges one-way, some two-way) is rejected: the
    hop-chain lowering assumes a uniform ring, and a mixed one needs a real routing decision.
    """
    if getattr(topology, "switch_indices", None):
        return None
    nodes = _data_nodes(topology)
    if len(nodes) < 3:
        return None
    cap = topology.capacity
    out = {i: [j for j in nodes if j != i and cap[i][j] > 0] for i in nodes}
    undirected = {i: sorted({j for j in nodes
                             if j != i and (cap[i][j] > 0 or cap[j][i] > 0)}) for i in nodes}
    if any(len(v) != 2 for v in undirected.values()):
        return None

    # Walk the undirected cycle; it must close after exactly len(nodes) steps and visit everything.
    start = nodes[0]
    order = [start]
    prev, cur = None, start
    while True:
        nxt = next((x for x in undirected[cur] if x != prev), None)
        if nxt is None:
            return None
        if nxt == start:
            break
        if nxt in order:
            return None
        order.append(nxt)
        prev, cur = cur, nxt
    if len(order) != len(nodes):
        return None

    ring = tuple(order)
    fwd = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    both = all(cap[u][v] > 0 and cap[v][u] > 0 for u, v in fwd)
    one_way = all(cap[u][v] > 0 and cap[v][u] <= 0 for u, v in fwd)
    if both:
        return RingOrder(gpus=ring, bidirectional=True, via_switch=None)
    if one_way:
        return RingOrder(gpus=ring, bidirectional=False, via_switch=None)
    # Consistently oriented the other way round: re-orient rather than reject.
    rev = tuple([ring[0]] + list(reversed(ring[1:])))
    rfwd = [(rev[i], rev[(i + 1) % len(rev)]) for i in range(len(rev))]
    if all(cap[u][v] > 0 and cap[v][u] <= 0 for u, v in rfwd):
        return RingOrder(gpus=rev, bidirectional=False, via_switch=None)
    if any(len(v) != 2 for v in out.values()):
        return None
    return None


def is_ring(topology) -> bool:
    """Does this level's graph consist of data nodes wired as one physical cycle?"""
    return ring_topology_order(topology) is not None


def ring_order_for_cell(topology, cell: Cell) -> RingOrder:
    """The order this cell's ring schedule runs around, physical if there is one and logical if the
    flag forced a crossbar over.

    The logical order is `cell.gpus` AS DECLARED, not sorted: that list is already the cell's
    canonical GPU order (it is what `lift_demand` uses to map sub-chunk c to gpus[c]), so using it
    keeps the ring's neighbour relation aligned with every other per-cell convention. On the
    rail-optimized topology it is also the rail order, which means ring neighbour == adjacent rail.
    """
    physical = ring_topology_order(topology)
    if physical is not None:
        # Restrict the detected cycle to this cell's GPUs, preserving cyclic order.
        in_cell = [g for g in physical.gpus if g in set(cell.gpus)]
        if len(in_cell) != len(cell.gpus):
            raise RuntimeError(
                f"cell declares gpus {list(cell.gpus)} but the detected physical ring covers "
                f"{list(physical.gpus)}; the ring solver needs the cell's GPUs to be exactly the "
                f"ring's nodes")
        return RingOrder(gpus=tuple(in_cell), bidirectional=physical.bidirectional,
                         via_switch=None)
    if not is_crossbar(topology):
        raise RuntimeError(
            f"ring_order_for_cell: the cell's fabric is neither a physical ring nor a single-switch "
            f"crossbar ({len(topology.capacity)} nodes, "
            f"{list(getattr(topology, 'switch_indices', []))} switches), so there is no ring order "
            f"to impose on it")
    if len(cell.gpus) < 2:
        raise RuntimeError(f"a ring needs at least 2 GPUs; cell has {list(cell.gpus)}")
    switches = list(cell.internal_switches)
    assert len(switches) == 1, (
        f"a logical ring over a crossbar cell needs the cell's single internal switch to annotate "
        f"its hops, got {switches}")
    return RingOrder(gpus=tuple(cell.gpus), bidirectional=False, via_switch=switches[0])


# --------------------------------------------------------------------------------------------
# Step 1: IntraCellDemand -> hop-chain jobs
# --------------------------------------------------------------------------------------------
@dataclass
class _HopNeed:
    """The tightest constraint any demand places on one hop of one (identity, direction) chain."""
    release: int
    deadline: float
    hard: bool

    def tighten(self, release: int, deadline: float, hard: bool) -> None:
        self.release = min(self.release, release)
        self.hard = self.hard or hard
        if hard:
            self.deadline = min(self.deadline, deadline)


def _arcs(demand: IntraCellDemand, order: RingOrder) -> Dict[int, int]:
    """How far this demand's data must travel in each direction: {direction: hops}.

    A ring delivery is a store-and-forward arc, so covering a set of wanters means reaching the
    FARTHEST of them; every node the arc passes through gets a copy on the way, which is what makes
    one arc serve a whole fan-out. On a bidirectional ring the wanters are split by which way round
    is shorter (the standard two-ring split, and what halves the chain depth); on a unidirectional
    ring there is a single arc.

    Returns {} when nothing has to move (the only wanter is the source itself).
    """
    wanters = [t for t in demand.dst_gpus if t != demand.src_gpu]
    reach: Dict[int, int] = {}
    for w in wanters:
        d = order.best_direction(demand.src_gpu, w)
        hops = order.distance(demand.src_gpu, w, d)
        if hops == 0:
            continue
        reach[d] = max(reach.get(d, 0), hops)
    return reach


def _ring_jobs(demands: Sequence[IntraCellDemand], order: RingOrder,
               debug: bool = False) -> List[_Job]:
    """Lower a cell's demands into ring hop chains.

    ONE CHAIN PER (identity, direction), never one per demand. Two demands that move the same
    identity the same way round share their hops -- an `egress_stage` relay to a gateway three hops
    along and a `self_distribution` of the same identity to the whole cell are the same bytes taking
    the same first three edges, and duplicating them would put the identity on one link twice. This
    is the ring's analogue of the crossbar's `(identity, src, dst)` delivery dedup, and it is why
    the chain is built from a per-hop requirement table rather than emitted demand by demand.

    PER-HOP, NOT PER-CHAIN, REQUIREMENTS -- and that is what keeps precedence sound across bands.
    Every demand covers a PREFIX of its chain (hops 1..L), so the set of demands covering hop k is a
    superset of those covering hop k+1, and therefore hop k's constraint is at least as tight as hop
    k+1's. Bands are assigned from those constraints, so a chain's bands are monotonically
    non-decreasing along the chain: a predecessor is never in a LATER band than its successor. That
    is the fact `_drop_cross_band_precedence` relies on.
    """
    # Where each identity lives in this cell, and how big it is. Built in ONE pass rather than
    # rescanned per chain: a rail alltoall cell carries hundreds of identities and hundreds of
    # demands, and re-deriving these inside the chain loop makes lowering quadratic in the cell's
    # demand count.
    src_of: Dict[Identity, int] = {}
    volume_of: Dict[Identity, float] = {}
    for d in demands:
        if not _arcs(d, order):
            continue                      # nothing to move: contributes no origin claim
        prev = src_of.setdefault(d.identity, d.src_gpu)
        if prev != d.src_gpu:
            raise RuntimeError(
                f"identity {d.identity} is sourced from both {prev} and {d.src_gpu} inside one "
                f"cell; a ring chain needs a single origin per identity (the gateway it arrived "
                f"on, or the GPU that owns it natively)")
        # MAX, never sum: a hop forwards the identity's bytes once and every demand wanting them
        # downstream is served by that one transfer (the same reasoning as
        # `reconstruct._coalesce_egress`, exact here because every demand carries a whole sub-chunk).
        volume_of[d.identity] = max(volume_of.get(d.identity, 0.0), d.volume)

    # (identity, direction) -> {hop index (1-based) -> _HopNeed}
    needs: Dict[Tuple[Identity, int], Dict[int, _HopNeed]] = defaultdict(dict)
    for d in demands:
        if d.kind == "egress_stage":
            # A network send is waiting on this relay, so every hop of it is hard with that send's
            # deadline: the whole arc has to land before the gateway transmits.
            hard, deadline = True, float(d.deadline_epoch)
            release = PROLOGUE_BAND
        else:
            hard = bool(getattr(d, "hard", False))
            deadline = float(d.deadline_epoch) if hard else inf
            # A network piece lands at the END of its arrival epoch, so the earliest band that can
            # forward it is the next one; native data exists before the collective starts.
            release = (d.deadline_epoch + 1 if d.kind == "ingress_distribution" else PROLOGUE_BAND)
        reach = _arcs(d, order)
        if not reach and debug:
            _p(debug, f"    demand id={d.identity} {d.kind} src={d.src_gpu} -> "
                      f"{list(d.dst_gpus)}: nothing to move")
        for direction, hops in sorted(reach.items()):
            table = needs[(d.identity, direction)]
            for k in range(1, hops + 1):
                need = table.get(k)
                if need is None:
                    table[k] = _HopNeed(release=release, deadline=deadline, hard=hard)
                else:
                    need.tighten(release, deadline, hard)
            _p(debug, f"    demand id={d.identity} {d.kind} src={d.src_gpu} -> "
                      f"{list(d.dst_gpus)}: arc {'cw' if direction == CW else 'ccw'} x{hops} hop(s)"
                      f"{' HARD' if hard else ''}")

    jobs: List[_Job] = []
    for (identity, direction), table in sorted(needs.items(), key=lambda kv: (str(kv[0][0]),
                                                                             kv[0][1])):
        src, volume = src_of[identity], volume_of[identity]
        prev: Optional[_Job] = None
        for k in sorted(table):
            need = table[k]
            u = order.node_at(src, k - 1, direction)
            v = order.node_at(src, k, direction)
            job = _Job(identity=identity, src=u, dst=v, volume=volume,
                       release_gap=need.release, deadline_gap=need.deadline, hard=need.hard,
                       kind="ring_hop", predecessor=prev)
            jobs.append(job)
            prev = job
    if debug:
        nh = sum(1 for j in jobs if j.hard)
        _p(debug, f"  [_ring_jobs] {len(demands)} demands -> {len(jobs)} hop job(s) "
                  f"({nh} hard, {len(jobs) - nh} soft) over {order.n}-node ring "
                  f"{'bidirectional' if order.bidirectional else 'unidirectional'}")
    return jobs


# --------------------------------------------------------------------------------------------
# Step 2: schedule one band's hop jobs onto the ring's links
# --------------------------------------------------------------------------------------------
def _link_load(jobs: Sequence[_Job]) -> Dict[Tuple[int, int], float]:
    """Total volume each directed ring edge must carry."""
    load: Dict[Tuple[int, int], float] = defaultdict(float)
    for j in jobs:
        load[(j.src, j.dst)] += j.volume
    return dict(load)


def _max_link_load(jobs: Sequence[_Job]) -> float:
    """The ring makespan lower bound: the busiest directed edge.

    On a ring the only contention is a link, so a link-bound schedule needs exactly
    ceil(this / link_cap) rounds. The other bound is the longest chain (its depth), and the real
    lower bound is the max of the two -- which is why `_schedule_band` reports both.
    """
    return max(_link_load(jobs).values(), default=0.0)


def _max_chain_depth(jobs: Sequence[_Job]) -> int:
    """Longest precedence chain, in rounds. A chain of L hops cannot finish before round L-1, so
    with few identities in flight this, not the link load, is what sets the makespan."""
    depth: Dict[int, int] = {}

    def _d(j: _Job) -> int:
        got = depth.get(id(j))
        if got is None:
            got = 1 if j.predecessor is None else 1 + _d(j.predecessor)
            depth[id(j)] = got
        return got

    return max((_d(j) for j in jobs), default=0)


def _hop_index(j: _Job) -> int:
    """Position of a job in its chain (1 for the first hop). Used as the scheduling tiebreak so an
    earlier hop is never held behind a later one, which is what lets chains pipeline."""
    k, cur = 1, j
    while cur.predecessor is not None:
        k += 1
        cur = cur.predecessor
    return k


def _schedule_band(jobs: List[_Job], order: RingOrder, band: int, link_cap: float = 1.0,
                   max_rounds: int = 100_000, cell_id: Optional[int] = None,
                   debug: bool = False) -> List[IntraFlow]:
    """Schedule one band's hop jobs into rounds, bounded by per-directed-link capacity.

    The round loop mirrors `crossbar_solve._schedule_band` -- greedy, priority-ordered, atomic
    multi-round spans -- with one substantive change: capacity is charged to the RING EDGE, not to
    the endpoints' ports. On this module's rings those coincide (a unidirectional logical ring gives
    each GPU exactly one out-edge, and a physical ring's two out-edges are distinct links), which is
    what makes edge-only accounting exact rather than merely optimistic; `_assert_links` re-checks
    it on the output.

    Priority: hard first, then earliest deadline, then HOP INDEX, then (src, dst). The hop-index
    term is the one that matters for a ring: without it a chain's later hops can win a link against
    an earlier hop of another chain and stall the pipeline behind the deepest chain.
    """
    def _key(j: _Job):
        return (not j.hard, j.deadline_gap, _hop_index(j), j.src, j.dst)

    link_lb = math.ceil(_max_link_load(jobs) / link_cap - EPS) if jobs else 0
    depth_lb = _max_chain_depth(jobs)
    lb = max(link_lb, depth_lb)
    if debug:
        nh = sum(1 for j in jobs if j.hard)
        _p(debug, f"    [band {band}] {len(jobs)} hop job(s) ({nh} hard); lower bound "
                  f"{lb} round(s) = max(link {link_lb}, chain depth {depth_lb})")

    def _span(volume: float) -> int:
        return max(1, math.ceil(volume / link_cap - EPS))

    def _needs(volume: float, span: int) -> List[float]:
        return [min(link_cap, volume - i * link_cap) for i in range(span)]

    link_used: Dict[Tuple[int, int, int], float] = defaultdict(float)

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
            # Impossible while the chains are chains: hop 1 has no predecessor and is always ready,
            # so an empty ready set means a predecessor was dropped without being scheduled (a
            # cross-band link that was not really cross-band) or a cycle was built.
            raise RuntimeError(
                f"ring schedule stalled at round {r}, band {band}: "
                f"{[(j.src, j.dst, j.identity) for j in pending][:8]}")
        ready.sort(key=_key)
        matched: List[str] = []
        for j in ready:
            span = _span(j.volume)
            needs = _needs(j.volume, span)
            if not all(link_used[(j.src, j.dst, r + i)] + needs[i] <= link_cap + EPS
                       for i in range(span)):
                continue          # this edge is busy in one of the rounds the job would span
            for i in range(span):
                link_used[(j.src, j.dst, r + i)] += needs[i]
            flows.append(IntraFlow(cell=cell_id, identity=j.identity, sender=j.src, receiver=j.dst,
                                   via_switch=order.via_switch, volume=j.volume, band=band,
                                   local_round=r, span=span, kind=j.kind, hard=j.hard,
                                   identities=j.identities))
            j.remaining = 0.0
            j.completion_round = r + span - 1
            pending.remove(j)
            if debug:
                matched.append(f"{j.src}->{j.dst}{'H' if j.hard else ''}"
                               f"(id{j.identity[0]},h{_hop_index(j)})")
        if debug and matched:
            _p(debug, f"      round {r}: " + "  ".join(matched))
    else:
        raise RuntimeError(f"ring schedule exceeded {max_rounds} rounds in band {band}")
    if debug:
        used = rounds_in(flows)
        verdict = ("OPTIMAL (= bound)" if used == lb else
                   f"{used - lb} round(s) above bound" if used > lb else "below bound?!")
        _p(debug, f"    [band {band}] used {used} round(s) vs bound {lb} -> {verdict}")
    return flows


def _drop_cross_band_precedence(by_band: Dict[int, List[_Job]]) -> None:
    """Cut a precedence edge whose endpoints landed in DIFFERENT bands.

    Bands are laid out sequentially on the absolute fine-epoch axis (band k occupies
    `[W + m*k, W + m*k + rounds_k)` and `rounds_k <= m` is asserted, so band k finishes before band
    k+1 starts -- see teccl/hierarchy/flat_schedule.py). A predecessor in a strictly earlier band is
    therefore already complete before this band's first round, and the in-band `completion_round`
    comparison -- which is band-local and would be meaningless across bands -- has nothing left to
    enforce. Dropping the link is the accurate statement, not a relaxation.

    A predecessor in a strictly LATER band would be a real violation, and cannot happen: a demand
    covers a prefix of its chain, so constraints only loosen along it and bands only increase (see
    `_ring_jobs`). Checked rather than assumed.
    """
    band_of_job: Dict[int, int] = {}
    for band, jobs in by_band.items():
        for j in jobs:
            band_of_job[id(j)] = band
    for band, jobs in by_band.items():
        for j in jobs:
            pred = j.predecessor
            if pred is None:
                continue
            pband = band_of_job.get(id(pred))
            if pband is None:
                raise RuntimeError(
                    f"hop {j.src}->{j.dst} (identity {j.identity}) depends on a job that was not "
                    f"assigned to any band")
            if pband > band:
                raise RuntimeError(
                    f"hop {j.src}->{j.dst} (identity {j.identity}) is in band {band} but its "
                    f"predecessor {pred.src}->{pred.dst} is in the LATER band {pband}; ring chain "
                    f"constraints must loosen along the chain")
            if pband < band:
                j.predecessor = None


# --------------------------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------------------------
def _assert_links(flows: Sequence[IntraFlow], order: RingOrder, link_cap: float = 1.0) -> None:
    """Every transfer is a ring hop, and no directed ring edge is over capacity in any round.

    The first half is the structural property the whole module exists for: if every emitted flow is
    a ring edge then each GPU has exactly one inbound and one outbound peer, so N:1 fan-in is
    unrepresentable no matter what a downstream consumer does with the ordering.
    """
    used: Dict[Tuple[int, int, int, int], float] = defaultdict(float)
    for f in flows:
        if not order.is_edge(f.sender, f.receiver):
            raise AssertionError(
                f"flow {f.sender}->{f.receiver} is not a ring edge of {list(order.gpus)}"
                f"{' (bidirectional)' if order.bidirectional else ''}")
        for i in range(f.span):
            share = min(link_cap, f.volume - i * link_cap)
            used[(f.band, f.local_round + i, f.sender, f.receiver)] += share
    for k, v in used.items():
        assert v <= link_cap + 1e-6, f"ring link over-subscribed {k}: {v}"


def _assert_ports(flows: Sequence[IntraFlow], order: RingOrder, link_cap: float = 1.0) -> None:
    """No GPU sends or receives more than its ring out-/in-edges can carry in any (band, round).

    The per-PORT statement, which is the one the stitch and the fine-link capacity checks care
    about. On a unidirectional ring it follows from `_assert_links` (one out-edge per node), and the
    bound is one link. On a bidirectional ring it is a genuinely separate statement -- a node has two
    out-edges, and on a LOGICAL ring they would share one physical port -- so it is checked rather
    than trusted, with the bound the physical ring actually provides.
    """
    cap = 2 * link_cap if order.bidirectional else link_cap
    eg: Dict[Tuple[int, int, int], float] = defaultdict(float)
    ing: Dict[Tuple[int, int, int], float] = defaultdict(float)
    for f in flows:
        for i in range(f.span):
            share = min(link_cap, f.volume - i * link_cap)
            eg[(f.band, f.local_round + i, f.sender)] += share
            ing[(f.band, f.local_round + i, f.receiver)] += share
    for k, v in eg.items():
        assert v <= cap + 1e-6, f"ring egress over-subscribed {k}: {v} > {cap}"
    for k, v in ing.items():
        assert v <= cap + 1e-6, f"ring ingress over-subscribed {k}: {v} > {cap}"


def _assert_deliveries(flows: Sequence[IntraFlow], demands: Sequence[IntraCellDemand],
                       order: RingOrder) -> None:
    """Every demand's destinations end up holding its identity, and only via hops whose sender
    already held it.

    Replayed in round order per band rather than checked structurally, so it catches a bad chain
    (an arc that skipped a node, a dropped precedence edge that let a hop run before its input
    arrived) and not just a missing edge.
    """
    holders: Dict[Identity, set] = defaultdict(set)
    for d in demands:
        holders[d.identity].add(d.src_gpu)
    for f in sorted(flows, key=lambda x: (x.band, x.local_round)):
        if f.sender not in holders[f.identity]:
            raise AssertionError(
                f"ring hop {f.sender}->{f.receiver} at band {f.band} round {f.local_round} "
                f"forwards identity {f.identity} that {f.sender} does not hold yet")
        holders[f.identity].add(f.receiver)
    for d in demands:
        missing = [t for t in d.dst_gpus if t not in holders[d.identity]]
        if missing:
            raise AssertionError(
                f"{d.kind} of identity {d.identity} from {d.src_gpu} never reached {missing} "
                f"(ring order {list(order.gpus)})")


# --------------------------------------------------------------------------------------------
# Step 3: per-cell orchestration
# --------------------------------------------------------------------------------------------
def schedule_cell(cell_id: int, cell: Cell, demands: Sequence[IntraCellDemand],
                  switch_copy: bool = False, ring_hint: Optional[Callable] = None,
                  port_cap: float = 1.0, debug: Optional[bool] = None,
                  subdivision: int = 1, topology=None) -> List[IntraFlow]:
    """Schedule every intra-cell demand of one cell as a ring and return the fine IntraFlows.

    Signature-compatible with `crossbar_solve.schedule_cell` so the base-case dispatch can pick
    either without knowing which it got. Three parameters are accepted and deliberately unused:

      switch_copy   a ring has no switch to multicast in -- the ring broadcast IS the fan-out, and
                    it already costs one send per link rather than N off one port.
      ring_hint     that hook exists on the crossbar row to nudge its edge-colouring toward a ring;
                    here the ring order IS the schedule, and it comes from the topology or from
                    `cell.gpus`.
      subdivision   the crossbar row coalesces co-travelling sub-chunks to keep them in one fine
                    epoch. A ring chain already moves an identity as one transfer per hop, so there
                    is nothing to merge; sub-chunks of one refined chunk simply form their own
                    chains. Accepted so the two rows are interchangeable.

    `topology` supplies the capacity view used to detect a physical ring; pass the cell's view (the
    dispatcher's `_CellView`). When omitted the ring is logical over `cell.gpus`, which is the
    forced-crossbar case.
    """
    debug = _ENV_DEBUG if debug is None else debug
    order = (ring_order_for_cell(topology, cell) if topology is not None
             else RingOrder(gpus=tuple(cell.gpus), bidirectional=False,
                            via_switch=(cell.internal_switches[0]
                                        if cell.internal_switches else None)))
    _p(debug, f"\n=== ring schedule_cell {cell_id}: order={list(order.gpus)} "
              f"{'bidirectional' if order.bidirectional else 'unidirectional'}, "
              f"via_switch={order.via_switch}, {len(demands)} intra demands ===")

    jobs = _ring_jobs(demands, order, debug=debug)
    by_band = _assign_bands(jobs)
    _drop_cross_band_precedence(by_band)

    flows: List[IntraFlow] = []
    for band, bjobs in sorted(by_band.items()):
        flows += _schedule_band(bjobs, order, band, link_cap=port_cap, cell_id=cell_id,
                                debug=debug)

    for j in jobs:
        if j.hard:
            assert j.completion_round is not None, (
                f"hard ring hop never completed: {j.src}->{j.dst} identity {j.identity}")
    _assert_links(flows, order, link_cap=port_cap)
    _assert_ports(flows, order, link_cap=port_cap)
    _assert_deliveries(flows, demands, order)
    if debug:
        bands = sorted({f.band for f in flows})
        peak = max(band_rounds(flows).values(), default=0)
        _p(debug, f"=== cell {cell_id}: {len(flows)} ring hop(s) across bands {bands}, peak "
                  f"{peak} rounds/band; 1 inbound + 1 outbound peer per GPU by construction ===")
    return flows


# --------------------------------------------------------------------------------------------
# Dispatch: this level's ROUTING decision, when the level's own graph is a ring
# --------------------------------------------------------------------------------------------
def ring_routing(coarse, mapping,
                 id_sets: Dict[Tuple[int, int], List[Identity]]
                 ) -> List[Tuple[object, Identity]]:
    """This level's ROUTING decision, as (coarse piece, identity) pairs for step A.

    The exact peer of `crossbar_solve.crossbar_routing`, and it has the same job: a memoized row
    knows its own routing, so instead of walking paths back out of a solved formulation it restates
    the level's demand in the piece vocabulary `reconstruct.assign_identities_preserving` consumes,
    keeping each identity attached so the level spends nothing from the refinement budget.

    On a ring the routing is "go the shorter way round", which unlike the crossbar's single hop is a
    PATH: a pair `dist` apart becomes `dist` pieces, one per ring edge, each cell on the way
    receiving and re-sending. Epochs come from `_colour_ring_epochs`, the ring's analogue of the
    crossbar's `_colour_epochs` -- a level owns its own timing, and leaving every piece in epoch 0
    would assert that one link carries the whole payload at once.

    Driven off `id_sets` rather than off IntraCellDemands for the same reason the crossbar row is:
    it is the one description both the root and a child level always have.
    """
    from teccl.hierarchy.reconstruct import make_piece

    order = ring_topology_order(coarse)
    if order is None:
        raise RuntimeError(
            f"ring_routing needs this level's graph to be a ring, but it is not "
            f"({len(coarse.capacity)} nodes, switches "
            f"{list(getattr(coarse, 'switch_indices', []))})")

    # One chain of hops per (identity, U, V), so the per-link colouring can keep a chain's hops in
    # increasing epochs.
    chains: List[Tuple[Identity, List[Tuple[int, int]]]] = []
    for (u, v), identities in sorted(id_sets.items()):
        direction = order.best_direction(u, v)
        dist = order.distance(u, v, direction)
        hops = [(order.node_at(u, k - 1, direction), order.node_at(u, k, direction))
                for k in range(1, dist + 1)]
        for ident in identities:
            chains.append((ident, hops))

    epochs = _colour_ring_epochs([hops for _ident, hops in chains])
    out: List[Tuple[object, Identity]] = []
    for (ident, hops), hop_epochs in zip(chains, epochs):
        for (a, b), k in zip(hops, hop_epochs):
            out.append((make_piece(src_cell=a, dst_cell=b,
                                   # A ring hop is a DIRECT link, so each side's coarse neighbor is
                                   # the other cell itself -- there is no passthrough node between
                                   # them the way a crossbar hop has its switch.
                                   egress_neighbor=b, ingress_neighbor=a,
                                   via_switches=(), volume=1.0,
                                   send_epoch=k, arrival_epoch=k), ident))
    return out


def _colour_ring_epochs(chains: Sequence[Sequence[Tuple[int, int]]]) -> List[List[int]]:
    """Assign each hop an epoch: one transfer per directed ring edge per epoch, and a chain's hops
    strictly increasing.

    The ring's counterpart to `crossbar_solve._colour_epochs`. There the only contention is a node's
    two ports and every transfer is independent, so a first-fit over both endpoints is the whole
    story. Here contention is the EDGE, and a multi-hop delivery additionally cannot leave a cell
    before it has arrived -- so the greedy is first-fit on the edge, seeded above the previous hop's
    epoch. On the regular all-to-all demand a ring level actually sees, that reproduces the standard
    ring pipeline: hop k of the chain starting at epoch k.
    """
    used: Dict[Tuple[int, int], set] = defaultdict(set)
    out: List[List[int]] = []
    for hops in chains:
        got: List[int] = []
        k = 0
        for (a, b) in hops:
            while k in used[(a, b)]:
                k += 1
            used[(a, b)].add(k)
            got.append(k)
            k += 1          # the next hop of this chain cannot leave in the same epoch
        out.append(got)
    return out
