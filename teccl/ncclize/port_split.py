"""Post-solve port splitting: realize one modeled link as P parallel ports.

The solver models a link as a single pipe of bandwidth B. Real fabrics often build that B out
of P parallel ports of B/P (two up ports into the same spine, say). Those are not the same
constraint: `sum(rate) <= B` on a link is strictly weaker than `sum(rate) <= B/P` on each port,
and a schedule that satisfies the first can be infeasible under the second.

This pass closes that gap AFTER the solve, without touching it. Volumes, epochs, rates and
routes are unchanged; only the port label is new. If the per-port packing succeeds, the
makespan is preserved exactly.

THE MODEL. Two objects, each a bijection with a physical thing:

    Flow     <-> a path                       (src, switches, dst)
    SubFlow  <-> a path plus its ports        (src, switches, dst, one port per hop)

A SubFlow is what actually crosses wires, so it is what everything downstream keys on: the
channel allocator, the flow-id/route bijection, and the switch forwarding table. Its identity is
purely spatial. Nothing about WHEN it runs belongs in it -- `(channel, peer)` has to stay one
logical connection with its own FIFO and in-order delivery, and an epoch-dependent key would
split one connection's traffic across two of them.

The packing's whole job is to give each Flow exactly ONE SubFlow. When a flow will not fit a
single port it falls back to several, and its pieces are partitioned among them. That fallback
is one mechanism, not two: partitioning across epochs yields subflows alive at different times,
partitioning within an epoch yields concurrent subflows at finer rates. Same object, same key,
same downstream treatment.

Design notes in port_split_design.md. The two decisions worth knowing here:

  * The port a flow occupies at hop k is decided by its own hop k-1 (leaving u on port p of
    (u,v) IS arriving at v on port p). That dependency runs forward in k within a flow, so
    sweeping by hop index is always well-founded. Sweeping by LINK is not: aggregating the
    same dependency to link level can produce cycles.
  * Capacity is per epoch and epochs are independent pools, but one port decision covers a
    piece's whole life -- so this is VECTOR bin packing, and a "size" is a vector, never a
    scalar. Order by peak relative load; score a placement by max over epochs.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Link = Tuple[int, int]
InPort = Tuple[object, int]     # (in-link, or ('origin', gpu) at hop 0; port index on it)
Address = Tuple[int, int, int]  # (origin, chunk, epoch) -- see Piece


# ----------------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Flow:
    """A path: the route the solve chose, exactly as the schedule names it."""
    src: int
    switches: Tuple[int, ...]
    dst: int

    def path(self) -> Tuple[int, ...]:
        return (self.src,) + self.switches + (self.dst,)

    def hops(self) -> List[Link]:
        p = self.path()
        return list(zip(p, p[1:]))

    def __str__(self) -> str:
        via = "->".join(str(s) for s in self.switches)
        return f"{self.src}->{self.dst}" + (f" via {via}" if via else "")


@dataclass(frozen=True)
class Piece:
    """One "7-Flows" line: the finest thing the emission can address separately.

    `(origin, chunk, epoch)` is the identity, because that is what a downstream path-key lookup
    is keyed on -- `flow_path_keys[(step_idx, chunk_id, src, dst)]`, where chunk_id derives from
    (origin, chunk) and step_idx from epoch. Two pieces of one flow can therefore carry
    different keys, which is what lets a flow be partitioned among several subflows.

    `span` is the grid epochs the piece occupies; `rate` is the bandwidth it holds in each.
    """
    origin: int
    chunk: int
    epoch: int
    rate: float
    span: Tuple[int, ...]

    @property
    def address(self) -> Address:
        return (self.origin, self.chunk, self.epoch)


@dataclass(frozen=True)
class SubFlow:
    """A path plus one port per hop. The unit that crosses wires, and the unit keyed on.

    `key` is the path key the emission uses in place of the bare switch tuple. It is a pure
    function of the route and its ports -- no epoch, no chunk -- so every piece riding this
    subflow lands on one connection with one FIFO.
    """
    flow: Flow
    ports: Tuple[int, ...]                     # one per hop, so len(switches) + 1

    @property
    def key(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (self.flow.switches, self.ports)

    def __str__(self) -> str:
        return f"{self.flow} ports {self.ports}"


@dataclass
class FlowLoad:
    """A flow and its pieces.

    A piece with no rate (a deliberately unpaced intra-cell hop) is kept but consumes no modeled
    capacity, exactly as it consumes none in `reconstruct._assert_rate_within_capacity`.
    """
    flow: Flow
    pieces: Tuple[Piece, ...] = ()

    def load(self) -> Dict[int, float]:
        return _load(self.pieces)


def _load(pieces: Iterable[Piece]) -> Dict[int, float]:
    """Grid epoch -> total rate held by these pieces. The vector a placement is scored against."""
    out: Dict[int, float] = defaultdict(float)
    for p in pieces:
        for e in p.span:
            out[e] += p.rate
    return dict(out)


def _peak(pieces: Iterable[Piece], cap: float) -> float:
    return max((v / cap for v in _load(pieces).values()), default=0.0)


@dataclass
class PortAssignment:
    """Which subflow each piece rides, plus the per-link diagnostics.

    `subflows[flow]` is normally one entry. More than one means the packing had to partition
    that flow -- the fallback, reported rather than silent, since each extra subflow is one more
    channel on that `(src, dst)` edge.
    """
    subflows: Dict[Flow, Tuple[SubFlow, ...]] = field(default_factory=dict)
    # (link, hop) -> distinct (in-port, out-port) pairs realized there
    combos: Dict[Tuple[Link, int], int] = field(default_factory=dict)
    by_piece: Dict[Tuple[Flow, Address], SubFlow] = field(default_factory=dict)

    @property
    def split_flows(self) -> List[Flow]:
        return [f for f, subs in self.subflows.items() if len(subs) > 1]

    def only(self, flow: Flow) -> SubFlow:
        """The flow's single subflow. Raises if it was partitioned -- use subflow_of."""
        subs = self.subflows[flow]
        if len(subs) > 1:
            raise KeyError(f"flow {flow} is partitioned into {len(subs)} subflows "
                           f"({[str(s) for s in subs]}); ask subflow_of(flow, piece address)")
        return subs[0]

    def subflow_of(self, flow: Flow, origin: int, chunk: int, epoch: int) -> SubFlow:
        """The subflow a given piece rides. THE accessor the emission should use."""
        subs = self.subflows[flow]
        if len(subs) == 1:
            return subs[0]
        return self.by_piece[(flow, (origin, chunk, epoch))]


# ----------------------------------------------------------------------------------------------
# Reading a schedule
# ----------------------------------------------------------------------------------------------
_FLOW_RE = re.compile(
    r'Chunk (\d+) from (\d+) traveled over (\d+)->(\d+)'
    r'(?:\s+with volume\s+(?P<volume>[\d.]+))?'
    r'\s+in epoch (?P<epoch>\d+)'
    r'(?:\s+at rate\s+(?P<rate>[\d.eE+-]+))?'
    r'(?:\s+via switches\s+(?P<switches>[\d\s\->]+))?$'
)


@dataclass(frozen=True)
class Grid:
    """A schedule's packing grid: `step` fine epochs per grid epoch, plus the occupancy map.

    Callable, so it can be handed straight to `flow_loads`. `epoch_of` is exposed separately for
    a consumer that has only a raw epoch and no rate to feed an occupancy call.
    """
    step: int
    spans: Dict[Tuple[int, float], int]
    chunk: float
    delta: float
    subdivision: int

    def epoch_of(self, epoch: int) -> int:
        return epoch // self.step

    def duration(self, rate: float, epoch: int) -> int:
        return (self.spans.get((epoch, rate))
                or max(1, round(self.chunk / (self.subdivision * rate * self.delta))))

    def __call__(self, epoch: int, volume: float, rate: float) -> Tuple[int, ...]:
        return tuple(range(epoch // self.step,
                           (epoch + self.duration(rate, epoch)) // self.step))


def occupancy_grid(schedule: dict, subdivision: int = 1) -> Grid:
    """Derive a schedule's packing grid, without being told it.

    A paced send occupies its link for `volume / rate` seconds, i.e. for
    `chunk_size / (M * rate * delta)` FINE epochs -- the same duration `parse_flows_lp` computes
    for its pacing gates. So the honest occupancy axis is the fine one, and on it a hierarchical
    schedule's network sends each span many epochs (1728 of them on the reference schedule).

    Packing per fine epoch would be correct and unaffordable. Instead take `g`, the gcd of every
    send's start AND duration: the coarsest grid on which every paced send both starts and ends
    on a boundary, so per-grid-epoch occupancy is EXACTLY per-fine-epoch occupancy, with nothing
    approximated. On the reference schedule g is 1728 and every send spans one grid epoch; on a
    flat schedule g is 1 and nothing changes.

    A schedule mixing levels with incommensurate pacing lands on g = 1 and pays the fine axis.
    That is the honest cost of the case, not a silent degradation.
    """
    from math import gcd
    delta = schedule['1-Epoch_Duration']
    chunk = schedule.get('9-Chunk_Size', 1.0)
    g = 0
    spans: Dict[Tuple[int, float], int] = {}
    for line in schedule['7-Flows']:
        m = _FLOW_RE.match(line)
        if m is None or m.group('rate') is None:
            continue
        rate = float(m.group('rate'))
        start = int(m.group('epoch'))
        dur = max(1, round(chunk / (subdivision * rate * delta)))
        spans[(start, rate)] = dur
        g = gcd(g, start, dur)
    return Grid(max(1, g), spans, chunk, delta, subdivision)


def flow_loads(schedule: dict, grid: Optional[Grid] = None) -> List[FlowLoad]:
    """Group a schedule's "7-Flows" lines into flows, keeping each line as its own Piece.

    Piece identity is retained rather than summed away: it is what lets the fallback partition a
    flow among subflows at the granularity the emission can actually address.
    """
    occ = grid or (lambda e, v, r: (e,))
    acc: Dict[Flow, Dict[Address, Piece]] = defaultdict(dict)
    for line in schedule['7-Flows']:
        m = _FLOW_RE.match(line)
        if m is None:
            raise ValueError(f"unparsable flow line: {line!r}")
        chunk, origin = int(m.group(1)), int(m.group(2))
        src, dst = int(m.group(3)), int(m.group(4))
        sw = m.group('switches')
        flow = Flow(src, tuple(int(x) for x in sw.split('->')) if sw else (), dst)
        epoch = int(m.group('epoch'))
        rate = 0.0 if m.group('rate') is None else float(m.group('rate'))
        volume = float(m.group('volume') or 1.0)
        span = tuple(occ(epoch, volume, rate)) if rate else ()
        addr = (origin, chunk, epoch)
        prev = acc[flow].get(addr)
        if prev is None:
            acc[flow][addr] = Piece(origin, chunk, epoch, rate, span)
        else:
            # Same bytes on the same route in the same epoch, listed twice: sum rather than
            # pick, so the capacity check still sees the whole of it.
            acc[flow][addr] = Piece(origin, chunk, epoch, prev.rate + rate, span or prev.span)
    return [FlowLoad(f, tuple(sorted(acc[f].values(), key=lambda p: p.address)))
            for f in sorted(acc, key=str)]


# ----------------------------------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------------------------------
def assign_ports(loads: Sequence[FlowLoad],
                 port_count: Callable[[int, int], int],
                 port_capacity: Callable[[int, int], float],
                 tol: float = 1e-6) -> PortAssignment:
    """Assign every piece a port on every hop of its flow, sweeping by HOP INDEX.

    Hop k is decided for every flow before hop k+1 is touched for any flow, so a piece's in-port
    is always already known when its out-port is chosen. That ordering is well-founded by
    construction -- see the module docstring.

    Within a hop each out-link is solved independently against the residual left by earlier hops,
    which is what makes a link used at several hop indices pack incrementally rather than jointly.
    """
    for fl in loads:
        p = fl.flow.path()
        if len(set(p)) != len(p):
            raise AssertionError(
                f"flow {fl.flow} repeats a node in its path; hop index is then not strictly "
                f"increasing along the flow and the hop sweep is not well-founded")

    # (flow, hop) -> piece address -> port. Working state; subflows are built from it below.
    chosen: Dict[Tuple[Flow, Link], Dict[Address, int]] = defaultdict(dict)
    used: Dict[Link, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))
    combos: Dict[Tuple[Link, int], int] = {}
    max_hops = max((len(fl.flow.hops()) for fl in loads), default=0)

    for k in range(max_hops):
        at_hop: Dict[Link, List[FlowLoad]] = defaultdict(list)
        for fl in loads:
            hops = fl.flow.hops()
            if k < len(hops):
                at_hop[hops[k]].append(fl)
        for link in sorted(at_hop):
            _solve_link(link, k, at_hop[link], chosen, used[link], combos,
                        port_count(*link), port_capacity(*link), tol)

    return _build(loads, chosen, combos)


def _in_port(fl: FlowLoad, piece: Piece, k: int,
             chosen: Dict[Tuple[Flow, Link], Dict[Address, int]]) -> InPort:
    """The port this piece arrived on: its own flow's decision one hop earlier.

    At hop 0 there is no inbound link -- the data comes out of the sender's memory -- so the
    label is the origin itself. That is also why a relay creates no cross-flow dependency: the
    relayed copy is a new flow whose hop 0 starts here.
    """
    if k == 0:
        return (('origin', fl.flow.src), 0)
    prev = fl.flow.hops()[k - 1]
    return (prev, chosen[(fl.flow, prev)][piece.address])


def _solve_link(link: Link, hop: int, flows: Sequence[FlowLoad],
                chosen: Dict[Tuple[Flow, Link], Dict[Address, int]],
                used: Dict[int, Dict[int, float]], combos: Dict[Tuple[Link, int], int],
                nports: int, cap: float, tol: float) -> None:
    """One local problem, solved in three tiers of the SAME rule: heavy-first, best-fit.

        tier 1  the whole in-port bucket on one port      -- one combo, no flow partitioned
        tier 2  each flow in the bucket, whole            -- still one subflow per flow
        tier 3  individual pieces                         -- the flow gains subflows

    Descending only on failure is what keeps subflow counts at one wherever that is possible.
    The tiers are the same placement rule at different grain, which is why a partition across
    epochs and a partition within one are the same mechanism rather than two.

    A single-port link still runs through here so its pieces get a recorded port (0) and the
    link gets a combo count, keeping the sweep uniform.
    """
    items = [(fl, p) for fl in flows for p in fl.pieces]
    inport = {(fl.flow, p.address): _in_port(fl, p, hop, chosen) for fl, p in items}

    if nports == 1:
        for fl, p in items:
            chosen[(fl.flow, link)][p.address] = 0
            for e in p.span:
                used[0][e] += p.rate
        combos[(link, hop)] = len(set(inport.values()))
        return

    def fit(load: Dict[int, float], q: int) -> Optional[float]:
        """Resulting max relative load on port q, or None if it does not fit.

        Max over epochs, not sum: epochs are independent capacity pools and a port is bound by
        its worst one. Evaluated per candidate port because the binding epoch differs per port
        as packing proceeds.
        """
        worst = 0.0
        for e, r in load.items():
            v = used[q][e] + r
            if v > cap + tol:
                return None
            worst = max(worst, v / cap)
        return worst

    def best(pieces: Sequence[Piece], prefer: int) -> Optional[int]:
        """Best-fit: the FULLEST port that still fits.

        Compaction, not balance. It preserves whole ports for later hops and reduces the number
        of active in-ports downstream; spreading would fragment both. Ties go to `prefer` (the
        in-port's own index, the only place affinity survives) and then to the lowest index, so
        the result is deterministic across runs.
        """
        load = _load(pieces)
        cand = [(q, f) for q in range(nports) for f in (fit(load, q),) if f is not None]
        if not cand:
            return None
        return min(cand, key=lambda qf: (-qf[1], 0 if qf[0] == prefer else 1, qf[0]))[0]

    def commit(flow: Flow, pieces: Sequence[Piece], q: int) -> None:
        for p in pieces:
            chosen[(flow, link)][p.address] = q
            for e in p.span:
                used[q][e] += p.rate

    buckets: Dict[InPort, List[Tuple[FlowLoad, Piece]]] = defaultdict(list)
    for fl, p in items:
        buckets[inport[(fl.flow, p.address)]].append((fl, p))

    # Heavy-first by PEAK relative load, not total: a port is bound by its worst epoch, and the
    # two orders genuinely differ (see the design note's measured inversion). Sorting is free --
    # every bucket is in hand at once, so there is no arrival order to be online against.
    order = sorted(buckets, key=lambda b: (-_peak([p for _, p in buckets[b]], cap), str(b)))
    realized = 0
    for b in order:
        members = buckets[b]
        prefer = b[1] % nports
        q = best([p for _, p in members], prefer)
        if q is not None:                                             # tier 1
            for fl, p in members:
                commit(fl.flow, [p], q)
            realized += 1
            continue

        by_flow: Dict[Flow, List[Piece]] = defaultdict(list)
        for fl, p in members:
            by_flow[fl.flow].append(p)
        touched = set()
        for flow in sorted(by_flow, key=lambda f: (-_peak(by_flow[f], cap), str(f))):
            pieces = by_flow[flow]
            q = best(pieces, prefer)
            if q is not None:                                         # tier 2
                commit(flow, pieces, q)
                touched.add(q)
                continue
            for p in sorted(pieces, key=lambda p: (-p.rate, p.address)):
                q = best([p], prefer)                                 # tier 3
                if q is None:
                    _refuse(flow, link, p, used, nports, cap, tol)
                commit(flow, [p], q)
                touched.add(q)
        realized += len(touched)
    combos[(link, hop)] = realized


def _refuse(flow: Flow, link: Link, piece: Piece, used: Dict[int, Dict[int, float]],
            nports: int, cap: float, tol: float) -> None:
    """Nothing left to subdivide: a single piece fits no port. Say which failure this is.

    Two very different situations wear the same shape here, and conflating them sends the reader
    looking in the wrong place.
    """
    stuck = next(e for e in piece.span
                 if all(used[q][e] + piece.rate > cap + tol for q in range(nports)))
    if piece.rate > cap + tol:
        why = (f"the piece alone carries {piece.rate:g} > one port's {cap:g}. A piece is the "
               f"smallest thing the emission can address separately, so there is nothing left "
               f"to divide -- this topology cannot carry this schedule")
    else:
        room = max(cap - used[q][stuck] for q in range(nports))
        why = (f"it needs {piece.rate:g} but the emptiest port has only {room:g} left, spent by "
               f"earlier placements. The greedy packing failed, not necessarily the instance -- "
               f"the aggregate may still fit")
    raise AssertionError(
        f"chunk {piece.chunk} from {piece.origin} on {flow}, link {link[0]}->{link[1]}, "
        f"cannot be placed in epoch {stuck}: {why}.")


def _build(loads: Sequence[FlowLoad],
           chosen: Dict[Tuple[Flow, Link], Dict[Address, int]],
           combos: Dict[Tuple[Link, int], int]) -> PortAssignment:
    """Turn the per-hop port choices into subflows.

    A subflow is one distinct port tuple over the flow's hops. Pieces agreeing on every hop ride
    the same one; the usual outcome is a single tuple and therefore a single subflow.
    """
    result = PortAssignment(combos=dict(combos))
    for fl in loads:
        hops = fl.flow.hops()
        groups: Dict[Tuple[int, ...], List[Piece]] = defaultdict(list)
        for p in fl.pieces:
            groups[tuple(chosen[(fl.flow, h)][p.address] for h in hops)].append(p)
        if not groups:                       # a flow with no pieces at all cannot arise
            continue
        subs = tuple(SubFlow(fl.flow, ports) for ports in sorted(groups))
        result.subflows[fl.flow] = subs
        if len(subs) > 1:
            for sub in subs:
                for p in groups[sub.ports]:
                    result.by_piece[(fl.flow, p.address)] = sub
    return result


# ----------------------------------------------------------------------------------------------
# The emitted key
# ----------------------------------------------------------------------------------------------
def unqualify_path_key(path_key) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Split an emitted path key into (switches, ports), with ports None if it carries none.

    The one place that knows the qualified key's shape, so consumers needing the raw switch
    sequence (`build_switch_routes`) do not each re-implement the discrimination. An unqualified
    key is a tuple of ints; a `SubFlow.key` is a 2-tuple of tuples.
    """
    if (isinstance(path_key, tuple) and len(path_key) == 2
            and isinstance(path_key[0], tuple) and isinstance(path_key[1], tuple)):
        return path_key
    return path_key, None


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------
# Relative slack when checking a summed rate against a port's bandwidth. Looser than
# `reconstruct.RATE_REL_TOL` (1e-6) on purpose: that one checks in-memory floats, this one checks
# rates that have been through `flat_schedule._segment`'s "{rate:g}" serialization and are
# therefore only 6 significant digits. A link the solver saturated exactly reads as 24.999984
# rather than 25.0 on the reference schedule, and could as easily have rounded the other way.
RATE_REL_TOL = 1e-5


def assert_port_capacity(loads: Sequence[FlowLoad], assignment: PortAssignment,
                         port_count: Callable[[int, int], int],
                         port_capacity: Callable[[int, int], float],
                         rel_tol: float = RATE_REL_TOL) -> None:
    """Per PORT per epoch, the assigned rates must fit one port's bandwidth.

    The port-granular sibling of `flat_schedule.assert_link_capacity`, and the whole point of
    the pass: passing this is what certifies the split preserved the makespan, because every
    send still starts and finishes exactly where the solve put it.
    """
    load: Dict[Tuple[Link, int, int], float] = defaultdict(float)
    for fl in loads:
        hops = fl.flow.hops()
        for p in fl.pieces:
            ports = assignment.subflow_of(fl.flow, *p.address).ports
            for hop, q in zip(hops, ports):
                for e in p.span:
                    load[(hop, q, e)] += p.rate
    over = []
    for (link, q, e), r in sorted(load.items()):
        cap = port_capacity(*link)
        if r > cap * (1.0 + rel_tol):
            over.append((f"{link[0]}->{link[1]}", q, e, round(r, 6), round(cap, 6)))
    if over:
        raise AssertionError(
            f"{len(over)} (port, epoch) pairs exceed one port's bandwidth "
            f"[(link, port, epoch, rate, cap)]: {over[:6]}")


def port_loads(loads: Sequence[FlowLoad], assignment: PortAssignment,
               link: Link) -> Dict[int, Dict[int, float]]:
    """port -> epoch -> assigned rate on one link. For reporting and tests."""
    out: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for fl in loads:
        hops = fl.flow.hops()
        if link not in hops:
            continue
        i = hops.index(link)
        for p in fl.pieces:
            q = assignment.subflow_of(fl.flow, *p.address).ports[i]
            for e in p.span:
                out[q][e] += p.rate
    return {q: dict(v) for q, v in out.items()}
