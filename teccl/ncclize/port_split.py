"""Post-solve port splitting: realize one modeled link as P parallel ports.

The solver models a link as a single pipe of bandwidth B. Real fabrics often build that B out
of P parallel ports of B/P (two up ports into the same spine, say). Those are not the same
constraint: `sum(rate) <= B` on a link is strictly weaker than `sum(rate) <= B/P` on each port,
and a schedule that satisfies the first can be infeasible under the second.

This pass closes that gap AFTER the solve, without touching it. Each FLOW -- the routed path
the solve already chose -- is placed on one port of each link it crosses. Volumes, epochs,
rates and routes are unchanged; only the port label is new. If the per-port packing succeeds,
the makespan is preserved exactly.

Design, including why the sweep is by hop and why the fit rule is best-fit, is in
port_split_design.md. The short version:

  * The port a flow occupies at hop k is decided by its own hop k-1 (leaving u on port p of
    (u,v) IS arriving at v on port p). That dependency runs forward in k within a flow, so
    sweeping by hop index is always well-founded. Sweeping by LINK is not: aggregating the
    same dependency to link level can produce cycles.
  * Capacity is per epoch and epochs are independent pools, but one port decision covers a
    flow's whole lifetime -- so this is VECTOR bin packing, and a flow's "size" is a vector,
    never a scalar. Order by peak relative load; score a placement by max over epochs.
  * Best-fit (fullest port that still fits) compacts and leaves whole ports for later hops.
    Emptiest-fit fragments and is actively wrong here.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Link = Tuple[int, int]
InPort = Tuple[object, int]     # (in-link or ('origin', gpu), port index on it)


# ----------------------------------------------------------------------------------------------
# The objects
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Flow:
    """A routed path, exactly as the schedule already names it.

    This is the atomic unit: every piece travelling `src -> switches -> dst` shares one port on
    each link, so a flow keeps exactly one port-qualified path key and the channel allocator
    (which counts distinct path keys per (src, dst) edge) sees no change at all.
    """
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


@dataclass
class FlowLoad:
    """A flow plus the bandwidth it occupies per epoch, in the schedule's own rate units.

    Rate rather than piece count, because rate is what the schedule carries and what the
    capacity is expressed in -- it needs no chunk_size / epoch_duration round trip and it is
    correct across a hierarchical schedule whose levels were paced against different epochs.

    A flow with no rate at all (a deliberately unpaced intra-cell hop) has an empty `load`: it
    is assigned a port for completeness but consumes no modeled capacity, exactly as it
    consumes none in `reconstruct._assert_rate_within_capacity`.
    """
    flow: Flow
    load: Dict[int, float] = field(default_factory=dict)

    def peak(self, cap: float) -> float:
        return max((v / cap for v in self.load.values()), default=0.0)


@dataclass
class SplitFlow:
    """A flow that did not fit any single port and had its PIECES divided across ports.

    The escape hatch, not the plan. Pieces are atomic and identical, so piece granularity is
    always feasible when the aggregate fits -- but it costs the flow its 1:1 path key, so each
    of these is +1 channel on that (src, dst) edge. Reported, never silent.
    """
    flow: Flow
    link: Link
    epoch_share: Dict[int, Dict[int, float]]   # port -> epoch -> rate


@dataclass
class PortAssignment:
    port: Dict[Tuple[Flow, Link], int] = field(default_factory=dict)
    splits: List[SplitFlow] = field(default_factory=list)
    # (link, hop) -> number of distinct (in-port, out-port) pairs realized
    combos: Dict[Tuple[Link, int], int] = field(default_factory=dict)

    def of(self, flow: Flow, link: Link) -> int:
        return self.port[(flow, link)]

    def qualified_path(self, flow: Flow) -> Tuple[object, ...]:
        """The flow's switch path with a port index attached to each hop.

        This is what a downstream path key becomes: `(24, 0, 28, 1, 26)` rather than
        `(24, 28, 26)`. Hops on single-port links keep index 0, so a topology that declares no
        ports produces exactly the tuple it produces today.
        """
        out: List[object] = []
        for i, hop in enumerate(flow.hops()):
            if i:
                out.append(flow.path()[i])
            out.append(self.port[(flow, hop)])
        return tuple(out)


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


def flow_loads(schedule: dict,
               occupancy: Optional[Callable[[int, float, float], Iterable[int]]] = None,
               ) -> List[FlowLoad]:
    """Group a schedule's "7-Flows" lines into flows carrying a per-epoch rate.

    `occupancy(epoch, volume, rate)` yields every GRID epoch a send occupies -- one callable,
    not a separate epoch remap plus a window, because the two do not compose: a send spanning
    many fine epochs occupies one grid epoch at its rate, it does not deposit that rate once per
    fine epoch. `occupancy_grid()` builds the right one for a schedule.

    The default is `(epoch,)`: the send occupies exactly the epoch it starts in, which is what a
    flat schedule does and what the fill-one-epoch pacing rule (`reconstruct._piece_rate`) makes
    true on any level's own grid.
    """
    occ = occupancy or (lambda e, v, r: (e,))
    acc: Dict[Flow, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    seen: set = set()
    for line in schedule['7-Flows']:
        m = _FLOW_RE.match(line)
        if m is None:
            raise ValueError(f"unparsable flow line: {line!r}")
        src, dst = int(m.group(3)), int(m.group(4))
        sw = m.group('switches')
        switches = tuple(int(x) for x in sw.split('->')) if sw else ()
        flow = Flow(src, switches, dst)
        seen.add(flow)
        if m.group('rate') is None:
            continue                                   # deliberately unpaced; see FlowLoad
        rate = float(m.group('rate'))
        volume = float(m.group('volume') or 1.0)
        epoch = int(m.group('epoch'))
        for e in occ(epoch, volume, rate):
            acc[flow][e] += rate
    loads = [FlowLoad(f, dict(acc[f])) for f in sorted(acc, key=str)]
    loads += [FlowLoad(f, {}) for f in sorted(seen - set(acc), key=str)]
    return loads


def occupancy_grid(schedule: dict, subdivision: int = 1
                   ) -> Callable[[int, float, float], Iterable[int]]:
    """Derive a schedule's occupancy callable, without being told the grid.

    A paced send occupies its link for `volume / rate` seconds, i.e. for
    `chunk_size / (M * rate * delta)` FINE epochs -- the same duration `parse_flows_lp`
    computes for its pacing gates. So the honest occupancy axis is the fine one, and on it a
    hierarchical schedule's network sends each span many epochs (1728 of them on the reference
    schedule, where one coarse epoch is 1728 fine ones).

    Packing per fine epoch would be correct and unaffordable. Instead take `g`, the gcd of
    every send's start AND duration: that is the coarsest grid on which every paced send both
    starts and ends on a boundary, so per-grid-epoch occupancy is EXACTLY per-fine-epoch
    occupancy, with no approximation. On the reference schedule g comes out 1728 and every send
    spans one grid epoch; on a flat schedule g is 1 and nothing changes.

    A schedule mixing levels with incommensurate pacing lands on g = 1 and pays the fine axis.
    That is the honest cost of the case, not a silent degradation -- and `window` still yields
    the exact epoch set, so the result is right either way.
    """
    delta = schedule['1-Epoch_Duration']
    chunk = schedule.get('9-Chunk_Size', 1.0)
    from math import gcd
    g = 0
    spans = {}
    for line in schedule['7-Flows']:
        m = _FLOW_RE.match(line)
        if m is None or m.group('rate') is None:
            continue
        rate = float(m.group('rate'))
        start = int(m.group('epoch'))
        dur = max(1, round(chunk / (subdivision * rate * delta)))
        spans[(start, rate)] = dur
        g = gcd(g, start, dur)
    g = max(1, g)

    def occupancy(epoch: int, volume: float, rate: float) -> Iterable[int]:
        dur = spans.get((epoch, rate)) or max(1, round(chunk / (subdivision * rate * delta)))
        return range(epoch // g, (epoch + dur) // g)

    return occupancy


# ----------------------------------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------------------------------
def assign_ports(loads: Sequence[FlowLoad],
                 port_count: Callable[[int, int], int],
                 port_capacity: Callable[[int, int], float],
                 tol: float = 1e-6) -> PortAssignment:
    """Assign every (flow, link) a port, sweeping by HOP INDEX.

    Hop k is decided for every flow before hop k+1 is touched for any flow, so each flow's
    in-port label is always already known when its out-port is chosen. That ordering is
    well-founded by construction -- see the module docstring.

    Within a hop, each out-link is solved independently against the residual left by earlier
    hops (`used`), which is what makes a link used at several hop indices pack incrementally
    rather than jointly.
    """
    for fl in loads:
        p = fl.flow.path()
        if len(set(p)) != len(p):
            raise AssertionError(
                f"flow {fl.flow} repeats a node in its path; hop index is then not strictly "
                f"increasing along the flow and the hop sweep is not well-founded")

    result = PortAssignment()
    # link -> port -> epoch -> occupied rate. Carried ACROSS hops, never reset.
    used: Dict[Link, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))
    max_hops = max((len(fl.flow.hops()) for fl in loads), default=0)

    for k in range(max_hops):
        at_hop: Dict[Link, List[FlowLoad]] = defaultdict(list)
        for fl in loads:
            hops = fl.flow.hops()
            if k < len(hops):
                at_hop[hops[k]].append(fl)
        for link in sorted(at_hop):
            flows = at_hop[link]
            inport = {fl.flow: _in_port(fl, k, result) for fl in flows}
            _solve_link(link, k, flows, inport, used[link],
                        port_count(*link), port_capacity(*link), result, tol)
    return result


def _in_port(fl: FlowLoad, k: int, result: PortAssignment) -> InPort:
    """The port this flow arrived on: its own decision one hop earlier.

    At hop 0 there is no inbound link -- the data comes out of the sender's memory -- so the
    label is the origin itself. That is also why a relay creates no cross-flow dependency: the
    relayed copy is a new flow whose hop 0 starts here.
    """
    if k == 0:
        return (('origin', fl.flow.src), 0)
    prev = fl.flow.hops()[k - 1]
    return (prev, result.port[(fl.flow, prev)])


def _solve_link(link: Link, hop: int, flows: Sequence[FlowLoad], inport: Dict[Flow, InPort],
                used: Dict[int, Dict[int, float]], nports: int, cap: float,
                result: PortAssignment, tol: float) -> None:
    """One local problem: heavy-first buckets, best-fit placement.

    A single-port link still runs through here so that its flows get a recorded port (0) and a
    combo count, keeping the sweep uniform and `qualified_path` total.
    """
    if nports == 1:
        for fl in flows:
            result.port[(fl.flow, link)] = 0
            for e, r in fl.load.items():
                used[0][e] += r
        result.combos[(link, hop)] = len({inport[fl.flow] for fl in flows})
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

    def place(load: Dict[int, float], q: int) -> None:
        for e, r in load.items():
            used[q][e] += r

    def best(load: Dict[int, float], prefer: int) -> Optional[int]:
        """Best-fit: the FULLEST port that still fits.

        Compaction, not balance. It preserves whole ports for later hops and reduces the number
        of active in-ports downstream; spreading load would fragment both. Ties go to `prefer`
        (the in-port's own index, the only place affinity survives) and then to the lowest
        index, so the result is deterministic across runs.
        """
        cand = [(q, f) for q in range(nports) for f in (fit(load, q),) if f is not None]
        if not cand:
            return None
        return min(cand, key=lambda qf: (-qf[1], 0 if qf[0] == prefer else 1, qf[0]))[0]

    buckets: Dict[InPort, List[FlowLoad]] = defaultdict(list)
    for fl in flows:
        buckets[inport[fl.flow]].append(fl)

    def bucket_peak(fs: Sequence[FlowLoad]) -> float:
        agg = _aggregate(fs)
        return max((v / cap for v in agg.values()), default=0.0)

    # Heavy-first by PEAK relative load, not total: a port is bound by its worst epoch, and the
    # two orders genuinely differ (see the design note's measured inversion). Sorting is free --
    # every bucket is in hand at once, so there is no arrival order to be online against.
    order = sorted(buckets, key=lambda b: (-bucket_peak(buckets[b]), str(b)))
    combos = 0
    for b in order:
        fs = buckets[b]
        prefer = b[1] % nports
        agg = _aggregate(fs)
        q = best(agg, prefer)
        if q is not None:                                   # whole bucket, one combo
            for fl in fs:
                result.port[(fl.flow, link)] = q
            place(agg, q)
            combos += 1
            continue
        # Bucket does not fit whole: break it up at FLOW granularity, same rule.
        touched = set()
        for fl in sorted(fs, key=lambda fl: (-fl.peak(cap), str(fl.flow))):
            q = best(fl.load, prefer)
            if q is not None:
                result.port[(fl.flow, link)] = q
                place(fl.load, q)
                touched.add(q)
                continue
            share = _piece_split(fl, used, nports, cap, tol)
            if share is None:
                raise AssertionError(
                    f"flow {fl.flow} on link {link} does not fit even when its pieces are "
                    f"divided across all {nports} ports -- the link is over capacity in "
                    f"aggregate, which means the SOLVE is infeasible for this topology, not "
                    f"the split")
            result.port[(fl.flow, link)] = min(share)
            result.splits.append(SplitFlow(fl.flow, link, share))
            touched |= set(share)
        combos += len(touched)
    result.combos[(link, hop)] = combos


def _aggregate(fs: Sequence[FlowLoad]) -> Dict[int, float]:
    agg: Dict[int, float] = defaultdict(float)
    for fl in fs:
        for e, r in fl.load.items():
            agg[e] += r
    return dict(agg)


def _piece_split(fl: FlowLoad, used: Dict[int, Dict[int, float]], nports: int, cap: float,
                 tol: float) -> Optional[Dict[int, Dict[int, float]]]:
    """Divide one flow's rate across ports, per epoch. Always feasible if the aggregate fits.

    This is what makes the pass total: flow granularity is bin packing and can fail, but pieces
    are atomic and interchangeable, so a flow can always be poured into whatever headroom
    exists. The cost is the flow's 1:1 path key, hence a SplitFlow report rather than silence.
    """
    share: Dict[int, Dict[int, float]] = defaultdict(dict)
    for e, r in fl.load.items():
        left = r
        for q in range(nports):
            if left <= tol:
                break
            room = cap - used[q][e]
            if room <= tol:
                continue
            take = min(room, left)
            share[q][e] = take
            used[q][e] += take
            left -= take
        if left > tol:
            return None
    return dict(share)


def qualify_path_key(path_key, ports: Sequence[int], any_split: bool):
    """The port-qualified form of a `_parse_switch_path` key: `(switches, ports)`.

    Returns the key UNCHANGED when the route touches no multi-port link, so a topology that
    declares no ports -- i.e. every topology today -- produces byte-identical output through
    every downstream consumer. Only a genuinely split route takes the new shape.

    `ports` has one entry per HOP, so it is one longer than `switches`: hop i is the link INTO
    switch i, and the final hop is the last switch to the destination GPU.
    """
    if not any_split:
        return path_key
    return (tuple(path_key or ()), tuple(ports))


def unqualify_path_key(path_key) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Split a path key back into (switches, ports), with ports None if it carries none.

    The single place that knows the qualified key's shape, so consumers that need the raw
    switch sequence (`build_switch_routes`) do not each re-implement the discrimination. A plain
    key is a tuple of ints; a qualified one is a 2-tuple whose first element is itself a tuple.
    """
    if (isinstance(path_key, tuple) and len(path_key) == 2
            and isinstance(path_key[0], tuple) and isinstance(path_key[1], tuple)):
        return path_key[0], path_key[1]
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
    split = {(s.flow, s.link) for s in assignment.splits}
    for s in assignment.splits:
        for q, per in s.epoch_share.items():
            for e, r in per.items():
                load[(s.link, q, e)] += r
    for fl in loads:
        for hop in fl.flow.hops():
            if (fl.flow, hop) in split:
                continue
            q = assignment.port[(fl.flow, hop)]
            for e, r in fl.load.items():
                load[(hop, q, e)] += r
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
    split = {s.flow: s for s in assignment.splits if s.link == link}
    for fl in loads:
        if link not in fl.flow.hops():
            continue
        if fl.flow in split:
            for q, per in split[fl.flow].epoch_share.items():
                for e, r in per.items():
                    out[q][e] += r
            continue
        q = assignment.port[(fl.flow, link)]
        for e, r in fl.load.items():
            out[q][e] += r
    return {q: dict(v) for q, v in out.items()}
