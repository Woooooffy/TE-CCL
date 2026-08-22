"""Unit tests for post-solve port splitting (port_split.py).

The reference schedule cannot discriminate the two rule choices the design rests on: its 50
GB/s links run at 100% in every loaded epoch, so every fit rule and both bucket orderings give
identical output there. The synthetic cases below are the only place those choices are
actually tested, which is why they exist rather than leaning on the regression alone.

Run from the repo root:
    python teccl/ncclize/port_split_test.py
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from teccl.input_data import TopologyParams                                  # noqa: E402
from teccl.ncclize.port_split import (Flow, FlowLoad, assign_ports,          # noqa: E402
                                      assert_port_capacity, flow_loads, port_loads)
from teccl.topologies.two_pod_rail import (TwoPodRailHostBound,              # noqa: E402
                                           TwoPodRailHostBoundSplitPorts)

SCHEDULE = 'Schedules/two_pod_rail_hostbound_allgather_fast_epoch_flat.json'
FINE_PER_COARSE = 1728

_passed = []


def ok(name):
    _passed.append(name)
    print(f"  [OK] {name}")


def fixed(counts, caps):
    """port_count / port_capacity callables from plain dicts, defaulting to one port."""
    return (lambda i, j: counts.get((i, j), 1),
            lambda i, j: caps.get((i, j), 1e9) / counts.get((i, j), 1))


def load(flow, **per_epoch):
    return FlowLoad(flow, {int(e[1:]): v for e, v in per_epoch.items()})


# ----------------------------------------------------------------------------------------------
def test_single_port_is_identity():
    """A topology declaring no ports must behave exactly as it does today."""
    flows = [load(Flow(0, (2,), 1), e0=5.0), load(Flow(3, (2,), 1), e0=5.0)]
    pc, cap = fixed({}, {})
    a = assign_ports(flows, pc, cap)
    assert all(q == 0 for q in a.port.values()), a.port
    # port, switch, port -- one port index per HOP, interleaved with the switch ids
    assert a.qualified_path(flows[0].flow) == (0, 2, 0), a.qualified_path(flows[0].flow)
    ok("single-port link is an identity relabeling")


def test_fan_in_hits_combo_floor():
    """P_in > P_out: no identity mapping exists, so max(P_in, P_out) is the floor.

    Four single-port in-links feeding a two-port out-link -- the leaf's permanent shape once
    spine0 is split (4 GPU downlinks against 2 up-ports).
    """
    flows = [load(Flow(i, (10,), 11), e0=25.0) for i in range(4)]
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 100.0})
    a = assign_ports(flows, pc, cap)
    assert_port_capacity(flows, a, pc, cap)
    pl = port_loads(flows, a, (10, 11))
    assert sorted(v[0] for v in pl.values()) == [50.0, 50.0], pl
    assert a.combos[((10, 11), 1)] == 4, a.combos          # floor max(4, 2)
    assert not a.splits
    ok("fan-in 4->2 packs to the combo floor with no splits")


def test_heavy_first_is_load_bearing():
    """{4,4,6,6} into two ports of 10.

    Best-fit alone is not enough: arrival order 4,4,6,6 fills one port to 8 and then neither 6
    fits anywhere, forcing a split. Sorting heavy-first (6,6,4,4) packs both ports exactly.
    Deleting the sort in _solve_link makes this test fail, which is its whole purpose.
    """
    sizes = [4.0, 4.0, 6.0, 6.0]
    flows = [load(Flow(i, (10,), 11), e0=s) for i, s in enumerate(sizes)]
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 20.0})
    a = assign_ports(flows, pc, cap)
    assert_port_capacity(flows, a, pc, cap)
    pl = port_loads(flows, a, (10, 11))
    assert sorted(v[0] for v in pl.values()) == [10.0, 10.0], pl
    assert not a.splits, a.splits
    ok("heavy-first packs {4,4,6,6} into 2x10 with no splits")


def test_best_fit_compacts_across_hops():
    """The fit rule, tested through the consequence it exists for.

    Nine ports of 100. Seven are filled at hop 0, then two half-port buckets arrive, then a
    full-port flow arrives at hop 2 on the SAME link. Best-fit compacts the two halves onto one
    port and leaves the last port whole, so the hop-2 flow fits intact. Emptiest-fit would
    scatter them onto the two free ports and the hop-2 flow would have to be split.
    """
    pc, cap = fixed({(10, 11): 9}, {(10, 11): 900.0})
    early = [FlowLoad(Flow(100 + i, (10,), 11), {0: 100.0}) for i in range(7)]
    halves = [FlowLoad(Flow(200 + i, (10,), 11), {0: 50.0}) for i in range(2)]
    late = [FlowLoad(Flow(300, (301, 10), 11), {0: 100.0})]   # link (10,11) at hop index 2
    flows = early + halves + late
    a = assign_ports(flows, pc, cap)
    assert_port_capacity(flows, a, pc, cap)
    assert not a.splits, f"best-fit should leave a whole port free; got splits {a.splits}"
    q0 = a.of(halves[0].flow, (10, 11))
    q1 = a.of(halves[1].flow, (10, 11))
    assert q0 == q1, f"the two half buckets should compact onto one port, got {q0} and {q1}"
    qlate = a.of(late[0].flow, (10, 11))
    assert qlate not in (q0, q1), f"the hop-2 flow should take a whole port, got {qlate}"
    ok("best-fit compacts, leaving a whole port for a later hop")


def test_vector_not_scalar():
    """Two flows with IDENTICAL totals pack differently depending on when they are active.

    Both halves use two flows of total 10 against ports of 10, so a scalar size model sees the
    same instance twice and must get one of them wrong. Disjoint in time -> they share a port.
    Overlapping -> they cannot.
    """
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 20.0})
    disjoint = [FlowLoad(Flow(1, (10,), 11), {0: 10.0}), FlowLoad(Flow(2, (10,), 11), {1: 10.0})]
    a = assign_ports(disjoint, pc, cap)
    assert_port_capacity(disjoint, a, pc, cap)
    assert not a.splits
    assert len({a.of(f.flow, (10, 11)) for f in disjoint}) == 1, \
        "time-disjoint flows should compact onto one port"
    overlap = [FlowLoad(Flow(1, (10,), 11), {0: 10.0}), FlowLoad(Flow(2, (10,), 11), {0: 10.0})]
    a2 = assign_ports(overlap, pc, cap)
    assert_port_capacity(overlap, a2, pc, cap)
    assert not a2.splits
    assert len({a2.of(f.flow, (10, 11)) for f in overlap}) == 2, \
        "time-overlapping flows must land on different ports"
    ok("placement is by epoch vector, not scalar total")


def test_piece_split_escape_hatch():
    """A flow too large for any single port is divided at PIECE granularity and reported."""
    flows = [FlowLoad(Flow(1, (10,), 11), {0: 60.0}), FlowLoad(Flow(2, (10,), 11), {0: 40.0})]
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 100.0})
    a = assign_ports(flows, pc, cap)
    assert_port_capacity(flows, a, pc, cap)
    assert len(a.splits) == 1 and a.splits[0].flow == flows[0].flow, a.splits
    assert abs(sum(sum(v.values()) for v in a.splits[0].epoch_share.values()) - 60.0) < 1e-9
    ok("oversized flow escapes to a piece split and is reported")


def test_hairpin_is_rejected():
    """A path revisiting a node breaks the hop ordering argument; refuse rather than guess."""
    pc, cap = fixed({}, {})
    try:
        assign_ports([FlowLoad(Flow(1, (2, 3, 2), 4), {0: 1.0})], pc, cap)
    except AssertionError as e:
        assert 'repeats a node' in str(e), e
        ok("a hairpin path is rejected, not silently mis-ordered")
        return
    raise AssertionError("expected a hairpin path to be rejected")


# ----------------------------------------------------------------------------------------------
def test_topology_ports_do_not_touch_the_solve():
    base = TwoPodRailHostBound(TopologyParams())
    split = TwoPodRailHostBoundSplitPorts(TopologyParams())
    assert base.capacity == split.capacity and base.alpha == split.alpha, \
        "declaring ports must not change what the solver sees"
    assert base.ports == [] and base.port_count(24, 28) == 1
    assert split.port_count(24, 28) == 2 and split.port_capacity(24, 28) == 25.0
    assert split.port_count(24, 29) == 1 and split.port_capacity(24, 29) == 25.0
    assert split.port_count(0, 24) == 1 and split.port_capacity(0, 24) == 25.0
    ok("port declaration leaves capacity/alpha byte-identical")


def _reference():
    topo = TwoPodRailHostBoundSplitPorts(TopologyParams())
    sched = json.load(open(SCHEDULE))
    loads = flow_loads(sched, epoch_of=lambda e: e // FINE_PER_COARSE)
    return topo, loads


def test_reference_schedule():
    topo, loads = _reference()
    a = assign_ports(loads, topo.port_count, topo.port_capacity)
    assert_port_capacity(loads, a, topo.port_count, topo.port_capacity)
    assert not a.splits, f"expected zero flow splits, got {len(a.splits)}"
    # The schedule's rate token is 6 significant digits (0.520833, not 0.5208333...), so a port
    # the solver saturated exactly reads 24.999984. Compare with tolerance, not equality.
    expect = {0: 25.0, 1: 25.0, 2: 25.0, 3: 25.0, 4: 0.0, 5: 25.0, 6: 12.5}
    for leaf in range(24, 28):
        for link in ((leaf, 28), (28, leaf)):
            pl = port_loads(loads, a, link)
            for q in (0, 1):
                got = {e: pl.get(q, {}).get(e, 0.0) for e in expect}
                assert all(abs(got[e] - v) < 1e-3 for e, v in expect.items()), \
                    f"{link} port {q}: {got}"
            hop = 1 if link[1] == 28 else 2
            assert a.combos[(link, hop)] == 4, (link, a.combos[(link, hop)])
    ok("reference schedule: exact 25/25 GB/s per port, 0 splits, 4 combos per link")


def test_path_keys_unchanged():
    """Channels are per (src,dst) edge; at flow granularity the split must cost none."""
    topo, loads = _reference()
    a = assign_ports(loads, topo.port_count, topo.port_capacity)
    before, after = collections.defaultdict(set), collections.defaultdict(set)
    for fl in loads:
        before[(fl.flow.src, fl.flow.dst)].add(fl.flow.switches)
        after[(fl.flow.src, fl.flow.dst)].add(a.qualified_path(fl.flow))
    hb = collections.Counter(len(v) for v in before.values())
    ha = collections.Counter(len(v) for v in after.values())
    assert hb == ha == collections.Counter({1: 104, 2: 64}), (hb, ha)
    ok("per-(src,dst) path-key count unchanged: {1: 104, 2: 64}")


def test_negative_control_random_hash():
    """An ECMP-style per-(flow, link) hash must lose, and by a lot.

    Without this the regression above proves only that SOME assignment works, not that finding
    it needed anything more than a coin flip.
    """
    topo, loads = _reference()
    links = [(l, s) for l in range(24, 28) for s in (28,)] + [(28, l) for l in range(24, 28)]
    rng = random.Random(1)
    worst = []
    for _ in range(2000):
        used = collections.defaultdict(lambda: collections.defaultdict(float))
        for fl in loads:
            for hop in fl.flow.hops():
                if topo.port_count(*hop) == 1:
                    continue
                q = rng.getrandbits(1)
                for e, r in fl.load.items():
                    used[(hop, q)][e] += r
        worst.append(max((max(v.values()) for v in used.values()), default=0.0))
    worst.sort()
    median = worst[len(worst) // 2]
    assert not any(w <= 25.0 + 1e-9 for w in worst), "a random hash should never be feasible here"
    assert median / 25.0 > 1.4, median
    ok(f"random hash: 0/{len(worst)} feasible, median {median:.1f}/25.0 "
       f"= {median / 25.0:.2f}x inflation")


def main():
    print("post-solve port split tests")
    print(" synthetic -- the rule choices the reference schedule cannot discriminate")
    test_single_port_is_identity()
    test_fan_in_hits_combo_floor()
    test_heavy_first_is_load_bearing()
    test_best_fit_compacts_across_hops()
    test_vector_not_scalar()
    test_piece_split_escape_hatch()
    test_hairpin_is_rejected()
    print(" topology")
    test_topology_ports_do_not_touch_the_solve()
    print(" regression -- two_pod_rail_hostbound_allgather_fast_epoch_flat")
    test_reference_schedule()
    test_path_keys_unchanged()
    test_negative_control_random_hash()
    print(f"\n{len(_passed)} passed")


if __name__ == '__main__':
    main()
