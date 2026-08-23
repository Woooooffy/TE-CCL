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
                                      assert_port_capacity, flow_loads, occupancy_grid,
                                      port_loads, qualify_path_key, unqualify_path_key)
from teccl.topologies.two_pod_rail import (TwoPodRailHostBound,              # noqa: E402
                                           TwoPodRailHostBoundSplitPorts,
                                           TwoPodRailSplitPorts)

SCHEDULE = 'Schedules/two_pod_rail_hostbound_allgather_fast_epoch_flat.json'
SPINE_BOUND = 'Schedules/two_pod_rail_allgather_flat.json'
# Every paced two_pod_rail schedule, with the split topology it was solved on.
ALL_CASES = [
    ('two_pod_rail_allgather_flat', TwoPodRailSplitPorts),
    ('two_pod_rail_alltoall_flat', TwoPodRailSplitPorts),
    ('two_pod_rail_hostbound_allgather_flat', TwoPodRailHostBoundSplitPorts),
    ('two_pod_rail_hostbound_alltoall_flat', TwoPodRailHostBoundSplitPorts),
    ('two_pod_rail_hostbound_allgather_fast_epoch_flat', TwoPodRailHostBoundSplitPorts),
    ('two_pod_rail_hostbound_alltoall_fast_epoch_flat', TwoPodRailHostBoundSplitPorts),
]

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
    assert a.qualified_path(flows[0].flow, 0) == (0, 2, 0), a.qualified_path(flows[0].flow, 0)
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


def test_sub_flows_become_distinct_routes():
    """A flow that fits neither port in ALL its epochs is sliced, and the slices are ROUTES.

    Two ports of 100. `a` loads port 0 in epoch 0 only; `b` is pushed to port 1 and loads it
    mostly in epoch 1. That leaves the two ports with complementary residuals -- port 0 tight in
    epoch 0, port 1 tight in epoch 1 -- so `x`, which spans both, fits neither port whole while
    fitting SOME port in each epoch. That is exactly the vector-packing failure epoch slicing
    exists for, and the two slices must produce different qualified path keys, or the emitted
    program would name one wire for bytes that cross two.
    """
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 200.0})       # 100 per port
    a_ = FlowLoad(Flow(1, (10,), 11), {0: 90.0})
    b_ = FlowLoad(Flow(2, (10,), 11), {0: 20.0, 1: 90.0})
    x_ = FlowLoad(Flow(3, (10,), 11), {0: 30.0, 1: 30.0})
    flows = [a_, b_, x_]
    a = assign_ports(flows, pc, cap)
    assert_port_capacity(flows, a, pc, cap)

    assert [str(s.flow) for s in a.splits] == [str(x_.flow)], [str(s.flow) for s in a.splits]
    sf = a.splits[0]
    assert set(sf.epoch_port) == {0, 1}, sf.epoch_port
    assert len(set(sf.epoch_port.values())) == 2, f"the slices should differ: {sf.epoch_port}"
    assert a.port_at(x_.flow, (10, 11), 0) != a.port_at(x_.flow, (10, 11), 1)

    # of() must refuse a split flow rather than hand back a representative port
    try:
        a.of(x_.flow, (10, 11))
    except KeyError:
        pass
    else:
        raise AssertionError("of() must refuse a split flow rather than pick a representative")

    # ... and the slices are distinct routes: different key => different flow id and channel
    k0, k1 = a.qualified_path(x_.flow, 0), a.qualified_path(x_.flow, 1)
    assert k0 != k1, (k0, k1)
    # an UNSPLIT flow is still epoch-invariant, so nothing changes when nothing splits
    assert a.qualified_path(b_.flow, 0) == a.qualified_path(b_.flow, 1)
    ok("a sub-flow is sliced by epoch and its slices are distinct routes")


def test_within_epoch_split_is_refused():
    """One epoch's load exceeding every port needs per-chunk slicing -- refuse, do not fudge."""
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 100.0})
    flows = [FlowLoad(Flow(1, (10,), 11), {0: 60.0}), FlowLoad(Flow(2, (10,), 11), {0: 40.0})]
    try:
        assign_ports(flows, pc, cap)
    except AssertionError as e:
        assert 'WITHIN an epoch' in str(e), e
        ok("a within-epoch split is refused with a clear error")
        return
    raise AssertionError("expected a within-epoch split to be refused")


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
    loads = flow_loads(sched, occupancy_grid(sched))
    return topo, loads


def test_occupancy_grid_is_derived():
    """The packing grid comes from the schedule, not from a hand-supplied divisor.

    Every paced send here starts on a multiple of 1728 fine epochs and lasts exactly 1728, so
    the gcd lands on 1728 and each send occupies exactly one grid epoch -- per-grid-epoch
    occupancy is then per-fine-epoch occupancy exactly, with nothing approximated away.
    """
    sched = json.load(open(SCHEDULE))
    occ = occupancy_grid(sched)
    assert list(occ(0, 1.0, 0.520833)) == [0], list(occ(0, 1.0, 0.520833))
    assert list(occ(1728, 1.0, 0.520833)) == [1], list(occ(1728, 1.0, 0.520833))
    assert list(occ(10368, 1.0, 0.520833)) == [6]
    ok("occupancy grid derived from the schedule (1728 fine epochs per grid epoch)")


def test_path_key_qualification_round_trip():
    """An unsplit route keeps its exact key; a split one takes the (switches, ports) form."""
    assert qualify_path_key((24, 28, 26), [0, 0, 0, 0], False) == (24, 28, 26)
    assert unqualify_path_key((24, 28, 26)) == ((24, 28, 26), None)
    q = qualify_path_key((24, 28, 26), [0, 1, 0, 0], True)
    assert q == ((24, 28, 26), (0, 1, 0, 0)), q
    assert unqualify_path_key(q) == ((24, 28, 26), (0, 1, 0, 0))
    assert unqualify_path_key(None) == (None, None)
    ok("path key qualification is opt-in and round-trips")


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
        for e in fl.load or {0: 0.0}:
            after[(fl.flow.src, fl.flow.dst)].add(a.qualified_path(fl.flow, e))
    hb = collections.Counter(len(v) for v in before.values())
    ha = collections.Counter(len(v) for v in after.values())
    assert hb == ha == collections.Counter({1: 104, 2: 64}), (hb, ha)
    ok("per-(src,dst) path-key count unchanged: {1: 104, 2: 64}")


def test_spine_bound_exercises_the_bucket_split():
    """The SLACK instance -- the one the host-bound schedule cannot provide.

    In the host-bound config every leaf uplink is pinned at line rate, so no in-port bucket can
    exceed a port and the packing has nothing to decide. Spine-bound (GPU_LEAF_BW = 50) leaves
    the GPU->leaf links with headroom, one ingress bucket outgrows a 25 GB/s port, and the
    bucket is broken up at FLOW granularity: 5 combos rather than the floor of 4, which is
    exactly the transportation bound r + c - 1 = 4 + 2 - 1. Still zero flow splits.
    """
    topo = TwoPodRailSplitPorts(TopologyParams())
    sched = json.load(open(SPINE_BOUND))
    loads = flow_loads(sched, occupancy_grid(sched))
    a = assign_ports(loads, topo.port_count, topo.port_capacity)
    assert_port_capacity(loads, a, topo.port_count, topo.port_capacity)
    assert not a.splits, a.splits
    for leaf in range(24, 28):
        assert a.combos[((leaf, 28), 1)] == 5, (leaf, a.combos[((leaf, 28), 1)])   # bucket split
        assert a.combos[((28, leaf), 2)] == 2, (leaf, a.combos[((28, leaf), 2)])   # at the floor
        for link in ((leaf, 28), (28, leaf)):
            pl = port_loads(loads, a, link)
            for q in (0, 1):
                assert all(abs(v - 25.0) < 1e-3 for v in pl[q].values()), (link, q, pl[q])
    ok("spine-bound config: bucket split to 5 combos (bound r+c-1), still 0 flow splits")


def test_every_paced_two_pod_rail_schedule():
    """Breadth: both collectives, both configs, both epoch granularities."""
    for name, cls in ALL_CASES:
        topo = cls(TopologyParams())
        sched = json.load(open(f'Schedules/{name}.json'))
        loads = flow_loads(sched, occupancy_grid(sched))
        a = assign_ports(loads, topo.port_count, topo.port_capacity)
        assert_port_capacity(loads, a, topo.port_count, topo.port_capacity)
        assert not a.splits, f"{name}: {len(a.splits)} flow splits"
    ok(f"all {len(ALL_CASES)} paced two_pod_rail schedules split with 0 flow splits")


def test_ncclize_emission():
    """Step 4: the port reaches the emitted program, and costs no channels.

    Skipped without the ncclize deps (lxml); everything above runs with none.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from teccl_ncclize import build_algorithm, build_switch_routes
    except ImportError as e:
        print(f"  [SKIP] ncclize emission ({e})")
        return
    sched = json.load(open(SCHEDULE))

    def emit(topo):
        algo, fpk, srm, _view, _rate, gates = build_algorithm(sched, topology=topo)
        ids, manifest = {}, []
        for si, step in enumerate(algo.steps):            # taccl's route bijection, mirrored
            for addr, src, dst in step.sends:
                key = fpk.get((si, addr, src, dst))
                fid = ids.setdefault((src, dst, key), len(ids))
                manifest.append({'flow_id': fid, 'step': si, 'src': src, 'dst': dst,
                                 'path_key': key})
        routes = build_switch_routes(manifest, srm, topo.programmable_switch_indices)
        per_edge = collections.defaultdict(set)
        for (_si, _a, src, dst), key in fpk.items():
            per_edge[(src, dst)].add(key)
        return fpk, routes, manifest, len(ids), len(gates), \
            collections.Counter(len(v) for v in per_edge.values())

    f0, r0, _m0, n0, g0, c0 = emit(TwoPodRailHostBound(TopologyParams()))
    f1, r1, m1, n1, g1, c1 = emit(TwoPodRailHostBoundSplitPorts(TopologyParams()))

    assert c0 == c1 == collections.Counter({1: 104, 2: 64}), (c0, c1)
    assert n0 == n1, f"route count changed: {n0} -> {n1}"
    assert g0 == g1, f"pacing gate count changed: {g0} -> {g1} (path keys and gate keys disagree)"
    assert f0 != f1, "the split topology should have produced port-qualified keys"

    # The emitted forwarding table gains next_hop_port and changes in no other way.
    def without_port(routes):
        return {s: {f: {k: v for k, v in e.items() if k != 'next_hop_port'}
                    for f, e in sw.items()} for s, sw in routes['switches'].items()}
    assert without_port(r0) == without_port(r1), "the split perturbed the forwarding table"
    assert r0['switch_id_map'] == r1['switch_id_map']

    # Every programmed port matches the hop it leaves on in the route's own port tuple.
    inv = {int(d): raw for d, raw in r1['switch_id_map'].items()}
    checked = 0
    for rec in m1:
        switches, ports = unqualify_path_key(rec['path_key'])
        if ports is None:
            continue
        for i, raw in enumerate(switches):
            dense = [d for d, r in inv.items() if r == raw]
            if not dense:
                continue
            entry = r1['switches'][str(dense[0])][str(rec['flow_id'])]
            assert entry['next_hop_port'] == ports[i + 1], (rec, entry)
            checked += 1
    assert checked, "no port-qualified entry reached the forwarding table"
    ok(f"emission: channels/routes/gates unchanged, {checked} next_hop_port entries verified")


def test_sub_flow_reaches_the_emitted_key():
    """End to end through the qualifier: a slice becomes a distinct path key.

    No shipped schedule splits, so this drives `_build_port_qualifier` on a hand-built one --
    the same complementary-residual shape as the unit test above, expressed as "7-Flows" lines.
    It is the only coverage of the wiring from a SubFlow to the key ncclize keys channels,
    flow ids and forwarding entries on.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from teccl_ncclize import _build_port_qualifier
    except ImportError as e:
        print(f"  [SKIP] sub-flow emission ({e})")
        return

    def line(src, dst, epoch, rate):
        return (f"Chunk 0 from {src} traveled over {src}->{dst} with volume 1 "
                f"in epoch {epoch} at rate {rate} via switches 10")
    schedule = {
        '1-Epoch_Duration': 1.0, '9-Chunk_Size': 1.0,
        '7-Flows': [line(1, 11, 0, 90),                       # fills port 0 in epoch 0
                    line(2, 11, 0, 20), line(2, 11, 1, 90),   # pushed to port 1, fills epoch 1
                    line(3, 11, 0, 30), line(3, 11, 1, 30)],  # fits neither port whole
    }

    class Stub:                       # only these three attributes are read
        ports = True
        programmable_switch_indices = [10]
        def port_count(self, i, j):
            return 2 if 10 in (i, j) else 1
        def port_capacity(self, i, j):
            return 100.0 if 10 in (i, j) else 1e9

    qualify, assignment = _build_port_qualifier(schedule, Stub(), lp_format=False)
    assert len(assignment.splits) == 1, [str(s.flow) for s in assignment.splits]
    k0 = qualify(3, 11, (10,), 0)
    k1 = qualify(3, 11, (10,), 1)
    assert k0 != k1, f"a sliced flow must emit distinct keys, got {k0} twice"
    assert k0[0] == k1[0] == (10,), (k0, k1)          # same switches, different ports
    # the unsplit flows stay epoch-invariant, so they stay ONE route each
    assert qualify(2, 11, (10,), 0) == qualify(2, 11, (10,), 1)
    ok(f"a sub-flow emits distinct path keys per slice: {k0} vs {k1}")


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
    test_sub_flows_become_distinct_routes()
    test_within_epoch_split_is_refused()
    test_hairpin_is_rejected()
    print(" topology")
    test_topology_ports_do_not_touch_the_solve()
    test_path_key_qualification_round_trip()
    print(" regression -- two_pod_rail_hostbound_allgather_fast_epoch_flat")
    test_occupancy_grid_is_derived()
    test_reference_schedule()
    test_path_keys_unchanged()
    test_spine_bound_exercises_the_bucket_split()
    test_every_paced_two_pod_rail_schedule()
    test_negative_control_random_hash()
    print(" emission")
    test_ncclize_emission()
    test_sub_flow_reaches_the_emitted_key()
    print(f"\n{len(_passed)} passed")


if __name__ == '__main__':
    main()
