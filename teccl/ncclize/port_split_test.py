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
from teccl.ncclize.port_split import (Flow, FlowLoad, Piece, SubFlow,        # noqa: E402
                                      assign_ports, assert_port_capacity, flow_loads,
                                      occupancy_grid, port_loads, unqualify_path_key)
from teccl.topologies.two_pod_rail import (TwoPodRail, TwoPodRailHostBound,  # noqa: E402
                                           TwoPodRailSplitPorts,
                                           two_pod_rail_variant)

# The unsplit twin of TwoPodRailHostBound: same graph, same capacities, one port per link.
# Every "declaring ports changes nothing" check compares against this.
TwoPodRailHostBoundOnePort = two_pod_rail_variant(
    "TwoPodRailHostBoundOnePort", gpu_leaf_bw=25.0, leaf_spine_ports=(1, 1))

SCHEDULE = 'Schedules/two_pod_rail_hostbound_allgather_fast_epoch_flat.json'
SPINE_BOUND = 'Schedules/two_pod_rail_allgather_flat.json'
# Every paced two_pod_rail schedule, with the split topology it was solved on.
ALL_CASES = [
    ('two_pod_rail_allgather_flat', TwoPodRailSplitPorts),
    ('two_pod_rail_alltoall_flat', TwoPodRailSplitPorts),
    ('two_pod_rail_hostbound_allgather_flat', TwoPodRailHostBound),
    ('two_pod_rail_hostbound_alltoall_flat', TwoPodRailHostBound),
    ('two_pod_rail_hostbound_allgather_fast_epoch_flat', TwoPodRailHostBound),
    ('two_pod_rail_hostbound_alltoall_fast_epoch_flat', TwoPodRailHostBound),
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
    """A FlowLoad with one piece per epoch, chunk index counting up."""
    return at(flow, {int(e[1:]): v for e, v in per_epoch.items()})


def at(flow, per_epoch):
    return FlowLoad(flow, tuple(
        Piece(flow.src, i, e, r, (e,)) for i, (e, r) in enumerate(sorted(per_epoch.items()))))


def ports_of(a, fl, link):
    """The distinct ports a flow's pieces take on one link."""
    i = fl.flow.hops().index(link)
    return {a.subflow_of(fl.flow, *p.address).ports[i] for p in fl.pieces}


# ----------------------------------------------------------------------------------------------
def test_single_port_is_identity():
    """A topology declaring no ports must behave exactly as it does today."""
    flows = [load(Flow(0, (2,), 1), e0=5.0), load(Flow(3, (2,), 1), e0=5.0)]
    pc, cap = fixed({}, {})
    a = assign_ports(flows, pc, cap)
    assert all(len(subs) == 1 and set(subs[0].ports) == {0}
               for subs in a.subflows.values()), a.subflows
    # an unsplit route keeps its bare switch tuple as the emitted key
    assert a.only(flows[0].flow).key == ((2,), (0, 0))
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
    assert not a.split_flows
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
    assert not a.split_flows, a.split_flows
    ok("heavy-first packs {4,4,6,6} into 2x10 with no splits")


def test_best_fit_compacts_across_hops():
    """The fit rule, tested through the consequence it exists for.

    Nine ports of 100. Seven are filled at hop 0, then two half-port buckets arrive, then a
    full-port flow arrives at hop 2 on the SAME link. Best-fit compacts the two halves onto one
    port and leaves the last port whole, so the hop-2 flow fits intact. Emptiest-fit would
    scatter them onto the two free ports and the hop-2 flow would have to be split.
    """
    pc, cap = fixed({(10, 11): 9}, {(10, 11): 900.0})
    early = [at(Flow(100 + i, (10,), 11), {0: 100.0}) for i in range(7)]
    halves = [at(Flow(200 + i, (10,), 11), {0: 50.0}) for i in range(2)]
    late = [at(Flow(300, (301, 10), 11), {0: 100.0})]   # link (10,11) at hop index 2
    flows = early + halves + late
    a = assign_ports(flows, pc, cap)
    assert_port_capacity(flows, a, pc, cap)
    assert not a.split_flows, f"best-fit should leave a whole port free; got {a.split_flows}"
    q0 = a.only(halves[0].flow).ports[1]
    q1 = a.only(halves[1].flow).ports[1]
    assert q0 == q1, f"the two half buckets should compact onto one port, got {q0} and {q1}"
    qlate = a.only(late[0].flow).ports[2]
    assert qlate not in (q0, q1), f"the hop-2 flow should take a whole port, got {qlate}"
    ok("best-fit compacts, leaving a whole port for a later hop")


def test_vector_not_scalar():
    """Two flows with IDENTICAL totals pack differently depending on when they are active.

    Both halves use two flows of total 10 against ports of 10, so a scalar size model sees the
    same instance twice and must get one of them wrong. Disjoint in time -> they share a port.
    Overlapping -> they cannot.
    """
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 20.0})
    disjoint = [at(Flow(1, (10,), 11), {0: 10.0}), at(Flow(2, (10,), 11), {1: 10.0})]
    a = assign_ports(disjoint, pc, cap)
    assert_port_capacity(disjoint, a, pc, cap)
    assert not a.split_flows
    assert len({a.only(f.flow).ports for f in disjoint}) == 1, \
        "time-disjoint flows should compact onto one port"
    overlap = [at(Flow(1, (10,), 11), {0: 10.0}), at(Flow(2, (10,), 11), {0: 10.0})]
    a2 = assign_ports(overlap, pc, cap)
    assert_port_capacity(overlap, a2, pc, cap)
    assert not a2.split_flows
    assert len({a2.only(f.flow).ports for f in overlap}) == 2, \
        "time-overlapping flows must land on different ports"
    ok("placement is by epoch vector, not scalar total")


def test_sub_flows_become_distinct_routes():
    """A flow that fits neither port whole becomes SEVERAL SUBFLOWS, each its own route.

    Two ports of 100. `a` loads port 0 in epoch 0 only; `b` is pushed to port 1 and loads it
    mostly in epoch 1. That leaves the two ports with complementary residuals -- port 0 tight in
    epoch 0, port 1 tight in epoch 1 -- so `x`, which spans both, fits neither port whole while
    fitting SOME port in each epoch. That is exactly the vector-packing failure epoch slicing
    exists for, and the two slices must produce different qualified path keys, or the emitted
    program would name one wire for bytes that cross two.
    """
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 200.0})       # 100 per port
    a_ = at(Flow(1, (10,), 11), {0: 90.0})
    b_ = at(Flow(2, (10,), 11), {0: 20.0, 1: 90.0})
    x_ = at(Flow(3, (10,), 11), {0: 30.0, 1: 30.0})
    flows = [a_, b_, x_]
    a = assign_ports(flows, pc, cap)
    assert_port_capacity(flows, a, pc, cap)

    assert [str(f) for f in a.split_flows] == [str(x_.flow)], a.split_flows
    subs = a.subflows[x_.flow]
    assert len(subs) == 2, [str(s) for s in subs]
    assert subs[0].flow == subs[1].flow, "same flow ..."
    assert subs[0].ports != subs[1].ports, "... different ports, so different routes"

    # the identity is the port-qualified path -- nothing temporal in it
    assert subs[0].key == (x_.flow.switches, subs[0].ports)
    assert subs[0].key != subs[1].key

    # only() must refuse a partitioned flow rather than pick a representative
    try:
        a.only(x_.flow)
    except KeyError:
        pass
    else:
        raise AssertionError("only() must refuse a partitioned flow")

    # every piece resolves to one of the two, and an unpartitioned flow to a single key
    rode = {a.subflow_of(x_.flow, *p.address) for p in x_.pieces}
    assert rode == set(subs), rode
    assert len({a.subflow_of(b_.flow, *p.address).key for p in b_.pieces}) == 1
    ok("a partitioned flow yields several subflows, each a distinct port-qualified route")


def test_indivisible_piece_is_refused():
    """A piece is the floor: bigger than every port and there is nothing left to partition.

    A piece is one addressable send, so once tier 3 is placing pieces individually there is no
    finer grain to fall back to. That is a statement about the TOPOLOGY, not the packing, and
    the error has to say so rather than blaming the split.
    """
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 100.0})       # 50 per port
    flows = [at(Flow(1, (10,), 11), {0: 60.0}), at(Flow(2, (10,), 11), {0: 40.0})]
    try:
        assign_ports(flows, pc, cap)
    except AssertionError as e:
        assert 'nothing left' in str(e) and 'cannot carry this schedule' in str(e), e
        ok("an indivisible over-capacity piece is refused, blaming the topology")
        return
    raise AssertionError("expected an over-capacity piece to be refused")


def test_greedy_exhaustion_is_named_as_such():
    """The other refusal: the piece fits a port in principle, but no residual is left.

    Same exception shape, completely different cause -- the greedy packing spent the room, and
    the instance may well be feasible. Merging the two sends the reader to the wrong place.
    """
    pc, cap = fixed({(10, 11): 2}, {(10, 11): 100.0})       # 50 per port
    flows = [at(Flow(1, (10,), 11), {0: 50.0}), at(Flow(2, (10,), 11), {0: 50.0}),
             at(Flow(3, (10,), 11), {0: 10.0})]
    try:
        assign_ports(flows, pc, cap)
    except AssertionError as e:
        assert 'greedy packing failed' in str(e), e
        assert 'nothing left' not in str(e), "must not be reported as an indivisible piece"
        ok("greedy exhaustion is reported as such, not as an over-capacity piece")
        return
    raise AssertionError("expected the third flow to find no residual")


def test_hairpin_is_rejected():
    """A path revisiting a node breaks the hop ordering argument; refuse rather than guess."""
    pc, cap = fixed({}, {})
    try:
        assign_ports([at(Flow(1, (2, 3, 2), 4), {0: 1.0})], pc, cap)
    except AssertionError as e:
        assert 'repeats a node' in str(e), e
        ok("a hairpin path is rejected, not silently mis-ordered")
        return
    raise AssertionError("expected a hairpin path to be rejected")


# ----------------------------------------------------------------------------------------------
def test_topology_ports_do_not_touch_the_solve():
    base = TwoPodRailHostBoundOnePort(TopologyParams())
    split = TwoPodRailHostBound(TopologyParams())
    assert base.capacity == split.capacity and base.alpha == split.alpha, \
        "declaring ports must not change what the solver sees"
    assert base.ports == [] and base.port_count(24, 28) == 1
    assert TwoPodRail(TopologyParams()).ports == [], "TwoPodRail stays the unsplit reference"
    assert split.port_count(24, 28) == 2 and split.port_capacity(24, 28) == 25.0
    assert split.port_count(24, 29) == 1 and split.port_capacity(24, 29) == 25.0
    assert split.port_count(0, 24) == 1 and split.port_capacity(0, 24) == 25.0
    ok("port declaration leaves capacity/alpha byte-identical")


def _reference():
    topo = TwoPodRailHostBound(TopologyParams())
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


def test_sub_flow_key_round_trip():
    """A SubFlow's key is (switches, ports), and unqualify_path_key splits it back."""
    sub = SubFlow(Flow(0, (24, 28, 26), 10), (0, 1, 0, 0))
    assert sub.key == ((24, 28, 26), (0, 1, 0, 0))
    assert unqualify_path_key(sub.key) == ((24, 28, 26), (0, 1, 0, 0))
    # an unqualified key (a topology with no ports declared) passes through untouched
    assert unqualify_path_key((24, 28, 26)) == ((24, 28, 26), None)
    assert unqualify_path_key(None) == (None, None)
    ok("SubFlow.key is (switches, ports) and round-trips")


def test_reference_schedule():
    topo, loads = _reference()
    a = assign_ports(loads, topo.port_count, topo.port_capacity)
    assert_port_capacity(loads, a, topo.port_count, topo.port_capacity)
    assert not a.split_flows, f"expected every flow to keep one subflow, got {a.split_flows}"
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
        for sub in a.subflows[fl.flow]:
            after[(fl.flow.src, fl.flow.dst)].add(sub.key)
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
    assert not a.split_flows, a.split_flows
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
        assert not a.split_flows, f"{name}: {len(a.split_flows)} partitioned flows"
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

    f0, r0, _m0, n0, g0, c0 = emit(TwoPodRailHostBoundOnePort(TopologyParams()))
    f1, r1, m1, n1, g1, c1 = emit(TwoPodRailHostBound(TopologyParams()))

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


def test_send_uplink_deconstructs_the_key():
    """The contention pool must DECONSTRUCT a path key, never index it.

    `_send_uplink` groups sends by the uplink they contend for, as (src, first_switch, port).
    Indexing `path_key[0]` gives the first switch for a bare key but the WHOLE switch tuple for
    a port-qualified one -- which silently hands every distinct route its own contention pool
    and stops sends that really do share an uplink from pacing against each other. It cost 396
    of 768 gate edges the first time, and nothing else would have caught it: the gate COUNT is
    unchanged, only the producers move.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from teccl_ncclize import _send_uplink
    except ImportError as e:
        print(f"  [SKIP] send uplink ({e})")
        return
    bare = _send_uplink((7, 0, 10, (24, 28, 26)))
    assert bare == (0, 24, 0), bare
    # same route, port-qualified: same uplink, same pool
    same = _send_uplink((7, 0, 10, ((24, 28, 26), (0, 1, 1, 0))))
    assert same == (0, 24, 0), same
    # a different route out of the SAME first switch shares the pool ...
    assert _send_uplink((7, 0, 11, ((24, 29, 27), (0, 0, 0, 0)))) == (0, 24, 0)
    # ... a different first switch does not ...
    assert _send_uplink((7, 0, 11, ((25, 29, 27), (0, 0, 0, 0)))) != (0, 24, 0)
    # ... and neither does the same first switch reached on a different PORT
    assert _send_uplink((7, 0, 10, ((24, 28, 26), (1, 0, 0, 0)))) == (0, 24, 1)
    assert _send_uplink((7, 0, 10, None)) == (0, None, 0)
    ok("_send_uplink deconstructs the key and takes the first hop's port")


def test_split_does_not_move_the_pacing_gates():
    """End to end: declaring ports must leave every gate edge exactly where it was.

    The gate manifest is keyed on (step, gpu, peer, path_key), so a key-shape change that any
    consumer mishandles shows up here as gates pointing at different producers -- with the same
    COUNT, which is why counting is not enough.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from teccl_ncclize import build_algorithm
    except ImportError as e:
        print(f"  [SKIP] gate stability ({e})")
        return
    sched = json.load(open(SCHEDULE))

    def gates_of(topo):
        _algo, _fpk, _srm, _v, _rate, gates = build_algorithm(sched, topology=topo)
        strip = lambda k: k[:3] + (unqualify_path_key(k[3])[0],)      # noqa: E731
        return {(strip(c), strip(p), kind) for c, p, kind in gates}

    g0 = gates_of(TwoPodRailHostBoundOnePort(TopologyParams()))
    g1 = gates_of(TwoPodRailHostBound(TopologyParams()))
    assert g0 == g1, f"{len(g0 - g1)} gate edges moved when ports were declared"
    ok(f"declaring ports moves none of the {len(g0)} pacing gate edges")


def test_sub_flow_reaches_the_emitted_key():
    """End to end through the qualifier: each subflow becomes a distinct path key.

    No shipped schedule splits, so this drives `_build_port_qualifier` on a hand-built one --
    the same complementary-residual shape as the unit test above, expressed as "7-Flows" lines.
    It is the only coverage of the wiring from a SubFlow to the key ncclize allocates channels,
    flow ids and forwarding entries on.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from teccl_ncclize import _build_port_qualifier
    except ImportError as e:
        print(f"  [SKIP] sub-flow emission ({e})")
        return

    def line(src, dst, chunk, epoch, rate):
        return (f"Chunk {chunk} from {src} traveled over {src}->{dst} with volume 1 "
                f"in epoch {epoch} at rate {rate} via switches 10")
    schedule = {
        '1-Epoch_Duration': 1.0, '9-Chunk_Size': 1.0,
        '7-Flows': [line(1, 11, 0, 0, 90),                          # fills port 0 in epoch 0
                    line(2, 11, 0, 0, 20), line(2, 11, 1, 1, 90),   # to port 1, fills epoch 1
                    line(3, 11, 0, 0, 30), line(3, 11, 1, 1, 30)],  # fits neither port whole
    }

    class Stub:                       # only these three attributes are read
        ports = True
        programmable_switch_indices = [10]
        def port_count(self, i, j):
            return 2 if 10 in (i, j) else 1
        def port_capacity(self, i, j):
            return 100.0 if 10 in (i, j) else 1e9

    qualify, assignment = _build_port_qualifier(schedule, Stub(), lp_format=False)
    assert len(assignment.split_flows) == 1, assignment.split_flows
    x = assignment.split_flows[0]

    # the two subflows of the partitioned flow reach the emitted key as two routes ...
    keys = {qualify(3, 11, (10,), 3, 0, 0),      # (origin, chunk, epoch) of each piece
            qualify(3, 11, (10,), 3, 1, 1)}
    assert keys == {s.key for s in assignment.subflows[x]}, keys
    assert len(keys) == 2, keys
    # ... and an unpartitioned flow emits ONE key for every one of its pieces, so its
    # (channel, peer) connection stays single
    assert qualify(2, 11, (10,), 2, 0, 0) == qualify(2, 11, (10,), 2, 1, 1)
    ok(f"a partitioned flow emits one key per subflow: {sorted(keys)}")


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
                q = rng.getrandbits(1)          # ONE choice per flow: ECMP hashes the 5-tuple,
                for p in fl.pieces:             # so a connection's packets all take one port
                    for e in p.span:
                        used[(hop, q)][e] += p.rate
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
    test_indivisible_piece_is_refused()
    test_greedy_exhaustion_is_named_as_such()
    test_hairpin_is_rejected()
    print(" topology")
    test_topology_ports_do_not_touch_the_solve()
    test_sub_flow_key_round_trip()
    print(" regression -- two_pod_rail_hostbound_allgather_fast_epoch_flat")
    test_occupancy_grid_is_derived()
    test_reference_schedule()
    test_path_keys_unchanged()
    test_spine_bound_exercises_the_bucket_split()
    test_every_paced_two_pod_rail_schedule()
    test_negative_control_random_hash()
    print(" emission")
    test_ncclize_emission()
    test_send_uplink_deconstructs_the_key()
    test_split_does_not_move_the_pacing_gates()
    test_sub_flow_reaches_the_emitted_key()
    print(f"\n{len(_passed)} passed")


if __name__ == '__main__':
    main()
