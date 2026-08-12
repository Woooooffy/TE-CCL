"""
Gurobi-free structural tests for the RING base-case row (teccl.hierarchy.ring_solve), the peer of
teccl.hierarchy.crossbar_solve in the recursion's closed-form dispatch table.

  [1] is_ring / ring_topology_order: what counts as a ring (uni- and bidirectional), and what does
      not (a chord, two cycles, a switch, a path).
  [2] the flag: TECCL_INTRA_ALGO / ring_solve.INTRA_ALGO routes a single-switch NVSwitch cell to the
      ring row, and the dispatch table agrees at both dispatch points.
  [3] ring allgather on an 8-GPU cell: recovers the textbook N-1 rounds with a perfect matching per
      round, which is the property the whole row exists for.
  [4] end-to-end on the real HeteroTaperedCluster Host-B resolution (relays + internal allgather),
      the same fixture the crossbar test uses: hard deadlines met, links within capacity, every
      demand delivered.
  [5] the A/B the flag is for: ring vs crossbar on identical input, reporting round count and
      per-GPU peer count so the trade is measured rather than asserted.

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_ring_solve_test
"""
from collections import defaultdict
from types import SimpleNamespace

from teccl.hierarchy import crossbar_solve, ring_solve, solve as solve_mod
from teccl.hierarchy.cell import Cell
from teccl.hierarchy.reconstruct import IntraCellDemand
from teccl.hierarchy.ring_solve import (CW, CCW, RingOrder, is_ring, ring_topology_order,
                                        schedule_cell)

from teccl.examples.hierarchy_crossbar_solve_test import _hostB_resolution, _by_cell


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _graph(n, edges, switches=(), passive=()):
    """A minimal Topology-shaped stand-in: is_ring/is_crossbar read only these three attributes."""
    cap = [[0.0] * n for _ in range(n)]
    for u, v in edges:
        cap[u][v] = 1.0
    return SimpleNamespace(capacity=cap, switch_indices=list(switches),
                           passive_indices=list(passive))


def _cycle(n, both=True):
    e = [(i, (i + 1) % n) for i in range(n)]
    if both:
        e += [((i + 1) % n, i) for i in range(n)]
    return _graph(n, e)


def _star(n_gpus):
    """n GPUs hanging off one switch: the NVSwitch / crossbar shape."""
    sw = n_gpus
    e = [(i, sw) for i in range(n_gpus)] + [(sw, i) for i in range(n_gpus)]
    return _graph(n_gpus + 1, e, switches=[sw])


def _nvswitch_cell(n_gpus=8):
    gpus = list(range(n_gpus))
    return Cell(members=gpus + [n_gpus], gpus=gpus, internal_switches=[n_gpus])


def _per_round(flows):
    """(band, round) -> (senders, receivers) counted per GPU."""
    tx, rx = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for f in flows:
        for i in range(f.span):
            tx[(f.band, f.local_round + i)][f.sender] += 1
            rx[(f.band, f.local_round + i)][f.receiver] += 1
    return tx, rx


def _peers(flows):
    """distinct inbound and outbound peers per GPU across the WHOLE schedule."""
    inc, out = defaultdict(set), defaultdict(set)
    for f in flows:
        inc[f.receiver].add(f.sender)
        out[f.sender].add(f.receiver)
    return inc, out


# ---------------------------------------------------------------------------------------------
# [1] detection
# ---------------------------------------------------------------------------------------------
def test_detection():
    bi = ring_topology_order(_cycle(8, both=True))
    assert bi is not None and bi.bidirectional and len(bi.gpus) == 8, bi
    assert set(bi.gpus) == set(range(8))
    # consecutive in the detected order means adjacent in the graph
    for i in range(8):
        u, v = bi.gpus[i], bi.gpus[(i + 1) % 8]
        assert bi.is_edge(u, v), (u, v)

    uni = ring_topology_order(_cycle(6, both=False))
    assert uni is not None and not uni.bidirectional, uni
    assert uni.directions() == (CW,)

    # a ring wired the other way round is still a ring, just re-oriented
    rev = ring_topology_order(_graph(5, [((i + 1) % 5, i) for i in range(5)]))
    assert rev is not None and not rev.bidirectional

    assert not is_ring(_star(8)), "an NVSwitch cell is a crossbar, not a physical ring"
    assert not is_ring(_graph(4, [(0, 1), (1, 2), (2, 3)])), "a path is not a ring"
    # a chord: every optimal-schedule claim the row makes would be wrong
    chorded = _cycle(6, both=True)
    chorded.capacity[0][3] = chorded.capacity[3][0] = 1.0
    assert not is_ring(chorded), "a cycle with a chord must be rejected"
    # two disjoint triangles
    two = _graph(6, [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2),
                     (3, 4), (4, 5), (5, 3), (4, 3), (5, 4), (3, 5)])
    assert not is_ring(two), "two disjoint cycles must be rejected"
    assert not is_ring(_cycle(2, both=True)), "a 2-node ring is a direct link, not a ring"

    # distances and directions
    o = RingOrder(gpus=(0, 1, 2, 3, 4, 5, 6, 7), bidirectional=True, via_switch=None)
    assert o.distance(0, 3, CW) == 3 and o.distance(0, 3, CCW) == 5
    assert o.best_direction(0, 3) == CW and o.best_direction(0, 5) == CCW
    assert o.best_direction(0, 4) == CW, "an exact tie goes clockwise"
    assert o.node_at(0, 3, CCW) == 5
    print("  [1] detection: uni/bi rings, re-orientation, and the four rejections OK")


# ---------------------------------------------------------------------------------------------
# [2] the flag and the dispatch table
# ---------------------------------------------------------------------------------------------
def test_flag_and_dispatch():
    star, ring = _star(8), _cycle(8)
    prev = ring_solve.INTRA_ALGO
    try:
        ring_solve.INTRA_ALGO = ring_solve.ALGO_CROSSBAR
        assert not ring_solve.should_use_ring(star, cell_fabric=True), \
            "default must leave the NVSwitch on crossbar"
        assert ring_solve.should_use_ring(ring), "a physical ring ignores the flag"
        assert solve_mod._memoized_row(star, cell_fabric=True).name == "crossbar"
        assert solve_mod._memoized_row(ring, cell_fabric=True).name == "ring"

        ring_solve.INTRA_ALGO = ring_solve.ALGO_RING
        assert ring_solve.should_use_ring(star, cell_fabric=True), \
            "the flag must force the NVSwitch node onto the ring"
        assert solve_mod._memoized_row(star, cell_fabric=True).name == "ring", \
            "row order must let the flag win on a cell fabric"
        assert solve_mod._memoized_row(ring, cell_fabric=True).name == "ring"

        # SHAPE selection is identical at both dispatch points -- a physical ring is a ring
        # whether it is a level's graph or a cell's interior, exactly as a crossbar is a crossbar.
        assert solve_mod._memoized_row(ring).name == "ring", "is_ring must claim a LEVEL too"
        # ...but the FLAG is scoped to a node fabric by its own definition, so a crossbar LEVEL
        # stays on the crossbar row. Forcing it would mean routing the level as a multi-hop ring,
        # i.e. relaying through an intermediate cell, which is unimplemented across the whole
        # lowering half rather than in either row.
        assert solve_mod._memoized_row(star).name == "crossbar", \
            "the node flag must not reach into level routing"

        # a shape neither row claims still falls through to a real solver
        assert solve_mod._memoized_row(_graph(4, [(0, 1), (1, 2), (2, 3)])) is None

        ring_solve.INTRA_ALGO = "nonsense"
        try:
            ring_solve.intra_algo()
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown INTRA_ALGO must be rejected, not silently defaulted")
    finally:
        ring_solve.INTRA_ALGO = prev
    print("  [2] flag: default=crossbar, forced=ring, physical ring unconditional, bad value "
          "rejected OK")


# ---------------------------------------------------------------------------------------------
# [3] the textbook ring allgather
# ---------------------------------------------------------------------------------------------
def test_ring_allgather():
    """8 GPUs, each broadcasting its own chunk to the other 7 -- the allgather shape.

    On a unidirectional ring the answer is the classic one: 7 rounds, and in EVERY round every GPU
    sends exactly one chunk and receives exactly one. That per-round perfect matching is the
    property the row exists to make structural, so it is checked directly rather than inferred from
    the round count.
    """
    n = 8
    cell = _nvswitch_cell(n)
    demands = [IntraCellDemand(cell=0, kind="self_distribution", identity=(g, 0), src_gpu=g,
                               dst_gpus=tuple(x for x in range(n) if x != g), volume=1.0,
                               deadline_epoch=0)
               for g in range(n)]
    flows = schedule_cell(0, cell, demands, debug=False)

    assert len(flows) == n * (n - 1), f"expected {n * (n - 1)} hops, got {len(flows)}"
    rounds = max(f.local_round for f in flows) + 1
    assert rounds == n - 1, f"ring allgather should take {n - 1} rounds, took {rounds}"

    tx, rx = _per_round(flows)
    for key in sorted(tx):
        assert set(tx[key].values()) == {1} and len(tx[key]) == n, (key, dict(tx[key]))
        assert set(rx[key].values()) == {1} and len(rx[key]) == n, (key, dict(rx[key]))

    inc, out = _peers(flows)
    assert all(len(v) == 1 for v in inc.values()), "a ring gives each GPU ONE inbound peer"
    assert all(len(v) == 1 for v in out.values()), "a ring gives each GPU ONE outbound peer"
    assert all(f.via_switch == n for f in flows), "a logical ring must name the cell's switch"
    print(f"  [3] ring allgather: {len(flows)} hops in {rounds} rounds, perfect matching every "
          f"round, 1 peer in / 1 peer out OK")


def test_ring_bidirectional_halves_depth():
    """A PHYSICAL bidirectional ring routes the shorter way, halving the chain depth."""
    n = 8
    gpus = list(range(n))
    cell = Cell(members=gpus, gpus=gpus, internal_switches=[])
    demands = [IntraCellDemand(cell=0, kind="self_distribution", identity=(0, 0), src_gpu=0,
                               dst_gpus=tuple(range(1, n)), volume=1.0, deadline_epoch=0)]
    flows = schedule_cell(0, cell, demands, debug=False, topology=_cycle(n, both=True))
    assert len(flows) == n - 1, f"one broadcast is still {n - 1} hops, got {len(flows)}"
    rounds = max(f.local_round for f in flows) + 1
    assert rounds == 4, f"splitting both ways should take ceil(7/2)=4 rounds, took {rounds}"
    assert {f.sender for f in flows} | {f.receiver for f in flows} == set(gpus)
    assert all(f.via_switch is None for f in flows), "a physical ring hop has no switch"
    print(f"  [4] bidirectional ring: {len(flows)} hops, depth {rounds} (vs 7 unidirectional) OK")


# ---------------------------------------------------------------------------------------------
# [4] the real hetero fixture
# ---------------------------------------------------------------------------------------------
def test_hetero_hostB():
    """The Host-B forced-relay resolution: hard egress staging plus an internal allgather.

    Same fixture the crossbar row is tested on, so a failure here is the ring lowering and not the
    input. The hard part is the interaction the crossbar row gets for free: an egress relay and a
    self_distribution of the SAME identity share their first hops, and the shared prefix has to
    inherit the relay's hard deadline while the rest of the arc stays soft.
    """
    _topo, mapping, res = _hostB_resolution()
    by_cell = _by_cell(res)
    total = 0
    for cid, demands in sorted(by_cell.items()):
        cell = mapping.coarse_cells[cid]
        # schedule_cell's own assertions do the verifying: every hop is a ring edge, no link is
        # oversubscribed in a round, every hard job completed, and every demand's destinations end
        # up holding its identity (replayed in round order, so a broken chain is caught too).
        flows = schedule_cell(cid, cell, demands, debug=False)
        inc, out = _peers(flows)
        assert all(len(v) <= 1 for v in inc.values()), (cid, {k: sorted(v) for k, v in inc.items()})
        assert all(len(v) <= 1 for v in out.values()), (cid, {k: sorted(v) for k, v in out.items()})
        total += len(flows)
    print(f"  [5] hetero Host-B: {total} ring hops across {len(by_cell)} cells; deadlines, links "
          f"and deliveries all verified OK")


# ---------------------------------------------------------------------------------------------
# [5] the A/B
# ---------------------------------------------------------------------------------------------
def _ab(name, demands, n=8, cell=None):
    cell = cell if cell is not None else _nvswitch_cell(n)
    xbar = crossbar_solve.schedule_cell(0, cell, demands, debug=False)
    ring = schedule_cell(0, cell, demands, debug=False)
    out = {}
    for label, flows in (("crossbar", xbar), ("ring", ring)):
        inc, _o = _peers(flows)
        out[label] = (len(flows), max(f.local_round for f in flows) + 1,
                      max((len(v) for v in inc.values()), default=0))
    print(f"      {name:<24} {'hops':>6} {'rounds':>7} {'max fan-in':>11}")
    for label in ("crossbar", "ring"):
        h, r, fi = out[label]
        print(f"        {label:<20} {h:>6} {r:>7} {fi:>11}")
    return out


def test_ab_summary():
    """Measure the trade the flag exists to explore, rather than asserting a preferred answer."""
    n = 8
    allgather = [IntraCellDemand(cell=0, kind="self_distribution", identity=(g, 0), src_gpu=g,
                                 dst_gpus=tuple(x for x in range(n) if x != g), volume=1.0,
                                 deadline_epoch=0)
                 for g in range(n)]
    # alltoall shape: one gateway holds 8 identities, each wanted by exactly one GPU.
    alltoall = [IntraCellDemand(cell=0, kind="self_distribution", identity=(0, t), src_gpu=0,
                                dst_gpus=(t,), volume=1.0, deadline_epoch=0)
                for t in range(1, n)]
    print("  [6] A/B on identical input (the point of the flag):")
    ag = _ab("allgather fan-out", allgather)
    a2a = _ab("point-to-point (a2a)", alltoall)

    # The allgather shape is where the ring is free: same hop count, same round count, and the
    # fan-in collapses from 7 to 1. This is the claim the row is worth having for.
    assert ag["ring"][0] == ag["crossbar"][0], "allgather: ring must not move more bytes"
    assert ag["ring"][1] == ag["crossbar"][1], "allgather: ring must not cost rounds"
    assert ag["ring"][2] == 1 < ag["crossbar"][2], "allgather: ring must collapse the fan-in"
    # The point-to-point shape is where it is not: the ring pays distance(src, dst) hops for what
    # the crossbar does in one. Asserted so a future change cannot quietly claim otherwise.
    assert a2a["ring"][0] > a2a["crossbar"][0], "point-to-point: ring is expected to move MORE"

    # The same comparison on the REAL hetero fixture, where the cells are 4/4/6 GPUs and the demand
    # mixes hard egress staging with an internal allgather -- i.e. not a shape either row was tuned
    # for. Reported rather than asserted: the interesting number is the fan-in column.
    _topo, mapping, res = _hostB_resolution()
    for cid, demands in sorted(_by_cell(res).items()):
        cell = mapping.coarse_cells[cid]
        _ab(f"hetero cell {cid} ({len(cell.gpus)} gpus)", demands, cell=cell)
    print("      -> ring is free on the fan-out shape and costs traffic on the point-to-point one")


def main():
    print("ring base-case row tests")
    test_detection()
    test_flag_and_dispatch()
    test_ring_allgather()
    test_ring_bidirectional_halves_depth()
    test_hetero_hostB()
    test_ab_summary()
    print("ring solve tests OK")


if __name__ == "__main__":
    main()
