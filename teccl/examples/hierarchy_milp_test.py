"""The hierarchical solver on the MILP (teccl/hierarchy/milp_design.md).

Gurobi-free:
  [1] the count demand: every milp_chunking policy's bundle size, column count and anchor
  [2] integral identity resolution UNCROSSES the MILP's arbitrary labels (the gpu0/gpu1 case)
  [3] a tree's shared leg is placed once and emitted once; the level stays Q == 1
  [4] occupancy pacing, and the rate check counting a multi-epoch piece in every epoch it covers
  [5] a hierarchical level defaults to LP; per-level formulation / chunking lists resolve by depth

Gurobi (size-limited license is enough; SKIPPED if the model does not fit):
  [6] end to end on the bridged-islands pair, co-located and RELAYED transit, through ncclize's
      check_implements

Run from the repo root:
    PYTHONPATH=. conda run -n teccl python teccl/examples/hierarchy_milp_test.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

from teccl.hierarchy.abstract import abstract, coarsify_demand
from teccl.hierarchy.milp_routing import (MilpGroup, PreservingDemand, assign_identities_integral,
                                          coarsify_demand_preserving)
from teccl.hierarchy.reconstruct import (IdentityResolution, ResolvedPiece, _CoarsePiece,
                                         _CoarseRoute, _assert_rate_within_capacity, _piece_rate,
                                         build_child_problems)
from teccl.hierarchy.scale import ChunkScale
from teccl.input_data import (Collective, EpochType, Formulation, InstanceParams, ObjectiveType,
                              SolutionMethod, TopologyParams, UserInputParams,
                              resolve_milp_chunking)
from teccl.solvers.demand import build_demand
from teccl.topologies.bridged_islands_cluster import (BridgedIslandsCluster,
                                                       BridgedIslandsSplitCluster)
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster


def _hetero():
    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    coarse, mapping = abstract(topo)
    return topo, coarse, mapping


def _columns(demand):
    return {g.cell: 0 for g in demand.groups} | {
        cell: sum(len(g.columns) for g in demand.groups if g.cell == cell)
        for cell in {g.cell for g in demand.groups}}


def test_count_demand():
    topo, _coarse, mapping = _hetero()
    ag1 = build_demand(Collective.ALLGATHER, topo, 1)
    ag4 = build_demand(Collective.ALLGATHER, topo, 4)
    a2a = build_demand(Collective.ALLTOALL, topo, 14)

    fine = coarsify_demand_preserving(ag1, mapping, "fine")
    assert fine.b == 1 and _columns(fine) == {0: 4, 1: 4, 2: 6}, (fine.b, _columns(fine))
    holder1 = coarsify_demand_preserving(ag1, mapping, "holder")
    holder4 = coarsify_demand_preserving(ag4, mapping, "holder")
    # "holder" bundles one GPU's chunks for one destination set: num_chunks stops multiplying the
    # model, and the bundle (b) absorbs it instead -- which is what the level then rescales by.
    assert holder1.b == 1 and holder4.b == 4, (holder1.b, holder4.b)
    assert _columns(holder1) == _columns(holder4) == {0: 4, 1: 4, 2: 6}
    assert coarsify_demand_preserving(ag4, mapping, "fine").b == 1
    assert _columns(coarsify_demand_preserving(ag4, mapping, "fine")) == {0: 16, 1: 16, 2: 24}
    cell = coarsify_demand_preserving(ag1, mapping, "cell")
    assert cell.b == 2 and _columns(cell) == {0: 2, 1: 2, 2: 3}, (cell.b, _columns(cell))
    two = coarsify_demand_preserving(ag4, mapping, 2)
    assert two.b == 2 and _columns(two) == {0: 8, 1: 8, 2: 12}
    try:
        coarsify_demand_preserving(ag4, mapping, 3)
        raise AssertionError("b=3 does not divide a holder's 4 chunks and must be refused")
    except ValueError:
        pass
    # alltoall: a holder's class toward cell V has |V| identities, and |V| differs (4, 4, 6), so
    # "holder" falls back to their gcd (2) rather than one chunk per class.
    h2 = coarsify_demand_preserving(a2a, mapping, "holder")
    assert h2.b == 2, h2.b
    # The tensor is binary and each column is wanted by exactly its group's destination cells.
    for d in (fine, holder4, cell, two, h2):
        for g in d.groups:
            for c in g.columns:
                wanted = {v for v in range(mapping.num_coarse) if d.tensor[g.cell][v][c]}
                assert wanted == set(g.dest_cells), (g, c, wanted)
    print("  [1] count demand: fine/holder/cell/int bundle sizes, column counts and anchor OK")


def _leg(src, dst, egress_nb, ingress_nb, via, epoch):
    return _CoarsePiece(src_cell=src, dst_cell=dst, egress_neighbor=egress_nb,
                        ingress_neighbor=ingress_nb, via_switches=via, volume=1.0,
                        send_epoch=epoch, arrival_epoch=epoch, origin=(src, src))


def _crossed_fixture():
    """Host A of HeteroTaperedCluster: g0 owns the uplink to T0, g1 the one to T1, g2/g3 own none.

    Four MILP columns, all of A's allgather data, each a tree to B and C. The MILP's labels make
    column i carry identity (i, 0) -- and they are CROSSED: column 0 leaves on T1 (g1's uplink)
    while identity (0, 0) lives on g0, and column 1 leaves on T0 (g0's) while (1, 0) lives on g1.
    Read off the labels, that is two egress relays the coarse level cannot see. Columns 2 and 3 do
    the same one epoch later.
    """
    topo, _coarse, mapping = _hetero()
    A, B, C, T0, T1 = 0, 1, 2, 3, 4
    t0, t1 = mapping.coarse_passthrough[T0], mapping.coarse_passthrough[T1]
    trees = {}
    for col, (nb, epoch) in enumerate([(T1, 0), (T0, 0), (T1, 1), (T0, 1)]):
        if nb == T1:
            to_c = _leg(A, C, T1, T1, (t1,), epoch)
            to_b = _leg(A, B, T1, T0, (t1, t0), epoch)
        else:
            to_c = _leg(A, C, T0, T0, (t0,), epoch)
            to_b = _leg(A, B, T0, T0, (t0,), epoch)
        trees[(A, col)] = {B: _CoarseRoute(origin=(A, B), legs=(to_b,), volume=1.0),
                           C: _CoarseRoute(origin=(A, C), legs=(to_c,), volume=1.0)}
    ids = [(g, 0) for g in range(4)]
    demand = PreservingDemand(
        tensor=[], b=1, policy="fine",
        groups=[MilpGroup(cell=A, dest_cells=(B, C), identities=ids,
                          holders={d: d[0] for d in ids}, columns=[0, 1, 2, 3])])
    return topo, mapping, trees, demand


def test_uncrossed():
    topo, mapping, trees, demand = _crossed_fixture()
    fine = build_demand(Collective.ALLGATHER, topo, 1)
    assignments, targets, epoch = assign_identities_integral(trees, demand, mapping, fine, topo,
                                                             epoch_duration=0.04)
    egress = {}
    for a in assignments:
        egress.setdefault(a.identity, set()).add((a.egress_gpu, a.route.first.send_epoch))
    assert egress[(0, 0)] == {(0, 0)}, f"g0's data must leave on g0's own uplink: {egress}"
    assert egress[(1, 0)] == {(1, 0)}, f"g1's data must leave on g1's own uplink: {egress}"
    # g2/g3 own no uplink, so they relay whatever they ride -- and the epoch tier puts the relayed
    # identities on the LATER columns, leaving epoch 0 to the natively-held data.
    assert all(e == 1 for d in ((2, 0), (3, 0)) for (_g, e) in egress[d]), egress
    relays = sum(1 for d, s in egress.items() for g, _e in s if g != d[0])
    assert relays == 2, (relays, egress)
    print(f"  [2] crossed MILP labels uncrossed: g0 -> T0, g1 -> T1 natively, only the 2 "
          f"uplink-less GPUs relay, and they take epoch 1")
    return assignments, targets, mapping, fine, topo


def test_shared_leg_once():
    # One tree, A -> B and A -> B -> C (B wants AND forwards), through the bridged co-located host.
    topo = BridgedIslandsCluster(TopologyParams(name="BridgedIslandsCluster", chunk_size=1))
    _coarse, mapping = abstract(topo)
    A, B, C, T0, T1 = 0, 1, 2, 3, 4
    t0, t1 = mapping.coarse_passthrough[T0], mapping.coarse_passthrough[T1]
    ab = _leg(A, B, T0, T0, (t0,), 0)
    bc = _leg(B, C, T1, T1, (t1,), 1)
    trees = {(A, c): {B: _CoarseRoute(origin=(A, B), legs=(ab,), volume=1.0),
                      C: _CoarseRoute(origin=(A, C), legs=(ab, bc), volume=1.0)}
             for c in (0,)}
    ids = [(0, 0)]
    demand = PreservingDemand(tensor=[], b=1, policy="fine", groups=[
        MilpGroup(cell=A, dest_cells=(B, C), identities=ids, holders={(0, 0): 0}, columns=[0])])
    fine = build_demand(Collective.ALLGATHER, topo, 1)
    # Restrict the fine demand to identity (0, 0) so step B's self-distribution is about it alone.
    for s in range(len(fine)):
        if s != 0:
            fine[s] = fine[s] * 0
    assignments, targets, epoch = assign_identities_integral(trees, demand, mapping, fine, topo,
                                                             epoch_duration=0.02)
    res = build_child_problems(assignments, targets, mapping, fine, topo, epoch, copy_aware=True)
    legs = [(p.identity, p.src_cell, p.dst_cell, p.send_epoch) for p in res.pieces]
    assert legs.count(((0, 0), A, B, 0)) == 1, f"the shared A -> B leg must ship once: {legs}"
    assert len(legs) == 2 and res.subdivision == 1, (legs, res.subdivision)
    # the same GPU of B lands A's data and forwards it (co-located), so B needs no relay
    landed = next(p.ingress_gpu for p in res.pieces if p.dst_cell == B)
    sent = next(p.egress_gpu for p in res.pieces if p.src_cell == B)
    assert landed == sent, (landed, sent)
    # without copy_aware the LP semantics would keep both copies -- the dedup is MILP-only
    res_lp = build_child_problems(assignments, targets, mapping, fine, topo, epoch)
    assert len(res_lp.pieces) == 3, len(res_lp.pieces)
    print("  [3] tree's shared leg placed once and shipped once (2 pieces, not 3), Q=1, "
          "co-located forward")


def test_transit_lands_where_it_can_forward():
    """Regression for the mini_2gpu_2nic remote failure (2026-10-12).

    Both GPUs of a mini_2gpu_2nic host hang off the SAME leaf, so either can land a chunk from
    the leaf and re-send it to the leaf -- co-location is possible on both. The landing used to be
    chosen on ingress room alone: it picked gpu 6 although gpu 6's uplink was already full in the
    forwarding epoch, then found it could not forward and gpu 7 (which had room) would need a relay
    the one-epoch dwell does not allow. The landing must be chosen JOINTLY with the outgoing room,
    exactly as the LP's _place_downstream_legs does with `room_limit`.
    """
    from teccl.hierarchy.reconstruct import _place_tree_route, snap_tolerance
    from teccl.topologies.dsl_topology import DslTopology

    topo = DslTopology(TopologyParams(
        name="Mini2GPU2NIC", chunk_size=1,
        topo_file="teccl/topologies/topology-dsl-frontend/examples/mini_2gpu_2nic.topo"))
    _coarse, mapping = abstract(topo)
    leaf0, leaf1, sp0 = 4, 5, 6
    f = mapping.coarse_passthrough
    assert mapping.boundary_gpu[(3, leaf1)] == [6, 7], mapping.boundary_gpu
    # identity (0, 0): cell 0 -> leaf0 -> spine -> leaf1 -> cell 3, then cell 3 -> leaf1 -> cell 2
    into3 = _leg(0, 3, leaf0, leaf1, (f[leaf0], f[sp0], f[leaf1]), 0)
    out3 = _leg(3, 2, leaf1, leaf1, (f[leaf1],), 1)
    route = _CoarseRoute(origin=(0, 2), legs=(into3, out3), volume=1.0)
    epoch = 0.16                                   # 12.5 GB/s fine uplink -> 2 chunks / epoch
    egress_ledger = {(6, leaf1, 1): 2.0}           # gpu 6's uplink is full in the forward epoch
    placed = _place_tree_route(route, (0, 0), 0, 1.0, {4, 5}, mapping, topo, epoch, {},
                               egress_ledger, tol=snap_tolerance(), placed_legs={})
    assert placed[0] == (0, 7) and placed[1][0] == 7, (
        f"must land on gpu 7, the bridging GPU with room to forward, and forward from it: {placed}")
    assert egress_ledger[(7, leaf1, 1)] == 1.0, egress_ledger
    print("  [4b] transit landing chosen jointly with the forward's uplink room (gpu 7, not 6)")


def test_occupancy_pacing():
    scale = ChunkScale(bytes_per_chunk=1.0, num_chunks=1)
    assert _piece_rate(1.0, scale, 0.01) == 100.0
    assert _piece_rate(1.0, scale, 0.01, occupancy=2) == 50.0
    # gpu 0 -> switch 1 -> gpu 2, 60 GB/s each way
    topo = SimpleNamespace(capacity=[[0, 60.0, 0], [60.0, 0, 60.0], [0, 60.0, 0]])
    # Two pieces on one 60 GB/s link, each a 2-epoch MILP leg at 50: alone fine, but the second
    # starts while the first is still on the link, so epoch 1 carries 100 > 60.
    p = dict(src_cell=0, dst_cell=1, identity=(0, 0), egress_gpu=0, ingress_gpu=2,
             via_switches=(1,), volume=1.0, rate=50.0)
    one = IdentityResolution(pieces=[ResolvedPiece(send_epoch=0, arrival_epoch=1, **p)],
                             scale=scale)
    _assert_rate_within_capacity(one, topo, 0.01)
    two = IdentityResolution(pieces=[ResolvedPiece(send_epoch=0, arrival_epoch=1, **p),
                                     ResolvedPiece(send_epoch=1, arrival_epoch=2, **p)],
                             scale=scale)
    try:
        _assert_rate_within_capacity(two, topo, 0.01)
        raise AssertionError("overlapping multi-epoch pieces must exceed the link")
    except AssertionError as e:
        assert "exceed fine link bandwidth" in str(e), e
    print("  [4] occupancy pacing: rate spread over the leg's epochs, and checked in each of them")


def test_level_policy():
    from teccl.hierarchy.solve import LevelContext, level_formulation, level_milp_chunking

    ui = UserInputParams()
    assert level_formulation(LevelContext(user_input=ui), 0) == Formulation.LP, (
        "a hierarchical level must not fall back to the flat ALLGATHER -> MILP default")
    ui.instance.formulation = [Formulation.MILP, Formulation.LP]
    ui.instance.milp_chunking = ["cell", 2]
    ctx = LevelContext(user_input=ui)
    assert level_formulation(ctx, 0) == Formulation.MILP
    assert level_formulation(ctx, 3) == Formulation.LP
    assert level_milp_chunking(ctx, 0) == "cell" and level_milp_chunking(ctx, 5) == 2
    for bad in ("bundle", 0, True):
        ui.instance.milp_chunking = bad
        try:
            resolve_milp_chunking(ui.instance)
            raise AssertionError(f"{bad!r} must be refused")
        except ValueError:
            pass
    print("  [5] hierarchical default LP; per-level formulation / chunking lists resolve by depth")


def _ui(coll, chunking):
    ui = UserInputParams()
    ui.instance = InstanceParams(
        collective=coll, formulation=Formulation.MILP, num_chunks=1,
        epoch_type=EpochType.SLOWEST_LINK, objective_type=ObjectiveType.PAPER,
        solution_method=SolutionMethod.ONE_SHOT, switch_copy=False, milp_chunking=chunking,
        schedule_output_file=os.path.join(tempfile.mkdtemp(), "coarse.json"))
    ui.gurobi.output_flag = 0
    return ui


def test_end_to_end():
    from teccl.examples.hierarchy_stitch_test import _check_ncclize
    from teccl.hierarchy.solve import solve_hierarchical

    # (topology, collective, policy, chunks per source per destination). The num_chunks=2 holder
    # case is the one whose bundle is b = 2, so it is what exercises the level RESCALE.
    for cls, coll, chunking, nc in [(BridgedIslandsCluster, Collective.ALLGATHER, "holder", 1),
                                    (BridgedIslandsSplitCluster, Collective.ALLGATHER, "holder", 1),
                                    (BridgedIslandsSplitCluster, Collective.ALLGATHER, "holder", 2),
                                    (BridgedIslandsCluster, Collective.ALLTOALL, "cell", 1),
                                    (BridgedIslandsSplitCluster, Collective.ALLTOALL, "cell", 1)]:
        topo = cls(TopologyParams(name=cls.__name__, chunk_size=1))
        n = len(topo.capacity) - len(topo.switch_indices)
        tag = f"{cls.__name__}/{coll.name.lower()}/{chunking}/{nc}chunks"
        try:
            info, sol = solve_hierarchical(topo, _ui(coll, chunking), coll,
                                           nc if coll == Collective.ALLGATHER else n * nc)
        except Exception as e:                     # gurobipy is imported lazily by the solve
            if "size-limited license" in str(e):
                print(f"  [6] SKIPPED {tag}: model too large for this Gurobi license")
                continue
            raise
        res = sol.resolution
        assert res.subdivision == 1, (tag, res.subdivision)
        relayed = [d for d in res.intra_demands
                   if d.kind == "egress_stage" and d.release_band is not None]
        if cls is BridgedIslandsSplitCluster:
            assert relayed, f"{tag}: the split bridge must forward through an NVSwitch relay"
            for d in relayed:
                fed = [p for p in res.pieces if p.identity == d.identity and p.src_cell == d.cell]
                assert all(p.send_epoch >= d.release_band + 1 for p in fed), (tag, d, fed)
        verdict = _check_ncclize(info, tag)
        assert "check_implements OK" in verdict or verdict.startswith("SKIPPED"), verdict
        print(f"  [6] {tag}: Q=1, {len(res.pieces)} pieces, {len(relayed)} transit relays, "
              f"finish {info['4-Collective_Finish_Time']:.4g}s; ncclize {verdict}")


if __name__ == "__main__":
    print("hierarchical MILP tests")
    test_count_demand()
    test_uncrossed()
    test_shared_leg_once()
    test_transit_lands_where_it_can_forward()
    test_occupancy_pacing()
    test_level_policy()
    if "--no-gurobi" not in sys.argv:
        test_end_to_end()
    print("hierarchical MILP tests OK")
