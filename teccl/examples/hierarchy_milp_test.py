"""The hierarchical solver on the MILP (teccl/hierarchy/milp_design.md).

Gurobi-free:
  [1] the count demand: every milp_chunking policy's bundle size, column count and anchor
  [2] integral identity resolution UNCROSSES the MILP's arbitrary labels (the gpu0/gpu1 case)
  [3] a column's sends are each placed once (A -> B -> C: two sends, co-located forward); Q == 1
  [4b] a transit cell lands where it can also forward
  [4c] a switch copy is ONE fine send with two branches, one gateway, one uplink charge
  [4d] a child level's switch copy stays one transfer through rebase and the stitch
  [4e] with switch copy, a cell's NVSwitch fan-out is one multicast transfer
  [4] occupancy pacing, and the rate check counting a multi-epoch send in every epoch it covers
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
from teccl.hierarchy.milp_routing import (MilpGroup, PreservingDemand, _place_column,
                                          coarsify_demand_preserving, lower_integral)
from teccl.hierarchy.reconstruct import (IdentityResolution, _assert_rate_within_capacity,
                                         snap_tolerance)
from teccl.hierarchy.scale import ChunkScale
from teccl.sends import Branch, Send, rate
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


def _send(column, src, epoch, *branches):
    """A coarse send from cell `src` in `epoch`; each branch is (dest cell, coarse switch path),
    landing the epoch after (ready = epoch + 1)."""
    return Send(identity=column, sender=src, epoch=epoch,
                fanout=tuple(Branch(dst, via, epoch + 1) for dst, via in branches))


def _crossed_fixture():
    """Host A of HeteroTaperedCluster: g0 owns the uplink to T0, g1 the one to T1, g2/g3 own none.

    Four MILP columns, all of A's allgather data, each one send that a switch copies to B and C. The
    MILP's labels make column i carry identity (i, 0) -- and they are CROSSED: column 0 leaves on T1
    (g1's uplink) while identity (0, 0) lives on g0, and column 1 leaves on T0 (g0's) while (1, 0)
    lives on g1. Read off the labels, that is two egress relays the coarse level cannot see.
    Columns 2 and 3 do the same one epoch later.
    """
    topo, _coarse, mapping = _hetero()
    A, B, C, T0, T1 = 0, 1, 2, 3, 4
    columns = {}
    for col, (nb, epoch) in enumerate([(T1, 0), (T0, 0), (T1, 1), (T0, 1)]):
        if nb == T1:
            columns[(A, col)] = [_send((A, col), A, epoch, (B, (T1, T0)), (C, (T1,)))]
        else:
            columns[(A, col)] = [_send((A, col), A, epoch, (B, (T0,)), (C, (T0,)))]
    ids = [(g, 0) for g in range(4)]
    demand = PreservingDemand(
        tensor=[], b=1, policy="fine",
        groups=[MilpGroup(cell=A, dest_cells=(B, C), identities=ids,
                          holders={d: d[0] for d in ids}, columns=[0, 1, 2, 3])])
    return topo, mapping, columns, demand


def test_uncrossed():
    topo, mapping, columns, demand = _crossed_fixture()
    fine = build_demand(Collective.ALLGATHER, topo, 1)
    res = lower_integral(columns, demand, mapping, fine, topo, epoch_duration=0.04)
    egress = {}
    for p in res.pieces:
        if p.sender in range(4):                  # host A's sends
            egress.setdefault(p.identity, set()).add((p.sender, p.epoch))
    assert egress[(0, 0)] == {(0, 0)}, f"g0's data must leave on g0's own uplink: {egress}"
    assert egress[(1, 0)] == {(1, 0)}, f"g1's data must leave on g1's own uplink: {egress}"
    # g2/g3 own no uplink, so they relay whatever they ride -- and the epoch tier puts the relayed
    # identities on the LATER columns, leaving epoch 0 to the natively-held data.
    assert all(e == 1 for d in ((2, 0), (3, 0)) for (_g, e) in egress[d]), egress
    relays = sum(1 for d, s in egress.items() for g, _e in s if g != d[0])
    assert relays == 2, (relays, egress)
    print(f"  [2] crossed MILP labels uncrossed: g0 -> T0, g1 -> T1 natively, only the 2 "
          f"uplink-less GPUs relay, and they take epoch 1")


def test_shared_leg_once():
    # One column, A -> B then B -> C (B wants AND forwards), through the bridged co-located host.
    topo = BridgedIslandsCluster(TopologyParams(name="BridgedIslandsCluster", chunk_size=1))
    _coarse, mapping = abstract(topo)
    A, B, C, T0, T1 = 0, 1, 2, 3, 4
    columns = {(A, 0): [_send((A, 0), A, 0, (B, (T0,))), _send((A, 0), B, 1, (C, (T1,)))]}
    ids = [(0, 0)]
    demand = PreservingDemand(tensor=[], b=1, policy="fine", groups=[
        MilpGroup(cell=A, dest_cells=(B, C), identities=ids, holders={(0, 0): 0}, columns=[0])])
    fine = build_demand(Collective.ALLGATHER, topo, 1)
    # Restrict the fine demand to identity (0, 0) so step B's self-distribution is about it alone.
    for s in range(len(fine)):
        if s != 0:
            fine[s] = fine[s] * 0
    res = lower_integral(columns, demand, mapping, fine, topo, epoch_duration=0.02)
    assert len(res.pieces) == 2 and res.subdivision == 1, (res.pieces, res.subdivision)
    first, second = sorted(res.pieces, key=lambda p: p.epoch)
    # the same GPU of B lands A's data and forwards it (co-located), so B needs no relay
    assert first.fanout[0].receiver == second.sender, (first, second)
    assert not [d for d in res.intra_demands if d.kind == "egress_stage" and d.cell == B]
    print("  [3] a column's sends are placed once each (2 sends, not 3), Q=1, co-located forward")


def test_transit_lands_where_it_can_forward():
    """Regression for the mini_2gpu_2nic remote failure (2026-10-12).

    Both GPUs of a mini_2gpu_2nic host hang off the SAME leaf, so either can land a chunk from
    the leaf and re-send it to the leaf -- co-location is possible on both. The landing used to be
    chosen on ingress room alone: it picked gpu 6 although gpu 6's uplink was already full in the
    forwarding epoch, then found it could not forward and gpu 7 (which had room) would need a relay
    the one-epoch dwell does not allow. The landing must be chosen JOINTLY with the outgoing room,
    exactly as the LP's _place_downstream_legs does with `room_limit`.
    """
    from teccl.topologies.dsl_topology import DslTopology

    topo = DslTopology(TopologyParams(
        name="Mini2GPU2NIC", chunk_size=1,
        topo_file="teccl/topologies/topology-dsl-frontend/examples/mini_2gpu_2nic.topo"))
    _coarse, mapping = abstract(topo)
    leaf0, leaf1, sp0 = 4, 5, 6
    assert mapping.boundary_gpu[(3, leaf1)] == [6, 7], mapping.boundary_gpu
    # identity (0, 0): cell 0 -> leaf0 -> spine -> leaf1 -> cell 3, then cell 3 -> leaf1 -> cell 2
    into3 = _send((0, 0), 0, 0, (3, (leaf0, sp0, leaf1)))
    out3 = _send((0, 0), 3, 1, (2, (leaf1,)))
    epoch = 0.16                                   # 12.5 GB/s fine uplink -> 2 chunks / epoch
    egress_ledger = {(6, leaf1, 1): 2.0}           # gpu 6's uplink is full in the forward epoch
    res = IdentityResolution()
    _place_column((0, 0), 0, [into3, out3], {into3: 0}, {((0, 0), 2): (4, 5)}, mapping, topo,
                  epoch, {}, egress_ledger, snap_tolerance(), res)
    landed = res.pieces[0].fanout[0].receiver
    assert landed == 7 and res.pieces[1].sender == 7, (
        f"must land on gpu 7, the bridging GPU with room to forward, and forward from it: "
        f"{res.pieces}")
    assert egress_ledger[(7, leaf1, 1)] == 1.0, egress_ledger
    print("  [4b] transit landing chosen jointly with the forward's uplink room (gpu 7, not 6)")


def test_switch_copy_send_charged_once():
    """Under switch copy one send reaches several cells: host B, whose single uplink g4 -> T0
    carries ONE chunk per epoch, sends each identity once and T0 copies it to A and to C. Charged
    per branch, every epoch would need two chunks on g4 -- infeasible. As one send it fits
    exactly: one gateway, one ledger entry, and one fine send with two branches.
    """
    topo, _coarse, mapping = _hetero()
    A, B, C, T0 = 0, 1, 2, 3
    columns = {(B, col): [_send((B, col), B, col, (A, (T0,)), (C, (T0,)))] for col in range(4)}
    ids = [(g, 0) for g in range(4, 8)]
    demand = PreservingDemand(tensor=[], b=1, policy="fine", groups=[
        MilpGroup(cell=B, dest_cells=(A, C), identities=ids, holders={d: d[0] for d in ids},
                  columns=[0, 1, 2, 3])])
    fine = build_demand(Collective.ALLGATHER, topo, 1)
    epoch = 0.02                                   # g4's 50 GB/s uplink -> 1 chunk per epoch
    res = lower_integral(columns, demand, mapping, fine, topo, epoch_duration=epoch)
    assert res.subdivision == 1
    for k in range(4):
        [send] = [p for p in res.pieces if p.sender == 4 and p.epoch == k]
        assert len(send.fanout) == 2, f"epoch {k}: one send from g4, branches to A and C: {send}"
    print("  [4c] switch copy: one fine send, two branches -- one gateway, charged once (g4 at 100%)")


def test_multicast_survives_the_fold():
    """A child level's switch copy is one transfer with several receivers ALL the way up: through
    pieces_as_flows (one IntraFlow, whole fan-out), re-indexing into the parent's ids, rebase, and
    the stitch -- where it becomes one record that back_trace delivers to both receivers, and that
    the link check charges once."""
    from teccl.hierarchy.flat_schedule import back_trace, build_records
    from teccl.hierarchy.flatten import rebase
    from teccl.hierarchy.problem import LevelSolution

    # child ids 0..3 are GPUs, 4 a switch; the parent sees them as 10..14
    child_send = Send((10, 0), 0, 1, (Branch(2, (4,), 2), Branch(3, (4,), 2)))
    sol = LevelSolution(pieces=[child_send], rounds_per_epoch=4)
    sub = SimpleNamespace(local_to_global={0: 10, 1: 11, 2: 12, 3: 13, 4: 14})
    [flow] = rebase(sol, cell_id=7, band=2, sub_topology=sub)
    assert flow.sender == 10 and [(b.receiver, b.via) for b in flow.fanout] == [
        (12, (14,)), (13, (14,))], flow
    assert flow.band == 2 and flow.local_round == (1 + 1) * 4, flow
    try:
        flow.receiver
        raise AssertionError("a multicast flow has no single receiver")
    except ValueError:
        pass

    res = IdentityResolution(scale=ChunkScale(bytes_per_chunk=1.0, num_chunks=1), subdivision=1)
    records, _w = build_records(res, [flow], m=16)
    [rec] = records
    assert len(rec.fanout) == 2 and {b.ready for b in rec.fanout} == {rec.epoch + rec.span}, rec
    demand = [[[0] for _ in range(15)] for _ in range(15)]
    demand[10][12][0] = demand[10][13][0] = 1
    paths = back_trace(records, demand, 1)
    assert {t for (t, _i) in paths} == {12, 13} and all(len(c) == 1 for c in paths.values())
    print("  [4d] a child's switch copy stays ONE transfer through rebase and the stitch, and "
          "back_trace delivers both receivers off it")


def test_crossbar_multicast():
    """With switch copy, a cell's NVSwitch replicates: an ingress fan-out to five GPUs is ONE
    transfer, one round, five branches -- where without it the source sends five times."""
    from teccl.hierarchy.crossbar_solve import schedule_cell
    from teccl.hierarchy.reconstruct import IntraCellDemand

    _topo, _coarse, mapping = _hetero()
    C = 2
    cell = mapping.coarse_cells[C]
    fan = IntraCellDemand(cell=C, kind="ingress_distribution", identity=(0, 0), src_gpu=8,
                          dst_gpus=(9, 10, 11, 12, 13), volume=1.0, deadline_epoch=0)
    [flow] = schedule_cell(C, cell, [fan], switch_copy=True)
    assert flow.sender == 8 and sorted(b.receiver for b in flow.fanout) == [9, 10, 11, 12, 13]
    unicast = schedule_cell(C, cell, [fan], switch_copy=False)
    assert len(unicast) == 5 and all(len(f.fanout) == 1 for f in unicast), unicast
    print("  [4e] NVSwitch copy: a 5-way fan-out is ONE transfer with five branches (5 without)")


def test_occupancy_pacing():
    assert rate(Send((0, 0), 0, 0, (Branch(1),)), 1.0, 0.01) == 100.0
    assert rate(Send((0, 0), 0, 0, (Branch(1),), span=2), 1.0, 0.01) == 50.0
    scale = ChunkScale(bytes_per_chunk=1.0, num_chunks=1)
    # gpu 0 -> switch 1 -> gpu 2, 60 GB/s each way
    topo = SimpleNamespace(capacity=[[0, 60.0, 0], [60.0, 0, 60.0], [0, 60.0, 0]])
    # Two sends on one 60 GB/s path, each a 2-epoch MILP send at 50: alone fine, but the second
    # starts while the first is still on the link, so epoch 1 carries 100 > 60.
    def send(epoch):
        return Send((0, 0), 0, epoch, (Branch(2, (1,), epoch + 2),), span=2)
    _assert_rate_within_capacity(IdentityResolution(pieces=[send(0)], scale=scale), topo, 0.01)
    try:
        _assert_rate_within_capacity(IdentityResolution(pieces=[send(0), send(1)], scale=scale),
                                     topo, 0.01)
        raise AssertionError("overlapping multi-epoch sends must exceed the link")
    except AssertionError as e:
        assert "exceed fine link bandwidth" in str(e), e
    print("  [4] occupancy pacing: rate spread over the send's epochs, and checked in each of them")


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


def _ui(coll, chunking, switch_copy=False):
    ui = UserInputParams()
    ui.instance = InstanceParams(
        collective=coll, formulation=Formulation.MILP, num_chunks=1,
        epoch_type=EpochType.SLOWEST_LINK, objective_type=ObjectiveType.LEXICOGRAPHIC,
        solution_method=SolutionMethod.ONE_SHOT, switch_copy=switch_copy, milp_chunking=chunking,
        schedule_output_file=os.path.join(tempfile.mkdtemp(), "coarse.json"))
    ui.gurobi.output_flag = 0
    return ui


def test_end_to_end():
    from teccl.examples.hierarchy_stitch_test import _check_ncclize
    from teccl.hierarchy.solve import solve_hierarchical

    # (topology, collective, policy, chunks per source per destination). The num_chunks=2 holder
    # case is the one whose bundle is b = 2, so it is what exercises the level RESCALE.
    # The last two turn switch copy ON: HeteroTaperedCluster's T0/T1 can fan one send out to
    # several hosts, which the assignment, the rate check and the stitch must charge once; the very
    # last also copies inside every cell (NVSwitch multicast), so each fan-out is one transfer.
    for cls, coll, chunking, nc, copy in [
            (BridgedIslandsCluster, Collective.ALLGATHER, "holder", 1, False),
            (BridgedIslandsSplitCluster, Collective.ALLGATHER, "holder", 1, False),
            (BridgedIslandsSplitCluster, Collective.ALLGATHER, "holder", 2, False),
            (BridgedIslandsCluster, Collective.ALLTOALL, "cell", 1, False),
            (BridgedIslandsSplitCluster, Collective.ALLTOALL, "cell", 1, False),
            (HeteroTaperedCluster, Collective.ALLGATHER, "cell", 1, [True, False]),
            (HeteroTaperedCluster, Collective.ALLGATHER, "cell", 1, True)]:
        topo = cls(TopologyParams(name=cls.__name__, chunk_size=1))
        n = len(topo.capacity) - len(topo.switch_indices)
        tag = f"{cls.__name__}/{coll.name.lower()}/{chunking}/{nc}chunks" + (
            "" if copy is False else "/copy" if copy is True else "/copy-root")
        try:
            info, sol = solve_hierarchical(topo, _ui(coll, chunking, copy), coll,
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
                fed = [p for p in res.pieces if p.identity == d.identity and p.sender in d.dst_gpus]
                assert all(p.epoch >= d.release_band + 1 for p in fed), (tag, d, fed)
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
    test_switch_copy_send_charged_once()
    test_multicast_survives_the_fold()
    test_crossbar_multicast()
    test_occupancy_pacing()
    test_level_policy()
    if "--no-gurobi" not in sys.argv:
        test_end_to_end()
    print("hierarchical MILP tests OK")
