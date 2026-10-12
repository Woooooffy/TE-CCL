"""Forwarding dwell at a collapsed cell: AllGatherFormulation.forward_dwell_constraints.

The bridged-islands pair is the minimal fixture. The only route from host A to host C is
A -> T0 -> B -> T1 -> C, through cell B, and the two variants differ in one edge:

    BridgedIslandsCluster       B's two uplinks are both on g2 -> forwarding is CO-LOCATED
    BridgedIslandsSplitCluster  B's uplinks are on g2 and g3   -> forwarding needs an NVSwitch relay

So on the coarse graph B may re-send in the epoch the chunk lands in the co-located variant, but
must hold it one full epoch in the split one -- and nothing else about the two solves differs.

[1] forward_bridges is derived from the real fine edges (Gurobi-free).
[2] co-located: B forwards in the arrival epoch.             (Gurobi, ~100 vars)
[3] split:      B forwards one epoch later, and the run is exactly one epoch longer.
[4] a cell that WANTS the chunk is not delayed by the dwell -- only its forwarding is. This is also
    the regression case for the extraction's no-copy switch rule: C's path shares B's prefix.

Run from the repo root (needs the `teccl` env for gurobipy):
    PYTHONPATH=. conda run -n teccl python teccl/examples/hierarchy_milp_forward_dwell_test.py
"""
import os
import tempfile

from teccl.hierarchy.abstract import abstract
from teccl.input_data import (Collective, EpochType, Formulation, InstanceParams, ObjectiveType,
                              SolutionMethod, TopologyParams, UserInputParams)
from teccl.topologies.bridged_islands_cluster import (BridgedIslandsCluster,
                                                       BridgedIslandsSplitCluster)


def _coarse(cls):
    fine = cls(TopologyParams(name=cls.__name__, chunk_size=1))
    coarse, mapping = abstract(fine)
    cell = {tuple(c.gpus): cid for cid, c in mapping.coarse_cells.items()}
    a, b, c = (cell[g] for g in sorted(cell))
    t0, t1 = sorted(coarse.switch_indices)
    return coarse, mapping, (a, b, c, t0, t1)


def _solve(coarse, demand_pairs, num_cells):
    n = len(coarse.capacity)
    demand = [[[0] for _ in range(n)] for _ in range(n)]
    for s, d in demand_pairs:
        demand[s][d][0] = 1
    coarse.demand_override = demand

    from teccl.scheduler import solve_on_topology
    ui = UserInputParams()
    ui.topology = TopologyParams(name=type(coarse).__name__, chunk_size=coarse.chunk_size)
    out = os.path.join(tempfile.mkdtemp(), "coarse.json")
    ui.instance = InstanceParams(
        collective=Collective.ALLGATHER, formulation=Formulation.MILP, num_chunks=1,
        epoch_type=EpochType.SLOWEST_LINK, objective_type=ObjectiveType.PAPER,
        solution_method=SolutionMethod.ONE_SHOT, switch_copy=False, num_epochs=8,
        schedule_output_file=out)
    ui.gurobi.output_flag = 0
    best = solve_on_topology(ui, coarse).best_solver
    flows, _ = best.get_schedule()
    _, demand_met, buffers, _ = best.get_flows_buffer_demand()
    # Every demand must trace back to its source. A relay that also WANTS the chunk shares its
    # path prefix with the demands it relays for; the no-copy switch rule used to forbid that
    # sharing and silently drop those demands (see dfs_remove_unnecessary_flows' claims).
    assert best.untraced_demands == [], best.untraced_demands
    assert set(best.per_demand_paths) == {(s, d, 0) for s, d in demand_pairs}, best.per_demand_paths
    return best, flows, buffers, demand_met


def _first(flows, s, i, j):
    ks = [k for (fs, fi, fj, _c, k) in flows if (fs, fi, fj) == (s, i, j)]
    assert ks, f"no flow {s}: {i} -> {j} in {sorted(flows)}"
    return min(ks)


def test_bridges_from_fine_edges():
    co, _, (a, b, c, t0, t1) = _coarse(BridgedIslandsCluster)
    sp, _, (a2, b2, c2, t02, t12) = _coarse(BridgedIslandsSplitCluster)
    # co-located: g2 lands from both islands and sends to both -> every pair bridges
    assert co.forward_bridges[(b, t1)] == frozenset({t0, t1}), co.forward_bridges
    assert co.forward_bridges[(b, t0)] == frozenset({t0, t1}), co.forward_bridges
    # split: g2 owns island 0 only, g3 island 1 only -> only same-island re-send bridges
    assert sp.forward_bridges[(b2, t12)] == frozenset({t12}), sp.forward_bridges
    assert sp.forward_bridges[(b2, t02)] == frozenset({t02}), sp.forward_bridges
    print("  [OK] [1] forward_bridges read off the fine edges")


def test_colocated_forwards_in_arrival_epoch():
    coarse, _, (a, b, c, t0, t1) = _coarse(BridgedIslandsCluster)
    best, flows, buffers, _ = _solve(coarse, [(a, c)], 3)
    landed = buffers[(a, b, 0)]          # first epoch B holds A's chunk
    sent = _first(flows, a, b, t1)
    assert sent == landed, f"co-located B should forward in its arrival epoch {landed}, sent {sent}"
    print(f"  [OK] [2] co-located: lands at B by epoch {landed}, forwarded in epoch {sent}")
    return best.find_demand_satisfied_k()


def test_split_waits_one_epoch(colocated_finish):
    coarse, _, (a, b, c, t0, t1) = _coarse(BridgedIslandsSplitCluster)
    best, flows, buffers, _ = _solve(coarse, [(a, c)], 3)
    landed = buffers[(a, b, 0)]
    sent = _first(flows, a, b, t1)
    assert sent == landed + 1, (
        f"split B must hold the chunk one epoch for the g2 -> g3 relay: landed {landed}, sent {sent}")
    finish = best.find_demand_satisfied_k()
    assert finish == colocated_finish + 1, (finish, colocated_finish)
    print(f"  [OK] [3] split: lands at B by epoch {landed}, forwarded in epoch {sent}; "
          f"finish {finish} = co-located {colocated_finish} + 1")


def test_consumption_not_delayed():
    # A's chunk is wanted by B AND C. B's own delivery must be as early as without the dwell;
    # only the onward leg to C waits.
    co, _, (a, b, c, *_r) = _coarse(BridgedIslandsCluster)
    _, _, _, met_co = _solve(co, [(a, b), (a, c)], 3)
    sp, _, (a2, b2, c2, *_r2) = _coarse(BridgedIslandsSplitCluster)
    _, _, _, met_sp = _solve(sp, [(a2, b2), (a2, c2)], 3)
    assert met_sp[(a2, b2, 0)] == met_co[(a, b, 0)], (met_sp, met_co)
    assert met_sp[(a2, c2, 0)] == met_co[(a, c, 0)] + 1, (met_sp, met_co)
    print(f"  [OK] [4] B's own delivery unchanged (epoch {met_sp[(a2, b2, 0)]}); "
          f"C's delayed by one ({met_co[(a, c, 0)]} -> {met_sp[(a2, c2, 0)]})")


if __name__ == "__main__":
    test_bridges_from_fine_edges()
    fin = test_colocated_forwards_in_arrival_epoch()
    test_split_waits_one_epoch(fin)
    test_consumption_not_delayed()
    print("all forward-dwell tests passed")
