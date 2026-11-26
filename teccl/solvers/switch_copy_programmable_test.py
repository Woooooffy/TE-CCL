"""Switch copy only at PROGRAMMABLE switches (AllGatherFormulation.copy_switches, solve.cell_switch_copy).

A copy is a forwarding decision: the switch must be told to replicate a flow to several ports. Only
the switches ncclize writes a forwarding program for (Topology.programmable_switch_indices) can be,
so only they may copy; a self-routing switch -- a host's PCIe root complex, an intra-host NVSwitch
on a topology that declares it so -- forwards one copy per copy received. Before this, switch_copy
let the MILP copy at EVERY switch, and the flat mini_2gpu_1nic schedule multicast through a root
complex, which the switch table then refused ("branches at switch 9, which takes no forwarding
program").

  [1] which switches may copy: the programmable set when copy is on, none when it is off, every
      switch for a topology that does not restrict it
  [2] the model (built, never optimized): a self-routing switch gets the conservation row, a
      programmable one the copy rows; a link INTO a switch that cannot copy may carry several
      same-class copies, every other link one
  [3] the trace: one send into a non-programmable switch feeds ONE of its two out-flows; the same
      flows through a programmable switch feed both
  [4] the crossbar row: a cell whose NVSwitch is self-routing does not multicast, and says so

Gurobi-free apart from model construction (no optimize, so the size-limited license is enough).
Run from the repo root:
    PYTHONPATH=. python teccl/solvers/switch_copy_programmable_test.py
"""
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from teccl.input_data import (Collective, EpochType, Formulation, InstanceParams, ObjectiveType,
                              SolutionMethod, TopologyParams, UserInputParams)
from teccl.topologies.dsl_topology import DslTopology

logging.disable(logging.WARNING)

MINI_2G1N = "teccl/topologies/topology-dsl-frontend/examples/mini_2gpu_1nic.topo"


def _ui(switch_copy, num_epochs=3):
    ui = UserInputParams()
    ui.instance = InstanceParams(
        collective=Collective.ALLGATHER, formulation=Formulation.MILP, num_chunks=1,
        epoch_type=EpochType.SLOWEST_LINK, objective_type=ObjectiveType.PAPER,
        solution_method=SolutionMethod.ONE_SHOT, switch_copy=switch_copy, num_epochs=num_epochs,
        schedule_output_file=os.path.join(tempfile.mkdtemp(), "s.json"))
    ui.gurobi.output_flag = 0
    return ui


def _mini():
    return DslTopology(TopologyParams(name="Mini2GPU1NIC", chunk_size=1, topo_file=MINI_2G1N))


def test_copy_switches():
    from teccl.solvers.allgather import AllGatherFormulation
    from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster

    topo = _mini()
    network, pcie = set(topo.programmable_switch_indices), set(topo.switch_indices)
    assert network and network < pcie, (network, pcie)
    f = AllGatherFormulation(_ui(True), topo)
    assert f.copy_switches == network, f.copy_switches
    assert not any(f.copies_at(rc) for rc in pcie - network)
    assert AllGatherFormulation(_ui(False), topo).copy_switches == frozenset()
    # A topology that does not restrict programmability keeps today's behaviour: every switch.
    het = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    assert AllGatherFormulation(_ui(True), het).copy_switches == set(het.switch_indices)
    print(f"  [1] copy only at programmable switches: {sorted(network)} copy, root complexes "
          f"{sorted(pcie - network)} do not; none with copy off; all on an unrestricted topology")


def test_model_structure():
    import gurobipy as gp
    from teccl.gurobi_env import get_gurobi_env
    from teccl.solvers.allgather import AllGatherFormulation

    topo = _mini()
    f = AllGatherFormulation(_ui(True), topo)
    f.model = gp.Model("structure", env=get_gurobi_env())
    f.initialize_variables()
    f.destination_constraints()
    f.node_constraints([])
    f.model.update()
    names = [c.ConstrName for c in f.model.getConstrs() if c.ConstrName.startswith("switch_flow_")]

    def rows(sw):
        """'copy' if every switch_flow row of `sw` is a per-out-link copy row, 'conserve' if every
        one is a single conservation row."""
        mine = [n for n in names if f"_i_{sw}_" in n]
        assert mine, f"switch {sw} got no switch rows"
        # copy rows end in "_<out-link index>" after the class tag; conservation rows end at the
        # tag (..._k_<k>_b_<b>)
        kinds = {"copy" if n.rsplit("_", 2)[-2] != "b" else "conserve" for n in mine}
        assert len(kinds) == 1, (sw, sorted(mine)[:6])
        return kinds.pop()

    network = set(topo.programmable_switch_indices)
    for sw in topo.switch_indices:
        want = "copy" if sw in network else "conserve"
        assert rows(sw) == want, f"switch {sw}: {rows(sw)} rows, expected {want}"

    # Copy bound per link. GPU -> its root complex: several same-class copies (the root complex
    # cannot replicate, so a chunk wanted in both a far host and the GPU's own leaf must leave
    # twice). Root complex -> leaf, and leaf -> anything: one (a copying switch, or a GPU).
    gpu, rc = next((i, j) for i in f.nodes for j in sorted(set(topo.switch_indices) - network)
                   if i not in topo.switch_indices and topo.capacity[i][j] > 0)
    leaf = next(j for j in network if topo.capacity[rc][j] > 0)
    assert f.class_copies(gpu, 0, rc) == f.max_copies(gpu, 0) > 1
    assert f.class_copies(gpu, 0, leaf) == 1
    other_gpu = next(j for j in f.nodes if j not in topo.switch_indices and j != gpu)
    assert f.class_copies(gpu, 0, other_gpu) == 1
    print(f"  [2] model: conservation rows at root complexes, copy rows at {sorted(network)}; a "
          f"link into root complex {rc} may carry {f.max_copies(gpu, 0)} copies, one into a leaf 1")


def test_trace():
    from milp_trace_test import FANOUT_BUF, FANOUT_FLOWS, FANOUT_MET, _hetero_coarse, _replay

    # The fan-out fixture: B -> T0 (3) once, then T0 -> C and T0 -> T1 (4) -> A in the same epoch.
    coarse = _hetero_coarse()
    coarse.programmable_switch_indices = [3, 4]
    both = _replay(coarse, True, FANOUT_FLOWS, FANOUT_MET, FANOUT_BUF)
    assert both.untraced_demands == [] and len(both.per_demand_paths) == 2, both.untraced_demands

    coarse = _hetero_coarse()
    coarse.programmable_switch_indices = [4]          # T0 self-routing: it cannot replicate
    one = _replay(coarse, True, FANOUT_FLOWS, FANOUT_MET, FANOUT_BUF)
    assert len(one.per_demand_paths) == 1 and len(one.untraced_demands) == 1, (
        f"a non-programmable T0 forwards ONE copy per copy received: {one.per_demand_paths}, "
        f"untraced {one.untraced_demands}")
    assert all(len(s.fanout) == 1 for s in one.sends), one.sends
    print("  [3] trace: through programmable T0 both demands share B's one send; with T0 "
          "self-routing only one is traced, and no send branches")


def test_crossbar_cell():
    from teccl.hierarchy.abstract import abstract
    from teccl.hierarchy.solve import LevelContext, _CellView, cell_switch_copy
    from teccl.hierarchy.problem import Subproblem
    from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster

    ui = UserInputParams()
    ui.instance.switch_copy = [True, True]          # ask for copy inside the cells too
    fine = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    _coarse, mapping = abstract(fine)
    cid, cell = next(iter(sorted(mapping.coarse_cells.items())))

    def problem():
        return Subproblem(topology=_CellView(fine, cell), demands=[], scale=None, depth=1,
                          cell_id=cid)

    # The class declares its NVSwitches programmable (Topology default), so the cell may copy.
    assert cell_switch_copy(LevelContext(user_input=ui), problem())
    # Declared self-routing, it may not -- and the drop is reported once.
    fine.programmable_switch_indices = [s for s in fine.programmable_switch_indices
                                        if s not in set(cell.internal_switches)]
    ctx = LevelContext(user_input=ui)
    logging.disable(logging.NOTSET)
    try:
        assert not cell_switch_copy(ctx, problem())
    finally:
        logging.disable(logging.WARNING)
    assert getattr(ctx, "warned_switch_copy", False)
    print("  [4] crossbar: a cell with a programmable NVSwitch multicasts; a self-routing one "
          "does not, with one warning")


if __name__ == "__main__":
    print("switch copy at programmable switches only")
    test_copy_switches()
    test_model_structure()
    test_trace()
    test_crossbar_cell()
    print("switch copy programmable tests OK")
