"""
Step 3 (go/no-go on the hierarchical premise): run BOTH the existing MILP and the copy-free
LP AllGather solvers on the *coarse* 44-node rail-optimized topology produced by abstract(),
and compare epochs / finish-time / bandwidth. No stitching yet -- this only checks that the
coarse problem is small enough to solve quickly and that the two formulations behave as the
design predicts (MILP with multicast copy vs. LP unicast-per-destination).

Requires Gurobi, so this runs on the remote solver host, not locally.

Run from the repo root:
    python -m teccl.examples.hierarchy_coarse_solve [num_sub_chunks]

num_sub_chunks defaults to GPUS_PER_NODE (8) -- the real host-level AllGather lift (each of a
host's 8 GPUs' data is a distinct coarse sub-chunk). Pass a smaller value (e.g. 1 or 2) for a
faster smoke test of the coarse solve mechanics.
"""
import copy
import json
import sys

from teccl.hierarchy.abstract import abstract, lift_demand
from teccl.input_data import (
    Collective, EpochType, Formulation, InstanceParams, ObjectiveType,
    SolutionMethod, TopologyParams, UserInputParams,
)
from teccl.scheduler import TECCLSolver
from teccl.topologies.rail_optimized_spine_leaf import RailOptimizedSpineLeaf
from teccl.topologies.topology import Topology


def _solve_on_topology(user_input: UserInputParams, topology: Topology) -> None:
    """Run TECCLSolver.solve() against an already-built Topology object, bypassing
    get_topology() (which only knows the named built-in topologies, not CoarseTopology)."""
    solver = TECCLSolver.__new__(TECCLSolver)
    solver.user_input = user_input
    solver.topology_obj = topology
    solver.solver = solver.get_solver(copy.deepcopy(user_input), topology)
    solver.solve()


def _make_input(formulation: Formulation, num_sub_chunks: int, out_file: str,
                epoch_multiplier: int = 1) -> UserInputParams:
    ui = UserInputParams()
    ui.topology = TopologyParams(name="RailOptimizedSpineLeaf_coarse", chunk_size=1)
    ui.instance = InstanceParams(
        collective=Collective.ALLGATHER,
        formulation=formulation,
        num_chunks=num_sub_chunks,
        # The coarse topology mixes two bandwidth tiers -- the 50 GB/s rail uplinks and the
        # 400 GB/s leaf-spine mesh (an 8x spread). FASTEST_LINK would pin the epoch to the
        # fast link (0.0025), so the slow-link-bottlenecked collective would need ~8x MORE
        # epochs (37 vs ~5), bloating the MILP ~8x in its largest dimension and OOM-ing the
        # branch-and-bound tree. The bottleneck is the slow rail uplink, so size the epoch to
        # it: SLOWEST_LINK (0.02). Epoch size is only a granularity dial for correctness
        # (see the epoch_duration_fastest_link memory note); this just right-sizes the model.
        epoch_type=EpochType.SLOWEST_LINK,
        epoch_multiplier=epoch_multiplier,
        objective_type=ObjectiveType.PAPER,
        solution_method=SolutionMethod.ONE_SHOT,
        # Enforce relay-twin symmetry (the emergent leaf twins + declared spine twins that
        # abstract() now puts in equivalent_node_indices). This pins the coarse solve onto
        # the symmetric barycenter optimum instead of an arbitrary degenerate interior point.
        # Consumed by both the MILP (aggregate) and LP (per-source) symmetry constraints.
        symmetry=True,
        # LP AllGather is copy-free and requires switch_copy=False (scheduler enforces this);
        # the MILP uses multicast copy through leaves/spines.
        switch_copy=(formulation == Formulation.MILP),
        schedule_output_file=out_file,
    )
    return ui


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    # argv: [num_sub_chunks] [epoch_multiplier] [which:milp|lp|both]
    num_sub_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else RailOptimizedSpineLeaf.GPUS_PER_NODE
    epoch_multiplier = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    which = sys.argv[3].lower() if len(sys.argv) > 3 else "both"

    fine = RailOptimizedSpineLeaf(TopologyParams(name="RailOptimizedSpineLeaf", chunk_size=1))
    coarse, mapping = abstract(fine)
    lift_demand(mapping, num_sub_chunks)
    print(f"coarse topology: {mapping.num_coarse} nodes "
          f"({len(mapping.coarse_cells)} hosts + {len(coarse.switch_indices)} switches), "
          f"num_sub_chunks={num_sub_chunks}, epoch_multiplier={epoch_multiplier}, which={which}")

    milp_out = f"Schedules/coarse_rail_milp_{num_sub_chunks}chunks.json"
    lp_out = f"Schedules/coarse_rail_lp_{num_sub_chunks}chunks.json"

    # Run each formulation in its own try-block so an OOM/failure in one (the MILP is the
    # memory-heavy one) doesn't prevent the other from producing a result. To fully isolate
    # memory, run them as separate jobs: `... <chunks> <mult> milp` and `... <chunks> <mult> lp`.
    if which in ("both", "milp"):
        print("\n=== MILP (switch_copy=True, multicast) ===")
        try:
            _solve_on_topology(_make_input(Formulation.MILP, num_sub_chunks, milp_out, epoch_multiplier), coarse)
        except Exception as e:
            print(f"MILP solve failed: {type(e).__name__}: {e}")
    if which in ("both", "lp"):
        print("\n=== LP (switch_copy=False, unicast) ===")
        try:
            _solve_on_topology(_make_input(Formulation.LP, num_sub_chunks, lp_out, epoch_multiplier), coarse)
        except Exception as e:
            print(f"LP solve failed: {type(e).__name__}: {e}")

    try:
        milp, lp = _load(milp_out), _load(lp_out)
    except FileNotFoundError:
        print("\n(one or both schedules missing; skipping comparison table)")
        return
    keys = ["3-Epochs_Required", "4-Collective_Finish_Time", "5-Algo_Bandwidth", "Solver_Time"]
    print("\n" + "=" * 62)
    print(f"{'Metric':<30}{'MILP':>15}{'LP (no-copy)':>17}")
    print("-" * 62)
    for k in keys:
        print(f"{k:<30}{str(milp.get(k)):>15}{str(lp.get(k)):>17}")
    print("=" * 62)
    mt, lt = milp.get("4-Collective_Finish_Time"), lp.get("4-Collective_Finish_Time")
    if mt and lt:
        print(f"LP finish / MILP finish = {lt / mt:.3f} (LP has no copy, expect >= 1.0)")


if __name__ == "__main__":
    main()
