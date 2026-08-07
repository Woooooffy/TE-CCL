"""
Full hierarchical pipeline on the SYMMETRIC RailOptimizedSpineLeaf (32 hosts x 8 GPUs = 256 GPUs
behind 8 leaf + 4 spine switches; 300 fine nodes -> 44 coarse nodes).

This is the production-shaped counterpart to hierarchy_coarse_solve_hetero: same four phases
(coarse solve -> identity resolution -> phase-3 intra-cell NVSwitch schedule -> phase-4 stitch),
same shared tail (teccl.examples.hierarchy_pipeline), different topology. Outputs
Schedules/coarse_rail_{tag}_{lp,milp,identities,intra,flat}.json, where _flat.json is an ordinary
FINE-topology schedule that feeds teccl/ncclize/teccl_ncclize.py with no hierarchy awareness.

What is different from the hetero driver, and why:

  * SCALE. 44 coarse nodes vs 8, and 256 fine GPUs vs 14. The coarse LP is the only Gurobi step
    and is the one that can blow up; everything below it is combinatorial and cheap. Phase-3
    debug narration is OFF by default here (32 structurally identical cells would each print a
    full derivation and bury the log) -- pass `debug` as the 3rd arg to turn it back on.

  * SYMMETRY IS ON. abstract() forwards the declared spine twins AND detects the emergent leaf
    twins (all 8 leaves have an identical coarse neighborhood once the hosts collapse). Without
    it the coarse rail LP is badly degenerate and lands on an arbitrary interior point with
    fractional flow smeared over every leaf/spine -- see the degeneracy notes. Only relay
    switches are grouped: the 32 hosts share a fingerprint too, but a host swap permutes the
    source index, so it is not a valid relay-twin symmetry.

  * The coarse demand still comes from the generic collective-agnostic path (build_demand on the
    fine topology -> coarsify_demand -> demand_override), not from a uniform per-cell generator.
    The rail cells happen to be uniform (8 GPUs each), so this reduces to the uniform lift; going
    through coarsify_demand anyway keeps one code path with the hetero driver.

KNOWN FAILURE MODE (as of this writing): the coarse rail AllGather LP *solves*, but flow
extraction can trip `assert consume == 0` in lp_formulation.account_for_consume on Gurobi
feasibility-tolerance noise -- flows of ~1e-9 or slightly negative that get_flows_and_consumes
admits because it filters on `v.x != 0.0` (lp_formulation.py:601) instead of clamping at
gurobi.feasibility_tol. If the run dies there, that is the bug, not the hierarchy: the fix is a
clamp in get_flows_and_consumes, deliberately not applied here.

Run from the repo root (needs Gurobi for the coarse solve):
    python -m teccl.examples.hierarchy_coarse_solve_rail [allgather|alltoall] [lp|milp|both] [debug]
"""
import sys
import traceback

from teccl.examples.hierarchy_pipeline import (
    print_solve_summary, run_identity_resolution, run_phase3_intra, run_stitch,
    solve_on_topology,
)
from teccl.hierarchy.abstract import abstract, coarsify_demand, lift_demand
from teccl.input_data import (
    Collective, EpochType, Formulation, InstanceParams, ObjectiveType,
    SolutionMethod, TopologyParams, UserInputParams,
)
from teccl.solvers.demand import build_demand
from teccl.topologies.rail_optimized_spine_leaf import RailOptimizedSpineLeaf

# Flat input num_chunks, per source per destination. Keep in lockstep with any flat rail input
# JSON this run is compared against -- the coarse volumes only correspond to the flat problem if
# the fine demand is built at the same resolution.
CHUNKS_PER_PAIR = 1


def _make_input(formulation: Formulation, collective: Collective, out_file: str,
                epoch_multiplier: int) -> UserInputParams:
    ui = UserInputParams()
    ui.topology = TopologyParams(name="RailOptimizedSpineLeaf_coarse", chunk_size=1)
    ui.instance = InstanceParams(
        collective=collective,
        formulation=formulation,
        # num_chunks is irrelevant for the coarse solve: the demand is injected via
        # demand_override (a single weighted slot), so BaseFormulation sets num_chunks == 1.
        num_chunks=1,
        # The coarse topology mixes two bandwidth tiers -- the 50 GB/s rail uplinks and the
        # 400 GB/s leaf-spine mesh (an 8x spread). FASTEST_LINK would pin the epoch to the fast
        # link (0.0025 s), so the slow-link-bottlenecked collective would need ~8x MORE epochs,
        # bloating the model in its largest dimension for nothing. The bottleneck is the rail
        # uplink, so size the epoch to it: SLOWEST_LINK (0.02 s). Epoch size is only a
        # granularity dial for correctness, so this is purely about model size.
        epoch_type=EpochType.SLOWEST_LINK,
        epoch_multiplier=epoch_multiplier,
        objective_type=ObjectiveType.PAPER,
        solution_method=SolutionMethod.ONE_SHOT,
        # Relay-twin symmetry: the declared spine twins plus the leaf twins abstract() detects
        # after collapsing. This pins the solve onto the symmetric barycenter instead of an
        # arbitrary degenerate interior point. Consumed by both the MILP (aggregate) and the LP
        # (per-source) symmetry constraints.
        symmetry=True,
        # LP AllGather/AllToAll are copy-free (switch_copy=False); the MILP uses multicast copy.
        switch_copy=(formulation == Formulation.MILP),
        schedule_output_file=out_file,
    )
    return ui


def main() -> None:
    coll_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "allgather"
    which = sys.argv[2].lower() if len(sys.argv) > 2 else "lp"
    debug_intra = len(sys.argv) > 3 and sys.argv[3].lower() in ("debug", "true", "1")
    collective = Collective.ALLGATHER if coll_arg == "allgather" else Collective.ALLTOALL
    epoch_multiplier = 1

    fine = RailOptimizedSpineLeaf(TopologyParams(name="RailOptimizedSpineLeaf", chunk_size=1))
    coarse, mapping = abstract(fine)
    # Uniform cells (8 GPUs each), so the default per-cell-GPU-count lift gives each host its 8
    # sub-chunk identities -- the real host-level AllGather lift.
    lift_demand(mapping)

    # AllGather passes CHUNKS_PER_PAIR through unscaled. AllToAll is different: the scheduler
    # scales the flat input by the participating-GPU count (scheduler.get_solver, ALLTOALL branch:
    # num_chunks *= num_gpus) and build_demand then lays down fine_chunks // num_gpus chunks per
    # ordered pair. build_demand is called DIRECTLY here, bypassing that scaling, so pre-scale.
    num_participating = sum(len(c.gpus) for c in mapping.coarse_cells.values())
    fine_chunks = (CHUNKS_PER_PAIR if collective == Collective.ALLGATHER
                   else CHUNKS_PER_PAIR * num_participating)
    fine_demand = build_demand(collective, fine, fine_chunks)
    coarse_demand = coarsify_demand(fine_demand, mapping)
    coarse.demand_override = coarse_demand

    vols = sorted({coarse_demand[u][v][0]
                   for u in range(mapping.num_coarse) for v in range(mapping.num_coarse)
                   if coarse_demand[u][v][0]})
    nonzero = sum(1 for u in range(mapping.num_coarse) for v in range(mapping.num_coarse)
                  if coarse_demand[u][v][0])
    print(f"coarse topology: {mapping.num_coarse} nodes "
          f"({len(mapping.coarse_cells)} cells x {num_participating // len(mapping.coarse_cells)} "
          f"gpus + {len(coarse.switch_indices)} switches), collective={coll_arg}, which={which}")
    print(f"twin groups (symmetry): {coarse.equivalent_node_indices}")
    # Uniform by construction, so the distinct-volume set should be a single value; printing the
    # set rather than the full 44x44 matrix keeps the log readable and still catches asymmetry.
    print(f"coarse demand: {nonzero} nonzero (U->V) pairs, distinct volumes {vols}")

    tag = coll_arg
    prefix = f"coarse_rail_{tag}"
    milp_out = f"Schedules/{prefix}_milp.json"
    lp_out = f"Schedules/{prefix}_lp.json"

    if which in ("both", "milp"):
        print("\n=== MILP (switch_copy=True, multicast) ===")
        try:
            solve_on_topology(_make_input(Formulation.MILP, collective, milp_out,
                                          epoch_multiplier), coarse)
        except Exception as e:
            print(f"MILP solve failed: {type(e).__name__}: {e}")
            traceback.print_exc()
    if which in ("both", "lp"):
        print("\n=== LP (switch_copy=False, unicast) ===")
        try:
            lp_solver = solve_on_topology(_make_input(Formulation.LP, collective, lp_out,
                                                      epoch_multiplier), coarse)
            # The coarse epoch is the coarse solve's OWN epoch duration -- m, the staging
            # deadlines and the network pacing rate all derive from it, so read it off the solved
            # formulation rather than restating 0.02 here.
            coarse_epoch = lp_solver.best_solver.epoch_duration
            res = run_identity_resolution(lp_solver, mapping, fine_demand, fine, coarse_epoch,
                                          prefix)
            intra_flows = run_phase3_intra(res, mapping, prefix, fine, coarse_epoch,
                                           debug=debug_intra)
            run_stitch(res, intra_flows, fine, fine_demand, coarse_epoch, tag, prefix)
        except Exception as e:
            print(f"LP solve / hierarchical reconstruction failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    print_solve_summary((("MILP", milp_out), ("LP", lp_out)))


if __name__ == "__main__":
    main()
