"""
Coarse solve of the IRREGULAR HeteroTaperedCluster (3 heterogeneous hosts A:4 / B:4 / C:6 GPUs,
tapered uplinks, single-homed Host B, multi-GPU boundary on Host C) -- the topology built to
exercise the phase-3 machinery the symmetric rail topology never triggers.

Unlike the rail driver, the coarse AllGather here is NON-UNIFORM: cells have different GPU
counts, so the coarse demand is the heterogeneous 4/4/6 volume matrix produced by
coarsify_demand (collective-agnostic) rather than a uniform num_chunks-per-source generator.
The coarse demand is injected via topology.demand_override, so the coarse solve satisfies the
aggregated inter-cell volumes directly. AllToAll is also supported (coarse volume |U|*|V|).

Runs the FULL hierarchical pipeline: coarse solve -> identity resolution -> phase-3 intra-cell
schedule -> phase-4 stitch, writing coarse_hetero_{tag}_{lp,identities,intra,flat}.json. The
_flat.json is a normal flat schedule on the FINE topology and feeds teccl/ncclize/teccl_ncclize.py
unchanged. Requires Gurobi for the coarse solve, so it runs on the remote solver host; everything
below the coarse solve is Gurobi-free and is replayed locally by hierarchy_stitch_test.py.

Everything below the coarse solve is topology-independent and lives in hierarchy_pipeline.

Run from the repo root:
    python -m teccl.examples.hierarchy_coarse_solve_hetero [allgather|alltoall] [lp|milp|both] [nocoarsen]
"""
import sys
import traceback

from teccl.examples.hierarchy_pipeline import (
    print_solve_summary, run_identity_resolution, run_phase3_intra, run_stitch,
    solve_on_topology,
)
from teccl.hierarchy.abstract import abstract, coarsify_demand, lift_demand, set_level_chunk
from teccl.input_data import (
    Collective, EpochType, Formulation, InstanceParams, ObjectiveType,
    SolutionMethod, TopologyParams, UserInputParams,
)
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster


def _make_input(formulation: Formulation, collective: Collective, out_file: str) -> UserInputParams:
    ui = UserInputParams()
    ui.topology = TopologyParams(name="HeteroTaperedCluster_coarse", chunk_size=1)
    ui.instance = InstanceParams(
        collective=collective,
        formulation=formulation,
        # num_chunks is irrelevant for the coarse solve: the demand is injected via
        # demand_override (a single weighted slot), so BaseFormulation sets num_chunks == 1.
        num_chunks=1,
        # Heterogeneous BW tiers (50 / 100 / 200). Size the epoch to the slow uplink so the
        # model stays small (see hierarchy_coarse_solve for the rationale); epoch size is only a
        # correctness-preserving granularity dial.
        epoch_type=EpochType.SLOWEST_LINK,
        epoch_multiplier=1,
        objective_type=ObjectiveType.PAPER,
        solution_method=SolutionMethod.ONE_SHOT,
        # The coarse graph has NO non-trivial automorphism (asymmetric, heterogeneous cells), so
        # abstract() emits no equivalent_node_indices -- nothing for symmetry enforcement to do.
        symmetry=False,
        # LP AllGather/AllToAll are copy-free (switch_copy=False); the MILP uses multicast copy.
        switch_copy=(formulation == Formulation.MILP),
        schedule_output_file=out_file,
    )
    return ui


def main() -> None:
    coll_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "allgather"
    which = sys.argv[2].lower() if len(sys.argv) > 2 else "lp"
    no_coarsen = "nocoarsen" in [a.lower() for a in sys.argv[3:]]
    collective = Collective.ALLGATHER if coll_arg == "allgather" else Collective.ALLTOALL

    fine = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    coarse, mapping = abstract(fine)
    lift_demand(mapping)  # heterogeneous: per-cell GPU-count chunk identities

    # Fine demand -> coarse demand (collective-agnostic aggregation). The fine demand MUST be
    # built at the same effective resolution the flat ground-truth solve uses, or the coarse
    # volumes won't correspond to the flat problem and the comparison is meaningless.
    #
    # CHUNKS_PER_PAIR is the flat input num_chunks (per source, per destination). AllGather
    # passes it through unscaled, so fine_chunks == CHUNKS_PER_PAIR. AllToAll is DIFFERENT: the
    # scheduler scales the flat input by the participating-GPU count (scheduler.get_solver,
    # ALLTOALL branch: num_chunks *= num_gpus), and build_demand then lays down
    # fine_chunks // num_gpus chunks per ordered pair -- so to reproduce the flat alltoall we
    # must pre-scale here to CHUNKS_PER_PAIR * num_gpus (build_demand is called DIRECTLY, it does
    # not go through the scheduler's scaling). Keep CHUNKS_PER_PAIR in lockstep with the flat
    # input JSON (hetero_alltoall_lp.json num_chunks).
    CHUNKS_PER_PAIR = 1
    num_participating = sum(len(c.gpus) for c in mapping.coarse_cells.values())
    if collective == Collective.ALLGATHER:
        fine_chunks = CHUNKS_PER_PAIR
    else:
        fine_chunks = CHUNKS_PER_PAIR * num_participating
    fine_demand = build_demand(collective, fine, fine_chunks)
    coarse_demand = coarsify_demand(fine_demand, mapping)

    # Put the coarse level into its own chunk unit (abstract.set_level_chunk): the GCD of the
    # coarse volumes. For these 4/4/6 cells that is 2 -- NOT any single cell's payload, which is
    # exactly why the rule has to be the common divisor rather than a largest/smallest choice.
    # `nocoarsen` forces g=1 to reproduce the pre-coarsening behaviour.
    coarse_demand, g, level_scale = set_level_chunk(coarse, coarse_demand,
                                                   g=1 if no_coarsen else None)
    coarse.demand_override = coarse_demand

    vols = {(u, v): coarse_demand[u][v][0]
            for u in range(mapping.num_coarse) for v in range(mapping.num_coarse)
            if coarse_demand[u][v][0]}
    print(f"coarse topology: {mapping.num_coarse} nodes "
          f"({len(mapping.coarse_cells)} cells + {len(coarse.switch_indices)} switches), "
          f"collective={coll_arg}, which={which}")
    print(f"level chunk: g={g} fine chunks -> {level_scale}, coarse epoch "
          f"{coarse.get_epoch_duration_slow_link()}s (SLOWEST_LINK)")
    print(f"coarse demand volumes (U->V): {vols}")

    tag = coll_arg
    prefix = f"coarse_hetero_{tag}"
    milp_out = f"Schedules/{prefix}_milp.json"
    lp_out = f"Schedules/{prefix}_lp.json"

    if which in ("both", "milp"):
        print("\n=== MILP (switch_copy=True, multicast) ===")
        try:
            solve_on_topology(_make_input(Formulation.MILP, collective, milp_out), coarse)
        except Exception as e:
            print(f"MILP solve failed: {type(e).__name__}: {e}")
    if which in ("both", "lp"):
        print("\n=== LP (switch_copy=False, unicast) ===")
        try:
            lp_solver = solve_on_topology(_make_input(Formulation.LP, collective, lp_out), coarse)
            # The coarse epoch is the coarse solve's own epoch duration -- every downstream
            # quantity (m, the staging deadlines, the network pacing rate) is derived from it, so
            # it must be read off the solved formulation rather than restated.
            coarse_epoch = lp_solver.best_solver.epoch_duration
            res = run_identity_resolution(lp_solver, mapping, fine_demand, fine, coarse_epoch,
                                          prefix, level_chunk=g)
            intra_flows = run_phase3_intra(res, mapping, prefix, fine, coarse_epoch)
            run_stitch(res, intra_flows, fine, fine_demand, coarse_epoch, tag, prefix)
        except Exception as e:
            print(f"LP solve / hierarchical reconstruction failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    print_solve_summary((("MILP", milp_out), ("LP", lp_out)))


if __name__ == "__main__":
    main()
