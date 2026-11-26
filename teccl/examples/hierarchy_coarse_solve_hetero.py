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

from teccl.examples.hierarchy_pipeline import make_reporter, print_solve_summary
from teccl.hierarchy.abstract import abstract, lift_demand
from teccl.hierarchy.solve import solve_hierarchical
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
    # abstract() only to size the demand; solve_hierarchical redoes it internally as level 0. The
    # per-cell GPU-count chunk identities lift_demand fills are a property of the topology, so
    # doing it here is harmless and keeps the participant count available for the alltoall scaling.
    _coarse, mapping = abstract(fine)
    lift_demand(mapping)

    # The fine demand MUST be built at the same effective resolution the flat ground-truth solve
    # uses, or the coarse volumes won't correspond to the flat problem and the comparison is
    # meaningless.
    #
    # CHUNKS_PER_PAIR is the flat input num_chunks (per source, per destination). AllGather passes
    # it through unscaled, so fine_chunks == CHUNKS_PER_PAIR. AllToAll is DIFFERENT: the scheduler
    # scales the flat input by the participating-GPU count (scheduler.get_solver, ALLTOALL branch:
    # num_chunks *= num_gpus), and build_demand then lays down fine_chunks // num_gpus chunks per
    # ordered pair -- so to reproduce the flat alltoall we must pre-scale here to
    # CHUNKS_PER_PAIR * num_gpus (build_demand is called DIRECTLY, it does not go through the
    # scheduler's scaling). Keep CHUNKS_PER_PAIR in lockstep with the flat input JSON
    # (hetero_alltoall_lp.json num_chunks).
    CHUNKS_PER_PAIR = 1
    num_participating = sum(len(c.gpus) for c in mapping.coarse_cells.values())
    if collective == Collective.ALLGATHER:
        fine_chunks = CHUNKS_PER_PAIR
    else:
        fine_chunks = CHUNKS_PER_PAIR * num_participating
    fine_demand = build_demand(collective, fine, fine_chunks)

    tag = coll_arg
    prefix = f"coarse_hetero_{tag}"
    milp_out = f"Schedules/{prefix}_milp.json"
    lp_out = f"Schedules/{prefix}_lp.json"

    print(f"fine topology: {len(fine.capacity)} nodes, {len(fine.cells)} cells, "
          f"collective={coll_arg}, which={which}")

    if which in ("both", "milp"):
        print("\n=== MILP (switch_copy=True, multicast) ===")
        try:
            solve_hierarchical(fine, _make_input(Formulation.MILP, collective, milp_out),
                               collective, fine_chunks, prefix=f"{prefix}_milp",
                               fine_demand=fine_demand,
                               level_chunk=1 if no_coarsen else None)
        except Exception as e:
            print(f"MILP solve failed: {type(e).__name__}: {e}")
    if which in ("both", "lp"):
        print("\n=== LP (switch_copy=False, unicast) ===")
        try:
            solve_hierarchical(fine, _make_input(Formulation.LP, collective, lp_out),
                               collective, fine_chunks, prefix=prefix,
                               fine_demand=fine_demand, write_outputs=True,
                               report=make_reporter(prefix, tag),
                               level_chunk=1 if no_coarsen else None)
        except Exception as e:
            print(f"LP solve / hierarchical reconstruction failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    print_solve_summary((("MILP", milp_out), ("LP", lp_out)))


if __name__ == "__main__":
    main()
