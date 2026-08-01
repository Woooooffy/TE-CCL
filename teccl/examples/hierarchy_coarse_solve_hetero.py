"""
Coarse solve of the IRREGULAR HeteroTaperedCluster (3 heterogeneous hosts A:4 / B:4 / C:6 GPUs,
tapered uplinks, single-homed Host B, multi-GPU boundary on Host C) -- the topology built to
exercise the phase-3 machinery the symmetric rail topology never triggers.

Unlike the rail driver, the coarse AllGather here is NON-UNIFORM: cells have different GPU
counts, so the coarse demand is the heterogeneous 4/4/6 volume matrix produced by
coarsify_demand (collective-agnostic) rather than a uniform num_chunks-per-source generator.
The coarse demand is injected via topology.demand_override, so the coarse solve satisfies the
aggregated inter-cell volumes directly. AllToAll is also supported (coarse volume |U|*|V|).

No stitching / phase-3 reconstruction yet -- this verifies that the coarse problem solves and
that the flows honor the forced-relay structure (uplinks < GPUs on every cell). Requires
Gurobi, so it runs on the remote solver host.

Run from the repo root:
    python -m teccl.examples.hierarchy_coarse_solve_hetero [allgather|alltoall] [lp|milp|both]
"""
import copy
import json
import sys

from dataclasses import asdict

from teccl.hierarchy.abstract import abstract, coarsify_demand, lift_demand
from teccl.hierarchy.reconstruct import resolve_identities
from teccl.input_data import (
    Collective, EpochType, Formulation, InstanceParams, ObjectiveType,
    SolutionMethod, TopologyParams, UserInputParams,
)
from teccl.scheduler import TECCLSolver
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster
from teccl.topologies.topology import Topology


def _solve_on_topology(user_input: UserInputParams, topology: Topology) -> TECCLSolver:
    """Run TECCLSolver.solve() against an already-built Topology (bypassing get_topology, which
    only knows the named built-ins, not CoarseTopology). Returns the TECCLSolver so the caller
    can reach the solved formulation (teccl_solver.best_solver) for post-processing."""
    solver = TECCLSolver.__new__(TECCLSolver)
    solver.user_input = user_input
    solver.topology_obj = topology
    solver.solver = solver.get_solver(copy.deepcopy(user_input), topology)
    solver.solve()
    return solver


def _run_identity_resolution(lp_solver, mapping, fine_demand, fine, tag: str) -> None:
    """Resolve the identity-free coarse LP solution into concrete fine identities + intra-cell
    demands, print a summary, and serialize to Schedules/coarse_hetero_{tag}_identities.json."""
    if not getattr(lp_solver, "best_solver", None):
        print("no solved LP formulation to resolve (best_solver unset)")
        return
    res = resolve_identities(lp_solver.best_solver, mapping, fine_demand, fine)

    egress = [d for d in res.intra_demands if d.kind == "egress_stage"]
    ingress = [d for d in res.intra_demands if d.kind == "ingress_distribution"]
    selfd = [d for d in res.intra_demands if d.kind == "self_distribution"]
    print(f"\n--- identity resolution ({tag}) ---")
    print(f"resolved inter-cell pieces: {len(res.pieces)}")
    print(f"intra-cell demands: {len(egress)} egress_stage, {len(ingress)} ingress_distribution, "
          f"{len(selfd)} self_distribution")
    per_cell_relay = {}
    for d in egress:
        per_cell_relay.setdefault(d.cell, []).append((d.identity, d.src_gpu, d.dst_gpus[0]))
    for cell in sorted(per_cell_relay):
        print(f"  cell {cell} egress relays: "
              f"{sorted((f'{s}->{g}') for (_id, s, g) in per_cell_relay[cell])}")

    out = {
        "pieces": [asdict(p) for p in res.pieces],
        "intra_demands": [asdict(d) for d in res.intra_demands],
    }
    path = f"Schedules/coarse_hetero_{tag}_identities.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=list)
    print(f"identity resolution written to {path}")


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
    coarse.demand_override = coarse_demand

    vols = {(u, v): coarse_demand[u][v][0]
            for u in range(mapping.num_coarse) for v in range(mapping.num_coarse)
            if coarse_demand[u][v][0]}
    print(f"coarse topology: {mapping.num_coarse} nodes "
          f"({len(mapping.coarse_cells)} cells + {len(coarse.switch_indices)} switches), "
          f"collective={coll_arg}, which={which}")
    print(f"coarse demand volumes (U->V): {vols}")

    tag = coll_arg
    milp_out = f"Schedules/coarse_hetero_{tag}_milp.json"
    lp_out = f"Schedules/coarse_hetero_{tag}_lp.json"

    if which in ("both", "milp"):
        print("\n=== MILP (switch_copy=True, multicast) ===")
        try:
            _solve_on_topology(_make_input(Formulation.MILP, collective, milp_out), coarse)
        except Exception as e:
            print(f"MILP solve failed: {type(e).__name__}: {e}")
    if which in ("both", "lp"):
        print("\n=== LP (switch_copy=False, unicast) ===")
        try:
            lp_solver = _solve_on_topology(_make_input(Formulation.LP, collective, lp_out), coarse)
            _run_identity_resolution(lp_solver, mapping, fine_demand, fine, tag)
        except Exception as e:
            print(f"LP solve / identity resolution failed: {type(e).__name__}: {e}")

    for label, path in (("MILP", milp_out), ("LP", lp_out)):
        try:
            with open(path) as f:
                d = json.load(f)
            print(f"\n{label}: epochs={d.get('3-Epochs_Required')} "
                  f"finish={d.get('4-Collective_Finish_Time')} "
                  f"bw={d.get('5-Algo_Bandwidth')} solver_time={d.get('Solver_Time')}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
