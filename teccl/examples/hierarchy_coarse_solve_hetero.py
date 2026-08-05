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

Run from the repo root:
    python -m teccl.examples.hierarchy_coarse_solve_hetero [allgather|alltoall] [lp|milp|both]
"""
import copy
import json
from collections import defaultdict
import sys
import traceback

from dataclasses import asdict

from teccl.hierarchy.abstract import abstract, coarsify_demand, lift_demand
from teccl.hierarchy.reconstruct import resolve_identities
from teccl.hierarchy.intra_solve import schedule_cell
from teccl.hierarchy.stitch import NETWORK, stitch
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


def _run_identity_resolution(lp_solver, mapping, fine_demand, fine, tag: str):
    """Resolve the identity-free coarse LP solution into concrete fine identities + intra-cell
    demands, print a summary, and serialize to Schedules/coarse_hetero_{tag}_identities.json.
    Returns the IdentityResolution (for phase-3), or None if there was nothing to resolve."""
    if not getattr(lp_solver, "best_solver", None):
        print("no solved LP formulation to resolve (best_solver unset)")
        return None
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
              f"{sorted(set(f'{s}->{g}' for (_id, s, g) in per_cell_relay[cell]))}")

    _report_resolution_invariants(res, fine)

    out = {
        "pieces": [asdict(p) for p in res.pieces],
        "intra_demands": [asdict(d) for d in res.intra_demands],
        "scale": asdict(res.scale) if res.scale else None,
        "subdivision": res.subdivision,
    }
    path = f"Schedules/coarse_hetero_{tag}_identities.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=list)
    print(f"identity resolution written to {path}")
    return res


def _report_resolution_invariants(res, fine) -> None:
    """Narrate (and assert) the invariants the joint assignment + sub-chunk refinement establish.

    These are the properties worth eyeballing in the remote .out log, because each replaced a
    silent defect:
      * every emitted volume is a whole sub-chunk, so no downstream merge can combine two disjoint
        byte ranges of one identity;
      * the ChunkScale conserves the per-GPU payload it re-denominates;
      * no fine uplink OR downlink is oversubscribed in any coarse epoch -- the downlink half was
        being violated by 150% because the ingress gateway was chosen without a capacity term;
      * the fraction of arrivals landing directly on a GPU that wants the data (the ingress half of
        the objective; 0 relays possible only where a target owns a boundary link).
    """
    scale, q = res.scale, res.subdivision
    print(f"  chunk scale: {scale}  (subdivision Q={q})")
    assert all(abs(p.volume - 1.0) < 1e-9 for p in res.pieces), "non-unit piece volume survived"
    assert all(abs(d.volume - 1.0) < 1e-9 for d in res.intra_demands), "non-unit demand volume"

    ep = 0.02 if not hasattr(fine, "epoch_duration") else fine.epoch_duration
    eg = defaultdict(float)
    ing = defaultdict(float)
    for p in res.pieces:
        gb = p.volume * scale.bytes_per_chunk
        eg[(p.egress_gpu, p.via_switches[0], p.send_epoch)] += gb
        ing[(p.ingress_gpu, p.via_switches[-1], p.arrival_epoch)] += gb
    over = []
    for (g, sw, k), vol in sorted(eg.items()):
        cap = fine.capacity[g][sw] * ep
        if vol > cap + 1e-9:
            over.append(("egress", g, sw, k, round(vol, 4), round(cap, 4)))
    for (h, sw, k), vol in sorted(ing.items()):
        cap = fine.capacity[sw][h] * ep
        if vol > cap + 1e-9:
            over.append(("ingress", h, sw, k, round(vol, 4), round(cap, 4)))
    worst_e = max((v / (fine.capacity[g][sw] * ep) for (g, sw, _), v in eg.items()), default=0)
    worst_i = max((v / (fine.capacity[sw][h] * ep) for (h, sw, _), v in ing.items()), default=0)
    print(f"  fine-link occupancy: egress peak {100 * worst_e:.0f}%, ingress peak "
          f"{100 * worst_i:.0f}% of per-epoch capacity; violations: {len(over)}")
    assert not over, f"fine link oversubscribed in some coarse epoch: {over[:6]}"

    single = [d for d in res.intra_demands
              if d.kind == "ingress_distribution" and len(d.dst_gpus) == 1]
    if single:
        landed = sum(1 for d in single if d.dst_gpus[0] == d.src_gpu)
        print(f"  ingress landing: {landed}/{len(single)} single-target arrivals "
              f"({100 * landed / len(single):.0f}%) landed directly on a GPU that wants them")
    relayed = sum(1 for p in res.pieces if p.egress_gpu != p.identity[0])
    print(f"  egress: {relayed}/{len(res.pieces)} pieces leave via a non-native gateway "
          f"(lexicographic tier 1 minimizes this; the ingress tier never trades against it)")


def _run_phase3_intra(res, mapping, tag: str, fine=None, coarse_epoch: float = None):
    """Phase-3: schedule every cell's intra-cell demands onto its NVSwitch (Gurobi-free, EDF
    edge-coloring). Debug narration is on so the .out log shows the full per-step derivation --
    fan-out density decisions, dedup, per-round matchings, and optimality vs the port-load bound.
    Serializes the fine IntraFlows to Schedules/coarse_hetero_{tag}_intra.json and RETURNS them,
    since the stitch consumes them together with the resolution."""
    if res is None:
        return []
    by_cell = {}
    for d in res.intra_demands:
        by_cell.setdefault(d.cell, []).append(d)

    print(f"\n=== phase-3 intra-cell scheduling ({tag}) ===")
    all_flows = []
    for cid in sorted(mapping.coarse_cells):
        cell = mapping.coarse_cells[cid]
        demands = by_cell.get(cid, [])
        if not demands:
            continue
        # switch_copy=False: the LP path is unicast, so the intra fabric is modeled unicast too.
        flows = schedule_cell(cid, cell, demands, switch_copy=False, debug=True,
                              subdivision=res.subdivision)
        all_flows.extend(flows)

    # kind/hard are the job provenance the stitch places a flow by: `gap` alone is ambiguous
    # (a self_distribution and an epoch-0 staging relay both land in gap 0).
    out = [dict(cell=f.cell, identity=list(f.identity), sender=f.sender, receiver=f.receiver,
                via_switch=f.via_switch, volume=f.volume, band=f.band, local_round=f.local_round,
                span=f.span, identities=[list(i) for i in f.identities],
                kind=f.kind, hard=f.hard)
           for f in all_flows]
    path = f"Schedules/coarse_hetero_{tag}_intra.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nphase-3 intra schedule: {len(all_flows)} fine flows written to {path}")
    if fine is not None and res.scale is not None and coarse_epoch is not None:
        _report_intra_fits_epoch(all_flows, res, fine, coarse_epoch)
    return all_flows


def _report_intra_fits_epoch(flows, res, fine, coarse_epoch: float) -> None:
    """The phase-3 feasibility certificate: intra-cell work must fit inside a coarse epoch.

    A "round" is one chunk across one NVSwitch port, so a round lasts
    scale.bytes_per_chunk / nvlink_bw and m = coarse_epoch / that is how many rounds a coarse epoch
    can hold. `port_cap = 1.0` is scale-invariant (it DEFINES the round), so refinement changes a
    round's duration, not its capacity -- both the round count and m scale with Q, and the margin
    is preserved. Asserting `peak rounds <= m` is what certifies the whole "the inner fabric is
    much faster than the outer" premise the per-gap-independent timing rests on."""
    nvlink_bw = max(max(row) for row in fine.capacity)
    delta = res.scale.epoch_duration(nvlink_bw)
    m = coarse_epoch / delta
    per_gap = defaultdict(int)
    for f in flows:
        per_gap[(f.cell, f.band)] = max(per_gap[(f.cell, f.band)], f.local_round + f.span)
    peak = max(per_gap.values(), default=0)
    hot = max(per_gap, key=lambda k: per_gap[k]) if per_gap else None
    print(f"  intra fits coarse epoch: fine epoch delta={delta:.3e}s, m={m:.1f} rounds per coarse "
          f"epoch; peak {peak} rounds at cell/gap {hot} -> {100 * peak / m:.1f}% of the budget")
    assert peak <= m + 1e-9, (
        f"intra-cell work does not fit a coarse epoch: {peak} rounds > m={m:.1f} at {hot}. The "
        f"inner fabric is not fast enough relative to the outer for per-gap-independent timing; "
        f"the windowed intra solver is the fallback.")


def _run_stitch(res, intra_flows, fine, fine_demand, coarse_epoch: float, tag: str):
    """Phase-4: merge the inter-cell pieces and the intra-cell flows into ONE flat schedule on the
    fine topology, written to Schedules/coarse_hetero_{tag}_flat.json.

    That file is an ordinary flat schedule -- ncclize consumes it with no hierarchy awareness:
        python teccl/ncclize/teccl_ncclize.py --schedule Schedules/coarse_hetero_{tag}_flat.json \\
               -o /tmp/{tag}.xml
    which runs check_implements() and so independently validates that the stitched schedule really
    implements the collective."""
    if res is None or not intra_flows and not res.pieces:
        return None
    print(f"\n=== phase-4 stitch ({tag}) ===")
    info, records = stitch(res, intra_flows, fine, fine_demand, coarse_epoch, tag)

    path = f"Schedules/coarse_hetero_{tag}_flat.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)

    net = [r for r in records if r.phase == NETWORK]
    by_phase = defaultdict(int)
    for r in records:
        by_phase[r.phase] += 1
    print(f"  {len(records)} delivery records {dict(by_phase)}; "
          f"{len(info['8-Chunk paths'])} demands traced (causality + coverage verified)")
    print(f"  fine epoch delta={info['1-Epoch_Duration']:.4e}s x "
          f"{info['3-Epochs_Required']} epochs = {info['4-Collective_Finish_Time']:.4f}s, "
          f"algo bw={info['5-Algo_Bandwidth']:.2f}, chunk={info['9-Chunk_Size']}")
    # Only the coarse level paces its flows; the intra level deliberately does not, so this ratio
    # should be exactly the network/intra split above.
    rates = sorted({r.rate for r in net})
    print(f"  paced network sends: {len(net)} at rate(s) {rates} GB/s; "
          f"{len(records) - len(net)} intra sends unpaced")
    print(f"  flat schedule written to {path}")
    return info


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
            # The coarse epoch is the coarse solve's own epoch duration -- every downstream
            # quantity (m, the staging deadlines, the network pacing rate) is derived from it, so
            # it must be read off the solved formulation rather than restated.
            coarse_epoch = lp_solver.best_solver.epoch_duration
            res = _run_identity_resolution(lp_solver, mapping, fine_demand, fine, tag)
            intra_flows = _run_phase3_intra(res, mapping, tag, fine, coarse_epoch)
            _run_stitch(res, intra_flows, fine, fine_demand, coarse_epoch, tag)
        except Exception as e:
            print(f"LP solve / hierarchical reconstruction failed: {type(e).__name__}: {e}")
            traceback.print_exc()

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
