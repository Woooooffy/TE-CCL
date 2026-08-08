"""
The topology-independent half of the hierarchical driver: everything below the coarse solve.

A driver picks a fine topology, abstracts it, builds the coarse demand and solves it; from there
the pipeline is identical whatever the topology looks like -- identity resolution (phase 2) ->
intra-cell NVSwitch schedule (phase 3) -> stitch (phase 4) -> flat schedule on the fine topology.
That shared tail lives here, so hierarchy_coarse_solve_hetero (irregular 3-host cluster) and
hierarchy_coarse_solve_rail (symmetric 32-host rail-optimized spine-leaf) differ only in the parts
that are genuinely topology-specific.

`prefix` is the schedule-path stem the driver owns: outputs are
Schedules/{prefix}_{lp,milp,identities,intra,flat}.json.
"""
import copy
import json
from collections import defaultdict
from dataclasses import asdict

from teccl.hierarchy.stitch import NETWORK, derive_grid, stitch
from teccl.hierarchy.intra_solve import schedule_cell
from teccl.hierarchy.reconstruct import resolve_identities
from teccl.input_data import UserInputParams
from teccl.scheduler import TECCLSolver
from teccl.topologies.topology import Topology


def solve_on_topology(user_input: UserInputParams, topology: Topology) -> TECCLSolver:
    """Run TECCLSolver.solve() against an already-built Topology (bypassing get_topology, which
    only knows the named built-ins, not CoarseTopology). Returns the TECCLSolver so the caller
    can reach the solved formulation (teccl_solver.best_solver) for post-processing."""
    solver = TECCLSolver.__new__(TECCLSolver)
    solver.user_input = user_input
    solver.topology_obj = topology
    solver.solver = solver.get_solver(copy.deepcopy(user_input), topology)
    solver.solve()
    return solver


def run_identity_resolution(lp_solver, mapping, fine_demand, fine, coarse_epoch: float,
                            prefix: str, level_chunk: int = 1):
    """Resolve the identity-free coarse LP solution into concrete fine identities + intra-cell
    demands, print a summary, and serialize to Schedules/{prefix}_identities.json.
    Returns the IdentityResolution (for phase-3), or None if there was nothing to resolve.

    `level_chunk` is the `g` the coarse level was solved in (abstract.set_level_chunk); it only
    tells the resolver how to read the coarse volumes, which it then re-denominates into fine
    identities, so the resolution itself is invariant to it."""
    if not getattr(lp_solver, "best_solver", None):
        print("no solved LP formulation to resolve (best_solver unset)")
        return None
    res = resolve_identities(lp_solver.best_solver, mapping, fine_demand, fine,
                             level_chunk=level_chunk)

    egress = [d for d in res.intra_demands if d.kind == "egress_stage"]
    ingress = [d for d in res.intra_demands if d.kind == "ingress_distribution"]
    selfd = [d for d in res.intra_demands if d.kind == "self_distribution"]
    print(f"\n--- identity resolution ({prefix}) ---")
    print(f"resolved inter-cell pieces: {len(res.pieces)}")
    print(f"intra-cell demands: {len(egress)} egress_stage, {len(ingress)} ingress_distribution, "
          f"{len(selfd)} self_distribution")
    per_cell_relay = {}
    for d in egress:
        per_cell_relay.setdefault(d.cell, []).append((d.identity, d.src_gpu, d.dst_gpus[0]))
    for cell in sorted(per_cell_relay):
        print(f"  cell {cell} egress relays: "
              f"{sorted(set(f'{s}->{g}' for (_id, s, g) in per_cell_relay[cell]))}")

    report_resolution_invariants(res, fine, coarse_epoch)

    out = {
        "pieces": [asdict(p) for p in res.pieces],
        "intra_demands": [asdict(d) for d in res.intra_demands],
        # to_json(), not asdict(): the scale holds exact Fractions and asdict passes them through
        # raw, which no JSON encoder can write.
        "scale": res.scale.to_json() if res.scale else None,
        "subdivision": res.subdivision,
    }
    path = f"Schedules/{prefix}_identities.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=list)
    print(f"identity resolution written to {path}")
    return res


def report_resolution_invariants(res, fine, coarse_epoch: float) -> None:
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

    ep = coarse_epoch
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


def run_phase3_intra(res, mapping, prefix: str, fine=None, coarse_epoch: float = None,
                     debug: bool = True):
    """Phase-3: schedule every cell's intra-cell demands onto its NVSwitch (Gurobi-free, EDF
    edge-coloring). Debug narration is on by default so the .out log shows the full per-step
    derivation -- fan-out density decisions, dedup, per-round matchings, and optimality vs the
    port-load bound. On a topology with many identical cells that narration is repeated per cell
    and can dominate the log, so drivers may pass debug=False.

    Serializes the fine IntraFlows to Schedules/{prefix}_intra.json and RETURNS them, since the
    stitch consumes them together with the resolution."""
    if res is None:
        return []
    by_cell = {}
    for d in res.intra_demands:
        by_cell.setdefault(d.cell, []).append(d)

    print(f"\n=== phase-3 intra-cell scheduling ({prefix}) ===")
    all_flows = []
    for cid in sorted(mapping.coarse_cells):
        cell = mapping.coarse_cells[cid]
        demands = by_cell.get(cid, [])
        if not demands:
            continue
        # switch_copy=False: the LP path is unicast, so the intra fabric is modeled unicast too.
        flows = schedule_cell(cid, cell, demands, switch_copy=False, debug=debug,
                              subdivision=res.subdivision)
        all_flows.extend(flows)

    # kind/hard are the job provenance the stitch places a flow by: `gap` alone is ambiguous
    # (a self_distribution and an epoch-0 staging relay both land in gap 0).
    out = [dict(cell=f.cell, identity=list(f.identity), sender=f.sender, receiver=f.receiver,
                via_switch=f.via_switch, volume=f.volume, band=f.band, local_round=f.local_round,
                span=f.span, identities=[list(i) for i in f.identities],
                kind=f.kind, hard=f.hard)
           for f in all_flows]
    path = f"Schedules/{prefix}_intra.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nphase-3 intra schedule: {len(all_flows)} fine flows written to {path}")
    if fine is not None and res.scale is not None and coarse_epoch is not None:
        report_intra_fits_epoch(all_flows, res, fine, coarse_epoch)
    return all_flows


def report_intra_fits_epoch(flows, res, fine, coarse_epoch: float) -> None:
    """The phase-3 feasibility certificate: intra-cell work must fit inside a coarse epoch.

    A "round" is one chunk across one NVSwitch port, so a round lasts
    scale.bytes_per_chunk / nvlink_bw and m = coarse_epoch / that is how many rounds a coarse epoch
    can hold. `port_cap = 1.0` is scale-invariant (it DEFINES the round), so refinement changes a
    round's duration, not its capacity -- both the round count and m scale with Q, and the margin
    is preserved. Asserting `peak rounds <= m` is what certifies the whole "the inner fabric is
    much faster than the outer" premise the per-gap-independent timing rests on.

    delta and m come from the stitch's own derive_grid rather than being recomputed here: this
    report and the axis the stitch actually lays out must agree by construction, and they used to
    be two independent copies of the same arithmetic."""
    delta, m = derive_grid(res.scale, fine, coarse_epoch)
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


def run_stitch(res, intra_flows, fine, fine_demand, coarse_epoch: float, tag: str, prefix: str):
    """Phase-4: merge the inter-cell pieces and the intra-cell flows into ONE flat schedule on the
    fine topology, written to Schedules/{prefix}_flat.json.

    That file is an ordinary flat schedule -- ncclize consumes it with no hierarchy awareness:
        python teccl/ncclize/teccl_ncclize.py --schedule Schedules/{prefix}_flat.json -o out.xml
    which runs check_implements() and so independently validates that the stitched schedule really
    implements the collective."""
    if res is None or not intra_flows and not res.pieces:
        return None
    print(f"\n=== phase-4 stitch ({tag}) ===")
    info, records = stitch(res, intra_flows, fine, fine_demand, coarse_epoch, tag)

    path = f"Schedules/{prefix}_flat.json"
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


def print_solve_summary(paths) -> None:
    """One line per solved formulation, from the schedule JSON it wrote."""
    for label, path in paths:
        try:
            with open(path) as f:
                d = json.load(f)
            print(f"\n{label}: epochs={d.get('3-Epochs_Required')} "
                  f"finish={d.get('4-Collective_Finish_Time')} "
                  f"bw={d.get('5-Algo_Bandwidth')} solver_time={d.get('Solver_Time')}")
        except FileNotFoundError:
            pass
