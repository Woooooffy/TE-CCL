"""
The drivers' shared REPORTING half: what a hierarchical run says about itself.

Orchestration used to live here too -- each driver called run_identity_resolution ->
run_phase3_intra -> run_stitch in order, which is what made "two levels" a property of the driver
rather than of the topology. That sequence is now `teccl.hierarchy.solve.solve_hierarchical`, which
recurses to whatever depth the topology declares, and a driver just calls it.

What remains is narration. Every line these functions print stands in for a defect that was once
silent -- a fine link oversubscribed by 150%, a merge combining two disjoint byte ranges of one
chunk, intra work that did not fit the coarse epoch it ran under -- so they are worth printing on
every remote run. They stay OUT of the solver because which of them a driver wants, and under what
name, is a driver's business; the recursion should not grow a reporting policy.

`solve_on_topology` also stays: it is how a level reaches TECCLSolver with an already-built
Topology object, which `get_topology` cannot do (it only knows the named built-ins).
"""
import copy
import json
from collections import defaultdict

from teccl.hierarchy.stitch import derive_grid
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


def make_reporter(prefix: str, tag: str):
    """Build the `report=` callback `solve.solve_hierarchical` calls once the root level has been
    lowered, so the remote .out log keeps narrating the invariants it always did.

    The narration is deliberately NOT inside the solver. Each of these lines replaced a silent
    defect, so they are worth printing on every run -- but which of them a given driver wants, and
    under what name, is a driver's business, and the recursion should not grow a reporting policy.
    """
    def report(res, flows, fine, coarse_epoch):
        if res is None:
            print("no resolution to report")
            return
        describe_resolution(res, prefix)
        report_resolution_invariants(res, fine, coarse_epoch)
        print(f"\n=== sub-level schedule ({tag}) ===")
        print(f"  {len(flows)} fine flows across "
              f"{len({(f.cell, f.band) for f in flows})} (cell, band) pairs")
        report_intra_fits_epoch(flows, res, fine, coarse_epoch)
    return report


def describe_resolution(res, prefix: str) -> None:
    """The per-cell relay summary the driver has always printed before the invariant checks."""
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
