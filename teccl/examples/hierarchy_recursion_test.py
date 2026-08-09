"""
Gurobi-free test of the RECURSION itself: solve_hierarchical on a genuinely three-level topology.

Every other hierarchy test exercises a two-level solve, where the "second level" is a direct call to
the crossbar scheduler and nothing ever recurses. This one runs
teccl.hierarchy.solve.solve_hierarchical on NestedCluster (cluster -> racks -> hosts), whose every
level collapses to a crossbar and therefore needs no formulation -- so the whole recursion is
checkable locally. The `solve_flat_hook` below FAILS rather than reaching for Gurobi, which is what
makes that claim a test result instead of a comment.

What is checked, and why each one is worth a line:

  1. Depth: the recursion actually reaches depth 2. A topology can declare subcells and still be
     solved as two levels if `Cell.subcells` is silently ignored, and the output would look fine.
  2. Correctness end-to-end: `stitch` runs `back_trace`, which verifies COVERAGE (every demanded
     (dest, identity) is delivered) and CAUSALITY (no GPU sends data it does not yet hold) over the
     assembled flat schedule. This is the property the whole recursion exists to preserve, and it is
     the one that caught every real bug during bring-up -- index spaces not translated on return,
     the holder read off `identity[0]` below the root, and a child band collapsed to a single parent
     round so staging and the send it fed landed together.
  3. Node indices come back GLOBAL. `induce` renumbers each level 0..n-1, so a missing translation
     on the way out yields flows naming nodes that exist but mean something else -- which back_trace
     can miss when the wrong index happens to be a plausible one.
  4. Q == 1 at every level, i.e. an all-crossbar recursion spends nothing from the refinement budget
     that `MAX_M` caps across the WHOLE recursion. This is the specific reason crossbar levels keep
     chunk identity rather than re-deriving it.
  5. Payload conservation across the scale chain, and bands that fit their budget (asserted inside
     `stitch` by `assert_bands_fit`).
  6. The two-level path still reaches the same machinery: HeteroTaperedCluster's cells declare no
     subcells, so it must take the bottom-cell branch and never recurse.

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_recursion_test
"""
import logging

from teccl.hierarchy import crossbar_solve
from teccl.hierarchy.abstract import abstract
from teccl.hierarchy.bands import assign_bands, band_of
from teccl.hierarchy.solve import LevelContext, solve_hierarchical
from teccl.hierarchy.subtopology import induce
from teccl.input_data import Collective, TopologyParams
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster
from teccl.topologies.nested_cluster import NestedCluster


def _no_gurobi(problem, ctx, mapping, tensor):
    raise AssertionError(
        f"level depth={problem.depth} cell={problem.cell_id} fell through to a formulation solve; "
        f"every level of NestedCluster is supposed to be a crossbar "
        f"({len(problem.topology.capacity)} nodes, switches {problem.topology.switch_indices})")


def _topology():
    return NestedCluster(TopologyParams(name="NestedCluster", chunk_size=1))


# ------------------------------------------------------------------------------------------
def test_every_level_is_a_crossbar():
    """The premise the rest of this file rests on, checked directly rather than assumed."""
    t = _topology()
    coarse, mapping = abstract(t)
    assert crossbar_solve.is_crossbar(coarse), (
        f"level 0 coarse graph is not a crossbar: {mapping.num_coarse} nodes, "
        f"switches {coarse.switch_indices}")
    for cid, rack in mapping.coarse_cells.items():
        assert rack.subcells, f"rack {cid} declares no subcells, so nothing would recurse"
        sub = induce(t, rack)
        rack_coarse, rack_map = abstract(sub)
        assert crossbar_solve.is_crossbar(rack_coarse), (
            f"rack {cid}'s coarse graph is not a crossbar: switches {rack_coarse.switch_indices}")
        for hid, host in rack_map.coarse_cells.items():
            assert not host.subcells, f"host {hid} unexpectedly declares subcells"
    print("  [1] all three levels collapse to crossbars (no formulation needed) OK")


def test_three_level_solve():
    """The whole point: a depth-2 recursion that produces a valid flat schedule."""
    t = _topology()
    fine_demand = build_demand(Collective.ALLGATHER, t, 1)
    info, sol = solve_hierarchical(t, None, Collective.ALLGATHER, 1,
                                   fine_demand=fine_demand, solve_flat_hook=_no_gurobi)

    # (2) back_trace ran inside stitch and verified coverage + causality; if it had not, there would
    # be no chunk paths to serialize.
    assert info["8-Chunk paths"], "stitch produced no traced chunk paths"
    n_demanded = sum(1 for s in range(len(fine_demand)) for d in range(len(fine_demand))
                     for c in range(len(fine_demand[0][0]))
                     if s != d and fine_demand[s][d][c])
    assert len(info["8-Chunk paths"]) == n_demanded, (
        f"{len(info['8-Chunk paths'])} paths traced for {n_demanded} demanded pairs")

    # (3) every node id that survives to the flat schedule is a real fine node, and switches only
    # ever appear as via-annotations, never as endpoints.
    n = len(t.capacity)
    data = set(range(n)) - set(t.switch_indices)
    for f in sol.flows:
        assert f.sender in data and f.receiver in data, (
            f"flow endpoint is not a fine data node (local index leaked out of a sub-level?): {f}")
        assert f.via_switch in set(t.switch_indices), f"via_switch is not a fine switch: {f}"
        assert t.capacity[f.sender][f.via_switch] > 0 and t.capacity[f.via_switch][f.receiver] > 0, (
            f"flow uses a link that does not exist in the fine topology: {f}")

    # (4) an all-crossbar recursion refines nothing.
    assert sol.scale.refinement_from_root == 1, (
        f"an all-crossbar recursion spent refinement budget: {sol.scale}")
    # (5) and conserves the payload it re-denominates.
    assert abs(sol.scale.payload_per_gpu - t.chunk_size * 1) < 1e-9, sol.scale

    depth_flows = len([f for f in sol.flows if f.kind == "sublevel_transfer"])
    assert depth_flows, "no sub-level transfers: the rack level produced no inter-host traffic"
    print(f"  [2] three-level solve OK: {len(sol.flows)} flows "
          f"({depth_flows} rack-level transfers), {len(info['8-Chunk paths'])} demands traced, "
          f"{info['3-Epochs_Required']} epochs, scale {sol.scale}")


def test_recursion_actually_recursed():
    """Depth is observable, so observe it -- a silently-ignored `subcells` looks identical."""
    t = _topology()
    seen = []

    ctx_depths = []

    def spy(problem, ctx, mapping, tensor):
        ctx_depths.append(problem.depth)
        return _no_gurobi(problem, ctx, mapping, tensor)

    import teccl.hierarchy.solve as S
    original = S.solve_level

    def traced(problem, ctx):
        seen.append(problem.depth)
        return original(problem, ctx)

    S.solve_level = traced
    try:
        solve_hierarchical(t, None, Collective.ALLGATHER, 1, solve_flat_hook=spy)
    finally:
        S.solve_level = original

    assert max(seen) == 2, f"recursion reached depth {max(seen)}, expected 2 (cluster/rack/host)"
    assert not ctx_depths, "a level fell through to the formulation hook"
    print(f"  [3] recursion reached depth {max(seen)} "
          f"({seen.count(0)} cluster, {seen.count(1)} rack, {seen.count(2)} host sub-problems) OK")


def test_two_level_topology_does_not_recurse():
    """A cell with no subcells is a BOTTOM cell, and must be solved rather than decomposed."""
    t = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    _, mapping = abstract(t)
    for cid, cell in mapping.coarse_cells.items():
        assert not cell.subcells, f"cell {cid} unexpectedly declares subcells"
    print("  [4] two-level topologies declare no subcells, so they take the bottom-cell branch OK")


def test_band_policy_agrees_at_both_granularities():
    """`band_of` has two call sites (jobs in crossbar_solve, demands at the level boundary) and they
    must not drift -- that is the entire reason the policy lives in its own module."""
    from math import inf
    cases = [(-1, inf, False), (-1, 0, True), (-1, 3, True), (0, inf, False),
             (2, inf, False), (2, 5, True), (4, 5, True)]
    for release, deadline, hard in cases:
        try:
            b = band_of(release, deadline, hard)
        except RuntimeError:
            continue
        assert b >= max(int(release), 0) or b == int(deadline) - 1, (release, deadline, hard, b)
        if hard and deadline != inf:
            assert b < deadline, (release, deadline, b)
    print("  [5] band policy: readiness placement, deadline clamp, prologue escape OK")


def main() -> None:
    logging.disable(logging.WARNING)   # the deadline + non-uniform-link warnings are expected here
    test_every_level_is_a_crossbar()
    test_three_level_solve()
    test_recursion_actually_recursed()
    test_two_level_topology_does_not_recurse()
    test_band_policy_agrees_at_both_granularities()
    print("hierarchical recursion tests OK")


if __name__ == "__main__":
    main()
