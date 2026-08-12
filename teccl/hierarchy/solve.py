"""
The hierarchical solver's entry point, and the recursion underneath it.

`solve_hierarchical(topology, ...)` is the whole pipeline: build the demand, solve the levels, stitch
the result into one flat schedule on the fine topology. Before this existed, every driver open-coded
`abstract -> coarsify_demand -> set_level_chunk -> solve -> resolve_identities -> schedule_cell ->
stitch`, and the "second level" was a direct call to the crossbar scheduler rather than a call to
anything recursive -- so a third level was not a configuration, it was a rewrite.

THE BASE CASE IS "A LEVEL THAT DECLARES NO SUB-CELLS", not "a single node". A level whose nodes are
real endpoints has nothing left to collapse, so it is solved rather than decomposed. Everything else
in the dispatch table is a degenerate or optimized instance of that:

    degenerate     no demand, or <= 1 data-bearing node        -> nothing
    crossbar memo  one switch, every data node hanging off it  -> closed form, no Gurobi
    flat           no sub-cells, not a crossbar                -> a real formulation (Gurobi)
    recursive      topology.cells non-empty                    -> abstract, solve, lower, recurse

That makes the crossbar a MEMOIZED ROW in the base-case table rather than a layer of the design,
which is what the forward-plan note always claimed it was.

ONE PASS THROUGH A LEVEL, in order, and each step's output is the next one's input:

    solve_flat        this level's routing, by whichever solver the table picked
    step A            which identity rode which piece (reconstruct.assign_identities_*)
    step B            integerize + emit -> THE NEXT LEVEL'S PROBLEM (build_child_problems)
    derive_grid       this level's m: how many of the child's rounds fit in one of my epochs
    assign_bands      split the child's demands by band (teccl.hierarchy.bands)
    recurse           one solve_level per (cell, band)
    rebase            flatten the children's schedules onto MY round axis

The last step is why nothing downstream needs to know how deep the recursion went: a level hands its
parent finished, flat `(band, local_round)` flows, so `stitch.py` remains a two-level flattener.

WHAT THE GENERAL PATH DOES NOT DO. The crossbar solver encodes hard egress deadlines and schedules
them first. A general level solver has no way to express "this commodity must land before epoch 3" --
neither the LP nor the MILP formulation models release times or per-commodity deadlines -- so the
general path does not enforce them; it WARNS, and relies on `stitch.assert_bands_fit` to fail loud if
the resulting schedule really does overrun its band. That is a deliberate, user-approved limitation,
not an oversight: the premise of the whole band construction is that an inner fabric is much faster
than the outer one, and where that premise holds the deadline is met with 6-25x headroom.
"""
import copy
import logging
from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from teccl.hierarchy import crossbar_solve, ring_solve
from teccl.hierarchy.abstract import (abstract, coarsify_demand, lift_demand, set_level_chunk)
from teccl.hierarchy.bands import assign_bands
from teccl.hierarchy.crossbar_solve import IntraFlow
from teccl.hierarchy.problem import CoarseSolution, LevelDemand, LevelSolution, Subproblem
from teccl.hierarchy.reconstruct import (IdentityResolution, IntraCellDemand,
                                         assign_identities_free, assign_identities_preserving,
                                         build_child_problems, identity_sets)
from teccl.hierarchy.scale import ChunkScale
from teccl.hierarchy.flat_schedule import build_flat_schedule
from teccl.hierarchy.flatten import assert_bands_fit, derive_grid, rebase
from teccl.hierarchy.subtopology import induce
from teccl.solvers.demand import build_demand
from teccl.topologies.topology import Topology

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------
# Solver context: the things every level needs but none of them owns
# ------------------------------------------------------------------------------------------------
class LevelContext:
    """Cross-level state: how to reach a real solver, and the memo table.

    `memo` is what makes "memoized" mean memoization rather than a hard-coded case. A cluster of 32
    identical hosts poses the same sub-problem 32 times, and the key is the shape of that problem --
    the topology's capacity fingerprint plus the demand's shape -- not the cell id, so twins hit even
    though their GPU numbering differs.
    """

    def __init__(self, user_input=None, solve_flat_hook=None, debug: bool = False,
                 max_depth: int = 8, root_level_chunk: Optional[int] = None) -> None:
        self.user_input = user_input
        self.solve_flat_hook = solve_flat_hook
        self.debug = debug
        self.max_depth = max_depth
        # Force the root level's chunk unit `g` instead of taking the GCD of its coarse demands.
        # Only a driver reproducing pre-coarsening behaviour should set it (the `nocoarsen` flag).
        self.root_level_chunk = root_level_chunk
        self.root_g = 1
        self.memo: Dict[Tuple, List[IntraFlow]] = {}
        self.memo_hits = 0
        self.warned_deadlines = False

    def warn_unmodelled_deadlines(self, problem: Subproblem, hard: int) -> None:
        """Say once, loudly, that a general level is being handed deadlines it will not honour."""
        if self.warned_deadlines:
            return
        self.warned_deadlines = True
        logger.warning(
            "level depth=%d cell=%d band=%d carries %d hard (egress_stage) demand(s), but the "
            "general recursion does not model deadlines -- no formulation here supports release "
            "times or per-commodity deadlines. The schedule is checked after the fact by "
            "stitch.assert_bands_fit (rounds <= m); if that passes, every deadline was met with "
            "room to spare. Only the crossbar solver prioritizes them explicitly.",
            problem.depth, problem.cell_id, problem.band, hard)


def _memo_key(problem: Subproblem) -> Tuple:
    """A fingerprint of a sub-problem, so the same one is not solved twice.

    IDENTITIES ARE PART OF THE KEY, and that is not an oversight. It is tempting to drop them -- two
    hosts distributing different chunks over the same fabric in the same pattern really do have the
    same schedule up to relabelling, which would let 32 twin hosts share one solve. But the cached
    value is a list of IntraFlows with identities baked into them, so serving it to a twin hands
    that twin the other cell's DATA. Sharing across twins needs the flows relabelled through the
    isomorphism, and a key can only fingerprint an isomorphism, not produce one. Until that
    relabelling exists, the memo is restricted to exact repeats -- which still fire, because a level
    routinely poses the identical single-demand problem in band after band.
    """
    topo = problem.topology
    cap = tuple(tuple(row) for row in topo.capacity)
    shape = tuple(sorted((d.src_gpu, tuple(sorted(d.dst_gpus)), d.identity, d.kind,
                          round(d.volume, 9), d.deadline_epoch - problem.band)
                         for d in problem.demands))
    # The base-case ALGORITHM is part of the fingerprint. The same cell with the same demands has a
    # different schedule under the crossbar row than under the ring row, so a key without it would
    # serve one row's flows to the other -- which is exactly what an in-process A/B does (solve
    # once, flip `ring_solve.INTRA_ALGO`, solve again).
    return (cap, tuple(sorted(topo.switch_indices)), shape, problem.band, problem.cell_id,
            ring_solve.intra_algo())


# ------------------------------------------------------------------------------------------------
# The memoized rows of the base-case dispatch table
# ------------------------------------------------------------------------------------------------
class _MemoRow(NamedTuple):
    """One shape whose optimal schedule is known in closed form.

    A row owns BOTH halves of what a memoized level owes, because they are the same solver seen from
    the two dispatch points and must never disagree about which shape they claim:

        routing        this level's own piece + epoch decision, for step A (`solve_flat` / `_lower`)
        schedule_cell  the fine schedule for a BOTTOM cell's interior      (`_solve_base`)

    Rows are peers. Neither is "the general one" and neither is restricted to a particular depth: a
    row is picked purely by the shape of the graph in front of it, at whichever dispatch point asks.
    Where the lowering half cannot yet express what a row routed -- a hop over a direct cell-to-cell
    coarse link, or a delivery that has to be relayed onward by an intermediate cell -- that is a
    general limitation of the piece/slot machinery (deliberately deferred, see
    `reconstruct._build_slots` and `bands.band_of`), and it raises the same way whichever row
    produced the routing.
    """
    name: str
    matches: Callable[..., bool]          # (topology, cell_fabric: bool) -> bool
    routing: Callable[..., object]
    schedule_cell: Callable[..., List[IntraFlow]]


# ORDER IS SIGNIFICANT, and only for one case: with the algorithm flag set, `should_use_ring` claims
# a single-switch CELL that `is_crossbar` would claim too, so ring is tested first and the flag can
# take effect. On every other shape the two predicates are disjoint -- a crossbar has exactly one
# switch, a ring has none -- so the order is immaterial there.
_MEMOIZED_ROWS: Tuple[_MemoRow, ...] = (
    _MemoRow("ring", ring_solve.should_use_ring, ring_solve.ring_routing,
             ring_solve.schedule_cell),
    _MemoRow("crossbar", lambda topo, cell_fabric=False: crossbar_solve.is_crossbar(topo),
             crossbar_solve.crossbar_routing, crossbar_solve.schedule_cell),
)


def _memoized_row(topology, cell_fabric: bool = False) -> Optional[_MemoRow]:
    """Which closed-form row serves this graph, or None if it needs a real solver.

    `cell_fabric` distinguishes the two dispatch points -- a bottom cell's INTERIOR versus a LEVEL's
    graph -- and is passed to every row rather than interpreted here, so a row that does not care
    (the crossbar) simply ignores it. Both rows are offered both kinds of graph; neither is
    restricted to a depth.
    """
    for row in _MEMOIZED_ROWS:
        if row.matches(topology, cell_fabric):
            return row
    return None


# ------------------------------------------------------------------------------------------------
# The base case: solve one level's routing, by whichever solver its shape calls for
# ------------------------------------------------------------------------------------------------
def solve_flat(problem: Subproblem, ctx: LevelContext,
               mapping=None, demand_tensor=None) -> CoarseSolution:
    """Route this level's demand on this level's graph. The base-case dispatch table.

    Returns a `CoarseSolution` whatever ran, so the lowering half never learns which solver it was
    -- only whether the solver kept chunk identity, which decides the step-A variant.
    """
    topo = problem.topology
    if _memoized_row(topo) is not None:
        return CoarseSolution(per_chunk_flow_paths=None, topology=topo,
                              epoch_duration=topo.get_epoch_duration_fast_link(),
                              preserves_identity=True)
    if ctx.solve_flat_hook is not None:
        return ctx.solve_flat_hook(problem, ctx, mapping, demand_tensor)
    raise NotImplementedError(
        f"level depth={problem.depth} on a {len(topo.capacity)}-node topology is neither a "
        f"crossbar nor degenerate, so it needs a real formulation, but no solver hook was supplied. "
        f"Pass LevelContext(solve_flat_hook=gurobi_level_solver) -- see solve_hierarchical -- or "
        f"declare the level's structure with Cell.subcells so it decomposes instead.")


def gurobi_level_solver(problem: Subproblem, ctx: LevelContext, mapping, demand_tensor
                        ) -> CoarseSolution:
    """The true base case: hand this level to a real formulation via TECCLSolver.

    Imported lazily and used only when the shape needs it, so every Gurobi-free path -- which is all
    of them below the root on today's topologies -- stays Gurobi-free and locally testable.
    """
    from teccl.examples.hierarchy_pipeline import solve_on_topology

    if ctx.user_input is None:
        raise RuntimeError(
            "gurobi_level_solver needs a UserInputParams template (LevelContext(user_input=...)) to "
            "know the formulation, collective and objective this level should be solved under")
    topo = problem.topology
    if demand_tensor is not None:
        topo.demand_override = demand_tensor
    ui = copy.deepcopy(ctx.user_input)
    # num_chunks is irrelevant with a demand_override in place -- BaseFormulation reads the injected
    # tensor and sets num_chunks from it -- but a level below the root must not inherit the root's
    # output path, or each one would overwrite the last.
    ui.instance.num_chunks = 1
    if problem.depth > 0:
        ui.instance.schedule_output_file = ""
    solver = solve_on_topology(ui, topo)
    best = getattr(solver, "best_solver", None)
    if best is None:
        raise RuntimeError(f"level depth={problem.depth} produced no solved formulation")
    return CoarseSolution(per_chunk_flow_paths=best.per_chunk_flow_paths, topology=best.topology,
                          epoch_duration=best.epoch_duration, preserves_identity=False)


# ------------------------------------------------------------------------------------------------
# The recursion
# ------------------------------------------------------------------------------------------------
def solve_level(problem: Subproblem, ctx: LevelContext) -> LevelSolution:
    """Solve one level and, recursively, everything beneath it.

    Returns flows already flat in THIS level's `(band, local_round)`; the caller only has to rebase
    them onto its own axis. `pieces` is this level's own inter-cell traffic, which the caller
    interprets according to its depth (see LevelSolution).
    """
    if problem.depth > ctx.max_depth:
        raise RecursionError(
            f"hierarchical recursion exceeded max_depth={ctx.max_depth}; a cell's subcells almost "
            f"certainly do not shrink (check that Cell.subcells is a strict partition of a subset "
            f"of the parent's members)")
    if problem.is_degenerate():
        return LevelSolution(scale=problem.scale)

    if not problem.topology.cells:
        return _solve_base(problem, ctx)
    return _solve_recursive(problem, ctx)


def _solve_base(problem: Subproblem, ctx: LevelContext) -> LevelSolution:
    """A level with nothing left to collapse: schedule its demands on its own fabric directly.

    The crossbar row of the table is served by `schedule_cell`, which already owns a complete
    schedule for a whole cell -- including its own band split, which it must keep doing itself
    because its dedup and fan-out density test need to see the cell's demands all at once (see
    teccl/hierarchy/bands.py). So this hands it everything it was given and lets it place the work;
    the band the parent asked for is honoured because the demands carry the deadlines that produced
    it.
    """
    key = _memo_key(problem)
    cached = ctx.memo.get(key)
    if cached is not None:
        ctx.memo_hits += 1
        return LevelSolution(flows=_replay_memo(cached, problem), scale=problem.scale)

    cell = getattr(problem.topology, "cell", None)
    if cell is None:
        raise RuntimeError(
            f"level depth={problem.depth} reached the base case without a Cell view; the crossbar "
            f"solver needs the cell's gpu order and internal switch, so bottom cells must be "
            f"presented through _CellView")
    row = _memoized_row(problem.topology, cell_fabric=True)
    if row is None:
        # Not a shape with a closed form: this is where a real formulation would run on the cell's
        # own interior. It needs the tensor adapter and a second lowering pass, which is the one
        # branch the current topologies never reach.
        raise NotImplementedError(
            f"cell {problem.cell_id} at depth {problem.depth} has an internal fabric matching no "
            f"closed-form row ({[r.name for r in _MEMOIZED_ROWS]}): "
            f"{len(problem.topology.capacity)} nodes, {len(problem.topology.switch_indices)} "
            f"switches, and it declares no subcells, so it would need a formulation solve of its "
            f"own interior. Declare its structure with Cell.subcells so the recursion can "
            f"decompose it.")

    hard = sum(1 for d in problem.demands if d.kind == "egress_stage")
    if hard and problem.depth > 1:
        ctx.warn_unmodelled_deadlines(problem, hard)

    flows = row.schedule_cell(problem.cell_id, cell, problem.demands,
                              switch_copy=False, debug=ctx.debug,
                              subdivision=_subdivision_of(problem.scale),
                              topology=problem.topology)
    ctx.memo[key] = flows
    return LevelSolution(flows=flows, scale=problem.scale)


def _solve_recursive(problem: Subproblem, ctx: LevelContext) -> LevelSolution:
    """A level that declares cells: collapse, solve, lower, then recurse into each cell."""
    topo = problem.topology
    coarse, mapping = abstract(topo)
    lift_demand(mapping)

    # The root's demand arrives as a tensor (nothing has resolved anything yet); every level below
    # builds one from the IntraCellDemands its parent emitted, and keeps the relabelling table.
    level_demand = problem.level_demand
    if problem.depth and level_demand is None:
        level_demand = LevelDemand.from_demands(problem.demands, topo)
    fine_tensor = level_demand.demand if level_demand is not None else problem.root_tensor
    assert fine_tensor is not None, (
        f"level depth={problem.depth} has neither a root tensor nor demands to build one from")

    coarse_demand = coarsify_demand(fine_tensor, mapping)
    # `level_chunk` is only ever forced at the root, by a driver reproducing pre-coarsening
    # behaviour; below it the GCD of the level's own demands is the right answer by construction.
    coarse_demand, g, _level_scale = set_level_chunk(
        coarse, coarse_demand, scale=problem.scale,
        g=ctx.root_level_chunk if problem.depth == 0 else None)
    if problem.depth == 0:
        ctx.root_g = g

    solution = solve_flat(Subproblem(topology=coarse, demands=problem.demands,
                                     scale=problem.scale.coarsen(g), depth=problem.depth,
                                     band=problem.band, cell_id=problem.cell_id),
                          ctx, mapping=mapping, demand_tensor=coarse_demand)

    res = _lower(solution, problem, coarse, mapping, fine_tensor, g, level_demand)
    if solution.preserves_identity and res.subdivision != 1:
        raise AssertionError(
            f"an identity-preserving level refined by Q={res.subdivision}; it should spend nothing "
            f"from the refinement budget. Its assignments carried fractional volumes, which means "
            f"the routing step split a chunk -- see reconstruct.assign_identities_preserving.")

    # THIS LEVEL's grid. `m` does two jobs, for two different audiences: here it is the BOUND every
    # band folded into this level must respect, and on the returned LevelSolution it is the STRIDE
    # this level's own caller folds us with. Deriving it once, from this level's topology and epoch,
    # is what keeps those two readings the same number -- see teccl/hierarchy/flatten.py.
    delta, m = derive_grid(res.scale, topo, solution.epoch_duration)
    flows: List[IntraFlow] = []
    net_scale = res.scale

    by_cell: Dict[int, List[IntraCellDemand]] = defaultdict(list)
    for d in res.intra_demands:
        by_cell[d.cell].append(d)

    for cid in sorted(mapping.coarse_cells):
        cell = mapping.coarse_cells[cid]
        demands = by_cell.get(cid, [])
        if not demands:
            continue
        if not cell.subcells:
            # No further structure declared: the cell IS the bottom, so hand it the whole band set
            # at once. Splitting here would change the crossbar solver's dedup and density test,
            # which need the cell's demands all at once (see teccl/hierarchy/bands.py).
            child = Subproblem(topology=_CellView(topo, cell), demands=demands,
                               scale=res.scale, depth=problem.depth + 1, band=problem.band,
                               budget_rounds=m, cell_id=cid)
            sol = solve_level(child, ctx)
            flows += sol.flows
            net_scale = _max_refined(net_scale, sol.scale)
            continue

        # A genuinely nested cell: one independent sub-problem per band, because a child level's
        # round budget is meaningless without knowing which band it is spending.
        sub_topo = induce(topo, cell)
        for band, band_demands in sorted(assign_bands(demands).items()):
            child = Subproblem(topology=sub_topo, demands=band_demands, scale=res.scale,
                               depth=problem.depth + 1, band=band, budget_rounds=m, cell_id=cid)
            sol = solve_level(child, ctx)
            flows += rebase(sol, cid, band, sub_topology=sub_topo)
            net_scale = _max_refined(net_scale, sol.scale)

    # The feasibility certificate, applied AT EVERY LEVEL rather than only at the root. It is what
    # makes this level's `m` a legitimate stride for its caller: a band that overruns its epoch
    # would, once folded, overlap the next one. Failing here names the level and the cell, instead
    # of surfacing as an unattributable overrun in the final schedule.
    assert_bands_fit(flows, m, num_coarse_epochs=max((p.send_epoch for p in res.pieces),
                                                     default=-1) + 1)

    return LevelSolution(flows=flows, pieces=res.pieces, scale=net_scale,
                         resolution=res, epoch_duration=solution.epoch_duration,
                         delta=delta, rounds_per_epoch=m)


def _lower(solution: CoarseSolution, problem: Subproblem, coarse, mapping, fine_tensor,
           g: int, level_demand) -> IdentityResolution:
    """Steps A and B for this level: recover identity, then build the child problems.

    Which step-A variant runs is the ONLY thing that depends on which solver produced the routing,
    and it is decided by `preserves_identity` rather than by the solver's type -- so a new level
    solver plugs in by answering that one question and nothing else here changes.

    `level_demand` is None at the ROOT and carries the two translations every level below needs:
    `relabel` (tensor coordinate -> global identity) and `holders` (identity -> the node holding it
    here). At the root both are the identity map -- axis 0 IS the source GPU and the chunk axis IS
    its chunk index -- so passing None keeps the original `identity[0]` behaviour exactly, which is
    what makes the two-level path byte-identical.
    """
    relabel = level_demand.relabel if level_demand is not None else None
    holder = level_demand.holders if level_demand is not None else None
    if not solution.preserves_identity:
        assignments, targets, epoch = assign_identities_free(
            solution, mapping, fine_tensor, problem.topology, level_chunk=g,
            relabel=relabel, holder_of=holder)
    else:
        # The same row `solve_flat` picked, asked for the other half of what it owes. Re-deriving it
        # from `coarse` rather than threading it through keeps `_lower` a pure function of its
        # arguments, and the two calls cannot disagree because they test the identical predicate on
        # the identical graph.
        row = _memoized_row(coarse)
        assert row is not None, (
            "a solution claiming preserves_identity came from a graph that matches no closed-form "
            "row; only a memoized row sets that flag")
        id_sets, targets = identity_sets(fine_tensor, mapping, relabel=relabel)
        carried = row.routing(coarse, mapping, id_sets)
        assignments = assign_identities_preserving(
            carried, holder or {}, targets, mapping, problem.topology, solution.epoch_duration)
        epoch = solution.epoch_duration

    return build_child_problems(assignments, targets, mapping, fine_tensor, problem.topology,
                                epoch, scale=problem.scale, relabel=relabel)


# ------------------------------------------------------------------------------------------------
# Small scaffold helpers (the per-layer flatten itself lives in teccl.hierarchy.flatten)
# ------------------------------------------------------------------------------------------------
def _replay_memo(flows: Sequence[IntraFlow], problem: Subproblem) -> List[IntraFlow]:
    """Return a cached schedule. A hit is an EXACT repeat (see `_memo_key`), so there is nothing to
    translate -- the guard is here to fail loudly rather than silently mis-attribute data if the key
    is ever loosened to share across twins without also relabelling the flows."""
    if not flows:
        return []
    assert flows[0].cell == problem.cell_id, (
        f"memo hit returned a schedule for cell {flows[0].cell} while solving cell "
        f"{problem.cell_id}; the key was loosened without adding the identity relabelling that "
        f"cross-cell sharing requires")
    return list(flows)


def _max_refined(a: Optional[ChunkScale], b: Optional[ChunkScale]) -> Optional[ChunkScale]:
    """The net scale of a subtree is the finest any of its branches needed.

    Refinement is resolved BOTTOM-UP: a Q discovered three levels down still has to be paid at the
    root, because `refinement_from_root` is what reaches ncclize's `chunk_up()` and MAX_M bounds the
    product over every level (see teccl/hierarchy/scale.py).
    """
    if a is None:
        return b
    if b is None:
        return a
    return a if a.refinement_from_root >= b.refinement_from_root else b


def _subdivision_of(scale: Optional[ChunkScale]) -> int:
    """The Q in force at this level, as the crossbar solver's `_coalesce_subchunks` wants it."""
    if scale is None:
        return 1
    r = scale.refinement_from_root
    return int(r) if r.denominator == 1 and r >= 1 else 1


class _CellView:
    """A read-only Topology-shaped view of one BOTTOM cell, in the PARENT's index space.

    Deliberately not `induce`. The crossbar solver works in global indices -- its `Cell` carries
    global gpu ids, and the flows it emits go straight to the stitch, which is global too -- so
    renumbering a level that has no sub-levels to renumber for would only mean translating every
    flow back on the way out. This presents the cell's own capacity view so `is_crossbar` can judge
    it, and leaves the indices alone.

    Only the handful of attributes the dispatch and the crossbar solver touch are provided. It is a
    view rather than a Topology subclass precisely so it cannot be handed to a formulation by
    accident: a formulation would index the capacity matrix densely and read the parent's other
    nodes as real participants.
    """

    def __init__(self, parent: Topology, cell) -> None:
        self.cell = cell
        members = set(cell.members)
        n = len(parent.capacity)
        self.capacity = [[parent.capacity[i][j] if (i in members and j in members) else 0.0
                          for j in range(n)] for i in range(n)]
        self.switch_indices = list(cell.internal_switches)
        self.passive_indices = [i for i in range(n) if i not in members]
        self.chunk_size = parent.chunk_size
        self.cells = []

    def get_epoch_duration_fast_link(self) -> float:
        best = max((c for row in self.capacity for c in row), default=0.0)
        return self.chunk_size / best if best > 0 else 0.0


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------
def write_side_outputs(res: IdentityResolution, flows: Sequence[IntraFlow], prefix: str) -> None:
    """Serialize the intermediate artifacts the drivers have always written next to the schedule.

    `Schedules/{prefix}_identities.json` is the resolved inter-cell traffic plus the child problems
    it produced; `_intra.json` is the assembled sub-level schedule. Neither feeds anything -- the
    flat schedule is self-contained -- but they are how a bad run is diagnosed without re-solving,
    which on the coarse LP is the expensive part.
    """
    import json
    import os
    from dataclasses import asdict

    os.makedirs("Schedules", exist_ok=True)
    with open(f"Schedules/{prefix}_identities.json", "w") as f:
        json.dump({
            "pieces": [asdict(p) for p in res.pieces],
            "intra_demands": [asdict(d) for d in res.intra_demands],
            # to_json(), not asdict(): the scale holds exact Fractions, which asdict passes through
            # raw and no JSON encoder can write.
            "scale": res.scale.to_json() if res.scale else None,
            "subdivision": res.subdivision,
        }, f, indent=2, default=list)

    with open(f"Schedules/{prefix}_intra.json", "w") as f:
        json.dump([dict(cell=x.cell, identity=list(x.identity), sender=x.sender,
                        receiver=x.receiver, via_switch=x.via_switch, volume=x.volume,
                        band=x.band, local_round=x.local_round, span=x.span,
                        identities=[list(i) for i in x.identities], kind=x.kind, hard=x.hard)
                   for x in flows], f, indent=2)


def solve_hierarchical(topology: Topology, user_input, collective, num_chunks: int,
                       prefix: str = "hierarchical", fine_demand=None,
                       solve_flat_hook=None, debug: bool = False,
                       report=None, level_chunk: Optional[int] = None,
                       write_outputs: bool = False) -> Tuple[Dict, LevelSolution]:
    """Solve `topology` hierarchically and return `(flat schedule dict, LevelSolution)`.

    The returned schedule is an ORDINARY flat schedule on the fine topology -- ncclize consumes it
    with no hierarchy awareness, and `check_implements()` independently verifies that it really
    implements the collective. That is the point of the stitch: however many levels ran, the output
    is indistinguishable from a flat solve's.

    fine_demand:     pre-built demand tensor; defaults to `build_demand(collective, topology,
                     num_chunks)`. Drivers that must match a flat run's resolution exactly (alltoall
                     pre-scales by the participating-GPU count) pass their own.
    solve_flat_hook: how to reach a real formulation. Defaults to `gurobi_level_solver`; pass a stub
                     to keep a run Gurobi-free.
    report:          optional callback(res, flows, fine, coarse_epoch) for the drivers' narration.
    """
    assert topology.cells, (
        "topology declares no cells, so there is no hierarchy to solve. Override "
        "Topology.build_hierarchy() (see teccl.hierarchy.Cell) or run the flat solver.")

    if fine_demand is None:
        fine_demand = build_demand(collective, topology, num_chunks)

    ctx = LevelContext(user_input=user_input,
                       solve_flat_hook=solve_flat_hook or gurobi_level_solver,
                       debug=debug, root_level_chunk=level_chunk)

    root = Subproblem(
        topology=topology, demands=[],
        # The root scale is the un-refined one: one chunk of the topology's own chunk_size per fine
        # chunk index. Every refinement below is recorded against it, and the net is what reaches
        # ncclize's chunk_up().
        scale=ChunkScale(bytes_per_chunk=topology.chunk_size,
                         num_chunks=len(fine_demand[0][0]) if len(fine_demand) else 1),
        depth=0, root_tensor=fine_demand)

    solution = solve_level(root, ctx)
    res, coarse_epoch = solution.resolution, solution.epoch_duration
    if res is None:
        raise RuntimeError("the root level produced no resolution; nothing to flatten")

    if report is not None:
        report(res, solution.flows, topology, coarse_epoch)
    if write_outputs:
        write_side_outputs(res, solution.flows, prefix)

    # Hand the final pass the ROOT's own grid rather than letting it re-derive one, so the absolute
    # axis it lays out is provably the axis the recursion folded onto.
    info, _records = build_flat_schedule(
        res, solution.flows, topology, fine_demand, coarse_epoch,
        getattr(collective, "name", str(collective)).lower(),
        grid=(solution.delta, solution.rounds_per_epoch))
    if write_outputs:
        import json
        with open(f"Schedules/{prefix}_flat.json", "w") as f:
            json.dump(info, f, indent=2, sort_keys=True)
        print(f"[hierarchy] flat schedule written to Schedules/{prefix}_flat.json")
    if ctx.memo_hits:
        print(f"[hierarchy] {ctx.memo_hits} sub-problem(s) served from the memo table "
              f"({len(ctx.memo)} distinct shapes solved)")
    return info, solution
