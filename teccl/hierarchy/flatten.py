"""
The PER-LAYER flatten: each level's time grid, and how a finished child is folded into its caller.

Every level of the recursion runs this, once per child. Its counterpart is
`teccl.hierarchy.flat_schedule`, which runs ONCE at the very end on the assembled result -- the two
were one module for as long as the solver had only two levels, at which point "fold the child in"
and "lay the whole thing on the absolute axis" were the same step. They are not, and separating them
is what makes the depth of the recursion invisible to everything downstream.

THE GRID IS PER LEVEL, AND ITS TWO NUMBERS DO DIFFERENT JOBS.

`derive_grid` gives a level `(delta, m)`: `delta` is the duration of one ROUND -- the finest thing
that level's children can do, one chunk across the fastest link inside one of its cells -- and
`m = level_epoch / delta` is how many of those rounds one of THIS level's epochs is worth. So `m` is
a unit conversion between adjacent levels, and every level owns exactly one.

Then `rebase` needs two of them, from two different levels:

    the CHILD's m   is the STRIDE. One child epoch occupies that many rounds on the output axis,
                    which is what makes a rack transfer that physically takes 4 NVLink rounds
                    occupy 4 of them rather than 1.
    THIS level's m  is the BOUND. Everything folded into one of this level's bands has to fit
                    inside it -- `assert_bands_fit`.

Getting this wrong is subtle in a specific way worth recording, because the code did it for a while.
`rebase` used to MEASURE its stride as `max(local_round + span)` over the child's own flows. That
conflates the unit conversion with the feasibility check: a window sized to its own contents can
never overflow, so the bound became true by construction and the conversion became wrong by
construction -- it knows nothing about link speed. On a rack fabric 4x slower than NVLink it
returned 1 where the answer was 4, and the recursion's ORDERING stayed verified while its DURATION
was silently optimistic by that factor. The stride is a property of the child's clock, so it must
come from the child's own `derive_grid`; the bound is a real property that has to be asserted.
"""
import logging
import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from teccl.hierarchy.bands import PROLOGUE_BAND
from teccl.hierarchy.crossbar_solve import IntraFlow, band_rounds, rounds_in
from teccl.hierarchy.scale import ChunkScale
from teccl.topologies.topology import Topology


# ----------------------------------------------------------------------------------------------
# The level's time grid
# ----------------------------------------------------------------------------------------------
def intra_link_bandwidth(fine_topology: Topology, cells=None) -> float:
    """The bandwidth the INTRA level's epoch is measured against: its own link set.

    A level's epoch is "one chunk (at that level's size) on a link of that level's graph", and the
    intra level's graph is a cell's internal fabric -- not the whole fine topology. For a
    single-NVSwitch cell every internal link is identical, so fastest and slowest coincide and the
    user's EpochType has nothing to choose between; that is why this returns one number rather than
    taking a selector. Restricting to the cell's own links is what makes that true by construction
    instead of by coincidence: `max` over the whole fine topology happens to land on the NVSwitch
    in both current topologies, and would silently pick an unrelated link in one where it does not.

    Falls back to the whole-topology max when the topology declares no cells (a flat schedule being
    replayed through the stitch), which is the historical behaviour.
    """
    cells = cells if cells is not None else getattr(fine_topology, "cells", None)
    caps = set()
    for cell in cells or []:
        members = set(cell.members)
        for i in members:
            for j in members:
                c = fine_topology.capacity[i][j]
                if c > 0:
                    caps.add(c)
    if not caps:
        return max(max(row) for row in fine_topology.capacity)
    if len(caps) > 1:
        # Not fatal -- pick the fastest, matching the historical reading -- but say so, because the
        # "a round is one chunk on a port" identity the whole band arithmetic rests on assumes the
        # ports are interchangeable.
        logging.warning(
            "intra-cell links are not uniform (%s); the fine epoch is sized to the fastest, so a "
            "round on a slower internal link takes more than one fine epoch", sorted(caps))
    return max(caps)


def derive_grid(scale: ChunkScale, fine_topology: Topology, coarse_epoch: float,
                max_refinement: int = 128) -> Tuple[float, int]:
    """The fine epoch duration and how many of them a coarse epoch holds.

    Both are DERIVED from the live ChunkScale, never written down: refinement changes the chunk
    size, and delta and m must move with it or the round count and the epoch grid silently desync.

    m is FLOORED and delta then back-solved as Delta/m, rather than requiring Delta/delta to come
    out whole. The fine grid must divide the coarse epoch exactly -- every band deadline is stated
    in fine epochs -- but nothing requires the natural chunk-crossing time to be the divisor. The
    old form asserted integrality and passed only because 1800/50 = 36 exactly on both current
    topologies; it already failed for a coarse FASTEST_LINK epoch (0.0025/delta = 4.5). Flooring
    makes delta >= the natural time, so an intra round still fits inside one fine epoch, and it
    keeps m*K*delta == K*Delta exactly, so the network half of the makespan is unrounded.
    """
    if scale is None:
        raise ValueError("IdentityResolution has no ChunkScale; cannot derive the fine epoch grid")
    if scale.refinement_from_root > max_refinement:
        raise ValueError(
            f"cumulative chunk refinement {scale.refinement_from_root} exceeds the budget "
            f"{max_refinement}: ncclize's chunk_up() would have to expand every chunk that finely. "
            f"The budget is shared by every level of the recursion.")
    natural = scale.epoch_duration(intra_link_bandwidth(fine_topology))
    m = int(math.floor(coarse_epoch / natural + 1e-9))
    if m < 1:
        raise AssertionError(
            f"a coarse epoch ({coarse_epoch}) is shorter than one intra round ({natural}): the "
            f"inner fabric is not faster than the outer one, so intra work cannot hide under "
            f"network time and the whole per-band timing premise fails here.")
    return coarse_epoch / m, m



# ----------------------------------------------------------------------------------------------
# Which bands are bounded by m, and the certificate that they fit
# ----------------------------------------------------------------------------------------------
def aligned_band(band: int, num_coarse_epochs: int) -> bool:
    """Is this band pinned to a network send, and therefore bounded by m?

    Bands 0..K-1 are: their leading edges ARE the sends, which are Delta apart by construction. The
    PROLOGUE and the EPILOGUE are not -- they have no send to align to, are as wide as their own
    schedules need, and are charged directly to the makespan instead. Anything checking "does this
    band fit in m" must ask this first, or it reports a long prologue as an overrun when the
    prologue was never claimed to fit.
    """
    return 0 <= band < num_coarse_epochs


def assert_bands_fit(intra_flows: Sequence[IntraFlow], m: int, num_coarse_epochs: int) -> None:
    """The feasibility certificate: a band aligned to a coarse epoch must fit inside it.

    Bands 0..K-1 are pinned to the network sends at their leading edges, so their width is fixed at
    m. If a cell needs more rounds than that, its intra work does NOT hide under network time and
    the whole premise (inner fabric much faster than outer) fails for that band -- the windowed
    intra solver would be the fallback. The prologue and epilogue are exempt: they have no send to
    align to and are charged their true length instead.
    """
    over = [(c, b, r) for (c, b), r in sorted(band_rounds(intra_flows).items())
            if aligned_band(b, num_coarse_epochs) and r > m]
    if over:
        raise AssertionError(
            f"intra-cell work does not fit the coarse epoch it runs under, on {len(over)} "
            f"(cell, band) pairs [(cell, band, rounds)] with m={m}: {over[:6]}")



# ----------------------------------------------------------------------------------------------
# Folding a finished child into the calling level
# ----------------------------------------------------------------------------------------------
def to_this_levels_indices(flows: Sequence[IntraFlow], sub_topology) -> List[IntraFlow]:
    """Undo `induce`'s renumbering: a CHILD's node ids -> the calling level's node ids.

    A child level is solved on a topology renumbered 0..n-1 (a `Topology` is dense by contract), so
    every node id in the flows it returns is in the CHILD's index space. Exactly one translation is
    owed per level, applied here on return, and because each level translates only its OWN children
    the compositions telescope: a flow from three levels down arrives at the root having been mapped
    through three tables, ending in global fine ids.

    Identities are deliberately untouched -- they are global at every depth, which is the whole
    reason this is a node-index problem and not an everything problem (see problem.py).
    """
    l2g = getattr(sub_topology, "local_to_global", None)
    if l2g is None:
        return list(flows)
    return [replace(f, sender=l2g[f.sender], receiver=l2g[f.receiver],
                    via_switch=l2g[f.via_switch]) for f in flows]


def rebase(sol, cell_id: int, band: int, sub_topology=None) -> List[IntraFlow]:
    """Translate a finished CHILD level into THIS level's coordinates. The level boundary, going up.

    Vocabulary, because it is easy to get turned around: THIS LEVEL is whoever called `rebase`;
    CHILD is the sub-level whose `LevelSolution` is being translated; `band` is a band on THIS
    LEVEL's epoch axis. (When this level's own parent later calls `rebase` on this level's output,
    what is "this level's axis" here will be "the child's axis" there -- the same function, one
    level up, which is how the translations telescope down to the root.)

    In:  `sol` in the CHILD's numbering -- node ids in the child's 0..n-1 space (`induce` renumbered
         them), `sol.flows` indexed by the child's band, `sol.pieces` by the child's epoch.
    Out: `IntraFlow`s in THIS LEVEL's numbering -- this level's node ids, `f.band` uniformly the
         `band` argument, `f.local_round` an offset within it.

    Identities are deliberately NOT translated: they are global `(fine GPU, fine chunk)` at every
    depth, so only a flow's address and its clock ever need fixing.

    THE CHILD RETURNS TWO ARTIFACTS ON TWO AXES and they must end up on ONE. Its own children's
    flows are indexed by the child's band; its inter-subcell pieces are indexed by the child's
    epoch. Both are folded onto a single round axis with stride `w = sol.rounds_per_epoch` -- THE
    CHILD'S OWN `m`, from the child's own `derive_grid`, never measured here (see the module
    docstring for what measuring it cost).

    Child band b' gets the window `[(b'+1)*w, (b'+2)*w)`; the epoch-b piece is placed at its own
    window's LEADING EDGE `(b+1)*w`, on that same axis. Two properties fall out:

      * ORDERING. Band b' completes by `(b'+2)*w`, and `band_of` guarantees hard work lands at
        band <= deadline-1, so the staging feeding an epoch-b piece is at some `b' <= b-1` and
        `(b'+2)*w <= (b+1)*w`. Equality at `b' = b-1` is fine: the piece departs exactly as the
        window that fed it closes, and back_trace's test is `held > epoch`. Putting the pieces on a
        DIFFERENT axis from the flows -- the flows shifted up one band for the prologue, the pieces
        at raw epoch -- is what broke this before: a piece then sat one round earlier than its own
        band, i.e. on top of the very staging it depended on.
      * DURATION. A child epoch occupies `w` rounds of real time, not one.

    The `+1` shift makes room for the child's PROLOGUE, which by definition precedes its epoch-0
    send and so occupies `[0, w)`.
    """
    flows = to_this_levels_indices(sol.flows, sub_topology)
    pieces = to_this_levels_indices(sol.pieces_as_flows(cell_id, band), sub_topology)

    w = sol.rounds_per_epoch
    assert w and w >= 1, (
        f"child of cell {cell_id} reports rounds_per_epoch={w!r}; the stride must come from the "
        f"child's own derive_grid, and a level that produced flows always has one")
    # The stride is only a valid window if the child's work actually fits inside one of its own
    # epochs. Measuring the stride used to make this true by definition; now it is a real claim, and
    # it is the same "inner fabric is faster than the outer" certificate `assert_bands_fit` applies,
    # asserted here at the level where it can still name the culprit.
    busiest = max(band_rounds(flows).values(), default=0)
    assert busiest <= w, (
        f"child of cell {cell_id} needs {busiest} rounds in its busiest band but one of its epochs "
        f"is only worth {w}; its own fabric cannot absorb the work its level scheduled, so folding "
        f"it into a band of this level would overlap two of its epochs")

    out = [replace(f, band=band, local_round=(f.band - PROLOGUE_BAND) * w + f.local_round)
           for f in flows]
    # `pieces_as_flows` already put the child's epoch in `local_round`; place it at that epoch's
    # window leading edge, on the same axis the flows above were just mapped onto.
    out += [replace(p, band=band, local_round=(p.local_round + 1) * w) for p in pieces]
    return out
