"""
The band policy: which coarse epoch of the level above a piece of intra-level work runs under.

A BAND is the unit that makes the recursion's timing compose. Each level's EPOCHS become the next
level's BANDS: level-0 coarse epoch k is level-1 band k, level-1's m rounds inside that band are
level-2's bands, and so on. Because a band is pinned to the network send at its leading edge, a
level's whole job is "finish this band's work inside the m rounds the band is worth", which is a
plain makespan problem with no deadlines in it -- and that is precisely what lets an arbitrary
solver serve as a level (see teccl/hierarchy/solve.py).

The policy lives HERE, in a module that neither imports the other two, because it has exactly two
call sites that must never disagree:

  * `crossbar_solve._assign_bands` applies it to `_Job`s, AFTER `_to_jobs` has done its fan-out
    lowering, its dedup and its density test. Those three steps need to see a cell's demands ALL AT
    ONCE -- dedup merges an egress relay with a self_distribution that happen to be the same send,
    and the DIRECT-vs-TREE density test compares a source's send load against the whole cell's max
    recv load -- so the crossbar solver cannot be handed one band at a time without changing its
    output.
  * `assign_bands` applies it to `IntraCellDemand`s, at the LEVEL BOUNDARY, before a child
    sub-problem is constructed. The general recursion does need the split up front, because a child
    level's `budget_rounds` is meaningless without knowing which band it is solving.

Same rule, two granularities. Keeping `band_of` as the single implementation is what stops the two
from drifting.
"""
from collections import defaultdict
from math import inf
from typing import Dict, List, Sequence

from teccl.hierarchy.reconstruct import IntraCellDemand, demand_is_hard

# The band that runs BEFORE coarse epoch 0's network sends. Unlike bands 0..K-1 it is not
# concurrent with any network traffic, so its width is whatever its own schedule needs rather than
# a full coarse epoch -- and it is the one band whose length is charged directly to the makespan.
PROLOGUE_BAND = -1


def band_of(release_gap: int, deadline_gap: float, hard: bool, what: str = "work") -> int:
    """Place one unit of work in the band where its data becomes READY -- as early as possible, not
    as late as its deadline allows.

    Readiness is the natural placement because it needs no lookahead: the inputs exist or they do
    not. `release_gap` already carries it (PROLOGUE_BAND for anything sourced from native data,
    arrival+1 for a fan-out of a network arrival, since a piece lands at the END of its arrival
    epoch).

    The one exception is the PROLOGUE. Work that is ready in band b but must COMPLETE before band
    b's network sends has nowhere to go inside b -- the sends sit at its leading edge -- so it moves
    to b-1. For the only case that occurs today, egress staging that feeds coarse epoch 0, that is
    band -1: work that happens before the collective's first network send. It is empty whenever
    gateways own the data they send (rail-optimized) and non-empty exactly where the boundary forces
    a relay first (the hetero cluster).

    This replaced an earlier deadline-gap policy for hard work. Deadline-pinning made `band`
    ambiguous downstream -- an epoch-0 staging relay and a self_distribution both landed in band 0
    while belonging on opposite sides of the first network send -- and it delayed staging for no
    gain, since the data was available all along.
    """
    # As early as ready, but not into the prologue merely because the data was always there -- the
    # prologue is reserved for work a deadline actually forces before the collective's first network
    # send, which is what keeps it empty on topologies with no forced relay.
    band = max(int(release_gap), 0)
    if hard and deadline_gap != inf and band >= deadline_gap:
        band = int(deadline_gap) - 1
        if band < release_gap:
            raise RuntimeError(
                f"{what} cannot run before band {int(release_gap)} but must complete before band "
                f"{int(deadline_gap)}; there is no band it can run in. For a transit forward this "
                f"means the coarse solve gave the cell a one-epoch dwell but the relay between "
                f"its two NICs needs two -- see cell_relay_design.md §6 (the per-cell forwarding "
                f"dwell). Identity resolution normally refuses such a route before it gets here, "
                f"so reaching this point means a caller built the demand directly.")
    return band


def release_of(demand: IntraCellDemand) -> int:
    """The first band a demand's data could possibly move in.

    A network piece lands at the END of its arrival epoch, so the first band that can fan it out is
    the next one. Anything sourced from data the cell already holds -- a native chunk being staged
    to a gateway, or a self_distribution -- exists before the collective even starts, which is what
    makes the prologue available to it.

    An explicit `release_band` overrides both rules. It exists for the TRANSIT forward (a route's
    middle leg): that demand is an `egress_stage`, so the kind-based rule would hand it the
    prologue, but its data is not in the cell until the inbound leg arrives. The kind says what the
    cell DOES with the data; only the route knows when the data got there.
    """
    if demand.release_band is not None:
        return demand.release_band
    if demand.kind == "ingress_distribution":
        return demand.deadline_epoch + 1
    return PROLOGUE_BAND


# Hardness is defined beside the field it reads (reconstruct.demand_is_hard) because this module
# imports that one, not the other way round -- but it is band policy, so it is named here too and
# every call site inside the band machinery goes through this name.
is_hard = demand_is_hard


def assign_bands(demands: Sequence[IntraCellDemand]) -> Dict[int, List[IntraCellDemand]]:
    """Split a cell's demands into one list per band -- the level boundary's view of the policy.

    `egress_stage` is the hard kind: a network send is waiting on it, so missing its band slips the
    level above. Everything else is soft (an `ingress_distribution` may be promoted by a caller for
    a transit cell, which is why `hard` is read off the demand rather than assumed from the kind).
    """
    by_band: Dict[int, List[IntraCellDemand]] = defaultdict(list)
    for d in demands:
        hard = is_hard(d)
        deadline = d.deadline_epoch if hard else inf
        what = f"{d.kind} {d.src_gpu}->{d.dst_gpus} (identity {d.identity})"
        by_band[band_of(release_of(d), deadline, hard, what)].append(d)
    return dict(by_band)
