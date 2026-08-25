"""
Phase-3 identity resolution for the LP hierarchical path.

The coarse LP is IDENTITY-FREE: it routes continuous aggregate volume per source CELL, not per
fine chunk. `coarsify_demand` collapsed each cell U's distinct fine identities into one aggregate
commodity per destination -- collective-agnostically, coarse[U][V] counts the fine identities of
U that some GPU in V wants -- and the coarse solve delivered exactly that many units of it to V.
Before phase-3 intra-cell reconstruction can run we must decide WHICH fine chunk identity each
unit of that aggregate carries. That is identity resolution, and it is driven entirely by the
demand shape (both the fine demand array and the coarse demand built from it are in hand), so it
works for ANY collective, not just AllGather.

For each ordered cell pair (U, V) this is a min-cost transportation problem: assign U's identities
(supply 1 each, since V must receive each exactly once) to the delivered egress volume. The
assignment is JOINT over both ends of the inter-cell hop and its objective is LEXICOGRAPHIC:

  1. minimize forced EGRESS relay -- an identity leaving via a gateway GPU that is not its native
     source GPU must first be relayed to that gateway inside U;
  2. then minimize forced INGRESS relay -- an identity landing on a boundary GPU of V that does not
     want it must be relayed onward inside V;
  3. then keep relayed identities off the earliest egress epochs (a relay feeding a coarse-epoch-0
     egress has to finish before ANY network send, which forces a staging prologue).

The ordering is strict, not a weighted trade: egress relays are HARD (deadline = the network send
epoch) while ingress relays are SOFT, and egress sits upstream of the network hop. Equal weights
would let the solver swap a native egress for an ingress saving at no apparent cost. Tier 3
replaces what used to be a hand-rolled post-hoc sort. See _solve_assignment.

Which fine GPU a piece LANDS on is then chosen under a global per-epoch fine downlink budget
(_pick_ingress), because a coarse link's capacity is the SUM of the fine downlinks behind it: left
unbounded, one fine downlink can be oversubscribed while a sibling idles, making the fine schedule
infeasible against the very coarse solution it implements.

Finally the chunk is REFINED so that every emitted volume is a whole sub-chunk (_snap_volumes ->
_subdivision_factor / _emit_refined). Fractional volumes are intrinsic here -- they come from the
coarse LP relaxation splitting a commodity across parallel paths, and from the abstraction summing
several fine links into one coarse link -- and refining at this boundary is what keeps every
downstream volume-merging step (this module's _coalesce_egress, crossbar_solve's _add_direct) from
having to reason about disjoint byte ranges it cannot represent. The resulting granularity is
reported on IdentityResolution.scale; see teccl/hierarchy/scale.py for why it must be threaded
rather than read off the topology.

_snap_volumes is the float -> exact BOUNDARY of that refinement, and the only tolerant step in the
lowering half: above it everything is float and noisy (the coarse LP, the slot split, the scipy
assignment), below it everything is exact `Fraction` arithmetic. The tolerance is sized against the
solver that produced the volumes rather than hardcoded (see snap_tolerance and grid_resolution --
MAX_DENOM and the solver's feasibility_tol are coupled, not independent constants).

Output (pure data, no Gurobi handles): resolved inter-cell pieces carrying a concrete fine chunk
identity on real GPUs/links, plus intra-cell demand descriptors (egress staging, ingress
distribution, self distribution) for the downstream phase-3 solve. This module STOPS before the
phase-3 intra-cell solve and the final flat stitching.

See the design note hierarchical_lp_identity_resolution / hierarchical_phase3_forward_plan.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import floor, gcd, lcm
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from teccl.hierarchy.abstract import HierarchyMapping
from teccl.hierarchy.scale import ChunkScale
from teccl.topologies.topology import Topology

# A fine data identity: (fine source GPU s, fine chunk index ci). This is already the global
# identity used across collectives -- the source index and chunk index together name the data.
Identity = Tuple[int, int]

EPS = 1e-6

# Largest denominator the identity-resolution grid represents; mirrors ncclize's MAX_DENOM
# (teccl/ncclize/teccl_ncclize.py). Defined here, ahead of _pick_ingress, because it is a default
# parameter value evaluated at function-definition time, not inside a function body where a later
# definition would still be visible by call time.
#
# NOT safe to change here as a blanket global bump: MAX_DENOM and GurobiParams.feasibility_tol are
# coupled through grid_resolution = 1/(2*MAX_DENOM^2) (see snap_tolerance), and this codebase's
# feasibility_tol DEFAULT is 1e-4 (teccl/input_data.py), which is only legal up to MAX_DENOM~70 --
# several existing hierarchical examples rely on that default. A single instance that needs a
# larger grid (e.g. a scattered dual-plane host's uneven 6/5/5 GPU-per-leaf split driving a
# proportional egress-split denominator, _build_slots, up to 125) should set
# InstanceParams.max_denom instead of raising this constant -- see resolve_max_denom, which is
# what every grid-sized comparison in this module actually reads.
MAX_DENOM = 64


@dataclass(frozen=True)
class ResolvedPiece:
    """One inter-cell egress piece with a concrete fine identity pinned onto real GPUs/links."""
    src_cell: int
    dst_cell: int
    identity: Identity                    # (s, ci); serialized as "for chunk {ci} from {s}"
    egress_gpu: int                       # fine GPU in src_cell that physically sends (== s iff no relay)
    ingress_gpu: int                      # fine GPU in dst_cell that physically receives
    via_switches: Tuple[int, ...]         # FINE switch ids along the coarse path
    volume: float
    send_epoch: int                       # coarse epoch it leaves src_cell
    arrival_epoch: int                    # coarse epoch it is consumed at dst_cell
    # Physical transmission rate in GB/s, or None for an unpaced flow. Set by THIS level
    # (see _piece_rate): the level that scheduled a flow is the only one that knows the
    # epoch it was paced against, so it computes the rate rather than leaving a consumer
    # to re-derive one from a single global epoch duration.
    rate: Optional[float] = None


@dataclass(frozen=True)
class IntraCellDemand:
    """A demand phase-3 must resolve inside a single cell.

    kind:
      "egress_stage"        native GPU -> gateway GPU, so an identity can leave on a non-native
                            uplink (dst_gpus is the single gateway).
      "ingress_distribution" gateway GPU -> the fine GPUs inside the cell that actually want the
                            identity (all of the cell for AllGather, one GPU for AllToAll).
      "self_distribution"   source GPU -> wanting GPUs, for demand whose source and destination
                            are BOTH inside this cell (dropped from the coarse graph). Structural,
                            independent of identity resolution.

    An `egress_stage` at a TRANSIT cell (a coarse route's middle leg: the cell neither produces
    nor wants the identity, it forwards it) is the one case that needs a release AND a deadline to
    be independent numbers. Everywhere else one of the two is implied by the kind -- native data is
    available from the prologue, an ingress fan-out is released the epoch after it lands -- so
    `deadline_epoch` alone has been enough. A forwarding relay is bounded on both sides: it cannot
    start before the inbound leg ARRIVES, and it must finish before the outbound leg SENDS. Hence
    the two optional overrides below; both default to None, meaning "use the kind's rule", which is
    what every non-transit demand does and why nothing else had to change.
    """
    cell: int
    kind: str
    identity: Identity
    src_gpu: int
    dst_gpus: Tuple[int, ...]
    volume: float
    deadline_epoch: int                   # egress: <= send_epoch of fed piece; ingress: > arrival_epoch
    # First band this demand's data could move in. None => bands.release_of's kind-based rule.
    # Set only for a transit forward, where the data is not present until its inbound leg lands.
    release_band: Optional[int] = None
    # Whether missing the deadline slips the level above. None => the kind's default
    # (egress_stage is hard, fan-out is soft). Read via `demand_is_hard`, never bare `getattr`:
    # the field EXISTS now, so a `getattr(d, "hard", default)` would read None and silently make
    # every egress_stage soft.
    hard: Optional[bool] = None


def demand_is_hard(demand: IntraCellDemand) -> bool:
    """Whether missing this demand's deadline slips the level above.

    `egress_stage` is the hard kind: a network send is waiting on it. Everything else is soft,
    absorbed by the intra fabric's slack. A demand may override either way (`hard=`), which is why
    this reads the field first rather than deciding from the kind alone.

    Read hardness through here rather than `getattr(d, "hard", <default>)`. The attribute now
    always EXISTS (defaulting to None), so a getattr default is dead code and None -- falsy --
    would silently demote every egress_stage to soft, dropping the deadline that keeps a staging
    relay ahead of the send it feeds.

    It lives HERE, beside the field, rather than in `bands.py` with the rest of the band policy:
    `bands` imports this module, so anything `bands` owns is unavailable to the emitters that
    build these records. `bands.is_hard` re-exports it, so the band policy still reads as one
    module from its own call sites.
    """
    if demand.hard is not None:
        return demand.hard
    return demand.kind == "egress_stage"


@dataclass
class IdentityResolution:
    pieces: List[ResolvedPiece] = field(default_factory=list)
    intra_demands: List[IntraCellDemand] = field(default_factory=list)
    # Granularity every volume above is expressed in, after sub-chunk refinement. Downstream
    # quantities derived from chunk size (fine epoch duration, epochs per coarse epoch,
    # "9-Chunk_Size", algorithmic bandwidth) must come from here, NOT from Topology.chunk_size,
    # which stays the un-refined root value. See teccl/hierarchy/scale.py.
    scale: Optional[ChunkScale] = None
    subdivision: int = 1              # the Q that was applied (1 == no refinement)


# --------------------------------------------------------------------------------------------
# Demand-shape derivation (collective-agnostic)
# --------------------------------------------------------------------------------------------
def _cell_of(mapping: HierarchyMapping) -> Dict[int, int]:
    return mapping.fine_to_coarse


def identity_sets(fine_demand, mapping: HierarchyMapping, relabel=None
                  ) -> Tuple[Dict[Tuple[int, int], List[Identity]],
                             Dict[Tuple[Identity, int], Tuple[int, ...]]]:
    """Derive, straight off the fine demand array, the per-(U,V) identity set and per-identity
    destination-GPU set. Mirrors abstract.coarsify_demand's counting exactly so that
    len(ID[(U,V)]) == coarse[U][V].

    `relabel` maps a tensor coordinate (axis-0 index, chunk-axis index) to the identity it stands
    for, and must be supplied BELOW THE ROOT. At the root the two coincide -- axis 0 is the source
    GPU and the chunk axis is its chunk index, so `(s, ci)` IS the identity -- but at a child level
    axis 0 is the local index of whichever GPU currently HOLDS the data and the chunk axis is a
    fresh per-holder enumeration, so the raw coordinate merely has the shape of an identity while
    naming the wrong chunk of the wrong GPU. Node indices need no such treatment: they are correct
    in the index space of the topology the tensor was built against, which is the one `mapping`
    describes. See teccl/hierarchy/problem.py (LevelDemand.relabel).

    Returns:
      id_sets[(U, V)]      = sorted list of identities with coarse(holder)=U that some GPU in
                             V wants (U != V).
      targets[(identity, V)] = tuple of fine GPUs t in cell V with fine_demand[s][t][ci] > 0.
    """
    f2c = _cell_of(mapping)
    n_fine = len(fine_demand)
    id_sets: Dict[Tuple[int, int], List[Identity]] = defaultdict(list)
    targets: Dict[Tuple[Identity, int], Tuple[int, ...]] = {}
    for s in range(n_fine):
        cs = f2c[s]
        chunks = len(fine_demand[s][s]) if n_fine else 0
        for ci in range(chunks):
            ident: Identity = relabel((s, ci)) if relabel else (s, ci)
            # who wants (s, ci), grouped by destination cell
            wanters_by_cell: Dict[int, List[int]] = defaultdict(list)
            for t in range(n_fine):
                if fine_demand[s][t][ci] > 0:
                    wanters_by_cell[f2c[t]].append(t)
            for cv, ts in wanters_by_cell.items():
                if cv == cs:
                    continue
                id_sets[(cs, cv)].append(ident)
                targets[(ident, cv)] = tuple(sorted(ts))
    for key in id_sets:
        id_sets[key].sort()
    return id_sets, targets


# --------------------------------------------------------------------------------------------
# Coarse-piece extraction from the solved LP formulation
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _CoarsePiece:
    src_cell: int                         # PHYSICAL sender cell of this store-and-forward leg
    dst_cell: int                         # PHYSICAL receiver cell of this leg
    egress_neighbor: int                  # coarse switch id of the first hop out of src_cell
    ingress_neighbor: int                 # coarse switch id of the last hop into dst_cell
    via_switches: Tuple[int, ...]         # FINE switch ids
    volume: float
    send_epoch: int
    arrival_epoch: int
    # LOGICAL (source cell, dest cell) of the coarse flow this leg belongs to -- the (s, d) key of
    # per_chunk_flow_paths. Equals (src_cell, dst_cell) for a normal single-hop inter-cell delivery,
    # but DIFFERS when the coarse path store-and-forwards through an intermediate cell: a path
    # A -> ... -> B -> ... -> C splits into a leg (src B, dst C) whose origin is still (A, C). The
    # walk keys pieces by physical endpoints, so origin is what preserves the true source identity
    # across the split (see _origin_diagnosis / the host-transit gap).
    origin: Tuple[int, int] = (-1, -1)


@dataclass(frozen=True)
class _CoarseRoute:
    """One WHOLE coarse delivery: every leg of one source-to-destination path, in order.

    This -- not the leg -- is the unit identity resolution assigns to, and the reason is the
    transit case. A coarse path that store-and-forwards through an intermediate CELL,
    `A -> sw -> B -> sw -> C`, splits into legs filed under the PHYSICAL pairs (A, B) and (B, C)
    while both belong to the LOGICAL flow (A, C). Keyed by leg, the two halves of one delivery are
    unrelated, `id_sets[(A,B)]` carries volume no GPU in B wants, and the demand anchor
    `len(ID(U,V)) == coarse[U][V]` fails at both ends (see _origin_diagnosis). Keyed by `origin`,
    it holds again:

        sum(route.volume for route in routes_by_pair[(U,V)]) == coarse[U][V] == |ID(U,V)|

    and `identity_sets` needs no change at all -- which is the point. The alternative fix,
    "track transit identities in identity_sets", would put identities into `id_sets[(A,B)]` that
    nobody in B wants, breaking the very mirror of `coarsify_demand` that anchors this module.

    Non-transit topologies are unaffected: every route has exactly one leg, and every step below
    is the one that ran before.
    """
    origin: Tuple[int, int]                # (logical source cell, logical dest cell)
    legs: Tuple[_CoarsePiece, ...]         # >= 1, chained: legs[i].dst_cell == legs[i+1].src_cell
    volume: float

    @property
    def first(self) -> _CoarsePiece:
        """The leg that leaves the SOURCE cell -- the one whose egress gateway the identity LP
        chooses, and the one the source-cell staging relay feeds."""
        return self.legs[0]

    @property
    def last(self) -> _CoarsePiece:
        """The leg that lands in the DESTINATION cell -- the one whose ingress candidates decide
        whether the identity needs an intra-cell hop after it arrives."""
        return self.legs[-1]

    @property
    def is_transit(self) -> bool:
        return len(self.legs) > 1


def _validate_route(route: _CoarseRoute) -> None:
    """§7's per-route checks. Cheap, and one of them guards a real failure mode.

    `dig_to_source` carries a path-local cycle-avoidance band-aid rather than a proper
    time-expanded flow decomposition, so a path that revisits a cell is a live possibility rather
    than a theoretical one. A route that revisits a cell would emit a forwarding demand whose data
    depends on a later leg of itself -- a deadlock at the child level, or an unsatisfiable band
    window. Fail loud here, where the path is still intact enough to name.
    """
    legs = route.legs
    if not legs:
        raise AssertionError(f"coarse route {route.origin} has no legs")
    if legs[0].src_cell != route.origin[0] or legs[-1].dst_cell != route.origin[1]:
        raise AssertionError(
            f"coarse route {route.origin} runs {legs[0].src_cell} -> {legs[-1].dst_cell}, which is "
            f"not its logical origin")
    cells = [legs[0].src_cell]
    for a, b in zip(legs, legs[1:]):
        if a.dst_cell != b.src_cell:
            raise AssertionError(
                f"coarse route {route.origin} does not chain: a leg ends at cell {a.dst_cell} but "
                f"the next starts at {b.src_cell}")
        if b.send_epoch <= a.arrival_epoch:
            # A forwarded leg may leave no earlier than the epoch AFTER its data arrived. The
            # transit node is a CELL -- a host, not a switch -- so the coarse solve's own
            # store-and-forward rule (lp_formulation's midFC) already gives the one-epoch dwell;
            # a violation means the walk mis-ordered the hops, or that the formulation let a host
            # cut through, and either way the fine level would be asked to forward bytes it does
            # not have yet.
            raise AssertionError(
                f"coarse route {route.origin} forwards without a dwell: a leg arrives at cell "
                f"{a.dst_cell} in epoch {a.arrival_epoch} but the next sends in epoch "
                f"{b.send_epoch}. A transit cell must hold the data at least one epoch.")
        cells.append(b.src_cell)
    cells.append(legs[-1].dst_cell)
    if len(set(cells)) != len(cells):
        raise AssertionError(
            f"coarse route {route.origin} revisits a cell (path {cells}). The coarse flow "
            f"decomposition produced a cyclic path; it cannot be lowered onto a fine schedule "
            f"(see dig_to_source's path-local cycle avoidance).")


def _extract_routes(coarse_solver, mapping: HierarchyMapping
                    ) -> Dict[Tuple[int, int], List[_CoarseRoute]]:
    """Group solved coarse flow into per-ORIGIN routes, reusing the switch-run grouping that
    lp_formulation.chunk_flow_paths_to_string uses (a leg is a maximal run of hops whose receiver
    is a switch: gpu-cell -> switch -> ... -> gpu-cell). Coarse switch ids on the path are
    translated to fine ids via coarse_passthrough. The LP aggregates one commodity per source
    cell, so the chunk axis c is always 0 here.

    The chaining between legs is EXACT, not reconstructed. `per_chunk_flow_paths[(s,d,c)]` entries
    are WHOLE source-to-destination paths -- `dig_to_source` back-traces from the destination all
    the way to the source and files one complete path -- and the switch-run walk below iterates
    successive runs of ONE such path. So "which arriving leg at B feeds which departing leg at B"
    is not missing information to be recovered; it is information this walk used to discard three
    lines after having it. Collecting the runs into a route instead of emitting them independently
    is the whole change: no reassembly heuristic, no matching pass, no temporal proof obligation.
    """
    switch_indices = set(coarse_solver.topology.switch_indices)
    passthrough = mapping.coarse_passthrough
    routes: Dict[Tuple[int, int], List[_CoarseRoute]] = defaultdict(list)

    for (s, d, _c), paths in coarse_solver.per_chunk_flow_paths.items():
        for each_path in paths:
            each_path = [x for x in each_path if len(x) != 0]
            if not each_path:
                continue
            # Reproduce chunk_flow_paths_to_string's exact normalization. dig_to_source appends
            # hops DEST-first (it back-traces from the destination), so a single epoch sort would
            # leave same-epoch hops in reverse-path order and the switch-run walk below would run
            # off the end. sort -> reverse -> sort flips same-epoch hops into SOURCE-first
            # (forward path) order so the walk goes source -> switch(es) -> dest.
            chunk_path = sorted(each_path, key=lambda x: x[-1])[::-1]
            chunk_path = sorted(chunk_path, key=lambda x: x[-1])
            chunk_path = [x for x in chunk_path if round(x[4], 6) > 0]
            if not chunk_path:
                continue
            start = nxt = 0
            legs: List[_CoarsePiece] = []
            while start < len(chunk_path):
                run = [chunk_path[start]]
                # extend while the current hop's RECEIVER is a switch
                while chunk_path[nxt][2] in switch_indices:
                    nxt += 1
                    run.append(chunk_path[nxt])
                start_node = run[0][1]            # sender of first hop  == src cell
                end_node = run[-1][2]             # receiver of last hop == dst cell
                sending_epoch = run[0][5]
                arrival_epoch = run[-1][5]
                volume = run[0][4]
                switches = [hop[2] for hop in run[:-1]]   # intermediate switches (coarse ids)
                if switches:
                    fine_switches = tuple(passthrough[sw] for sw in switches)
                    legs.append(_CoarsePiece(
                        src_cell=start_node, dst_cell=end_node,
                        egress_neighbor=switches[0], ingress_neighbor=switches[-1],
                        via_switches=fine_switches, volume=volume,
                        send_epoch=sending_epoch, arrival_epoch=arrival_epoch,
                        origin=(s, d)))
                start = nxt = nxt + 1
            if not legs:
                continue
            # `dig_to_source` decomposes at the path bottleneck, so every leg of one path carries
            # the same volume. Take the min and check the agreement rather than assuming it: a
            # disagreement would mean the decomposition split mid-path, which this route form
            # cannot represent (an identity cannot ride two thirds of a leg).
            volume = min(leg.volume for leg in legs)
            spread = max(leg.volume for leg in legs) - volume
            if spread > EPS:
                raise AssertionError(
                    f"coarse path {(s, d)} has legs of differing volume (spread {spread:g}); a "
                    f"store-and-forward path must carry one volume end to end. Its flow "
                    f"decomposition split mid-path, which a route cannot represent.")
            legs = [replace(leg, volume=volume) for leg in legs]
            route = _CoarseRoute(origin=(s, d), legs=tuple(legs), volume=volume)
            _validate_route(route)
            routes[(s, d)].append(route)
    return routes


def _origin_diagnosis(routes_by_pair: Dict[Tuple[int, int], List["_CoarseRoute"]],
                      U: int, V: int) -> str:
    """Explain a mismatch between a demanded pair (U, V) and the extracted routes. Returns "" when
    nothing unusual is found, so the normal case adds no noise.

    Routes are keyed by their LOGICAL origin, so transit is no longer a gap this has to explain --
    it is modelled, and a route's legs carry it. What remains is a genuine-malformation aid: if a
    leg's physical endpoints disagree with its route's origin in a way the route form should have
    absorbed, say where the volume went. Two tells:
      * DISPLACED: legs whose logical origin is (U, V) sit under routes filed elsewhere.
      * FOREIGN:   routes filed under (U, V) carry legs of a different logical pair.
    Both should now be empty for every input; a non-empty report means the extraction is
    malformed, not that transit happened."""
    displaced = defaultdict(float)   # physical leg key -> volume of legs whose origin is (U, V)
    foreign = defaultdict(float)     # foreign origin -> volume filed under (U, V)
    for key, rlist in routes_by_pair.items():
        for r in rlist:
            for p in r.legs:
                if p.origin == (U, V) and key != (U, V):
                    displaced[(p.src_cell, p.dst_cell)] += p.volume
                if key == (U, V) and p.origin != (U, V):
                    foreign[p.origin] += p.volume
    msgs = []
    if displaced:
        legs = ", ".join(f"{k}:vol={v:g}" for k, v in sorted(displaced.items()))
        msgs.append(f"legs of the {(U, V)} flow are filed under other origins [{legs}]")
    if foreign:
        legs = ", ".join(f"origin {o}:vol={v:g}" for o, v in sorted(foreign.items()))
        msgs.append(f"routes filed under {(U, V)} carry legs of other flows [{legs}]")
    return "; ".join(msgs)


# --------------------------------------------------------------------------------------------
# Joint identity assignment (identities x (coarse piece, egress gateway) slots)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Slot:
    """One (coarse ROUTE, FIRST-leg egress gateway GPU) pair an identity can be assigned to.

    `capacity` is the route's volume scaled by this gateway's share of the coarse egress link,
    proportional to fine link capacity. Splitting EVERY route proportionally (rather than only
    matching aggregate budgets) is what makes the per-(U, V) decomposition sound: for any epoch k,
    gateway g's egress is sum over that epoch's routes of volume * cap_g/cap_sum, which coarse
    feasibility bounds by cap_g * Delta -- so the per-fine-uplink per-epoch bound holds by
    construction, globally, even though each demand pair is solved independently.

    ONE gateway per slot, chosen on the first leg -- not a gateway TUPLE over all the legs. The
    tuple generalization is sound (summing `volume * prod_legs(cap_g/cap_sum)` over the
    combinations containing g at leg i gives back exactly `volume * cap_g/cap_sum` there) but it
    costs a `prod |gateways(leg)|` column count and buys nothing: whether leg i's landing GPU
    co-locates with leg i+1's egress gateway is a property of THE LEG AND THE TOPOLOGY, not of
    which chunk rides it, so every identity on a route pays the same transit cost. Pricing an
    identity-INDEPENDENT decision inside the identity LP is a combinatorial product spent on a term
    that is constant across the rows it would discriminate between. Downstream legs are placed
    afterwards, by `_place_downstream_legs`.

    `ingress_candidates` are the fine boundary GPUs of the DESTINATION cell that own the link from
    the LAST leg's ingress neighbor -- the identity's final landing, which is what the objective's
    ingress tier is priced against. Usually a single GPU (the choice is forced by the coarse
    path); where there are several, _pick_ingress selects one.
    """
    route: "_CoarseRoute"
    egress_gpu: int
    capacity: float
    ingress_candidates: Tuple[int, ...]


def _build_slots(routes: Sequence["_CoarseRoute"], U: int, V: int,
                 mapping: HierarchyMapping, fine_topology: Topology) -> List[_Slot]:
    """Expand the coarse routes of one (U, V) pair into per-(route, first-leg egress gateway)
    slots."""
    slots: List[_Slot] = []
    for r in routes:
        p = r.first
        last = r.last
        egress_gws = mapping.boundary_gpu[(U, p.egress_neighbor)]
        # Every leg is placed onto real GPUs, so every leg's neighbours must be measurable -- not
        # just the first leg's. A transit route whose middle hop is a direct cell-to-cell coarse
        # link is as unplaceable as a single-leg one.
        neighbours = [(side, nb) for leg in r.legs
                      for side, nb in (("egress", leg.egress_neighbor),
                                       ("ingress", leg.ingress_neighbor))]
        for side, nb in neighbours:
            if nb not in mapping.coarse_passthrough:
                # A CURRENT, GENERAL limitation of the piece/slot machinery, not a property of
                # whichever solver routed this hop: a piece is placed onto real GPUs by splitting
                # the coarse link across the boundary GPUs that own it, and that split reads
                # `capacity[gpu][fine_neighbor]` -- which needs the coarse neighbor to be an
                # un-collapsed fine node (a leaf, a spine, an NVSwitch). A DIRECT cell-to-cell
                # coarse link has no such node between the two cells, so there is nothing to
                # measure the split against and the gateway choice is undetermined. Deliberately
                # deferred: every topology solved so far reaches its peers through a switch.
                raise NotImplementedError(
                    f"piece {U} -> {V} uses coarse node {nb} as its {side} neighbor, but {nb} is a "
                    f"collapsed CELL rather than an un-collapsed passthrough node -- i.e. this is a "
                    f"direct cell-to-cell coarse link. Placing such a hop onto fine GPUs is not "
                    f"implemented (see _build_slots): the boundary split needs a fine neighbor node "
                    f"to measure link capacity against. Topologies whose cells reach each other "
                    f"through a switch are unaffected.")
        fine_egress_nb = mapping.coarse_passthrough[p.egress_neighbor]
        caps = [fine_topology.capacity[g][fine_egress_nb] for g in egress_gws]
        cap_sum = sum(caps) or 1.0
        # The LAST leg's ingress neighbour: where the identity actually lands in V. For a
        # single-leg route this is the same leg, which is why nothing changes off the transit path.
        ingress_cands = tuple(mapping.boundary_gpu[(V, last.ingress_neighbor)])
        for g, cap in zip(egress_gws, caps):
            share = r.volume * cap / cap_sum
            if share > EPS:
                slots.append(_Slot(route=r, egress_gpu=g, capacity=share,
                                   ingress_candidates=ingress_cands))
    return slots


def _solve_assignment(identities: Sequence[Identity],
                      native_gpu: Dict[Identity, int],
                      target_gpus: Dict[Identity, Tuple[int, ...]],
                      slots: Sequence[_Slot]) -> Dict[Tuple[Identity, int], float]:
    """Assign identities to slots, minimizing intra-cell relay work at BOTH ends.

    rows d in identities  (supply 1 each: the destination cell receives each identity once)
    cols j in slots        (capacity slots[j].capacity)

    The objective is LEXICOGRAPHIC, not a plain sum, and the ordering matters:

        1. EGRESS relay        w = nD + 1     g != native_gpu[d]
        2. INGRESS relay       w = 1          no ingress candidate of this slot wants d
        3. epoch preference    w = tiny       a RELAYED identity taking an early epoch

    Egress must dominate ingress rather than trade against it. The two relays are not
    interchangeable: an `egress_stage` relay is HARD (its deadline is the network send epoch, and
    missing it slips the internode schedule) while `ingress_distribution` is SOFT (no deadline,
    absorbed by the intra-fabric slack) -- see crossbar_solve's hard/soft split. An egress relay also
    sits UPSTREAM of the network hop and adds pressure to the pre-epoch-0 staging prologue, whereas
    an ingress relay is downstream and off the network critical path. With equal weights the solver
    would be indifferent between "relay at the source" and "relay at the destination" (both cost 1)
    and could silently give up a native egress -- a regression against this function's original
    goal. The big-M form is required rather than merely w_egress > w_ingress because the slot
    capacities COUPLE identities: one identity's native claim can displace another's.

    Tier 2 is measured against the slot's ingress CANDIDATES, not a chosen gateway: an identity can
    be landed directly iff some boundary GPU owning this piece's ingress neighbor is one of its
    targets. For AllGather every GPU of the cell is a target, so tier 2 is uniformly 0 and the
    objective reduces to the original egress-only one. For AllToAll a target is a single GPU and the
    tier bites. _pick_ingress then realizes the choice this tier priced.

    Tier 3 preserves the old `_egress_order` heuristic's intent -- keep relayed identities off the
    earliest egress epochs -- as an objective term instead of a post-hoc sort. It matters because a
    relay feeding a coarse-epoch-0 egress must complete before ANY network send, which is what
    forces a staging prologue and delays the whole timeline.

    Returns x[(d, slot_index)] for the positive entries.
    """
    D = list(identities)
    nD, nS = len(D), len(slots)
    if nD == 0 or nS == 0:
        return {}

    max_epoch = max(s.route.first.send_epoch for s in slots)
    K = max_epoch + 1
    W_EGRESS = float(nD + 1)
    W_INGRESS = 1.0
    # Total tier-3 cost is < 0.5, strictly below one unit of tier 2; and W_EGRESS exceeds the
    # largest possible tier-2 + tier-3 total (nD * 1 + 0.5), so the ordering is exactly
    # lexicographic rather than a weighted compromise.
    W_EPOCH = 1.0 / (2.0 * nD * K + 1.0)

    cost = np.empty(nD * nS, dtype=float)
    for di, d in enumerate(D):
        nat = native_gpu[d]
        wanted = set(target_gpus.get(d, ()))
        for j, s in enumerate(slots):
            relayed = s.egress_gpu != nat
            c = W_EGRESS if relayed else 0.0
            if not (wanted & set(s.ingress_candidates)):
                c += W_INGRESS
            if relayed:
                c += W_EPOCH * (K - s.route.first.send_epoch)
            cost[di * nS + j] = c

    A_eq, b_eq = [], []
    for di in range(nD):                       # each identity delivered exactly once
        row = np.zeros(nD * nS)
        row[di * nS:(di + 1) * nS] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    A_ub, b_ub = [], []
    for j, s in enumerate(slots):              # each slot bounded by its capacity share
        row = np.zeros(nD * nS)
        for di in range(nD):
            row[di * nS + j] = 1.0
        A_ub.append(row)
        b_ub.append(s.capacity)

    res = linprog(cost, A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"identity slot assignment infeasible: {res.message}")
    x: Dict[Tuple[Identity, int], float] = {}
    for di, d in enumerate(D):
        for j in range(nS):
            v = res.x[di * nS + j]
            if v > EPS:
                x[(d, j)] = float(v)
    return x


def _pick_ingress(leg: "_CoarsePiece", candidates: Sequence[int], identity: Identity,
                  volume: float, preferred: AbstractSet[int],
                  epoch_capacity: Dict[int, float],
                  ledger: Dict[Tuple[int, int, int], float],
                  tol: float = EPS, max_denom: int = MAX_DENOM) -> List[Tuple[int, float]]:
    """Choose the fine GPU(s) one LEG lands on, preferring a `preferred` GPU and respecting fine
    downlink capacity.

    `preferred` is supplied by the caller rather than read off `target_gpus` here, because what
    counts as a good landing depends on WHICH leg this is. At the destination cell it is the
    identity's target GPUs -- landing on one saves an intra-cell fan-out hop. At a TRANSIT cell
    nobody wants the data at all; what matters instead is landing on the GPU that owns the
    OUTGOING uplink, because a transit that is not co-located needs an extra coarse epoch the
    solve did not budget (design §6). Same mechanism, two different notions of "preferred", so the
    notion is the caller's to name.

    This replaces taking `boundary_gpu[...][0]` unconditionally, which had two defects. It ignored
    capacity: a coarse link whose capacity is the SUM of several fine downlinks (abstract() sums
    them) could have all of its per-epoch volume dumped on one fine link, oversubscribing it while
    a sibling sat idle -- the fine schedule then being infeasible against the very coarse solution
    it implements. And it ignored the destination, so an identity wanted by a boundary GPU would
    land elsewhere and pay an avoidable intra-cell hop.

    `ledger` is keyed (gpu, ingress_neighbor, arrival_epoch) and spans the WHOLE resolution, not
    one demand pair, because a destination cell's ingress link is shared by every source cell
    sending to it. The neighbor is part of the key, not just the gpu, because a gpu can own MORE
    THAN ONE independent ingress port -- e.g. a dual-plane host, where every gpu has a separate
    physical link into each plane's leaf. Two pieces arriving over different planes in the same
    epoch are not competing for the same budget, even though they land on the same gpu; keying on
    (gpu, epoch) alone would wrongly merge two independent ports into one shared budget and see
    "no room" on the second plane while its own physical port sits untouched -- exactly the
    accounting `_assert_rate_within_capacity` avoids by keying ingress on
    (via_switches[-1], gpu, epoch).

    Usually returns a single (gpu, volume) pair. But the SUM of the candidates' fine downlinks is
    exactly what the coarse capacity was built from, so a piece can legitimately need more than any
    single fine downlink has room for while still fitting the candidates collectively -- e.g. a
    coarse link backed by 6 fine downlinks, each capped near its own epoch budget, delivering one
    piece that only one of them alone can't absorb. In that case the piece is split across however
    many candidates it takes, biggest room first, rather than raising.

    `tol` bounds how much "room" and "volume" may disagree and still be treated as equal. It
    should be `snap_tolerance(coarse_solver)`, not the bare EPS default: `room` is a per-(gpu,
    neighbor, epoch) ledger accumulated across every piece placed so far in the whole resolution,
    so by the time a late identity is placed it carries the SUM of many upstream floats -- the
    coarse LP, `_solve_assignment`'s scipy LP, and every earlier `commit` in this same ledger --
    and a comparison at the bare 1e-6 module EPS can reject a piece that fits to within noise, the
    same way `_snap_group` would reject an off-by-noise volume without its own amplified budget.
    """
    candidates = list(candidates)
    if not candidates:
        raise RuntimeError(
            f"no ingress boundary GPU for cell {leg.dst_cell} from coarse neighbor "
            f"{leg.ingress_neighbor}")
    epoch = leg.arrival_epoch
    neighbor = leg.ingress_neighbor
    wanted = set(preferred)

    def room(h: int) -> float:
        return epoch_capacity[h] - ledger.get((h, neighbor, epoch), 0.0)

    def commit(h: int, v: float) -> None:
        ledger[(h, neighbor, epoch)] = ledger.get((h, neighbor, epoch), 0.0) + v

    if len(candidates) == 1:
        # Forced by the coarse path; capacity is still checked by the caller's assert.
        commit(candidates[0], volume)
        return [(candidates[0], volume)]

    fits = [h for h in candidates if room(h) >= volume - tol]
    on_target = [h for h in fits if h in wanted]
    if on_target:
        # BEST fit (tightest room that still fits), not least-loaded: leaving the roomiest
        # candidates untouched is what keeps them available for a LATER, larger volume in this
        # same epoch. Least-loaded (worst fit) spreads small volumes evenly across every downlink
        # instead, fragmenting them -- which is exactly the shape that forces the split fallback
        # below to trigger on a piece that a less fragmented packing would have placed whole.
        h = min(sorted(on_target), key=lambda h: room(h))
        commit(h, volume)
        return [(h, volume)]
    if fits:
        h = min(sorted(fits), key=lambda h: room(h))
        commit(h, volume)
        return [(h, volume)]

    # `total_room` is a SUM over candidates of a ledger that has itself been summed over every
    # earlier commit in the whole resolution -- `tol` bounds one comparison, not a sum of them, so
    # the deficit this comparison can legitimately see from pure accumulation is bigger than `tol`
    # by the time many pieces have landed on the same (gpu, neighbor, epoch). `dust_threshold`
    # already names exactly this "is a leftover amount real or is it residue" question -- it is
    # what `_snap_group` uses to decide whether a group's shortfall against 1.0 is genuine -- so
    # reuse it here rather than inventing a second noise budget.
    budget = dust_threshold(tol, max_denom)
    total_room = sum(max(room(h), 0.0) for h in candidates)
    if total_room < volume - budget:
        raise RuntimeError(
            f"ingress capacity exhausted for cell {leg.dst_cell} epoch {epoch}: "
            f"identity {identity} needs {volume:g} but candidates "
            f"{ {h: round(room(h), 6) for h in candidates} } have no room "
            f"(per-epoch capacities { {h: round(epoch_capacity[h], 6) for h in candidates} }). The "
            f"coarse solve routed more into this cell in one epoch than its fine downlinks can "
            f"absorb.")

    # No single candidate fits, but the candidates collectively have (up to dust) enough room --
    # spread the piece across them, largest room first (targets first) so it lands on as few GPUs
    # as possible rather than smearing thinly. Any residue at or below `budget` when the candidates
    # run dry is left uncommitted rather than forced onto a GPU past its own room: the identity's
    # total across all its pieces then falls short of its true volume by at most `budget`, which is
    # exactly the shortfall `_snap_group`'s own dust handling downstream is sized to absorb.
    remaining = volume
    picks: List[Tuple[int, float]] = []
    for h in sorted(candidates, key=lambda h: (h not in wanted, -room(h))):
        if remaining <= budget:
            break
        take = min(room(h), remaining)
        if take <= tol:
            continue
        commit(h, take)
        picks.append((h, take))
        remaining -= take
    if remaining > budget:
        raise RuntimeError(
            f"ingress capacity exhausted for cell {leg.dst_cell} epoch {epoch}: "
            f"identity {identity} needs {volume:g} but only {volume - remaining:g} could be "
            f"placed across candidates {candidates} after splitting.")
    return picks


def _bridging_gpus(cell: int, inbound: "_CoarsePiece", outbound: "_CoarsePiece",
                   mapping: HierarchyMapping) -> List[int]:
    """The GPUs of a transit cell that own BOTH the link the data arrives on and the link it
    leaves on -- the co-located transits.

    Co-location is a REQUIREMENT here, not a preference, and that is worth being explicit about
    (design §6). The coarse solve already models a cell as store-and-forward: `lp_formulation`'s
    midFC lets flow arriving in epoch k leave in epoch k+1, so the one-epoch dwell is budgeted.
    But that +1 models a host that receives and re-sends ON THE SAME PORT. The abstraction
    collapsed the cell's GPUs into a single coarse node, so when the landing GPU and the outgoing
    gateway differ, the intra-cell hop between them is invisible to the coarse level and
    unbudgeted -- and with sends pinned to the leading edge of a band, a piece landing at the end
    of epoch k is ready in band k+1 and can only feed a send from epoch k+2. The coarse solve
    gives k+1. So a non-co-located transit is INFEASIBLE rather than merely expensive, which is
    why it is a constraint on this pass and not a term in the objective.
    """
    return [g for g in mapping.boundary_gpu[(cell, inbound.ingress_neighbor)]
            if g in set(mapping.boundary_gpu[(cell, outbound.egress_neighbor)])]


def _place_downstream_legs(route: "_CoarseRoute", identity: Identity, first_egress_gpu: int,
                           volume: float, target_gpus: AbstractSet[int],
                           mapping: HierarchyMapping, fine_topology: Topology,
                           epoch_duration: float,
                           ingress_ledger: Dict[Tuple[int, int, int], float],
                           egress_ledger: Dict[Tuple[int, int, int], float],
                           tol: float, max_denom: int
                           ) -> List[Tuple[Tuple[Tuple[int, int], ...], float]]:
    """Walk a route's legs in order, placing each onto real GPUs, after the LP has chosen the
    first leg's egress gateway.

    This is the half of the decomposition the identity LP deliberately does not do (design §3.1).
    Whether leg i's landing GPU co-locates with leg i+1's egress gateway is a property of THE LEGS
    AND THE TOPOLOGY, not of which chunk rides them -- every identity on a route pays the same
    transit cost -- so pricing it inside the identity LP would pay a `prod |gateways(leg)|` column
    count for a term that is constant across the rows it would discriminate between.

    Returns a list of (per-leg (egress GPU, ingress GPU) tuples, volume) fragments. Usually one
    fragment carrying the whole volume; more when `_pick_ingress` has to split a landing across
    several fine downlinks, which is legitimate (a coarse link's capacity is the SUM of the fine
    downlinks behind it, so one piece can exceed any single one while fitting them collectively).

    What this decomposition gives up, exactly once and worth naming: when a commodity (A, C)
    splits across routes with DIFFERENT downstream ingress costs at C, the first-leg LP chooses
    identities blind to that difference. When a commodity uses a single route shape it is exactly
    optimal -- the whole pool transits the same cells, so the final landing preference is fully
    recoverable here. And the loss is confined to the objective's SOFT tier: it can never cost
    feasibility, only an avoidable intra-cell hop at the destination.
    """
    # Each fragment is (legs placed so far, volume, the GPU that egresses the NEXT leg).
    frags: List[Tuple[Tuple[Tuple[int, int], ...], float, int]] = [((), volume, first_egress_gpu)]

    for i, leg in enumerate(route.legs):
        is_last = (i == len(route.legs) - 1)
        outbound = None if is_last else route.legs[i + 1]
        nxt: List[Tuple[Tuple[Tuple[int, int], ...], float, int]] = []
        for placed, vol, egress_gpu in frags:
            if is_last:
                candidates = list(mapping.boundary_gpu[(leg.dst_cell, leg.ingress_neighbor)])
                preferred = set(target_gpus)
            else:
                candidates = _bridging_gpus(leg.dst_cell, leg, outbound, mapping)
                if not candidates:
                    raise RuntimeError(
                        f"cell {leg.dst_cell} transits identity {identity} from coarse neighbor "
                        f"{leg.ingress_neighbor} to {outbound.egress_neighbor}, but no GPU owns "
                        f"both links -- the two NICs hang off different GPUs. Forwarding then "
                        f"needs an intra-cell hop between them, which costs a SECOND coarse epoch "
                        f"(arrival + 2) that the coarse solve did not budget: it models a "
                        f"store-and-forward dwell of exactly one epoch. This is a true statement "
                        f"about the topology, not a resolution failure; it needs a per-cell "
                        f"forwarding dwell in the coarse formulation (see cell_relay_design.md "
                        f"§6), which is deliberately out of scope here.")
                # Prefer a bridging GPU whose OUTGOING link still has room this epoch, using the
                # same `preferred` channel the destination case uses for targets. Preference, not
                # restriction: if none has room, _pick_ingress still places the piece and the
                # egress commit below is what fails loud about it.
                out_cap = _egress_epoch_capacity(outbound, candidates, mapping, fine_topology,
                                                 epoch_duration)
                preferred = {g for g in candidates
                             if out_cap[g] - egress_ledger.get(
                                 (g, outbound.egress_neighbor, outbound.send_epoch), 0.0)
                             >= vol - tol}

            in_cap = _ingress_epoch_capacity(leg, candidates, mapping, fine_topology,
                                             epoch_duration)
            for landed, part in _pick_ingress(leg, candidates, identity, vol, preferred, in_cap,
                                              ingress_ledger, tol=tol, max_denom=max_denom):
                legs_so_far = placed + ((egress_gpu, landed),)
                if is_last:
                    nxt.append((legs_so_far, part, -1))
                else:
                    # Co-located: the GPU that landed this leg is the one that sends the next.
                    _commit_egress(outbound, landed, part, identity, mapping, fine_topology,
                                   epoch_duration, egress_ledger, tol)
                    nxt.append((legs_so_far, part, landed))
        frags = nxt

    return [(placed, vol) for placed, vol, _ in frags]


def _commit_egress(leg: "_CoarsePiece", gpu: int, volume: float, identity: Identity,
                   mapping: HierarchyMapping, fine_topology: Topology, epoch_duration: float,
                   ledger: Dict[Tuple[int, int, int], float], tol: float) -> None:
    """Book one forwarded leg's volume against its sender's per-epoch uplink budget, and fail loud
    if the uplink has no room.

    The honest version of what a gateway-tuple slot capacity would only approximate: the product
    formula bounds the EXPECTED per-epoch load on a forwarding gateway, this accounts for the
    actual one. Keyed and compared exactly like `ingress_ledger` -- (gpu, neighbor, epoch), with
    the neighbor in the key because one GPU can own several independent ports -- and with the same
    amplified tolerance, because it accumulates the same way.
    """
    cap = _egress_epoch_capacity(leg, [gpu], mapping, fine_topology, epoch_duration)[gpu]
    key = (gpu, leg.egress_neighbor, leg.send_epoch)
    used = ledger.get(key, 0.0)
    if used + volume > cap + tol:
        raise RuntimeError(
            f"forwarding egress capacity exhausted at cell {leg.src_cell} epoch "
            f"{leg.send_epoch}: gpu {gpu} already forwards {used:g} toward coarse neighbor "
            f"{leg.egress_neighbor} and identity {identity} needs {volume:g} more, but the link "
            f"absorbs {cap:g} per epoch. All of this cell's transit volume is forced onto the "
            f"GPUs that bridge the two islands (co-location is required, see "
            f"_bridging_gpus/_place_downstream_legs), and there is more of it than they can "
            f"carry.")
    ledger[key] = used + volume


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------
# MAX_DENOM is defined near EPS, above _pick_ingress -- see the comment there for why it lives
# there and how a single instance gets a larger grid without a global bump (InstanceParams.
# max_denom / resolve_max_denom). Mirrors ncclize's MAX_M too (teccl/ncclize/teccl_ncclize.py).
# Integerization now happens HERE, at the recursion boundary, rather than at the ncclize boundary
# -- see _subdivision_factor.
MAX_SUBDIVISION = 128
# Relative slack when re-checking a paced rate against a link bandwidth. Relative, not absolute:
# a rate is in GB/s and spans orders of magnitude across a heterogeneous fabric, so one absolute
# epsilon is either meaningless on a 900 GB/s NVLink or punitive on a 25 GB/s uplink.
RATE_REL_TOL = 1e-6
# How much the reconstruction chain amplifies the coarse solver's declared tolerance by the time a
# volume reaches _snap_volumes. The coarse LP's feasibility_tol bounds ITS constraint violation;
# what gets snapped has since been through the proportional slot split, the `len(identities) /
# total_cap` rescale, and a whole second LP (`_solve_assignment`, scipy/HiGHS). Measured on the
# TwoPodRailHostBound runs at feasibility_tol=1e-5: allgather tops out at 3.3e-06, alltoall at
# 2.3e-05 -- so the chain costs about 2.3x, and 4x leaves margin while staying well inside
# grid_resolution (1.22e-04 at MAX_DENOM=64). Sourcing the tolerance verbatim from the coarse
# solver, as this did first, rejected 20 of alltoall's 1024 assignment volumes.
RECONSTRUCTION_NOISE_FACTOR = 4.0


def grid_resolution(max_denom: int = MAX_DENOM) -> float:
    """Smallest possible gap between two DISTINCT rationals with denominator <= max_denom.

    |p/d - p'/d'| = |p d' - p' d| / (d d') >= 1 / (d d') >= 1 / max_denom^2 whenever the two are
    distinct, so half that is the radius within which a snap to the 1/max_denom grid is
    UNAMBIGUOUS -- i.e. within which the nearest grid point is the true value rather than a
    neighbour. This is the quantity that couples MAX_DENOM to the solver's tolerance; see
    snap_tolerance.
    """
    return 1.0 / (2.0 * max_denom * max_denom)


def resolve_max_denom(coarse_solver=None) -> int:
    """This solve's identity-resolution grid size, from InstanceParams.max_denom if it overrides
    the module default.

    Mirrors snap_tolerance's own feasibility_tol lookup, because the two travel together (see
    InstanceParams.max_denom): a `CoarseSolution` carries `max_denom` directly -- flattened out of
    `ui.instance` at construction (teccl/hierarchy/solve.py), the same way it carries
    `feasibility_tol` -- a raw solver object carries it nested under `.user_input.instance`, and
    anything else (a closed-form row, a test shim) has neither, so it gets the module default.
    """
    md = getattr(coarse_solver, "max_denom", None)
    if md is None:
        instance = getattr(getattr(coarse_solver, "user_input", None), "instance", None)
        md = getattr(instance, "max_denom", None)
    return int(md) if md is not None else MAX_DENOM


def snap_tolerance(coarse_solver=None, max_denom: Optional[int] = None) -> float:
    """How far a solved volume may sit from the 1/max_denom grid and still be snapped to it.

    THE ONE PLACE A NUMERIC TOLERANCE IS NAMED in the lowering half. Everything downstream of
    `_snap_volumes` holds exact `Fraction`s, so no other step here has -- or needs -- an epsilon.
    That is deliberate: the previous arrangement snapped with `limit_denominator(64)` (tolerating
    ~1.2e-4) and then re-checked the raw float against `volume * q` with a fixed 1e-6, which is a
    tolerance of 1e-6/q ~ 8e-9 on the volume itself. The two disagreed by four orders of magnitude
    and the disagreement GREW with q, so any solution with non-dyadic volumes and a moderate q
    failed a check that the snap had already accepted.

    The tolerance comes from the COARSE solver, because that is the only place a numeric error is
    actually declared: Gurobi's `feasibility_tol` bounds its constraint violation. Treat it as a
    FLOOR rather than the true bound on what reaches `_snap_volumes` -- the volumes snapped there
    are emitted by `_solve_assignment` (scipy) running on slot capacities that already carry the
    coarse error, so that chain can produce residue larger than this number. Near-zero residue is
    handled by `dust_threshold`, which is sized independently for exactly that reason; what remains
    under this tolerance is the genuine "is a SUBSTANTIAL volume on the grid" question.

    Raises if the requested tolerance is COARSER than the grid can resolve -- with error that large
    the nearest grid point is not necessarily the right one, and snapping would silently invent a
    volume. Warns when the margin is under 4x, which is where MAX_DENOM = 64 against the default
    feasibility_tol = 1e-4 lands (grid resolution 1.22e-4): compatible, but only by 20%.

    max_denom: None (the default) resolves via `resolve_max_denom(coarse_solver)`, i.e. this
    solve's InstanceParams.max_denom override if it set one, else the module default. Passed
    explicitly only by a caller intentionally checking a DIFFERENT grid than the one this solve
    will actually use (see the coupling test in hierarchy_volume_snap_test.py).
    """
    max_denom = resolve_max_denom(coarse_solver) if max_denom is None else max_denom
    resolution = grid_resolution(max_denom)
    raw = tol = getattr(coarse_solver, "feasibility_tol", None)
    if tol is None:
        gurobi = getattr(getattr(coarse_solver, "user_input", None), "gurobi", None)
        tol = getattr(gurobi, "feasibility_tol", None)
    if tol is None:
        # No solver to ask (a closed-form row, or a test shim). Its volumes are constructed
        # exactly, so demand the comfortable margin rather than inventing a solver tolerance.
        return resolution / 4.0
    tol = float(tol)
    if tol < resolution:
        # Amplify for the reconstruction chain (see RECONSTRUCTION_NOISE_FACTOR), but never past
        # what the grid can resolve -- and never below the raw tolerance, so this can only ever be
        # more permissive than the solver's own declared bound, not less.
        amplified = min(tol * RECONSTRUCTION_NOISE_FACTOR, resolution * 0.9)
        tol = max(tol, amplified)
    if tol >= resolution:
        raise ValueError(
            f"solver feasibility_tol={tol:g} is coarser than the identity grid can resolve "
            f"(MAX_DENOM={max_denom} -> two distinct grid points can be only {2 * resolution:g} "
            f"apart, so the snap radius is {resolution:g}). Snapping would pick a grid point that "
            f"is not necessarily the true volume. Tighten GurobiParams.feasibility_tol to below "
            f"{resolution:g} (ideally {resolution / 4.0:g}) or lower MAX_DENOM to "
            f"{int((1.0 / (2.0 * tol)) ** 0.5)}.")
    if tol > resolution / 4.0:
        logging.warning(
            "identity-grid margin is thin: snap tolerance %g (feasibility_tol %g amplified %gx for "
            "the reconstruction chain) against a snap radius of %g (MAX_DENOM=%d). Volumes will "
            "snap, but a slightly noisier solve would not. Consider feasibility_tol <= %g.",
            tol, raw, RECONSTRUCTION_NOISE_FACTOR, resolution, max_denom,
            resolution / 4.0 / RECONSTRUCTION_NOISE_FACTOR)
    return tol


@dataclass(frozen=True)
class _Assignment:
    """One resolved (identity -> coarse piece) decision, before sub-chunk refinement."""
    src_cell: int
    dst_cell: int
    identity: Identity
    # The WHOLE coarse route, not one leg: a transit delivery is two network flows that must share
    # a sub-chunk index and be ordered by a forwarding demand between them, and only the route
    # relates them. `egress_gpu` / `ingress_gpu` below are the FIRST leg's sender and the LAST
    # leg's receiver -- the identity's two endpoints inside this level. Intermediate legs are
    # placed by `_place_downstream_legs` and carried in `leg_gpus`.
    route: _CoarseRoute
    egress_gpu: int
    ingress_gpu: int
    volume: float
    # (egress GPU, ingress GPU) for EVERY leg, in route order. `egress_gpu` / `ingress_gpu` above
    # are leg_gpus[0][0] and leg_gpus[-1][1] -- the identity's two endpoints in this level -- and
    # are kept as their own fields because that is what every existing reader wants. Empty means
    # "single leg, use those two"; `legs` normalizes it either way.
    leg_gpus: Tuple[Tuple[int, int], ...] = ()
    # Which node in THIS LEVEL's index space physically holds the identity, and therefore stages it
    # onto the gateway. -1 means "identity[0]", which is correct at the ROOT and only there: an
    # identity names its native source GPU, but below the root the data has already been relayed and
    # is held by some other GPU -- in a different index space, at that. Reading the holder off the
    # identity below the root silently emits a staging relay from a node id that means something
    # else entirely at this level.
    holder: int = -1
    # `volume` snapped onto a rational grid, set by _snap_volumes at the float->exact boundary.
    # None until then. Every step from _subdivision_factor onward reads THIS, never `volume`: the
    # float is the solver's noisy answer, this is the answer on the grid the recursion contract
    # requires. Keeping both means a diagnostic can still show what was solved vs what was emitted.
    exact: Optional[Fraction] = None

    @property
    def native(self) -> int:
        return self.holder if self.holder >= 0 else self.identity[0]

    @property
    def legs(self) -> Tuple[Tuple[int, int], ...]:
        """Per-leg (egress GPU, ingress GPU), normalized. A single-leg assignment need not carry
        `leg_gpus` explicitly -- its two endpoint fields already say everything -- so this fills
        it in, which is what lets `_emit_refined` have one code path for both."""
        if self.leg_gpus:
            return self.leg_gpus
        return ((self.egress_gpu, self.ingress_gpu),)


def _egress_epoch_capacity(leg: "_CoarsePiece", candidates: Sequence[int],
                           mapping: HierarchyMapping,
                           fine_topology: Topology, epoch_duration: float) -> Dict[int, float]:
    """Per-epoch volume each candidate egress GPU of one LEG can push, in chunk units.

    The mirror of `_ingress_epoch_capacity`, and note the direction: an egress leg is gpu ->
    switch, so this indexes capacity[gpu][fine_neighbor].

    Needed only for the DOWNSTREAM legs of a transit route. The first leg's egress is already
    bounded by construction -- `_build_slots` splits every route's volume across that cell's
    gateways in proportion to fine link capacity, so per-epoch gateway load is bounded by coarse
    feasibility without any global accounting (see _Slot). No such split covers a forwarded leg:
    the coarse level cannot see it, because the abstraction collapsed the transit cell's GPUs into
    one node. So the forward pass accounts for the ACTUAL load instead, against a ledger.
    """
    fine_nb = mapping.coarse_passthrough[leg.egress_neighbor]
    return {g: fine_topology.capacity[g][fine_nb] * epoch_duration for g in candidates}


def _ingress_epoch_capacity(leg: "_CoarsePiece", candidates: Sequence[int],
                            mapping: HierarchyMapping,
                            fine_topology: Topology, epoch_duration: float) -> Dict[int, float]:
    """Per-epoch volume each candidate ingress GPU of one LEG can absorb, in chunk units.

    Note the direction: the ingress leg is switch -> gpu, so this indexes
    capacity[fine_neighbor][gpu], whereas the egress split in _build_slots uses
    capacity[gpu][fine_neighbor]."""
    fine_nb = mapping.coarse_passthrough[leg.ingress_neighbor]
    return {h: fine_topology.capacity[fine_nb][h] * epoch_duration
            for h in candidates}


# A volume below this fraction of the FINEST representable one (1/MAX_DENOM) is not a small share
# of an identity -- no point on the grid is that close to zero -- it is solver residue. 1% leaves
# two orders of magnitude between "residue" and the smallest real slot, so the two can never be
# confused whichever way the arithmetic drifts.
DUST_FRACTION_OF_GRID = 0.01


def dust_threshold(snap_tol: float, max_denom: int = MAX_DENOM) -> float:
    """Below this, a volume is solver residue rather than a share of the identity.

    Distinct from `snap_tol`, and deliberately larger, because it answers a different question.
    `snap_tol` asks "is this value ON the grid?" and its consequence is a hard error. This asks
    "is this value ZERO?" and its consequence is that the value's share is handed to a sibling
    slot -- the identity still reaches the same cell, just on a different piece -- so being
    slightly generous here is cheap while being strict is not.

    It also covers a gap in how `snap_tol` is sourced: that comes from the COARSE solver, but the
    volumes being snapped are produced downstream by `_solve_assignment` (scipy) on slot capacities
    that already carry the coarse solver's error. The residue that chain emits can exceed the
    coarse tolerance -- an observed 2.3e-05 against a 1e-05 coarse feasibility_tol -- and it is
    near-zero residue, not a genuinely off-grid volume, so it belongs here rather than in an error.

    This threshold may exceed `grid_resolution`, and that is not a contradiction: it measures
    distance from ZERO, whose nearest nonzero neighbour is 1/max_denom, not the interior spacing
    between two adjacent grid points. `_snap_group` separately caps the TOTAL residue it will
    absorb at `grid_resolution`, so residue large enough to make the grid ambiguous still raises.
    """
    return max(snap_tol, DUST_FRACTION_OF_GRID / max_denom)


def _snap_group(vols: Sequence[float], snap_tol: float, key,
                max_denom: int = MAX_DENOM) -> List[Fraction]:
    """Snap ONE (identity, dst_cell) group's volumes onto a shared rational grid summing to 1.

    Group-aware on purpose. Snapping each volume independently would be simpler, but the group's
    volumes PARTITION the identity -- `_emit_refined`'s `cursor == q` check is exactly that -- and
    independent rounding perturbs a sum of 1.0 off 1.0, turning a float-noise problem into a hard
    failure at the partition check. Here the sum is exact by construction: floor onto the shared
    grid, then hand the remaining units to the largest fractional remainders.

    The grid G is the LCM of the per-volume `limit_denominator(MAX_DENOM)` denominators, so every
    volume that really is on the 1/MAX_DENOM grid is exactly representable on G, and the
    post-repair check below is a genuine test of that -- not a rounding allowance.

    Two failure modes are kept apart, because they call for opposite responses and conflating them
    turned solver residue into a crash (see dust_threshold):
      * DUST -- a volume whose nearest grid point is zero. Dropped; the largest-remainder repair
        hands its share to a sibling slot of the same identity.
      * OFF-GRID -- a substantial volume far from every grid point. Raised; this is the real
        "the coarse solution is split more finely than MAX_DENOM admits" signal.
    """
    n = len(vols)
    total = sum(vols)
    # The group must already partition the identity to within solver noise. If it does not, no
    # rounding scheme can fix it and the largest-remainder repair below would quietly paper over a
    # real assignment bug, so say so here instead.
    if abs(total - 1.0) > snap_tol * max(1, n):
        raise AssertionError(
            f"identity {key[0]} -> cell {key[1]}: assignment volumes sum to {total!r}, not 1 "
            f"(tolerance {snap_tol * max(1, n):g} over {n} assignments). They must partition the "
            f"identity; this is an assignment defect, not rounding.")

    dust_tol = dust_threshold(snap_tol, max_denom)
    live = [i for i, v in enumerate(vols) if abs(v) >= dust_tol]
    dust_total = sum(vols[i] for i in range(n) if i not in set(live))
    if not live:
        raise AssertionError(
            f"identity {key[0]} -> cell {key[1]}: every one of its {n} assignment volumes is below "
            f"the {dust_tol:g} dust threshold, so nothing carries it. This is an assignment defect.")

    # The budget for "is this value on the grid". Solver noise PLUS the dropped residue: the group
    # sums to 1, so whatever the dust holds was taken from its siblings, and each of them sits off
    # its own grid point by up to that much. Using snap_tol alone here rejects the very siblings
    # that make dust droppable -- the residue has to come from somewhere.
    budget = snap_tol + abs(dust_total)
    resolution = grid_resolution(max_denom)
    if budget >= resolution:
        raise AssertionError(
            f"identity {key[0]} -> cell {key[1]}: {abs(dust_total):g} of residue across "
            f"{n - len(live)} dust entries pushes the snap budget to {budget:g}, at or beyond the "
            f"{resolution:g} radius within which a grid point is unambiguous. There is too much "
            f"residue to snap safely -- tighten the solver or regularize the coarse LP.")

    dens = []
    for i in live:
        v = vols[i]
        frac = Fraction(v).limit_denominator(max_denom)
        if abs(float(frac) - v) > budget:
            raise AssertionError(
                f"identity {key[0]} -> cell {key[1]}: volume {v!r} is not within {budget:g} of "
                f"any rational with denominator <= {max_denom} (nearest is {frac} = "
                f"{float(frac)!r}), and it is too large ({dust_tol:g}) to be solver residue. The "
                f"coarse solution is off the declared grid: either it is split more finely than "
                f"MAX_DENOM admits (regularize the coarse LP) or the solver tolerance is looser "
                f"than the grid can resolve (see snap_tolerance).")
        dens.append(frac.denominator)

    G = 1
    for d in dens:
        G = lcm(G, d)

    # Only the live entries take sub-chunks; dust gets exactly zero. Their volumes sum to
    # 1 - dust_total, so the residual below absorbs the dropped dust as well as the floor loss and
    # largest-remainder hands it back to whichever entry the dust was taken from.
    scaled = [vols[i] * G for i in live]
    counts = [floor(x) for x in scaled]
    residual = G - sum(counts)
    if abs(residual) > len(live):
        # Each entry can absorb at most one unit, so a residual this large is not floor loss.
        raise AssertionError(
            f"identity {key[0]} -> cell {key[1]}: cannot place {residual} residual sub-chunks on a "
            f"grid of {G} across {len(live)} assignments; the volumes are not a partition of the "
            f"identity.")
    # Largest fractional remainder first, so the repair lands where the floor lost the most.
    order = sorted(range(len(live)), key=lambda i: scaled[i] - counts[i], reverse=True)
    for i in order[:residual] if residual > 0 else order[::-1][:-residual]:
        counts[i] += 1 if residual > 0 else -1

    # Same budget as the grid check above: an entry that gave up residue may take it back.
    fracs = [Fraction(0)] * n
    for i, c in zip(live, counts):
        if c < 0:
            raise AssertionError(
                f"identity {key[0]} -> cell {key[1]}: assignment {i} snapped to a negative volume "
                f"{c}/{G} from {vols[i]!r}")
        if abs(c / G - vols[i]) > budget:
            raise AssertionError(
                f"identity {key[0]} -> cell {key[1]}: snapping volume {vols[i]!r} to {c}/{G} moved "
                f"it by {abs(c / G - vols[i]):g}, beyond the {budget:g} budget. The group does "
                f"not lie on a common 1/{max_denom} grid.")
        fracs[i] = Fraction(c, G)
    return fracs


def _snap_volumes(assignments: Sequence[_Assignment], snap_tol: float,
                  max_denom: int = MAX_DENOM) -> List[_Assignment]:
    """Populate `_Assignment.exact` for every assignment: the float -> rational boundary.

    This is the seam the whole integerization rests on. Above it everything is float and tolerant
    (the coarse LP, the slot split, the scipy assignment); below it everything is exact Fraction
    arithmetic and no step carries an epsilon. Grouping is by (identity, dst_cell) because that is
    the set `_emit_refined` requires to partition the identity.

    `max_denom` should be `resolve_max_denom(coarse_solver)`, paired with a `snap_tol` sourced from
    the SAME solver (`snap_tolerance(coarse_solver)`) -- the two are coupled (see
    InstanceParams.max_denom), so mixing a tolerance from one solve with a grid size from another
    would check volumes against a budget that was never actually derived for that grid.
    """
    by_id_dst: Dict[Tuple[Identity, int], List[int]] = defaultdict(list)
    for i, a in enumerate(assignments):
        by_id_dst[(a.identity, a.dst_cell)].append(i)

    out = list(assignments)
    for key, idxs in sorted(by_id_dst.items()):
        vols = [assignments[i].volume for i in idxs]
        for i, frac in zip(idxs, _snap_group(vols, snap_tol, key, max_denom)):
            out[i] = replace(assignments[i], exact=frac)
    return out


def _subdivision_factor(assignments: Sequence[_Assignment]) -> int:
    """The refinement Q that makes every assigned volume an integer number of sub-chunks.

    Fractional volumes reach here from two independent relaxations, and BOTH are intrinsic rather
    than artifacts to be fixed upstream:
      * the coarse LP splitting one commodity across parallel paths (the point of an LP relaxation
        -- expected wherever the coarse graph has real multipath, e.g. leaf -> 4 spines);
      * the ABSTRACTION summing several fine links into one coarse link, so that even a perfectly
        integral coarse flow does not decompose into integral fine flows (a coarse link owned by
        two equal gateways forces halves).

    Refining here means nothing downstream ever holds a fractional volume. That is what protects
    the volume-MERGING steps below (_coalesce_egress here, _add_direct in crossbar_solve): both merge
    by max() over an (identity, src, dst) key, which is only correct when a merge cannot combine two
    DISJOINT byte ranges of one identity. Sub-chunk identities make such a merge unrepresentable,
    because the two ranges are distinct commodities with distinct keys. Integerizing at the ncclize
    boundary instead -- where it used to happen -- is structurally too late for that.

    Reads the SNAPPED volume, not the float: `_snap_volumes` already chose the grid (and already
    checked that the solver's answer lies on it), so the LCM here is over denominators that are
    settled rather than re-derived. That is what makes `_emit_refined` exact -- q is by
    construction a multiple of every `exact.denominator`, so `exact * q` is an integer identically
    and there is no tolerance left to get wrong.
    """
    q = 1
    for a in assignments:
        if a.exact is None:
            raise AssertionError(
                "assignment volumes must be snapped by _snap_volumes before the subdivision "
                "factor is computed; build_child_problems does this on entry")
        den = a.exact.denominator
        q = q * den // gcd(q, den)
        if q > MAX_SUBDIVISION:
            raise RuntimeError(
                f"identity resolution needs a sub-chunk refinement of {q} > "
                f"MAX_SUBDIVISION={MAX_SUBDIVISION} to integerize its volumes. The coarse solution "
                f"is split too finely to lower onto whole chunks; regularize the coarse LP (or "
                f"quantize its flow onto a declared 1/Q grid) before resolving identities.")
    return q


def _piece_rate(volume: float, scale: ChunkScale, coarse_epoch: float) -> float:
    """This level's pacing rule: send a piece at exactly the rate that fills ONE COARSE
    EPOCH with its volume.

    Rate bookkeeping belongs to the level that produced the flow. This level solved on a
    coarse epoch grid, so its flows are paced to that grid -- a consumer downstream sees a
    fine epoch axis and could not reconstruct this number. (The level BELOW, the memoized
    NVSwitch schedule, deliberately paces nothing: its flows carry ordering only, so they
    get no rate at all.)

    The rule is self-clocking (a send completes exactly at its epoch boundary, so a chain of
    same-link sends stays on the grid) and heterogeneity-blind: it never reads a link
    bandwidth, yet the coarse solve's own feasibility, sum(volume) <= bw * Delta / chunk,
    gives sum(rate) <= bw -- with equality exactly where the solver saturated the link. That
    implication is a statement about THIS level's capacity constraint, which is why this
    level is also the one that asserts it (see _assert_rate_within_capacity).
    """
    return volume * scale.bytes_per_chunk / coarse_epoch


def _assert_rate_within_capacity(result: IdentityResolution, fine_topology: Topology,
                                 coarse_epoch: float) -> None:
    """Per fine link per coarse epoch, the paced rates must fit the link bandwidth.

    This is the capacity guarantee the coarse solve was entitled to assume, re-expressed in
    rate units: it holds by construction if every gateway's share of each coarse link stayed
    within its fine capacity (egress is split proportionally to fine capacity; ingress is
    assigned under an explicit per-epoch ledger, see _pick_ingress). Checking it here is what
    makes "the fine schedule implements the coarse solution" a verified property rather than
    an assumed one.
    """
    egress: Dict[Tuple[int, int, int], float] = defaultdict(float)
    ingress: Dict[Tuple[int, int, int], float] = defaultdict(float)
    for p in result.pieces:
        if p.rate is None:
            continue
        egress[(p.egress_gpu, p.via_switches[0], p.send_epoch)] += p.rate
        ingress[(p.via_switches[-1], p.ingress_gpu, p.arrival_epoch)] += p.rate
    over = []
    for (a, b, k), rate in sorted(egress.items()) + sorted(ingress.items()):
        bw = fine_topology.capacity[a][b]
        # Relative, not absolute: rates here span a 900 GB/s NVLink and a 25 GB/s spine uplink in
        # the same schedule, and one absolute epsilon cannot be meaningful on both.
        if rate > bw * (1.0 + RATE_REL_TOL):
            over.append((a, b, k, round(rate, 4), round(bw, 4)))
    if over:
        raise AssertionError(
            f"paced sends exceed fine link bandwidth on {len(over)} (link, coarse epoch) "
            f"pairs [(src, dst, epoch, rate, bw)]: {over[:6]}. The coarse solve's capacity "
            f"guarantee did not survive the split onto fine links.")


def _emit_refined(assignments: Sequence[_Assignment],
                  targets: Dict[Tuple[Identity, int], Tuple[int, ...]],
                  q: int, scale: ChunkScale, coarse_epoch: float) -> IdentityResolution:
    """Expand each assignment into `volume * q` whole sub-chunk pieces plus their intra demands.

    Sub-chunk indices are allocated per (identity, dst_cell): the assignments delivering one
    identity to one cell PARTITION it (their volumes sum to exactly 1), so consecutive index ranges
    over a deterministic ordering give every sub-chunk exactly one carrier. That the cursor lands
    exactly on q is the partition check."""
    result = IdentityResolution()
    by_id_dst: Dict[Tuple[Identity, int], List[_Assignment]] = defaultdict(list)
    for a in assignments:
        by_id_dst[(a.identity, a.dst_cell)].append(a)

    for (identity, V), group in sorted(by_id_dst.items()):
        # Tie-break on the SNAPPED volume, not the float: sub-chunk indices are allocated in this
        # order, so two runs that snap to the same grid must order identically even if the solver
        # returned volumes differing in the last bits.
        group.sort(key=lambda a: (a.route.first.send_epoch, a.egress_gpu, a.ingress_gpu,
                                  a.route.first.via_switches, a.exact))
        s, ci = identity
        cursor = 0
        for a in group:
            native = a.native
            legs, placed = a.route.legs, a.legs
            if len(legs) != len(placed):
                raise AssertionError(
                    f"identity {identity}: assignment places {len(placed)} legs but its route "
                    f"{a.route.origin} has {len(legs)}")
            # Exact by construction: q is the LCM of every assignment's snapped denominator, so
            # `exact * q` has denominator 1. No tolerance is involved or wanted -- the float
            # volume's error was spent once, at _snap_volumes, and never again. (The guard is kept
            # because it is free and because it pins the invariant for anyone who changes how q is
            # derived.)
            scaled = a.exact * q
            if scaled.denominator != 1:
                raise AssertionError(
                    f"identity {identity} -> cell {V}: snapped volume {a.exact} is not a whole "
                    f"number of 1/{q} sub-chunks. q must be a multiple of every snapped "
                    f"denominator -- see _subdivision_factor.")
            count = int(scaled)
            for _ in range(count):
                sub: Identity = (s, ci * q + cursor)
                cursor += 1
                # ONE ResolvedPiece PER LEG, all sharing this sub-chunk index. That shared index is
                # what makes a transit chain expressible at all: the two halves of one delivery are
                # the same commodity, and `_emit_refined`'s per-(identity, dst_cell) cursor is what
                # guarantees it. Downstream needs nothing else -- each leg is an ordinary network
                # flow, and the demand between them is what orders the two.
                for li, (leg, (eg, ig)) in enumerate(zip(legs, placed)):
                    result.pieces.append(ResolvedPiece(
                        src_cell=leg.src_cell, dst_cell=leg.dst_cell, identity=sub, egress_gpu=eg,
                        ingress_gpu=ig, via_switches=leg.via_switches, volume=1.0,
                        send_epoch=leg.send_epoch, arrival_epoch=leg.arrival_epoch,
                        rate=_piece_rate(1.0, scale, coarse_epoch)))
                    if li == 0:
                        # Source cell: stage the NATIVE data onto the gateway, if it isn't there.
                        if eg != native:
                            result.intra_demands.append(IntraCellDemand(
                                cell=leg.src_cell, kind="egress_stage", identity=sub,
                                src_gpu=native, dst_gpus=(eg,), volume=1.0,
                                deadline_epoch=leg.send_epoch))
                    elif eg != placed[li - 1][1]:
                        # TRANSIT cell: forward from the GPU that landed the previous leg onto the
                        # one that sends this one. The cell's ROLE here is "sender of the outgoing
                        # leg", which is what egress_stage means -- it is not a new kind. What IS
                        # new is that the data is not there yet, hence the explicit release.
                        # Unreachable while co-location is required (_bridging_gpus makes the two
                        # GPUs equal), and kept so the record exists for the two-epoch dwell.
                        result.intra_demands.append(IntraCellDemand(
                            cell=leg.src_cell, kind="egress_stage", identity=sub,
                            src_gpu=placed[li - 1][1], dst_gpus=(eg,), volume=1.0,
                            deadline_epoch=leg.send_epoch,
                            release_band=legs[li - 1].arrival_epoch + 1, hard=True))
                # Destination cell: fan the identity out to the GPUs that actually want it.
                result.intra_demands.append(IntraCellDemand(
                    cell=V, kind="ingress_distribution", identity=sub, src_gpu=a.ingress_gpu,
                    dst_gpus=targets[(identity, V)], volume=1.0,
                    deadline_epoch=a.route.last.arrival_epoch))
        if cursor != q:
            raise AssertionError(
                f"identity {identity} -> cell {V}: assignments cover {cursor}/{q} sub-chunks; "
                f"they must partition the identity exactly once")
    return result


def assign_identities_free(coarse_solver, mapping: HierarchyMapping,
                           fine_demand, fine_topology: Topology,
                           level_chunk: int = 1, relabel=None, holder_of=None
                           ) -> Tuple[List[_Assignment],
                                      Dict[Tuple[Identity, int], Tuple[int, ...]],
                                      float]:
    """STEP A for an IDENTITY-FREE level: recover which fine identity rode which coarse piece.

    The coarse LP is solved on a demand whose chunk axis `coarsify_demand` collapsed to a single
    aggregated slot per (cell, cell) pair, so its solution says "8 units flowed U -> sw -> V in
    epoch 3" and nothing about WHICH chunks. This step puts identity back: expand the pieces into
    (piece, egress gateway) slots, assign identities to slots by the joint lexicographic min-cost
    program (_solve_assignment), and choose each piece's landing GPU under a global fine downlink
    budget (_pick_ingress).

    A coarse path that store-and-forwards through an intermediate CELL is handled by assigning
    identities to whole ROUTES (see _CoarseRoute); the LP still sees one gateway per slot, and the
    downstream legs are placed afterwards by _place_downstream_legs. That decomposition gives up
    exactly one thing, worth stating here rather than only at the pass: when a commodity (A, C)
    SPLITS across routes with different downstream ingress costs at C, the first-leg LP chooses
    identities blind to that difference. With a single route shape it is exactly optimal, and the
    loss is confined to the objective's soft tier -- it can never cost feasibility, only an
    avoidable intra-cell hop at the destination.

    Returns the `_Assignment` list -- the A/B interface -- plus the per-(identity, cell) target
    map and the level's epoch duration, both of which step B needs. A level whose solver PRESERVES
    identity skips this entirely and builds its `_Assignment`s directly (see
    assign_identities_preserving); either way `build_child_problems` is what runs next.

    coarse_solver:  the solved LPFormulation (needs .per_chunk_flow_paths, .topology,
                    .epoch_duration).
    mapping:        HierarchyMapping from abstract().
    fine_demand:    the fine demand tensor build_demand produced (same one coarsify_demand used).
    fine_topology:  the fine Topology (for per-link capacities and switch ids).
    level_chunk:    how many FINE identities one unit of the coarse solution's volume represents
                    -- the `g` from abstract.set_level_chunk, 1 when the coarse level was solved
                    in the fine chunk unit. It is needed ONLY to check the coarse volumes against
                    the identity count; the same rescale then converts everything to identity
                    units, so no other step here is unit-aware.
    relabel:        tensor coordinate -> identity, required below the root (see identity_sets).
    holder_of:      identity -> the node HOLDING it in this level's index space, required below the
                    root for the same reason `_Assignment.holder` exists: `identity[0]` names the
                    identity's native source, which below the root is neither where the data
                    currently is nor even an index in this level's space.
    """
    id_sets, targets = identity_sets(fine_demand, mapping, relabel=relabel)
    routes_by_pair = _extract_routes(coarse_solver, mapping)

    epoch_duration = getattr(coarse_solver, "epoch_duration", None)
    if epoch_duration is None:
        raise RuntimeError(
            "coarse_solver has no .epoch_duration; it is required to bound each fine ingress "
            "downlink per epoch (a coarse link's capacity is the SUM of the fine downlinks behind "
            "it, so without this the resolution can oversubscribe one of them).")

    # Spans the whole resolution, not one demand pair: a destination cell's ingress link is shared
    # by every source cell sending to it.
    ingress_ledger: Dict[Tuple[int, int, int], float] = {}
    # The egress mirror, for FORWARDED legs only. A first leg's egress needs no ledger: the
    # proportional slot split already bounds it by construction (see _Slot). A forwarded leg has
    # no such split behind it, because the coarse level cannot see it. Empty on every topology
    # with no transit.
    egress_ledger: Dict[Tuple[int, int, int], float] = {}
    assignments: List[_Assignment] = []
    # _pick_ingress compares a piece's volume against `room`, a ledger accumulated across every
    # earlier commit in the WHOLE resolution -- by the time a late identity is placed that ledger
    # carries noise from the coarse LP, from `_solve_assignment`'s own scipy LP, and from every
    # prior float addition into the same accumulator. The bare EPS module default is sized for a
    # single comparison, not that accumulation, so use the same amplified budget `_snap_volumes`
    # trusts for exactly this chain (see snap_tolerance / RECONSTRUCTION_NOISE_FACTOR).
    ingress_tol = snap_tolerance(coarse_solver)
    ingress_max_denom = resolve_max_denom(coarse_solver)

    for (U, V), identities in sorted(id_sets.items()):
        routes = routes_by_pair.get((U, V), [])
        if not routes:
            # Coarse solve delivered nothing for this demanded pair. Transit is no longer a
            # candidate explanation -- routes are keyed by logical origin, so a store-and-forward
            # path is filed HERE, under (U, V), rather than split across its physical legs.
            detail = _origin_diagnosis(routes_by_pair, U, V)
            raise RuntimeError(
                f"no coarse pieces for demanded pair {(U, V)} (|ID|={len(identities)}): "
                + (detail if detail else "coarse solve delivered nothing for this pair"))

        # Where each identity lives at THIS level. At the root that is the identity's own source;
        # below it, `holder_of` says, because the data has already been relayed elsewhere.
        native = {d: (holder_of.get(d, d[0]) if holder_of else d[0]) for d in identities}
        slots = _build_slots(routes, U, V, mapping, fine_topology)
        # The coarse solve delivers exactly coarse[U][V] units to V, and coarse[U][V] is
        # |ID(U,V)| / level_chunk -- the identity count re-expressed in the COARSE LEVEL'S OWN
        # chunk (abstract.set_level_chunk), which is the identity count itself in the flat case
        # where level_chunk == 1. Rescale away float noise so the assignment stays feasible; a
        # gross mismatch is an upstream bug.
        #
        # This rescale is also where the level's unit change is ABSORBED: it normalizes the slot
        # capacities to identity units, so every step below -- the assignment, the subdivision
        # factor, the emitted volumes -- is denominated in fine identities exactly as before,
        # whatever chunk the coarse level solved in. That is why coarsening needs no other change
        # here, and why the ChunkScale handed to the stitch stays the FINE root refined by q.
        total_cap = sum(s.capacity for s in slots) * level_chunk
        assert abs(total_cap - len(identities)) < 1e-3, (
            f"pair {(U, V)}: egress volume {total_cap} (level chunk {level_chunk}) != identity "
            f"count {len(identities)}. "
            f"{_origin_diagnosis(routes_by_pair, U, V) or 'coarse volume disagrees with demand count'}")
        if total_cap:
            f = len(identities) / total_cap * level_chunk
            slots = [replace(s, capacity=s.capacity * f) for s in slots]

        tgt = {d: targets[(d, V)] for d in identities}
        x = _solve_assignment(identities, native, tgt, slots)
        # Largest volume first: _pick_ingress commits greedily against a shared per-(gpu, epoch)
        # ledger, so whichever assignment it sees first gets the roomiest pick. Descending volume
        # means a piece that genuinely needs a big chunk of a downlink claims one before smaller
        # pieces have fragmented every candidate's remaining room -- the same fragmentation the
        # best-fit tie-break inside _pick_ingress guards against, but at the granularity of
        # processing order rather than of a single choice. (kv[0][1], kv[0][0]) only breaks ties
        # among equal volumes, for determinism.
        for (d, j), vol in sorted(x.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0])):
            slot = slots[j]
            for leg_gpus, v in _place_downstream_legs(
                    slot.route, d, slot.egress_gpu, vol, set(tgt.get(d, ())),
                    mapping, fine_topology, epoch_duration, ingress_ledger, egress_ledger,
                    tol=ingress_tol, max_denom=ingress_max_denom):
                assignments.append(_Assignment(
                    src_cell=U, dst_cell=V, identity=d, route=slot.route,
                    egress_gpu=leg_gpus[0][0], ingress_gpu=leg_gpus[-1][1], volume=v,
                    leg_gpus=leg_gpus,
                    holder=native[d] if holder_of else -1))

    return assignments, targets, epoch_duration


def make_piece(src_cell: int, dst_cell: int, egress_neighbor: int, ingress_neighbor: int,
               via_switches: Tuple[int, ...], volume: float, send_epoch: int,
               arrival_epoch: int) -> "_CoarsePiece":
    """Public constructor for a coarse piece, for level solvers that know their own routing
    instead of having it walked out of a solved formulation by `_extract_routes` (the crossbar
    solver: its answer is always `U -> switch -> V`, so there is nothing to walk)."""
    return _CoarsePiece(src_cell=src_cell, dst_cell=dst_cell, egress_neighbor=egress_neighbor,
                        ingress_neighbor=ingress_neighbor, via_switches=via_switches,
                        volume=volume, send_epoch=send_epoch, arrival_epoch=arrival_epoch,
                        origin=(src_cell, dst_cell))


def assign_identities_preserving(carried: Sequence[Tuple["_CoarsePiece", Identity]],
                                 holder: Dict[Identity, int],
                                 targets: Dict[Tuple[Identity, int], Tuple[int, ...]],
                                 mapping: HierarchyMapping, fine_topology: Topology,
                                 epoch_duration: float) -> List[_Assignment]:
    """STEP A for a level whose solver KEPT chunk identity (the crossbar; the MILP path too, if it
    ever lands). Each coarse piece already names the identity it carries, so there is no assignment
    problem left -- only the gateway question that `assign_identities_free` answers as a side
    effect: which fine GPU egresses, and which lands it.

    Not sharing `_solve_assignment` here is deliberate and load-bearing. Re-deriving an assignment
    that is already known would let the lexicographic optimizer move an identity onto a different
    piece, and -- worse -- would reintroduce the fractional volumes `_subdivision_factor` has to
    integerize. Reading the assignment off the input keeps every volume whole, so `Q == 1` at these
    levels and they spend nothing from the `refinement_from_root` budget that `MAX_M` caps across
    the WHOLE recursion (see teccl/hierarchy/scale.py).

    `holder` is where the identity physically lives at this level -- the `src_gpu` of the
    IntraCellDemand that produced this flow, NOT `identity[0]`: below the root an identity has
    already been relayed and its native source may not even be in this cell.
    """
    ingress_ledger: Dict[Tuple[int, int, int], float] = {}
    assignments: List[_Assignment] = []
    # No coarse_solver here to source a feasibility_tol from (see assign_identities_free's
    # ingress_tol) -- snap_tolerance()'s no-solver default is the same comfortable margin
    # `_snap_volumes` falls back to for volumes it cannot trace to a solver either.
    ingress_tol = snap_tolerance()

    for piece, identity in carried:
        U, V = piece.src_cell, piece.dst_cell
        # A one-leg route around the caller's piece. This path is character-for-character what it
        # was: a solver that keeps chunk identity routes U -> switch -> V directly, so it never
        # produces a transit, and every route here has exactly one leg.
        route = _CoarseRoute(origin=(U, V), legs=(piece,), volume=piece.volume)
        slots = _build_slots([route], U, V, mapping, fine_topology)
        if not slots:
            raise RuntimeError(
                f"no egress gateway for identity {identity} on cell {U} -> {V} via coarse neighbor "
                f"{piece.egress_neighbor}; boundary_gpu has no entry for that pair")
        # Egress: the identity's current holder if it owns the uplink, else the widest gateway --
        # the same preference _solve_assignment's first lexicographic tier encodes, applied
        # directly because there is only one identity in play.
        native = holder.get(identity, identity[0])
        on_native = [s for s in slots if s.egress_gpu == native]
        slot = on_native[0] if on_native else max(slots, key=lambda s: (s.capacity, -s.egress_gpu))
        cap = _ingress_epoch_capacity(piece, slot.ingress_candidates, mapping, fine_topology,
                                      epoch_duration)
        # Single-leg by construction (see the route built above), so no forward pass is involved:
        # place the one landing directly, exactly as this path always has.
        for h, v in _pick_ingress(piece, slot.ingress_candidates, identity, piece.volume,
                                  set(targets.get(identity, ())), cap, ingress_ledger,
                                  tol=ingress_tol):
            assignments.append(_Assignment(
                src_cell=U, dst_cell=V, identity=identity, route=slot.route,
                egress_gpu=slot.egress_gpu, ingress_gpu=h, volume=v,
                holder=native))
    return assignments


def build_child_problems(assignments: Sequence[_Assignment],
                         targets: Dict[Tuple[Identity, int], Tuple[int, ...]],
                         mapping: HierarchyMapping, fine_demand, fine_topology: Topology,
                         epoch_duration: float,
                         scale: Optional[ChunkScale] = None,
                         relabel=None,
                         snap_tol: Optional[float] = None,
                         max_denom: Optional[int] = None) -> IdentityResolution:
    """STEP B: turn this level's `_Assignment`s into its flows plus THE NEXT LEVEL'S PROBLEM.

    This runs at EVERY level, whatever produced the assignments, and it is not optional plumbing:
    `_subdivision_factor` + `_emit_refined` ARE the integerization the recursion contract demands
    (see teccl/hierarchy/scale.py: "every level receives INTEGER demands, expressed in that level's
    own chunk unit"), and `_coalesce_egress` / `_dedup_deliveries` are what stop the child being
    handed redundant demands it would schedule real traffic for. A level that skips this emits an
    illegal child problem.

    The refinement is reported on the returned `scale`, and every downstream quantity derived from
    chunk size -- fine epoch duration, epochs per coarse epoch, "9-Chunk_Size", algorithmic
    bandwidth -- must be taken from there.

    scale: granularity the incoming volumes are expressed in; defaults to the root
           (fine_topology.chunk_size, one chunk per fine chunk index).
    snap_tol: how far a solved volume may sit off the rational grid and still be snapped onto it.
           None => snap_tolerance()'s no-solver default. A caller holding the solver that produced
           these volumes should pass snap_tolerance(solver) so the grid is checked against the
           tolerance the volumes were actually computed to.
    max_denom: the grid `snap_tol` was sized against. None => resolve_max_denom()'s no-solver
           default (the module MAX_DENOM). Must be paired with a `snap_tol` from the SAME solver
           (see resolve_max_denom / InstanceParams.max_denom) -- passing one solver's tolerance
           with another's grid size checks volumes against a budget nobody actually derived.
    """
    # The float -> exact boundary, and the only tolerant step below this line. Everything after it
    # is Fraction arithmetic, which is what lets the refinement and the partition check be exact.
    assignments = _snap_volumes(
        assignments, snap_tolerance() if snap_tol is None else snap_tol,
        resolve_max_denom() if max_denom is None else max_denom)
    q = _subdivision_factor(assignments)
    root = scale or ChunkScale(bytes_per_chunk=fine_topology.chunk_size,
                               num_chunks=_num_fine_chunks(fine_demand))
    refined = root.refine(q)
    root.assert_conserves(refined)

    result = _emit_refined(assignments, targets, q, refined, epoch_duration)
    result.scale = refined
    result.subdivision = q
    _coalesce_egress(result)
    _emit_self_distribution(result, fine_demand, mapping, q, relabel=relabel)
    _dedup_deliveries(result)
    _assert_rate_within_capacity(result, fine_topology, epoch_duration)
    return result


def resolve_identities(coarse_solver, mapping: HierarchyMapping,
                       fine_demand, fine_topology: Topology,
                       scale: Optional[ChunkScale] = None,
                       level_chunk: int = 1) -> IdentityResolution:
    """Resolve an identity-free coarse solution into concrete fine identities and emit the
    intra-cell demands the next level must satisfy: step A (identity-free variant) then step B.

    Collective-agnostic: every input is read off `fine_demand` / the coarse solution, nothing
    branches on the collective name. See assign_identities_free and build_child_problems for what
    each half does and why the seam sits where it does.
    """
    assignments, targets, epoch_duration = assign_identities_free(
        coarse_solver, mapping, fine_demand, fine_topology, level_chunk=level_chunk)
    return build_child_problems(assignments, targets, mapping, fine_demand, fine_topology,
                                epoch_duration, scale=scale,
                                snap_tol=snap_tolerance(coarse_solver),
                                max_denom=resolve_max_denom(coarse_solver))


def _num_fine_chunks(fine_demand) -> int:
    """Chunks per GPU in the fine demand array (its innermost dimension)."""
    if len(fine_demand) == 0:
        return 0
    return len(fine_demand[0][0])


def _coalesce_egress(result: IdentityResolution) -> None:
    """A GPU node buffers (store-and-forward), so ONE relay native->gateway serves every later
    epoch that gateway egresses the identity. Merge duplicate egress_stage demands by
    (cell, identity, src_gpu, dst_gpu), keeping the earliest deadline.

    Volume is the MAX of the merged demands, not their sum: the gateway holds the identity's data
    once and re-sends the same buffered bytes to each network destination, so summing would
    re-count one physical relay per egress. Since _emit_refined runs first, every demand reaching
    here carries volume 1.0 of a whole SUB-CHUNK, which is what makes max exact -- two demands with
    the same key are necessarily the same bytes. (Before sub-chunk refinement this was only true
    for whole-identity relays: a fractionally split identity whose shares covered disjoint byte
    ranges needed their union, and max under-counted it, because the aggregate model cannot track
    ranges. Refinement replaces that union with distinct commodities.)

    A TRANSIT forward carries a release as well as a deadline, and the merge must respect both
    ends: one relay serves every merged demand only if it runs late enough for the LATEST arrival
    (max release) and early enough for the EARLIEST send (min deadline). Merging on the deadline
    alone -- which is all this did while every egress_stage was native-sourced and released in the
    prologue -- would keep the earliest send while quietly dropping the constraint that the data
    for the later arrival is not there yet, and emit one relay standing in for two that cannot in
    fact be the same send. If the merged window is empty the two are genuinely distinct sends and
    the key is lying, so fail loud rather than pick one."""
    merged: Dict[Tuple, IntraCellDemand] = {}
    others: List[IntraCellDemand] = []
    for dem in result.intra_demands:
        if dem.kind != "egress_stage":
            others.append(dem)
            continue
        key = (dem.cell, dem.identity, dem.src_gpu, dem.dst_gpus)
        prev = merged.get(key)
        if prev is None:
            merged[key] = dem
        else:
            release = _merge_release(prev.release_band, dem.release_band)
            deadline = min(prev.deadline_epoch, dem.deadline_epoch)
            if release is not None and release > deadline:
                raise AssertionError(
                    f"cell {dem.cell}: egress_stage {dem.src_gpu}->{dem.dst_gpus} for identity "
                    f"{dem.identity} merges two relays whose windows are disjoint -- the merged "
                    f"one could not run before band {release} but must complete before epoch "
                    f"{deadline}. One buffered relay cannot serve both sends; they are not the "
                    f"same bytes at the same time (see _coalesce_egress).")
            merged[key] = IntraCellDemand(
                cell=dem.cell, kind="egress_stage", identity=dem.identity,
                src_gpu=dem.src_gpu, dst_gpus=dem.dst_gpus,
                volume=max(prev.volume, dem.volume),
                deadline_epoch=deadline,
                release_band=release,
                # Hardness is a property of the kind here, and both sides are egress_stage. Take
                # the stricter of the two anyway, so an explicitly-soft relay merged with a hard
                # one keeps the deadline the hard one needs.
                hard=(demand_is_hard(prev) or demand_is_hard(dem)))
    result.intra_demands = list(merged.values()) + others


def _merge_release(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """The release of one relay standing in for two: the LATER of the two, with None meaning
    "the kind's default" rather than "unconstrained".

    None is the prologue for an egress_stage -- the earliest band there is -- so a None merged
    with a real band must yield that band, not None. Returning None there would hand the merged
    relay the prologue and let it run before the transit data arrived, which is exactly the bug
    the explicit release exists to prevent."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _dedup_deliveries(result: IdentityResolution) -> int:
    """Make the demand kinds DISJOINT: no (identity, src, dst) delivery is asked for twice.

    An egress_stage relay native->gateway and a self_distribution native->wanters both name the
    gateway when the gateway also wants the data -- which for AllGather is every gateway, since
    every GPU wants every chunk. They are one physical send, so the requirement belongs to exactly
    one demand, and it belongs to the egress_stage: that one is HARD (a network send waits on it)
    and it must exist regardless, while the self_distribution's copy is redundant.

    Doing this HERE rather than in the scheduler is what makes it reliable. crossbar_solve does dedup
    overlapping deliveries, but only on its DIRECT path (_add_direct); a fan-out lowered to a
    binomial tree bypasses that table entirely, so the redundancy survived exactly when the density
    test chose a tree -- 3 of 26 overlaps in the hetero allgather, delivering the same bytes twice
    over one NVLink edge. Removing the overlap from the demand set makes the outcome independent of
    which lowering branch is picked, and leaves the scheduler's own dedup as a safety net rather
    than a load-bearing mechanism.

    Returns the number of redundant targets removed. Note the delivery requirement itself is NOT
    lost: the gateway still appears in the egress_stage's dst_gpus, so delivery-coverage checks
    still demand it.
    """
    staged: Dict[Tuple[Identity, int], set] = defaultdict(set)
    for d in result.intra_demands:
        if d.kind == "egress_stage":
            staged[(d.identity, d.src_gpu)].update(d.dst_gpus)
    if not staged:
        return 0

    kept: List[IntraCellDemand] = []
    removed = 0
    for d in result.intra_demands:
        covered = staged.get((d.identity, d.src_gpu)) if d.kind != "egress_stage" else None
        if not covered:
            kept.append(d)
            continue
        targets = tuple(t for t in d.dst_gpus if t not in covered)
        removed += len(d.dst_gpus) - len(targets)
        if targets:
            kept.append(replace(d, dst_gpus=targets))
        # else: every target was already staged, so the demand is entirely redundant -- drop it.
    result.intra_demands = kept
    return removed


def _emit_self_distribution(result: IdentityResolution, fine_demand,
                            mapping: HierarchyMapping, q: int = 1, relabel=None) -> None:
    """Emit the intra-cell demand coarsify_demand dropped: fine entries whose source and
    destination are both inside one cell. Structural (read straight off the fine array),
    independent of identity resolution; deadline_epoch=0 (available from the start).

    `q` is the sub-chunk refinement in force, so these demands are expressed in the same unit as
    the resolved pieces: one fine identity becomes q whole sub-chunk demands. Self-distribution
    never introduces fractions of its own -- the data is already local -- so it only follows.

    `relabel` is the tensor-coordinate-to-identity map, required below the root for the same reason
    identity_sets needs it."""
    f2c = _cell_of(mapping)
    n_fine = len(fine_demand)
    per_cell_targets: Dict[Tuple[int, Identity, int], List[int]] = defaultdict(list)
    for s in range(n_fine):
        cs = f2c[s]
        chunks = len(fine_demand[s][s]) if n_fine else 0
        for ci in range(chunks):
            ident: Identity = relabel((s, ci)) if relabel else (s, ci)
            for t in range(n_fine):
                if fine_demand[s][t][ci] > 0 and f2c[t] == cs:
                    per_cell_targets[(cs, ident, s)].append(t)
    for (cell, identity, src), tgts in sorted(per_cell_targets.items()):
        s, ci = identity
        for j in range(q):
            result.intra_demands.append(IntraCellDemand(
                cell=cell, kind="self_distribution", identity=(s, ci * q + j), src_gpu=src,
                dst_gpus=tuple(sorted(tgts)), volume=1.0, deadline_epoch=0))


# --------------------------------------------------------------------------------------------
# Identity -> global chunk id (ncclize)
# --------------------------------------------------------------------------------------------
def identity_to_global_chunk(num_gpus: int, s: int, ci: int) -> int:
    """Map an identity (fine source GPU s, chunk index ci) to the flat global chunk id, using the
    same `s + ci * num_gpus` layout demand.py assigns (device index in the low digits, chunk
    round in the high digits). For 1-chunk-per-GPU AllGather this is just s."""
    return s + ci * num_gpus
