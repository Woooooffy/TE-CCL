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

Finally the chunk is REFINED so that every emitted volume is a whole sub-chunk
(_subdivision_factor / _emit_refined). Fractional volumes are intrinsic here -- they come from the
coarse LP relaxation splitting a commodity across parallel paths, and from the abstraction summing
several fine links into one coarse link -- and refining at this boundary is what keeps every
downstream volume-merging step (this module's _coalesce_egress, crossbar_solve's _add_direct) from
having to reason about disjoint byte ranges it cannot represent. The resulting granularity is
reported on IdentityResolution.scale; see teccl/hierarchy/scale.py for why it must be threaded
rather than read off the topology.

Output (pure data, no Gurobi handles): resolved inter-cell pieces carrying a concrete fine chunk
identity on real GPUs/links, plus intra-cell demand descriptors (egress staging, ingress
distribution, self distribution) for the downstream phase-3 solve. This module STOPS before the
phase-3 intra-cell solve and the final flat stitching.

See the design note hierarchical_lp_identity_resolution / hierarchical_phase3_forward_plan.
"""
from collections import defaultdict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from teccl.hierarchy.abstract import HierarchyMapping
from teccl.hierarchy.scale import ChunkScale
from teccl.topologies.topology import Topology

# A fine data identity: (fine source GPU s, fine chunk index ci). This is already the global
# identity used across collectives -- the source index and chunk index together name the data.
Identity = Tuple[int, int]

EPS = 1e-6


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
    """
    cell: int
    kind: str
    identity: Identity
    src_gpu: int
    dst_gpus: Tuple[int, ...]
    volume: float
    deadline_epoch: int                   # egress: <= send_epoch of fed piece; ingress: > arrival_epoch


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


def _extract_pieces(coarse_solver, mapping: HierarchyMapping
                    ) -> Dict[Tuple[int, int], List[_CoarsePiece]]:
    """Group solved coarse flow into per-(U,V) egress pieces, reusing the switch-run grouping
    that lp_formulation.chunk_flow_paths_to_string uses (a piece is a maximal run of hops whose
    receiver is a switch: gpu-cell -> switch -> ... -> gpu-cell). Coarse switch ids on the path
    are translated to fine ids via coarse_passthrough. The LP aggregates one commodity per
    source cell, so the chunk axis c is always 0 here."""
    switch_indices = set(coarse_solver.topology.switch_indices)
    passthrough = mapping.coarse_passthrough
    pieces: Dict[Tuple[int, int], List[_CoarsePiece]] = defaultdict(list)

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
                    pieces[(start_node, end_node)].append(_CoarsePiece(
                        src_cell=start_node, dst_cell=end_node,
                        egress_neighbor=switches[0], ingress_neighbor=switches[-1],
                        via_switches=fine_switches, volume=volume,
                        send_epoch=sending_epoch, arrival_epoch=arrival_epoch,
                        origin=(s, d)))
                start = nxt = nxt + 1
    return pieces


def _origin_diagnosis(pieces_by_pair: Dict[Tuple[int, int], List[_CoarsePiece]],
                      U: int, V: int) -> str:
    """Explain a mismatch between a demanded pair (U, V) and the extracted pieces by surfacing the
    host-transit signature (see _extract_pieces / _CoarsePiece.origin). Returns "" when nothing
    unusual is found, so the normal single-hop case adds no noise.

    Two tells, both meaning the coarse path store-and-forwarded through an intermediate cell (which
    identity resolution does not yet model):
      * DISPLACED: legs whose logical origin is (U, V) were filed under other physical endpoints
        (the (U, V) flow was split at a transit cell), so pieces_by_pair[(U, V)] is missing volume.
      * FOREIGN:   legs filed under (U, V) actually originate from a different logical pair (they
        are transit legs of someone else's flow passing through this boundary)."""
    displaced = defaultdict(float)   # physical key -> volume of legs whose origin is (U, V)
    foreign = defaultdict(float)     # foreign origin -> volume filed under (U, V)
    for key, plist in pieces_by_pair.items():
        for p in plist:
            if p.origin == (U, V) and key != (U, V):
                displaced[key] += p.volume
            if key == (U, V) and p.origin != (U, V):
                foreign[p.origin] += p.volume
    msgs = []
    if displaced:
        legs = ", ".join(f"{k}:vol={v:g}" for k, v in sorted(displaced.items()))
        msgs.append(f"the {(U, V)} flow was SPLIT across physical legs [{legs}] -- it transits an "
                    f"intermediate cell (host store-and-forward), unmodeled by identity resolution")
    if foreign:
        legs = ", ".join(f"origin {o}:vol={v:g}" for o, v in sorted(foreign.items()))
        msgs.append(f"pieces filed under {(U, V)} are TRANSIT legs of other flows [{legs}]")
    return "; ".join(msgs)


# --------------------------------------------------------------------------------------------
# Joint identity assignment (identities x (coarse piece, egress gateway) slots)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Slot:
    """One (coarse piece, egress gateway GPU) pair an identity can be assigned to.

    `capacity` is the piece's volume scaled by this gateway's share of the coarse egress link,
    proportional to fine link capacity. Splitting EVERY piece proportionally (rather than only
    matching aggregate budgets) is what makes the per-(U, V) decomposition sound: for any epoch k,
    gateway g's egress is sum over that epoch's pieces of volume * cap_g/cap_sum, which coarse
    feasibility bounds by cap_g * Delta -- so the per-fine-uplink per-epoch bound holds by
    construction, globally, even though each demand pair is solved independently.

    `ingress_candidates` are the fine boundary GPUs of the DESTINATION cell that own the link from
    this piece's ingress neighbor. Usually a single GPU (the choice is forced by the coarse path);
    where there are several, _pick_ingress selects one.
    """
    piece: "_CoarsePiece"
    egress_gpu: int
    capacity: float
    ingress_candidates: Tuple[int, ...]


def _build_slots(pieces: Sequence["_CoarsePiece"], U: int, V: int,
                 mapping: HierarchyMapping, fine_topology: Topology) -> List[_Slot]:
    """Expand the coarse pieces of one (U, V) pair into per-(piece, egress gateway) slots."""
    slots: List[_Slot] = []
    for p in pieces:
        egress_gws = mapping.boundary_gpu[(U, p.egress_neighbor)]
        for side, nb in (("egress", p.egress_neighbor), ("ingress", p.ingress_neighbor)):
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
        ingress_cands = tuple(mapping.boundary_gpu[(V, p.ingress_neighbor)])
        for g, cap in zip(egress_gws, caps):
            share = p.volume * cap / cap_sum
            if share > EPS:
                slots.append(_Slot(piece=p, egress_gpu=g, capacity=share,
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

    max_epoch = max(s.piece.send_epoch for s in slots)
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
                c += W_EPOCH * (K - s.piece.send_epoch)
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


def _pick_ingress(slot: _Slot, identity: Identity, volume: float,
                  target_gpus: Dict[Identity, Tuple[int, ...]],
                  epoch_capacity: Dict[int, float],
                  ledger: Dict[Tuple[int, int], float]) -> int:
    """Choose the fine GPU a piece lands on, preferring a target and respecting fine downlink
    capacity.

    This replaces taking `boundary_gpu[...][0]` unconditionally, which had two defects. It ignored
    capacity: a coarse link whose capacity is the SUM of several fine downlinks (abstract() sums
    them) could have all of its per-epoch volume dumped on one fine link, oversubscribing it while
    a sibling sat idle -- the fine schedule then being infeasible against the very coarse solution
    it implements. And it ignored the destination, so an identity wanted by a boundary GPU would
    land elsewhere and pay an avoidable intra-cell hop.

    `ledger` is keyed (gpu, arrival_epoch) and spans the WHOLE resolution, not one demand pair,
    because a destination cell's ingress link is shared by every source cell sending to it.
    """
    candidates = list(slot.ingress_candidates)
    if not candidates:
        raise RuntimeError(
            f"no ingress boundary GPU for cell {slot.piece.dst_cell} from coarse neighbor "
            f"{slot.piece.ingress_neighbor}")
    epoch = slot.piece.arrival_epoch
    wanted = set(target_gpus.get(identity, ()))

    def room(h: int) -> float:
        return epoch_capacity[h] - ledger.get((h, epoch), 0.0)

    def commit(h: int) -> int:
        ledger[(h, epoch)] = ledger.get((h, epoch), 0.0) + volume
        return h

    if len(candidates) == 1:
        # Forced by the coarse path; capacity is still checked by the caller's assert.
        return commit(candidates[0])

    fits = [h for h in candidates if room(h) >= volume - EPS]
    on_target = sorted(h for h in fits if h in wanted)
    if on_target:
        return commit(on_target[0])
    if fits:
        # Least loaded, deterministic tie-break, so a sibling downlink is used before overloading.
        return commit(max(sorted(fits), key=lambda h: room(h)))
    raise RuntimeError(
        f"ingress capacity exhausted for cell {slot.piece.dst_cell} epoch {epoch}: "
        f"identity {identity} needs {volume:g} but candidates "
        f"{ {h: round(room(h), 6) for h in candidates} } have no room "
        f"(per-epoch capacities { {h: round(epoch_capacity[h], 6) for h in candidates} }). The "
        f"coarse solve routed more into this cell in one epoch than its fine downlinks can absorb.")


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------
# Mirrors ncclize's MAX_DENOM / MAX_M (teccl/ncclize/teccl_ncclize.py). Integerization now happens
# HERE, at the recursion boundary, rather than at the ncclize boundary -- see _subdivision_factor.
MAX_DENOM = 64
MAX_SUBDIVISION = 128


@dataclass(frozen=True)
class _Assignment:
    """One resolved (identity -> coarse piece) decision, before sub-chunk refinement."""
    src_cell: int
    dst_cell: int
    identity: Identity
    piece: _CoarsePiece
    egress_gpu: int
    ingress_gpu: int
    volume: float
    # Which node in THIS LEVEL's index space physically holds the identity, and therefore stages it
    # onto the gateway. -1 means "identity[0]", which is correct at the ROOT and only there: an
    # identity names its native source GPU, but below the root the data has already been relayed and
    # is held by some other GPU -- in a different index space, at that. Reading the holder off the
    # identity below the root silently emits a staging relay from a node id that means something
    # else entirely at this level.
    holder: int = -1

    @property
    def native(self) -> int:
        return self.holder if self.holder >= 0 else self.identity[0]


def _ingress_epoch_capacity(slot: _Slot, mapping: HierarchyMapping,
                            fine_topology: Topology, epoch_duration: float) -> Dict[int, float]:
    """Per-epoch volume each of a slot's candidate ingress GPUs can absorb, in chunk units.

    Note the direction: the ingress leg is switch -> gpu, so this indexes
    capacity[fine_neighbor][gpu], whereas the egress split in _build_slots uses
    capacity[gpu][fine_neighbor]."""
    fine_nb = mapping.coarse_passthrough[slot.piece.ingress_neighbor]
    return {h: fine_topology.capacity[fine_nb][h] * epoch_duration
            for h in slot.ingress_candidates}


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
    """
    q = 1
    for a in assignments:
        den = Fraction(a.volume).limit_denominator(MAX_DENOM).denominator
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
        if rate > bw + 1e-6:
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
        group.sort(key=lambda a: (a.piece.send_epoch, a.egress_gpu, a.ingress_gpu,
                                  a.piece.via_switches, a.volume))
        s, ci = identity
        cursor = 0
        for a in group:
            native = a.native
            count = int(round(a.volume * q))
            if abs(a.volume * q - count) > 1e-6:
                raise AssertionError(
                    f"volume {a.volume} is not a whole number of 1/{q} sub-chunks "
                    f"(identity {identity} -> cell {V})")
            for _ in range(count):
                sub: Identity = (s, ci * q + cursor)
                cursor += 1
                result.pieces.append(ResolvedPiece(
                    src_cell=a.src_cell, dst_cell=V, identity=sub, egress_gpu=a.egress_gpu,
                    ingress_gpu=a.ingress_gpu, via_switches=a.piece.via_switches, volume=1.0,
                    send_epoch=a.piece.send_epoch, arrival_epoch=a.piece.arrival_epoch,
                    rate=_piece_rate(1.0, scale, coarse_epoch)))
                if a.egress_gpu != native:
                    result.intra_demands.append(IntraCellDemand(
                        cell=a.src_cell, kind="egress_stage", identity=sub, src_gpu=native,
                        dst_gpus=(a.egress_gpu,), volume=1.0,
                        deadline_epoch=a.piece.send_epoch))
                result.intra_demands.append(IntraCellDemand(
                    cell=V, kind="ingress_distribution", identity=sub, src_gpu=a.ingress_gpu,
                    dst_gpus=targets[(identity, V)], volume=1.0,
                    deadline_epoch=a.piece.arrival_epoch))
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
    pieces_by_pair = _extract_pieces(coarse_solver, mapping)

    epoch_duration = getattr(coarse_solver, "epoch_duration", None)
    if epoch_duration is None:
        raise RuntimeError(
            "coarse_solver has no .epoch_duration; it is required to bound each fine ingress "
            "downlink per epoch (a coarse link's capacity is the SUM of the fine downlinks behind "
            "it, so without this the resolution can oversubscribe one of them).")

    # Spans the whole resolution, not one demand pair: a destination cell's ingress link is shared
    # by every source cell sending to it.
    ingress_ledger: Dict[Tuple[int, int], float] = {}
    assignments: List[_Assignment] = []

    for (U, V), identities in sorted(id_sets.items()):
        pieces = pieces_by_pair.get((U, V), [])
        if not pieces:
            # Coarse solve delivered nothing filed under this demanded pair. Either the coarse
            # solve genuinely dropped it, or (the common cause) the path store-and-forwarded
            # through an intermediate cell and got split into legs filed elsewhere.
            detail = _origin_diagnosis(pieces_by_pair, U, V)
            raise RuntimeError(
                f"no coarse pieces for demanded pair {(U, V)} (|ID|={len(identities)}): "
                + (detail if detail else "coarse solve delivered nothing for this pair"))

        # Where each identity lives at THIS level. At the root that is the identity's own source;
        # below it, `holder_of` says, because the data has already been relayed elsewhere.
        native = {d: (holder_of.get(d, d[0]) if holder_of else d[0]) for d in identities}
        slots = _build_slots(pieces, U, V, mapping, fine_topology)
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
            f"{_origin_diagnosis(pieces_by_pair, U, V) or 'coarse volume disagrees with demand count'}")
        if total_cap:
            f = len(identities) / total_cap * level_chunk
            slots = [replace(s, capacity=s.capacity * f) for s in slots]

        tgt = {d: targets[(d, V)] for d in identities}
        x = _solve_assignment(identities, native, tgt, slots)
        for (d, j), vol in sorted(x.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            slot = slots[j]
            cap = _ingress_epoch_capacity(slot, mapping, fine_topology, epoch_duration)
            h = _pick_ingress(slot, d, vol, tgt, cap, ingress_ledger)
            assignments.append(_Assignment(
                src_cell=U, dst_cell=V, identity=d, piece=slot.piece,
                egress_gpu=slot.egress_gpu, ingress_gpu=h, volume=vol,
                holder=native[d] if holder_of else -1))

    return assignments, targets, epoch_duration


def make_piece(src_cell: int, dst_cell: int, egress_neighbor: int, ingress_neighbor: int,
               via_switches: Tuple[int, ...], volume: float, send_epoch: int,
               arrival_epoch: int) -> "_CoarsePiece":
    """Public constructor for a coarse piece, for level solvers that know their own routing
    instead of having it walked out of a solved formulation by `_extract_pieces` (the crossbar
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
    ingress_ledger: Dict[Tuple[int, int], float] = {}
    assignments: List[_Assignment] = []

    for piece, identity in carried:
        U, V = piece.src_cell, piece.dst_cell
        slots = _build_slots([piece], U, V, mapping, fine_topology)
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
        cap = _ingress_epoch_capacity(slot, mapping, fine_topology, epoch_duration)
        h = _pick_ingress(slot, identity, piece.volume, targets, cap, ingress_ledger)
        assignments.append(_Assignment(
            src_cell=U, dst_cell=V, identity=identity, piece=piece,
            egress_gpu=slot.egress_gpu, ingress_gpu=h, volume=piece.volume,
            holder=native))
    return assignments


def build_child_problems(assignments: Sequence[_Assignment],
                         targets: Dict[Tuple[Identity, int], Tuple[int, ...]],
                         mapping: HierarchyMapping, fine_demand, fine_topology: Topology,
                         epoch_duration: float,
                         scale: Optional[ChunkScale] = None,
                         relabel=None) -> IdentityResolution:
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
    """
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
                                epoch_duration, scale=scale)


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
    ranges. Refinement replaces that union with distinct commodities.)"""
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
            merged[key] = IntraCellDemand(
                cell=dem.cell, kind="egress_stage", identity=dem.identity,
                src_gpu=dem.src_gpu, dst_gpus=dem.dst_gpus,
                volume=max(prev.volume, dem.volume),
                deadline_epoch=min(prev.deadline_epoch, dem.deadline_epoch))
    result.intra_demands = list(merged.values()) + others


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
