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

For each ordered cell pair (U, V) the problem is a min-cost transportation: assign U's identities
(supply 1 each, since V must receive each exactly once) to the delivered egress volume so as to
minimize forced intra-cell egress relay (an identity leaving via a gateway GPU that is not its
native source GPU must first be relayed to that gateway inside U).

Output (pure data, no Gurobi handles): resolved inter-cell pieces carrying a concrete fine chunk
identity on real GPUs/links, plus intra-cell demand descriptors (egress staging, ingress
distribution, self distribution) for the downstream phase-3 solve. This module STOPS before the
phase-3 intra-cell solve and the final flat stitching.

Egress epoch ordering (see back-distribution in resolve_identities): at each gateway GPU, NATIVE
identities are pinned to the earliest egress epochs and RELAYED identities to the later ones, so
an early slot is never spent on data not yet on the gateway and relays get maximal staging slack.
The order AMONG relayed identities is left as identity index for now -- the runtime-optimal choice
depends on when each relay actually completes, which is only known once phase-3 fixes the
intra-cell (NVSwitch) schedule. That is the intended home for an intra-cell ordering-heuristic
knob (earliest-relay-ready-first, longest-intra-path-first, deadline-driven, ...); revisit when
phase-3 lands.

See the design note hierarchical_lp_identity_resolution / hierarchical_phase3_forward_plan.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from teccl.hierarchy.abstract import HierarchyMapping
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
    identity: Identity                    # (s, ci); -> ncclize global chunk id via identity_to_global_chunk
    egress_gpu: int                       # fine GPU in src_cell that physically sends (== s iff no relay)
    ingress_gpu: int                      # fine GPU in dst_cell that physically receives
    via_switches: Tuple[int, ...]         # FINE switch ids along the coarse path
    volume: float
    send_epoch: int                       # coarse epoch it leaves src_cell
    arrival_epoch: int                    # coarse epoch it is consumed at dst_cell


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


# --------------------------------------------------------------------------------------------
# Demand-shape derivation (collective-agnostic)
# --------------------------------------------------------------------------------------------
def _cell_of(mapping: HierarchyMapping) -> Dict[int, int]:
    return mapping.fine_to_coarse


def identity_sets(fine_demand, mapping: HierarchyMapping
                  ) -> Tuple[Dict[Tuple[int, int], List[Identity]],
                             Dict[Tuple[Identity, int], Tuple[int, ...]]]:
    """Derive, straight off the fine demand array, the per-(U,V) identity set and per-identity
    destination-GPU set. Mirrors abstract.coarsify_demand's counting exactly so that
    len(ID[(U,V)]) == coarse[U][V].

    Returns:
      id_sets[(U, V)]      = sorted list of identities (s, ci) with coarse(s)=U that some GPU in
                             V wants (U != V).
      targets[((s,ci), V)] = tuple of fine GPUs t in cell V with fine_demand[s][t][ci] > 0.
    """
    f2c = _cell_of(mapping)
    n_fine = len(fine_demand)
    id_sets: Dict[Tuple[int, int], List[Identity]] = defaultdict(list)
    targets: Dict[Tuple[Identity, int], Tuple[int, ...]] = {}
    for s in range(n_fine):
        cs = f2c[s]
        chunks = len(fine_demand[s][s]) if n_fine else 0
        for ci in range(chunks):
            # who wants (s, ci), grouped by destination cell
            wanters_by_cell: Dict[int, List[int]] = defaultdict(list)
            for t in range(n_fine):
                if fine_demand[s][t][ci] > 0:
                    wanters_by_cell[f2c[t]].append(t)
            for cv, ts in wanters_by_cell.items():
                if cv == cs:
                    continue
                id_sets[(cs, cv)].append((s, ci))
                targets[((s, ci), cv)] = tuple(sorted(ts))
    for key in id_sets:
        id_sets[key].sort()
    return id_sets, targets


# --------------------------------------------------------------------------------------------
# Coarse-piece extraction from the solved LP formulation
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _CoarsePiece:
    src_cell: int
    dst_cell: int
    egress_neighbor: int                  # coarse switch id of the first hop out of src_cell
    ingress_neighbor: int                 # coarse switch id of the last hop into dst_cell
    via_switches: Tuple[int, ...]         # FINE switch ids
    volume: float
    send_epoch: int
    arrival_epoch: int


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
                        send_epoch=sending_epoch, arrival_epoch=arrival_epoch))
                start = nxt = nxt + 1
    return pieces


# --------------------------------------------------------------------------------------------
# Min-cost transportation (identities x egress gateway GPUs), solved exactly
# --------------------------------------------------------------------------------------------
def _solve_transport(identities: Sequence[Identity],
                     native_gpu: Dict[Identity, int],
                     gateways: Sequence[int],
                     load: Dict[int, float]) -> Dict[Tuple[Identity, int], float]:
    """Exact min-cost transportation.

    rows d in identities  (supply 1 each: V receives each identity once)
    cols g in gateways     (demand load[g]: fill each gateway's coarse-solved egress budget)
    cost(d, g) = 0 if g == native_gpu[d] else 1     (1 = a forced intra-cell relay)

    sum_g load[g] == len(identities) by construction (coarse volume == identity count), so the
    problem is balanced and feasible. Returns x[(d, g)] for the positive entries.
    """
    D = list(identities)
    G = list(gateways)
    nD, nG = len(D), len(G)
    if nD == 0:
        return {}
    # flatten variable (d, g) -> index d*nG + g
    cost = np.empty(nD * nG, dtype=float)
    for di, d in enumerate(D):
        nat = native_gpu[d]
        for gi, g in enumerate(G):
            cost[di * nG + gi] = 0.0 if g == nat else 1.0

    # equality: each identity's row sums to 1
    A_eq = []
    b_eq = []
    for di in range(nD):
        row = np.zeros(nD * nG)
        row[di * nG:(di + 1) * nG] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    # equality: each gateway column sums to its load
    for gi, g in enumerate(G):
        row = np.zeros(nD * nG)
        for di in range(nD):
            row[di * nG + gi] = 1.0
        A_eq.append(row)
        b_eq.append(load[g])

    res = linprog(cost, A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"identity transportation infeasible: {res.message}")
    x: Dict[Tuple[Identity, int], float] = {}
    for di, d in enumerate(D):
        for gi, g in enumerate(G):
            v = res.x[di * nG + gi]
            if v > EPS:
                x[(d, g)] = float(v)
    return x


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------
def _gateway_loads(pieces: List[_CoarsePiece], mapping: HierarchyMapping,
                   fine_topology: Topology, coarse_solver
                   ) -> Tuple[Dict[int, float], Dict[int, List[_CoarsePiece]]]:
    """Split each coarse egress link's total volume across the fine gateway GPUs that own it,
    proportional to their fine link capacity (an even split for equal-capacity gateways). Returns
    per-gateway egress budget `load[g]` and, for back-distribution, the pieces feeding each
    coarse egress neighbor."""
    src_cell = pieces[0].src_cell
    by_neighbor: Dict[int, List[_CoarsePiece]] = defaultdict(list)
    for p in pieces:
        by_neighbor[p.egress_neighbor].append(p)

    load: Dict[int, float] = defaultdict(float)
    for neighbor, plist in by_neighbor.items():
        total = sum(p.volume for p in plist)
        gws = mapping.boundary_gpu[(src_cell, neighbor)]
        fine_neighbor = mapping.coarse_passthrough[neighbor]
        caps = [fine_topology.capacity[g][fine_neighbor] for g in gws]
        cap_sum = sum(caps) or 1.0
        for g, cap in zip(gws, caps):
            load[g] += total * cap / cap_sum
    return dict(load), by_neighbor


def _distribute_ingress_gpu(dst_cell: int, ingress_neighbor: int, mapping: HierarchyMapping) -> int:
    """Pick the fine ingress gateway a piece lands on (identity-independent: the destination owns
    none of the data). For a single-GPU boundary this is forced; for a multi-GPU boundary any
    owner works -- take the first deterministically (capacity balancing across arrivals is a
    later refinement)."""
    return mapping.boundary_gpu[(dst_cell, ingress_neighbor)][0]


def resolve_identities(coarse_solver, mapping: HierarchyMapping,
                       fine_demand, fine_topology: Topology) -> IdentityResolution:
    """Resolve the identity-free coarse LP solution into concrete fine identities and emit the
    intra-cell demands phase-3 must satisfy. Collective-agnostic: every input is read off
    `fine_demand` / the coarse solution, nothing branches on the collective name.

    coarse_solver:  the solved LPFormulation (has .per_chunk_flow_paths and .topology).
    mapping:        HierarchyMapping from abstract().
    fine_demand:    the fine demand tensor build_demand produced (same one coarsify_demand used).
    fine_topology:  the fine Topology (for per-link capacities and switch ids).
    """
    id_sets, targets = identity_sets(fine_demand, mapping)
    pieces_by_pair = _extract_pieces(coarse_solver, mapping)
    result = IdentityResolution()

    for (U, V), identities in id_sets.items():
        pieces = pieces_by_pair.get((U, V), [])
        if not pieces:
            # Coarse solve delivered nothing for a demanded pair -> upstream inconsistency.
            raise RuntimeError(f"no coarse pieces for demanded pair {(U, V)} "
                               f"(|ID|={len(identities)})")

        native = {d: d[0] for d in identities}       # native GPU of identity (s, ci) is s
        load, by_neighbor = _gateway_loads(pieces, mapping, fine_topology, coarse_solver)
        # The coarse solve delivers exactly coarse[U][V] == |ID(U,V)| units to V, so the gateway
        # loads must sum to the identity count. Rescale away float noise so the balanced
        # transportation stays feasible; a gross mismatch is an upstream bug.
        total_load = sum(load.values())
        assert abs(total_load - len(identities)) < 1e-3, (
            f"pair {(U, V)}: egress volume {total_load} != identity count {len(identities)}")
        scale = len(identities) / total_load if total_load else 1.0
        load = {g: v * scale for g, v in load.items()}
        gateways = list(load.keys())
        x = _solve_transport(identities, native, gateways, load)

        # Per-gateway queue of (piece, remaining volume), earliest send epoch first, for
        # deterministic back-distribution of an identity's assigned volume onto concrete pieces.
        gw_pieces: Dict[int, List[List]] = {}
        for neighbor, plist in by_neighbor.items():
            gws = mapping.boundary_gpu[(U, neighbor)]
            fine_neighbor = mapping.coarse_passthrough[neighbor]
            caps = [fine_topology.capacity[g][fine_neighbor] for g in gws]
            cap_sum = sum(caps) or 1.0
            ordered = sorted(plist, key=lambda p: p.send_epoch)
            for g, cap in zip(gws, caps):
                # extend (not overwrite): a gateway GPU homed to several neighbors accumulates
                # its slot capacity across all of them.
                gw_pieces.setdefault(g, []).extend(
                    [[p, p.volume * cap / cap_sum] for p in ordered])

        # Choose which epoch slot each identity takes on its gateway. Ordering rule per gateway g:
        #   1. NATIVE identities first (g == native[d]): they already sit on g at t=0, so they
        #      claim the EARLIEST egress epochs -- an early slot is never spent on data that is
        #      not yet on the gateway.
        #   2. RELAYED identities after: they take the later slots, which maximizes the staging
        #      slack of their egress_stage relay (the relay only has to finish before the -- now
        #      later -- epoch g actually sends them). This is the "own volume out first, relayed
        #      volume in later epochs" ordering.
        # The order AMONG relayed identities (and native ties) is left as identity index for now.
        # That secondary order is a PLACEHOLDER: the runtime-optimal relayed-egress order depends
        # on when each relay can actually complete, which is only known once phase-3 fixes the
        # intra-cell (NVSwitch) schedule. That is the natural home for an intra-cell ordering
        # heuristic knob (e.g. earliest-relay-ready-first, longest-intra-path-first, deadline
        # driven) selecting among strategies; wire it here when phase-3 lands. See module docstring.
        def _egress_order(kv):
            (d, g), _vol = kv
            return (g, 0 if g == native[d] else 1, d)   # per gateway: native (0) before relayed (1)

        for (d, g), vol in sorted(x.items(), key=_egress_order):
            # Egress-stage relay demand if the gateway is not the identity's native GPU.
            if g != native[d]:
                # deadline = earliest send epoch of the pieces this identity will feed on g
                deadline = min(p.send_epoch for p, rem in gw_pieces[g] if rem > EPS)
                result.intra_demands.append(IntraCellDemand(
                    cell=U, kind="egress_stage", identity=d, src_gpu=native[d],
                    dst_gpus=(g,), volume=vol, deadline_epoch=deadline))
            # Back-distribute `vol` onto concrete pieces on gateway g (earliest-first).
            remaining = vol
            for slot in gw_pieces[g]:
                if remaining <= EPS:
                    break
                p, rem = slot
                take = min(rem, remaining)
                if take <= EPS:
                    continue
                slot[1] -= take
                remaining -= take
                ingress_gpu = _distribute_ingress_gpu(V, p.ingress_neighbor, mapping)
                result.pieces.append(ResolvedPiece(
                    src_cell=U, dst_cell=V, identity=d, egress_gpu=g,
                    ingress_gpu=ingress_gpu, via_switches=p.via_switches, volume=take,
                    send_epoch=p.send_epoch, arrival_epoch=p.arrival_epoch))
                # Ingress-distribution demand: fan the identity from the ingress gateway to the
                # fine GPUs of V that actually want it (all of V for allgather, one for alltoall).
                tgt = targets[(d, V)]
                result.intra_demands.append(IntraCellDemand(
                    cell=V, kind="ingress_distribution", identity=d, src_gpu=ingress_gpu,
                    dst_gpus=tgt, volume=take, deadline_epoch=p.arrival_epoch))

    _coalesce_egress(result)
    _emit_self_distribution(result, fine_demand, mapping)
    return result


def _coalesce_egress(result: IdentityResolution) -> None:
    """A GPU node buffers (store-and-forward), so one relay native->gateway serves every later
    epoch that gateway egresses the identity. Merge duplicate egress_stage demands by
    (cell, identity, src_gpu, dst_gpu), summing volume and keeping the earliest deadline."""
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
                volume=prev.volume + dem.volume,
                deadline_epoch=min(prev.deadline_epoch, dem.deadline_epoch))
    result.intra_demands = list(merged.values()) + others


def _emit_self_distribution(result: IdentityResolution, fine_demand,
                            mapping: HierarchyMapping) -> None:
    """Emit the intra-cell demand coarsify_demand dropped: fine entries whose source and
    destination are both inside one cell. Structural (read straight off the fine array),
    independent of identity resolution; deadline_epoch=0 (available from the start)."""
    f2c = _cell_of(mapping)
    n_fine = len(fine_demand)
    per_cell_targets: Dict[Tuple[int, Identity, int], List[int]] = defaultdict(list)
    for s in range(n_fine):
        cs = f2c[s]
        chunks = len(fine_demand[s][s]) if n_fine else 0
        for ci in range(chunks):
            for t in range(n_fine):
                if fine_demand[s][t][ci] > 0 and f2c[t] == cs:
                    per_cell_targets[(cs, (s, ci), s)].append(t)
    for (cell, identity, src), tgts in per_cell_targets.items():
        result.intra_demands.append(IntraCellDemand(
            cell=cell, kind="self_distribution", identity=identity, src_gpu=src,
            dst_gpus=tuple(sorted(tgts)), volume=1.0, deadline_epoch=0))


# --------------------------------------------------------------------------------------------
# Identity -> global chunk id (ncclize)
# --------------------------------------------------------------------------------------------
def identity_to_global_chunk(num_gpus: int, s: int, ci: int) -> int:
    """Map an identity (fine source GPU s, chunk index ci) to the flat global chunk id, using the
    same `s + ci * num_gpus` layout demand.py assigns (device index in the low digits, chunk
    round in the high digits). For 1-chunk-per-GPU AllGather this is just s."""
    return s + ci * num_gpus
