"""
A hierarchy level solved by the chunk-level MILP (AllGatherFormulation): its coarse demand, how its
solution is read back, and the integral identity resolution that lowers it.

The MILP is the LP's peer at a level, not a separate pipeline: its output is the same
IdentityResolution, built with the same helpers. What differs, and lives here:

  * DEMAND. The MILP's chunks are binary (`total_demand_sat` is a {0,1} var pinned to the demand),
    so it cannot take the LP's weighted single slot -- a demand of 4 is structurally infeasible.
    `coarsify_demand_preserving` builds a COUNT tensor instead: one column per BUNDLE of `b` fine
    identities, b chosen by the level's `milp_chunking` policy.

  * READ-BACK. The MILP copies at a cell (a node may send what it holds to several neighbours)
    and, with switch copy, at a switch. Each column's solution is therefore a forest of SENDS
    (teccl/sends.py), which the extraction outputs directly (`AllGatherFormulation.sends`,
    `level_sends`): one Send per transmission out of a cell, a switch copy being one send with
    several branches. Co-located forwarding is `send >= ready` and a relayed one
    `send >= ready + 1`, exactly as for the LP.

  * IDENTITY. The MILP keeps chunk identity only up to interchangeable columns. Two columns of one
    cell with the same destination-cell set look identical to the coarse level, which collapsed the
    cell's GPUs into one node: it cannot see that putting gpu1's data on gpu0's uplink and gpu0's on
    gpu1's costs two staging relays the uncrossed labelling does not. So step A is still an
    assignment (`_solve_integral_assignment`) -- but of WHOLE identities to WHOLE columns, which is
    a transportation problem and integral, so this level never refines (Q == 1) and never snaps.
    The placement is then one causal walk over each column's sends (`_place_column`); it shares the
    LP's placement helpers (`_leg_gateways`, `_pick_ingress`, `_commit_egress`, ...) and the step-B
    tail (`finish_resolution`), but not its route machinery, because a send is placed exactly once.

See teccl/hierarchy/milp_design.md.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field, replace
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from teccl.hierarchy.abstract import HierarchyMapping, classify_identities
from teccl.hierarchy.reconstruct import (
    Identity, IdentityResolution, IntraCellDemand, _CoarsePiece, _TierWeights, _commit_egress,
    _egress_epoch_capacity, _ingress_epoch_capacity, _leg_gateways, _leg_ingress_candidates,
    _num_fine_chunks, _pick_ingress, finish_resolution, identity_sets, snap_tolerance)
from teccl.hierarchy.scale import ChunkScale
from teccl.sends import Branch, Send, delivery_index
from teccl.topologies.topology import Topology

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------
# M1: the count tensor
# ------------------------------------------------------------------------------------------------
@dataclass
class MilpGroup:
    """The identities of one cell that share a destination-cell set, and the MILP columns that
    carry them. Columns of one group are interchangeable to the MILP; which identity rides which
    column's sends is decided after the solve, by `lower_integral`."""
    cell: int
    dest_cells: Tuple[int, ...]
    identities: List[Identity]
    holders: Dict[Identity, int]          # identity -> node holding it, in this level's indices
    columns: List[int] = field(default_factory=list)


@dataclass
class PreservingDemand:
    """A MILP level's demand: the {0,1} count tensor the formulation solves, plus what each
    column stands for. `b` is the bundle size -- identities per column -- and the factor the level
    is rescaled by (abstract.rescale_level)."""
    tensor: List[List[List[int]]]
    groups: List[MilpGroup]
    b: int
    policy: Union[str, int]

    def total_identity_crossings(self) -> int:
        return sum(len(g.identities) * len(g.dest_cells) for g in self.groups)


def coarsify_demand_preserving(fine_demand, mapping: HierarchyMapping,
                               policy: Union[str, int] = "holder",
                               relabel=None) -> PreservingDemand:
    """The MILP's coarse demand: identities bundled `b` to a column, `b` set by `policy`.

    Every policy is one mechanism, `(class key, b)`:

        "fine"    b = 1
        "holder"  key (holder node, dest-cell set); b = the class size, their gcd if they differ
        "cell"    key (holder cell, dest-cell set); b likewise
        int       key (holder node, dest-cell set); b as given, which must divide every class

    The key only decides `b`. Columns are counted per (cell, dest-cell set) GROUP, because that is
    all the coarse level can tell apart; bundle composition is left to the resolution, whose egress
    tier already prefers single-holder sets. Anchored to the LP's coarse demand, which is the same
    fold over `classify_identities`:  sum over groups |identities| * |dests| == sum coarsify_demand.
    """
    from teccl.hierarchy.abstract import coarsify_demand

    if policy not in ("fine", "holder", "cell") and not (isinstance(policy, int) and policy > 0):
        raise ValueError(f"unknown milp_chunking policy {policy!r}")

    groups: Dict[Tuple[int, Tuple[int, ...]], MilpGroup] = {}
    class_sizes: Dict[Tuple, int] = defaultdict(int)
    for identity, holder, cs, wanters in classify_identities(fine_demand, mapping, relabel=relabel):
        dests = tuple(sorted(cv for cv in wanters if cv != cs))
        if not dests:
            continue                                  # intra-cell only: never crosses this level
        g = groups.setdefault((cs, dests), MilpGroup(cell=cs, dest_cells=dests, identities=[],
                                                     holders={}))
        g.identities.append(identity)
        g.holders[identity] = holder
        class_sizes[(cs, dests) if policy == "cell" else (holder, dests)] += 1

    sizes = sorted(set(class_sizes.values()))
    if policy == "fine":
        b = 1
    elif isinstance(policy, int):
        b = policy
        bad = {k: n for k, n in class_sizes.items() if n % b}
        if bad:
            raise ValueError(
                f"milp_chunking={b} does not divide every holder's class size; classes "
                f"(holder, dest cells) -> size: {dict(list(bad.items())[:6])}. Pick a common "
                f"divisor of {sizes}, or a named policy.")
    else:
        b = 0
        for n in sizes:
            b = gcd(b, n)
        b = b or 1
        if len(sizes) > 1:
            logger.warning(
                "milp_chunking=%r: class sizes differ (%s), so one class cannot be one chunk; "
                "using their gcd b=%d (a class of n identities gets n/%d chunks)",
                policy, sizes, b, b)

    width = 1
    per_cell: Dict[int, int] = defaultdict(int)
    for (cs, _d), g in sorted(groups.items()):
        g.identities.sort()
        n = len(g.identities) // b
        g.columns = list(range(per_cell[cs], per_cell[cs] + n))
        per_cell[cs] += n
        width = max(width, per_cell[cs])

    num = mapping.num_coarse
    tensor = [[[0] * width for _ in range(num)] for _ in range(num)]
    for g in groups.values():
        for c in g.columns:
            for v in g.dest_cells:
                tensor[g.cell][v][c] = 1

    demand = PreservingDemand(tensor=tensor, groups=[groups[k] for k in sorted(groups)], b=b,
                              policy=policy)
    lp_total = sum(v[0] for row in coarsify_demand(fine_demand, mapping) for v in row)
    assert demand.total_identity_crossings() == lp_total, (
        f"preserving demand covers {demand.total_identity_crossings()} identity crossings but the "
        f"weighted coarse demand counts {lp_total}; the two folds of classify_identities disagree")
    return demand


# ------------------------------------------------------------------------------------------------
# M2: the solution as sends
# ------------------------------------------------------------------------------------------------
Columns = Dict[Tuple[int, int], List[Send]]      # (cell, column) -> its sends, epoch (= causal) order


def level_sends(milp, demand: PreservingDemand) -> Columns:
    """Read a solved MILP level back as its sends (AllGatherFormulation.sends), grouped by column.

    A column's sends form a forest rooted at its cell: each is one transmission out of a cell,
    fanning out through coarse switches, and every cell receives the column at most once (the
    forest invariant the extraction builds). Epoch order is causal order -- a cell forwards only
    after it holds the chunk. Checked here, failing loud: every demanded (column, destination cell)
    is delivered. An untraced demand is a delivery the hierarchy would otherwise silently drop.
    """
    if getattr(milp, "untraced_demands", None):
        raise RuntimeError(
            f"the MILP level delivered {len(milp.untraced_demands)} demand(s) the extraction could "
            f"not trace back to a source (e.g. {milp.untraced_demands[:3]}). A flat solve logs and "
            f"skips these; a hierarchy level cannot, because each is a delivery with no route.")
    columns: Columns = defaultdict(list)
    for send in milp.sends:
        columns[send.identity].append(send)
    index = delivery_index(milp.sends)
    for g in demand.groups:
        for c in g.columns:
            for v in g.dest_cells:
                if ((g.cell, c), v) not in index:
                    raise RuntimeError(f"MILP solution delivers cell {g.cell} column {c} to no "
                                       f"send reaching cell {v}")
    return {key: sorted(sends, key=lambda x: (x.epoch, x.sender)) for key, sends in columns.items()}


def _leg(send: Send, branch: Branch, mapping: HierarchyMapping) -> _CoarsePiece:
    """One branch of a coarse send, as the cell -> switch(es) -> cell leg the placement helpers
    (`_leg_gateways`, `_pick_ingress`, `_commit_egress`, ...) take. Every branch of a send shares
    its first switch, so any branch speaks for the send's egress side."""
    if not branch.via:
        raise AssertionError(f"coarse send {send} reaches cell {branch.receiver} without a switch; "
                             f"cells are joined only through passthrough switches")
    return _CoarsePiece(src_cell=send.sender, dst_cell=branch.receiver,
                        egress_neighbor=branch.via[0], ingress_neighbor=branch.via[-1],
                        via_switches=tuple(mapping.coarse_passthrough[sw] for sw in branch.via),
                        volume=1.0, send_epoch=send.epoch, arrival_epoch=branch.ready - 1,
                        occupancy=send.span)


def _window(send: Send) -> range:
    """The epochs a send holds its links."""
    return range(send.epoch, send.epoch + send.span)


# ------------------------------------------------------------------------------------------------
# M4: integral identity resolution
# ------------------------------------------------------------------------------------------------
def _solve_integral_assignment(cell: int, groups: Sequence[MilpGroup], columns: Columns, b: int,
                               targets, mapping: HierarchyMapping, fine_topology: Topology,
                               epoch_duration: float, egress_ledger) -> Dict[Identity, Tuple[int, Dict]]:
    """Assign one cell's identities to its columns, and each column's first sends to gateways.

        y[d, c]       in {0,1}  identity d rides column c                  (c in d's group)
        z[d, c, S, g] in {0,1}  ...and its first send S leaves through gateway GPU g

        sum_c y[d, c] = 1                 every identity rides exactly one column
        sum_d y[d, c] = b                 every column carries a full bundle
        sum_g z[d, c, S, g] = y[d, c]     each first send of a ridden column gets one gateway
        sum z / span <= room(g, e)        per gateway per epoch, the fine uplink's capacity

    A first send is one transmission out of the cell (a switch copy is one send, several branches),
    so it gets one gateway and one capacity charge however many cells it reaches.

    Cost: `_TierWeights` -- egress relay (g is not d's holder) on z, ingress miss (a destination's
    landing candidates do not include a GPU that wants d) on y, and the epoch tie-break on relayed
    z. Solved as a MIP rather than an LP because the capacity rows couple columns and can break the
    transportation structure; the instances are one cell's identities, so they are tiny. One MIP
    per CELL, not per group, because all of a cell's groups share its uplinks.

    Returns identity -> (column, {first send: gateway}).
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    ids: List[Tuple[MilpGroup, Identity]] = [(g, d) for g in groups for d in g.identities]
    first = {c: [S for S in columns[(cell, c)] if S.sender == cell]
             for g in groups for c in g.columns}
    delivered = {c: delivery_index(columns[(cell, c)]) for g in groups for c in g.columns}
    y_index: Dict[Tuple[Identity, int], int] = {}
    z_index: Dict[Tuple[Identity, int, Send, int], int] = {}
    gateways: Dict[Send, List[Tuple[int, float]]] = {}
    for g, d in ids:
        for c in g.columns:
            y_index[(d, c)] = len(y_index)
    n_y = len(y_index)
    for g, d in ids:
        for c in g.columns:
            for S in first[c]:
                gws = gateways.setdefault(S, _leg_gateways(_leg(S, S.fanout[0], mapping), mapping,
                                                           fine_topology))
                if not gws:
                    raise RuntimeError(f"cell {cell}: no fine gateway owns the coarse link toward "
                                       f"{S.fanout[0].via[0]}")
                for gw, _cap in gws:
                    z_index[(d, c, S, gw)] = n_y + len(z_index)
    n = n_y + len(z_index)

    horizon = 1 + max((S.epoch for sends in first.values() for S in sends), default=0)
    n_ids = len(ids)
    max_dests = max(len(g.dest_cells) for g in groups)
    max_first = max(len(v) for v in first.values())
    w = _TierWeights(egress=float(n_ids * max_dests + 1), ingress=1.0,
                     epoch=1.0 / (2.0 * n_ids * max_first * horizon + 1.0), horizon=horizon)

    cost = np.zeros(n)
    holder = {d: g.holders[d] for g, d in ids}
    for g, d in ids:
        for c in g.columns:
            misses = 0
            for v in g.dest_cells:
                send, branch = delivered[c][((cell, c), v)]
                landing = _leg_ingress_candidates(_leg(send, branch, mapping), mapping)
                misses += not (set(targets.get((d, v), ())) & set(landing))
            cost[y_index[(d, c)]] = misses * w.ingress
    for (d, c, S, gw), j in z_index.items():
        cost[j] = w.cost(relayed=gw != holder[d], ingress_miss=False, send_epoch=S.epoch)

    rows: List[Tuple[Dict[int, float], float, float]] = []
    for g, d in ids:
        rows.append(({y_index[(d, c)]: 1.0 for c in g.columns}, 1.0, 1.0))
    for g in groups:
        for c in g.columns:
            rows.append(({y_index[(d, c)]: 1.0 for d in g.identities}, float(b), float(b)))
    for g, d in ids:
        for c in g.columns:
            for S in first[c]:
                row = {z_index[(d, c, S, gw)]: 1.0 for gw, _ in gateways[S]}
                row[y_index[(d, c)]] = -1.0
                rows.append((row, 0.0, 0.0))
    cap_rows: Dict[Tuple[int, Send, int], Dict[int, float]] = defaultdict(dict)
    for (d, c, S, gw), j in z_index.items():
        for e in _window(S):
            cap_rows[(gw, S.fanout[0].via[0], e)][j] = 1.0 / S.span
    tol = snap_tolerance()
    for (gw, nb, e), row in cap_rows.items():
        S = next(S for S in gateways if S.fanout[0].via[0] == nb)
        cap = _egress_epoch_capacity(_leg(S, S.fanout[0], mapping), [gw], mapping, fine_topology,
                                     epoch_duration)[gw]
        rows.append((row, -np.inf, cap - egress_ledger.get((gw, nb, e), 0.0) + tol))

    A = lil_matrix((len(rows), n))
    lb, ub = np.empty(len(rows)), np.empty(len(rows))
    for r, (coeffs, lo, hi) in enumerate(rows):
        for j, a in coeffs.items():
            A[r, j] = a
        lb[r], ub[r] = lo, hi
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lb, ub),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(
            f"cell {cell}: integral identity assignment infeasible ({res.message}). The MILP's "
            f"sends need more of some gateway's per-epoch uplink than it has -- the coarse link "
            f"aggregates fine uplinks, and no whole-identity split of the bundles fits them.")
    x = res.x
    if np.any(np.minimum(np.abs(x), np.abs(x - 1)) > 1e-6):
        raise AssertionError(f"cell {cell}: integral assignment returned non-0/1 values")

    out: Dict[Identity, Tuple[int, Dict]] = {}
    for (d, c), j in y_index.items():
        if x[j] > 0.5:
            out[d] = (c, {})
    for (d, c, S, gw), j in z_index.items():
        if x[j] > 0.5:
            assert out[d][0] == c
            out[d][1][S] = gw
    return out


def _forward_room(send: Send, mapping, fine_topology, epoch_duration, egress_ledger) -> Dict[int, float]:
    """Per-epoch uplink room left on each gateway of `send` (its first link), over its window."""
    leg = _leg(send, send.fanout[0], mapping)
    gateways = [g for g, _cap in _leg_gateways(leg, mapping, fine_topology)]
    cap = _egress_epoch_capacity(leg, gateways, mapping, fine_topology, epoch_duration)
    return {g: cap[g] - max(egress_ledger.get((g, leg.egress_neighbor, e), 0.0) for e in _window(send))
            for g in gateways}


def _commit_send(send: Send, gpu: int, identity: Identity, mapping, fine_topology, epoch_duration,
                 egress_ledger, tol) -> None:
    """Charge one transmission to its gateway's uplink: 1/span chunk in every epoch it holds it."""
    leg = _leg(send, send.fanout[0], mapping)
    for e in _window(send):
        _commit_egress(replace(leg, send_epoch=e), gpu, 1.0 / send.span, identity, mapping,
                       fine_topology, epoch_duration, egress_ledger, tol)


def _place_column(d: Identity, holder: int, sends: Sequence[Send], gateways: Dict[Send, int],
                  targets, mapping: HierarchyMapping, fine_topology: Topology,
                  epoch_duration: float, ingress_ledger, egress_ledger, tol: float,
                  result: IdentityResolution) -> None:
    """Lay one identity's column onto real GPUs, send by send in causal order, appending the fine
    sends and the intra-cell demands they create to `result`.

    `at[cell]` is the GPU holding `d` in that cell and `ready[cell]` the epoch it holds it from. For
    each coarse send out of a cell:

      * its sender GPU is the assignment's gateway (a first send), or -- forwarding -- the GPU that
        landed `d` when that GPU owns the outgoing link and has room (co-located), else a gateway
        with room when the send leaves at least an epoch after `d` lands (the dwell
        `forward_dwell_constraints` charged for the relay), else this is a contract breach;
      * a sender other than `at[cell]` needs an `egress_stage` relay inside the cell first;
      * each branch lands on a GPU of its cell: one that can also forward `d` on every send the cell
        makes next in this column, when the cell forwards; otherwise one that wants `d`;
      * a cell that wants `d` gets an `ingress_distribution` from its landing GPU.

    Each send is placed exactly once, so a switch copy is one fine send with one branch per cell.
    """
    first_cell = sends[0].sender if sends else None
    at: Dict[int, int] = {first_cell: holder}
    ready: Dict[int, Optional[int]] = {first_cell: None}
    out_of: Dict[int, List[Send]] = defaultdict(list)
    for S in sends:
        out_of[S.sender].append(S)

    def land(leg, candidates, preferred, what):
        in_cap = _ingress_epoch_capacity(leg, candidates, mapping, fine_topology, epoch_duration)
        [(landed, _v)] = _pick_ingress(leg, candidates, d, 1.0 / leg.occupancy, preferred, in_cap,
                                       ingress_ledger, tol=tol, what=what, allow_split=False)
        return landed

    for S in sends:
        U = S.sender
        if U not in at:
            raise AssertionError(f"identity {d}: cell {U} sends in epoch {S.epoch} before any send "
                                 f"delivers to it")
        if U == first_cell:
            g = gateways[S]                 # chosen, and already charged, by the assignment
        else:
            if S.epoch < ready[U]:
                raise AssertionError(f"identity {d}: cell {U} forwards in epoch {S.epoch} but holds "
                                     f"the chunk only from {ready[U]}")
            room = _forward_room(S, mapping, fine_topology, epoch_duration, egress_ledger)
            need = 1.0 / S.span
            if room.get(at[U], -1.0) >= need - tol:
                g = at[U]                   # co-located: no intra-cell work
            elif S.epoch >= ready[U] + 1:
                fits = [gw for gw in sorted(room) if room[gw] >= need - tol]
                if not fits:
                    raise RuntimeError(
                        f"forwarding egress capacity exhausted at cell {U} epoch {S.epoch}: identity "
                        f"{d} needs {need:g} toward coarse neighbor {S.fanout[0].via[0]}, gateway "
                        f"room { {gw: round(r, 6) for gw, r in room.items()} }")
                g = min(fits, key=lambda gw: room[gw])          # best fit, as _pick_ingress
            else:
                raise RuntimeError(
                    f"cell {U} forwards identity {d} in epoch {S.epoch}, the epoch it is ready "
                    f"({ready[U]}), but its landing gpu {at[U]} cannot send toward coarse neighbor "
                    f"{S.fanout[0].via[0]} (room {room}). A relay to another gateway needs a "
                    f"second epoch, which the MILP's forward-dwell constraint should have charged "
                    f"-- was the level solved without CoarseTopology.forward_bridges?")
            _commit_send(S, g, d, mapping, fine_topology, epoch_duration, egress_ledger, tol)
        if g != at[U]:
            result.intra_demands.append(IntraCellDemand(
                cell=U, kind="egress_stage", identity=d, src_gpu=at[U], dst_gpus=(g,), volume=1.0,
                deadline_epoch=S.epoch, release_band=ready[U],
                hard=True if ready[U] is not None else None))

        branches = []
        for br in S.fanout:
            V, leg = br.receiver, _leg(S, br, mapping)
            candidates = list(_leg_ingress_candidates(leg, mapping))
            landed = None
            if out_of[V]:
                # V forwards d next: land where it can also send, on every one of V's sends, with
                # uplink room for each -- chosen jointly, or the only way out is a relay the dwell
                # may not allow.
                joint = [h for h in candidates
                         if all(_forward_room(T, mapping, fine_topology, epoch_duration,
                                              egress_ledger).get(h, -1.0) >= 1.0 / T.span - tol
                                for T in out_of[V])]
                if joint:
                    try:
                        landed = land(leg, joint, set(targets.get((d, V), ())), "transit")
                    except RuntimeError:
                        landed = None       # no joint bridge has ingress room either
            if landed is None:
                landed = land(leg, candidates, set(targets.get((d, V), ())), "ingress")
            at[V], ready[V] = landed, br.ready
            branches.append(Branch(landed, leg.via_switches, br.ready))
            if targets.get((d, V)):
                result.intra_demands.append(IntraCellDemand(
                    cell=V, kind="ingress_distribution", identity=d, src_gpu=landed,
                    dst_gpus=targets[(d, V)], volume=1.0, deadline_epoch=br.ready - 1))
        result.pieces.append(Send(identity=d, sender=g, epoch=S.epoch, span=S.span,
                                  fanout=tuple(branches)))


def lower_integral(columns: Columns, demand: PreservingDemand, mapping: HierarchyMapping,
                   fine_demand, fine_topology: Topology, epoch_duration: float,
                   scale: Optional[ChunkScale] = None, relabel=None, holder_of=None
                   ) -> IdentityResolution:
    """Lower a MILP level: decide which identity rides which column (and leaves through which
    gateway), lay every column's sends onto real GPUs, and build the child problems.

    Volumes are whole identities throughout, so the level never refines (Q == 1) and never snaps:
    the result is in the level's own scale. Every first send's gateway is committed to the egress
    ledger BEFORE anything is placed, so a cell's forwarded sends (its transit role) see the room
    its own data already uses on each uplink.
    """
    _id_sets, targets = identity_sets(fine_demand, mapping, relabel=relabel)
    egress_ledger: Dict[Tuple[int, int, int], float] = {}
    ingress_ledger: Dict[Tuple[int, int, int], float] = {}
    tol = snap_tolerance()
    result = IdentityResolution(
        scale=scale or ChunkScale(bytes_per_chunk=fine_topology.chunk_size,
                                  num_chunks=_num_fine_chunks(fine_demand)),
        subdivision=1)

    by_cell: Dict[int, List[MilpGroup]] = defaultdict(list)
    for g in demand.groups:
        by_cell[g.cell].append(g)
    chosen: Dict[Identity, Tuple[MilpGroup, int, Dict]] = {}
    for cell in sorted(by_cell):
        picks = _solve_integral_assignment(cell, by_cell[cell], columns, demand.b, targets, mapping,
                                           fine_topology, epoch_duration, egress_ledger)
        for g in by_cell[cell]:
            for d in g.identities:
                c, gws = picks[d]
                chosen[d] = (g, c, gws)
                for S, gw in gws.items():
                    _commit_send(S, gw, d, mapping, fine_topology, epoch_duration, egress_ledger,
                                 tol)
    # Columns that FORWARD through a cell go first, for the reason _place_decisions gives the LP's
    # transit routes: a forwarding cell needs ingress AND egress room on one GPU, and ordinary
    # landings placed first fragment the two until no GPU has both.
    order = sorted(chosen, key=lambda d: not any(S.sender != chosen[d][0].cell
                                                 for S in columns[(chosen[d][0].cell, chosen[d][1])]))
    for d in order:
        g, c, gws = chosen[d]
        holder = g.holders[d] if holder_of is not None else d[0]
        _place_column(d, holder, columns[(g.cell, c)], gws, targets, mapping, fine_topology,
                      epoch_duration, ingress_ledger, egress_ledger, tol, result)
    return finish_resolution(result, mapping, fine_demand, fine_topology, epoch_duration,
                             relabel=relabel)
