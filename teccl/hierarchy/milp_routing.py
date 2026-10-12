"""
A hierarchy level solved by the chunk-level MILP (AllGatherFormulation): its coarse demand, how its
solution is read back, and the integral identity resolution that lowers it.

The MILP is the LP's peer at a level, not a separate pipeline. It re-enters the LP's machinery at
the step-A seam and shares everything else -- `_legs_of_hops`, `_leg_gateways`, `_TierWeights`,
`_place_decisions`, and all of step B. What differs, and lives here:

  * DEMAND. The MILP's chunks are binary (`total_demand_sat` is a {0,1} var pinned to the demand),
    so it cannot take the LP's weighted single slot -- a demand of 4 is structurally infeasible.
    `coarsify_demand_preserving` builds a COUNT tensor instead: one column per BUNDLE of `b` fine
    identities, b chosen by the level's `milp_chunking` policy.

  * READ-BACK. The MILP copies at a cell (a node may send what it holds to several neighbours), so
    each column's solution is a multicast TREE, read off the per-demand paths the extraction already
    traces (`extract_trees`). Its leg epochs are normalized to the LP's convention -- `arrival_epoch`
    is the epoch the bytes land DURING -- so co-located forwarding is `send >= arrival + 1` and a
    relayed one `send >= arrival + 2` in both.

  * IDENTITY. The MILP keeps chunk identity only up to interchangeable columns. Two columns of one
    cell with the same destination-cell set look identical to the coarse level, which collapsed the
    cell's GPUs into one node: it cannot see that putting gpu1's data on gpu0's uplink and gpu0's on
    gpu1's costs two staging relays the uncrossed labelling does not. So step A is still an
    assignment (`assign_identities_integral`) -- but of WHOLE identities to WHOLE trees, which is a
    transportation problem and integral, so this level never refines (Q == 1) and never snaps.

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
    Identity, _Assignment, _CoarsePiece, _CoarseRoute, _Hop, _Slot, _TierWeights,
    _commit_egress, _egress_epoch_capacity, _leg_gateways, _leg_ingress_candidates,
    _legs_of_hops, _place_decisions, _validate_route, identity_sets, resolve_max_denom,
    send_group, snap_tolerance)
from teccl.topologies.topology import Topology

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------
# M1: the count tensor
# ------------------------------------------------------------------------------------------------
@dataclass
class MilpGroup:
    """The identities of one cell that share a destination-cell set, and the MILP columns that
    carry them. Columns of one group are interchangeable to the MILP; which identity rides which
    column's tree is decided after the solve, by `assign_identities_integral`."""
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
# M2: the solution as trees
# ------------------------------------------------------------------------------------------------
Trees = Dict[Tuple[int, int], Dict[int, _CoarseRoute]]       # (cell, column) -> {dest cell: route}


def extract_trees(milp, mapping: HierarchyMapping, demand: PreservingDemand) -> Trees:
    """Read a solved MILP level back as one multicast tree per column.

    Built from `milp.per_demand_paths`, the tuple paths `dfs_remove_unnecessary_flows` traced, one
    per (source cell, destination cell, column). A column's per-destination routes share prefix
    legs; they are kept as routes (what placement walks) and the sharing is recovered by the
    placement memo and the piece dedup in step B. Checked here, failing loud:

      * every demanded (column, destination) was traced -- an untraced demand is a delivery the
        hierarchy would otherwise silently drop;
      * each route is a valid chain (`_validate_route`: chains, dwells, no cell revisited);
      * the routes of one column form a TREE: each cell is entered by at most one leg.

    `milp` needs `.per_demand_paths`, `.untraced_demands`, `.topology` and `.hop_timing(j, i)`.
    """
    if getattr(milp, "untraced_demands", None):
        raise RuntimeError(
            f"the MILP level delivered {len(milp.untraced_demands)} demand(s) the extraction could "
            f"not trace back to a source (e.g. {milp.untraced_demands[:3]}). A flat solve logs and "
            f"skips these; a hierarchy level cannot, because each is a delivery with no route.")
    switches = set(milp.topology.switch_indices)
    trees: Trees = {}
    for g in demand.groups:
        for c in g.columns:
            routes: Dict[int, _CoarseRoute] = {}
            entering: Dict[int, _CoarsePiece] = {}
            # One leg OBJECT per physical transfer. `_legs_of_hops` stamps each leg with its
            # route's origin, so the A -> B leg on the route to B and on the route to C through B
            # would otherwise compare unequal -- and equality is what the tree check, the placement
            # memo (keyed (identity, leg)) and the step-B piece dedup all rely on.
            canonical: Dict[_CoarsePiece, _CoarsePiece] = {}
            for v in g.dest_cells:
                path = milp.per_demand_paths.get((g.cell, v, c))
                if path is None:
                    raise RuntimeError(f"MILP solution has no traced path for cell {g.cell} column "
                                       f"{c} -> cell {v}")
                hops = []
                for (_s, i, j, _c, k) in reversed(path):      # traced destination-first
                    offset, occupancy = milp.hop_timing(i, j)
                    hops.append(_Hop(sender=i, receiver=j, volume=1.0, send_epoch=k,
                                     arrival_epoch=k + offset - 1, occupancy=occupancy))
                legs = [canonical.setdefault(replace(leg, origin=(g.cell, g.cell)), leg)
                        for leg in _legs_of_hops(hops, switches, mapping.coarse_passthrough,
                                                 origin=(g.cell, v))]
                route = _CoarseRoute(origin=(g.cell, v), legs=tuple(legs), volume=1.0)
                _validate_route(route)
                for leg in legs:
                    prev = entering.setdefault(leg.dst_cell, leg)
                    if prev != leg:
                        raise AssertionError(
                            f"cell {g.cell} column {c}: cell {leg.dst_cell} is entered twice "
                            f"({prev} and {leg}); a copy-aware solution must be a tree")
                routes[v] = route
            trees[(g.cell, c)] = routes
    return trees


# ------------------------------------------------------------------------------------------------
# M4: integral identity resolution
# ------------------------------------------------------------------------------------------------
def _first_legs(routes: Dict[int, _CoarseRoute]) -> Dict[_CoarsePiece, List[_CoarsePiece]]:
    """The physical sends leaving the tree's root cell, each as {representative leg: every leg it
    starts}. A cell may copy to several neighbours (several groups), and under switch copy one
    send may start several legs (one group, several members) -- see `send_group`."""
    groups: Dict[Tuple[int, int, int], List[_CoarsePiece]] = {}
    for v in sorted(routes):
        leg = routes[v].first
        members = groups.setdefault(send_group(leg), [])
        if leg not in members:
            members.append(leg)
    return {members[0]: members for members in groups.values()}


def _window(leg: _CoarsePiece) -> range:
    """The epochs a leg holds its link: from its send, for its occupancy."""
    return range(leg.send_epoch, leg.send_epoch + leg.occupancy)


def _solve_integral_assignment(cell: int, groups: Sequence[MilpGroup], trees: Trees, b: int,
                               targets, mapping: HierarchyMapping, fine_topology: Topology,
                               epoch_duration: float, egress_ledger) -> Dict[Identity, Tuple[int, Dict]]:
    """Assign one cell's identities to its columns' trees, and each tree's first legs to gateways.

        y[d, c]       in {0,1}  identity d rides column c's tree            (c in d's group)
        z[d, c, l, g] in {0,1}  ...and leaves on first leg l through gateway GPU g

        sum_c y[d, c] = 1                 every identity rides exactly one tree
        sum_d y[d, c] = b                 every tree carries a full bundle
        sum_g z[d, c, l, g] = y[d, c]     each first leg of a ridden tree gets one gateway
        sum z / occupancy <= room(g, e)   per gateway per epoch, the fine uplink's capacity

    Cost: `_TierWeights` -- egress relay (g is not d's holder) on z, ingress miss (a destination's
    landing candidates do not include a GPU that wants d) on y, and the epoch tie-break on relayed
    z. Solved as a MIP rather than an LP because the capacity rows couple trees and can break the
    transportation structure; the instances are one cell's identities, so they are tiny. One MIP
    per CELL, not per group, because all of a cell's groups share its uplinks.

    A first leg is one per physical SEND (`send_group`): under switch copy several legs of a tree
    can start with the same send, and they get one gateway and one capacity charge between them.

    Returns identity -> (column, {first leg: gateway}), with every leg of a send group listed.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    ids: List[Tuple[MilpGroup, Identity]] = [(g, d) for g in groups for d in g.identities]
    first = {(g.cell, c): _first_legs(trees[(g.cell, c)]) for g in groups for c in g.columns}
    y_index: Dict[Tuple[Identity, int], int] = {}
    z_index: Dict[Tuple[Identity, int, _CoarsePiece, int], int] = {}
    gateways: Dict[_CoarsePiece, List[Tuple[int, float]]] = {}
    for g, d in ids:
        for c in g.columns:
            y_index[(d, c)] = len(y_index)
    n_y = len(y_index)
    for g, d in ids:
        for c in g.columns:
            for leg in first[(cell, c)]:
                gws = gateways.setdefault(leg, _leg_gateways(leg, mapping, fine_topology))
                if not gws:
                    raise RuntimeError(f"cell {cell}: no fine gateway owns the coarse link toward "
                                       f"{leg.egress_neighbor}")
                for gw, _cap in gws:
                    z_index[(d, c, leg, gw)] = n_y + len(z_index)
    n = n_y + len(z_index)

    horizon = 1 + max((leg.send_epoch for legs in first.values() for leg in legs), default=0)
    n_ids = len(ids)
    max_dests = max(len(g.dest_cells) for g in groups)
    max_first = max(len(v) for v in first.values())
    w = _TierWeights(egress=float(n_ids * max_dests + 1), ingress=1.0,
                     epoch=1.0 / (2.0 * n_ids * max_first * horizon + 1.0), horizon=horizon)

    cost = np.zeros(n)
    holder = {d: g.holders[d] for g, d in ids}
    for g, d in ids:
        for c in g.columns:
            routes = trees[(cell, c)]
            misses = sum(1 for v, r in routes.items()
                         if not (set(targets.get((d, v), ())) & set(_leg_ingress_candidates(r.last, mapping))))
            cost[y_index[(d, c)]] = w.cost(relayed=False, ingress_miss=False, send_epoch=0) + misses * w.ingress
    for (d, c, leg, gw), j in z_index.items():
        cost[j] = w.cost(relayed=gw != holder[d], ingress_miss=False, send_epoch=leg.send_epoch)

    rows: List[Tuple[Dict[int, float], float, float]] = []
    for g, d in ids:
        rows.append(({y_index[(d, c)]: 1.0 for c in g.columns}, 1.0, 1.0))
    for g in groups:
        for c in g.columns:
            rows.append(({y_index[(d, c)]: 1.0 for d in g.identities}, float(b), float(b)))
    for g, d in ids:
        for c in g.columns:
            for leg in first[(cell, c)]:
                row = {z_index[(d, c, leg, gw)]: 1.0 for gw, _ in gateways[leg]}
                row[y_index[(d, c)]] = -1.0
                rows.append((row, 0.0, 0.0))
    cap_rows: Dict[Tuple[int, int, int], Dict[int, float]] = defaultdict(dict)
    for (d, c, leg, gw), j in z_index.items():
        for e in _window(leg):
            cap_rows[(gw, leg.egress_neighbor, e)][j] = 1.0 / leg.occupancy
    tol = snap_tolerance()
    for (gw, nb, e), row in cap_rows.items():
        leg = next(l for l in gateways if l.egress_neighbor == nb)
        cap = _egress_epoch_capacity(leg, [gw], mapping, fine_topology, epoch_duration)[gw]
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
            f"trees need more of some gateway's per-epoch uplink than it has -- the coarse link "
            f"aggregates fine uplinks, and no whole-identity split of the bundles fits them.")
    x = res.x
    if np.any(np.minimum(np.abs(x), np.abs(x - 1)) > 1e-6):
        raise AssertionError(f"cell {cell}: integral assignment returned non-0/1 values")

    out: Dict[Identity, Tuple[int, Dict]] = {}
    for (d, c), j in y_index.items():
        if x[j] > 0.5:
            out[d] = (c, {})
    for (d, c, leg, gw), j in z_index.items():
        if x[j] > 0.5:
            assert out[d][0] == c
            for member in first[(cell, c)][leg]:
                out[d][1][member] = gw
    return out


def assign_identities_integral(trees: Trees, demand: PreservingDemand, mapping: HierarchyMapping,
                               fine_demand, fine_topology: Topology, epoch_duration: float,
                               relabel=None, holder_of=None
                               ) -> Tuple[List[_Assignment], Dict, float]:
    """STEP A for a MILP level: decide which identity rides which tree (and leaves through which
    gateway), then lay every route onto real GPUs. Returns what `assign_identities_free` returns --
    `_Assignment`s, the target map and the epoch -- so step B (`build_child_problems`) is shared.

    Volumes are whole identities throughout, so step B finds Q == 1. Placement is the shared
    `_place_decisions`, in its copy-aware mode: a leg shared by several routes of one identity's
    tree is placed once (`placed_legs`), and its pieces are deduplicated in step B.

    Every first leg's gateway is committed to the egress ledger BEFORE any route is placed, so a
    cell's forwarded legs (its transit role) see the room its own data already uses on each uplink.
    """
    _id_sets, targets = identity_sets(fine_demand, mapping, relabel=relabel)
    egress_ledger: Dict[Tuple[int, int, int], float] = {}
    ingress_ledger: Dict[Tuple[int, int, int], float] = {}
    decisions = []
    holder_by_id: Dict[Tuple[int, int, Identity], int] = {}

    by_cell: Dict[int, List[MilpGroup]] = defaultdict(list)
    for g in demand.groups:
        by_cell[g.cell].append(g)
    tol = snap_tolerance()
    for cell in sorted(by_cell):
        groups = by_cell[cell]
        chosen = _solve_integral_assignment(cell, groups, trees, demand.b, targets, mapping,
                                            fine_topology, epoch_duration, egress_ledger)
        committed = set()
        for g in groups:
            for d in g.identities:
                c, gws = chosen[d]
                for leg, gw in gws.items():
                    if (d, send_group(leg)) in committed:
                        continue                      # one send, several legs (switch copy)
                    committed.add((d, send_group(leg)))
                    for e in _window(leg):
                        _commit_egress(replace(leg, send_epoch=e), gw, 1.0 / leg.occupancy, d,
                                       mapping, fine_topology, epoch_duration, egress_ledger, tol)
                for v, route in sorted(trees[(cell, c)].items()):
                    slot = _Slot(route=route, egress_gpu=gws[route.first], capacity=1.0,
                                 ingress_candidates=_leg_ingress_candidates(route.last, mapping))
                    decisions.append((cell, v, d, slot, 1.0))
                    holder_by_id[(cell, v, d)] = g.holders[d] if holder_of is not None else -1

    assignments = _place_decisions(decisions, holder_by_id, targets, mapping, fine_topology,
                                   epoch_duration, ingress_ledger, egress_ledger,
                                   tol=tol, max_denom=resolve_max_denom(), placed_legs={})
    return assignments, targets, epoch_duration
