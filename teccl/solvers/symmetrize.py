"""Orbit averaging: project an LP optimum onto the symmetric (balanced) optimum.

WHY THIS EXISTS. A degenerate LP face leaves the solver free to return any point on it, and
simplex returns a VERTEX -- by construction the most lopsided point of that face. On a topology
whose hosts are interchangeable this shows up as twin hosts splitting a shared uplink budget
arbitrarily (e.g. 0.667 / 1.333 instead of 1.0 / 1.0), which is not merely cosmetic: a host that
draws the short share can end up IDLE for an epoch and then burst, and an idle epoch is exactly
what leaves a send with no same-GPU op to gate against, so ncclize reports it as unrealizable on
real hardware.

No linear objective can fix this. If the face is optimal then every linear objective is constant
on it, so no tier weighting reaches the balanced point; and both solver paths actively move away
from it (simplex -> an extreme vertex, barrier without crossover -> the analytic centre of the
whole optimal face, which is some other arbitrary interior point). Preferring balance needs
either a strictly convex term (LP -> QP) or integrality (LP -> MILP). This module takes the third
route: leave the model alone and average the FINISHED solution over the topology's symmetry
group, which costs nothing in the solver and is provably optimality-preserving.

CORRECTNESS. Let sigma be a permutation of node indices with
    (A1) capacity[u][v] == capacity[sigma u][sigma v] for all u, v, and sigma preserving node
         type (host / switch / programmable switch), and
    (A2) demand[s][d] == demand[sigma s][sigma d].
Define sigma(x)[s][i][j][k] = x[sigma^-1 s][sigma^-1 i][sigma^-1 j][k]. Then:

  1. sigma(x) is FEASIBLE. Every constraint maps to another instance of itself: a capacity row on
     (u, v) becomes the row on (sigma u, sigma v) whose bound is equal by A1 (and whose left side
     sums over all sources, an index set sigma merely permutes); conservation at n for source s
     becomes conservation at sigma n for source sigma s; the demand row for (s, d) becomes the row
     for (sigma s, sigma d), equal by A2.
  2. sigma(x) has the SAME OBJECTIVE, tier by tier, exactly -- which matters because the
     lexicographic solve pins tier 1 to its optimum before optimising tier 2, and only an exactly
     equal value still satisfies that constraint. Tier 1 is an unweighted sum over all (s, d, k),
     invariant under permuting the index set. Tier 2 sums relay flow over all hosts and all
     sources with the guard s != n, which maps to sigma s != sigma n; sigma keeps hosts hosts. Tier
     3 sums flow entering a switch, and sigma keeps switch-ingress links switch-ingress.
  3. The AVERAGE is feasible and optimal. The feasible set is a polyhedron, hence convex, so the
     mean over a finite group G is feasible; each tier is linear, so its value at the mean is the
     mean of values, all equal to its value at x. An optimal x therefore averages to an optimal,
     G-invariant x_bar.
  4. x_bar is BALANCED where it matters. G-invariance forces equal per-epoch flow across each
     orbit, so if G acts transitively on the GPUs behind a shared bottleneck they all carry an
     equal share, and none of them is idle while the orbit as a whole is active.

WHAT THIS DOES NOT DO. Averaging projects onto the fixed-point subspace of G, so it removes
exactly the degeneracy that is ALIGNED WITH A SYMMETRY ORBIT and nothing else -- ties between
genuinely non-equivalent paths survive untouched. It also DENSIFIES: filling a zero is the very
mechanism that removes an idle epoch (a 0 averaged against its twin's 0.667 becomes 0.333), so
the support grows, which can mean more ops emitted and more work for a flow decomposition that
is known to be delicate. Both are the reason this is opt-in rather than default; measure the
fraction histogram, the op count and the ncclize warning count before trusting it more widely.

SAFETY. Every generator this module proposes is VERIFIED exactly against the capacity matrix,
the type vector and the demand matrix before use, so a candidate built by an incomplete search is
rejected rather than trusted. Finding fewer generators than exist only means a smaller group and
less averaging; it can never make the result wrong.
"""

import logging
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

# A permutation is a list p with p[i] = image of node i.
Perm = List[int]

# Enumerating the group is exponential in the number of independent generators, and averaging
# costs O(|G| x support), so cap it. Past the cap we average over NOTHING rather than over a
# truncated set: a truncated set is not a group, so its mean is not G-invariant -- it would be
# just another arbitrary point, which is the thing this module exists to get away from.
#
# For scale: the 4-host / 8-GPU mini topology has a group of exactly 256 (2^4 within-host GPU
# swaps, x4 host swaps within a leaf, x2 leaf swap, x2 spine swap). A 32-host / 256-GPU rail
# topology's group is astronomically larger and will skip -- exact enumeration is the wrong tool
# there, and the honest behaviour is to decline rather than to half-symmetrize.
MAX_GROUP = 4096


def _neighbors(capacity: Sequence[Sequence[float]], i: int) -> List[int]:
    return [j for j, c in enumerate(capacity[i]) if c > 0]


def _type_key(topology, i: int) -> Tuple[bool, bool]:
    """Node class for automorphism purposes: (is switch, is programmable switch).

    Programmability is part of the key because tier 3 and ncclize both treat a programmable
    switch differently from a self-routing one, so a permutation that mixed the two would not
    preserve the objective even if it preserved the graph.
    """
    return (i in set(topology.switch_indices),
            i in set(topology.programmable_switch_indices))


def stable_colors(capacity: Sequence[Sequence[float]], types: Sequence[Tuple]) -> List[int]:
    """Colour-refinement (1-WL) classes. Nodes in different classes cannot be in one orbit."""
    n = len(capacity)
    colors = list({t: idx for idx, t in enumerate(sorted(set(types)))}[t] for t in types)
    while True:
        signature = [
            (colors[i], tuple(sorted((colors[j], capacity[i][j]) for j in _neighbors(capacity, i))))
            for i in range(n)
        ]
        table = {sig: idx for idx, sig in enumerate(sorted(set(signature)))}
        new = [table[sig] for sig in signature]
        if new == colors:
            return colors
        colors = new


def _candidate_involution(capacity, colors, a: int, b: int) -> Optional[Perm]:
    """Try to grow the transposition (a b) into a full involution by matching neighbourhoods.

    Walks outward from the pair, matching each side's neighbours by (colour, capacity) in sorted
    order. Sorted order is a heuristic -- it finds the generators these topologies actually have
    without a backtracking search -- and anything it builds is verified by the caller, so a wrong
    guess costs a rejected candidate and never a wrong answer.
    """
    n = len(capacity)
    sigma: Dict[int, int] = {a: b, b: a}
    queue = [(a, b)]
    while queue:
        u, v = queue.pop()
        bucket_u, bucket_v = defaultdict(list), defaultdict(list)
        for w in _neighbors(capacity, u):
            bucket_u[(colors[w], capacity[u][w])].append(w)
        for w in _neighbors(capacity, v):
            bucket_v[(colors[w], capacity[v][w])].append(w)
        if set(bucket_u) != set(bucket_v):
            return None
        for key in bucket_u:
            left, right = sorted(bucket_u[key]), sorted(bucket_v[key])
            if len(left) != len(right):
                return None
            for x, y in zip(left, right):
                if x in sigma:
                    if sigma[x] != y:
                        return None
                    continue
                sigma[x] = y
                if y not in sigma:
                    sigma[y] = x
                elif sigma[y] != x:
                    return None
                if x != y:
                    queue.append((x, y))
    perm = list(range(n))
    for src, dst in sigma.items():
        perm[src] = dst
    return perm


def is_automorphism(capacity, types, perm: Perm) -> bool:
    """A1: exact check that `perm` preserves every capacity entry and every node type."""
    n = len(capacity)
    if sorted(perm) != list(range(n)):
        return False
    if any(types[i] != types[perm[i]] for i in range(n)):
        return False
    return all(capacity[i][j] == capacity[perm[i]][perm[j]]
               for i in range(n) for j in range(n))


def preserves_demand(demand, perm: Perm) -> bool:
    """A2: exact check that `perm` maps the demand matrix onto itself, AGGREGATED OVER CHUNKS.

    This is the gate that makes the pass collective-aware without being told the collective. A
    uniform all-pairs demand (alltoall, allgather) is invariant under every permutation; a rooted
    collective's is not, and its root-moving permutations are rejected here.

    AGGREGATED, and that is load-bearing rather than a convenience. AllToAll builds its demand as
    demand[s][t][device_chunk_map[t] + c*gpus] = 1, i.e. THE CHUNK INDEX ENCODES THE DESTINATION
    (see solvers/demand.py), so the per-chunk vector for (s, t) is a one-hot whose position moves
    with t. Compared chunk-by-chunk, every permutation that moves any GPU would be rejected and
    this pass could never fire on the collective it was written for.

    Comparing the aggregate is not a weakening, because the aggregate is exactly what the LP
    sees: `self.demand` reaches the model in three places only -- all_demand, node_demand and
    demand_at_i -- and every one of them sums over c before use. No LP constraint is indexed by a
    chunk. The chunk dimension is resolved later, in account_for_consume, which splits each (s, d)
    aggregate across that pair's chunks using demand_copy -- a structure this pass never touches.
    So invariance of the aggregate is precisely the condition under which sigma(x) is feasible for
    the LP with an unchanged objective, which is all steps 1-3 of the correctness argument need.
    """
    n = len(perm)
    total = [[sum(demand[s][d]) for d in range(n)] for s in range(n)]
    return all(total[s][d] == total[perm[s]][perm[d]]
               for s, d in product(range(n), range(n)))


def find_generators(topology, demand) -> List[Perm]:
    """Verified generators: involutions that satisfy A1 and A2."""
    capacity = topology.capacity
    n = len(capacity)
    types = [_type_key(topology, i) for i in range(n)]
    colors = stable_colors(capacity, types)
    by_color = defaultdict(list)
    for i in range(n):
        by_color[colors[i]].append(i)

    generators: List[Perm] = []
    seen = set()
    for nodes in by_color.values():
        for a, b in product(nodes, nodes):
            if a >= b:
                continue
            perm = _candidate_involution(capacity, colors, a, b)
            if perm is None or tuple(perm) in seen:
                continue
            if not is_automorphism(capacity, types, perm):
                continue
            if not preserves_demand(demand, perm):
                logging.debug("symmetrize: rejecting graph automorphism that moves demand "
                              "(%d<->%d)", a, b)
                continue
            seen.add(tuple(perm))
            generators.append(perm)
    return generators


def close_group(generators: Sequence[Perm], n: int) -> Optional[List[Perm]]:
    """The group generated, or None if it exceeds MAX_GROUP (see the note on MAX_GROUP)."""
    identity = tuple(range(n))
    group = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for gen in generators:
            composed = tuple(gen[current[i]] for i in range(n))
            if composed not in group:
                if len(group) >= MAX_GROUP:
                    return None
                group.add(composed)
                frontier.append(composed)
    return [list(g) for g in group]


def average_over_group(values: Dict[Tuple, float], group: Sequence[Perm],
                       node_axes: Sequence[int]) -> Dict[Tuple, float]:
    """Mean of g(x) over the group, for a dict keyed by a tuple whose `node_axes` are node ids.

    `node_axes` says which positions of the key are node indices and must be relabelled; the rest
    (the epoch, say) are carried through untouched.
    """
    size = len(group)
    out: Dict[Tuple, float] = defaultdict(float)
    for perm in group:
        for key, value in values.items():
            mapped = list(key)
            for axis in node_axes:
                mapped[axis] = perm[key[axis]]
            out[tuple(mapped)] += value / size
    return dict(out)
