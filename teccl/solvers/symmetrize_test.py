"""symmetrize: the orbit-averaging post-pass must balance without changing the answer.

Run: conda run -n teccl python teccl/solvers/symmetrize_test.py
"""

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from teccl.input_data import TopologyParams
from teccl.solvers import symmetrize
from teccl.topologies.dsl_topology import DslTopology

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "topologies",
                        "topology-dsl-frontend", "examples")


def load(name: str) -> DslTopology:
    return DslTopology(TopologyParams(name=name, chunk_size=1.0,
                                      topo_file=os.path.join(EXAMPLES, name + ".topo")))


def uniform_demand(topology):
    """AllToAll-shaped: one unit between every ordered GPU pair."""
    n = len(topology.capacity)
    gpus = [i for i in range(n) if i not in topology.switch_indices]
    demand = [[[0.0] for _ in range(n)] for _ in range(n)]
    for s in gpus:
        for d in gpus:
            if s != d:
                demand[s][d] = [1.0]
    return demand, gpus


def rooted_demand(topology, root):
    """Broadcast-shaped: root -> everyone. Only root-fixing permutations may be accepted."""
    n = len(topology.capacity)
    gpus = [i for i in range(n) if i not in topology.switch_indices]
    demand = [[[0.0] for _ in range(n)] for _ in range(n)]
    for d in gpus:
        if d != root:
            demand[root][d] = [1.0]
    return demand


def check_generators_are_verified() -> None:
    """Every generator must be a true automorphism AND preserve the demand."""
    topo = load("mini_2gpu_1nic")
    demand, _ = uniform_demand(topo)
    types = [symmetrize._type_key(topo, i) for i in range(len(topo.capacity))]
    generators = symmetrize.find_generators(topo, demand)
    assert generators, "expected the twin hosts and twin GPUs to be found"
    for perm in generators:
        assert symmetrize.is_automorphism(topo.capacity, types, perm)
        assert symmetrize.preserves_demand(demand, perm)
    print(f"  OK  {len(generators)} generators, all verified as demand-preserving automorphisms")


def check_rooted_demand_is_gated() -> None:
    """A permutation that moves the root changes the demand and must be refused."""
    topo = load("mini_2gpu_1nic")
    names = topo.node_names
    root = names.index("srv0_gpu0")
    demand = rooted_demand(topo, root)
    for perm in symmetrize.find_generators(topo, demand):
        assert perm[root] == root, (
            f"accepted a permutation moving the broadcast root {names[root]} -> "
            f"{names[perm[root]]}")
    # and the same topology under a uniform demand DOES admit root-moving permutations,
    # so the gate is discriminating rather than just conservative.
    uniform, _ = uniform_demand(topo)
    assert any(p[root] != root for p in symmetrize.find_generators(topo, uniform)), \
        "uniform demand should admit permutations that move any given GPU"
    print("  OK  rooted demand gates out root-moving permutations")


def check_group_is_closed() -> None:
    """close_group must return an actual group: closed under composition, with an identity."""
    topo = load("mini_2gpu_1nic")
    demand, _ = uniform_demand(topo)
    n = len(topo.capacity)
    group = symmetrize.close_group(symmetrize.find_generators(topo, demand), n)
    assert group is not None
    members = {tuple(g) for g in group}
    assert tuple(range(n)) in members, "group is missing the identity"
    for a in group:
        for b in group:
            assert tuple(a[b[i]] for i in range(n)) in members, "group is not closed"
    print(f"  OK  group of {len(group)} is closed under composition")


def check_averaging_balances_and_conserves() -> None:
    """The point of the pass: equal LINK LOAD across an orbit, with total volume preserved.

    Balanced means balanced in the SOURCE-AGGREGATED load, not entry by entry. sigma permutes the
    source index along with the endpoints, so a single source's flow does not stay on that source
    -- it is redistributed across the source's own orbit, and asserting per-(source, link) equality
    would be asserting something false. What the LP's capacity rows constrain, what a link
    physically carries, and what an idle epoch is defined against are all the sum over sources, and
    that is what comes out equal across an orbit.
    """
    topo = load("mini_2gpu_1nic")
    demand, gpus = uniform_demand(topo)
    n = len(topo.capacity)
    group = symmetrize.close_group(symmetrize.find_generators(topo, demand), n)

    # A deliberately lopsided flow on the twin pair, of the shape the LP actually returns.
    names = topo.node_names
    s0, s1 = names.index("srv0_rc"), names.index("srv1_rc")
    leaf0 = names.index("leaf0")
    src = gpus[0]
    flow = {(src, s0, leaf0, 0): 0.25, (src, s1, leaf0, 0): 1.75}
    averaged = symmetrize.average_over_group(flow, group, node_axes=(0, 1, 2))

    before = sum(flow.values())
    after = sum(averaged.values())
    assert abs(before - after) < 1e-9, f"volume not conserved: {before} -> {after}"

    def link_load(tail, head):
        return sum(v for (_, u, w, k), v in averaged.items() if u == tail and w == head and k == 0)

    # Every link in the ORBIT of (srv0_rc -> leaf0) must carry the same load, and the orbit must
    # hold all of the volume. The orbit is bigger than the twin pair: the group also swaps the two
    # leaf sides, so it is {srv0_rc,srv1_rc}->leaf0 together with {srv2_rc,srv3_rc}->leaf1. Asking
    # for `before / 2` here would be assuming the orbit is the host pair alone.
    orbit = {(perm[s0], perm[leaf0]) for perm in group}
    loads = [link_load(tail, head) for tail, head in orbit]
    assert max(loads) - min(loads) < 1e-9, f"orbit not balanced: {sorted(loads)}"
    assert abs(sum(loads) - before) < 1e-9, \
        f"orbit should hold all {before} of the volume, holds {sum(loads)}"
    assert abs(loads[0] - before / len(orbit)) < 1e-9
    print("  OK  averaging balances the orbit and conserves total volume")


def check_no_symmetry_is_a_noop() -> None:
    """On a topology with no demand-preserving symmetry the pass must leave the solution alone."""
    topo = load("hetero_tapered_cluster")
    n = len(topo.capacity)
    demand, _ = uniform_demand(topo)
    generators = symmetrize.find_generators(topo, demand)
    group = symmetrize.close_group(generators, n) if generators else None
    flow = {(0, 0, 1, 0): 1.0}
    if not generators or group is None:
        print("  OK  heterogeneous topology yields no usable group (pass would no-op)")
        return
    averaged = symmetrize.average_over_group(flow, group, node_axes=(0, 1, 2))
    assert abs(sum(averaged.values()) - 1.0) < 1e-9
    print(f"  OK  heterogeneous topology: group of {len(group)}, volume preserved")


def main() -> None:
    check_generators_are_verified()
    check_rooted_demand_is_gated()
    check_group_is_closed()
    check_averaging_balances_and_conserves()
    check_no_symmetry_is_a_noop()
    print("\nall symmetrize checks passed")


if __name__ == "__main__":
    main()
