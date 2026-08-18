"""
Behavioral tests for ObjectiveType.LEXICOGRAPHIC (the latency > host-relay > switch-chain
hierarchy in teccl/solvers/base_formulation.py).

Each test runs an AllGather between two GPUs on a tiny topology (teccl.topologies.
objective_probes) that offers two routes, and asserts WHICH route the objective picks:

  [1] direct_vs_switch   equal-bandwidth direct link vs one-switch path.
      Latency and host relay tie, so tier 3 decides: take the direct link, 0 switch hops.

  [2] one_vs_two_switch  equal-bandwidth one-switch path vs two-switch path.
      Latency and host relay tie again, so tier 3 decides: take the shorter chain.

  [3] slow_switch_vs_relay  a slow one-switch path vs a fast path relaying through a passive
      GPU. Tier 2 wants to avoid the relay, but tier 1 outranks it: when the relay is strictly
      faster it MUST be used, and when the two paths are equally fast the relay must be
      dropped. Both directions are checked -- that is what makes it a test of the hierarchy
      rather than of any single term.

Needs Gurobi. Run from the repo root (in the teccl env):
    python -m teccl.examples.objective_lexicographic_test
"""
from itertools import product

from teccl.input_data import (Collective, Formulation, InstanceParams, ObjectiveType,
                              SolutionMethod, TopologyParams, UserInputParams)
from teccl.solvers.allgather import AllGatherFormulation
from teccl.topologies.objective_probes import (DirectVsSwitch, FastSwitchVsRelay,
                                               OneVsTwoSwitch, SlowSwitchVsRelay)

EPOCH_DURATION = 1.0    # one chunk per epoch on a LINK_CAPACITY link
NUM_EPOCHS = 8          # far more than any of these probes needs


def build_topology(cls, passive=()):
    return cls(TopologyParams(name=cls.__name__, chunk_size=1, passive_node_indices=tuple(passive)))


def solve(topology, objective_type, latency_rel_tol=0.0, relay_rel_tol=0.0):
    """Solve AllGather on `topology` with the given objective and return the solved formulation."""
    user_input = UserInputParams()
    user_input.topology = TopologyParams(name=type(topology).__name__, chunk_size=1)
    user_input.instance = InstanceParams(
        collective=Collective.ALLGATHER,
        formulation=Formulation.MILP,
        num_chunks=1,
        epoch_duration=EPOCH_DURATION,
        num_epochs=NUM_EPOCHS,
        solution_method=SolutionMethod.ONE_SHOT,
        objective_type=objective_type,
        objective_latency_rel_tol=latency_rel_tol,
        objective_relay_rel_tol=relay_rel_tol,
    )
    user_input.gurobi.output_flag = 0
    solver = AllGatherFormulation(user_input, topology)
    solver.encode_problem()
    return solver


# ---------------------------------------------------------------------------------------
# Reading the solved schedule
# ---------------------------------------------------------------------------------------

def _value(var):
    """Flow entries that initialize_variables skipped are plain 0.0 floats, not Vars."""
    return var.X if hasattr(var, "X") else float(var)


def link_volume(solver):
    """{(i, j): total chunks sent on that link across all sources/chunks/epochs}."""
    volume = {}
    for i, j in product(solver.nodes, solver.nodes):
        if solver.topology.capacity[i][j] <= 0:
            continue
        total = 0.0
        for s, c, k in product(solver.sources, solver.chunks, solver.epochs):
            total += _value(solver.flow[s][i][j][c][k])
        if total > 1e-6:
            volume[(i, j)] = total
    return volume


def metrics(solver):
    """The three tier quantities plus the epoch count, read back off the solved model."""
    volume = link_volume(solver)
    switches = set(solver.topology.switch_indices)
    switch_hops = sum(v for (i, j), v in volume.items() if j in switches)

    relay = 0.0
    for n in solver.host_indices():
        for j in solver.nodes:
            if solver.topology.capacity[n][j] <= 0:
                continue
            for s, c, k in product(solver.sources, solver.chunks, solver.epochs):
                if s == n:
                    continue
                relay += _value(solver.flow[s][n][j][c][k])

    return {
        "epochs": solver.find_demand_satisfied_k() + 1,
        "switch_hops": switch_hops,
        "host_relay": relay,
        "volume": volume,
    }


# ---------------------------------------------------------------------------------------
# The three tests
# ---------------------------------------------------------------------------------------

def test_direct_vs_switch():
    """[1] Direct link beats an equal-bandwidth one-switch path (tier 3, chain length 0 vs 1)."""
    topo = build_topology(DirectVsSwitch)
    m = metrics(solve(topo, ObjectiveType.LEXICOGRAPHIC))

    assert m["switch_hops"] == 0, \
        f"expected the direct link only, but {m['switch_hops']} chunk-hops entered the switch: {m['volume']}"
    assert m["volume"].get((0, 1), 0) >= 1 and m["volume"].get((1, 0), 0) >= 1, \
        f"both directions should ride the direct link: {m['volume']}"
    assert m["host_relay"] == 0, f"a 2-GPU direct exchange relays nothing: {m['volume']}"
    print(f"[1] direct_vs_switch      OK  epochs={m['epochs']} switch_hops={m['switch_hops']} "
          f"links={sorted(m['volume'])}")
    return m


def test_one_vs_two_switch():
    """[2] The one-switch path beats the equal-bandwidth two-switch path (chain length 1 vs 2)."""
    topo = build_topology(OneVsTwoSwitch)
    m = metrics(solve(topo, ObjectiveType.LEXICOGRAPHIC))

    long_path_links = [(0, 3), (3, 4), (4, 3), (3, 0), (1, 4), (4, 1)]
    on_long_path = sum(m["volume"].get(link, 0) for link in long_path_links)
    assert on_long_path == 0, \
        f"the two-switch path should carry nothing, carried {on_long_path}: {m['volume']}"
    assert m["switch_hops"] == 2, \
        f"two chunks over a one-switch path = 2 switch hops, got {m['switch_hops']}: {m['volume']}"
    print(f"[2] one_vs_two_switch     OK  epochs={m['epochs']} switch_hops={m['switch_hops']} "
          f"links={sorted(m['volume'])}")
    return m


def test_slow_switch_vs_relay():
    """
    [3] Latency outranks the relay penalty, and only outranks it when it actually pays.

    (a) switch path 4x slower than the relay path -> the relay through the passive GPU is used,
        even though tier 2 charges for it, because tier 1 is strictly better with it.
    (b) same shape with the switch path made just as fast -> latency ties, so tier 2 takes over
        and the relay is dropped.
    """
    slow = metrics(solve(build_topology(SlowSwitchVsRelay, passive=SlowSwitchVsRelay.PASSIVE),
                         ObjectiveType.LEXICOGRAPHIC))
    fast = metrics(solve(build_topology(FastSwitchVsRelay, passive=SlowSwitchVsRelay.PASSIVE),
                         ObjectiveType.LEXICOGRAPHIC))

    assert slow["host_relay"] > 0, \
        f"the relay path is strictly faster here, so tier 1 must overrule tier 2: {slow['volume']}"
    assert slow["epochs"] <= 3, \
        f"the relay path takes 2 epochs (2 hops); the slow switch path takes at least 4, so a " \
        f"latency-first objective must land at or under 3, got {slow['epochs']}: {slow['volume']}"
    assert fast["host_relay"] == 0, \
        f"with both paths equally fast the relay is pure tier-2 cost and should be dropped: {fast['volume']}"
    assert fast["switch_hops"] > 0, \
        f"dropping the relay means going through the switch: {fast['volume']}"
    print(f"[3] slow_switch_vs_relay  OK  slow: epochs={slow['epochs']} relay={slow['host_relay']} "
          f"| fast: epochs={fast['epochs']} relay={fast['host_relay']} switch_hops={fast['switch_hops']}")
    return {"slow": slow, "fast": fast}


TESTS = [test_direct_vs_switch, test_one_vs_two_switch, test_slow_switch_vs_relay]


def main():
    for test in TESTS:
        test()
    print(f"\nAll {len(TESTS)} lexicographic-objective behavior tests passed.")


if __name__ == "__main__":
    main()
