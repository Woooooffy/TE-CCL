"""
Behavioral tests for ObjectiveType.LEXICOGRAPHIC, run against BOTH formulations. The tiers
differ by formulation:

  LP    latency > host_relay > switch_chain    (LPFormulation.hierarchical_objective_tiers)
  MILP  makespan > host_relay > latency_chain  (AllGatherFormulation.hierarchical_objective_tiers),
        where latency_chain = latency + switch_chain at weight 1.

On the MILP only makespan outranks the relay penalty; latency and switch hops are traded in
the last tier. The probes below decide every case either by makespan or by a tie on it, so the
same routing assertions hold on both.

Each test runs an AllGather between two GPUs on a tiny topology (teccl.topologies.
objective_probes) that offers two routes, and asserts WHICH route the objective picks:

  [1] direct_vs_switch   equal-bandwidth direct link vs one-switch path.
      Latency and host relay tie, so the switch-chain term decides: take the direct link,
      0 switch hops.

  [2] one_vs_two_switch  equal-bandwidth one-switch path vs two-switch path.
      Latency and host relay tie again, so the switch-chain term decides: take the shorter
      chain.

  [3] slow_switch_vs_relay  a slow one-switch path vs a fast path relaying through a passive
      GPU. The relay tier wants to avoid the relay, but the finish time outranks it (latency on
      the LP, makespan on the MILP): when the relay is strictly faster it MUST be used, and when
      the two paths are equally fast the relay must be dropped. Both directions are checked --
      that is what makes it a test of the hierarchy rather than of any single term.

Every test runs on the MILP (AllGatherFormulation) and on the LP (LPFormulation, which is what
the hierarchical solver uses). The LP is the harder case: its flow is continuous and
per-source aggregated, so it can SPLIT a chunk across both routes at identical cost -- exactly
the degeneracy a tie-breaking tier has to resolve. Assertions are therefore written with a
numeric tolerance (TOL) rather than exact equality, and "route not used" means "carries less
than TOL", not "carries a structurally absent variable".

AllGather on the LP requires switch_copy=False (the LP cannot replicate); with two GPUs
exchanging one chunk each there is nothing to replicate, so the probes are unaffected.

Needs Gurobi. Run from the repo root (in the teccl env):
    python -m teccl.examples.objective_lexicographic_test
"""
from itertools import product

from teccl.input_data import (Collective, Formulation, InstanceParams, ObjectiveType,
                              SolutionMethod, TopologyParams, UserInputParams)
from teccl.solvers.allgather import AllGatherFormulation
from teccl.solvers.lp_formulation import LPFormulation
from teccl.topologies.objective_probes import (DirectVsSwitch, FastSwitchVsRelay,
                                               OneVsTwoSwitch, SlowSwitchVsRelay)

EPOCH_DURATION = 1.0    # one chunk per epoch on a LINK_CAPACITY link
NUM_EPOCHS = 8          # far more than any of these probes needs
# Gurobi's default feasibility tolerance here is 1e-4, and the LP reports flows as continuous
# values, so anything at or below this is solver noise rather than a routing decision.
TOL = 1e-4

FORMULATIONS = [Formulation.MILP, Formulation.LP]


def build_topology(cls, passive=()):
    return cls(TopologyParams(name=cls.__name__, chunk_size=1, passive_node_indices=tuple(passive)))


def solve(topology, objective_type, formulation=Formulation.MILP,
          latency_rel_tol=0.0, relay_rel_tol=0.0):
    """Solve AllGather on `topology` with the given objective and return the solved formulation."""
    user_input = UserInputParams()
    user_input.topology = TopologyParams(name=type(topology).__name__, chunk_size=1)
    user_input.instance = InstanceParams(
        collective=Collective.ALLGATHER,
        formulation=formulation,
        num_chunks=1,
        epoch_duration=EPOCH_DURATION,
        num_epochs=NUM_EPOCHS,
        solution_method=SolutionMethod.ONE_SHOT,
        objective_type=objective_type,
        objective_latency_rel_tol=latency_rel_tol,
        objective_relay_rel_tol=relay_rel_tol,
        # The LP aggregates flow per source and has no notion of copy; AllGather on it is only
        # meaningful with switch replication off (see TECCLSolver.get_solver).
        switch_copy=(formulation != Formulation.LP),
    )
    user_input.gurobi.output_flag = 0
    solver_cls = LPFormulation if formulation == Formulation.LP else AllGatherFormulation
    solver = solver_cls(user_input, topology)
    solver.encode_problem()
    return solver


# ---------------------------------------------------------------------------------------
# Reading the solved schedule (formulation-agnostic)
# ---------------------------------------------------------------------------------------

def _value(var):
    """Flow entries the formulation skipped are plain 0.0 floats, not Vars."""
    return var.X if hasattr(var, "X") else float(var)


def _link_flow(solver, s, i, j):
    """
    Total flow of source s on link (i, j) over all epochs. The MILP indexes flow by chunk
    (flow[s][i][j][c][k]) and the LP does not (flow[s][i][j][k]), so branch once here and keep
    every metric below shared between the two.
    """
    per_link = solver.flow[s][i][j]
    if isinstance(solver, LPFormulation):
        return sum(_value(per_link[k]) for k in solver.epochs)
    return sum(_value(per_link[c][k]) for c, k in product(solver.chunks, solver.epochs))


def link_volume(solver):
    """{(i, j): total chunk-volume sent on that link, summed over sources/chunks/epochs}."""
    volume = {}
    for i, j in product(solver.nodes, solver.nodes):
        if solver.topology.capacity[i][j] <= 0:
            continue
        total = sum(_link_flow(solver, s, i, j) for s in solver.sources)
        if total > TOL:
            volume[(i, j)] = total
    return volume


def objective_report(solver):
    """
    What objective the SOLVED model actually carried: the number of Gurobi objectives and,
    for a multi-objective model, each tier's name and optimal value straight from Gurobi
    (ObjNVal), not recomputed from the flows.

    This is the check that the formulation really installed the hierarchy: a single-objective
    run reports NumObj == 1, a LEXICOGRAPHIC run reports 3 named tiers on either formulation.
    Cross-read the tier values against the metrics below -- makespan is minus the epochs from
    the finish on, host_relay and switch_chain must match exactly, latency is the negated sum
    of total_demand_sat, and the MILP's latency_chain is latency + switch_chain.
    """
    model = solver.model
    num_obj = model.NumObj
    tiers = []
    if num_obj > 1:
        for index in range(num_obj):
            model.params.ObjNumber = index
            tiers.append((model.ObjNName, model.ObjNVal))
    return {"num_obj": num_obj, "tiers": tiers}


def metrics(solver):
    """The tier quantities plus the epoch count, read back off the solved model."""
    volume = link_volume(solver)
    switches = set(solver.topology.switch_indices)
    switch_hops = sum(v for (i, j), v in volume.items() if j in switches)

    relay = 0.0
    for n in solver.host_indices():
        for j in solver.nodes:
            if solver.topology.capacity[n][j] <= 0:
                continue
            for s in solver.sources:
                if s == n:
                    continue
                relay += _link_flow(solver, s, n, j)

    # latency = -sum of total_demand_sat over demanded (s, d[, c], k), as both formulations define it
    if isinstance(solver, LPFormulation):
        latency = -sum(_value(solver.total_demand_sat[s][d][k])
                       for s, d, k in product(solver.sources, solver.nodes, solver.epochs)
                       if solver.demand[s][d])
    else:
        latency = -sum(_value(solver.total_demand_sat[s][d][c][k])
                       for s, d, c, k in product(solver.sources, solver.nodes, solver.chunks,
                                                 solver.epochs)
                       if solver.demand[s][d][c])

    report = objective_report(solver)
    return {
        "formulation": Formulation.MILP if isinstance(solver, AllGatherFormulation) else Formulation.LP,
        "epochs": solver.find_demand_satisfied_k() + 1,
        "switch_hops": switch_hops,
        "host_relay": relay if relay > TOL else 0.0,
        "latency": latency,
        "volume": volume,
        "num_obj": report["num_obj"],
        "tiers": report["tiers"],
    }


def _tier_str(m):
    """Gurobi's per-tier optimal values, e.g. "makespan=-7 host_relay=0 latency_chain=-14"."""
    return " ".join(f"{name}={value:g}" for name, value in m["tiers"]) or f"NumObj={m['num_obj']}"


def assert_hierarchy_installed(m):
    """
    The tests below assert routing OUTCOMES, which a formulation could also produce by
    accident (or by its single-objective flow penalty). This asserts the mechanism: the
    solved model carried the named objectives (makespan first on the MILP), and Gurobi's own
    optimal value for the makespan/relay/chain tiers equals what we measured off the solution.
    """
    expected = (["makespan", "host_relay", "latency_chain"] if m["formulation"] == Formulation.MILP
                else ["latency", "host_relay", "switch_chain"])
    assert m["num_obj"] == len(expected), \
        f"expected {len(expected)} lexicographic objectives on the solved model, found " \
        f"{m['num_obj']}: the formulation did not install the hierarchy"
    names = [name for name, _ in m["tiers"]]
    assert names == expected, f"tiers installed in the wrong order: {names}"
    by_name = dict(m["tiers"])
    if "makespan" in by_name:
        # the tier counts the epochs with everything met: finish epoch (epochs - 1) through the end
        # The tier is a sum of CONTINUOUS all_met indicators, and later passes may relax a pinned
        # tier by about its pass's relative MIPGap (1e-4 x 8 here), which they absorb (-7.9992 on
        # probe [1] since the merged latency_chain tier). The finish itself cannot move: each
        # all_met <= a binary met, so any slack below one epoch leaves the finish epoch intact.
        assert abs(by_name["makespan"] + (NUM_EPOCHS - m["epochs"] + 1)) < 0.5, \
            f"Gurobi's makespan tier ({by_name['makespan']:g}) disagrees with the finish " \
            f"epoch ({m['epochs']} of {NUM_EPOCHS})"
    assert abs(by_name["host_relay"] - m["host_relay"]) <= TOL, \
        f"Gurobi's host_relay tier ({by_name['host_relay']:g}) disagrees with the flows " \
        f"({m['host_relay']:g}) -- the tier is not measuring what the test measures"
    if "switch_chain" in by_name:
        assert abs(by_name["switch_chain"] - m["switch_hops"]) <= TOL, \
            f"Gurobi's switch_chain tier ({by_name['switch_chain']:g}) disagrees with the flows " \
            f"({m['switch_hops']:g})"
    if "latency_chain" in by_name:
        measured = m["latency"] + m["switch_hops"]
        assert abs(by_name["latency_chain"] - measured) <= TOL, \
            f"Gurobi's latency_chain tier ({by_name['latency_chain']:g}) disagrees with latency " \
            f"{m['latency']:g} + switch hops {m['switch_hops']:g} measured off the solution"


def _used(metrics_dict, links):
    """Volume carried on any of `links` (both directions of each are counted separately)."""
    return sum(metrics_dict["volume"].get(link, 0.0) for link in links)


# ---------------------------------------------------------------------------------------
# The three tests
# ---------------------------------------------------------------------------------------

def test_direct_vs_switch(formulation=Formulation.MILP):
    """[1] Direct link beats an equal-bandwidth one-switch path (chain length 0 vs 1)."""
    topo = build_topology(DirectVsSwitch)
    m = metrics(solve(topo, ObjectiveType.LEXICOGRAPHIC, formulation))

    assert_hierarchy_installed(m)
    assert m["switch_hops"] <= TOL, \
        f"expected the direct link only, but {m['switch_hops']:g} chunk-hops entered the switch: {m['volume']}"
    assert m["volume"].get((0, 1), 0) >= 1 - TOL and m["volume"].get((1, 0), 0) >= 1 - TOL, \
        f"both directions should ride the direct link in full: {m['volume']}"
    assert m["host_relay"] <= TOL, f"a 2-GPU direct exchange relays nothing: {m['volume']}"
    print(f"    [1] direct_vs_switch      OK  epochs={m['epochs']} switch_hops={m['switch_hops']:g} "
          f"tiers={_tier_str(m)} links={sorted(m['volume'])}")
    return m


def test_one_vs_two_switch(formulation=Formulation.MILP):
    """[2] The one-switch path beats the equal-bandwidth two-switch path (chain length 1 vs 2)."""
    topo = build_topology(OneVsTwoSwitch)
    m = metrics(solve(topo, ObjectiveType.LEXICOGRAPHIC, formulation))

    assert_hierarchy_installed(m)
    long_path_links = [(0, 3), (3, 4), (4, 3), (3, 0), (1, 4), (4, 1)]
    on_long_path = _used(m, long_path_links)
    assert on_long_path <= TOL, \
        f"the two-switch path should carry nothing, carried {on_long_path:g}: {m['volume']}"
    assert abs(m["switch_hops"] - 2) <= TOL, \
        f"two chunks over a one-switch path = 2 switch hops, got {m['switch_hops']:g}: {m['volume']}"
    print(f"    [2] one_vs_two_switch     OK  epochs={m['epochs']} switch_hops={m['switch_hops']:g} "
          f"tiers={_tier_str(m)} links={sorted(m['volume'])}")
    return m


def test_slow_switch_vs_relay(formulation=Formulation.MILP):
    """
    [3] Finish time outranks the relay penalty, and only outranks it when it actually pays.

    (a) switch path 4x slower than the relay path -> the relay through the passive GPU is used,
        even though the relay tier charges for it, because the finish is strictly earlier with
        it (latency on the LP; makespan on the MILP, where it is the only tier above relay).
    (b) same shape with the switch path made just as fast -> the finish ties, so the relay tier
        takes over and the relay is dropped.
    """
    passive = SlowSwitchVsRelay.PASSIVE
    slow = metrics(solve(build_topology(SlowSwitchVsRelay, passive=passive),
                         ObjectiveType.LEXICOGRAPHIC, formulation))
    fast = metrics(solve(build_topology(FastSwitchVsRelay, passive=passive),
                         ObjectiveType.LEXICOGRAPHIC, formulation))

    assert_hierarchy_installed(slow)
    assert_hierarchy_installed(fast)
    assert slow["host_relay"] > TOL, \
        f"the relay path is strictly faster here, so the finish time must overrule the relay tier: {slow['volume']}"
    assert slow["epochs"] <= 3, \
        f"the relay path takes 2 epochs (2 hops); the slow switch path takes at least 4, so a " \
        f"finish-first objective must land at or under 3, got {slow['epochs']}: {slow['volume']}"
    assert fast["host_relay"] <= TOL, \
        f"with both paths equally fast the relay is pure relay-tier cost and should be dropped: {fast['volume']}"
    assert fast["switch_hops"] > TOL, \
        f"dropping the relay means going through the switch: {fast['volume']}"
    print(f"    [3] slow_switch_vs_relay  OK  slow: epochs={slow['epochs']} relay={slow['host_relay']:g} "
          f"| fast: epochs={fast['epochs']} relay={fast['host_relay']:g} switch_hops={fast['switch_hops']:g}")
    return {"slow": slow, "fast": fast}


TESTS = [test_direct_vs_switch, test_one_vs_two_switch, test_slow_switch_vs_relay]


def run_all(formulation):
    print(f"\n{formulation.name}:")
    for test in TESTS:
        test(formulation)


def main():
    for formulation in FORMULATIONS:
        run_all(formulation)
    print(f"\nAll {len(TESTS) * len(FORMULATIONS)} lexicographic-objective behavior tests passed "
          f"({', '.join(f.name for f in FORMULATIONS)}).")


if __name__ == "__main__":
    main()
