"""
Run all three objective probes (teccl.examples.objective_lexicographic_test) under the NEW
hierarchical objective (ObjectiveType.LEXICOGRAPHIC, 6) and the paper's objective
(ObjectiveType.PAPER, 3), and aggregate what each one chose.

For every probe it reports the three tier quantities read back off the solved model --
epochs to finish (latency), chunk-volume relayed through a GPU (host relay), and chunk-hops
entering a switch (switch chain length) -- plus the links that actually carried traffic. The
probes are built so latency ties between the alternatives (except case 3a, where it is meant
not to), so any difference in the last two columns is the objective's doing.

Needs Gurobi. Run from the repo root (in the teccl env):
    python -m teccl.examples.compare_objective_lexicographic
"""
from teccl.input_data import ObjectiveType
from teccl.examples.objective_lexicographic_test import (
    build_topology, metrics, solve, TESTS)
from teccl.topologies.objective_probes import (DirectVsSwitch, FastSwitchVsRelay,
                                               OneVsTwoSwitch, SlowSwitchVsRelay)


# (label, topology class, passive nodes, what the new objective is expected to do)
CASES = [
    ("1. direct vs 1-switch", DirectVsSwitch, (), "use the direct link (0 switch hops)"),
    ("2. 1-switch vs 2-switch", OneVsTwoSwitch, (), "use the shorter chain (2 switch hops)"),
    ("3a. slow switch vs relay", SlowSwitchVsRelay, SlowSwitchVsRelay.PASSIVE,
     "relay through the passive GPU (latency outranks relay)"),
    ("3b. fast switch vs relay", FastSwitchVsRelay, SlowSwitchVsRelay.PASSIVE,
     "drop the relay, go through the switch"),
]

OBJECTIVES = [
    ("PAPER (3)", ObjectiveType.PAPER),
    ("LEXICOGRAPHIC (6)", ObjectiveType.LEXICOGRAPHIC),
]


def _row(label, objective_name, m):
    links = ", ".join(f"{i}->{j}:{v:g}" for (i, j), v in sorted(m["volume"].items()))
    return (f"{label:<26}{objective_name:<20}{m['epochs']:>8}{m['host_relay']:>12g}"
            f"{m['switch_hops']:>14g}  {links}")


def run_comparison():
    results = {}
    print("\n" + "=" * 120)
    print(f"{'Case':<26}{'Objective':<20}{'Epochs':>8}{'Host relay':>12}{'Switch hops':>14}  Links used")
    print("-" * 120)
    for label, topo_cls, passive, expectation in CASES:
        for objective_name, objective_type in OBJECTIVES:
            # A fresh topology per solve: the formulation reads capacity/alpha off it and the
            # solved model holds references, so sharing one across runs would blur the results.
            topology = build_topology(topo_cls, passive=passive)
            m = metrics(solve(topology, objective_type))
            results[(label, objective_name)] = m
            print(_row(label, objective_name, m))
        print(f"{'':<26}expected of LEXICOGRAPHIC: {expectation}")
        print("-" * 120)
    return results


def summarize(results):
    """One line per case: did the new objective improve on PAPER, tie, or regress."""
    print("\nSummary (LEXICOGRAPHIC vs PAPER, lower is better on every column):")
    for label, _, _, _ in CASES:
        paper = results[(label, "PAPER (3)")]
        lex = results[(label, "LEXICOGRAPHIC (6)")]
        deltas = []
        for key, name in (("epochs", "epochs"), ("host_relay", "relay"), ("switch_hops", "hops")):
            delta = lex[key] - paper[key]
            deltas.append(f"{name} {paper[key]:g}->{lex[key]:g}"
                          + ("" if delta == 0 else f" ({delta:+g})"))
        print(f"  {label:<26}{' | '.join(deltas)}")


def main():
    results = run_comparison()
    summarize(results)

    print("\nRunning the behavioral assertions on the new objective:")
    failures = []
    for test in TESTS:
        try:
            test()
        except AssertionError as e:
            failures.append(f"{test.__name__}: {e}")
            print(f"    FAILED {test.__name__}: {e}")
    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} behavior tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
