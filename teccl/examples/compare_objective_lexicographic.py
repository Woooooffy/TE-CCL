"""
Run all the objective probes (teccl.examples.objective_lexicographic_test) under the NEW
hierarchical objective (ObjectiveType.LEXICOGRAPHIC, 6) and the paper's objective
(ObjectiveType.PAPER, 3), on BOTH the MILP and the LP formulation, and aggregate what each
combination chose.

For every probe it reports the three tier quantities read back off the solved model --
epochs to finish (latency), chunk-volume relayed through a GPU (host relay), and chunk-hops
entering a switch (switch chain length) -- plus the links that actually carried traffic. The
probes are built so latency ties between the alternatives (except case 3a, where it is meant
not to), so any difference in the last two columns is the objective's doing.

The LP column is the interesting one: with continuous, per-source-aggregated flow the two
routes of each probe cost exactly the same, so a solution can split across them arbitrarily
(LP degeneracy). Watch for PAPER splitting a chunk over both routes while LEXICOGRAPHIC
commits to one -- the "Links used" column shows it directly.

Needs Gurobi. Run from the repo root (in the teccl env):
    python -m teccl.examples.compare_objective_lexicographic
"""
from teccl.input_data import Formulation, ObjectiveType
from teccl.examples.objective_lexicographic_test import (
    FORMULATIONS, TESTS, build_topology, metrics, solve)
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


def _row(label, formulation, objective_name, m):
    links = ", ".join(f"{i}->{j}:{v:g}" for (i, j), v in sorted(m["volume"].items()))
    # NumObj says which objective the SOLVED model actually carried: 1 for the single
    # weighted objective, 3 when the lexicographic hierarchy was installed.
    return (f"{label:<26}{formulation.name:<7}{objective_name:<20}{m['num_obj']:>6}"
            f"{m['epochs']:>8}{m['host_relay']:>12g}{m['switch_hops']:>14g}  {links}")


def run_comparison():
    results = {}
    print("\n" + "=" * 130)
    print(f"{'Case':<26}{'Form.':<7}{'Objective':<20}{'NumObj':>6}{'Epochs':>8}"
          f"{'Host relay':>12}{'Switch hops':>14}  Links used")
    print("-" * 130)
    for label, topo_cls, passive, expectation in CASES:
        for formulation in FORMULATIONS:
            for objective_name, objective_type in OBJECTIVES:
                # A fresh topology per solve: the formulation reads capacity/alpha off it and
                # the solved model holds references, so sharing one would blur the results.
                topology = build_topology(topo_cls, passive=passive)
                m = metrics(solve(topology, objective_type, formulation))
                results[(label, formulation, objective_name)] = m
                print(_row(label, formulation, objective_name, m))
                if m["tiers"]:
                    print(f"{'':<53}Gurobi tier values: "
                          + "  ".join(f"{name}={value:g}" for name, value in m["tiers"]))
        print(f"{'':<26}expected of LEXICOGRAPHIC: {expectation}")
        print("-" * 130)
    return results


def summarize(results):
    """One line per case/formulation: how the new objective moved each tier quantity."""
    print("\nSummary (LEXICOGRAPHIC vs PAPER, lower is better on every column):")
    for label, _, _, _ in CASES:
        for formulation in FORMULATIONS:
            paper = results[(label, formulation, "PAPER (3)")]
            lex = results[(label, formulation, "LEXICOGRAPHIC (6)")]
            deltas = []
            for key, name in (("epochs", "epochs"), ("host_relay", "relay"),
                              ("switch_hops", "hops")):
                delta = lex[key] - paper[key]
                deltas.append(f"{name} {paper[key]:g}->{lex[key]:g}"
                              + ("" if abs(delta) < 1e-9 else f" ({delta:+g})"))
            print(f"  {label:<26}{formulation.name:<7}{' | '.join(deltas)}")


def run_assertions():
    """Re-run the behavioral tests themselves, per formulation, collecting failures."""
    print("\nRunning the behavioral assertions on the new objective:")
    failures = []
    for formulation in FORMULATIONS:
        print(f"  {formulation.name}:")
        for test in TESTS:
            try:
                test(formulation)
            except AssertionError as e:
                failures.append(f"{formulation.name}/{test.__name__}: {e}")
                print(f"    FAILED {test.__name__}: {e}")
    total = len(TESTS) * len(FORMULATIONS)
    print(f"\n{total - len(failures)}/{total} behavior tests passed.")
    return failures


def main():
    results = run_comparison()
    summarize(results)
    return 1 if run_assertions() else 0


if __name__ == "__main__":
    raise SystemExit(main())
