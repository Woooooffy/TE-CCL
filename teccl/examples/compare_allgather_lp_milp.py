"""
Compare the AllGather schedule produced by the MILP formulation vs. the (copy-free) LP
formulation on a switchless topology (DGX1).

Both runs use identical inputs except InstanceParams.formulation (MILP vs LP). The LP is
demand-matrix driven and collective-agnostic (see teccl/solvers/lp_formulation.py); for
AllGather it aggregates flow per source and cannot replicate/copy, so it finds unicast-style
flows and is expected to be equal-or-worse than the MILP (which allows multicast copy),
never better. This script surfaces that gap.

Run from the repo root:
    python -m teccl.examples.compare_allgather_lp_milp
"""
import json
import pathlib

from teccl.input_data import UserInputParams, Collective, Formulation, EpochType, ObjectiveType, SolutionMethod
from teccl.scheduler import TECCLSolver

HERE = pathlib.Path(__file__).parent
INPUTS = HERE / "sample_inputs"


def _load_input(path: pathlib.Path) -> UserInputParams:
    user_input = UserInputParams()
    with open(path, "r") as jf:
        args = json.load(jf)
    for k, v in args["TopologyParams"].items():
        setattr(user_input.topology, k, v)
    for k, v in args["GurobiParams"].items():
        setattr(user_input.gurobi, k, v)
    for k, v in args["InstanceParams"].items():
        if k == "objective_type":
            setattr(user_input.instance, k, ObjectiveType(v))
        elif k == "solution_method":
            setattr(user_input.instance, k, SolutionMethod(v))
        elif k == "collective":
            setattr(user_input.instance, k, Collective(v))
        elif k == "formulation":
            setattr(user_input.instance, k, Formulation(v))
        elif k == "epoch_type":
            setattr(user_input.instance, k, EpochType(v))
        else:
            setattr(user_input.instance, k, v)
    return user_input


def _run(input_path: pathlib.Path) -> dict:
    user_input = _load_input(input_path)
    TECCLSolver(user_input).solve()
    out_path = pathlib.Path(user_input.instance.schedule_output_file)
    with open(out_path, "r") as f:
        return json.load(f)


def main() -> None:
    milp = _run(INPUTS / "dgx1_allgather_milp.json")
    lp = _run(INPUTS / "dgx1_allgather_lp.json")

    keys = [
        "3-Epochs_Required",
        "4-Collective_Finish_Time",
        "5-Algo_Bandwidth",
    ]
    print("\n" + "=" * 60)
    print(f"{'Metric':<28}{'MILP':>15}{'LP (no-copy)':>17}")
    print("-" * 60)
    for k in keys:
        print(f"{k:<28}{str(milp.get(k)):>15}{str(lp.get(k)):>17}")
    print("=" * 60)
    milp_t = milp.get("4-Collective_Finish_Time")
    lp_t = lp.get("4-Collective_Finish_Time")
    if milp_t is not None and lp_t is not None:
        print(f"LP finish time / MILP finish time = {lp_t / milp_t:.3f} "
              f"(LP has no copy, so expect >= 1.0)")


if __name__ == "__main__":
    main()
