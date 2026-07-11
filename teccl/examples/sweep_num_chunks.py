"""
Sweeps num_chunks (holding each source's total data volume fixed, i.e.
chunk_size = total_data / num_chunks) on a given topology, to empirically
test whether the atomic-chunk MILP formulation is leaving meaningful
performance on the table relative to a continuous-rate model -- before
committing to a larger solver reformulation.

Run on a machine with gurobipy installed:

    python teccl/examples/sweep_num_chunks.py
    python teccl/examples/sweep_num_chunks.py --num-chunks 1 2 4 8 16 32 64
    python teccl/examples/sweep_num_chunks.py --oversubscription 3.7  # isolate ceil()-rounding waste

Compare a run with a clean divisor (--oversubscription 4, the default
IncastSwitch ratio) against a run with an awkward ratio (e.g. 3.7) to
separate two distinct effects:
  - generic chunk-granularity approximation (present in both runs; should
    shrink and plateau as num_chunks grows in both cases)
  - ceil()-rounding waste specific to non-integer capacity ratios (extra
    gap that persists in the 3.7 run but not the clean 4 run)
"""
import argparse
import copy
import json
import pathlib
import time

from teccl.input_data import (Collective, EpochType, ObjectiveType,
                               SolutionMethod, UserInputParams)
from teccl.scheduler import TECCLSolver
from teccl.topologies import incast_switch

SAMPLE_INPUT = pathlib.Path(__file__).parent / \
    "sample_inputs" / "incast_switch_sample.json"
OUTPUT_DIR = pathlib.Path(__file__).parent / \
    "experiments" / "num_chunks_sweep"


def load_base_user_input(sample_path: pathlib.Path) -> UserInputParams:
    user_input = UserInputParams()
    with open(sample_path, 'r') as f:
        raw = json.load(f)
    for k, v in raw['TopologyParams'].items():
        user_input.topology.__setattr__(k, v)
    for k, v in raw['GurobiParams'].items():
        user_input.gurobi.__setattr__(k, v)
    for k, v in raw['InstanceParams'].items():
        if k == 'objective_type':
            user_input.instance.objective_type = ObjectiveType(v)
        elif k == 'solution_method':
            user_input.instance.solution_method = SolutionMethod(v)
        elif k == 'collective':
            user_input.instance.collective = Collective(v)
        elif k == 'epoch_type':
            user_input.instance.epoch_type = EpochType(v)
        else:
            user_input.instance.__setattr__(k, v)
    return user_input


def run_sweep(num_chunks_values, oversubscription: float = 4,
              time_limit_hours: float = None, mip_gap: float = None):
    # IncastSwitch.OVERSUBSCRIPTION is a plain class attribute read fresh
    # at construction time, so overriding it here before building any
    # topology instances is enough to change the ratio for this sweep.
    incast_switch.IncastSwitch.OVERSUBSCRIPTION = oversubscription

    base = load_base_user_input(SAMPLE_INPUT)
    total_data = base.instance.num_chunks * base.topology.chunk_size

    # Applied uniformly to every n in this sweep, not just the ones that
    # are slow to solve -- so points stay comparable (mixing tight-gap and
    # loose-gap results across n would bias the finish_time trend we're
    # trying to read off this sweep).
    if time_limit_hours is not None:
        base.gurobi.time_limit = time_limit_hours
    if mip_gap is not None:
        base.gurobi.mip_gap = mip_gap

    # epoch_type=FASTEST_LINK defines epoch_duration = chunk_size / fast_rate,
    # so it moves in lockstep with chunk_size as num_chunks grows -- that
    # keeps every capacity ratio (and beta_num_back) invariant to n, which
    # defeats the point of this sweep (see 2026-07-11 finding: it measures
    # atomic per-chunk reservation overhead, not discretization fidelity).
    # Fix epoch_duration once, from the n=1 baseline, and hold it fixed
    # across the whole sweep so smaller chunks actually pack more tightly
    # into a fixed real-time epoch.
    baseline_topology = incast_switch.IncastSwitch(base.topology)
    fixed_epoch_duration = baseline_topology.epoch_duration_fast_link

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for n in num_chunks_values:
        user_input = copy.deepcopy(base)
        user_input.instance.num_chunks = n
        user_input.topology.chunk_size = total_data / n
        user_input.instance.epoch_type = EpochType.USER_INPUT
        user_input.instance.epoch_duration = fixed_epoch_duration
        user_input.instance.num_epochs = -1
        user_input.instance.schedule_output_file = str(
            OUTPUT_DIR / f"n{n}_osub{str(oversubscription).replace('.', '_')}.json")

        start = time.time()
        TECCLSolver(user_input).solve()
        wall = time.time() - start

        output_path = pathlib.Path(user_input.instance.schedule_output_file)
        if not output_path.exists():
            # scheduler.py's solve() only writes a file when at least one
            # epoch-count probe in the iterative binary search reaches
            # GRB.OPTIMAL; a probe that only hits GRB.TIME_LIMIT is
            # discarded even if Gurobi already had a feasible incumbent
            # (see encode_problem()'s `if status != GRB.OPTIMAL: return`).
            # Record the miss and keep going rather than crash the sweep.
            print(f"num_chunks={n:>4}  NO SCHEDULE FOUND in {wall:.1f}s wall time "
                  f"(likely hit GurobiParams.time_limit={user_input.gurobi.time_limit}h "
                  f"before reaching GRB.OPTIMAL at mip_gap={user_input.gurobi.mip_gap} -- "
                  f"try --time-limit-hours / --mip-gap, or drop this n)")
            results.append({
                "num_chunks": n,
                "chunk_size": user_input.topology.chunk_size,
                "oversubscription": oversubscription,
                "epochs_required": None,
                "collective_finish_time": None,
                "algo_bandwidth": None,
                "solver_time_s": None,
                "wall_time_s": wall,
                "status": "no_schedule_found",
            })
            continue

        with open(output_path) as f:
            out = json.load(f)
        row = {
            "num_chunks": n,
            "chunk_size": user_input.topology.chunk_size,
            "oversubscription": oversubscription,
            "epochs_required": out["3-Epochs_Required"],
            "collective_finish_time": out["4-Collective_Finish_Time"],
            "algo_bandwidth": out["5-Algo_Bandwidth"],
            "solver_time_s": out["Solver_Time"],
            "wall_time_s": wall,
        }
        results.append(row)
        print(f"num_chunks={n:>4}  finish_time={row['collective_finish_time']:.6f}  "
              f"algo_bw={row['algo_bandwidth']:.3f}  solver_time={row['solver_time_s']:.2f}s")

    summary_path = OUTPUT_DIR / f"summary_osub{str(oversubscription).replace('.', '_')}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, nargs="+",
                         default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--oversubscription", type=float, default=4)
    parser.add_argument("--time-limit-hours", type=float, default=None,
                         help="Overrides GurobiParams.time_limit for every n in this sweep "
                              "(sample default: 0.5). Raise this if larger num_chunks values "
                              "hit the limit without reaching GRB.OPTIMAL.")
    parser.add_argument("--mip-gap", type=float, default=None,
                         help="Overrides GurobiParams.mip_gap for every n in this sweep "
                              "(sample default: 1e-4). Loosen this (e.g. 1e-2) to let Gurobi "
                              "accept a near-optimal incumbent sooner on harder n.")
    args = parser.parse_args()
    run_sweep(args.num_chunks, args.oversubscription,
              args.time_limit_hours, args.mip_gap)
