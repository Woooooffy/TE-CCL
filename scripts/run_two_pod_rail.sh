#!/bin/bash
# Launch the hierarchical pipeline for TwoPodRail (16 GPUs / 8 nodes / 4 leaves / 2 UNEQUAL
# spines, 22 fine nodes -> 14 coarse) on the remote (lanka/SLURM), plus the FLAT ground-truth
# solve of the same problem, and diff them.
#
# This is the CONTRIVED sibling of run_rail_phase3.sh and run_hetero_phase3.sh. The hetero
# cluster is the irregularity stress test and the rail topology is the at-scale target; this one
# is the CORRECTNESS ORACLE. It is deliberately small enough (22 nodes) that the flat solve is
# tractable, so for the first time the hierarchical answer can be compared against a real
# optimum instead of only against its own invariants.
#
# Unlike the other two scripts this one goes through the UNIFIED entry point -- `python -m teccl
# solve -i <json>` with "hierarchical": true -- rather than a per-topology driver module. There
# is no hierarchy_coarse_solve_two_pod_rail.py and there should not be: solve_hierarchical owns
# the orchestration now, and the JSON carries everything that used to be driver argv.
#
# WHAT THIS TOPOLOGY IS FOR. Two design goals, and they want OPPOSITE ends of one knob (the
# GPU->leaf bandwidth H), so they are two configurations of the same graph:
#
#   [multipath]  H = 50, spines 50/25.  The spine cut binds. Both spine links out of every leaf
#       are saturated for the whole makespan, so the optimal split is FORCED and non-dyadic at
#       2/3 via spine0 and 1/3 via spine1. An equal-split (ECMP-style) router is not merely
#       arbitrary here, it is provably wrong -- it lands at 0.64 against an optimum of 0.4267,
#       a 1.50x regression. That is the multipath signal.
#   [rate]  H = 30, same spines.  The binding cut moves onto the GPU->leaf link, which pins BOTH
#       streams on it with zero slack: cross-pod 17.14 + intra-pod 12.86 = 30.00 = exactly line
#       rate. Under the multipath config that link has 34% headroom, which silently absorbs a
#       rate-oblivious emitter's mistakes; here it cannot. That is the rate signal.
#
# Both cuts can only bind at once at the knife edge U/H = 2.286, which hands the LP a degenerate
# vertex -- hence two configs rather than one tuned value.
#
# Outputs: Schedules/two_pod_rail[_hostbound]_allgather_{coarse,identities,intra,flat}.json,
# Schedules/two_pod_rail_allgather_flatsolve.json (the oracle), xml/two_pod_rail*.xml, and the
# log in logs/teccl-<jobid>.out|.err.
#
# What to look for in the .out log:
#   "T*" / finish time -- THE MAIN THING. The multipath run should land at 0.4267 s at
#       chunk_size 1 GB, and the FLAT oracle in step [4/4] should agree. Hierarchical ABOVE flat
#       is the abstraction's cost and is the number worth reporting; hierarchical BELOW flat
#       means one of the two is wrong, and it is nearly always the hierarchical one.
#   RAW LP FLOW VARS on the leaf->spine links -- the multipath payoff. Every leaf should show
#       its cross-pod traffic split 2:1 between spine0 and spine1, i.e. per-GPU 12.5 and 6.25.
#       A 50/50 split means the solver is ignoring the capacity asymmetry; a 100/0 split means
#       it found a single path and the cut is not as tight as the analysis claims (check that
#       LEAF_SPINE_BW was not overridden).
#   per-op `rate` in the emitted XML -- the rate payoff, and the known-weak seam. ncclize has
#       dropped solver rate information four separate times. The host-bound run in step [3/4] is
#       the one that catches it: with zero slack on the GPU->leaf link, a schedule that emits
#       line-rate sends instead of 17.14 / 12.86 cannot meet 0.4667 and check_implements or the
#       pacing residuals will say so.
#   "twin groups (symmetry)" -- should be ABSENT/EMPTY. TwoPodRail declares no
#       equivalent_node_indices on purpose (unequal spines, no two leaves alike). If a twin
#       group appears, LEAF_SPINE_BW has been made uniform somewhere and the topology has
#       reverted to the degenerate symmetric tie it exists to avoid.
#
# Usage (from repo root):
#   sbatch scripts/run_two_pod_rail.sh              # both configs + flat oracle
#   sbatch scripts/run_two_pod_rail.sh multipath    # H=50 only
#   sbatch scripts/run_two_pod_rail.sh rate         # H=30 only
#   sbatch scripts/run_two_pod_rail.sh flat         # flat oracle only
#
# Run locally instead (needs a Gurobi license on this box) -- it is small, this is realistic:
#   python -m teccl solve -i teccl/examples/sample_inputs/two_pod_rail_hierarchical_allgather.json
#SBATCH --job-name=teccl-2pod
#SBATCH --output=logs/teccl-%j.out
#SBATCH --error=logs/teccl-%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --account=commit
#SBATCH --qos=commit-main
#SBATCH --partition=lanka-v3

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs Schedules xml

source /data/commit/graphit/wangyj05/workspace/setup.sh

# `python -m` from SLURM_SUBMIT_DIR puts the repo tree ahead of site-packages, so the run uses the
# checked-out code either way -- but say so loudly rather than letting a failed build look like a
# successful one (a wheel build has failed here before on an NFS "Stale file handle").
if ! pip install .; then
    echo "WARNING: pip install failed. Continuing: 'python -m' from the repo root shadows the"
    echo "         installed package, so the run still uses this checkout -- but the installed"
    echo "         teccl is now STALE for anything that imports it differently."
fi

WHICH="${1:-all}"
SAMPLES=teccl/examples/sample_inputs

run_multipath() { [ "$WHICH" = all ] || [ "$WHICH" = multipath ]; }
run_rate()      { [ "$WHICH" = all ] || [ "$WHICH" = rate ]; }
run_flat()      { [ "$WHICH" = all ] || [ "$WHICH" = flat ]; }

# Structural checks first: seconds, no Gurobi, and they cover the whole lower half (identity
# resolution, phase-3, stitch epoch layout, ncclize pacing gates). Failing here means not burning
# solver time on a broken lower half. These run against the HETERO fixtures -- deliberately, they
# are the irregular cases -- so this is a regression gate on shared machinery.
echo "=== [1/4] Gurobi-free structural tests (shared machinery, hetero fixtures) ==="
srun python -m teccl.examples.hierarchy_level_chunk_test
srun python -m teccl.examples.hierarchy_identity_resolution_test
srun python -m teccl.examples.hierarchy_crossbar_solve_test
srun python -m teccl.examples.hierarchy_ring_solve_test
srun python teccl/ncclize/pacing_gates_test.py

if run_multipath; then
    echo "=== [2/4] MULTIPATH config (H=50): hierarchical solve -> flat schedule ==="
    srun python -m teccl solve -i "${SAMPLES}/two_pod_rail_hierarchical_allgather.json"

    echo "--- ncclize round trip -> MSCCL XML (runs check_implements on all 16 GPUs) ---"
    # The per-op `rate` comes from the schedule itself (the level that solved each flow supplied
    # it), so `--no-rate` is NOT wanted here: stripping it would discard the coarse level's
    # pacing, which is half of what this topology exists to test.
    srun python teccl/ncclize/teccl_ncclize.py \
        --schedule "Schedules/two_pod_rail_allgather_flat.json" \
        --hierarchical \
        --topology TwoPodRail \
        -o "xml/two_pod_rail_allgather.xml" \
        --epoch-debug-output "logs/two_pod_rail_allgather_epochs.txt"
fi

if run_rate; then
    echo "=== [3/4] RATE config (H=30, host-bound): hierarchical solve -> flat schedule ==="
    srun python -m teccl solve -i "${SAMPLES}/two_pod_rail_hostbound_allgather.json"

    echo "--- ncclize round trip -> MSCCL XML (the pacing-sensitive arm) ---"
    srun python teccl/ncclize/teccl_ncclize.py \
        --schedule "Schedules/two_pod_rail_hostbound_allgather_flat.json" \
        --hierarchical \
        --topology TwoPodRailHostBound \
        -o "xml/two_pod_rail_hostbound_allgather.xml" \
        --epoch-debug-output "logs/two_pod_rail_hostbound_allgather_epochs.txt"
fi

if run_flat; then
    # The oracle. No --hierarchical on the ncclize call: this schedule was produced by the flat
    # solver on the fine topology, so there is no per-level rate to carry.
    echo "=== [4/4] FLAT ground truth (same problem, no abstraction) ==="
    srun python -m teccl solve -i "${SAMPLES}/two_pod_rail_flat_allgather.json"

    echo "--- ncclize round trip -> MSCCL XML ---"
    srun python teccl/ncclize/teccl_ncclize.py \
        --schedule "Schedules/two_pod_rail_allgather_flatsolve.json" \
        --topology TwoPodRail \
        -o "xml/two_pod_rail_allgather_flatsolve.xml"
fi

if [ "$WHICH" = all ]; then
    echo "=== COMPARE: hierarchical vs flat ground truth ==="
    # Finish time is the headline; print both rather than asserting, so a mismatch is read in
    # context (an abstraction gap and a bug look identical to a threshold, but not to a human).
    srun python - <<'PY'
import json, pathlib
for label, path in [("hierarchical (H=50)", "Schedules/two_pod_rail_allgather_flat.json"),
                    ("flat oracle  (H=50)", "Schedules/two_pod_rail_allgather_flatsolve.json"),
                    ("hierarchical (H=30)", "Schedules/two_pod_rail_hostbound_allgather_flat.json")]:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"{label}: MISSING ({path})")
        continue
    info = json.loads(p.read_text())
    keys = [k for k in info if "time" in k.lower() or "epoch" in k.lower()]
    print(f"{label}: " + ", ".join(f"{k}={info[k]}" for k in sorted(keys)))
print("\nexpected: H=50 -> 0.4267 s, H=30 -> 0.4667 s (chunk_size 1 GB);"
      " hierarchical >= flat, and the gap IS the abstraction cost.")
PY
fi
