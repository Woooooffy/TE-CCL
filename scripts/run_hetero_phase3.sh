#!/bin/bash
# Launch the full hierarchical pipeline for the HeteroTaperedCluster example on the remote
# (lanka/SLURM): coarse LP solve (Gurobi) -> identity resolution -> phase-3 intra-cell schedule.
#
# Phase 3 runs with TECCL_INTRA_DEBUG so the .out log narrates every step (fan-out density
# decisions, delivery dedup, per-round NVSwitch matchings, and optimality vs the max-port-load
# bound). Outputs land in Schedules/coarse_hetero_<coll>_{lp,identities,intra}.json and the log
# in logs/teccl-<jobid>.out|.err (git-synced back as usual).
#
# Usage (from repo root):
#   sbatch scripts/run_hetero_phase3.sh                # allgather (default)
#   sbatch scripts/run_hetero_phase3.sh alltoall       # alltoall
#   COLL=alltoall sbatch scripts/run_hetero_phase3.sh  # same via env
#
# Run locally instead (needs a Gurobi license on this box):
#   TECCL_INTRA_DEBUG=1 python -m teccl.examples.hierarchy_coarse_solve_hetero allgather lp
#
# NOTE: adjust the #SBATCH partition/time/account and CONDA_ROOT below to your cluster if needed.
#SBATCH --job-name=teccl
#SBATCH --output=logs/teccl-%j.out
#SBATCH --error=logs/teccl-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# #SBATCH --partition=<your-partition>     # uncomment + set if your cluster requires it

set -euo pipefail

# --- environment ------------------------------------------------------------
CONDA_ROOT="${CONDA_ROOT:-/data/commit/graphit/wangyj05/miniconda3}"
CONDA_ENV="${CONDA_ENV:-teccl}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# --- run --------------------------------------------------------------------
# Collective: first positional arg, or $COLL, default allgather. The coarse solve uses the LP arm
# (the MILP is intractable / gives no allgather makespan benefit -- see the design notes).
COLL="${1:-${COLL:-allgather}}"

mkdir -p logs Schedules
pip install ./. >/dev/null

echo "=== running hetero phase-3 pipeline: collective=${COLL}, LP arm, intra-debug on ==="
TECCL_INTRA_DEBUG=1 python -m teccl.examples.hierarchy_coarse_solve_hetero "${COLL}" lp
