#!/bin/bash
# Launch the full hierarchical pipeline for the HeteroTaperedCluster example on the remote
# (lanka/SLURM): coarse LP solve (Gurobi) -> identity resolution -> phase-3 intra-cell schedule.
#
# Phase 3 narrates every step to the .out log (fan-out density decisions, delivery dedup, per-round
# NVSwitch matchings, optimality vs the max-port-load bound). The driver forces that debug on, so no
# env var is required. Outputs: Schedules/coarse_hetero_<coll>_{lp,identities,intra}.json and the
# log in logs/teccl-<jobid>.out|.err.
#
# Usage (from repo root):
#   sbatch scripts/run_hetero_phase3.sh              # allgather (default)
#   sbatch scripts/run_hetero_phase3.sh alltoall     # alltoall
#
# Run locally instead (needs a Gurobi license on this box):
#   python -m teccl.examples.hierarchy_coarse_solve_hetero allgather lp
#SBATCH --job-name=teccl-solve
#SBATCH --output=logs/teccl-%j.out
#SBATCH --error=logs/teccl-%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=100G
#SBATCH --account=commit
#SBATCH --qos=commit-main
#SBATCH --partition=lanka-v3
#SBATCH --exclusive

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs Schedules

source /data/commit/graphit/wangyj05/workspace/setup.sh
pip install .

# Collective: first positional arg, default allgather. The coarse solve uses the LP arm (the MILP
# is intractable / gives no allgather makespan benefit -- see the design notes).
COLL="${1:-allgather}"
echo "=== running hetero phase-3 pipeline: collective=${COLL}, LP arm, intra-debug on ==="
srun python -m teccl.examples.hierarchy_coarse_solve_hetero "${COLL}" lp
