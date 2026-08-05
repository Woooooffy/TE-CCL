#!/bin/bash
# Launch the full hierarchical pipeline for the HeteroTaperedCluster example on the remote
# (lanka/SLURM): Gurobi-free structural tests -> coarse LP solve (Gurobi) -> identity resolution ->
# phase-3 intra-cell schedule -> cross-stage invariant replay.
#
# Phase 3 narrates every step to the .out log (fan-out density decisions, delivery dedup, per-round
# NVSwitch matchings, optimality vs the max-port-load bound). The driver forces that debug on, so no
# env var is required. Outputs: Schedules/coarse_hetero_<coll>_{lp,identities,intra}.json and the
# log in logs/teccl-<jobid>.out|.err.
#
# What to look for in the .out log:
#   "chunk scale: ChunkScale(...)  (subdivision Q=...)"  -- the sub-chunk refinement that makes
#       every downstream volume a whole chunk; Q comes from the coarse relaxation, so it is data.
#   "fine-link occupancy: egress peak N%, ingress peak N% ... violations: 0"  -- the ingress half
#       was silently running at 150% before the landing GPU became capacity-aware.
#   "ingress landing: X/Y ... landed directly"      -- the ingress tier of the joint objective.
#   "egress: X/Y pieces leave via a non-native gateway"  -- tier 1; must not get worse when the
#       ingress tier is added, which is what the lexicographic weighting guarantees.
#   "intra fits coarse epoch: ... peak R rounds ... % of the budget"  -- the phase-3 certificate.
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

# `python -m` from SLURM_SUBMIT_DIR puts the repo tree ahead of site-packages, so the run uses the
# checked-out code either way -- but say so loudly rather than letting a failed build look like a
# successful one (a wheel build has failed here before on an NFS "Stale file handle").
if ! pip install .; then
    echo "WARNING: pip install failed. Continuing: 'python -m' from the repo root shadows the"
    echo "         installed package, so the run still uses this checkout -- but the installed"
    echo "         teccl is now STALE for anything that imports it differently."
fi

# Collective: first positional arg, default allgather. The coarse solve uses the LP arm (the MILP
# is intractable / gives no allgather makespan benefit -- see the design notes).
COLL="${1:-allgather}"

# Gurobi-free structural tests first: they take seconds and cover identity resolution (the joint
# lexicographic assignment + capacity-aware ingress + sub-chunk refinement) and the phase-3
# scheduler. Failing here means not burning solver time on a broken lower half.
echo "=== [1/3] Gurobi-free structural tests ==="
srun python -m teccl.examples.hierarchy_identity_resolution_test
srun python -m teccl.examples.hierarchy_intra_solve_test

echo "=== [2/3] hetero coarse solve + identity resolution + phase-3: collective=${COLL}, LP arm ==="
srun python -m teccl.examples.hierarchy_coarse_solve_hetero "${COLL}" lp

# End-to-end replay of the freshly written coarse LP through both downstream stages, asserting the
# cross-stage invariants: whole sub-chunk volumes, delivery coverage for every (identity, gpu),
# fine-link capacity on BOTH ends per coarse epoch, and peak intra rounds within one coarse epoch.
echo "=== [3/3] pipeline replay invariants ==="
srun python -m teccl.examples.hierarchy_pipeline_replay_test
