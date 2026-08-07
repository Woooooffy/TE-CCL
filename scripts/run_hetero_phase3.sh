#!/bin/bash
# Launch the full hierarchical pipeline for the HeteroTaperedCluster example on the remote
# (lanka/SLURM): Gurobi-free structural tests -> coarse LP solve (Gurobi) -> identity resolution ->
# phase-3 intra-cell schedule -> phase-4 STITCH -> ncclize round trip to MSCCL XML.
#
# Phase 3 narrates every step to the .out log (fan-out density decisions, delivery dedup, per-round
# NVSwitch matchings, optimality vs the max-port-load bound). The driver forces that debug on, so no
# env var is required. Outputs: Schedules/coarse_hetero_<coll>_{lp,identities,intra,flat}.json,
# xml/hetero_<coll>.xml, and the log in logs/teccl-<jobid>.out|.err.
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
#   "=== phase-4 stitch ===" then:
#     "N delivery records {...}; M demands traced (causality + coverage verified)" -- the stitch's
#         back-trace is both the emitter of "8-Chunk paths" and the proof every demand is met.
#     "fine epoch delta=... x E epochs = T s"  -- T must be ~0.20 s (allgather) / ~0.80 s (alltoall).
#         REFINEMENT MUST NOT MOVE T: Q halves delta and doubles E, so their product is invariant.
#         If T jumps to ~0.22 s, delta and m have desynced from the ChunkScale.
#     "paced network sends: N at rate(s) [...]"  -- ONLY inter-cell sends carry a rate (each level
#         paces its own flows against its own epoch); intra-cell NVLink hops are deliberately
#         unpaced, so "intra sends unpaced" should equal the intra record count exactly.
#   "[5/5] ... check_implements" -- ncclize independently verifying the stitched flat schedule
#       really implements the collective. This is the strongest free correctness check in the run.
#       Expect ~50 "cannot be realized as-is" WARNINGS: those are the send-gating gap that the
#       deferred pacing follow-up closes, not a stitch failure. Record the count as the baseline.
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

# Collective: first positional arg, default allgather. The coarse solve uses the LP arm (the MILP
# is intractable / gives no allgather makespan benefit -- see the design notes).
COLL="${1:-allgather}"

# Gurobi-free structural tests first: they take seconds and cover identity resolution (the joint
# lexicographic assignment + capacity-aware ingress + sub-chunk refinement), the phase-3 scheduler,
# and the phase-4 stitch (epoch layout, precedence levels, chunk-label addressing). Failing here
# means not burning solver time on a broken lower half.
echo "=== [1/5] Gurobi-free structural tests ==="
srun python -m teccl.examples.hierarchy_identity_resolution_test
srun python -m teccl.examples.hierarchy_intra_solve_test
srun python teccl/ncclize/pacing_gates_test.py

echo "=== [2/5] hetero coarse solve -> identity resolution -> phase-3 -> phase-4 stitch: collective=${COLL}, LP arm ==="
srun python -m teccl.examples.hierarchy_coarse_solve_hetero "${COLL}" lp

# End-to-end replay of the freshly written coarse LP through both downstream stages, asserting the
# cross-stage invariants: whole sub-chunk volumes, delivery coverage for every (identity, gpu),
# fine-link capacity on BOTH ends per coarse epoch, and peak intra rounds within one coarse epoch.
echo "=== [3/5] pipeline replay invariants (resolution + phase-3) ==="
srun python -m teccl.examples.hierarchy_pipeline_replay_test

# Same replay carried one stage further, through the stitch, plus the unit oracles for the epoch
# layout. Re-runs against the schedule the coarse solve just wrote.
echo "=== [4/5] stitch replay + epoch-layout oracles ==="
srun python -m teccl.examples.hierarchy_stitch_test

# The flat schedule is an ordinary fine-topology schedule, so ncclize consumes it with no hierarchy
# awareness. Algorithm.make_implementation runs check_implements(), which independently validates
# that the schedule implements the collective -- the real point of this step. The per-op `rate`
# now comes from the schedule itself (the level that solved each flow supplied it), so `--no-rate`
# is NOT wanted here: stripping it would discard the coarse level's pacing.
echo "=== [5/5] ncclize round trip: flat schedule -> MSCCL XML (runs check_implements) ==="
srun python teccl/ncclize/teccl_ncclize.py \
    --schedule "Schedules/coarse_hetero_${COLL}_flat.json" \
    --hierarchical \
    -o "xml/hetero_${COLL}.xml" \
    --epoch-debug-output "logs/hetero_${COLL}_epochs.txt"
