#!/bin/bash
# Launch the full hierarchical pipeline for the RailOptimizedSpineLeaf topology (32 hosts x 8 GPUs
# = 256 GPUs, 8 leaves + 4 spines, 300 fine nodes -> 44 coarse nodes) on the remote (lanka/SLURM):
# Gurobi-free structural tests -> coarse LP solve (Gurobi) -> identity resolution -> phase-3
# intra-cell schedule -> phase-4 STITCH -> ncclize round trip to MSCCL XML.
#
# This is the at-scale sibling of run_hetero_phase3.sh. The hetero cluster is the irregularity
# stress test (3 uneven cells, tapered uplinks, forced relay); the rail topology is the shape the
# solver actually targets, and it is where SIZE, not irregularity, is the risk. Read that script
# for the phase-by-phase narration; only what differs is repeated here.
#
# Outputs: Schedules/coarse_rail_<coll>_{lp,identities,intra,flat}.json, xml/rail_<coll>.xml, and
# the log in logs/teccl-<jobid>.out|.err.
#
# What to look for in the .out log, beyond the hetero checks:
#   "twin groups (symmetry): [[...8 leaves...], [...4 spines...]]"  -- symmetry is ON here. If the
#       leaf group is MISSING, abstract()'s emergent-twin detection did not fire and the coarse LP
#       is about to land on a degenerate fractional interior point; the identity resolution will
#       then look like a mess of tiny pieces. That is the first thing to check on a bad run.
#   "level chunk: g=8 fine chunks ... coarse epoch 0.16s"  -- THE MAIN THING THIS RUN TESTS. The
#       coarse level is now solved in its own chunk (abstract.set_level_chunk: the GCD of its
#       demand volumes), so a host's whole payload for one peer is ONE chunk and one epoch of its
#       egress is EIGHT. Job 1336765 ran at g=1 / 0.02s, where a pair's demand was exactly one
#       epoch of egress and the LP was free to smear it across destinations: it was optimal (31
#       epochs, 100% uplink utilisation) but 2347 distinct flow values. Expect g=8 for allgather
#       and g=64 for alltoall. `sbatch scripts/run_rail_phase3.sh allgather nocoarsen` reproduces
#       the old behaviour for an A/B.
#   "coarse demand: N nonzero (U->V) pairs, distinct volumes [1]"  -- the rail cells are uniform,
#       so there must be exactly ONE distinct volume, and after coarsening it must be 1: one coarse
#       chunk per ordered pair. More than one value means coarsify_demand or the cell declaration
#       is wrong; a value != 1 means the GCD did not come out to a whole cell payload.
#   RAW LP FLOW VARS -- the payoff. At g=1 the leaf->host values were an unstructured smear
#       (0.306379 / 0.454924 / ...) because a source served 2-5 destinations per epoch. If the
#       degeneracy diagnosis is right these should now be dominated by whole-chunk deliveries
#       (one destination per epoch, 0.125 on each of the 8 rails). If they are STILL smeared, the
#       tie is not coming from the demand/epoch ratio and the objective-regularization route is
#       the remaining lever.
#   "resolved inter-cell pieces: N"  -- expect this to be LARGE (32x31 host pairs x 8 identities,
#       times the sub-chunk refinement Q). Wall time below the coarse solve is combinatorial, not
#       solver-bound, so a slow run here points at Q having blown up.
#   "intra fits coarse epoch: ... % of the budget"  -- the phase-3 certificate. The rail NVSwitch
#       is 1800 GB/s against a 50 GB/s uplink, so this should sit at a small percentage; a number
#       near 100% means the intra fabric is no longer comfortably faster than the outer one.
#       m grows with the level chunk (36 per coarse epoch at g=1, 288 at g=8), and so does the
#       intra work per band, so the RATIO is what to watch, not the absolute round count.
#   "fine epoch delta=... x E epochs = T s"  -- REFINEMENT MUST NOT MOVE T (Q halves delta and
#       doubles E). Their product is the scale-invariance check.
#
# KNOWN FAILURE MODE: the coarse rail AllGather LP solves, but flow extraction can trip
# `assert consume == 0` in lp_formulation.account_for_consume on Gurobi feasibility-tolerance
# noise (get_flows_and_consumes filters on `v.x != 0.0` instead of clamping at
# gurobi.feasibility_tol -- lp_formulation.py:601). If step [2/5] dies there, that is the known
# extraction bug, not a hierarchy failure.
#
# Usage (from repo root):
#   sbatch scripts/run_rail_phase3.sh                        # allgather (default)
#   sbatch scripts/run_rail_phase3.sh alltoall               # alltoall (heavier: 256x256 fine demand)
#   sbatch scripts/run_rail_phase3.sh allgather nocoarsen    # A/B: pre-coarsening behaviour (g=1)
#
# Run locally instead (needs a Gurobi license on this box):
#   python -m teccl.examples.hierarchy_coarse_solve_rail allgather lp
# Extra args: `debug` turns phase-3's per-cell narration back on (off by default here: 32 identical
# cells would each print a full derivation and bury the log); `nocoarsen` pins the level chunk to 1.
#SBATCH --job-name=teccl-rail
#SBATCH --output=logs/teccl-%j.out
#SBATCH --error=logs/teccl-%j.err
#SBATCH --time=12:00:00
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

COLL="${1:-allgather}"
# Any extra args (e.g. `nocoarsen`, `debug`) pass straight through to the driver.
shift || true
EXTRA=("$@")

# Gurobi-free structural tests first: seconds, and they cover the whole lower half (identity
# resolution, phase-3 scheduler, stitch epoch layout, ncclize pacing gates). They run against the
# HETERO fixtures -- that is deliberate, those are the irregular cases -- so this step is a
# regression gate on shared machinery, not a rail-specific check. Failing here means not burning
# solver time on a broken lower half.
echo "=== [1/5] Gurobi-free structural tests (shared machinery, hetero fixtures) ==="
srun python -m teccl.examples.hierarchy_level_chunk_test
srun python -m teccl.examples.hierarchy_identity_resolution_test
srun python -m teccl.examples.hierarchy_intra_solve_test
srun python teccl/ncclize/pacing_gates_test.py

echo "=== [2/5] rail coarse solve -> identity resolution -> phase-3 -> phase-4 stitch: collective=${COLL}, LP arm ==="
srun python -m teccl.examples.hierarchy_coarse_solve_rail "${COLL}" lp "${EXTRA[@]}"

# The replay tests are hetero-fixture-bound (they rebuild HeteroTaperedCluster and read
# Schedules/coarse_hetero_*_lp.json), so they re-verify the cross-stage invariants on the small
# case only. They are cheap and catch seam regressions; the rail-specific invariants are asserted
# inline by step [2/5] itself (whole sub-chunk volumes, fine-link capacity per coarse epoch, the
# phase-3 rounds-fit-epoch certificate, and the stitch's causality + coverage back-trace).
echo "=== [3/5] pipeline replay invariants (resolution + phase-3, hetero fixture) ==="
srun python -m teccl.examples.hierarchy_pipeline_replay_test

echo "=== [4/5] stitch replay + epoch-layout oracles (hetero fixture) ==="
srun python -m teccl.examples.hierarchy_stitch_test

# The flat schedule is an ordinary fine-topology schedule, so ncclize consumes it with no hierarchy
# awareness. Algorithm.make_implementation runs check_implements(), which independently validates
# that the schedule implements the collective on all 256 GPUs -- the real point of this step. The
# per-op `rate` comes from the schedule itself (the level that solved each flow supplied it), so
# `--no-rate` is NOT wanted here: stripping it would discard the coarse level's pacing.
echo "=== [5/5] ncclize round trip: flat schedule -> MSCCL XML (runs check_implements) ==="
srun python teccl/ncclize/teccl_ncclize.py \
    --schedule "Schedules/coarse_rail_${COLL}_flat.json" \
    --hierarchical \
    --topology RailOptimizedSpineLeaf \
    -o "xml/rail_${COLL}.xml" \
    --epoch-debug-output "logs/rail_${COLL}_epochs.txt"
