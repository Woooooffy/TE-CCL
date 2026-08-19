#!/bin/bash
# A/B the two GPU->leaf PLACEMENTS on the 96-GPU dual-plane 64-port cluster: CLUSTERED (each host
# wholly on one leaf) vs SCATTERED (GPU g -> leaf g % 3). Hierarchical solve for each, plus the
# flat ground-truth oracle for each, then a comparison.
#
# WHY THIS IS THE ONLY VARIABLE. DualPlaneHeteroClusterScattered subclasses DualPlaneHeteroCluster
# and overrides exactly one method, _gpu_leaf. Capacities, alphas, indexing, spine mesh, cell
# structure and the port budget (32-down/32-up leaves, 48/64 spines, 10 switches) are shared code
# and verified identical. So a solve difference is attributable to placement and nothing else.
#
# WHAT TO EXPECT -- read this before reading the output, because the headline number is a TIE and
# that is the correct result, not a broken run:
#
#   T* SHOULD BE EQUAL (0.880 s at chunk_size 1 GB, 44 epochs). The fabric is non-blocking, so the
#       binding cut is the HOST uplink -- 8-GPU hosts must move 704 GB across their boundary at
#       800 GB/s -- and that cut is identical under both placements. The spine cut, which is the
#       only thing placement moves, sits slack at 0.640 s. A non-blocking fabric cannot express a
#       placement preference in makespan. If the two makespans DIFFER, either an uplink got
#       tapered or something is wrong.
#   SWITCH HOPS ARE WHERE THEY DIVERGE, which is why every input uses objective_type 5
#       (TOTAL_DEMAND_MIN_SWITCH_HOPS): the tiny per-hop penalty is a pure tie-breaker among
#       equal-makespan solutions. Scattered places 33.3% of cross-host GPU pairs on a shared leaf
#       vs clustered's 22.6% (clustered spends its leaf locality on same-host pairs that use
#       NVLink and never touch the leaf), so SCATTERED SHOULD WIN on leaf->spine volume.
#   COARSE GRAPH SHAPE differs even though the fine hardware does not: clustered gives each host
#       2 fat uplinks (800 GB/s), scattered gives it 6 thin ones (250-300 GB/s). Watch whether the
#       extra routing freedom helps the coarse LP or just makes it degenerate.
#   TWIN GROUPS should be the 2 intra-plane spine pairs and NOT a cross-plane group. A cross-plane
#       twin group means a cross-plane link got introduced somewhere and the planes are no longer
#       separate failure domains.
#
# EPOCHS ARE PINNED (epoch_duration 0.02 s, epoch_type USER_INPUT) in all four inputs. Derived
# SLOWEST_LINK sizing would hand the two placements different epochs, because their coarse link
# bandwidths differ, and the makespans would then differ by quantization rather than by placement.
#
# COST WARNING. The hierarchical runs are cheap (18-node coarse LP). The FLAT runs are not: ~96
# sources x 600 directed links x 64 epochs is ~3.7M flow variables on a 114-node graph. TwoPodRail
# was sized so its flat oracle was tractable; this topology is a real cluster and is not. Expect
# hours, expect possible OOM (the known dense-flow-cube limit), and treat the flat arm as optional
# validation rather than the deliverable.
#
# Usage (from repo root):
#   sbatch scripts/run_dual_plane.sh              # both placements, hierarchical + flat
#   sbatch scripts/run_dual_plane.sh hier         # both placements, hierarchical only (cheap)
#   sbatch scripts/run_dual_plane.sh flat         # both placements, flat oracle only (expensive)
#SBATCH --job-name=teccl-dualplane
#SBATCH --output=logs/teccl-%j.out
#SBATCH --error=logs/teccl-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH --account=commit
#SBATCH --qos=commit-main
#SBATCH --partition=lanka-v3

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs Schedules xml Logs logs/gurobi

source /data/commit/graphit/wangyj05/workspace/setup.sh

if ! pip install .; then
    echo "WARNING: pip install failed. Continuing: 'python -m' from the repo root shadows the"
    echo "         installed package, so the run still uses this checkout."
fi

WHICH="${1:-all}"
SAMPLES=teccl/examples/sample_inputs

run_hier() { [ "$WHICH" = all ] || [ "$WHICH" = hier ]; }
run_flat() { [ "$WHICH" = all ] || [ "$WHICH" = flat ]; }

# See run_two_pod_rail.sh for the full Logs/ vs logs/ naming trap: GurobiParams.log_file is a
# SUFFIX on the auto-generated name, not a path, and the hierarchical path writes one log per
# (epoch_duration, level).
collect_gurobi_logs() {
    local tag="$1"
    mkdir -p logs/gurobi
    cp -f Logs/*"${tag}"*.log logs/gurobi/ 2>/dev/null
    echo "--- Gurobi algorithm summary (${tag}) ---"
    for f in Logs/*"${tag}"*.log; do
        [ -f "$f" ] || { echo "  (no Logs/*${tag}*.log)"; break; }
        echo "== $(basename "$f")"
        grep -inE "Solved with|Barrier|Crossover|Optimal objective|Model fingerprint" "$f" | head -10
    done
}

# Structural checks first: seconds, no Gurobi. Failing here means not burning solver time.
echo "=== [1/4] Gurobi-free structural checks ==="
srun python -m teccl.examples.hierarchy_identity_resolution_test
srun python -m teccl.examples.hierarchy_crossbar_solve_test
srun python - <<'PY'
from teccl.input_data import TopologyParams
from teccl.hierarchy.abstract import abstract
from teccl.topologies.dual_plane_hetero_cluster import (
    DualPlaneHeteroCluster as C, DualPlaneHeteroClusterScattered as S)
PORT = 50.0
for nm, cls in (("clustered", C), ("scattered", S)):
    t = cls(TopologyParams(name=nm, chunk_size=1)); n = len(t.capacity)
    for p in range(2):
        for l in range(3):
            L = t._leaf(p, l)
            down = sum(t.capacity[L][g]/PORT for g in range(96) if t.capacity[L][g] > 0)
            up = sum(t.capacity[L][t._spine(p, x)]/PORT for x in range(2))
            assert down == 32 and up == 32, (nm, p, l, down, up)
    p0 = {t._leaf(0,l) for l in range(3)} | {t._spine(0,s) for s in range(2)}
    p1 = {t._leaf(1,l) for l in range(3)} | {t._spine(1,s) for s in range(2)}
    assert not any(t.capacity[i][j] > 0 for i in p0 for j in p1), "CROSS-PLANE LINK"
    c, _ = abstract(t)
    print(f"{nm}: 114 fine -> {len(c.capacity)} coarse, 32/32 leaves, planes disjoint, "
          f"twins={c.equivalent_node_indices}")
PY

if run_hier; then
    echo "=== [2/4] HIERARCHICAL: clustered placement ==="
    srun python -m teccl solve -i "${SAMPLES}/dual_plane_clustered_hierarchical_allgather.json"
    collect_gurobi_logs "_dp_clustered_hier"
    srun python teccl/ncclize/teccl_ncclize.py \
        --schedule "Schedules/dual_plane_clustered_allgather_flat.json" \
        --hierarchical --topology DualPlaneHeteroCluster \
        -o "xml/dual_plane_clustered_allgather.xml" \
        --epoch-debug-output "logs/dual_plane_clustered_epochs.txt"

    echo "=== [3/4] HIERARCHICAL: scattered placement ==="
    srun python -m teccl solve -i "${SAMPLES}/dual_plane_scattered_hierarchical_allgather.json"
    collect_gurobi_logs "_dp_scattered_hier"
    srun python teccl/ncclize/teccl_ncclize.py \
        --schedule "Schedules/dual_plane_scattered_allgather_flat.json" \
        --hierarchical --topology DualPlaneHeteroClusterScattered \
        -o "xml/dual_plane_scattered_allgather.xml" \
        --epoch-debug-output "logs/dual_plane_scattered_epochs.txt"
fi

if run_flat; then
    echo "=== [4/4] FLAT ground truth (expensive; see the cost warning at the top) ==="
    srun python -m teccl solve -i "${SAMPLES}/dual_plane_clustered_flat_allgather.json"
    srun python -m teccl solve -i "${SAMPLES}/dual_plane_scattered_flat_allgather.json"
fi

echo "=== COMPARE ==="
# Print, do not assert. An abstraction gap and a bug look identical to a threshold but not to a
# human, and here the EXPECTED result is a tie -- which an assert would happily hide.
srun python - <<'PY'
import json, pathlib
rows = [("clustered  hierarchical", "Schedules/dual_plane_clustered_allgather_flat.json"),
        ("scattered  hierarchical", "Schedules/dual_plane_scattered_allgather_flat.json"),
        ("clustered  flat oracle ", "Schedules/dual_plane_clustered_allgather_flatsolve.json"),
        ("scattered  flat oracle ", "Schedules/dual_plane_scattered_allgather_flatsolve.json")]
for label, path in rows:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"{label}: MISSING ({path})"); continue
    info = json.loads(p.read_text())
    keys = [k for k in info if "time" in k.lower() or "epoch" in k.lower()]
    print(f"{label}: " + ", ".join(f"{k}={info[k]}" for k in sorted(keys)))
print("\nexpected: all four T* = 0.880 s (44 epochs @ 0.02 s, chunk_size 1 GB).")
print("A TIE on latency is the CORRECT result -- the fabric is non-blocking, so the binding cut")
print("is the host uplink, which placement does not move. Look for the difference in leaf->spine")
print("volume instead (objective_type 5's switch-hop term); scattered should carry less.")
PY
