"""
Generate AllGather sample inputs on the RailOptimizedSpineLeaf (32-node / 256-GPU)
topology where only a SUBSET of GPUs participate in the collective, using the
`passive_node_indices` feature (present-but-no-demand relay nodes).

Motivation: MoE inference. For a given batch only the GPUs holding *activated*
experts take part in the expert exchange; the rest hold dormant experts and are
pure forwarders. WHERE the active experts sit on the rail-optimized fabric
decides which part of the network is stressed. We craft three placements, each
isolating one phenomenon you can then look for in the produced schedule.

GPU index convention (see rail_optimized_spine_leaf.py):
    GPU(node n, rail r) = n * 8 + r      n in [0,32), r in [0,8)   -> [0,256)
Switches (256-299) are never passive; passive = every GPU not in the active set.

Run:  python3 -m teccl.examples.moe_subsets.gen_moe_subset_inputs
"""
import json
import pathlib

NUM_NODES = 32
GPUS_PER_NODE = 8
NUM_GPUS = NUM_NODES * GPUS_PER_NODE  # 256


def gpu(node: int, rail: int) -> int:
    return node * GPUS_PER_NODE + rail


def passive_of(active):
    """All GPUs not in the active set become passive relays."""
    active = set(active)
    return sorted(set(range(NUM_GPUS)) - active)


# ---------------------------------------------------------------------------
# Three MoE-motivated active-expert placements.
# ---------------------------------------------------------------------------

# E1 - RAIL-ALIGNED  (incast on a single leaf, no spine needed)
#   32 experts, one per node, all pinned to rail 0. Every participant hangs off
#   leaf switch 0. AllGather among them is a pure incast/broadcast through ONE
#   leaf: the 50 GB/s leaf<->GPU ports are the only way in/out of each node.
#   Look for: leaf 0 (node 288) saturated; spines 296-299 UNUSED; completion
#   bound by 31 chunks funneling through 50 GB/s per-receiver ingress. A detour
#   via NVSwitch->other rail->spine cannot relieve the final leaf0->dest hop, so
#   this checks whether the solver recognizes an irreducible incast bottleneck.
E1_active = [gpu(n, 0) for n in range(NUM_NODES)]

# E2 - NODE-DENSE  (small cluster, maximal path choice -> multipath)
#   Both nodes 0 and 1 fully activated (16 GPUs). Small enough to actually solve.
#   Contains all three pair types: intra-node (NVSwitch), cross-node same-rail
#   (one leaf), cross-node cross-rail (spine OR intra-node rail-bridge).
#   Look for: for GPU(0,r1)->GPU(1,r2) with r1!=r2, does the solver take the
#   spine (leaf r1->spine->leaf r2) or bridge rails over NVSwitch(0) first? Does
#   it SPLIT cross-rail flow across the 4 spines (multipath)? This is the best
#   example to inspect for genuine multipath and NVSwitch rail-bridging.
E2_active = [gpu(n, r) for n in (0, 1) for r in range(GPUS_PER_NODE)]

# E3 - HOT-NODE ASYMMETRY  (one dense node + a scatter of single experts)
#   Node 0 fully active (spans all 8 leaves) PLUS the rail-0 GPU of nodes 1..15.
#   The 15 remote participants live ONLY on leaf 0, while node 0 touches every
#   leaf. So node 0's rail-1..7 chunks must fan out to the leaf-0 crowd via spine
#   (leaf r->spine->leaf 0), and the 15 remote chunks must reach node 0's rails
#   1..7 the same way. Leaf 0 is a hot fan-in point; leaves 1-7 each see one
#   participant. Load is deliberately ASYMMETRIC.
#   Look for: uneven leaf load (leaf 0 hot), spine fan-in, and whether the 4
#   spines are balanced (symmetry constraint) or one is favored.
E3_active = [gpu(0, r) for r in range(GPUS_PER_NODE)] + [gpu(n, 0) for n in range(1, 16)]


CONFIGS = {
    "rail_aligned_incast": E1_active,
    "node_dense_multipath": E2_active,
    "hot_node_asymmetry": E3_active,
}


def build_input(name: str, active):
    return {
        "TopologyParams": {
            "name": "RailOptimizedSpineLeaf",
            "chassis": 1,
            "chunk_size": 1,
            "passive_node_indices": passive_of(active),
        },
        "GurobiParams": {
            "time_limit": 2,
            "feasibility_tol": 1e-4,
            "intfeas_tol": 1e-4,
            "optimality_tol": 1e-4,
            "output_flag": 1,
            "log_file": "",
            "log_to_console": 1,
            "mip_gap": 1e-3,
            "mip_focus": 1,
            "crossover": -1,
            "method": -1,
            "heuristics": 0.05,
        },
        "InstanceParams": {
            "collective": 1,            # ALLGATHER
            "num_chunks": 1,
            "epoch_type": 1,            # FASTEST_LINK
            "epoch_duration": -1,
            "num_epochs": -1,
            "alpha_threshold": 0.1,
            "switch_copy": True,
            "switch_pipeline": True,
            "debug": False,
            "debug_output_file": "",
            "objective_type": 2,        # TOTAL_DEMAND
            "solution_method": 1,       # ONE_SHOT (switch to 2 if it times out)
            "symmetry": True,
            "schedule_output_file": f"teccl/examples/schedules/moe_{name}_schedule.json",
        },
    }


def main():
    out_dir = pathlib.Path(__file__).parent
    for name, active in CONFIGS.items():
        cfg = build_input(name, active)
        path = out_dir / f"moe_{name}_sample.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"{name:24s} active={len(active):3d} gpus  passive={len(cfg['TopologyParams']['passive_node_indices']):3d}  -> {path}")
        print(f"    active indices: {active}")


if __name__ == "__main__":
    main()
