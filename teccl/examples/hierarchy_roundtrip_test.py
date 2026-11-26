"""
Gurobi-free structural round-trip test for the hierarchical abstraction.

Verifies that abstract() collapses the 300-node rail-optimized spine-leaf topology into the
expected 44-node coarse topology (32 hosts + 8 leaves + 4 spines), that capacities/switch
roles/boundary port map are correct, and that lift_demand builds the right sub-chunk<->GPU
correspondence. Does NOT build or solve any Gurobi model.

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_roundtrip_test
"""
from teccl.hierarchy.abstract import abstract, lift_demand
from teccl.input_data import TopologyParams
from teccl.topologies.rail_optimized_spine_leaf import RailOptimizedSpineLeaf


def main() -> None:
    topo = RailOptimizedSpineLeaf(TopologyParams(name="RailOptimizedSpineLeaf", chunk_size=1))

    NUM_NODES, GPN, NUM_LEAF, NUM_SPINE = 32, 8, 8, 4
    n_fine = len(topo.capacity)
    assert n_fine == 300, n_fine
    assert len(topo.cells) == NUM_NODES, len(topo.cells)

    coarse, m = abstract(topo)

    # --- coarse size ---------------------------------------------------------
    expected_coarse = NUM_NODES + NUM_LEAF + NUM_SPINE  # 44
    assert m.num_coarse == expected_coarse, m.num_coarse
    assert len(coarse.capacity) == expected_coarse, len(coarse.capacity)

    # cells are coarse ids 0..31 (data-bearing); leaves/spines are switches.
    assert set(m.coarse_cells.keys()) == set(range(NUM_NODES))
    assert len(m.coarse_passthrough) == NUM_LEAF + NUM_SPINE
    # switch coarse ids == the passthrough leaves+spines (hosts are NOT switches).
    assert set(coarse.switch_indices) == set(m.coarse_passthrough.keys()), coarse.switch_indices
    assert len(coarse.switch_indices) == NUM_LEAF + NUM_SPINE

    # --- coarse capacities ---------------------------------------------------
    # host->leaf coarse link == one physical gpu(n,r)->leaf(r) edge = 400 Gbps = 50 GB/s.
    # leaf->spine coarse link == 8x400 Gbps = 400 GB/s. chunk_size=1 so caps are as-is.
    leaf_coarse = {m.fine_to_coarse[topo._leaf(r)] for r in range(NUM_LEAF)}
    spine_coarse = {m.fine_to_coarse[topo._spine(s)] for s in range(NUM_SPINE)}
    host0 = 0  # cell 0
    host0_leaf_links = [coarse.capacity[host0][lc] for lc in leaf_coarse if coarse.capacity[host0][lc] > 0]
    assert len(host0_leaf_links) == GPN, host0_leaf_links  # host reaches all 8 leaves
    assert all(abs(c - 50.0) < 1e-9 for c in host0_leaf_links), host0_leaf_links
    for lc in leaf_coarse:
        for sc in spine_coarse:
            assert abs(coarse.capacity[lc][sc] - 400.0) < 1e-9, (lc, sc, coarse.capacity[lc][sc])
    # no host<->host or host<->spine direct links (rail-optimized: only via leaves).
    for a in range(NUM_NODES):
        for b in range(NUM_NODES):
            assert coarse.capacity[a][b] == 0.0, (a, b)
        for sc in spine_coarse:
            assert coarse.capacity[a][sc] == 0.0, (a, sc)

    # --- boundary port map ---------------------------------------------------
    # coarse link (host n, leaf r) must be owned physically by fine gpu(n, r).
    for n in range(NUM_NODES):
        for r in range(NUM_LEAF):
            lc = m.fine_to_coarse[topo._leaf(r)]
            owners = m.boundary_gpu[(n, lc)]
            assert owners == [topo._gpu(n, r)], (n, r, owners)

    # --- symmetry & node_per_chassis ----------------------------------------
    # the 4 twin spines survive as an equivalent coarse group.
    assert any(sorted(g) == sorted(spine_coarse) for g in coarse.equivalent_node_indices), \
        coarse.equivalent_node_indices
    assert coarse.node_per_chassis == NUM_NODES

    # --- lift_demand ---------------------------------------------------------
    lift_demand(m, num_sub_chunks=GPN)
    for n in range(NUM_NODES):
        for c in range(GPN):
            assert m.chunk_origin[(n, c)] == topo._gpu(n, c), (n, c)

    print("hierarchy round-trip OK: "
          f"{n_fine} fine nodes -> {m.num_coarse} coarse nodes "
          f"({len(m.coarse_cells)} hosts + {len(coarse.switch_indices)} switches); "
          "capacities, boundary port map, symmetry, lift_demand all correct.")


if __name__ == "__main__":
    main()
