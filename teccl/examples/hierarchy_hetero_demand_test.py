"""
Gurobi-free structural test for HETEROGENEOUS demand coarsification on HeteroTaperedCluster.

Exercises the phase-3 groundwork that the symmetric rail topology never triggers: cells with
DIFFERENT GPU counts (A:4, B:4, C:6) whose coarse AllGather is non-uniform, plus the
collective-agnostic coarsify_demand aggregation shared by AllGather and AllToAll. Does NOT
build or solve any Gurobi model.

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_hetero_demand_test
"""
from teccl.hierarchy.abstract import abstract, coarsify_demand, lift_demand
from teccl.input_data import Collective, TopologyParams
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster


def _vol(coarse):
    """coarse[U][V][0] -> a plain U->V volume dict (only the non-zero entries)."""
    return {(u, v): coarse[u][v][0]
            for u in range(len(coarse)) for v in range(len(coarse)) if coarse[u][v][0]}


def main() -> None:
    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    assert len(topo.capacity) == 19, len(topo.capacity)
    coarse, m = abstract(topo)

    # Cells A,B,C -> coarse 0,1,2 (data-bearing); T0,T1 -> coarse 3,4 (switch passthrough).
    A, B, C, T0, T1 = 0, 1, 2, 3, 4
    assert m.num_coarse == 5, m.num_coarse
    assert set(m.coarse_cells) == {A, B, C}
    assert set(coarse.switch_indices) == {T0, T1}
    gpu_count = {A: 4, B: 4, C: 6}
    for cid, n in gpu_count.items():
        assert len(m.coarse_cells[cid].gpus) == n, (cid, len(m.coarse_cells[cid].gpus))

    # --- AllGather coarsification (num_chunks=1) -----------------------------
    # Each identity (source gpu, chunk 0) is wanted by every participating GPU, so it collapses
    # to ONE crossing per destination CELL: coarse[U][V] = |U| (the cell's GPU count). This is
    # the non-uniform 4/4/6 volume the uniform generator could not express. Switches are not
    # destinations, so no volume flows into T0/T1.
    ag = coarsify_demand(build_demand(Collective.ALLGATHER, topo, num_chunks=1), m)
    expected_ag = {}
    for u in (A, B, C):
        for v in (A, B, C):
            if u != v:
                expected_ag[(u, v)] = gpu_count[u]
    assert _vol(ag) == expected_ag, (_vol(ag), expected_ag)

    # num_chunks scales volume linearly: 2 chunks/GPU -> 2*|U|.
    ag2 = coarsify_demand(build_demand(Collective.ALLGATHER, topo, num_chunks=2), m)
    assert _vol(ag2) == {k: 2 * v for k, v in expected_ag.items()}, _vol(ag2)

    # --- AllToAll coarsification (1 chunk per ordered GPU pair) ---------------
    # Identities are per-destination-GPU distinct (no fan-out sharing), so no collapse happens
    # and coarse[U][V] = |U|*|V| -- the full pairwise volume. Same code path, different result,
    # driven entirely by the fine demand's identity structure.
    num_gpus = sum(gpu_count.values())  # 14 participating GPUs; 1 chunk per pair
    a2a = coarsify_demand(build_demand(Collective.ALLTOALL, topo, num_chunks=num_gpus), m)
    expected_a2a = {}
    for u in (A, B, C):
        for v in (A, B, C):
            if u != v:
                expected_a2a[(u, v)] = gpu_count[u] * gpu_count[v]
    assert _vol(a2a) == expected_a2a, (_vol(a2a), expected_a2a)

    # --- lift_demand (heterogeneous, per-cell chunk counts) ------------------
    # Default (num_sub_chunks=None): every cell contributes exactly its own GPU count of
    # sub-chunk identities, mapped to its GPUs in order. This is the identity map phase-3
    # AllGather resolution splits each cell's aggregate coarse commodity back onto.
    lift_demand(m)
    for cid in (A, B, C):
        cell = m.coarse_cells[cid]
        for c in range(len(cell.gpus)):
            assert m.chunk_origin[(cid, c)] == cell.gpus[c], (cid, c)
        # no extra identities beyond the cell's GPU count
        assert (cid, len(cell.gpus)) not in m.chunk_origin, cid
    assert len(m.chunk_origin) == sum(gpu_count.values()), len(m.chunk_origin)

    print("hetero demand coarsification OK: "
          f"AllGather coarse volumes {expected_ag} (non-uniform 4/4/6), "
          f"AllToAll coarse volumes {expected_a2a} (|U|*|V|), "
          f"lift_demand identity map has {len(m.chunk_origin)} per-cell chunks.")


if __name__ == "__main__":
    main()
