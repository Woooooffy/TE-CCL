"""
Gurobi-free structural test for LP hierarchical IDENTITY RESOLUTION (teccl.hierarchy.reconstruct).

Identity resolution is driven entirely by the demand shape, so it needs no solver: we feed it a
HAND-BUILT coarse flow decomposition (the same per_chunk_flow_paths structure the solved
LPFormulation exposes) plus a fine demand array, and check the assignment + emitted intra-cell
demands against hand-derived oracles. Covers:

  1. identity_sets() matches coarsify_demand's counts for BOTH AllGather and AllToAll (the
     collective-general derivation), on the real HeteroTaperedCluster abstraction.
  2. Forced-relay oracle: single-homed Host B (one gateway g4, GPUs 4..7) must relay the 3
     non-gateway identities to g4 -- exactly 3 egress_stage demands, coalesced across
     destinations, and none for the native identity.
  3. Zero-relay symmetric case: every GPU is a gateway (uplinks == GPUs) -> no egress_stage.
  4. AllToAll ingress target is a SINGLE GPU (vs all-of-cell for AllGather).

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_identity_resolution_test
"""
from types import SimpleNamespace

import numpy as np

from teccl.hierarchy.abstract import abstract, coarsify_demand
from teccl.hierarchy.cell import Cell
from teccl.hierarchy.reconstruct import HierarchyMapping, identity_sets, resolve_identities
from teccl.input_data import Collective, TopologyParams
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster


def _fake_solver(per_chunk_flow_paths, switch_indices):
    """Minimal stand-in for a solved LPFormulation: resolve_identities only reads
    .per_chunk_flow_paths and .topology.switch_indices."""
    return SimpleNamespace(
        per_chunk_flow_paths=per_chunk_flow_paths,
        topology=SimpleNamespace(switch_indices=switch_indices),
    )


def _single_switch_path(cell_s, cell_d, switch, vol, epoch):
    """A U -> switch -> V coarse path as the two 6-tuples (s, i, j, c, vol, k)."""
    return [(cell_s, cell_s, switch, 0, vol, epoch),
            (cell_s, switch, cell_d, 0, vol, epoch)]


def _two_switch_path(cell_s, cell_d, sw1, sw2, vol, epoch):
    """A U -> sw1 -> sw2 -> V coarse path (single-homed relay across the top mesh)."""
    return [(cell_s, cell_s, sw1, 0, vol, epoch),
            (cell_s, sw1, sw2, 0, vol, epoch),
            (cell_s, sw2, cell_d, 0, vol, epoch)]


# ------------------------------------------------------------------------------------------
def test_identity_sets_match_coarsify():
    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    _, m = abstract(topo)
    for coll, nchunks in ((Collective.ALLGATHER, 1), (Collective.ALLTOALL, 14)):
        fine_demand = build_demand(coll, topo, num_chunks=nchunks)
        coarse = coarsify_demand(fine_demand, m)
        id_sets, _ = identity_sets(fine_demand, m)
        for u in range(m.num_coarse):
            for v in range(m.num_coarse):
                expect = coarse[u][v][0]
                got = len(id_sets.get((u, v), []))
                assert got == expect, (coll, (u, v), got, expect)
    print("  [1] identity_sets counts == coarsify_demand (AllGather + AllToAll) OK")


# ------------------------------------------------------------------------------------------
def test_hostB_forced_relay():
    """Host B single-homed (gateway g4). B sources to A and C; identities native to g5/g6/g7
    must relay to g4."""
    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    _, m = abstract(topo)
    A, B, C, T0, T1 = 0, 1, 2, 3, 4
    B_gpus = [4, 5, 6, 7]
    A_gpus, C_gpus = [0, 1, 2, 3], [8, 9, 10, 11, 12, 13]

    # fine demand: only B sources (to A, C, and internally) -> id_sets has just (B,A),(B,C).
    n = 19
    fine_demand = np.zeros((n, n, 1), dtype=np.int32)
    for s in B_gpus:
        for t in A_gpus + C_gpus + [g for g in B_gpus if g != s]:
            fine_demand[s][t][0] = 1

    # coarse pieces: B->A via T0 (4 units); B->C via T0->T1 (4 units). All egress on g4's link.
    pcp = {(B, A, 0): [_single_switch_path(B, A, T0, 1.0, k) for k in range(4)],
           (B, C, 0): [_two_switch_path(B, C, T0, T1, 1.0, k) for k in range(4)]}
    solver = _fake_solver(pcp, switch_indices=[T0, T1])
    res = resolve_identities(solver, m, fine_demand, topo)

    # egress_stage: exactly the 3 non-gateway identities, native gpu -> g4, coalesced (one each
    # even though each feeds both A and C).
    egress = [d for d in res.intra_demands if d.kind == "egress_stage"]
    got = sorted((d.identity, d.src_gpu, d.dst_gpus) for d in egress)
    want = sorted(((s, 0), s, (4,)) for s in (5, 6, 7))
    assert got == want, (got, want)
    assert all(d.cell == B for d in egress)
    assert (4, 0) not in {d.identity for d in egress}  # native identity: no relay

    # every (U,V) delivers each identity exactly once (sum of resolved piece volume per identity)
    for (U, V) in ((B, A), (B, C)):
        per_id = {}
        for p in res.pieces:
            if (p.src_cell, p.dst_cell) == (U, V):
                per_id[p.identity] = per_id.get(p.identity, 0.0) + p.volume
        assert set(per_id) == {(s, 0) for s in B_gpus}, (U, V, per_id)
        assert all(abs(v - 1.0) < 1e-6 for v in per_id.values()), per_id

    # ingress distribution for AllGather fans out to ALL of the destination cell.
    for dem in res.intra_demands:
        if dem.kind == "ingress_distribution" and dem.dst_gpus:
            if dem.identity[0] in B_gpus:
                pieces_dst = {p.dst_cell for p in res.pieces if p.identity == dem.identity}
                # target set is all gpus of whichever cell this ingress belongs to
                assert dem.dst_gpus in (tuple(A_gpus), tuple(C_gpus)), dem

    # self_distribution for B's internal allgather (source & dest both in B).
    selfd = [d for d in res.intra_demands if d.kind == "self_distribution"]
    assert {d.identity for d in selfd} == {(s, 0) for s in B_gpus}, selfd
    for d in selfd:
        assert d.cell == B and d.dst_gpus == tuple(g for g in B_gpus if g != d.src_gpu), d
    print("  [2] Host B forced-relay oracle (3 coalesced egress relays, native exempt) OK")


# ------------------------------------------------------------------------------------------
def _symmetric_mapping():
    """Synthetic 2-cell topology where every GPU is its own gateway (uplinks == GPUs):
    cell0 gpus {0,1}, cell1 gpus {2,3}, both cells homed to one switch (fine 4 / coarse 2)."""
    fine_to_coarse = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
    mapping = HierarchyMapping(
        fine_to_coarse=fine_to_coarse,
        coarse_cells={0: Cell(members=[0, 1], gpus=[0, 1]),
                      1: Cell(members=[2, 3], gpus=[2, 3])},
        coarse_passthrough={2: 4},
        boundary_gpu={(0, 2): [0, 1], (1, 2): [2, 3]},
        num_coarse=3,
    )
    cap = [[0.0] * 5 for _ in range(5)]
    for g in (0, 1, 2, 3):
        cap[g][4] = cap[4][g] = 100.0
    fine_topo = SimpleNamespace(capacity=cap, switch_indices=[4])
    return mapping, fine_topo


def test_zero_relay_symmetric():
    mapping, fine_topo = _symmetric_mapping()
    n = 5
    fine_demand = np.zeros((n, n, 1), dtype=np.int32)
    for s in (0, 1):                      # cell0 gpus sourced to cell1 gpus (allgather-shape)
        for t in (2, 3):
            fine_demand[s][t][0] = 1
    # both units cross on the shared coarse link (0 -> switch2 -> 1); each gpu owns half.
    pcp = {(0, 1, 0): [_single_switch_path(0, 1, 2, 1.0, 0), _single_switch_path(0, 1, 2, 1.0, 1)]}
    solver = _fake_solver(pcp, switch_indices=[2])
    res = resolve_identities(solver, mapping, fine_demand, fine_topo)
    egress = [d for d in res.intra_demands if d.kind == "egress_stage"]
    assert egress == [], egress   # identity native to its own gateway -> zero relay
    # each identity delivered exactly once, on its native gateway
    for p in res.pieces:
        assert p.egress_gpu == p.identity[0], p
    print("  [3] symmetric case: zero egress relay OK")


# ------------------------------------------------------------------------------------------
def test_alltoall_single_target():
    """AllToAll: identity (s, ci) is wanted by exactly ONE destination GPU -> ingress
    distribution target is a singleton (vs all-of-cell for AllGather)."""
    mapping, fine_topo = _symmetric_mapping()
    n = 5
    # 2 chunks per source (one per cell1 GPU): chunk 0 -> gpu2, chunk 1 -> gpu3 (alltoall shape).
    fine_demand = np.zeros((n, n, 2), dtype=np.int32)
    for s in (0, 1):
        fine_demand[s][2][0] = 1
        fine_demand[s][3][1] = 1
    id_sets, targets = identity_sets(fine_demand, mapping)
    assert len(id_sets[(0, 1)]) == 4, id_sets[(0, 1)]          # |U|*|V| = 2*2
    assert targets[((0, 0), 1)] == (2,)
    assert targets[((0, 1), 1)] == (3,)

    pcp = {(0, 1, 0): [_single_switch_path(0, 1, 2, 1.0, k) for k in range(4)]}
    solver = _fake_solver(pcp, switch_indices=[2])
    res = resolve_identities(solver, mapping, fine_demand, fine_topo)
    for dem in res.intra_demands:
        if dem.kind == "ingress_distribution":
            assert len(dem.dst_gpus) == 1, dem       # single-GPU scatter target
    print("  [4] AllToAll single-GPU ingress target OK")


def main() -> None:
    test_identity_sets_match_coarsify()
    test_hostB_forced_relay()
    test_zero_relay_symmetric()
    test_alltoall_single_target()
    print("identity resolution structural tests OK")


if __name__ == "__main__":
    main()
