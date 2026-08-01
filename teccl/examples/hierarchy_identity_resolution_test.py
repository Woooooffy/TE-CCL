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
  5. REPLAY: reconstruct the exact per_chunk_flow_paths from the real solved
     Schedules/coarse_hetero_allgather_lp.json and run the resolver end-to-end (this is the input
     that crashed with the single-sort bug; it exercises the C->T1 multi-GPU boundary and
     two-switch relay paths that the synthetic fixtures only partially cover).

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_identity_resolution_test
"""
import json
import os
import re
from collections import defaultdict
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
    """A U -> switch -> V coarse path as two 6-tuples (s, i, j, c, vol, k). Hops are listed
    DEST-first, the way dig_to_source (which back-traces from the destination) emits them; the
    resolver's sort->reverse->sort normalization flips them to source-first."""
    return [(cell_s, switch, cell_d, 0, vol, epoch),
            (cell_s, cell_s, switch, 0, vol, epoch)]


def _two_switch_path(cell_s, cell_d, sw1, sw2, vol, epoch):
    """A U -> sw1 -> sw2 -> V coarse path (single-homed relay across the top mesh), DEST-first."""
    return [(cell_s, sw2, cell_d, 0, vol, epoch),
            (cell_s, sw1, sw2, 0, vol, epoch),
            (cell_s, cell_s, sw1, 0, vol, epoch)]


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


_SEG_RE = re.compile(
    r"(\d+)->(\d+) with volume ([\d.]+) in epoch (\d+)(?: via switches (.+))?$")


def _replay_per_chunk_flow_paths(schedule_json: dict):
    """Invert a solved schedule's '8-Chunk paths' back into the per_chunk_flow_paths structure
    resolve_identities consumes. Each grouped segment 'U->V volume X epoch K via switches S1->S2'
    expands to the raw hops U->S1, S1->S2, S2->V (emitted DEST-first, as dig_to_source does)."""
    key_re = re.compile(r"Demand at (\d+) for chunk (\d+) from (\d+) met by epoch \d+")
    pcp = defaultdict(list)
    for key, paths in schedule_json["8-Chunk paths"].items():
        km = key_re.match(key)
        d, c, s = int(km.group(1)), int(km.group(2)), int(km.group(3))
        for path in paths:                         # each path = list of [epoch, segment_str]
            each_path = []
            for _epoch, seg in path:
                m = _SEG_RE.search(seg)
                start, end = int(m.group(1)), int(m.group(2))
                vol, ep = float(m.group(3)), int(m.group(4))
                switches = [int(x) for x in m.group(5).split("->")] if m.group(5) else []
                nodes = [start] + switches + [end]
                hops = [(s, nodes[h], nodes[h + 1], c, vol, ep) for h in range(len(nodes) - 1)]
                each_path.extend(reversed(hops))   # dest-first, matching dig_to_source
            pcp[(s, d, c)].append(each_path)
    return dict(pcp)


def test_replay_real_allgather_json():
    path = "Schedules/coarse_hetero_allgather_lp.json"
    if not os.path.exists(path):
        print(f"  [5] SKIP replay: {path} not found")
        return
    with open(path) as f:
        schedule_json = json.load(f)

    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    coarse, m = abstract(topo)
    B, C = 1, 2
    fine_demand = build_demand(Collective.ALLGATHER, topo, num_chunks=1)

    pcp = _replay_per_chunk_flow_paths(schedule_json)
    solver = _fake_solver(pcp, switch_indices=list(coarse.switch_indices))
    res = resolve_identities(solver, m, fine_demand, topo)   # must not raise (was the crash)

    id_sets, _ = identity_sets(fine_demand, m)
    # every demanded (U,V) delivers each identity exactly once
    for (U, V), ids in id_sets.items():
        per_id = defaultdict(float)
        for p in res.pieces:
            if (p.src_cell, p.dst_cell) == (U, V):
                per_id[p.identity] += p.volume
        assert set(per_id) == set(ids), (U, V, sorted(per_id), sorted(ids))
        assert all(abs(v - 1.0) < 1e-5 for v in per_id.values()), (U, V, dict(per_id))

    # single-homed Host B: identities native to g5,g6,g7 relay to its lone gateway g4.
    b_relays = sorted((d.src_gpu, d.dst_gpus[0]) for d in res.intra_demands
                      if d.kind == "egress_stage" and d.cell == B)
    assert b_relays == [(5, 4), (6, 4), (7, 4)], b_relays

    # Host C's multi-GPU boundary to T1 (g11, g13) is actually used on some egress piece.
    c_egress_gpus = {p.egress_gpu for p in res.pieces if p.src_cell == C}
    assert {11, 13} & c_egress_gpus, sorted(c_egress_gpus)

    # NATIVE-FIRST egress ordering, checked at the granularity it is enforced: per (src_cell,
    # dst_cell, gateway). Within one demand pair on a gateway, a native identity never sits on a
    # later epoch than a relayed one -- native_epoch <= every relayed epoch. (Equality is the
    # forced case, e.g. g8 where the coarse LP packed two units into the same epoch.) NOTE the
    # property is intentionally NOT asserted across different destinations sharing one physical
    # uplink -- that cross-pair interleaving is set by the coarse LP's epoch assignment, not by
    # this per-pair ordering.
    by_gateway = defaultdict(list)   # (src_cell, dst_cell, egress_gpu) -> [(send_epoch, is_native)]
    for p in res.pieces:
        by_gateway[(p.src_cell, p.dst_cell, p.egress_gpu)].append(
            (p.send_epoch, p.identity[0] == p.egress_gpu))
    for (U, V, g), rows in by_gateway.items():
        native_eps = [e for e, nat in rows if nat]
        relay_eps = [e for e, nat in rows if not nat]
        if native_eps and relay_eps:
            assert max(native_eps) <= min(relay_eps), (U, V, g, sorted(rows))

    print(f"  [5] replay real allgather JSON OK: {len(res.pieces)} pieces, "
          f"Host-B relays {b_relays}, C egress gpus {sorted(c_egress_gpus)}, native-first ordering holds")


def main() -> None:
    test_identity_sets_match_coarsify()
    test_hostB_forced_relay()
    test_zero_relay_symmetric()
    test_alltoall_single_target()
    test_replay_real_allgather_json()
    print("identity resolution structural tests OK")


if __name__ == "__main__":
    main()
