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
  6. LEXICOGRAPHIC objective: when landing an identity on its target would cost a NATIVE egress,
     the native egress wins. The two relays are not interchangeable (egress is hard and upstream of
     the network hop, ingress is soft and downstream), so this ordering must be strict rather than
     a weighted trade -- with equal weights the solver is indifferent and can silently regress.
  7. Ingress gateway PREFERS a target when the choice is free, instead of always taking
     boundary_gpu[...][0].
  8. Ingress fine-downlink capacity is respected per epoch, on the real replay -- a coarse link's
     capacity is the SUM of the fine downlinks behind it, so an unbounded choice can oversubscribe
     one while a sibling idles.
  9. Sub-chunk refinement: every emitted volume is exactly 1.0, the sub-chunks of one identity
     partition it, and the ChunkScale conserves the per-GPU payload.
 10. CELL RELAY (host transit) on BridgedIslandsCluster, where A -> C has no route but
     A -> T0 -> B -> T1 -> C: both legs of one delivery carry the SAME sub-chunk identity, the
     co-located variant emits ZERO transit relays, the split variant fails loud with the §6
     forwarding-dwell message, a non-co-located forward gets the right release/deadline pair, a
     saturated bridge is refused rather than oversubscribed, and a cyclic route is rejected.

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_identity_resolution_test
"""
import collections
import json
import os
import re
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from teccl.hierarchy.abstract import abstract, coarsify_demand, level_chunk_units
from teccl.hierarchy.cell import Cell
from teccl.hierarchy.reconstruct import (HierarchyMapping, _Assignment, _bridging_gpus,
                                          _CoarsePiece, _CoarseRoute, _emit_refined,
                                          _pick_ingress, _place_downstream_legs, _validate_route,
                                          identity_sets, resolve_identities)
from teccl.hierarchy.bands import PROLOGUE_BAND, assign_bands, release_of
from teccl.hierarchy.scale import ChunkScale
from teccl.topologies.bridged_islands_cluster import (BridgedIslandsCluster,
                                                      BridgedIslandsSplitCluster)
from teccl.input_data import Collective, TopologyParams
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster


def _fake_solver(per_chunk_flow_paths, switch_indices, epoch_duration=0.02):
    """Minimal stand-in for a solved LPFormulation: resolve_identities reads
    .per_chunk_flow_paths, .topology.switch_indices and .epoch_duration.

    epoch_duration is load-bearing, not decoration: it converts a fine link's bandwidth into the
    volume that link can absorb in one coarse epoch, which is what bounds the ingress gateway
    choice. 0.02 s matches the hetero coarse solve (1 GB chunk over its 50 GB/s slowest uplink)."""
    return SimpleNamespace(
        per_chunk_flow_paths=per_chunk_flow_paths,
        topology=SimpleNamespace(switch_indices=switch_indices),
        epoch_duration=epoch_duration,
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

    # coarse pieces: B->A via T0 (4 units); B->C via T0->T1 (4 units). All egress on g4's link,
    # so the 8 units must occupy 8 DISTINCT coarse epochs: B's single uplink carries one chunk
    # per epoch, and identity resolution now paces each piece to fill one coarse epoch and
    # asserts the result fits the fine link (_assert_rate_within_capacity). Overlapping B->A and
    # B->C on epochs 0..3 would be an infeasible coarse solution -- 2 GB in 0.02 s out of a
    # 50 GB/s link -- which no real coarse solve emits. The relay structure this test asserts is
    # epoch-independent, so spreading them changes nothing it checks.
    pcp = {(B, A, 0): [_single_switch_path(B, A, T0, 1.0, k) for k in range(4)],
           (B, C, 0): [_two_switch_path(B, C, T0, T1, 1.0, k) for k in range(4, 8)]}
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

    # self_distribution for B's internal allgather (source & dest both in B), MINUS anything the
    # egress_stage already delivers: 5/6/7 each relay to gateway 4, so their internal fan-out no
    # longer names 4 -- one physical send cannot be owed to two demands. Gateway 4 stages nothing
    # (its data is native to the uplink), so its own fan-out still covers all of 5/6/7.
    staged = {(d.identity, d.src_gpu): set(d.dst_gpus) for d in egress}
    selfd = [d for d in res.intra_demands if d.kind == "self_distribution"]
    assert {d.identity for d in selfd} == {(s, 0) for s in B_gpus}, selfd
    for d in selfd:
        covered = staged.get((d.identity, d.src_gpu), set())
        want = tuple(g for g in B_gpus if g != d.src_gpu and g not in covered)
        assert d.cell == B and d.dst_gpus == want, (d, want)
    assert {d.src_gpu: d.dst_gpus for d in selfd} == {
        4: (5, 6, 7), 5: (6, 7), 6: (5, 7), 7: (5, 6)}, {d.src_gpu: d.dst_gpus for d in selfd}

    # The three demand kinds must be DISJOINT: no (identity, src, dst) delivery is asked twice.
    seen = defaultdict(set)
    for d in res.intra_demands:
        for t in d.dst_gpus:
            if t != d.src_gpu:
                seen[(d.identity, d.src_gpu, t)].add(d.kind)
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dup, f"a delivery is claimed by more than one demand kind: {dict(list(dup.items())[:4])}"
    print("  [2] Host B forced-relay oracle (3 coalesced egress relays, native exempt; "
          "self_distribution disjoint from staging) OK")


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
    fine_topo = SimpleNamespace(capacity=cap, switch_indices=[4], chunk_size=1)
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
    # Both the epoch and the chunk unit must come FROM THE SCHEDULE, not from this file's defaults.
    # They move together (abstract.set_level_chunk coarsens the topology and the demand at once), so
    # a hardcoded 0.02/g=1 silently describes a different problem than the one being replayed -- and
    # then trips the resolver's own capacity assert, since every piece is paced against the epoch.
    solver = _fake_solver(pcp, switch_indices=list(coarse.switch_indices),
                          epoch_duration=schedule_json["1-Epoch_Duration"])
    # The checked-in schedule was solved in the coarse level's OWN chunk unit (the driver calls
    # abstract.set_level_chunk), so its volumes are in units of g fine identities. Re-derive that g
    # the same way the driver does instead of assuming 1 -- assuming it made this replay fail the
    # moment coarsening was turned on, since the coarse volumes then no longer count identities.
    level_chunk = level_chunk_units(coarsify_demand(fine_demand, m))
    res = resolve_identities(solver, m, fine_demand, topo,
                             level_chunk=level_chunk)      # must not raise (was the crash)

    q = res.subdivision
    assert q >= 1 and res.scale is not None, (q, res.scale)
    # Sub-chunk refinement: nothing downstream should ever see a fractional volume, and the
    # refinement must conserve the payload it re-denominates.
    assert all(abs(p.volume - 1.0) < 1e-9 for p in res.pieces), \
        sorted({p.volume for p in res.pieces})
    assert all(abs(d.volume - 1.0) < 1e-9 for d in res.intra_demands), \
        sorted({d.volume for d in res.intra_demands})
    assert abs(res.scale.payload_per_gpu - topo.chunk_size * 1) < 1e-9, res.scale
    assert res.scale.refinement_from_root == q, (res.scale, q)

    id_sets, _ = identity_sets(fine_demand, m)
    # Every demanded (U,V) delivers each identity exactly once. Identities are now sub-chunks
    # (s, ci*q + j), so fold them back to the original (s, ci) and require exactly q sub-chunks --
    # i.e. the pieces PARTITION each identity rather than merely summing to its volume.
    for (U, V), ids in id_sets.items():
        per_id = defaultdict(list)
        for p in res.pieces:
            if (p.src_cell, p.dst_cell) == (U, V):
                s, sub = p.identity
                per_id[(s, sub // q)].append(sub % q)
        assert set(per_id) == set(ids), (U, V, sorted(per_id), sorted(ids))
        for orig, subs in per_id.items():
            assert sorted(subs) == list(range(q)), (U, V, orig, sorted(subs))

    # single-homed Host B: identities native to g5,g6,g7 relay to its lone gateway g4. Each
    # original identity is q sub-chunks now, and each sub-chunk is its own relay, so the distinct
    # (src, gateway) pairs are unchanged and each occurs exactly q times.
    b_counts = collections.Counter((d.src_gpu, d.dst_gpus[0]) for d in res.intra_demands
                                   if d.kind == "egress_stage" and d.cell == B)
    b_relays = sorted(b_counts)
    assert b_relays == [(5, 4), (6, 4), (7, 4)], b_relays
    assert set(b_counts.values()) == {q}, dict(b_counts)

    # Host C's multi-GPU boundary to T1 (g11, g13) is actually used on some egress piece.
    c_egress_gpus = {p.egress_gpu for p in res.pieces if p.src_cell == C}
    assert {11, 13} & c_egress_gpus, sorted(c_egress_gpus)

    # NATIVE-FIRST egress ordering. This used to be a hard post-hoc sort; it is now objective
    # tier 3 (keep relayed identities off the earliest egress epochs), so it is a PREFERENCE that
    # the two dominating tiers may override -- assert the aggregate, not a per-gateway inequality.
    # Aggregate mean epoch of relayed egress >= that of native egress: relays drift late, which is
    # what keeps them out of the pre-epoch-0 staging prologue.
    native_eps = [p.send_epoch for p in res.pieces if p.identity[0] == p.egress_gpu]
    relay_eps = [p.send_epoch for p in res.pieces if p.identity[0] != p.egress_gpu]
    if native_eps and relay_eps:
        assert (sum(relay_eps) / len(relay_eps)) >= (sum(native_eps) / len(native_eps)), \
            (sorted(native_eps), sorted(relay_eps))

    # Ingress fine-downlink capacity, per (gpu, epoch), in absolute GB. Volumes are sub-chunks now,
    # so scale by the refined chunk size rather than assuming 1 GB -- and take the epoch from the
    # schedule being replayed, for the same reason the solver shim does.
    ep = schedule_json["1-Epoch_Duration"]
    used = defaultdict(float)
    for p in res.pieces:
        used[(p.ingress_gpu, p.arrival_epoch)] += p.volume * res.scale.bytes_per_chunk
    over = []
    for (h, k), vol in sorted(used.items()):
        cap = topo.capacity[18][h] if topo.capacity[18][h] else topo.capacity[17][h]
        if vol > cap * ep + 1e-9:
            over.append((h, k, vol, cap * ep))
    assert not over, f"ingress downlink oversubscribed: {over}"

    # Host C's ingress must be SPREAD, not piled onto one GPU. Taking boundary_gpu[...][0]
    # unconditionally ran g11 at 150% of its downlink while a sibling sat idle; the capacity-aware
    # choice is what fixed it, and the oversubscription assert above is the direct guard.
    #
    # This deliberately does NOT name g13. That expectation was over-fitted to a replay parameterized
    # with the wrong epoch (0.02 against a schedule solved at 0.04): under half the real budget the
    # resolver was forced to spill onto g13, and the test read the spill as the property. With the
    # correct budget it lands on a GPU that actually wants the data instead -- strictly better, and
    # the objective's ingress tier asking for exactly that. What must hold in both worlds is that
    # more than one boundary GPU is used.
    c_ingress = collections.Counter(p.ingress_gpu for p in res.pieces if p.dst_cell == C)
    assert len(c_ingress) > 1, dict(c_ingress)
    assert max(c_ingress.values()) < sum(c_ingress.values()), dict(c_ingress)

    print(f"  [5] replay real allgather JSON OK: {len(res.pieces)} pieces (Q={q}, "
          f"{res.scale}), Host-B relays {b_relays}, C egress gpus {sorted(c_egress_gpus)}, "
          f"C ingress {dict(c_ingress)}, ingress capacity respected")


# ------------------------------------------------------------------------------------------
def _two_path_mapping():
    """Two cells, two switches, and a RAIL-LIKE constraint: each cell reaches switch A only via
    its gpu0 and switch B only via its gpu1. So the egress gateway and the landing GPU are both
    determined by which switch a piece crosses -- the setup where the egress and ingress
    objectives can genuinely conflict.

        cell0 gpus {0,1}   boundary {swA: [0], swB: [1]}
        cell1 gpus {2,3}   boundary {swA: [2], swB: [3]}
        fine swA = 4, swB = 5;  coarse cell0 = 0, cell1 = 1, swA = 2, swB = 3
    """
    mapping = HierarchyMapping(
        fine_to_coarse={0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3},
        coarse_cells={0: Cell(members=[0, 1], gpus=[0, 1]),
                      1: Cell(members=[2, 3], gpus=[2, 3])},
        coarse_passthrough={2: 4, 3: 5},
        boundary_gpu={(0, 2): [0], (0, 3): [1], (1, 2): [2], (1, 3): [3]},
        num_coarse=4,
    )
    cap = [[0.0] * 6 for _ in range(6)]
    for g, sw in ((0, 4), (1, 5), (2, 4), (3, 5)):
        cap[g][sw] = cap[sw][g] = 100.0
    return mapping, SimpleNamespace(capacity=cap, switch_indices=[4, 5], chunk_size=1)


def test_lexicographic_egress_dominates():
    """Landing on the target must NOT be bought with a native egress.

    Identity (0,0) is native to gpu0 (which owns swA) but wanted by gpu3 (which owns swB);
    identity (1,0) is native to gpu1 (swB) but wanted by gpu2 (swA). Both assignments are
    available and each costs exactly one relay, so a SUMMED objective is indifferent:

        native egress + ingress relay   vs   egress relay + landing on target

    The lexicographic objective must take the first: an egress_stage relay is HARD (its deadline is
    the network send epoch) and sits upstream of the network hop, while ingress distribution is
    SOFT and downstream. Assert zero egress relays, i.e. both identities left on their native
    gateway and paid the ingress hop instead."""
    mapping, fine_topo = _two_path_mapping()
    fine_demand = np.zeros((6, 6, 1), dtype=np.int32)
    fine_demand[0][3][0] = 1          # native gpu0 (swA) -> wanted by gpu3 (swB)
    fine_demand[1][2][0] = 1          # native gpu1 (swB) -> wanted by gpu2 (swA)
    pcp = {(0, 1, 0): [_single_switch_path(0, 1, 2, 1.0, 0),      # via swA: gpu0 -> gpu2
                       _single_switch_path(0, 1, 3, 1.0, 0)]}     # via swB: gpu1 -> gpu3
    res = resolve_identities(_fake_solver(pcp, switch_indices=[2, 3]),
                             mapping, fine_demand, fine_topo)
    egress = [d for d in res.intra_demands if d.kind == "egress_stage"]
    assert egress == [], f"native egress was sacrificed to save an ingress relay: {egress}"
    landed = {(p.identity, p.egress_gpu, p.ingress_gpu) for p in res.pieces}
    assert landed == {((0, 0), 0, 2), ((1, 0), 1, 3)}, sorted(landed)
    # ...and the ingress relays it chose instead are emitted, one per identity, off-target.
    relays = sorted((d.identity, d.src_gpu, d.dst_gpus) for d in res.intra_demands
                    if d.kind == "ingress_distribution")
    assert relays == [((0, 0), 2, (3,)), ((1, 0), 3, (2,))], relays
    print("  [6] lexicographic objective: native egress beats landing on target OK")


def test_ingress_prefers_target():
    """When the ingress choice is FREE, prefer the GPU that actually wants the data.

    _symmetric_mapping's cell1 boundary owns both gpu2 and gpu3, so a piece arriving there can land
    on either. Identity (0,1) is wanted only by gpu3; the old code took boundary_gpu[...][0] == 2
    unconditionally and paid a 2->3 relay. Nothing about the egress side differs between the two
    choices here, so the ingress tier decides uncontested."""
    mapping, fine_topo = _symmetric_mapping()
    fine_demand = np.zeros((5, 5, 2), dtype=np.int32)
    fine_demand[0][2][0] = 1          # (0,0) wanted by gpu2
    fine_demand[0][3][1] = 1          # (0,1) wanted by gpu3
    pcp = {(0, 1, 0): [_single_switch_path(0, 1, 2, 1.0, k) for k in range(2)]}
    res = resolve_identities(_fake_solver(pcp, switch_indices=[2]),
                             mapping, fine_demand, fine_topo)
    landing = {p.identity: p.ingress_gpu for p in res.pieces}
    for identity, h in landing.items():
        want = 2 if identity[1] // res.subdivision == 0 else 3
        assert h == want, (identity, h, want, landing)
    assert not [d for d in res.intra_demands
                if d.kind == "ingress_distribution" and d.src_gpu not in d.dst_gpus], \
        "an identity landed off-target although a target gateway was available"
    print("  [7] ingress gateway prefers a target when the choice is free OK")


def test_level_chunk_unit_invariance():
    """The resolution must not depend on WHICH CHUNK the coarse level solved in.

    abstract.set_level_chunk lets a coarse level re-express itself in a `g`-times coarser unit (the
    GCD of its demands), so the same physical solution arrives here with volumes divided by g. The
    resolution is denominated in FINE IDENTITIES, so it must come out bit-identical -- the slot
    renormalization against the identity count is what absorbs the unit change, and this test is
    what says so out loud. If it ever fails, some step below has grown a unit assumption.

    Note what is held fixed: `epoch_duration` is the ONE place the coarse epoch's absolute value
    enters (via the per-epoch ingress downlink budget). A real coarsening scales the epoch by g AND
    the volume per epoch by g, so that budget's tightness is unchanged; holding it fixed here is
    what makes this a like-for-like comparison rather than a quietly loosened one.
    """
    path = "Schedules/coarse_hetero_allgather_lp.json"
    if not os.path.exists(path):
        print(f"  [8] SKIP level-chunk invariance: {path} not found")
        return
    with open(path) as f:
        schedule_json = json.load(f)
    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    coarse, m = abstract(topo)
    fine_demand = build_demand(Collective.ALLGATHER, topo, num_chunks=1)
    # Normalize the replayed solution to ONE FINE IDENTITY PER UNIT before starting, because the
    # checked-in schedule was itself solved in a coarsened unit (`9-Chunk_Size` fine chunks per
    # coarse chunk). This test's whole subject is the g-dependence, so its baseline has to be the
    # g=1 denomination rather than whatever the file happened to be written in -- otherwise the
    # "base" is already a coarsened case and the comparison is against the wrong thing.
    g_sched = level_chunk_units(coarsify_demand(fine_demand, m))
    raw = _replay_per_chunk_flow_paths(schedule_json)
    pcp = {k: [[(s, i, j, c, v * g_sched, e) for (s, i, j, c, v, e) in path] for path in paths]
           for k, paths in raw.items()}
    switches = list(coarse.switch_indices)
    ep = schedule_json["1-Epoch_Duration"]

    def fingerprint(res):
        return ([(p.identity, p.egress_gpu, p.ingress_gpu, p.send_epoch, round(p.volume, 9))
                 for p in res.pieces],
                sorted((d.cell, d.kind, d.identity, d.src_gpu, tuple(d.dst_gpus))
                       for d in res.intra_demands),
                res.subdivision, res.scale)

    base = fingerprint(resolve_identities(_fake_solver(pcp, switches, epoch_duration=ep), m, fine_demand, topo))
    for g in (2, 4, 8):
        rescaled = {k: [[(s, i, j, c, v / g, e) for (s, i, j, c, v, e) in path] for path in paths]
                    for k, paths in pcp.items()}
        res = resolve_identities(_fake_solver(rescaled, switches, epoch_duration=ep), m, fine_demand, topo,
                                 level_chunk=g)
        assert fingerprint(res) == base, f"resolution changed at level_chunk={g}"

    # And the volume/identity-count check must still fire on a genuine mismatch rather than be
    # defeated by the new factor: claiming g=1 for a solution denominated in halves is exactly the
    # upstream bug the assert exists to catch.
    halved = {k: [[(s, i, j, c, v / 2, e) for (s, i, j, c, v, e) in path] for path in paths]
              for k, paths in pcp.items()}
    try:
        resolve_identities(_fake_solver(halved, switches, epoch_duration=ep), m, fine_demand, topo, level_chunk=1)
    except AssertionError:
        pass
    else:
        raise AssertionError("a coarse volume / identity-count mismatch was not caught")
    print("  [8] resolution is invariant to the coarse level's chunk unit (g=2,4,8); "
          "mismatched g still rejected OK")




# ------------------------------------------------------------------------------------------
# 10. CELL RELAY: a coarse path that store-and-forwards through an intermediate CELL.
# ------------------------------------------------------------------------------------------
# Coarse node ids on BridgedIslandsCluster: cells A=0, B=1, C=2; top switches T0=3, T1=4.
_A, _B, _C, _T0, _T1 = 0, 1, 2, 3, 4


def _bridged_paths():
    """A coarse AllGather flow on BridgedIslandsCluster, DEST-first as dig_to_source emits.

    Every link here absorbs 2 units per epoch (100 GB/s x 0.02 s, 1 GB chunks), and the schedule
    is laid out to stay inside that -- the point of the fixture is the TRANSIT, so nothing else
    should be able to fail first. A -> C and C -> A have no route but the bridge, and each dwells
    exactly one epoch at host B (arrival + 1, which is what the coarse solve's midFC gives).
    """
    def hop(s, i, j, vol, k):
        return (s, i, j, 0, vol, k)
    return {
        # epoch 0: the four direct neighbour deliveries
        (_A, _B, 0): [[hop(_A, _T0, _B, 2.0, 0), hop(_A, _A, _T0, 2.0, 0)]],
        (_B, _A, 0): [[hop(_B, _T0, _A, 2.0, 0), hop(_B, _B, _T0, 2.0, 0)]],
        (_B, _C, 0): [[hop(_B, _T1, _C, 2.0, 0), hop(_B, _B, _T1, 2.0, 0)]],
        (_C, _B, 0): [[hop(_C, _T1, _B, 2.0, 0), hop(_C, _C, _T1, 2.0, 0)]],
        # epochs 1 -> 2: the two TRANSIT deliveries through the bridge host
        (_A, _C, 0): [[hop(_A, _T1, _C, 2.0, 2), hop(_A, _B, _T1, 2.0, 2),
                       hop(_A, _T0, _B, 2.0, 1), hop(_A, _A, _T0, 2.0, 1)]],
        (_C, _A, 0): [[hop(_C, _T0, _A, 2.0, 2), hop(_C, _B, _T0, 2.0, 2),
                       hop(_C, _T1, _B, 2.0, 1), hop(_C, _C, _T1, 2.0, 1)]],
    }


def _resolve_bridged(cls, paths=None):
    topo = cls(TopologyParams(name=cls.__name__, chunk_size=1))
    coarse, m = abstract(topo)
    fine_demand = build_demand(Collective.ALLGATHER, topo, num_chunks=1)
    solver = _fake_solver(paths or _bridged_paths(), list(coarse.switch_indices),
                          epoch_duration=0.02)
    return resolve_identities(solver, m, fine_demand, topo)


def test_transit_route_both_legs_one_identity():
    """The two halves of a transit delivery are ONE commodity, and that is what makes the chain
    expressible: they share a sub-chunk identity, so the intervening demand orders them and the
    per-(identity, dst_cell) partition check still holds.

    Keyed by leg -- the old shape -- the halves would be unrelated, filed under (A,B) and (B,C),
    and neither the demand anchor nor the sub-chunk index could relate them.
    """
    res = _resolve_bridged(BridgedIslandsCluster)
    # A's chunks are (0, 0) and (1, 0); they reach cell C only through the bridge.
    for identity in ((0, 0), (1, 0)):
        legs = sorted((p for p in res.pieces if p.identity == identity
                       and (p.src_cell, p.dst_cell) in ((_A, _B), (_B, _C))
                       and p.send_epoch >= 1),
                      key=lambda p: p.send_epoch)
        assert len(legs) == 2, f"{identity}: expected 2 transit legs, got {len(legs)}"
        first, second = legs
        assert (first.src_cell, first.dst_cell) == (_A, _B)
        assert (second.src_cell, second.dst_cell) == (_B, _C)
        assert first.identity == second.identity, "the two legs must be the same commodity"
        # The dwell the coarse solve budgets: land at the end of epoch k, leave in k+1.
        assert second.send_epoch == first.arrival_epoch + 1, (
            f"{identity}: transit dwell is {second.send_epoch - first.arrival_epoch}, want 1")
        # And it really is the bridge host doing the forwarding, on the co-located GPU.
        assert first.ingress_gpu == second.egress_gpu == 2, (
            f"{identity}: expected the bridge GPU 2 to land and re-send, got "
            f"{first.ingress_gpu} -> {second.egress_gpu}")
    print("  [10a] transit: both legs of one delivery carry the same identity, dwell 1 epoch, "
          "forwarded by the bridge GPU OK")


def test_colocated_transit_costs_no_intra_work():
    """When one GPU owns both of the bridge host's uplinks, forwarding is free: the bytes land on
    the GPU that also sends them onward, so there is no intra-cell hop and no relay demand.

    This is the design's §6 claim made checkable. Any egress_stage at the bridge cell must be a
    NATIVE staging (some non-gateway GPU handing its own chunk to the gateway), never a forward.
    """
    res = _resolve_bridged(BridgedIslandsCluster)
    transit_ids = {(0, 0), (1, 0), (4, 0), (5, 0)}   # A's and C's chunks -- the ones that transit
    forwards = [d for d in res.intra_demands
                if d.kind == "egress_stage" and d.cell == _B and d.identity in transit_ids]
    assert not forwards, f"co-located transit should need no relay, got {forwards}"
    # The bridge cell's own staging is unaffected: g3 still hands its native chunk to gateway g2.
    native_staging = [d for d in res.intra_demands
                      if d.kind == "egress_stage" and d.cell == _B]
    assert all(d.src_gpu == 3 and d.dst_gpus == (2,) for d in native_staging), native_staging
    assert all(d.release_band is None for d in native_staging), (
        "native staging is available from the prologue and must carry no explicit release")
    print(f"  [10b] co-located transit needs zero relay work ({len(native_staging)} native "
          f"stagings at the bridge, all prologue-released) OK")


def test_split_bridge_fails_loud():
    """With the bridge host's two uplinks on DIFFERENT GPUs there is no co-located transit at all,
    and no amount of work in identity resolution fixes it: forwarding needs arrival + 2 and the
    coarse solve budgets arrival + 1. The failure is a true statement about the topology, so it
    must be a clear one -- and it must point at the coarse forwarding dwell (§6), which is the
    actual fix and is deliberately not attempted here."""
    try:
        _resolve_bridged(BridgedIslandsSplitCluster)
    except RuntimeError as e:
        msg = str(e)
        assert "no GPU owns both links" in msg, msg
        assert "forwarding dwell" in msg, msg
        print("  [10c] split bridge (no co-located GPU) fails loud, pointing at the coarse "
              "forwarding dwell OK")
        return
    raise AssertionError("a bridge host with no co-located GPU must be refused, not resolved")


def _transit_assignment(leg_gpus, send_epochs=(1, 2), arrival_epochs=(1, 2)):
    """A hand-built two-leg A -> B -> C assignment, so the emission can be checked without a
    solver and with the leg placement chosen rather than derived."""
    legs = (
        _CoarsePiece(src_cell=_A, dst_cell=_B, egress_neighbor=_T0, ingress_neighbor=_T0,
                     via_switches=(9,), volume=1.0, send_epoch=send_epochs[0],
                     arrival_epoch=arrival_epochs[0], origin=(_A, _C)),
        _CoarsePiece(src_cell=_B, dst_cell=_C, egress_neighbor=_T1, ingress_neighbor=_T1,
                     via_switches=(10,), volume=1.0, send_epoch=send_epochs[1],
                     arrival_epoch=arrival_epochs[1], origin=(_A, _C)),
    )
    route = _CoarseRoute(origin=(_A, _C), legs=legs, volume=1.0)
    return _Assignment(src_cell=_A, dst_cell=_C, identity=(0, 0), route=route,
                       egress_gpu=leg_gpus[0][0], ingress_gpu=leg_gpus[-1][1], volume=1.0,
                       leg_gpus=leg_gpus, exact=Fraction(1))


def test_transit_forward_release_and_deadline():
    """A forward that is NOT co-located gets both of its bounds, and they are different numbers.

    Unreachable through the resolver while co-location is required, so it is exercised here on a
    hand-placed assignment -- the record has to be right BEFORE the coarse dwell makes it
    reachable, or the dwell work would land on a representation that silently drops one bound.
    The release is the inbound leg's arrival + 1 (a piece lands at the END of its epoch); the
    deadline is the outbound leg's send.
    """
    a = _transit_assignment(leg_gpus=((0, 2), (3, 4)))   # lands on g2, leaves from g3
    res = _emit_refined([a], {((0, 0), _C): (4, 5)}, q=1,
                        scale=ChunkScale(bytes_per_chunk=1, num_chunks=1), coarse_epoch=0.02)
    fwd = [d for d in res.intra_demands if d.kind == "egress_stage" and d.cell == _B]
    assert len(fwd) == 1, f"expected one transit forward, got {fwd}"
    d = fwd[0]
    assert (d.src_gpu, d.dst_gpus) == (2, (3,)), d
    assert d.release_band == 2, f"release must be inbound arrival + 1 = 2, got {d.release_band}"
    assert d.deadline_epoch == 2, f"deadline must be the outbound send = 2, got {d.deadline_epoch}"
    assert d.hard is True, "a network send waits on a forward, so it is hard"
    # And the band policy actually reads both: released in band 2, deadline 2, so band_of must
    # pull it back to band 1 rather than dropping it in the prologue.
    assert release_of(d) == 2, release_of(d)
    # A native staging in the same cell still gets the prologue -- the override is per demand.
    native = [x for x in res.intra_demands if x.kind == "egress_stage" and x.cell == _A]
    assert all(release_of(x) == PROLOGUE_BAND for x in native), native
    print("  [10d] a non-co-located forward carries release=arrival+1 AND deadline=send, and "
          "bands.release_of honours it OK")


def test_transit_legs_share_subchunk_index():
    """Both legs of one route must carry the SAME sub-chunk index out of _emit_refined, and the
    cursor == q partition check -- this module's integrity anchor -- must still hold.

    This is what a leg-keyed design could not do: sub-chunk indices are allocated per (identity,
    dst_cell), so independently assigned legs would have to chain indices across separately
    refined groups.
    """
    a = _transit_assignment(leg_gpus=((0, 2), (2, 4)))
    res = _emit_refined([a], {((0, 0), _C): (4, 5)}, q=1,
                        scale=ChunkScale(bytes_per_chunk=1, num_chunks=1), coarse_epoch=0.02)
    assert len(res.pieces) == 2, f"a two-leg route must emit two flows, got {len(res.pieces)}"
    assert res.pieces[0].identity == res.pieces[1].identity, res.pieces
    assert (res.pieces[0].src_cell, res.pieces[0].dst_cell) == (_A, _B)
    assert (res.pieces[1].src_cell, res.pieces[1].dst_cell) == (_B, _C)
    # Fractional split: two half-volume assignments over the same route must still partition the
    # identity exactly, and every leg of both must share its own half's index.
    halves = [_transit_assignment(leg_gpus=((0, 2), (2, 4))),
              _transit_assignment(leg_gpus=((1, 2), (2, 5)))]
    halves = [replace(h, volume=0.5, exact=Fraction(1, 2)) for h in halves]
    res2 = _emit_refined(halves, {((0, 0), _C): (4, 5)}, q=2,
                         scale=ChunkScale(bytes_per_chunk=1, num_chunks=1), coarse_epoch=0.02)
    assert len(res2.pieces) == 4, len(res2.pieces)
    by_sub = defaultdict(list)
    for p in res2.pieces:
        by_sub[p.identity].append((p.src_cell, p.dst_cell))
    assert len(by_sub) == 2, f"q=2 must produce 2 sub-chunks, got {sorted(by_sub)}"
    for ident, legs in by_sub.items():
        assert sorted(legs) == [(_A, _B), (_B, _C)], f"{ident}: legs {legs}"
    print("  [10e] both legs of a route share one sub-chunk index; the cursor == q partition "
          "check holds for whole and split volumes OK")


def test_saturated_bridge_is_refused():
    """The egress ledger must refuse a forward the bridge GPU has no room for, rather than
    silently oversubscribing it.

    Co-location makes this a real risk rather than a theoretical one: ALL of a cell's transit
    volume is forced onto the GPUs that bridge the two islands, so a bridge that is wide enough in
    aggregate can still be too narrow per epoch. And nothing upstream covers it -- `_build_slots`'
    proportional split bounds a FIRST leg's gateway by construction, but no such split exists for
    a forwarded leg, because the coarse level cannot see it. The ledger is the only accounting
    there is.

    Driven straight at `_place_downstream_legs` with a shared ledger rather than through a whole
    schedule: it is the second route landing on an already-full bridge that must be refused, and a
    hand-built pair says that without depending on how a coarse solve happened to pack epochs.
    (Routed through `resolve_identities`, an oversubscription like this would also trip the
    downstream `_assert_rate_within_capacity` -- but that is a last-line check on emitted flows,
    not the accounting that is supposed to catch it first.)
    """
    topo = BridgedIslandsCluster(TopologyParams(name="BridgedIslandsCluster", chunk_size=1))
    _, m = abstract(topo)
    epoch_duration = 0.02                      # 100 GB/s x 0.02 s = 2 chunk-units per epoch
    ingress_ledger, egress_ledger = {}, {}
    route = _transit_assignment(leg_gpus=((0, 2), (2, 4))).route

    def place(volume):
        return _place_downstream_legs(
            route, (0, 0), first_egress_gpu=0, volume=volume, target_gpus={4, 5},
            mapping=m, fine_topology=topo, epoch_duration=epoch_duration,
            ingress_ledger=ingress_ledger, egress_ledger=egress_ledger,
            tol=1e-6, max_denom=64)

    placed = place(2.0)                        # fills the bridge's outbound link for that epoch
    assert placed == [(((0, 2), (2, 4)), 2.0)], placed
    try:
        place(2.0)                             # a second route wanting the same epoch
    except RuntimeError as e:
        msg = str(e)
        # Reported by the JOINT budget, which is what actually binds: the placement never picks a
        # bridge without outgoing room, so the shortage surfaces at the choice rather than at the
        # commit afterwards. The message must name both budgets, because "no room" on a bridge is
        # ambiguous otherwise -- the landing side may be wide open.
        assert "transit capacity exhausted" in msg, msg
        assert "jointly bounded by the outgoing leg" in msg, msg
        print("  [10f] a second route whose co-located bridge GPU is already saturated is "
              "refused by the joint budget, naming both sides OK")
        return
    raise AssertionError("an oversubscribed bridge GPU must be refused")


def test_transit_needs_both_budgets_on_one_gpu():
    """A bridging GPU with landing room but NO forwarding room must not be chosen, and the piece
    must spread onto bridges that have both.

    This is the failure the dual-plane AllGather hit for real. A transit piece needs a MATCHED
    pair of budgets on ONE GPU -- it lands and forwards on the same one -- while an ordinary
    delivery needs only ingress room anywhere. Treating the outgoing side as a `preferred` hint is
    not enough: a hint is abandoned whenever no preferred candidate has ingress room, and the
    piece then lands on a GPU that cannot forward it. The two budgets have to intersect.

    Built with two bridges, one of them pre-saturated on its OUTGOING link only, so the ingress
    side alone would happily choose it (it is the roomiest landing).
    """
    topo = BridgedIslandsCluster(TopologyParams(name="BridgedIslandsCluster", chunk_size=1))
    _, m = abstract(topo)
    route = _transit_assignment(leg_gpus=((0, 2), (2, 4))).route
    inbound, outbound = route.legs
    bridges = _bridging_gpus(1, inbound, outbound, m)
    assert bridges == [2], f"this fixture has one bridge; got {bridges}"

    # One bridge in this topology, so drive _pick_ingress directly with a hand-made pair: gpu 2 is
    # untouched on ingress but full on egress, gpu 3 has room on both.
    ingress_ledger = {}
    picks = _pick_ingress(
        inbound, [2, 3], (0, 0), volume=1.0, preferred=set(),
        epoch_capacity={2: 2.0, 3: 1.0}, ledger=ingress_ledger,
        tol=1e-6, room_limit={2: 0.0, 3: 1.0}, what="transit")
    assert picks == [(3, 1.0)], (
        f"a bridge with no outgoing room must not be chosen even though it has the most landing "
        f"room; got {picks}")
    # And when neither alone suffices, the piece spreads across both rather than being dropped on
    # one and rejected afterwards.
    picks = _pick_ingress(
        inbound, [2, 3], (1, 0), volume=1.0, preferred=set(),
        epoch_capacity={2: 2.0, 3: 2.0}, ledger={},
        tol=1e-6, room_limit={2: 0.5, 3: 0.5}, what="transit")
    assert sorted(picks) == [(2, 0.5), (3, 0.5)], picks
    print("  [10h] a bridge with landing room but no forwarding room is never chosen; the piece "
          "spreads across bridges that have both OK")


def test_cyclic_route_rejected():
    """A route that revisits a cell must fail loud. dig_to_source carries a path-local
    cycle-avoidance band-aid rather than a real time-expanded flow decomposition, so a cyclic path
    is a live possibility -- and one would ask a child level to forward bytes whose arrival
    depends on a later leg of itself."""
    legs = (
        _CoarsePiece(src_cell=_A, dst_cell=_B, egress_neighbor=_T0, ingress_neighbor=_T0,
                     via_switches=(9,), volume=1.0, send_epoch=0, arrival_epoch=0,
                     origin=(_A, _A)),
        _CoarsePiece(src_cell=_B, dst_cell=_A, egress_neighbor=_T0, ingress_neighbor=_T0,
                     via_switches=(9,), volume=1.0, send_epoch=1, arrival_epoch=1,
                     origin=(_A, _A)),
    )
    try:
        _validate_route(_CoarseRoute(origin=(_A, _A), legs=legs, volume=1.0))
    except AssertionError as e:
        assert "revisits a cell" in str(e), e
        print("  [10g] a route revisiting a cell is rejected OK")
        return
    raise AssertionError("a cyclic route must be rejected")


def main() -> None:
    test_identity_sets_match_coarsify()
    test_hostB_forced_relay()
    test_zero_relay_symmetric()
    test_alltoall_single_target()
    test_replay_real_allgather_json()
    test_lexicographic_egress_dominates()
    test_ingress_prefers_target()
    test_level_chunk_unit_invariance()
    test_transit_route_both_legs_one_identity()
    test_colocated_transit_costs_no_intra_work()
    test_split_bridge_fails_loud()
    test_transit_forward_release_and_deadline()
    test_transit_legs_share_subchunk_index()
    test_saturated_bridge_is_refused()
    test_transit_needs_both_budgets_on_one_gpu()
    test_cyclic_route_rejected()
    print("identity resolution structural tests OK")


if __name__ == "__main__":
    main()
