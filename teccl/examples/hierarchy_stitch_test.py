"""
Gurobi-free tests for PHASE 4, the stitch (teccl.hierarchy.stitch).

Two layers, deliberately separate:

  UNIT, on hand-built input -- the epoch layout and the precedence-level projection, where an
  oracle can be written down by hand:
    1. levels(): a binomial tree has depth ceil(log2 n); a ring/direct fan-out is all zeros.
    2. Epoch layout monotonicity: every staging relay lands strictly before the network send it
       feeds; every fan-out lands strictly after its piece arrives; the prologue precedes
       everything. This is the property the whole "fine epochs are a numbering convention" design
       rests on, and it is the one thing a wrong band offset breaks silently.
    3. Causality/coverage: back_trace() rejects a schedule where a relay sends data it does not
       hold yet, and rejects one where a demand is never delivered.
    4. Chunk labels follow the COLLECTIVE's addressing, not the refinement's: a dst-major
       (alltoall) label keeps the destination in the low digits under sub-chunk refinement. This is
       the one the naive `ci * Q + j` gets wrong, and ncclize rejects it downstream with a
       confusing "chunk label implies dense destination X" error.

  END-TO-END REPLAY, on the real solved schedules -- the stitch over both collectives, asserting
  the cross-stage invariants and (when lxml/taccl are importable) that ncclize's own
  check_implements() accepts the result. That last one is the strongest free check available: it
  independently verifies the schedule implements the collective.

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_stitch_test
"""
import collections
import json
import os
import sys

import numpy as np

from teccl.hierarchy.abstract import abstract
from teccl.hierarchy.intra_solve import IntraFlow, schedule_cell
from teccl.hierarchy.reconstruct import (IdentityResolution, IntraCellDemand, ResolvedPiece,
                                         resolve_identities)
from teccl.hierarchy.scale import ChunkScale
from teccl.hierarchy.stitch import (INGRESS, NETWORK, PROLOGUE, STAGE, DeliveryRecord,
                                    _chunk_label, back_trace, build_records, derive_grid, levels,
                                    stitch)
from teccl.input_data import Collective, TopologyParams
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster

COARSE_EPOCH = 0.02


def _flow(cell, identity, src, dst, gap, rnd, kind, hard=False, switch=99):
    return IntraFlow(cell=cell, identity=identity, sender=src, receiver=dst, via_switch=switch,
                     volume=1.0, gap=gap, local_round=rnd, kind=kind, hard=hard)


# ------------------------------------------------------------------------------------------
def test_levels_tree_and_ring():
    """A broadcast tree's depth is its forwarding depth; a direct fan-out has none."""
    # binomial tree rooted at 0 over {1,2,3}: 0->1 (r0), 0->2 / 1->3 (r1)
    tree = [_flow(0, (0, 0), 0, 1, 0, 0, "self_distribution"),
            _flow(0, (0, 0), 0, 2, 0, 1, "self_distribution"),
            _flow(0, (0, 0), 1, 3, 0, 1, "self_distribution")]
    lv = levels(tree)
    assert [lv[id(f)] for f in tree] == [0, 0, 1], [lv[id(f)] for f in tree]

    # direct fan-out: every edge leaves the source, nothing forwards
    ring = [_flow(0, (0, 0), 0, d, 0, r, "self_distribution") for r, d in enumerate((1, 2, 3))]
    assert set(levels(ring).values()) == {0}

    # a deeper chain 0->1->2->3 is depth 2
    chain = [_flow(0, (0, 0), 0, 1, 0, 0, "self_distribution"),
             _flow(0, (0, 0), 1, 2, 0, 1, "self_distribution"),
             _flow(0, (0, 0), 2, 3, 0, 2, "self_distribution")]
    assert sorted(levels(chain).values()) == [0, 1, 2]
    print("  [1] levels(): binomial tree depth, ring all-zero, chain 0/1/2 OK")


# ------------------------------------------------------------------------------------------
def test_epoch_layout_monotonicity():
    """Staging precedes its send; fan-out follows its arrival; the prologue precedes both."""
    scale = ChunkScale(bytes_per_chunk=1.0, num_chunks=1)
    res = IdentityResolution(scale=scale, subdivision=1)
    # one piece leaving in coarse epoch 2, arriving in epoch 2
    res.pieces.append(ResolvedPiece(
        src_cell=0, dst_cell=1, identity=(0, 0), egress_gpu=1, ingress_gpu=4,
        via_switches=(90,), volume=1.0, send_epoch=2, arrival_epoch=2, rate=50.0))
    flows = [
        _flow(0, (0, 0), 0, 1, 2, 0, "egress_stage", hard=True),   # stages that send
        _flow(0, (3, 0), 3, 2, 0, 0, "self_distribution"),          # prologue
        _flow(1, (0, 0), 4, 5, 2, 0, "ingress_distribution"),       # fan-out of the arrival
        _flow(1, (0, 0), 5, 6, 2, 1, "ingress_distribution"),       # ... one level deeper
    ]
    m = 8
    records, = (build_records(res, flows, m),)
    recs, P = records
    by_phase = {r.phase: r for r in recs if r.phase != INGRESS}
    ingress = sorted((r for r in recs if r.phase == INGRESS), key=lambda r: r.epoch)
    net = by_phase[NETWORK]

    assert by_phase[PROLOGUE].epoch < P, (by_phase[PROLOGUE].epoch, P)
    assert net.epoch == P + m * 2, (net.epoch, P, m)
    # staging sits in the band BEFORE the send, immediately ahead of it
    assert by_phase[STAGE].epoch == net.epoch - 1, (by_phase[STAGE].epoch, net.epoch)
    assert P + m * 1 <= by_phase[STAGE].epoch < net.epoch
    # fan-out sits in the band AFTER the arrival, and after the piece is held
    assert net.completion == P + m * 3, (net.completion, P, m)
    assert all(r.epoch > net.completion for r in ingress), (
        [r.epoch for r in ingress], net.completion)
    assert ingress[0].epoch < ingress[1].epoch, "a deeper level must sit later"
    print(f"  [2] epoch layout: prologue < stage({by_phase[STAGE].epoch}) < "
          f"send({net.epoch}) -> held({net.completion}) < fan-out{[r.epoch for r in ingress]} OK")


# ------------------------------------------------------------------------------------------
def test_back_trace_rejects_violations():
    """back_trace is the verifier, so it must actually reject bad input."""
    scale = ChunkScale(bytes_per_chunk=1.0, num_chunks=1)
    demand = np.zeros((3, 3, 1), dtype=np.int32)
    demand[0][2][0] = 1                       # GPU 2 wants GPU 0's chunk

    # (a) never delivered
    res = IdentityResolution(scale=scale, subdivision=1)
    recs, _ = build_records(res, [], 8)
    try:
        back_trace(recs, demand, 1)
        raise SystemExit("back_trace accepted an undelivered demand")
    except AssertionError as e:
        assert "never delivered" in str(e), e

    # (b) delivered, but the relay sends before it holds the data. Note this cannot be built out
    #     of IntraFlows: levels() gives 1->2 depth 1 precisely because 0->1 produced the data on
    #     GPU 1, so build_records already separates them. The records are therefore constructed
    #     directly -- back_trace has to stand on its own as the verifier, not lean on the layout
    #     that feeds it, since a future placement policy could get this wrong.
    def _rec(src, dst, epoch):
        return DeliveryRecord(identity=(0, 0), sender=src, receiver=dst, via_switches=(99,),
                              volume=1.0, epoch=epoch, completion=epoch + 1, rate=None,
                              phase=PROLOGUE, cell=0, level=0)

    try:
        back_trace([_rec(0, 1, 1), _rec(1, 2, 0)], demand, 1)
        raise SystemExit("back_trace accepted a send-before-hold")
    except AssertionError as e:
        assert "causality" in str(e), e

    # (c) the same two hops, correctly levelled, are accepted
    good = [_flow(0, (0, 0), 0, 1, 0, 0, "self_distribution"),
            _flow(0, (0, 0), 1, 2, 0, 1, "self_distribution")]
    recs, _ = build_records(res, good, 8)
    paths = back_trace(recs, demand, 1)
    assert len(paths[(2, (0, 0))]) == 2, paths
    print("  [3] back_trace rejects undelivered + send-before-hold, accepts the relay chain OK")


# ------------------------------------------------------------------------------------------
def test_chunk_label_addressing():
    """Refinement must not move a dst-major collective's destination out of the low digits."""
    N, Q = 4, 2
    # src-major: the refined index IS the label
    assert _chunk_label(7, Q, dst_major=False, num_gpus=N, dst_dense=3) == 7
    # dst-major: demand.py lays alltoall down as chunk = dst + c*N, so original chunk 6 is
    # (dst=2, c=1); its two sub-chunks must stay decodable as destination 2.
    for j, want in ((0, 2 + 2 * N), (1, 2 + 3 * N)):
        got = _chunk_label(6 * Q + j, Q, dst_major=True, num_gpus=N, dst_dense=2)
        assert got == want, (j, got, want)
        assert got % N == 2, "the destination must survive in the low digits"
    # and it fails loud if the label and the demand disagree
    try:
        _chunk_label(6 * Q, Q, dst_major=True, num_gpus=N, dst_dense=3)
        raise SystemExit("mismatched destination accepted")
    except AssertionError as e:
        assert "encodes destination" in str(e), e
    print("  [4] chunk labels: src-major passthrough, dst-major keeps the destination OK")


# ------------------------------------------------------------------------------------------
def _replay_pipeline(tag, collective):
    """Coarse LP JSON -> resolution -> phase 3 -> stitch, on the real hetero schedules."""
    from teccl.examples.hierarchy_identity_resolution_test import (
        _fake_solver, _replay_per_chunk_flow_paths)

    path = f"Schedules/coarse_hetero_{tag}_lp.json"
    if not os.path.exists(path):
        return None
    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    coarse, mapping = abstract(topo)
    n_gpus = sum(len(c.gpus) for c in mapping.coarse_cells.values())
    fine_chunks = 1 if collective == Collective.ALLGATHER else n_gpus
    fine_demand = build_demand(collective, topo, fine_chunks)

    with open(path) as f:
        schedule_json = json.load(f)
    solver = _fake_solver(_replay_per_chunk_flow_paths(schedule_json),
                          switch_indices=list(coarse.switch_indices))
    res = resolve_identities(solver, mapping, fine_demand, topo)

    by_cell = collections.defaultdict(list)
    for d in res.intra_demands:
        by_cell[d.cell].append(d)
    flows = []
    for cid in sorted(mapping.coarse_cells):
        if by_cell.get(cid):
            flows += schedule_cell(cid, mapping.coarse_cells[cid], by_cell[cid], debug=False)

    info, records = stitch(res, flows, topo, fine_demand, COARSE_EPOCH, tag)
    return topo, res, flows, info, records, fine_demand


def _check_ncclize(info, tag):
    """Round-trip through ncclize; its check_implements() independently validates the schedule."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ncclize")
    sys.path.insert(0, os.path.abspath(here))
    try:
        from teccl_ncclize import build_algorithm
    except ImportError as e:                     # lxml / z3 not installed locally
        return f"SKIPPED ({e})"
    algo, _, _, _, _, piece_rate = build_algorithm(info)
    assert isinstance(piece_rate, dict) and piece_rate, (
        "the stitched schedule must supply per-flow rates, not a global scalar")
    unpaced = sum(1 for v in piece_rate.values() if v is None)
    return (f"check_implements OK, {len(algo.steps)} steps, "
            f"{len(piece_rate)} paced flows ({unpaced} unpaced)")


def test_replay(tag, collective):
    out = _replay_pipeline(tag, collective)
    if out is None:
        print(f"  SKIP {tag}: Schedules/coarse_hetero_{tag}_lp.json not found")
        return False
    topo, res, flows, info, records, fine_demand = out

    delta, m = derive_grid(res.scale, topo, COARSE_EPOCH)
    # Refinement re-denominates the payload without creating or destroying bytes, and the reported
    # time must be exactly the grid it is derived from.
    fine_chunks = 1 if collective == Collective.ALLGATHER else len(fine_demand[0][0])
    assert abs(res.scale.payload_per_gpu - topo.chunk_size * fine_chunks) < 1e-9, (
        res.scale, fine_chunks)
    assert abs(info["9-Chunk_Size"] * res.scale.num_chunks
               - topo.chunk_size * fine_chunks) < 1e-9
    assert abs(info["4-Collective_Finish_Time"] - delta * info["3-Epochs_Required"]) < 1e-12

    # Every network record carries this level's pacing; no intra record does.
    net = [r for r in records if r.phase == NETWORK]
    intra = [r for r in records if r.phase != NETWORK]
    assert all(r.rate is not None for r in net), "a network send must be paced to the coarse epoch"
    assert all(r.rate is None for r in intra), "intra flows are deliberately unpaced"
    assert len(net) == len(res.pieces), (len(net), len(res.pieces))
    assert len(intra) == len(flows), (len(intra), len(flows))

    # Every demanded (dest, sub-identity) has a path; back_trace already proved causality.
    n = len(fine_demand)
    want = sum(1 for s in range(n) for t in range(n) for ci in range(len(fine_demand[s][t]))
               if fine_demand[s][t][ci] and s != t) * res.subdivision
    assert len(info["8-Chunk paths"]) == want, (len(info["8-Chunk paths"]), want)

    # SCALE INVARIANCE: refining the chunk halves delta and doubles m, and the NETWORK time
    # (Delta per coarse epoch) is then exactly invariant. What does move is the prologue and the
    # trailing fan-out: they occupy a fixed number of FINE epochs whatever m is, so they cost
    # O(delta) and shrink as the grid refines. The reported time therefore CONVERGES DOWN to the
    # coarse-grid time rather than sitting still, and the property to assert is that the residual
    # HALVES when the grid doubles -- which is what pins delta and m together. If one were ever
    # derived from a stale chunk size the residual would scale wrongly (or the time would jump to
    # ~0.22s, a plausible-looking number rather than an obvious bug).
    finish_at = {}
    for factor in (1, 2, 4):
        recs_f, _ = build_records(res, flows, m * factor)
        finish_at[factor] = (max(r.completion for r in recs_f) + 1) * (delta / factor)
    assert finish_at[1] > finish_at[2] > finish_at[4], (
        f"a finer grid must not report MORE time: {finish_at}")
    r1, r2 = finish_at[1] - finish_at[2], finish_at[2] - finish_at[4]
    assert abs(r1 / r2 - 2.0) < 0.25, (
        f"the grid-dependent residual does not halve when the grid doubles (ratio {r1 / r2:.3f}, "
        f"expected 2.0): {finish_at}. delta and m have desynced from the ChunkScale.")
    # And the whole residual is a couple of fine epochs' worth, not a structural offset.
    assert (finish_at[1] - finish_at[4]) < 0.02 * finish_at[1], finish_at

    print(f"  {tag}: Q={res.subdivision}, delta={delta:.3e}s, m={m}, "
          f"{info['3-Epochs_Required']} epochs -> {info['4-Collective_Finish_Time']:.4f}s, "
          f"bw={info['5-Algo_Bandwidth']:.1f}")
    print(f"    scale-invariant: finish at m x1/x2/x4 = "
          f"{'/'.join(f'{finish_at[f]:.4f}' for f in (1, 2, 4))}s "
          f"-> residual halves ({r1:.2e} then {r2:.2e}), converging to the coarse-grid time")
    print(f"    {len(net)} network + {len(intra)} intra records, {want} demands traced, "
          f"chunk={info['9-Chunk_Size']}")
    print(f"    ncclize: {_check_ncclize(info, tag)}")
    return True


def main() -> None:
    print("stitch (phase 4) tests")
    test_levels_tree_and_ring()
    test_epoch_layout_monotonicity()
    test_back_trace_rejects_violations()
    test_chunk_label_addressing()
    print("  --- end-to-end replay on the real coarse schedules ---")
    ran = False
    for tag, coll in (("allgather", Collective.ALLGATHER), ("alltoall", Collective.ALLTOALL)):
        ran |= bool(test_replay(tag, coll))
    if not ran:
        print("  (no coarse LP schedules under Schedules/ -- run the coarse solve first)")
    print("stitch tests OK")


if __name__ == "__main__":
    main()
