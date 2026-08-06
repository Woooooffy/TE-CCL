"""
Gurobi-free tests for PHASE 4, the stitch (teccl.hierarchy.stitch).

Two layers, deliberately separate:

  UNIT, on hand-built input -- band assignment and the flattening, where an oracle can be written
  by hand:
    1. _assign_bands(): a job goes to the band its data is READY in, and only work that must
       precede coarse epoch 0's sends falls into the prologue. This is what makes the prologue
       empty on a topology with no forced relay and non-empty on one that has them.
    2. Epoch layout monotonicity: every staging relay lands strictly before the network send it
       feeds; every fan-out lands at or after its piece is held; the prologue precedes everything.
       This is the property the whole banding design rests on, and a wrong band offset breaks it
       silently.
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

from math import inf

import numpy as np

from teccl.hierarchy.abstract import abstract
from teccl.hierarchy.intra_solve import (PROLOGUE_BAND, IntraFlow, _assign_bands, _Job,
                                         schedule_cell)
from teccl.hierarchy.reconstruct import (IdentityResolution, IntraCellDemand, ResolvedPiece,
                                         resolve_identities)
from teccl.hierarchy.scale import ChunkScale
from teccl.hierarchy.stitch import (NETWORK, PROLOGUE, DeliveryRecord, _chunk_label,
                                    back_trace, build_records, derive_grid, stitch)
from teccl.input_data import Collective, TopologyParams
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster

COARSE_EPOCH = 0.02


def _flow(cell, identity, src, dst, band, rnd, kind, hard=False, switch=99):
    return IntraFlow(cell=cell, identity=identity, sender=src, receiver=dst, via_switch=switch,
                     volume=1.0, band=band, local_round=rnd, kind=kind, hard=hard)


def _job(src, dst, release, deadline, hard, kind):
    return _Job(identity=(src, 0), src=src, dst=dst, volume=1.0, release_gap=release,
                deadline_gap=deadline, hard=hard, kind=kind)


# ------------------------------------------------------------------------------------------
def test_band_assignment():
    """A job runs in the band its data is ready; only work that must precede epoch 0 is prologue."""
    # staging feeding a send in coarse epoch 3: native data, so ready at 0 -- it does NOT wait
    # until band 2 just because its deadline allows it.
    assert list(_assign_bands([_job(5, 4, PROLOGUE_BAND, 3, True, "egress_stage")])) == [0]
    # staging feeding the FIRST send has nowhere to go inside band 0 (the send is its leading
    # edge), so it is the prologue case.
    assert list(_assign_bands([_job(5, 4, PROLOGUE_BAND, 0, True, "egress_stage")])) == [PROLOGUE_BAND]
    # a fan-out of a piece arriving in epoch 4 is ready in band 5 (_to_jobs sets release=arrival+1)
    assert list(_assign_bands([_job(4, 5, 5, inf, False, "ingress_distribution")])) == [5]
    # native self-distribution is ready immediately
    assert list(_assign_bands([_job(0, 1, PROLOGUE_BAND, inf, False, "self_distribution")])) == [0]

    # a cell with no forced relay has an EMPTY prologue -- the property that makes this cheap on
    # rail-optimized topologies, where every gateway owns the data it sends.
    no_relay = [_job(0, 1, PROLOGUE_BAND, inf, False, "self_distribution"),
                _job(4, 5, 2, inf, False, "ingress_distribution")]
    assert PROLOGUE_BAND not in _assign_bands(no_relay)

    # host transit -- data that arrives in band b and must be forwarded before band b -- has no
    # band and must fail loud rather than be silently placed.
    try:
        _assign_bands([_job(4, 5, 3, 3, True, "egress_stage")])
        raise SystemExit("_assign_bands accepted an unschedulable job")
    except RuntimeError as e:
        assert "no band it can run in" in str(e), e
    print("  [1] _assign_bands: readiness placement, prologue only when forced, empty when not OK")


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
        # prologue: 2 rounds of work that must precede the first network send -> W = 2
        _flow(0, (3, 0), 3, 2, PROLOGUE_BAND, 0, "egress_stage", hard=True),
        _flow(0, (3, 0), 2, 1, PROLOGUE_BAND, 1, "egress_stage", hard=True),
        # staging for the epoch-2 send, ready immediately -> band 0
        _flow(0, (0, 0), 0, 1, 0, 0, "egress_stage", hard=True),
        # fan-out of the arrival: ready in band 3, two rounds deep
        _flow(1, (0, 0), 4, 5, 3, 0, "ingress_distribution"),
        _flow(1, (0, 0), 5, 6, 3, 1, "ingress_distribution"),
    ]
    m = 8
    recs, W = build_records(res, flows, m)
    assert W == 2, W
    net = next(r for r in recs if r.phase == NETWORK)
    pro = sorted((r.epoch for r in recs if r.phase == PROLOGUE))
    stage = next(r for r in recs if r.phase == "egress_stage")
    fan = sorted(r.epoch for r in recs if r.phase == "ingress_distribution")

    # the prologue is [0, W) and everything else is after it
    assert pro == [0, 1], pro
    assert net.epoch == W + m * 2, (net.epoch, W, m)
    # staging sits in band 0, well before the send it feeds
    assert stage.epoch == W and stage.epoch < net.epoch, (stage.epoch, net.epoch)
    # the piece is held at the leading edge of the band after its arrival epoch, and its fan-out
    # starts there -- at, not after, since "held at epoch E" means available from the start of E.
    assert net.completion == W + m * 3, (net.completion, W, m)
    assert fan == [net.completion, net.completion + 1], (fan, net.completion)
    print(f"  [2] epoch layout: prologue{pro} < stage({stage.epoch}) < send({net.epoch}) "
          f"-> held({net.completion}) <= fan-out{fan} OK")


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
    #     of IntraFlows: _schedule_band gives 1->2 a strictly later round than 0->1 precisely
    #     because it enforces the precedence, so build_records already separates them. The records
    #     are therefore constructed directly -- back_trace has to stand on its own as the verifier,
    #     not lean on the schedule that feeds it, since a future level could get this wrong.
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
    good = [_flow(0, (0, 0), 0, 1, PROLOGUE_BAND, 0, "self_distribution"),
            _flow(0, (0, 0), 1, 2, PROLOGUE_BAND, 1, "self_distribution")]
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
            flows += schedule_cell(cid, mapping.coarse_cells[cid], by_cell[cid], debug=False,
                                   subdivision=res.subdivision)

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
    algo, _, _, _, piece_rate, pacing_gates = build_algorithm(info)
    assert isinstance(piece_rate, dict) and piece_rate, (
        "the stitched schedule must supply per-flow rates, not a global scalar")
    unpaced = sum(1 for v in piece_rate.values() if v is None)
    return (f"check_implements OK, {len(algo.steps)} steps, "
            f"{len(piece_rate)} paced flows ({unpaced} unpaced), "
            f"{len(pacing_gates)} pacing gates")


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
    # One record per sub-chunk, but a coalesced flow carries several of them -- so records track
    # the sub-chunks moved, not the transfers, and the gap between the two is the merge that
    # ncclize turns into cnt=Q ops.
    assert len(intra) == sum(len(f.identities) for f in flows), (len(intra), len(flows))
    # A coalesced transfer only pays off if ncclize can merge its sub-chunks, which needs BOTH
    # conditions its make_intervals checks: consecutive chunk labels and a shared step. Assert both
    # -- they are the entire point of the coalescing, and either one silently failing just returns
    # the op count to where it was.
    coalesced = [f for f in flows if len(f.identities) > 1]
    # Records grouped the way ncclize groups sends: one bucket per (step, edge).
    buckets = collections.defaultdict(set)
    for r in intra:
        buckets[(r.epoch, r.sender, r.receiver)].add(r.identity)
    for f in coalesced:
        labels = [ci for _s, ci in f.identities]
        assert labels == list(range(labels[0], labels[0] + len(labels))), (
            f"coalesced sub-chunk labels {labels} are not consecutive; ncclize cannot merge them")
        # All of this transfer's sub-chunks must land in ONE bucket. (Scoped to the transfer, not
        # to the edge: the same identity can also be delivered over the same edge by a separate
        # flow -- see the tree/direct duplicate noted below -- and that is a different transfer.)
        assert any(set(f.identities) <= ids for ids in buckets.values()), (
            f"sub-chunks of one transfer {f.sender}->{f.receiver} did not share a step")

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
    test_band_assignment()
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
