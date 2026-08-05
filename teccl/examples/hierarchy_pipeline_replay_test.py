"""
Gurobi-free END-TO-END replay of the hierarchical pipeline below the coarse solve.

Replays a REAL solved coarse LP (Schedules/coarse_hetero_{coll}_lp.json) back into the
per_chunk_flow_paths structure, then runs the two stages that consume it -- identity resolution
(teccl.hierarchy.reconstruct) and the phase-3 intra-cell NVSwitch schedule
(teccl.hierarchy.intra_solve) -- and checks the invariants that hold across BOTH stages. The
per-stage structural oracles live in hierarchy_identity_resolution_test / hierarchy_intra_solve_test;
this file is about the seam between them, which is where volume bookkeeping goes wrong.

Checked here:

  1. Sub-chunk refinement: every piece and every intra demand carries volume exactly 1.0, and the
     ChunkScale conserves the per-GPU payload. Nothing below identity resolution should ever hold a
     fractional volume -- that is what keeps the volume-MERGING steps (reconstruct's
     _coalesce_egress, intra_solve's _add_direct, both max() over an (identity, src, dst) key) from
     silently combining two disjoint byte ranges of one identity.

  2. DELIVERY COVERAGE: every (identity, gpu) some demand asks for actually receives at least the
     demanded volume once phase-3 has scheduled it. This is the check that catches the merge class
     of bug directly, and it is deliberately independent of HOW the fan-out was lowered. It was
     previously satisfied only by luck: the two 0.5 shares of a split identity collided on
     _add_direct's dedup key and would have been merged down to 0.5 -- half the chunk never
     arriving -- except that the density test happened to route them through the tree branch, which
     is not deduped. One load-balance tweak away from silent data loss, with no assertion anywhere.

  3. Fine-link capacity per coarse epoch, on BOTH ends. Egress holds by construction (each piece is
     split across gateways proportionally to fine capacity). Ingress did NOT: a coarse link's
     capacity is the SUM of the fine downlinks behind it, and the landing GPU used to be
     boundary_gpu[...][0] unconditionally, so one downlink ran at 150% while its sibling sat idle.

  4. The phase-3 feasibility certificate: peak rounds per (cell, gap) <= m, the number of rounds a
     coarse epoch can hold. Both the round count and m scale with the refinement Q (port_cap = 1.0
     DEFINES a round, so refinement changes a round's duration, not its capacity), so this also
     doubles as a scale-invariance check.

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_pipeline_replay_test
"""
import collections
import json
import os

from teccl.examples.hierarchy_identity_resolution_test import (
    _fake_solver, _replay_per_chunk_flow_paths,
)
from teccl.hierarchy.abstract import abstract
from teccl.hierarchy.intra_solve import schedule_cell
from teccl.hierarchy.reconstruct import resolve_identities
from teccl.input_data import Collective, TopologyParams
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster

COARSE_EPOCH = 0.02          # matches the hetero coarse solve (1 GB chunk / 50 GB/s slowest uplink)


def _check_refinement(res, topo, fine_chunks: int) -> None:
    assert res.scale is not None and res.subdivision >= 1, (res.scale, res.subdivision)
    bad_p = sorted({p.volume for p in res.pieces if abs(p.volume - 1.0) > 1e-9})
    bad_d = sorted({d.volume for d in res.intra_demands if abs(d.volume - 1.0) > 1e-9})
    assert not bad_p, f"fractional piece volumes survived refinement: {bad_p}"
    assert not bad_d, f"fractional demand volumes survived refinement: {bad_d}"
    # Refinement re-denominates the payload; it must not create or destroy bytes. The per-GPU
    # payload is chunk_size * chunks-per-GPU, which is 1 for AllGather but num_gpus for AllToAll
    # (one chunk per ordered pair), so it has to come from the demand rather than be assumed.
    expected = topo.chunk_size * fine_chunks
    assert abs(res.scale.payload_per_gpu - expected) < 1e-9, (res.scale, expected)
    assert res.scale.refinement_from_root == res.subdivision, (res.scale, res.subdivision)


def _check_link_capacity(res, topo) -> tuple:
    """Per (fine link, coarse epoch) occupancy in absolute GB, both directions."""
    eg = collections.defaultdict(float)
    ing = collections.defaultdict(float)
    for p in res.pieces:
        gb = p.volume * res.scale.bytes_per_chunk
        eg[(p.egress_gpu, p.via_switches[0], p.send_epoch)] += gb
        ing[(p.ingress_gpu, p.via_switches[-1], p.arrival_epoch)] += gb
    over, worst_e, worst_i = [], 0.0, 0.0
    for (g, sw, k), vol in sorted(eg.items()):
        cap = topo.capacity[g][sw] * COARSE_EPOCH
        worst_e = max(worst_e, vol / cap)
        if vol > cap + 1e-9:
            over.append(("egress", g, sw, k, round(vol, 4), round(cap, 4)))
    for (h, sw, k), vol in sorted(ing.items()):
        cap = topo.capacity[sw][h] * COARSE_EPOCH
        worst_i = max(worst_i, vol / cap)
        if vol > cap + 1e-9:
            over.append(("ingress", h, sw, k, round(vol, 4), round(cap, 4)))
    assert not over, f"fine link oversubscribed within a coarse epoch: {over[:6]}"
    return worst_e, worst_i


def _check_delivery_coverage(res, flows) -> int:
    """Every (identity, gpu) a demand asks for must actually receive it.

    Required volume is the MAX over the demands wanting it, not the sum: one physical delivery
    satisfies every demand asking for that identity at that GPU (an egress_stage relay 5->4 and a
    self_distribution 5->4 are the same send), which is exactly what the dedup in _to_jobs exploits.
    Delivered volume is the total inbound over all intra flows, so a relay chain (tree fan-out)
    counts the same as a direct send."""
    required = collections.defaultdict(float)
    for d in res.intra_demands:
        for t in d.dst_gpus:
            if t != d.src_gpu:
                required[(d.identity, t)] = max(required[(d.identity, t)], d.volume)
    delivered = collections.defaultdict(float)
    for f in flows:
        # A flow may carry several co-travelling sub-chunks as ONE transfer (intra_solve's
        # _coalesce_subchunks), so credit each sub-chunk it moves -- `f.identity` is only the
        # representative, and counting it alone would report the rest as never delivered.
        for identity in f.identities:
            delivered[(identity, f.receiver)] += f.volume / len(f.identities)
    short = {k: (round(v, 6), round(delivered.get(k, 0.0), 6))
             for k, v in required.items() if delivered.get(k, 0.0) < v - 1e-6}
    assert not short, (
        f"{len(short)} (identity, gpu) pairs under-delivered (required, delivered): "
        f"{dict(list(short.items())[:6])}")
    return len(required)


def _check_intra_fits_epoch(res, flows, topo) -> tuple:
    nvlink_bw = max(max(row) for row in topo.capacity)
    delta = res.scale.epoch_duration(nvlink_bw)
    m = COARSE_EPOCH / delta
    per_gap = collections.defaultdict(int)
    for f in flows:
        per_gap[(f.cell, f.band)] = max(per_gap[(f.cell, f.band)], f.local_round + f.span)
    peak = max(per_gap.values(), default=0)
    hot = max(per_gap, key=lambda k: per_gap[k]) if per_gap else None
    assert peak <= m + 1e-9, (
        f"intra work does not fit a coarse epoch: {peak} rounds > m={m:.1f} at cell/gap {hot}")
    return peak, m, hot


def replay(tag: str, collective: Collective) -> bool:
    path = f"Schedules/coarse_hetero_{tag}_lp.json"
    if not os.path.exists(path):
        print(f"  SKIP {tag}: {path} not found")
        return False
    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    coarse, mapping = abstract(topo)
    n_participating = sum(len(c.gpus) for c in mapping.coarse_cells.values())
    # AllToAll lays down fine_chunks // num_gpus chunks per ordered pair, so the flat
    # num_chunks must be pre-scaled -- mirrors hierarchy_coarse_solve_hetero's CHUNKS_PER_PAIR.
    fine_chunks = 1 if collective == Collective.ALLGATHER else n_participating
    fine_demand = build_demand(collective, topo, fine_chunks)

    with open(path) as f:
        schedule_json = json.load(f)
    solver = _fake_solver(_replay_per_chunk_flow_paths(schedule_json),
                          switch_indices=list(coarse.switch_indices))
    res = resolve_identities(solver, mapping, fine_demand, topo)

    _check_refinement(res, topo, fine_chunks)
    worst_e, worst_i = _check_link_capacity(res, topo)

    by_cell = collections.defaultdict(list)
    for d in res.intra_demands:
        by_cell[d.cell].append(d)
    flows = []
    for cid in sorted(mapping.coarse_cells):
        if by_cell.get(cid):
            flows += schedule_cell(cid, mapping.coarse_cells[cid], by_cell[cid],
                                   switch_copy=False, debug=False,
                                   subdivision=res.subdivision)

    n_required = _check_delivery_coverage(res, flows)
    peak, m, hot = _check_intra_fits_epoch(res, flows, topo)

    ingress_used = collections.Counter(p.ingress_gpu for p in res.pieces)
    single = [d for d in res.intra_demands
              if d.kind == "ingress_distribution" and len(d.dst_gpus) == 1]
    landed = sum(1 for d in single if d.dst_gpus[0] == d.src_gpu)
    relayed = sum(1 for p in res.pieces if p.egress_gpu != p.identity[0])

    print(f"  {tag}: Q={res.subdivision} {res.scale}")
    print(f"    {len(res.pieces)} pieces, {len(res.intra_demands)} intra demands, "
          f"{len(flows)} fine intra flows")
    print(f"    fine-link peak occupancy: egress {100 * worst_e:.0f}%, ingress {100 * worst_i:.0f}%"
          f"  (no violations)")
    print(f"    ingress gateways used: {dict(sorted(ingress_used.items()))}")
    print(f"    delivery coverage: all {n_required} (identity, gpu) pairs satisfied")
    print(f"    intra fits epoch: peak {peak} rounds vs m={m:.0f} "
          f"({100 * peak / m:.0f}% of budget) at cell/gap {hot}")
    if single:
        print(f"    ingress landing on target: {landed}/{len(single)} "
              f"({100 * landed / len(single):.0f}%)")
    print(f"    non-native egress: {relayed}/{len(res.pieces)} pieces")
    return True


def main() -> None:
    print("hierarchical pipeline replay (identity resolution + phase-3, Gurobi-free)")
    ran = False
    for tag, coll in (("allgather", Collective.ALLGATHER), ("alltoall", Collective.ALLTOALL)):
        ran |= replay(tag, coll)
    if not ran:
        print("no coarse LP schedules found under Schedules/ -- run the coarse solve first")
        return
    print("pipeline replay OK")


if __name__ == "__main__":
    main()
