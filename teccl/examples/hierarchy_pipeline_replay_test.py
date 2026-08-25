"""
Gurobi-free END-TO-END replay of the hierarchical pipeline below the coarse solve.

Replays a REAL solved coarse LP (Schedules/coarse_hetero_{coll}_lp.json) back into the
per_chunk_flow_paths structure, then runs the two stages that consume it -- identity resolution
(teccl.hierarchy.reconstruct) and the phase-3 intra-cell NVSwitch schedule
(teccl.hierarchy.crossbar_solve) -- and checks the invariants that hold across BOTH stages. The
per-stage structural oracles live in hierarchy_identity_resolution_test / hierarchy_crossbar_solve_test;
this file is about the seam between them, which is where volume bookkeeping goes wrong.

Checked here:

  1. Sub-chunk refinement: every piece and every intra demand carries volume exactly 1.0, and the
     ChunkScale conserves the per-GPU payload. Nothing below identity resolution should ever hold a
     fractional volume -- that is what keeps the volume-MERGING steps (reconstruct's
     _coalesce_egress, crossbar_solve's _add_direct, both max() over an (identity, src, dst) key) from
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

  5. The same four, on a route that STORE-AND-FORWARDS through an intermediate cell
     (BridgedIslandsCluster). The cross-stage invariants are what a transit chain is most likely to
     break, because a delivery is now two network flows rather than one: coverage has to count a
     two-leg chain as arriving once, and both legs have to fit their own end's link budget. The
     hetero replay above cannot cover it -- no real solved schedule in this repo contains a
     transit, which is exactly why the case stayed dormant.

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_pipeline_replay_test
"""
import collections
import json
import os

from teccl.examples.hierarchy_identity_resolution_test import (
    _bridged_paths, _fake_solver, _replay_per_chunk_flow_paths,
)
from teccl.hierarchy.abstract import abstract
from teccl.hierarchy.crossbar_solve import band_rounds, schedule_cell
from teccl.hierarchy.reconstruct import resolve_identities
from teccl.hierarchy.flatten import aligned_band, derive_grid
from teccl.input_data import Collective, TopologyParams
from teccl.solvers.demand import build_demand
from teccl.topologies.bridged_islands_cluster import BridgedIslandsCluster
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster

def _coarse_epoch(schedule_json) -> float:
    """Read the coarse epoch off the schedule being replayed rather than restating it. It is a
    property of the solve that produced the file, and it moves as soon as a level is re-expressed
    in its own chunk unit (abstract.set_level_chunk) -- a stale literal would silently put the
    capacity budgets and the epoch grid below on a different scale than the flows they bound."""
    return float(schedule_json["1-Epoch_Duration"])


def _level_chunk(schedule_json, fine_topology) -> int:
    """How many FINE chunks one unit of this schedule's volume is worth.

    A solved schedule records the chunk it was solved in as "9-Chunk_Size", so a coarsened level
    (abstract.set_level_chunk) is self-describing and a replay needs no out-of-band knowledge.
    Schedules written before coarsening existed carry the fine chunk size and yield 1, which is an
    exact no-op -- so old and new fixtures both replay correctly through the same path."""
    g = float(schedule_json.get("9-Chunk_Size", fine_topology.chunk_size)) / fine_topology.chunk_size
    assert abs(g - round(g)) < 1e-9, f"schedule chunk {g}x the fine chunk is not a whole multiple"
    return int(round(g))


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


def _check_link_capacity(res, topo, coarse_epoch) -> tuple:
    """Per (fine link, coarse epoch) occupancy in absolute GB, both directions."""
    eg = collections.defaultdict(float)
    ing = collections.defaultdict(float)
    for p in res.pieces:
        gb = p.volume * res.scale.bytes_per_chunk
        eg[(p.egress_gpu, p.via_switches[0], p.send_epoch)] += gb
        ing[(p.ingress_gpu, p.via_switches[-1], p.arrival_epoch)] += gb
    over, worst_e, worst_i = [], 0.0, 0.0
    for (g, sw, k), vol in sorted(eg.items()):
        cap = topo.capacity[g][sw] * coarse_epoch
        worst_e = max(worst_e, vol / cap)
        if vol > cap + 1e-9:
            over.append(("egress", g, sw, k, round(vol, 4), round(cap, 4)))
    for (h, sw, k), vol in sorted(ing.items()):
        cap = topo.capacity[sw][h] * coarse_epoch
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
        # A flow may carry several co-travelling sub-chunks as ONE transfer (crossbar_solve's
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


def _check_intra_fits_epoch(res, flows, topo, coarse_epoch) -> tuple:
    # delta/m from the stitch's own derive_grid, so this check and the axis the stitch lays out
    # cannot drift apart (they were two independent copies of the same arithmetic).
    # band_rounds/aligned_band are the stitch's too, for the same reason: only bands PINNED to a
    # network send are bounded by m. The prologue and epilogue have none to hide under and are
    # charged their true length, so including them here would assert something stronger than the
    # property the stitch actually guarantees.
    delta, m = derive_grid(res.scale, topo, coarse_epoch)
    num_coarse_epochs = max((p.send_epoch for p in res.pieces), default=-1) + 1
    bounded = {k: r for k, r in band_rounds(flows).items()
               if aligned_band(k[1], num_coarse_epochs)}
    peak = max(bounded.values(), default=0)
    hot = max(bounded, key=lambda k: bounded[k]) if bounded else None
    assert peak <= m + 1e-9, (
        f"intra work does not fit a coarse epoch: {peak} rounds > m={m:.1f} at cell/band {hot}")
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
    coarse_epoch = _coarse_epoch(schedule_json)
    solver = _fake_solver(_replay_per_chunk_flow_paths(schedule_json),
                          switch_indices=list(coarse.switch_indices),
                          epoch_duration=coarse_epoch)
    res = resolve_identities(solver, mapping, fine_demand, topo,
                             level_chunk=_level_chunk(schedule_json, topo))

    _check_refinement(res, topo, fine_chunks)
    worst_e, worst_i = _check_link_capacity(res, topo, coarse_epoch)

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
    peak, m, hot = _check_intra_fits_epoch(res, flows, topo, coarse_epoch)

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




def _run_stages(topo, mapping, res, coarse_epoch, fine_chunks):
    """The cross-stage half of `replay`, factored out so the transit fixture runs exactly the same
    checks rather than a parallel set that could drift from them."""
    _check_refinement(res, topo, fine_chunks)
    worst_e, worst_i = _check_link_capacity(res, topo, coarse_epoch)
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
    peak, m, hot = _check_intra_fits_epoch(res, flows, topo, coarse_epoch)
    return flows, worst_e, worst_i, n_required, peak, m, hot


def replay_transit() -> bool:
    """The cross-stage invariants on a coarse solution containing a HOST TRANSIT.

    Needs no checked-in schedule: the coarse flow is the hand-built one from the resolution test,
    which is the only way to get a transit at all -- no solved schedule in this repo has one.
    """
    topo = BridgedIslandsCluster(TopologyParams(name="BridgedIslandsCluster", chunk_size=1))
    coarse, mapping = abstract(topo)
    coarse_epoch = 0.02
    fine_demand = build_demand(Collective.ALLGATHER, topo, 1)
    solver = _fake_solver(_bridged_paths(), switch_indices=list(coarse.switch_indices),
                          epoch_duration=coarse_epoch)
    res = resolve_identities(solver, mapping, fine_demand, topo)

    flows, worst_e, worst_i, n_required, peak, m, hot = _run_stages(
        topo, mapping, res, coarse_epoch, fine_chunks=1)

    # The transit-specific structure, on top of the shared checks: the two legs of each
    # store-and-forward delivery are one commodity, and cell B (the bridge) really is relaying
    # traffic it neither produces nor wants.
    legs_by_id = collections.defaultdict(list)
    for pc in res.pieces:
        legs_by_id[pc.identity].append(pc)
    chains = {i: sorted(ps, key=lambda p: p.send_epoch)
              for i, ps in legs_by_id.items() if len(ps) > 2}
    assert chains, "the bridged fixture must produce store-and-forward chains"
    n_forwarded = 0
    for identity, ps in chains.items():
        forwarded = [p for p in ps if p.src_cell == 1 and p.send_epoch > 0]
        for f in forwarded:
            inbound = [p for p in ps if p.dst_cell == 1 and p.arrival_epoch < f.send_epoch]
            assert inbound, f"{identity}: leg {f} forwards data that never arrived at the bridge"
            latest = max(p.arrival_epoch for p in inbound)
            assert f.send_epoch >= latest + 1, (
                f"{identity}: forwarded in epoch {f.send_epoch} but its data lands in {latest}")
            # Co-located bridge: whoever landed it is whoever re-sends it, so no intra hop.
            assert f.egress_gpu in {p.ingress_gpu for p in inbound}, (
                f"{identity}: forwarded from gpu {f.egress_gpu}, which did not land the data")
            n_forwarded += 1

    print(f"  transit (BridgedIslandsCluster): Q={res.subdivision} {res.scale}")
    print(f"    {len(res.pieces)} pieces, {len(res.intra_demands)} intra demands, "
          f"{len(flows)} fine intra flows")
    print(f"    fine-link peak occupancy: egress {100 * worst_e:.0f}%, ingress "
          f"{100 * worst_i:.0f}%  (no violations)")
    print(f"    delivery coverage: all {n_required} (identity, gpu) pairs satisfied")
    print(f"    intra fits epoch: peak {peak} rounds vs m={m:.0f} at cell/gap {hot}")
    print(f"    {n_forwarded} forwarded legs across {len(chains)} chains, every one dwelling "
          f">= 1 epoch on the GPU that landed it")
    return True


def main() -> None:
    print("hierarchical pipeline replay (identity resolution + phase-3, Gurobi-free)")
    ran = False
    for tag, coll in (("allgather", Collective.ALLGATHER), ("alltoall", Collective.ALLTOALL)):
        ran |= replay(tag, coll)
    # Needs no checked-in schedule, so it runs even where the hetero ones are absent.
    ran |= replay_transit()
    if not ran:
        print("no coarse LP schedules found under Schedules/ -- run the coarse solve first")
        return
    print("pipeline replay OK")


if __name__ == "__main__":
    main()
