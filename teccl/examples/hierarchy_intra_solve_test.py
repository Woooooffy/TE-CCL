"""
Gurobi-free structural tests for the phase-3 intra-cell scheduler (teccl.hierarchy.intra_solve).

Steps 1-3 of the algorithm phase:
  [1] _to_jobs: IntraCellDemand -> scheduler jobs, incl. fan-out density test (direct vs binomial
      tree) and precedence links.
  [2] _schedule_band: EDF-weighted greedy b-matching -> ring recovery on symmetric input, relay
      priority, fractional volumes, port safety.
  [3] schedule_cell: end-to-end on the real HeteroTaperedCluster Host-B resolution (relays +
      internal allgather together): deadlines met, ports never oversubscribed, demand satisfied.

Run from the repo root (in the teccl env):
    python -m teccl.examples.hierarchy_intra_solve_test
"""
from collections import defaultdict

import numpy as np

from teccl.hierarchy.abstract import abstract
from teccl.hierarchy.reconstruct import IntraCellDemand, resolve_identities
from teccl.hierarchy import intra_solve
from teccl.hierarchy.intra_solve import (
    _Job, _to_jobs, _schedule_band, _assert_ports, _assert_deadlines, schedule_cell)
from teccl.input_data import Collective, TopologyParams
from teccl.solvers.demand import build_demand
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster

# reuse the hand-built coarse-path helpers / fake solver from the identity-resolution test
from teccl.examples.hierarchy_identity_resolution_test import (
    _fake_solver, _single_switch_path, _two_switch_path)


def _hostB_resolution():
    """The Host-B forced-relay scenario from the identity-resolution test: only B sources, so B
    has 3 egress relays (5/6/7 -> g4) plus its internal allgather, and A/C get ingress fan-out."""
    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    _, m = abstract(topo)
    A, B, C, T0, T1 = 0, 1, 2, 3, 4
    B_gpus, A_gpus, C_gpus = [4, 5, 6, 7], [0, 1, 2, 3], [8, 9, 10, 11, 12, 13]
    n = 19
    fine_demand = np.zeros((n, n, 1), dtype=np.int32)
    for s in B_gpus:
        for t in A_gpus + C_gpus + [g for g in B_gpus if g != s]:
            fine_demand[s][t][0] = 1
    # 8 units all leaving on g4's single uplink, so they occupy 8 DISTINCT coarse epochs: one
    # chunk per epoch is what that link can carry, and identity resolution now paces each piece to
    # fill a coarse epoch and asserts the result fits (_assert_rate_within_capacity). Overlapping
    # B->A and B->C on epochs 0..3 would be an infeasible coarse solution no real solve emits.
    pcp = {(B, A, 0): [_single_switch_path(B, A, T0, 1.0, k) for k in range(4)],
           (B, C, 0): [_two_switch_path(B, C, T0, T1, 1.0, k) for k in range(4, 8)]}
    solver = _fake_solver(pcp, switch_indices=[T0, T1])
    res = resolve_identities(solver, m, fine_demand, topo)
    return topo, m, res


def _by_cell(res):
    by_cell = defaultdict(list)
    for d in res.intra_demands:
        by_cell[d.cell].append(d)
    return by_cell


# ------------------------------------------------------------------------------------------
def test_to_jobs_hostB():
    topo, m, res = _hostB_resolution()
    B = 1
    cellB = m.coarse_cells[B]
    demands_B = _by_cell(res)[B]
    jobs = _to_jobs(demands_B, cellB)

    # egress relays -> exactly 3 HARD deliveries 5->4, 6->4, 7->4. Each absorbs its redundant
    # self_distribution twin (5->4 etc), so those 3 self deliveries are deduped away and promoted
    # to hard.
    egress = sorted((j.src, j.dst) for j in jobs if j.hard)
    assert egress == [(5, 4), (6, 4), (7, 4)], egress
    assert all(j.kind == "egress_stage" for j in jobs if j.hard)

    # after dedup the internal allgather leaves 9 soft self deliveries (the 3 that landed on g4
    # merged into the hard relays); 12 direct deliveries total, none with precedence.
    selfj = [j for j in jobs if not j.hard]
    assert len(selfj) == 9, len(selfj)
    assert all(j.kind == "self_distribution" and j.predecessor is None for j in selfj)
    assert len(jobs) == 12, len(jobs)
    # every B gpu still sends its own chunk to the other 3 (invariant across all 12 deliveries)
    src_counts = defaultdict(int)
    for j in jobs:
        src_counts[j.src] += 1
    assert dict(src_counts) == {4: 3, 5: 3, 6: 3, 7: 3}, dict(src_counts)
    print("  [1a] Host-B _to_jobs: 3 hard relays (redundant self-twins deduped) + 9 soft OK")


def test_to_jobs_tree_when_egress_bound():
    """An ISOLATED fan-out (one gateway broadcasting to 7, nothing else) is egress-bound -> lower
    to a binomial tree (ceil(log2(8))=3 rounds, 7 edges, precedence links)."""
    from teccl.hierarchy.cell import Cell
    gw, wanters = 0, [1, 2, 3, 4, 5, 6, 7]
    cell = Cell(members=list(range(8)) + [8], gpus=list(range(8)), internal_switches=[8])
    dem = [IntraCellDemand(cell=0, kind="ingress_distribution", identity=(0, 0),
                           src_gpu=gw, dst_gpus=tuple(wanters), volume=1.0, deadline_epoch=0)]
    jobs = _to_jobs(dem, cell)
    assert len(jobs) == 7, len(jobs)                    # 7 receivers => 7 tree edges
    # exactly the root's children have no predecessor; every other edge waits on its parent
    roots = [j for j in jobs if j.predecessor is None]
    assert all(j.src == gw for j in roots), [(j.src, j.dst) for j in roots]
    # tree depth: max chain length from root should be ceil(log2(8)) = 3
    depth = {gw: 0}
    for j in jobs:
        depth[j.dst] = depth[j.src] + 1
    assert max(depth.values()) == 3, depth
    # every wanter is reached exactly once
    assert sorted(j.dst for j in jobs) == wanters
    print("  [1b] isolated fan-out -> binomial tree (7 edges, depth 3, precedence) OK")


# ------------------------------------------------------------------------------------------
def _symmetric_allgather_jobs(n):
    """Every GPU sends its own chunk to every other GPU (soft), the dense allgather pattern."""
    return [_Job(identity=(s, 0), src=s, dst=d, volume=1.0, release_gap=0, deadline_gap=float("inf"),
                 hard=False, kind="self_distribution")
            for s in range(n) for d in range(n) if s != d]


def test_ring_recovery():
    n = 8
    gpus = list(range(n))
    jobs = _symmetric_allgather_jobs(n)
    flows = _schedule_band(jobs, gpus, switch=n, band=0)
    _assert_ports(flows)
    rounds = max(f.local_round for f in flows) + 1
    assert rounds == n - 1, rounds                         # max port load = n-1
    # every round is a permutation: each GPU sends exactly 1 and receives exactly 1
    for r in range(rounds):
        rf = [f for f in flows if f.local_round == r]
        senders = [f.sender for f in rf]
        receivers = [f.receiver for f in rf]
        assert sorted(senders) == gpus and sorted(receivers) == gpus, (r, senders, receivers)
        # ring structure: round r moves every GPU's data to the (i+r+1)-th neighbor
        for f in rf:
            assert (f.receiver - f.sender) % n == r + 1, (r, f.sender, f.receiver)
    print(f"  [2a] ring recovery: {rounds} rounds, each a distance-(r+1) permutation OK")


def test_relay_priority():
    """A hard relay with a far ring-distance still lands in round 0 because hard/EDF outranks the
    symmetric soft traffic sharing its ports."""
    n = 8
    gpus = list(range(n))
    jobs = _symmetric_allgather_jobs(n)
    hard = _Job(identity=(99, 0), src=0, dst=7, volume=1.0, release_gap=0, deadline_gap=0,
                hard=True, kind="egress_stage")
    jobs.append(hard)
    _schedule_band(jobs, gpus, switch=n, band=0)
    _assert_deadlines([hard])
    assert hard.completion_round == 0, hard.completion_round
    print("  [2b] relay priority: far-distance hard job pulled to round 0 OK")


def test_fractional_pack():
    """Two 0.5-volume transfers on the same src/dst pack into ONE round (b-matching, not a strict
    permutation)."""
    gpus = [0, 1, 2]
    j1 = _Job(identity=(0, 0), src=0, dst=1, volume=0.5, release_gap=0, deadline_gap=float("inf"),
              hard=False, kind="self_distribution")
    j2 = _Job(identity=(0, 1), src=0, dst=1, volume=0.5, release_gap=0, deadline_gap=float("inf"),
              hard=False, kind="self_distribution")
    flows = _schedule_band([j1, j2], gpus, switch=3, band=0)
    _assert_ports(flows)
    assert j1.completion_round == 0 and j2.completion_round == 0
    assert max(f.local_round for f in flows) == 0
    print("  [2c] fractional 0.5+0.5 packs into one round, ports safe OK")


def test_tree_precedence_schedule():
    """A binomial-tree fan-out schedules respecting precedence: a child forwards only in a round
    strictly after it received."""
    from teccl.hierarchy.cell import Cell
    gw, wanters = 0, [1, 2, 3, 4, 5, 6, 7]
    cell = Cell(members=list(range(8)) + [8], gpus=list(range(8)), internal_switches=[8])
    dem = [IntraCellDemand(cell=0, kind="ingress_distribution", identity=(0, 0),
                           src_gpu=gw, dst_gpus=tuple(wanters), volume=1.0, deadline_epoch=0)]
    jobs = _to_jobs(dem, cell)
    flows = _schedule_band(jobs, cell.gpus, switch=8, band=0)
    _assert_ports(flows)
    # a GPU can only send after it holds the data: recv_round[child] < send_round for its children
    recv_round = {gw: -1}
    for f in sorted(flows, key=lambda x: x.local_round):
        recv_round[f.receiver] = f.local_round
    for j in jobs:
        # find the round this edge fired
        fr = next(f.local_round for f in flows if f.sender == j.src and f.receiver == j.dst)
        assert recv_round[j.src] < fr, (j.src, j.dst, recv_round.get(j.src), fr)
    rounds = max(f.local_round for f in flows) + 1
    assert rounds == 3, rounds                              # log2(8) broadcast depth
    print(f"  [2d] tree fan-out respects precedence, {rounds} rounds (log2 depth) OK")


# ------------------------------------------------------------------------------------------
def _check_fanout_delivery(flows, demands, cell_gpus):
    """Every wanter of every fan-out demand receives exactly the demanded volume (once)."""
    for d in demands:
        if d.kind == "egress_stage":
            continue
        for w in d.dst_gpus:
            if w == d.src_gpu:
                continue
            got = sum(f.volume for f in flows if f.identity == d.identity and f.receiver == w)
            assert abs(got - d.volume) < 1e-6, (d.kind, d.identity, w, got, d.volume)


def test_schedule_cell_hostB_combined():
    """End-to-end on the real HeteroTaperedCluster Host-B resolution: relays + internal allgather
    scheduled together (cell B), and the egress-bound ingress fan-out into A and C (trees)."""
    topo, m, res = _hostB_resolution()
    by_cell = _by_cell(res)
    A, B, C = 0, 1, 2

    # ---- cell B: 3 hard egress relays + dense internal allgather ----
    demands_B = by_cell[B]
    flows_B = schedule_cell(B, m.coarse_cells[B], demands_B)   # asserts deadlines + ports inside
    _check_fanout_delivery(flows_B, demands_B, m.coarse_cells[B].gpus)
    # every hard relay lands in a round of its own deadline gap; the switch id is B's NVSwitch (15)
    assert all(f.via_switch == 15 for f in flows_B)
    relay_flows = [f for f in flows_B if f.sender in (5, 6, 7) and f.receiver == 4]
    assert {f.identity for f in relay_flows} == {(5, 0), (6, 0), (7, 0)}, relay_flows
    bands_B = sorted({f.band for f in flows_B})

    # ---- cells A and C: isolated ingress fan-out from one gateway -> binomial trees ----
    flows_A = schedule_cell(A, m.coarse_cells[A], by_cell[A])
    _check_fanout_delivery(flows_A, by_cell[A], m.coarse_cells[A].gpus)
    flows_C = schedule_cell(C, m.coarse_cells[C], by_cell[C])
    _check_fanout_delivery(flows_C, by_cell[C], m.coarse_cells[C].gpus)
    # A's ingress is egress-bound (one gateway g0 broadcasting 4 identities) -> tree relaying, so
    # some non-gateway A gpu forwards (is a sender) despite owning none of the source data.
    a_senders = {f.sender for f in flows_A}
    assert a_senders - {0}, ("expected tree relaying in A", a_senders)

    print(f"  [3] schedule_cell Host-B combined: B gaps {bands_B}, "
          f"{len(flows_B)} B-flows / {len(flows_A)} A-flows / {len(flows_C)} C-flows, "
          f"relays land + fan-out delivered, ports & deadlines OK")


def main():
    test_to_jobs_hostB()
    test_to_jobs_tree_when_egress_bound()
    test_ring_recovery()
    test_relay_priority()
    test_fractional_pack()
    test_tree_precedence_schedule()
    test_schedule_cell_hostB_combined()
    print("intra-solve step-1/2/3 tests OK")


if __name__ == "__main__":
    main()
