"""Unit tests for pacing gate derivation (_finish_before_start_gates).

The gate manifest decides what each paced send waits on so that a rate-paced send
holds to its intended epoch on real hardware. The edge is derived from LINK-OCCUPANCY
WINDOWS (start, finish=start+volume/rate), NOT from step order -- which is what makes
it correct for two cases a plain sort gets wrong:

  1. Concurrent sends (same epoch, different peers/channels/uplinks) must never be
     serialized against each other.
  2. A send paced to span multiple epochs must not serialize a later send that runs
     alongside its tail on capacity a *different*, earlier send already freed.

There are two clocks. P2 is another paced SEND FROM THE SAME GPU completing; P3 is a
paced RECV ARRIVING at the sending GPU, which pins a send whose GPU sent nothing in
time. Both pools are per source GPU -- a clock only has to be a rate-paced event that
GPU's proxy observes, not one that frees the particular link the send needs. They
ride different runtime carriers -- netdepid/netdeps for P2 (the proxy withholds the
isend against the NIC) and depid/deps for P3 (the kernel already waits on the proxy
marking a recv slot filled) -- so the manifest tags each edge with its kind.

Run from the repo root (in an env with the ncclize deps, or none -- this file imports
only teccl_ncclize, which needs lxml/z3 for the rest of the module):
    python teccl/ncclize/pacing_gates_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teccl_ncclize import _finish_before_start_gates as gates  # noqa: E402
from teccl_ncclize import unpinned_sends  # noqa: E402


def check(name, paced, expect):
    got = sorted(gates(paced), key=repr)
    if got != sorted(expect, key=repr):
        raise AssertionError(f"{name}: expected {sorted(expect, key=repr)}, got {got}")
    print(f"  [OK] {name}")


def main():
    print("send-pacing gate derivation tests")

    # key = (step, src, dst, path_key) -> (start_fine, finish_fine)

    # Concern 1a: three concurrent sends from gpu0 in the SAME epoch (finish 6) to
    # different peers. None finishes before another starts -> no edges.
    check("concern1a: multi-peer, same epoch, not serialized",
          {(5, 0, 1, None): (5, 6), (5, 0, 2, None): (5, 6), (5, 0, 3, None): (5, 6)},
          [])

    # Concern 1b: one peer, two multipath channels (distinct path_keys), same epoch.
    check("concern1b: multipath channels, same epoch, not serialized",
          {(5, 0, 1, ('A',)): (5, 6), (5, 0, 1, ('B',)): (5, 6)},
          [])

    # Contiguous epochs: send finishing at 6 pins the send starting at 6.
    check("contiguous chain: gate on the send that frees the link",
          {(5, 0, 1, None): (5, 6), (6, 0, 1, None): (6, 7)},
          [((6, 0, 1, None), (5, 0, 1, None), 'send')])

    # Concern 2: X spans epochs 0..2 (half rate, finish 2); Y in epoch 0 finishes at 1;
    # Z starts at epoch 1. Z runs alongside X's tail on the half Y just freed, so Z must
    # gate on Y (finish 1 <= 1), NOT on the still-running X (finish 2 > 1).
    check("concern2: multi-epoch send does not serialize a later one",
          {(0, 0, 1, None): (0, 2),   # X
           (0, 0, 2, None): (0, 1),   # Y
           (1, 0, 3, None): (1, 2)},  # Z
          [((1, 0, 3, None), (0, 0, 2, None), 'send')])

    # A gap with nothing arriving still pins to the latest send that finishes in time
    # (best-effort; the residual earliness is what the stitch reports).
    check("gap: pin to the latest finishing predecessor",
          {(0, 0, 1, None): (0, 1), (3, 0, 1, None): (3, 4)},
          [((3, 0, 1, None), (0, 0, 1, None), 'send')])

    # Sends from different source GPUs are never in each other's pool -> no edges.
    check("cross-gpu: independent senders, no edges",
          {(0, 0, 1, None): (0, 1), (1, 7, 1, None): (1, 2)},
          [])

    # Multi-uplink: one GPU, two sends entering the fabric at DIFFERENT first
    # switches (path (9,..) vs (10,..)). The send pool is per SOURCE GPU, so the
    # completion on uplink 9 is a valid clock for the send leaving on uplink 10 --
    # it is an op-completion edge between two threadblocks of one GPU, which is what
    # a netdep already is. (Per-uplink grouping would leave this send unpinned; that
    # is the dual-plane/multi-rail gap.)
    check("multi-uplink: a completion on the other uplink still pins",
          {(0, 0, 1, (9, 5)): (0, 1), (1, 0, 2, (10, 5)): (1, 2)},
          [((1, 0, 2, (10, 5)), (0, 0, 1, (9, 5)), 'send')])

    # Same epoch on two uplinks is still concurrency, not a chain: neither finishes
    # before the other starts.
    check("multi-uplink: same epoch on two uplinks is not serialized",
          {(0, 0, 1, (9, 5)): (0, 1), (0, 0, 2, (10, 5)): (0, 1)},
          [])

    # Tie at equal finish prefers a producer on the CONSUMER's OWN uplink: it frees the
    # very link the gated send needs, not just the clock.
    check("multi-uplink: tie at equal finish prefers the same uplink",
          {(0, 0, 1, (9, 5)): (0, 1), (0, 0, 2, (10, 5)): (0, 1),
           (1, 0, 3, (10, 5)): (1, 2)},
          [((1, 0, 3, (10, 5)), (0, 0, 2, (10, 5)), 'send')])

    # Same first switch (same uplink) DOES chain across the epoch boundary, even
    # when the downstream switch path differs.
    check("same uplink, different downstream path, still chains",
          {(0, 0, 1, (9, 5)): (0, 1), (1, 0, 2, (9, 6)): (1, 2)},
          [((1, 0, 2, (9, 6)), (0, 0, 1, (9, 5)), 'send')])

    # --- P3: the recv pool -------------------------------------------------------
    # gpu0's uplink is idle going into epoch 3 (its own last send freed the link back at 1),
    # but a paced delivery from gpu9 ARRIVES at gpu0 exactly at 3. That arrival is the only
    # rate-paced tick available, so it -- not the stale send at 1 -- is the gate. The producer
    # key is the recv's MIRROR of the delivering send: (step, dst, src, path_key).
    check("p3: recv arriving at k pins a send the idle uplink cannot",
          {(0, 0, 1, None): (0, 1),     # gpu0's earlier send, frees the uplink at 1
           (2, 9, 0, None): (2, 3),     # delivery into gpu0, arrives at 3
           (3, 0, 1, None): (3, 4)},    # the send to pin
          [((3, 0, 1, None), (2, 0, 9, None), 'recv')])

    # Tie at the same epoch prefers the SEND pool: a netdep costs no XML step, a depid/deps
    # edge may cost an expansion nop. Here the multi-epoch send finishes at 3 and the recv
    # also arrives at 3.
    check("p3: tie at k prefers the send pool",
          {(0, 0, 1, None): (0, 3),     # multi-epoch send, frees the uplink at 3
           (2, 9, 0, None): (2, 3),     # delivery into gpu0, also lands at 3
           (3, 0, 1, None): (3, 4)},
          [((3, 0, 1, None), (0, 0, 1, None), 'send')])

    # An arrival at a DIFFERENT gpu is not a clock for this send: the recv pool is indexed by
    # the sending gpu, so gpu5's delivery cannot pin gpu0.
    check("p3: an arrival at another gpu does not pin",
          {(2, 9, 5, None): (2, 3),     # delivery into gpu5, not gpu0
           (3, 0, 1, None): (3, 4)},
          [])

    # An arrival AFTER the send starts is not a candidate -- same finish-before-start rule the
    # send pool uses. Nothing else is eligible either, so the send stays ungated.
    check("p3: an arrival after the send starts is not a candidate",
          {(3, 9, 0, None): (3, 4),     # arrives at 4, after the send is due
           (3, 0, 1, None): (3, 4)},
          [])

    # A GPU with no send to gate against but a steady stream of arrivals pins on the latest
    # one that lands in time: 5, not 2 (and not the one arriving at 8, after it is due).
    # gpu9's own three sends share an uplink and chain among themselves as usual.
    check("p3: pick the latest arrival that lands in time",
          {(0, 9, 0, None): (0, 2),
           (4, 9, 0, None): (4, 5),
           (7, 9, 0, None): (7, 8),     # arrives at 8, too late
           (5, 0, 1, None): (5, 6)},
          [((5, 0, 1, None), (4, 0, 9, None), 'recv'),
           ((4, 9, 0, None), (0, 9, 0, None), 'send'),
           ((7, 9, 0, None), (4, 9, 0, None), 'send')])

    remote_derivation_tests()
    unpinned_report_tests()
    realization_tests()
    remote_realization_tests()
    remote_emission_tests()
    rate_widening_tests()
    print("pacing gate tests OK")


def unpinned_report_tests():
    """The pacing REPORT, read straight off the manifest: a send counts as pinned only when some
    emitted edge's producer lands exactly at its start. A best-effort edge from an early producer
    orders the send but does not hold it, and must be reported."""

    def ucheck(name, paced, expect, remote=False):
        got = unpinned_sends(paced, gates(paced, remote_gates=remote))
        if got != sorted(expect):
            raise AssertionError(f"{name}: expected {sorted(expect)}, got {got}")
        print(f"  [OK] {name}")

    # gpu0's uplink freed at 1 but the send is due at 3: the manifest carries a 'send' edge to
    # that stale producer, which orders it and nothing more. This is the case the old
    # res.pieces-based check reported as pinned.
    stalled = {(0, 0, 5, None): (0, 1),
               (2, 9, 5, None): (2, 3),
               (3, 0, 5, None): (3, 4)}
    # (9, 2) comes along for free and is also correct: gpu9's own send has no producer at all,
    # so nothing pins it either. A send with no gate and a send with an early gate are the same
    # kind of residual.
    ucheck("report: an early producer is a gate but not a pin", stalled, [(0, 3), (9, 2)])

    # Same fixture with P4 on: the arrival at the DESTINATION lands exactly at 3, so the send is
    # genuinely pinned and the report is clean. This is the 16-send dual-plane case in miniature.
    ucheck("report: a remote gate pins what no local clock could", stalled, [(9, 2)], remote=True)

    # A producer landing exactly on time pins, via either local carrier.
    ucheck("report: an on-time send producer pins",
           {(0, 0, 5, None): (0, 3), (3, 0, 5, None): (3, 4)}, [])
    ucheck("report: an on-time arrival pins",
           {(2, 9, 0, None): (2, 3), (3, 0, 5, None): (3, 4)}, [(9, 2)])

    # Epoch 0 needs no clock -- it is held by the egress staging it depends on (P1).
    ucheck("report: epoch 0 is never unpinned", {(0, 0, 5, None): (0, 1)}, [])

    # One entry per (gpu, epoch), however many sends that GPU has due at once.
    ucheck("report: collapsed per (gpu, epoch)",
           {(0, 0, 5, None): (0, 1), (3, 0, 5, None): (3, 4), (3, 0, 6, None): (3, 4)},
           [(0, 3)])


def remote_derivation_tests():
    """P4, the EXPERIMENTAL remote pool: when neither local clock pins a send, gate it on a
    paced delivery into its DESTINATION -- the remote op holding the capacity it is waiting
    behind. Skipped when remote_gates=False, and never spent when a local gate already pins."""

    def rcheck(name, paced, expect, remote=True):
        got = sorted(gates(paced, remote_gates=remote), key=repr)
        if got != sorted(expect, key=repr):
            raise AssertionError(f"{name}: expected {sorted(expect, key=repr)}, got {got}")
        print(f"  [OK] {name}")

    # gpu0's own uplink went quiet at 1 and nothing lands on gpu0, so no local clock pins its
    # send at 3. What actually holds that send is the receiver: gpu5 is taking delivery from
    # gpu9 through epoch 2, finishing exactly at 3. Gate on that arrival -- the producer key is
    # the RECV MIRROR at gpu5, whose op carries remotenotify back to gpu0.
    stalled = {(0, 0, 5, None): (0, 1),     # gpu0's stale send, freed the uplink at 1
               (2, 9, 5, None): (2, 3),     # delivery into gpu5 (the destination), lands at 3
               (3, 0, 5, None): (3, 4)}     # the send to pin
    rcheck("p4: an arrival at the DESTINATION pins a send no local clock can",
           stalled, [((3, 0, 5, None), (2, 5, 9, None), 'remote')])

    # Same fixture with the pool disabled: the derivation falls back to the stale local netdep,
    # which is best-effort only (finishes at 1, the send is due at 3) -- i.e. the residual.
    rcheck("p4: disabled, falls back to the stale local gate",
           stalled, [((3, 0, 5, None), (0, 0, 5, None), 'send')], remote=False)

    # A local clock that DOES pin is never traded for a notification: gpu0's own send frees the
    # uplink exactly at 3, so the free netdep wins and no remote edge is emitted.
    rcheck("p4: not spent when a local send already pins",
           {(0, 0, 5, None): (0, 3),
            (2, 9, 5, None): (2, 3),
            (3, 0, 5, None): (3, 4)},
           [((3, 0, 5, None), (0, 0, 5, None), 'send')])

    # A remote arrival that lands EARLY does not pin either, and a gate that does not pin is not
    # worth a notification: fall back to the local best-effort edge.
    rcheck("p4: a stale remote arrival is not worth a notification",
           {(0, 0, 4, None): (0, 1),   # gpu0's stale send, to a peer other than the one below
            (1, 9, 5, None): (1, 2),   # into gpu5, but lands at 2 while the send is due at 3
            (3, 0, 5, None): (3, 4)},
           [((3, 0, 5, None), (0, 0, 4, None), 'send')])

    # Two arrivals land at the destination at the same time over DIFFERENT ingress links. The
    # tie goes to the one sharing the gated send's own ingress link (last switch 9), which is
    # the link the send is actually queued behind.
    # Two sends from one GPU to one peer, due at the SAME instant on different planes. Either
    # arrival pins both, so they share ONE notification -- per-consumer ingress-link picks would
    # otherwise emit two for one instant, which is also the only way a stream can issue two
    # notifications in a single step (the case the ordinal's no-reordering assumption misses).
    rcheck("p4: sends due at one instant share a single notification",
           {(3, 8, 5, (7, 9)): (3, 4),
            (3, 9, 5, (7, 8)): (3, 4),
            (4, 0, 5, (7, 9)): (4, 5),
            (4, 0, 5, (7, 8)): (4, 5)},
           [((4, 0, 5, (7, 9)), (3, 5, 9, (7, 8)), 'remote'),
            ((4, 0, 5, (7, 8)), (3, 5, 9, (7, 8)), 'remote')])

    rcheck("p4: tie prefers the arrival on the same ingress link",
           {(2, 8, 5, (7, 9)): (2, 3),
            (2, 9, 5, (7, 8)): (2, 3),
            (3, 0, 5, (7, 9)): (3, 4)},
           [((3, 0, 5, (7, 9)), (2, 5, 8, (7, 9)), 'remote')])


def remote_realization_tests():
    """A 'remote' edge lands on remote_dep/remote_notify, numbered per ordered (notifier,
    waiter) pair in the notifier's own step order, and the deadlock check follows it."""
    T = _load_taccl_ncclize()

    def send(gpu, peer, step):
        return T._Op(gpu, peer, step, True, 's', 'i', 0, 'o', 0, 1, [])

    def recv(gpu, peer, step):
        return T._Op(gpu, peer, step, False, 'r', 'i', 0, 'o', 0, 1, [])

    def build(specs):
        """specs: {gpu: [(rbid, [ops]), ...]} -> gpus dict."""
        gpus = {}
        for rank, tb_specs in specs.items():
            gpu = T._Gpu([], {}, {}, 0, 0)
            gpu.threadblocks = []
            for rbid, ops in tb_specs:
                tb = T._Threadblock(channel=0, rbid=rbid)
                tb.steps = list(ops)
                tb.ops = list(ops)
                for op in ops:
                    op.block_rbid = rbid
                gpu.threadblocks.append(tb)
            gpus[rank] = gpu
        return gpus

    # The basic edge: gpu5's recv notifies gpu0, gpu0's send waits for notification #1 from
    # gpu5. Neither local carrier is touched -- this is not a local dependency and must not
    # become one.
    r5, s0 = recv(5, 9, 2), send(0, 5, 3)
    gpus = build({5: [(0, [r5])], 0: [(0, [s0])]})
    T._realize_pacing_gates(gpus, [((3, 0, 5, None), (2, 5, 9, None), 'remote')])
    assert s0.remote_dep == (r5, 1), f"remote gate not realized: {s0.remote_dep}"
    assert r5.remote_notify == [0], f"notifier did not record the waiter: {r5.remote_notify}"
    assert s0.net_dep is None and s0.depends == [], "remote gate leaked onto a local carrier"
    print("  [OK] remote: gate lands on remote_dep/remote_notify, not a local carrier")

    # Ordinals count within the ordered PAIR, in the notifier's step order: gpu5's two recvs
    # notify gpu0 as #1 and #2 even though a third recv (to gpu7) sits between them. gpu7's
    # stream is numbered independently, from 1.
    a, b, c = recv(5, 9, 1), recv(5, 9, 2), recv(5, 9, 3)
    w1, w2, w3 = send(0, 5, 2), send(7, 5, 3), send(0, 5, 4)
    gpus = build({5: [(0, [a, b, c])], 0: [(0, [w1]), (1, [w3])], 7: [(0, [w2])]})
    T._realize_pacing_gates(gpus, [((2, 0, 5, None), (1, 5, 9, None), 'remote'),
                                   ((3, 7, 5, None), (2, 5, 9, None), 'remote'),
                                   ((4, 0, 5, None), (3, 5, 9, None), 'remote')])
    assert w1.remote_dep == (a, 1) and w3.remote_dep == (c, 2), \
        f"gpu5->gpu0 stream misnumbered: {w1.remote_dep[1]}, {w3.remote_dep[1]}"
    assert w2.remote_dep == (b, 1), f"gpu5->gpu7 stream not numbered independently: {w2.remote_dep}"
    assert a.remote_notify == [0] and b.remote_notify == [7] and c.remote_notify == [0]
    print("  [OK] remote: ordinals count per (notifier, waiter) pair in the notifier's order")

    # One recv serving two sends on the same waiter is ONE notification: both sends wait for
    # ordinal 1 and the notifier lists the waiter once. (A recv feeding two DIFFERENT waiters
    # does carry two list entries, each numbered in its own stream.)
    r, x, y, z = recv(5, 9, 1), send(0, 5, 2), send(0, 6, 2), send(7, 5, 2)
    gpus = build({5: [(0, [r])], 0: [(0, [x]), (1, [y])], 7: [(0, [z])]})
    T._realize_pacing_gates(gpus, [((2, 0, 5, None), (1, 5, 9, None), 'remote'),
                                   ((2, 0, 6, None), (1, 5, 9, None), 'remote'),
                                   ((2, 7, 5, None), (1, 5, 9, None), 'remote')])
    assert x.remote_dep == (r, 1) and y.remote_dep == (r, 1) and z.remote_dep == (r, 1)
    assert r.remote_notify == [0, 7], f"notify list wrong: {r.remote_notify}"
    print("  [OK] remote: one recv serving several waiters notifies each stream once")

    # A remote gate is the only edge here that crosses GPUs on its own, so it is the one that
    # can deadlock: gpu0's send waits on gpu5's recv, which sits behind a recv of gpu0's LATER
    # send in program order. The check must follow remote_dep to see it.
    T._assert_gates_acyclic(gpus)   # the fixtures above are legitimate
    g = send(0, 5, 1)
    p = recv(5, 0, 1)               # the far end's recv of g -- so g must happen first ...
    q = recv(5, 9, 1)               # ... and q, behind p in program order, gates g. Cycle.
    p.depends = [g]
    g.remote_dep = (q, 1)
    q.remote_notify = [0]
    cyc = build({5: [(0, [p, q])], 0: [(0, [g])]})
    try:
        T._assert_gates_acyclic(cyc)
    except ValueError as e:
        assert 'deadlock' in str(e), e
        print("  [OK] remote: the deadlock check follows a cross-GPU remote gate")
    else:
        raise AssertionError("a deadlocking remote gate was accepted")


def remote_emission_tests():
    """The XML carries remotedep/remotedeps on the gated send and remotenotify on the remote
    recv -- and carries NOTHING when no remote gate was derived, so enabling the pool cannot
    perturb a schedule that does not need it."""
    import json
    import os
    schedule_path = 'Schedules/two_pod_rail_hostbound_allgather_flat.json'
    try:
        from teccl_ncclize import build_algorithm
        from taccl_ncclize import ncclize, ChannelPolicy
    except ImportError as e:
        print(f"  [SKIP] remote emission ({e})")
        return
    if not os.path.exists(schedule_path):
        print(f"  [SKIP] remote emission (no {schedule_path})")
        return
    sched = json.load(open(schedule_path))
    algo, fpk, _srm, _view, rate, derived, _grm = build_algorithm(sched)

    def emit(gate_list):
        return ncclize(algo, channel_policy=ChannelPolicy.MatchTopology, old_format=True,
                       use_scratch=True, flow_path_keys=fpk, piece_rate=rate,
                       pacing_gates=gate_list)

    base = emit(derived)
    assert 'remotedep' not in base and 'remotenotify' not in base, \
        "a schedule with no remote gate must emit no remote attributes"
    print("  [OK] remote emission: no remote gate derived -> no attributes emitted")

    # Inject one remote edge over the real op graph: take a send and gate it on a recv that
    # lives on the peer it sends to. (The derivation is unit-tested above; this exercises
    # realization and serialization against real threadblocks and nop expansion.)
    sends = _send_keys(algo, fpk)
    recvs = _recv_keys(algo, fpk)
    pair = next((((step, src, dst, pk), r)
                 for (step, src, dst, pk) in sends
                 for r in recvs
                 if r[1] == dst and r[0] < step and r[2] != src), None)
    if pair is None:
        print("  [SKIP] remote emission (no suitable recv on the destination)")
        return
    consumer, producer = pair
    step, src, dst, pk = consumer
    xml = emit([(consumer, producer, 'remote')])
    assert f'remotedep="{dst}"' in xml, "the gated send did not carry remotedep"
    assert 'remotedeps="1"' in xml, "the gated send did not carry an ordinal"
    assert f'remotenotify="{src}"' in xml, "the remote recv did not carry remotenotify"
    print("  [OK] remote emission: remotedep/remotedeps on the send, remotenotify on the recv")


def _send_keys(algo, fpk):
    """Op keys (step, src, dst, path_key) for every send in the algorithm, in step order."""
    seen = []
    for si, step in enumerate(algo.steps):
        for addr, src, dst in step.sends:
            key = (si, src, dst, fpk.get((si, addr, src, dst)))
            if key not in seen:
                seen.append(key)
    return seen


def _recv_keys(algo, fpk):
    """The mirror keys (step, dst, src, path_key) -- one per recv op."""
    return [(si, dst, src, pk) for (si, src, dst, pk) in _send_keys(algo, fpk)]


def _load_taccl_ncclize():
    """Import taccl_ncclize with lxml/z3 stubbed out.

    The module pulls both in at import time for parts this test never touches (XML
    serialization uses the same ElementTree API; z3 is only used by scratch remapping),
    so stubbing keeps the gate tests runnable in a bare env, matching this file's docstring.
    """
    import types
    import xml.etree.ElementTree as _ET
    lxml = types.ModuleType('lxml'); lxml.etree = _ET
    sys.modules.setdefault('lxml', lxml)
    sys.modules.setdefault('lxml.etree', _ET)
    sys.modules.setdefault('z3', types.ModuleType('z3'))
    import taccl_ncclize
    return taccl_ncclize


def realization_tests():
    """_realize_pacing_gates routes each edge to the carrier its kind names -- net_dep for a
    send-sourced gate, depends for a recv-sourced one -- and _assert_gates_acyclic catches the
    deadlock the runtime would otherwise hang on."""
    T = _load_taccl_ncclize()

    def send(gpu, peer, step):
        return T._Op(gpu, peer, step, True, 's', 'i', 0, 'o', 0, 1, [])

    def recv(gpu, peer, step):
        return T._Op(gpu, peer, step, False, 'r', 'i', 0, 'o', 0, 1, [])

    def build(tb_specs):
        """tb_specs: list of (rbid, [ops]) for one gpu; returns (gpus, ops)."""
        gpu = T._Gpu([], {}, {}, 0, 0)
        gpu.threadblocks = []
        for rbid, ops in tb_specs:
            tb = T._Threadblock(channel=0, rbid=rbid)
            tb.steps = list(ops)
            tb.ops = list(ops)
            for op in ops:
                op.block_rbid = rbid
            gpu.threadblocks.append(tb)
        return {0: gpu}

    # A SEND-sourced gate lands on net_dep and leaves depends untouched: it waits on bytes
    # being off the NIC, which the kernel cannot observe, so it must not become a data
    # dependency (nor an expansion nop).
    a, b = send(0, 1, 0), send(0, 2, 1)
    gpus = build([(0, [a]), (1, [b])])
    T._realize_pacing_gates(gpus, [((1, 0, 2, None), (0, 0, 1, None), 'send')])
    assert b.net_dep is a, "gate not realized onto net_dep"
    assert b.depends == [] and a.depends == [], "gate leaked into depends"
    assert not a.has_dependence, "gate producer must not need a kernel dep flag"
    print("  [OK] realization: cross-tb send gate lands on net_dep, not depends")

    # A RECV-sourced gate is the mirror case: it waits on a reception, the exact event the
    # proxy already signals to the kernel, so it rides depends/depid/deps and leaves net_dep
    # alone. The producing recv then picks up hasdep from ncclize's ordinary marking pass.
    r, c = recv(0, 9, 0), send(0, 2, 1)
    gpus = build([(0, [r]), (1, [c])])
    T._realize_pacing_gates(gpus, [((1, 0, 2, None), (0, 0, 9, None), 'recv')])
    assert c.net_dep is None, "recv gate must not take the netdep carrier"
    assert c.depends == [r], f"recv gate not realized onto depends: {c.depends}"
    print("  [OK] realization: cross-tb recv gate lands on depends, not net_dep")

    # A recv that is ALREADY a data dependency of the gated send needs no second edge: the
    # wait exists, and a duplicate would only buy an expansion nop.
    r, c = recv(0, 9, 0), send(0, 2, 1)
    c.depends = [r]
    gpus = build([(0, [r]), (1, [c])])
    T._realize_pacing_gates(gpus, [((1, 0, 2, None), (0, 0, 9, None), 'recv')])
    assert c.depends == [r], f"recv gate duplicated an existing dependency: {c.depends}"
    print("  [OK] realization: recv gate not duplicated onto an existing dependency")

    # The producer key is looked up in the pool its kind names. A send and a recv can share a
    # key when traffic is bidirectional over the same route at the same step -- here gpu0
    # both sends to and receives from gpu9 at step 0, so (0, 0, 9, None) names two ops. The
    # kind is what disambiguates them.
    sr, rr, c = send(0, 9, 0), recv(0, 9, 0), send(0, 2, 1)
    gpus = build([(0, [sr]), (1, [rr]), (2, [c])])
    T._realize_pacing_gates(gpus, [((1, 0, 2, None), (0, 0, 9, None), 'recv')])
    assert c.depends == [rr] and c.net_dep is None, "kind did not disambiguate the shared key"
    c2 = send(0, 3, 1)
    gpus = build([(0, [sr]), (1, [rr]), (2, [c2])])
    T._realize_pacing_gates(gpus, [((1, 0, 3, None), (0, 0, 9, None), 'send')])
    assert c2.net_dep is sr and c2.depends == [], "kind did not disambiguate the shared key"
    print("  [OK] realization: kind disambiguates a key shared by a send and a recv")

    # Two gated sends in ONE threadblock are an ordinary schedule: both survive.
    p0, p1 = send(0, 1, 0), send(0, 1, 2)
    c0, c1 = send(0, 2, 1), send(0, 2, 3)
    gpus = build([(0, [p0, p1]), (1, [c0, c1])])
    T._realize_pacing_gates(gpus, [((1, 0, 2, None), (0, 0, 1, None), 'send'),
                                   ((3, 0, 2, None), (2, 0, 1, None), 'send')])
    assert (c0.net_dep, c1.net_dep) == (p0, p1), "a threadblock may carry several gates"
    print("  [OK] realization: several gates in one threadblock all survive")

    # Same threadblock on both ends: one threadblock is one connection, whose send
    # FIFO already orders its own sends, so the edge is dropped as redundant.
    s0, s1 = send(0, 1, 0), send(0, 1, 1)
    gpus = build([(0, [s0, s1])])
    T._realize_pacing_gates(gpus, [((1, 0, 1, None), (0, 0, 1, None), 'send')])
    assert s1.net_dep is None, "same-connection gate should be dropped as redundant"
    print("  [OK] realization: same-threadblock gate dropped as redundant")

    # Acyclicity. Legitimate shape: B = [X, gated-send] -- no cycle.
    ax, ay = send(0, 1, 0), send(0, 1, 2)
    bx, by = send(0, 2, 1), send(0, 2, 3)
    ay.depends = [bx]          # A's second send needs B's first
    by.net_dep = ax            # B's second send gated on A's first
    T._assert_gates_acyclic(build([(0, [ax, ay]), (1, [bx, by])]))
    print("  [OK] acyclicity: legitimate gate accepted")

    # The failing shape from the runtime plan: B = [gated-send, X], A = [Y needs B.X, send],
    # gate A.1 -> B.0. Closes through both threadblocks' program order.
    ax, ay = send(0, 1, 0), send(0, 1, 2)
    bx, by = send(0, 2, 1), send(0, 2, 3)
    ax.depends = [by]          # A.0 needs B.1
    bx.net_dep = ay            # B.0 gated on A.1
    try:
        T._assert_gates_acyclic(build([(0, [ax, ay]), (1, [bx, by])]))
    except ValueError as e:
        assert 'deadlock' in str(e), e
        print("  [OK] acyclicity: deadlocking gate rejected")
    else:
        raise AssertionError("expected the gate cycle to be rejected")


def rate_widening_tests():
    """_widen_rates_for_tb_serialization widens a same-tb, same-epoch flow group to its total.

    A threadblock has no intra-block concurrency, so the group's ops run back to back. Widening
    each to the group total makes the group finish exactly when concurrent transmission would
    have, while the connection draws the same total for the same span -- so the per-link load is
    unchanged. Ops of different epochs, of different flows, and unpaced ops are all left alone.
    """
    T = _load_taccl_ncclize()

    def op(step, cnt, rate, path_key='p'):
        o = T._Op(0, 1, step, True, 's', 'i', 0, 'o', 0, cnt, [])
        o.piece_rate = rate
        o.path_key = path_key
        return o

    def build(ops):
        gpu = T._Gpu([], {}, {}, 0, 0)
        tb = T._Threadblock(channel=0, rbid=0)
        tb.steps = list(ops)
        gpu.threadblocks = [tb]
        return {0: gpu}

    # Two ops of one flow at one epoch: each emits the group total (1*4 + 3*4 = 16), so the
    # serialized pair finishes at the same instant the concurrent pair would have.
    a, b = op(0, 1, 4.0), op(0, 3, 4.0)
    T._widen_rates_for_tb_serialization(build([a, b]))
    assert a.emit_rate == 16.0 and b.emit_rate == 16.0, (a.emit_rate, b.emit_rate)
    print("  [OK] rate widening: a same-epoch group emits its total rate")

    # A lone op at an epoch is already the whole group -- nothing to compensate, and it must
    # keep emitting cnt * piece_rate (emit_rate stays None) so unaffected XML is unchanged.
    a, b = op(0, 1, 4.0), op(1, 1, 4.0)
    T._widen_rates_for_tb_serialization(build([a, b]))
    assert a.emit_rate is None and b.emit_rate is None
    print("  [OK] rate widening: different epochs are not a group")

    # Different flows sharing a threadblock at one epoch may leave the gpu on different physical
    # ports, so they are widened only against themselves, never summed together.
    a, b = op(0, 1, 4.0, 'p'), op(0, 1, 4.0, 'q')
    T._widen_rates_for_tb_serialization(build([a, b]))
    assert a.emit_rate is None and b.emit_rate is None
    print("  [OK] rate widening: different flows are not summed together")

    # An unpaced op carries ordering only and was never sized against an epoch.
    a, b = op(0, 1, None), op(0, 1, None)
    T._widen_rates_for_tb_serialization(build([a, b]))
    assert a.emit_rate is None and b.emit_rate is None
    print("  [OK] rate widening: unpaced ops are skipped")


if __name__ == '__main__':
    main()
