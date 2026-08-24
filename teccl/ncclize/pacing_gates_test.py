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

    realization_tests()
    print("pacing gate tests OK")


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


if __name__ == '__main__':
    main()
