"""Unit tests for send-pacing gate derivation (_finish_before_start_gates).

The gate manifest decides which paced send waits on which so that a rate-paced send
holds to its intended epoch on real hardware. The edge is derived from each send's
LINK-OCCUPANCY WINDOW (start, finish=start+volume/rate), NOT from step order -- which
is what makes it correct for two cases a plain sort gets wrong:

  1. Concurrent sends (same epoch, different peers/channels/uplinks) must never be
     serialized against each other.
  2. A send paced to span multiple epochs must not serialize a later send that runs
     alongside its tail on capacity a *different*, earlier send already freed.

Run from the repo root (in an env with the ncclize deps, or none -- this file imports
only teccl_ncclize, which needs lxml/z3 for the rest of the module):
    python teccl/ncclize/pacing_gates_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teccl_ncclize import _finish_before_start_gates as gates  # noqa: E402


def check(name, paced, expect):
    got = sorted(gates(paced))
    if got != sorted(expect):
        raise AssertionError(f"{name}: expected {sorted(expect)}, got {got}")
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
          [((6, 0, 1, None), (5, 0, 1, None))])

    # Concern 2: X spans epochs 0..2 (half rate, finish 2); Y in epoch 0 finishes at 1;
    # Z starts at epoch 1. Z runs alongside X's tail on the half Y just freed, so Z must
    # gate on Y (finish 1 <= 1), NOT on the still-running X (finish 2 > 1).
    check("concern2: multi-epoch send does not serialize a later one",
          {(0, 0, 1, None): (0, 2),   # X
           (0, 0, 2, None): (0, 1),   # Y
           (1, 0, 3, None): (1, 2)},  # Z
          [((1, 0, 3, None), (0, 0, 2, None))])

    # A gap still pins to the latest send that finishes in time (best-effort; the
    # residual earliness is the deferred-P3 case the stitch reports).
    check("gap: pin to the latest finishing predecessor",
          {(0, 0, 1, None): (0, 1), (3, 0, 1, None): (3, 4)},
          [((3, 0, 1, None), (0, 0, 1, None))])

    # Sends from different source GPUs share no uplink -> never gate each other.
    check("cross-gpu: independent uplinks, no edges",
          {(0, 0, 1, None): (0, 1), (1, 7, 1, None): (1, 2)},
          [])

    # Multi-uplink: one GPU, two sends entering the fabric at DIFFERENT first
    # switches (path (9,..) vs (10,..)) use different uplinks, so even across an
    # epoch boundary they do NOT gate each other.
    check("multi-uplink: different first switch, independent, no edges",
          {(0, 0, 1, (9, 5)): (0, 1), (1, 0, 2, (10, 5)): (1, 2)},
          [])

    # Same first switch (same uplink) DOES chain across the epoch boundary, even
    # when the downstream switch path differs.
    check("same uplink, different downstream path, still chains",
          {(0, 0, 1, (9, 5)): (0, 1), (1, 0, 2, (9, 6)): (1, 2)},
          [((1, 0, 2, (9, 6)), (0, 0, 1, (9, 5)))])

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
    """_realize_pacing_gates puts the edge on net_dep (never depends), and
    _assert_gates_acyclic catches the deadlock the runtime would otherwise hang on."""
    T = _load_taccl_ncclize()

    def send(gpu, peer, step):
        return T._Op(gpu, peer, step, True, 's', 'i', 0, 'o', 0, 1, [])

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

    # Cross-threadblock gate lands on net_dep and leaves depends untouched: a gate is
    # not a data dependency and must not become one (nor an expansion nop).
    a, b = send(0, 1, 0), send(0, 2, 1)
    gpus = build([(0, [a]), (1, [b])])
    T._realize_pacing_gates(gpus, [((1, 0, 2, None), (0, 0, 1, None))])
    assert b.net_dep is a, "gate not realized onto net_dep"
    assert b.depends == [] and a.depends == [], "gate leaked into depends"
    assert not a.has_dependence, "gate producer must not need a kernel dep flag"
    print("  [OK] realization: cross-tb gate lands on net_dep, not depends")

    # Two gated sends in ONE threadblock are an ordinary schedule: both survive.
    p0, p1 = send(0, 1, 0), send(0, 1, 2)
    c0, c1 = send(0, 2, 1), send(0, 2, 3)
    gpus = build([(0, [p0, p1]), (1, [c0, c1])])
    T._realize_pacing_gates(gpus, [((1, 0, 2, None), (0, 0, 1, None)),
                                   ((3, 0, 2, None), (2, 0, 1, None))])
    assert (c0.net_dep, c1.net_dep) == (p0, p1), "a threadblock may carry several gates"
    print("  [OK] realization: several gates in one threadblock all survive")

    # Same threadblock on both ends: one threadblock is one connection, whose send
    # FIFO already orders its own sends, so the edge is dropped as redundant.
    s0, s1 = send(0, 1, 0), send(0, 1, 1)
    gpus = build([(0, [s0, s1])])
    T._realize_pacing_gates(gpus, [((1, 0, 1, None), (0, 0, 1, None))])
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
