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

    print("pacing gate tests OK")


if __name__ == '__main__':
    main()
