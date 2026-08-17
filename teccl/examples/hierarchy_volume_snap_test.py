"""
Oracles for the float -> exact boundary in identity resolution: snap_tolerance, _snap_group,
_snap_volumes, and the exactness they buy _subdivision_factor / _emit_refined.

WHAT BROKE, AND WHY IT NEEDED A STRUCTURAL FIX RATHER THAN A LOOSER EPSILON.
`_subdivision_factor` chose q by snapping each volume with `Fraction(v).limit_denominator(64)`,
which tolerates error up to ~1.2e-4. `_emit_refined` then re-checked the RAW float against
`volume * q` with a fixed 1e-6 -- a tolerance of 1e-6/q on the volume itself, ~8e-9 at q=120. The
two steps disagreed about the same number by four orders of magnitude, and the disagreement GREW
with q, so a coarse solve with non-dyadic volumes failed a check its own snap had already passed:

    AssertionError: volume 0.466666 is not a whole number of 1/120 sub-chunks

0.466666 IS 7/15 to within Gurobi's feasibility_tol; 0.466666 * 120 = 55.99992 is not 56 to within
1e-6. No single epsilon fixes that, because the error is absolute at the solver and gets multiplied
by q downstream. The fix is to spend the tolerance exactly once -- at the snap -- and hold exact
`Fraction`s after it, so every later step is exact arithmetic with no epsilon to get wrong.

The properties asserted here:
  1. MAX_DENOM and the solver tolerance are COUPLED by grid_resolution = 1/(2 D^2), and
     snap_tolerance enforces the coupling (raises when the solver is coarser than the grid).
  2. The reported regression snaps cleanly and the group sums to EXACTLY 1.
  3. Snapping is group-aware: it repairs a sum that independent rounding would break.
  4. Downstream is exact -- q is a multiple of every snapped denominator, so counts sum to q with
     no tolerance anywhere.
  5. The three failure modes are DISTINGUISHABLE: off-grid volume, non-partitioning group, and
     too-finely-split solution each raise their own message.

Deliberately Gurobi-free (a hand-built _Assignment needs no solver):

    python -m teccl.examples.hierarchy_volume_snap_test
"""
from dataclasses import replace
from fractions import Fraction

from teccl.hierarchy.reconstruct import (
    MAX_DENOM, MAX_SUBDIVISION, _Assignment, _CoarsePiece, _snap_group, _snap_volumes,
    _subdivision_factor, grid_resolution, snap_tolerance,
)


class _Solver:
    """Minimal stand-in for a solved formulation: snap_tolerance reads only this attribute."""

    def __init__(self, feasibility_tol):
        self.feasibility_tol = feasibility_tol


def _assignment(volume, identity=(0, 0), dst_cell=1):
    piece = _CoarsePiece(src_cell=0, dst_cell=dst_cell, egress_neighbor=2, ingress_neighbor=2,
                         via_switches=(2,), volume=volume, send_epoch=0, arrival_epoch=0)
    return _Assignment(src_cell=0, dst_cell=dst_cell, identity=identity, piece=piece,
                       egress_gpu=0, ingress_gpu=1, volume=volume)


def _raises(fn, needle):
    try:
        fn()
    except (AssertionError, ValueError, RuntimeError) as exc:
        assert needle in str(exc), f"expected {needle!r} in the message, got: {exc}"
        return str(exc)
    raise AssertionError(f"expected a raise mentioning {needle!r}, but the call succeeded")


def test_tolerance_is_coupled_to_the_grid():
    """MAX_DENOM is not a free constant: it is bounded by the solver tolerance, and vice versa."""
    # Two distinct rationals with denominator <= D differ by at least 1/D^2, so the unambiguous
    # snap radius is half that.
    for d in (16, 32, 64, 128):
        assert grid_resolution(d) == 1.0 / (2.0 * d * d)
        # Exhibit the witness: two distinct grid points really are that close.
        a, b = Fraction(1, d), Fraction(1, d - 1)
        assert abs(float(a - b)) >= 2 * grid_resolution(d) - 1e-18

    # A solver at least 4x tighter than the grid: accepted verbatim, no warning.
    assert snap_tolerance(_Solver(1e-6)) == 1e-6
    # No solver at all (closed-form row, test shim): demand the comfortable margin.
    assert snap_tolerance(None) == grid_resolution(MAX_DENOM) / 4.0
    # Coarser than the grid can resolve: snapping could pick the WRONG grid point, so refuse.
    msg = _raises(lambda: snap_tolerance(_Solver(1e-3)), "coarser than the identity grid")
    assert "lower MAX_DENOM to 22" in msg, msg
    # The default pairing (feasibility_tol 1e-4, MAX_DENOM 64) is legal but thin -- 1e-4 against a
    # 1.22e-4 radius. It must still be accepted; the thinness is a warning, not an error.
    assert snap_tolerance(_Solver(1e-4)) == 1e-4
    print("  [1] MAX_DENOM <-> feasibility_tol coupling enforced by snap_tolerance OK")


def test_reported_regression_snaps():
    """The exact volumes from the failing run: 0.466666 -> 7/15, and the group sums to exactly 1."""
    vols = [0.466666, 0.533334]
    tol = snap_tolerance(_Solver(1e-4))
    fracs = _snap_group(vols, tol, ((0, 0), 1))
    assert fracs == [Fraction(7, 15), Fraction(8, 15)], fracs
    assert sum(fracs) == 1, sum(fracs)          # exact, not "within 1e-6"
    assert isinstance(sum(fracs), Fraction)

    # And the old failure mode is gone end to end: q is a multiple of 15 and the counts are exact.
    asg = _snap_volumes([_assignment(v) for v in vols], tol)
    q = _subdivision_factor(asg)
    assert q % 15 == 0, q
    counts = [a.exact * q for a in asg]
    assert all(c.denominator == 1 for c in counts), counts
    assert sum(counts) == q, (counts, q)
    print("  [2] the reported 0.466666 / q=120 regression snaps exactly OK")


def test_snapping_is_group_aware():
    """Independent rounding breaks a sum of 1; the largest-remainder repair restores it exactly."""
    # Floors alone would give 1 + 1 + 1 = 3 sub-chunks on a grid of 4 -- one short. The repair must
    # give the missing unit to the largest fractional remainder (the 0.5 entry).
    vols = [0.4999999, 0.2500001, 0.25]
    tol = snap_tolerance(_Solver(1e-4))
    fracs = _snap_group(vols, tol, ((0, 0), 1))
    assert sum(fracs) == 1, fracs
    assert fracs == [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)], fracs

    # Same property over a three-way split whose floors lose two units.
    thirds = [1 / 3 + 1e-9, 1 / 3, 1 / 3 - 1e-9]
    fr3 = _snap_group(thirds, tol, ((0, 1), 1))
    assert sum(fr3) == 1, fr3
    assert all(f == Fraction(1, 3) for f in fr3), fr3
    print("  [3] group-aware snapping repairs the partition sum exactly OK")


def test_downstream_is_exact():
    """q is a multiple of every snapped denominator, so no downstream step needs a tolerance."""
    tol = snap_tolerance(_Solver(1e-6))
    # Two identities with incompatible denominators (thirds and quarters) -> q = 12.
    asg = _snap_volumes(
        [_assignment(1 / 3, identity=(0, 0)), _assignment(2 / 3, identity=(0, 0)),
         _assignment(1 / 4, identity=(0, 1)), _assignment(3 / 4, identity=(0, 1))], tol)
    q = _subdivision_factor(asg)
    assert q == 12, q
    for a in asg:
        assert (a.exact * q).denominator == 1, (a.exact, q)
    # Each identity's counts partition q -- this is _emit_refined's `cursor == q`, in exact form.
    for identity in ((0, 0), (0, 1)):
        counts = [int(a.exact * q) for a in asg if a.identity == identity]
        assert sum(counts) == q, (identity, counts, q)
    print("  [4] exact Fraction arithmetic downstream of the snap OK")


def test_failure_modes_are_distinguishable():
    """Off-grid, non-partitioning, and too-finely-split must not all look like float noise."""
    tol = snap_tolerance(_Solver(1e-6))

    # (a) A volume genuinely off the 1/MAX_DENOM grid: 1/97 needs a denominator above MAX_DENOM.
    off = 1.0 / 97.0
    _raises(lambda: _snap_group([off, 1 - off], tol, ((0, 0), 1)),
            f"not within {tol:g} of any rational with denominator <= {MAX_DENOM}")

    # (b) Volumes that do not partition the identity at all -- an assignment defect, not rounding.
    _raises(lambda: _snap_group([0.5, 0.2], tol, ((0, 0), 1)), "not 1")

    # (c) Legal volumes whose LCM exceeds MAX_SUBDIVISION: the coarse solve is split too finely.
    #     Denominators 61, 62, 63 are each <= MAX_DENOM but their LCM is far above the ceiling.
    fine = _snap_volumes([_assignment(1.0, identity=(0, i)) for i in range(3)], tol)
    fine = [replace(a, exact=Fraction(1, d)) for a, d in zip(fine, (61, 62, 63))]
    _raises(lambda: _subdivision_factor(fine), f"MAX_SUBDIVISION={MAX_SUBDIVISION}")
    print("  [5] off-grid / non-partition / too-finely-split raise distinct errors OK")


def main() -> None:
    print("identity-resolution volume snapping (the float -> exact boundary)")
    test_tolerance_is_coupled_to_the_grid()
    test_reported_regression_snaps()
    test_snapping_is_group_aware()
    test_downstream_is_exact()
    test_failure_modes_are_distinguishable()
    print("volume snap tests OK")


if __name__ == "__main__":
    main()
