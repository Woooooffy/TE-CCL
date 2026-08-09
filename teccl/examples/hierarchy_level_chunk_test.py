"""
Oracles for the level-chunk boundary: ChunkScale.coarsen, level_chunk_units (the GCD rule), and
CoarseTopology.rescale_to_chunk.

This is the ASCENDING half of the recursion -- abstraction re-expressing a coarse level in its own
chunk unit -- and it is the mirror of the refinement the identity resolution already does when
descending. The properties that matter are the ones a silent unit mismatch would break:

  1. coarsen/refine round-trip exactly, and both conserve payload_per_gpu, so a level boundary can
     be crossed in either direction without bytes drifting.
  2. refinement_from_root is a RATIO: coarsening credits the budget back, so a coarsen-by-g
     followed by a clean refine-by-g nets to 1 and spends nothing from ncclize's chunk_up budget.
  3. The GCD is the coarsest unit keeping every demand whole -- and for heterogeneous cells it is
     NOT any single cell's payload, which is the case a largest/smallest rule would get wrong.
  4. g == 1 is an exact no-op everywhere (the graceful-degradation path for coprime volumes).
  5. Rescaling the topology moves the epoch by exactly g and leaves per-epoch link capacity, in the
     level's own units, invariant. That equality is the whole point: it is what makes "one chunk on
     the selected link" true at every level for both LP and MILP.

Deliberately Gurobi-free AND numpy-free (the fine demand is built by hand), so it runs anywhere:

    python -m teccl.examples.hierarchy_level_chunk_test
"""
import json
from fractions import Fraction

from teccl.hierarchy.abstract import (
    abstract, coarsify_demand, level_chunk_units, rescale_demand, set_level_chunk,
)
from teccl.hierarchy.scale import ChunkScale
from teccl.input_data import TopologyParams
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster
from teccl.topologies.rail_optimized_spine_leaf import RailOptimizedSpineLeaf


def _all_gather_demand(topo):
    """build_demand(ALLGATHER, topo, 1) as nested lists -- numpy-free so this runs locally."""
    n = len(topo.capacity)
    parts = [d for d in range(n)
             if d not in set(topo.switch_indices) and d not in set(topo.passive_indices)]
    dem = [[[0] for _ in range(n)] for _ in range(n)]
    for s in parts:
        for t in parts:
            if s != t:
                dem[s][t][0] = 1
    return dem, parts


def _all_to_all_demand(topo):
    """build_demand(ALLTOALL, topo, len(parts)) as nested lists: chunk index encodes the target."""
    n = len(topo.capacity)
    parts = [d for d in range(n)
             if d not in set(topo.switch_indices) and d not in set(topo.passive_indices)]
    idx = {d: i for i, d in enumerate(parts)}
    dem = [[[0] * len(parts) for _ in range(n)] for _ in range(n)]
    for s in parts:
        for t in parts:
            if s != t:
                dem[s][t][idx[t]] = 1
    return dem, parts


def test_scale_round_trip():
    root = ChunkScale(bytes_per_chunk=1.0, num_chunks=1)
    assert root.refinement_from_root == 1 and root.payload_per_gpu == 1.0

    up = root.coarsen(8)
    assert up.bytes_per_chunk == 8.0
    assert up.num_chunks == Fraction(1, 8)
    assert up.refinement_from_root == Fraction(1, 8)
    # The invariant that makes the ascent legal at all.
    root.assert_conserves(up)
    assert abs(up.payload_per_gpu - 1.0) < 1e-12

    # A clean descent (q == g) returns to the root and spends nothing from the chunk_up budget.
    back = up.refine(8)
    assert back.bytes_per_chunk == 1.0 and back.num_chunks == 1
    assert back.refinement_from_root == 1, back

    # A descent that also absorbs a relaxation leaves exactly that relaxation on the budget.
    relaxed = up.refine(16)
    assert relaxed.refinement_from_root == 2, relaxed
    assert abs(relaxed.bytes_per_chunk - 0.5) < 1e-12
    root.assert_conserves(relaxed)

    # g == 1 / q == 1 are exact no-ops (identity, not a copy).
    assert root.coarsen(1) is root and root.refine(1) is root

    for bad in (0, -2, 1.5):
        for op in ("coarsen", "refine"):
            try:
                getattr(root, op)(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{op}({bad!r}) should have been rejected")

    # The epoch scales with the chunk, in both directions, and takes whichever link the caller
    # selects -- there is no built-in fastest/slowest preference any more.
    assert abs(root.epoch_duration(50.0) - 0.02) < 1e-12
    assert abs(up.epoch_duration(50.0) - 0.16) < 1e-12
    assert abs(up.epoch_duration(400.0) - 0.02) < 1e-12
    print("  [1] ChunkScale coarsen/refine round-trip, payload conserved, budget is a ratio OK")


def test_gcd_rule():
    # Uniform cells: the GCD is exactly one host's payload.
    rail = RailOptimizedSpineLeaf(TopologyParams(name="RailOptimizedSpineLeaf", chunk_size=1))
    _, mapping = abstract(rail)
    dem, parts = _all_gather_demand(rail)
    cd = coarsify_demand(dem, mapping)
    vols = {v[0] for row in cd for v in row if v[0]}
    assert vols == {8}, vols
    assert level_chunk_units(cd) == 8

    # Heterogeneous cells (4/4/6): the GCD is 2, which is NOT any cell's payload. A "largest
    # chunk" rule would pick 6 and leave the 4-cells at 4/6 of an epoch; "smallest" would pick 4
    # and leave the 6-cell at 1.5. Only the common divisor keeps every demand whole.
    het = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    _, hm = abstract(het)
    hdem, hparts = _all_gather_demand(het)
    hcd = coarsify_demand(hdem, hm)
    hvols = sorted({v[0] for row in hcd for v in row if v[0]})
    assert hvols == [4, 6], hvols
    g = level_chunk_units(hcd)
    assert g == 2, g
    scaled = rescale_demand(hcd, g)
    assert sorted({v[0] for row in scaled for v in row if v[0]}) == [2, 3]

    # AllToAll on the same cells: identities are per-destination-GPU distinct, so the volumes are
    # |U|*|V| and the GCD is a different number. It must be derived, never assumed to be a cell size.
    a2a, _ = _all_to_all_demand(het)
    acd = coarsify_demand(a2a, hm)
    avols = sorted({v[0] for row in acd for v in row if v[0]})
    ag = level_chunk_units(acd)
    assert all(v % ag == 0 for v in avols), (avols, ag)
    rescale_demand(acd, ag)      # must not raise

    # Coprime volumes -> 1 -> everything downstream is an exact no-op.
    coprime = [[[0], [3]], [[5], [0]]]
    assert level_chunk_units(coprime) == 1
    assert rescale_demand(coprime, 1) is coprime
    assert level_chunk_units([[[0]]]) == 1        # no demand at all
    print(f"  [2] GCD rule: rail AG={8}, hetero AG={g} (cells are 4/4/6), hetero A2A={ag}, "
          f"coprime/empty -> 1 OK")


def test_rescale_topology():
    rail = RailOptimizedSpineLeaf(TopologyParams(name="RailOptimizedSpineLeaf", chunk_size=1))
    coarse, mapping = abstract(rail)
    dem, _ = _all_gather_demand(rail)
    cd = coarsify_demand(dem, mapping)

    before_slow = coarse.get_epoch_duration_slow_link()
    before_fast = coarse.get_epoch_duration_fast_link()
    host, leaf = 0, 32
    cap_before = coarse.capacity[host][leaf]
    alpha_before = coarse.alpha[host][leaf]
    # Volume one host uplink can carry in one epoch, in that level's own chunk unit. THIS is the
    # quantity that must not move: it is the sense in which the epoch is "one chunk on the link".
    per_epoch_before = cap_before * before_slow

    scaled, g, level = set_level_chunk(coarse, cd)
    assert g == 8, g

    assert abs(coarse.get_epoch_duration_slow_link() - before_slow * g) < 1e-12
    assert abs(coarse.get_epoch_duration_fast_link() - before_fast * g) < 1e-12
    assert abs(coarse.get_epoch_duration_slow_link() - 0.16) < 1e-12
    assert abs(coarse.capacity[host][leaf] - cap_before / g) < 1e-12
    assert coarse.chunk_size == g
    assert coarse.alpha[host][leaf] == alpha_before, "alpha is a time; it must not scale"

    per_epoch_after = coarse.capacity[host][leaf] * coarse.get_epoch_duration_slow_link()
    assert abs(per_epoch_after - per_epoch_before) < 1e-12, (per_epoch_before, per_epoch_after)
    assert abs(per_epoch_after - 1.0) < 1e-12, per_epoch_after

    # Demand and topology moved together: a pair's demand is now exactly one chunk, so a host's
    # whole payload for one peer fits one epoch on one uplink. That is the degeneracy fix, stated
    # as an equality rather than a hope.
    assert {v[0] for row in scaled for v in row if v[0]} == {1}
    assert level.bytes_per_chunk == coarse.chunk_size
    assert level.refinement_from_root == Fraction(1, 8)

    # Zero-capacity entries stay exactly zero (non-links must not become tiny positives).
    assert coarse.capacity[host][host] == 0.0
    print(f"  [3] rescale_to_chunk: epoch {before_slow} -> {coarse.get_epoch_duration_slow_link()} "
          f"(x{g}), per-epoch link capacity invariant at 1 chunk, demand -> 1/pair OK")


def test_rescaled_level_is_solver_ready():
    """After set_level_chunk, the level must be handable to a formulation AS IS.

    This reads the cached ATTRIBUTES, not the getters, because that is what
    `BaseFormulation.set_epoch_duration` does -- it never calls
    `get_epoch_duration_{fast,slow}_link`. `rescale_to_chunk` invalidates those caches, so a level
    that only invalidates and waits for a lazy recompute hands the solver a 0 and dies on
    `assert self.epoch_duration > 0, "Epoch Multiplier in the user input is not positive"` -- an
    error message that points at the wrong thing entirely.

    That is not hypothetical: it worked for months only because a driver's log line happened to
    call the getter between the rescale and the solve, and it broke the moment that print was
    removed. Every other test here goes through the getters and so cannot see it.
    """
    for topo in (RailOptimizedSpineLeaf(TopologyParams(name="RailOptimizedSpineLeaf", chunk_size=1)),
                 HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))):
        coarse, mapping = abstract(topo)
        dem, _ = _all_gather_demand(topo)
        _scaled, g, _level = set_level_chunk(coarse, coarsify_demand(dem, mapping))
        assert coarse.epoch_duration_fast_link > 0, (
            f"{type(topo).__name__} (g={g}): cached fast-link epoch is "
            f"{coarse.epoch_duration_fast_link} after rescaling; a formulation reading the "
            f"attribute would assert on a non-positive epoch")
        assert coarse.epoch_duration_slow_link > 0, (
            f"{type(topo).__name__} (g={g}): cached slow-link epoch is "
            f"{coarse.epoch_duration_slow_link} after rescaling")
        # and the cached value is the RESCALED one, not a stale pre-rescale leftover
        assert abs(coarse.epoch_duration_slow_link
                   - coarse.get_epoch_duration_slow_link()) < 1e-12
    print("  [4] rescaled level is solver-ready: cached epoch attributes positive and current "
          "(BaseFormulation reads them directly, never via the getter) OK")


def test_g1_is_a_noop():
    """The whole change must be byte-identical when the GCD is 1 -- that is the flat-solve
    fallback and the coprime-volume path, and it is the only guarantee that this cannot regress a
    topology it was never meant to touch."""
    het = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    coarse, mapping = abstract(het)
    dem, _ = _all_gather_demand(het)
    cd = coarsify_demand(dem, mapping)

    slow, fast = coarse.get_epoch_duration_slow_link(), coarse.get_epoch_duration_fast_link()
    cap = [row[:] for row in coarse.capacity]
    chunk = coarse.chunk_size

    scaled, g, level = set_level_chunk(coarse, cd, g=1)
    assert g == 1
    assert scaled is cd, "demand must be returned unchanged, not rebuilt"
    assert coarse.capacity == cap and coarse.chunk_size == chunk
    assert coarse.get_epoch_duration_slow_link() == slow
    assert coarse.get_epoch_duration_fast_link() == fast
    assert level.refinement_from_root == 1 and level.bytes_per_chunk == chunk
    print("  [5] g == 1 is an exact no-op (topology, demand, scale all unchanged) OK")


def test_scale_is_serializable():
    """The driver writes the live scale into Schedules/{prefix}_identities.json, and that path is
    reachable only through a Gurobi solve -- so it is exactly the kind of code the structural tests
    never touch. It escaped once already: `dataclasses.asdict` passes the Fractions through raw and
    json.dump died on them ONLY on the remote, after the coarse LP had already been paid for.
    Serialize every scale a level boundary can produce, here, for free."""
    root = ChunkScale(bytes_per_chunk=1.0, num_chunks=1)
    for scale in (root, root.refine(2), root.coarsen(8), root.coarsen(8).refine(16),
                  root.coarsen(2).refine(2)):
        blob = json.dumps({"scale": scale.to_json()}, indent=2)
        back = json.loads(blob)["scale"]
        assert Fraction(back["refinement_from_root"]["num"],
                        back["refinement_from_root"]["den"]) == scale.refinement_from_root
        assert Fraction(back["num_chunks"]["num"],
                        back["num_chunks"]["den"]) == scale.num_chunks
        assert abs(back["payload_per_gpu"] - 1.0) < 1e-12, (scale, back)
    print("  [6] every level-boundary scale round-trips through JSON exactly OK")


def main() -> None:
    print("level-chunk boundary tests (ascending half of the recursion)")
    test_scale_round_trip()
    test_gcd_rule()
    test_rescale_topology()
    test_rescaled_level_is_solver_ready()
    test_g1_is_a_noop()
    test_scale_is_serializable()
    print("level-chunk tests OK")


if __name__ == "__main__":
    main()
