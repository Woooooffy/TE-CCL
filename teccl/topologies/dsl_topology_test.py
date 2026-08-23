"""DslTopology parity: a .topo file must build the SAME topology as the class it ports.

Four of the DSL frontend's examples were written as ports of hand-written TE-CCL classes, which
makes them the strongest test available: capacity, ports, switch/programmable sets, the port map
and the hierarchy cells are all compared element-wise against the Python class (alpha is not a
parity claim -- see UNIFORM_ALPHA below). A mismatch is either a real bug
here or a deliberate divergence in the .topo file -- and the deliberate ones are listed in
EXPECTED_DIVERGENCES rather than being quietly tolerated, so the list stays reviewable.

Run: conda run -n teccl python teccl/topologies/dsl_topology_test.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from teccl.input_data import TopologyParams
from teccl.topologies.dsl_topology import DslTopology
from teccl.topologies.fat_tree_pod import FatTreePod
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster
from teccl.topologies.nested_cluster import NestedCluster
from teccl.topologies.rail_optimized_spine_leaf import RailOptimizedSpineLeaf
from teccl.topologies.topology import Topology
from teccl.topologies.two_pod_rail import TwoPodRailHostBound

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "topology-dsl-frontend", "examples")

# ALPHA IS NOT A PARITY CLAIM. The .topo files give every link the same 700 ns propagation delay,
# while the classes give intra-host links a shorter one (nvswitch_alpha = 0.35e-6); NestedCluster
# uses 0 throughout. Nothing measured here turns on it -- every binding constraint in these
# topologies is a capacity cut, and the three-level hierarchical solve is byte-identical either
# way -- so the files keep one value rather than tracking each class. What IS asserted is that
# the rule holds: check_uniform_alpha below requires every link of every ported file to be
# exactly UNIFORM_ALPHA, which is a stronger statement than "differs from the class somewhere".
UNIFORM_ALPHA = 700e-9

# Where a file deliberately disagrees with the class it ports, and why.
EXPECTED_DIVERGENCES = {
    # HeteroTaperedCluster does not override default_programmable_switch_indices, so it leaves
    # its per-host NVSwitches in the emitted forwarding table -- unlike RailOptimizedSpineLeaf
    # and TwoPodRail, which both exclude them because an NVSwitch self-routes and never takes a
    # forwarding entry. The DSL follows the latter (its `nvswitch` type IS that distinction), so
    # it programs only the two top switches. The class is the odd one out here, not the DSL.
    ("hetero_tapered_cluster.topo", "programmable_switch_indices"):
        "class programs its NVSwitches too; the DSL programs only `switch` nodes",
    ("nested_cluster.topo", "programmable_switch_indices"):
        "class programs its NVSwitches too; the DSL programs only `switch` nodes",
}

# (.topo file, class, chunk_size, {field: reason} of fields to skip)
CASES = [
    ("two_pod_rail_hostbound_splitports.topo", TwoPodRailHostBound, 1.0, {}),
    ("hetero_tapered_cluster.topo", HeteroTaperedCluster, 1.0, {}),
    ("rail_optimized_256gpu.topo", RailOptimizedSpineLeaf, 1.0, {}),
    # FatTreePod declares GPU twins ([0,1], [2,3]) alongside the spine twins. A GPU swap permutes
    # the source index, so the DSL's switch-restricted detector deliberately does not report them
    # (same restriction as hierarchy.abstract); the switch group [6, 7] must still match.
    ("fat_tree_pod.topo", FatTreePod, 1.0, {"equivalent_node_indices": "GPU twins, see below"}),
    # Three levels (cluster -> racks -> hosts): the only case where a cell has subcells, and so
    # the only one that tests nesting rather than a flat list of cells.
    ("nested_cluster.topo", NestedCluster, 1.0, {}),
]

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsl_test_files")

# Built, not compared -- no Python class to compare against.
SMOKE = ["fat_tree_pod_incast.topo", "two_pod_rail_hostbound.topo", "3gpus_ring.topo",
         "2gpus1sw.topo", "nvswitch_test.topo", "hubs1-multipath.topo"]


def build_dsl(filename: str, chunk_size: float = 1.0) -> DslTopology:
    return DslTopology(TopologyParams(name=filename, chunk_size=chunk_size,
                                      topo_file=os.path.join(EXAMPLES, filename)))


def _matrix_diff(a, b, names, label, limit=6):
    n = len(a)
    diffs = []
    for i in range(n):
        for j in range(n):
            if abs(a[i][j] - b[i][j]) > 1e-9:
                diffs.append(f"  {label}[{names[i]}({i})][{names[j]}({j})]: dsl {a[i][j]} != "
                             f"class {b[i][j]}")
    if len(diffs) > limit:
        diffs = diffs[:limit] + [f"  ... and {len(diffs) - limit} more"]
    return diffs


def check_parity(filename: str, cls, chunk_size: float, skip: dict) -> None:
    dsl = build_dsl(filename, chunk_size)
    ref = cls(TopologyParams(name=cls.__name__, chunk_size=chunk_size))
    names = dsl.node_names

    assert len(dsl.capacity) == len(ref.capacity), \
        f"{filename}: {len(dsl.capacity)} nodes, {cls.__name__} has {len(ref.capacity)}"

    diffs = _matrix_diff(dsl.capacity, ref.capacity, names, "capacity")
    assert not diffs, f"{filename}: capacity differs from {cls.__name__}\n" + "\n".join(diffs)

    check_uniform_alpha(filename, dsl, names)

    n = len(dsl.capacity)
    for i in range(n):
        for j in range(n):
            assert dsl.port_count(i, j) == ref.port_count(i, j), (
                f"{filename}: ports[{names[i]}][{names[j]}] = {dsl.port_count(i, j)}, "
                f"{cls.__name__} says {ref.port_count(i, j)}")

    assert dsl.switch_indices == sorted(ref.switch_indices), \
        f"{filename}: switch_indices {dsl.switch_indices} != {sorted(ref.switch_indices)}"
    programmable_same = (sorted(dsl.programmable_switch_indices)
                         == sorted(ref.programmable_switch_indices))
    if (filename, "programmable_switch_indices") in EXPECTED_DIVERGENCES:
        assert not programmable_same, (
            f"{filename}: EXPECTED_DIVERGENCES claims a programmable-switch mismatch but there "
            f"is none -- drop the entry")
        # Whatever the class does, the DSL must program exactly the `switch`-typed nodes.
        assert sorted(dsl.programmable_switch_indices) == dsl.network_switch_indices
    else:
        assert programmable_same, (
            f"{filename}: programmable {sorted(dsl.programmable_switch_indices)} != "
            f"{sorted(ref.programmable_switch_indices)}")

    # The port map is where the DSL's declaration order meets the base class's ascending default.
    # They must agree on a file that ports a class, or the emitted forwarding table would name
    # different sockets for the same fabric.
    assert set(dsl.port_map) == set(dsl.programmable_switch_indices), \
        f"{filename}: the port map must cover exactly the programmable switches"
    if (filename, "programmable_switch_indices") not in EXPECTED_DIVERGENCES:
        assert set(dsl.port_map) == set(ref.port_map), \
            f"{filename}: port-mapped nodes differ"
    for node in set(dsl.port_map) & set(ref.port_map):
        assert dsl.port_map[node] == ref.port_map[node], (
            f"{filename}: port map of {names[node]}({node}) differs\n"
            f"  dsl:   {sorted(dsl.port_map[node].items(), key=lambda kv: kv[1])}\n"
            f"  class: {sorted(ref.port_map[node].items(), key=lambda kv: kv[1])}")

    if "equivalent_node_indices" not in skip:
        assert dsl.equivalent_node_indices == sorted(
            (sorted(g) for g in ref.equivalent_node_indices), key=lambda g: g[0]), (
            f"{filename}: twins {dsl.equivalent_node_indices} != {ref.equivalent_node_indices}")
    else:
        # Every switch group the class declares must still be found.
        switches = set(dsl.switch_indices)
        for group in ref.equivalent_node_indices:
            if set(group) <= switches:
                assert sorted(group) in dsl.equivalent_node_indices, (
                    f"{filename}: {cls.__name__} declares switch twins {group}, which the DSL's "
                    f"detector missed ({dsl.equivalent_node_indices})")
        for group in dsl.equivalent_node_indices:
            assert set(group) <= switches, \
                f"{filename}: inferred a non-switch twin group {group}"

    check_cells(filename, dsl.cells, ref.cells, names, path=cls.__name__)

    print(f"  OK  {filename:<48} == {cls.__name__} ({n} nodes)"
          + ("  [programs fewer switches than the class, expected]"
             if (filename, "programmable_switch_indices") in EXPECTED_DIVERGENCES else ""))


def check_uniform_alpha(filename, dsl, names) -> None:
    """Every link of a ported file carries exactly UNIFORM_ALPHA -- see the note above."""
    n = len(dsl.capacity)
    wrong = [f"  alpha[{names[i]}({i})][{names[j]}({j})] = {dsl.alpha[i][j]}"
             for i in range(n) for j in range(n)
             if dsl.capacity[i][j] > 0 and abs(dsl.alpha[i][j] - UNIFORM_ALPHA) > 1e-18]
    assert not wrong, (f"{filename}: these links are not at the uniform {UNIFORM_ALPHA}s the "
                       f"ported files are supposed to use\n" + "\n".join(wrong[:6]))


def check_cells(filename, got, want, names, path) -> None:
    """Cells must match the class's, recursively.

    `gpus` is compared as an ordered LIST because that order defines the sub-chunk <-> origin-GPU
    correspondence (see Cell); `members` and `internal_switches` are compared as SETS, since
    nothing reads their order and the two builders assemble them in different sweeps.
    """
    assert len(got) == len(want), \
        f"{filename}: {len(got)} cells at {path}, {len(want)} expected"
    for i, (g, w) in enumerate(zip(got, want)):
        where = f"{path}.cells[{i}]"
        assert g.gpus == w.gpus, f"{filename}: {where}.gpus {g.gpus} != {w.gpus}"
        assert set(g.members) == set(w.members), \
            f"{filename}: {where}.members {sorted(g.members)} != {sorted(w.members)}"
        assert set(g.internal_switches) == set(w.internal_switches), \
            f"{filename}: {where}.internal_switches {sorted(g.internal_switches)} != " \
            f"{sorted(w.internal_switches)}"
        assert {k: sorted(v) for k, v in g.boundary.items()} == \
            {k: sorted(v) for k, v in w.boundary.items()}, \
            f"{filename}: {where}.boundary {dict(g.boundary)} != {dict(w.boundary)}"
        check_cells(filename, g.subcells, w.subcells, names, path=where)


def check_node_attributes() -> None:
    """radix and passive come off the `node` statement, and both are enforced."""
    dsl = DslTopology(TopologyParams(name="attrs",
                                     topo_file=os.path.join(FIXTURES, "attrs.topo")))
    assert dsl.radix(3) == 4, dsl.radix(3)          # sw, declared radix=4
    assert dsl.radix(0) is None, "a node with no radix attribute does not claim a width"
    assert dsl.passive_indices == [2], dsl.passive_indices
    # TopologyParams passives are unioned with the file's, not replaced by them.
    both = DslTopology(TopologyParams(name="attrs", passive_node_indices=(1,),
                                      topo_file=os.path.join(FIXTURES, "attrs.topo")))
    assert both.passive_indices == [1, 2], both.passive_indices
    print("  OK  node attributes (radix, passive)")


def check_symmetry() -> None:
    """Declared groups, inferred groups, and the check that a declaration is honest."""
    dsl = DslTopology(TopologyParams(name="sym",
                                     topo_file=os.path.join(FIXTURES, "symmetry.topo")))
    # The GPU pairs are DECLARED (inference will not report a source-permuting group) and the
    # spine pair is both declared and inferred -- it must appear once, not twice. This is
    # FatTreePod's own [[0,1],[2,3],[6,7]], reached from the file instead of from Python.
    assert dsl.equivalent_node_indices == [[0, 1], [2, 3], [6, 7]], dsl.equivalent_node_indices
    print("  OK  symmetry (declared + inferred, deduplicated)")


def expect_failure(filename: str, needle: str) -> None:
    try:
        DslTopology(TopologyParams(name=filename, topo_file=os.path.join(FIXTURES, filename)))
    except (AssertionError, ValueError) as e:
        assert needle in str(e), f"{filename}: raised {e!r}, which does not mention {needle!r}"
        print(f"  OK  rejected {filename:<30} ({needle})")
        return
    raise AssertionError(f"{filename} was accepted; it must be rejected ({needle})")


def check_rejections() -> None:
    expect_failure("radix_overflow.topo", "radix")
    expect_failure("unequal_cables.topo", "parallel links")
    expect_failure("cell_switch_boundary.topo", "link outside the cell")
    expect_failure("symmetry_bad.topo", "not interchangeable")


def check_declaration_port_order(filename: str) -> None:
    """The declaration order must be a permutation of the node's connections (the base class
    asserts this) -- and here we also report whether it coincides with ascending order."""
    dsl = build_dsl(filename)
    same = all(dsl.port_order(node) == Topology.port_order(dsl, node) for node in dsl.port_map)
    print(f"  {'==' if same else '!='} declaration vs ascending port order: {filename}")


def check_units() -> None:
    from teccl.topologies.dsl_topology import _to_gigabytes_per_second, _to_seconds
    assert _to_gigabytes_per_second((400, "Gbps"), "t") == 50.0
    assert _to_gigabytes_per_second((3200, "Gbps"), "t") == 400.0
    assert _to_gigabytes_per_second((900, "GBps"), "t") == 900.0
    assert _to_gigabytes_per_second((8000, "Mbps"), "t") == 1.0
    assert _to_gigabytes_per_second((1000, "MBps"), "t") == 1.0
    assert abs(_to_gigabytes_per_second((8000000, "Kbps"), "t") - 1.0) < 1e-9
    assert abs(_to_gigabytes_per_second((1000000, "KBps"), "t") - 1.0) < 1e-9
    # Compared with a tolerance: 700 * 1e-9 and 700e-9 are different floats, and the difference
    # is 22 orders of magnitude below any latency this models.
    for number, unit, seconds in [(700, "ns", 700e-9), (350, "ns", 0.35e-6),
                                  (2, "us", 2e-6), (5, "ms", 5e-3)]:
        assert abs(_to_seconds((number, unit), "t") - seconds) < 1e-18, (number, unit)
    for bad, fn in [(700, _to_gigabytes_per_second), ((700, "ns"), _to_gigabytes_per_second),
                    (400, _to_seconds), ((400, "Gbps"), _to_seconds)]:
        try:
            fn(bad, "t")
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} accepted {bad!r}")
    print("  OK  units")


def check_chunk_size() -> None:
    """Capacity is in units of chunks/second, i.e. divided by chunk_size, like every class."""
    one = build_dsl("fat_tree_pod.topo", 1.0)
    half = build_dsl("fat_tree_pod.topo", 0.5)
    assert half.capacity[0][4] == one.capacity[0][4] * 2, \
        f"{half.capacity[0][4]} != 2 * {one.capacity[0][4]}"
    assert half.alpha[0][4] == one.alpha[0][4], "alpha must not scale with chunk_size"
    print("  OK  chunk_size scaling")


def check_split_ports() -> None:
    """The two-cable leaf<->spine0 edge: aggregate capacity, ports=2, and 2 leaf ports used."""
    dsl = build_dsl("two_pod_rail_hostbound_splitports.topo")
    leaf0, spine0, spine1 = 24, 28, 29
    assert dsl.capacity[leaf0][spine0] == 50.0, dsl.capacity[leaf0][spine0]
    assert dsl.port_count(leaf0, spine0) == 2 and dsl.port_count(leaf0, spine1) == 1
    assert dsl.port_capacity(leaf0, spine0) == 25.0
    assert dsl.physical_port(leaf0, spine0, 0) == 4 and dsl.physical_port(leaf0, spine0, 1) == 5
    assert dsl.physical_port(leaf0, spine1, 0) == 6
    assert len(dsl.port_map[spine0]) == 8, "spine0 is exactly full at 8 cables"
    # GPUs and NVSwitches carry no port map -- nothing numbered their sockets.
    assert dsl.physical_port(0, 16) is None and dsl.physical_port(16, 0) is None
    print("  OK  split ports (2 x 25 GBps on leaf<->spine0)")


def check_smoke() -> None:
    for filename in SMOKE:
        dsl = build_dsl(filename)
        assert len(dsl.capacity) > 0
        print(f"  OK  built {filename:<48} ({len(dsl.capacity)} nodes, "
              f"{len(dsl.switch_indices)} switches)")


def main() -> None:
    print("parity against the class each .topo ports:")
    for filename, cls, chunk_size, skip in CASES:
        check_parity(filename, cls, chunk_size, skip)
    print("port order:")
    for filename, _, _, _ in CASES:
        check_declaration_port_order(filename)
    print("cells and attributes:")
    check_node_attributes()
    check_symmetry()
    check_rejections()
    print("units and scaling:")
    check_units()
    check_chunk_size()
    check_split_ports()
    print("smoke:")
    check_smoke()
    print("\nall dsl_topology checks passed")


if __name__ == "__main__":
    main()
