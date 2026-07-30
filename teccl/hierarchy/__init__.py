"""
Hierarchical solving support.

A hierarchical topology (see Topology.build_hierarchy) declares `cells`: groups of
fine-grained nodes (e.g. an 8-GPU + 1-NVSwitch host) that collapse into a single coarse
node. `abstract()` builds the smaller inter-cell topology plus a `HierarchyMapping` that
records how each coarse link/chunk corresponds to fine GPUs, so the intra-cell schedule can
be reconstructed later. See the `hierarchical_solver_design` design note.

Only `Cell` is re-exported here. `abstract`, `CoarseTopology`, `HierarchyMapping` and
`lift_demand` live in teccl.hierarchy.abstract and must be imported from there directly:
that module imports Topology, and Topology imports Cell, so eagerly importing abstract here
would create an import cycle when Topology is first loaded. Import it lazily instead:

    from teccl.hierarchy.abstract import abstract, lift_demand
"""
from teccl.hierarchy.cell import Cell

__all__ = ["Cell"]
