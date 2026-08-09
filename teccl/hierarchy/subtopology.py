"""
Descending a level: the induced sub-topology of one cell.

`abstract()` goes UP -- it collapses cells into coarse nodes and drops their internal edges, because
those edges belong to a level it is not solving. `induce()` goes DOWN and picks exactly those edges
back up: the sub-graph on a cell's members, renumbered 0..n-1, which is the topology the cell's own
level is solved on. The two are the only ways the recursion crosses a level boundary, and they are
inverses in the sense that matters -- the edges one drops are the edges the other keeps, so no link
is solved against twice and none is forgotten.

Renumbering is what makes the recursion uniform rather than special-cased. A `Topology` is dense
0..n-1 by contract (every formulation indexes its capacity matrix directly), so a cell whose members
are scattered global ids cannot be handed to a solver as-is. `local_to_global` is therefore carried
on the result and is the ONLY thing that knows the correspondence: identities stay global at every
depth (see teccl/hierarchy/problem.py), and node indices are translated at exactly this boundary.
"""
from typing import Dict, List

from teccl.hierarchy.cell import Cell
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class InducedTopology(Topology):
    """A Topology built from precomputed data (like CoarseTopology), for one cell's interior.

    Carries `local_to_global` / `global_to_local` and re-declares the cell's `subcells` in local
    indices, so `build_hierarchy` reports a hierarchy exactly when the cell has one -- which is what
    the base-case dispatcher tests to decide whether to recurse or to solve.
    """

    def __init__(self, topo_input: TopologyParams, capacity: List[List[float]],
                 alpha: List[List[float]], switch_indices: List[int],
                 local_to_global: List[int], subcells: List[Cell],
                 node_per_chassis: int) -> None:
        self._c_capacity = capacity
        self._c_alpha = alpha
        self._c_switch_indices = switch_indices
        self._c_node_per_chassis = node_per_chassis
        self._c_subcells = subcells
        self.local_to_global = local_to_global
        self.global_to_local = {g: i for i, g in enumerate(local_to_global)}
        super().__init__(topo_input)

    def construct_topology(self, topo_input: TopologyParams) -> None:
        self.capacity = self._c_capacity
        self.alpha = self._c_alpha
        # No symmetry groups are forwarded. The fine topology's `equivalent_node_indices` are
        # statements about the WHOLE graph, and restricting a graph can destroy an automorphism
        # (two nodes interchangeable in the cluster need not be interchangeable inside one host once
        # their differing external links are cut away). Emitting them unchecked would let a solver
        # enforce a symmetry the sub-graph does not have, which silently prunes the optimum. Leaving
        # them empty only forgoes a speedup.
        self.equivalent_node_indices = []
        self.node_per_chassis = self._c_node_per_chassis

    def set_switch_indicies(self) -> None:
        self.switch_indices = self._c_switch_indices

    def build_hierarchy(self) -> None:
        self.cells = self._c_subcells


def induce(topology: Topology, cell: Cell) -> InducedTopology:
    """Build the sub-topology of `cell`: the induced subgraph on its members, renumbered 0..n-1.

    Local index = position in `sorted(cell.members)`, so the numbering is a pure function of the
    cell and two calls on the same cell agree. Only edges with BOTH endpoints inside the cell are
    kept -- an edge leaving the cell is the level above's to schedule, and it already did
    (`abstract` aggregated it into a coarse link and `boundary_gpu` recorded which member owns it).
    """
    members = sorted(cell.members)
    local: Dict[int, int] = {g: i for i, g in enumerate(members)}
    n = len(members)

    capacity = [[0.0] * n for _ in range(n)]
    alpha = [[-1.0] * n for _ in range(n)]
    for u in members:
        for v in members:
            c = topology.capacity[u][v]
            if c > 0:
                capacity[local[u]][local[v]] = c
                alpha[local[u]][local[v]] = topology.alpha[u][v]

    missing = [s for s in cell.internal_switches if s not in local]
    assert not missing, f"cell declares internal switches {missing} that are not among its members"

    subcells: List[Cell] = []
    for sc in cell.subcells:
        outside = [m for m in sc.members if m not in local]
        assert not outside, (
            f"subcell members {outside} are outside the parent cell; a subcell must partition part "
            f"of its parent's members")
        subcells.append(Cell(
            members=[local[m] for m in sc.members],
            gpus=[local[g] for g in sc.gpus],
            internal_switches=[local[s] for s in sc.internal_switches],
            # A subcell's boundary names neighbors INSIDE the parent cell (the rack switch, say),
            # which is exactly what makes it a valid boundary map at this level.
            boundary={local[nb]: [local[g] for g in gs]
                      for nb, gs in sc.boundary.items() if nb in local},
            subcells=[],   # deeper nesting is re-derived by the next induce(), not carried here
        ))
        # Deeper levels: translate recursively rather than dropping them on the floor.
        if sc.subcells:
            subcells[-1].subcells = [
                Cell(members=[local[m] for m in d.members],
                     gpus=[local[g] for g in d.gpus],
                     internal_switches=[local[s] for s in d.internal_switches],
                     boundary={local[nb]: [local[g] for g in gs]
                               for nb, gs in d.boundary.items() if nb in local},
                     subcells=d.subcells)
                for d in sc.subcells]

    topo_input = TopologyParams(
        name=f"{topology.__class__.__name__}_cell",
        chassis=1,
        # The chunk unit is the LEVEL's, not the topology's, and it travels on the ChunkScale rather
        # than here (see teccl/hierarchy/scale.py). Inheriting the parent's value keeps
        # `capacity * epoch_duration` and the demand in the same unit at construction time; a level
        # that then coarsens re-expresses both together via CoarseTopology.rescale_to_chunk.
        chunk_size=topology.chunk_size,
        passive_node_indices=tuple(local[p] for p in getattr(topology, "passive_indices", [])
                                   if p in local),
    )
    return InducedTopology(
        topo_input,
        capacity=capacity,
        alpha=alpha,
        switch_indices=sorted(local[s] for s in cell.internal_switches),
        local_to_global=members,
        subcells=subcells,
        node_per_chassis=len(cell.gpus),
    )
