from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from teccl.input_data import TopologyParams
from teccl.hierarchy.cell import Cell
from teccl.topologies.topology import Topology


@dataclass
class HierarchyMapping:
    """
    Records how a coarse topology (produced by abstract()) relates to the fine one, so the
    coarse solution can be lowered back to fine GPUs.

    fine_to_coarse:     every fine node index -> its coarse node index.
    coarse_cells:       coarse id -> Cell, for the coarse nodes that are collapsed cells
                        (the data-bearing "host" nodes).
    coarse_passthrough: coarse id -> fine id, for the coarse nodes that are un-collapsed
                        fine nodes carried through verbatim (leaves, spines).
    boundary_gpu:       (coarse_cell_id, coarse_neighbor_id) -> the fine gpu(s) inside the
                        cell that physically own the coarse link to that neighbor. This is
                        the port map used to place a coarse inter-cell flow onto a real GPU
                        egress/ingress during reconstruction (phase 3 / stitching).
    num_coarse:         number of coarse nodes.
    chunk_origin:       (coarse_cell_id, sub_chunk) -> fine gpu the sub-chunk originates on.
                        Empty until lift_demand() fills it for a specific num_sub_chunks.
    """
    fine_to_coarse: Dict[int, int]
    coarse_cells: Dict[int, Cell]
    coarse_passthrough: Dict[int, int]
    boundary_gpu: Dict[Tuple[int, int], List[int]]
    num_coarse: int
    chunk_origin: Dict[Tuple[int, int], int] = field(default_factory=dict)


class CoarseTopology(Topology):
    """
    A Topology whose capacity/alpha/switch data are supplied precomputed by abstract(),
    rather than constructed from scratch. It is a normal Topology in every other respect, so
    the existing LP/MILP formulations run on it unmodified.

    The coarse topology is itself flat (no nested hierarchy) for now; build_hierarchy stays a
    no-op. Nested hierarchies (a coarse topology that itself declares cells) are a later step.
    """

    def __init__(self, topo_input: TopologyParams, capacity: List[List[float]],
                 alpha: List[List[float]], switch_indices: List[int],
                 equivalent_node_indices: List[List[int]], node_per_chassis: int) -> None:
        # Stash the precomputed data; construct_topology / set_switch_indicies (called by the
        # base __init__) install it. These must exist before super().__init__ runs.
        self._c_capacity = capacity
        self._c_alpha = alpha
        self._c_switch_indices = switch_indices
        self._c_equivalent_node_indices = equivalent_node_indices
        self._c_node_per_chassis = node_per_chassis
        super().__init__(topo_input)

    def construct_topology(self, topo_input: TopologyParams) -> None:
        self.capacity = self._c_capacity
        self.alpha = self._c_alpha
        self.equivalent_node_indices = self._c_equivalent_node_indices
        self.node_per_chassis = self._c_node_per_chassis

    def set_switch_indicies(self) -> None:
        self.switch_indices = self._c_switch_indices


def abstract(topology: Topology) -> Tuple[CoarseTopology, HierarchyMapping]:
    """
    Collapse each declared cell of `topology` into a single coarse node and return the coarse
    topology plus a HierarchyMapping.

    Coarse node numbering: the cells get ids 0..C-1 (in declaration order), then every
    un-collapsed fine node gets a subsequent id (in ascending fine order). Cells become
    data-bearing GPU nodes; un-collapsed switches stay switches. Intra-cell edges are dropped
    (they live in the cell's internal topology, used later by phase 3); inter-cell edges are
    aggregated into coarse links, with the owning fine gpu recorded in boundary_gpu.
    """
    assert topology.cells, "topology declares no cells; nothing to abstract"
    n_fine = len(topology.capacity)

    # --- assign coarse ids -------------------------------------------------
    fine_to_coarse: Dict[int, int] = {}
    coarse_cells: Dict[int, Cell] = {}
    coarse_passthrough: Dict[int, int] = {}

    covered: set = set()
    next_id = 0
    for cell in topology.cells:
        # check that the cell's members are disjoint
        overlap = covered & set(cell.members)
        assert not overlap, f"cells overlap on fine nodes {sorted(overlap)}"
        covered.update(cell.members)
        # assign coarse ids
        coarse_cells[next_id] = cell
        for m in cell.members:
            fine_to_coarse[m] = next_id
        next_id += 1
    for f in range(n_fine):
        if f in covered:
            continue
        coarse_passthrough[next_id] = f
        fine_to_coarse[f] = next_id
        next_id += 1
    num_coarse = next_id

    # --- aggregate inter-cell edges into coarse links ----------------------
    capacity = [[0.0] * num_coarse for _ in range(num_coarse)]
    alpha = [[-1.0] * num_coarse for _ in range(num_coarse)]
    boundary_gpu: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    cell_gpus = {cid: set(cell.gpus) for cid, cell in coarse_cells.items()}

    for u in range(n_fine):
        for v in range(n_fine):
            cap = topology.capacity[u][v]
            if cap <= 0:
                continue
            cu, cv = fine_to_coarse[u], fine_to_coarse[v]
            if cu == cv:
                # intra-cell edge: dropped from the coarse graph (handled in phase 3).
                continue
            capacity[cu][cv] += cap
            # Contributing fine links are assumed to share an alpha (true for the
            # rail-optimized topology); record it. If they differ, this keeps the last
            # one -- fine for the current topologies, revisit if a mixed-alpha coarse
            # link ever appears.
            alpha[cu][cv] = topology.alpha[u][v]
            # If the source endpoint u lives inside a cell, it is that cell's physical
            # boundary gpu toward coarse node cv (egress for cu->cv, ingress for cv->cu).
            if cu in cell_gpus and u in cell_gpus[cu]:
                if u not in boundary_gpu[(cu, cv)]:
                    boundary_gpu[(cu, cv)].append(u)

    # --- switches, symmetry, passives on the coarse graph ------------------
    coarse_switch_indices = [
        cid for cid, f in coarse_passthrough.items() if f in topology.switch_indices
    ]

    # Coarse symmetry groups = the fine equivalences forwarded through the collapse
    # PLUS emergent twins that only exist after coarsening. The fine graph declares
    # only the 4 spines: the 8 leaves are NOT twins in the fine graph (leaf r touches
    # only rail-r GPUs, so each leaf's fine neighborhood differs), so no forwarding step
    # could ever produce a leaf group. Collapsing each host's rail GPUs into one node
    # erases the rail identity from the capacity matrix (it moves into boundary_gpu),
    # after which every leaf connects to all hosts @50 and all spines @400 -- identical
    # neighborhoods, hence interchangeable. We discover these by neighbor-profile hashing
    # on the COARSE matrix: nodes whose (sorted out-edges, sorted in-edges) fingerprints
    # match are twins (swapping them is a graph automorphism), and grouping by fingerprint
    # finds every twin class in one pass.
    #
    # RESTRICTED TO SWITCHES on purpose. The data-bearing hosts ALSO share an identical
    # fingerprint (each connects to exactly the 8 leaves @50), so an unrestricted detector
    # would report them as a twin group too. But a host swap permutes the source index
    # (host a carries source-a's data, host b does not), so it is a source-permuting
    # symmetry, valid only when the demand is host-symmetric -- that belongs to a separate,
    # demand-gated mechanism, not to the relay-twin groups consumed here. Only relay
    # switches (whose swap fixes every source) are safe to emit as equivalent_node_indices.
    #
    # Generalization note: this catches only "identical-neighborhood" twins, and it works
    # unmasked because every twin class here is an independent set (no leaf<->leaf or
    # spine<->spine edges). For twins that ARE adjacent to each other, the mutual edge makes
    # their raw fingerprints differ, so mask group-mates out of the key before comparing.
    # Symmetries that are not identical-neighborhood (e.g. rotational, or twins-of-twins)
    # are not detected here and would need real graph-automorphism refinement (nauty-style).
    def _twin_key(i: int) -> Tuple:
        outs = tuple(sorted((j, capacity[i][j]) for j in range(num_coarse) if capacity[i][j] > 0))
        ins = tuple(sorted((j, capacity[j][i]) for j in range(num_coarse) if capacity[j][i] > 0))
        return (outs, ins)

    coarse_equivalent_set = {
        tuple(sorted({fine_to_coarse[x] for x in group}))
        for group in topology.equivalent_node_indices
        if len({fine_to_coarse[x] for x in group}) > 1
    }
    emergent: Dict[Tuple, List[int]] = defaultdict(list)
    for cid in coarse_switch_indices:
        emergent[_twin_key(cid)].append(cid)
    for cids in emergent.values():
        if len(cids) > 1:
            coarse_equivalent_set.add(tuple(sorted(cids)))
    coarse_equivalent: List[List[int]] = sorted(
        (list(g) for g in coarse_equivalent_set), key=lambda g: g[0])

    # Coarse "GPUs per chassis" is a diagnostic only (drives Algo_Bandwidth in the coarse
    # solve); the final fine-grained bandwidth is recomputed during stitching. Use the number
    # of data-bearing coarse nodes.
    coarse_node_per_chassis = len(coarse_cells)

    coarse_passive = sorted(
        cid for cid, f in coarse_passthrough.items() if f in topology.passive_indices
    )

    topo_input = TopologyParams(
        name=f"{topology.__class__.__name__}_coarse",
        chassis=1,
        chunk_size=topology.chunk_size,
        passive_node_indices=tuple(coarse_passive),
    )
    coarse = CoarseTopology(
        topo_input,
        capacity=capacity,
        alpha=alpha,
        switch_indices=sorted(coarse_switch_indices),
        equivalent_node_indices=coarse_equivalent,
        node_per_chassis=coarse_node_per_chassis,
    )

    mapping = HierarchyMapping(
        fine_to_coarse=fine_to_coarse,
        coarse_cells=coarse_cells,
        coarse_passthrough=coarse_passthrough,
        boundary_gpu={k: v for k, v in boundary_gpu.items()},
        num_coarse=num_coarse,
    )
    return coarse, mapping


def coarsify_demand(fine_demand, mapping: HierarchyMapping) -> List[List[List[int]]]:
    """
    Aggregate a FINE demand tensor into a COARSE demand matrix by cell membership,
    COLLECTIVE-AGNOSTICALLY. A demand is "what volume goes from where to where"; coarsifying
    replaces each fine GPU endpoint with its coarse node (cell or passthrough) and counts how
    much distinct data must cross each inter-cell boundary.

    General rule (no branch on collective type). Each fine entry is a distinct data identity
    (source s, chunk index ci) that some set of destinations wants. Its inter-cell contribution
    is ONE unit crossing from cell(s) into each DISTINCT destination cell that wants it:

        for each identity (s, ci):
            dest_cells = { coarse(t) : fine_demand[s][t][ci] > 0 }
            for cv in dest_cells, cv != coarse(s):
                coarse[coarse(s)][cv] += 1

    Counting an identity once per destination CELL (not once per destination GPU) is exactly
    what produces AllGather's fan-out saving: all GPUs of cell V wanting identity (s, ci)
    collapse to a SINGLE coarse crossing, and the intra-cell replication to V's other GPUs is
    deferred to phase-3 ingress distribution. AllToAll identities are per-destination-GPU
    distinct (the chunk index encodes the target GPU), so no collapse happens and the coarse
    volume is the full |U|*|V| -- the same code yields both. Intra-cell demand (coarse(s) ==
    coarse(t)) is dropped; it is reconstructed by phase 3, never carried on the coarse graph.

    `fine_demand` may be a numpy array or a nested list -- it is only indexed and compared,
    never reshaped. Returns a coarse demand tensor as a nested list of shape
    (num_coarse, num_coarse, 1): coarse[U][V][0] is the aggregate volume U -> V. A single
    weighted chunk-slot is exact for the copy-free LP, which sums demand over the chunk axis
    into a per-(source, dest) volume.
    """
    f2c = mapping.fine_to_coarse
    num_coarse = mapping.num_coarse
    n_fine = len(fine_demand)
    coarse = [[[0] for _ in range(num_coarse)] for _ in range(num_coarse)]
    for s in range(n_fine):
        cs = f2c[s]
        chunks = len(fine_demand[s][s]) if n_fine else 0
        for ci in range(chunks):
            dest_cells = set()
            for t in range(n_fine):
                if fine_demand[s][t][ci] > 0:
                    dest_cells.add(f2c[t])
            for cv in dest_cells:
                if cv != cs:
                    coarse[cs][cv][0] += 1
    return coarse


def lift_demand(mapping: HierarchyMapping, num_sub_chunks: int = None) -> None:
    """
    Fill mapping.chunk_origin: coarse sub-chunk c of a cell originates on the cell's c-th gpu.
    This is the identity map identity-resolution uses to split a cell's aggregate coarse
    commodity back into its fine chunk identities.

    Two modes:
      * num_sub_chunks=None (default, HETEROGENEOUS): each cell contributes exactly its own
        GPU count of sub-chunks -- cell H gets sub-chunks 0..len(H.gpus)-1, one per fine GPU.
        This is the host-level AllGather lift for topologies whose cells have DIFFERENT GPU
        counts (e.g. HeteroTaperedCluster's 4/4/6), matching the weighted
        1-commodity-per-cell demand produced by the AllGather demand generator from
        topology.node_volumes.
      * num_sub_chunks=int (UNIFORM, backward compatible): every cell contributes exactly
        num_sub_chunks sub-chunks. Requires every cell to have at least that many GPUs. Used
        by the symmetric rail-optimized topology where all cells have the same GPU count.
    """
    for cid, cell in mapping.coarse_cells.items():
        if num_sub_chunks is None:
            count = len(cell.gpus)
        else:
            assert num_sub_chunks <= len(cell.gpus), (
                f"cell {cid} has {len(cell.gpus)} gpus but {num_sub_chunks} sub-chunks requested"
            )
            count = num_sub_chunks
        for c in range(count):
            mapping.chunk_origin[(cid, c)] = cell.gpus[c]
