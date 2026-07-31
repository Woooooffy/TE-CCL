from teccl.hierarchy.cell import Cell
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class HeteroTaperedCluster(Topology):
    """
    Small IRREGULAR topology built specifically to exercise the hierarchical solver's phase-3
    machinery (identity resolution + intra-cell relay), which the symmetric rail-optimized
    topology never triggers.

    rail_optimized_spine_leaf abstracts to a highly symmetric coarse graph (8 leaf-twins,
    32 host-twins), so the coarse LP's clean 1-unit-per-rail solution gives ZERO identity
    leftover and the intra-cell problem is trivial. This topology instead abstracts to an
    asymmetric, heterogeneous coarse graph with NO non-trivial automorphism, and every cell
    has FEWER uplinks than GPUs, so egress relay is FORCED and identity resolution must
    produce real intra-node demand.

    3 heterogeneous hosts (each: full-mesh NVSwitch internal at 1800 GB/s) + 2 top switches:
        Host A: 4 GPUs, uplinks {g0 -> T0, g1 -> T1}            (2 uplinks)
        Host B: 4 GPUs, uplinks {g0 -> T0}                      (1 uplink, single-homed to T0)
        Host C: 6 GPUs, uplinks {g0 -> T0, g3 -> T1, g5 -> T1}  (3 uplinks, TWO landing on T1)
    T0 <-> T1 form the top mesh. Heterogeneous BW: A/C host uplinks 100 GB/s, B's lone uplink
    50 GB/s, T0<->T1 200 GB/s.

    Fine node indexing (19 nodes):
        GPUs:        A 0-3,  B 4-7,  C 8-13
        NVSwitches:  A 14,   B 15,   C 16
        Top switch:  T0 17,  T1 18

    Coarse abstraction (hosts collapse to one data-bearing node each; top switches stay
    switch nodes) -> 3 host nodes + 2 switches = 5 coarse nodes:
        A->T0 100, A->T1 100, B->T0 50, C->T0 100, C->T1 200 (g3+g5 capacities SUMMED),
        T0<->T1 200.
    The coarse graph has no automorphism (T0 is adjacent to B, T1 is not; the three cells
    differ in GPU count and uplink pattern), so abstract() emits NO equivalent_node_indices
    -> the coarse LP has no symmetry to exploit and produces genuinely asymmetric flows.

    Why each irregularity matters for testing (see hierarchical_solver_design):
      - uplinks < GPUs (all hosts): a chunk on a non-uplink GPU MUST relay to a gateway GPU
        before egress -> identity resolution has non-zero leftover -> real intra-node egress
        demand (gpu(H,c) -> gpu(H,gateway)). This is the core phase-3 behavior rail-optimized
        never triggers.
      - single-homed Host B (only reaches T0): B<->C and B's cross traffic must take a
        multi-hop coarse path B->T0->T1->C, and all of B's GPUs 5,6,7 funnel through g4.
      - multi-GPU boundary (C -> T1 via g11 and g13): exercises boundary_gpu as a LIST and
        capacity summing (200), plus a choice of which gateway to relay to.
      - heterogeneous BW + non-isomorphic cells: no symmetry shortcut; identity resolution
        and per-destination conservation are tested on non-uniform flows.
      - intra-cell stays a single full-mesh NVSwitch, so intra-node resolution is the trivial
        memoized "send through the switch" -- this test isolates the coarse + identity layer.
        (A recursive-intra-cell variant -- e.g. an internal ring instead of a switch -- is a
        separate, later test for nested recursion.)
    """

    # (num_gpus, {local_gpu_index: (top_switch_local_id, uplink_capacity_GBs)})
    HOST_SPECS = [
        (4, {0: (0, 100.0), 1: (1, 100.0)}),   # Host A
        (4, {0: (0, 50.0)}),                    # Host B (single-homed, slow)
        (6, {0: (0, 100.0), 3: (1, 100.0), 5: (1, 100.0)}),  # Host C
    ]
    NUM_TOP = 2
    TOP_TOP_CAP = 200.0

    def __init__(self, topo_input: TopologyParams):
        # Precompute per-host GPU offsets and the key node ranges before super().__init__
        # calls construct_topology.
        self._gpu_offset = []
        off = 0
        for num_gpus, _ in self.HOST_SPECS:
            self._gpu_offset.append(off)
            off += num_gpus
        self._num_gpus = off
        self._nvswitch_base = self._num_gpus
        self._top_base = self._num_gpus + len(self.HOST_SPECS)
        self._total = self._top_base + self.NUM_TOP
        super().__init__(topo_input)
        self.node_per_chassis = self.HOST_SPECS[0][0]

    # --- index helpers -----------------------------------------------------
    def _gpu(self, host: int, local: int) -> int:
        return self._gpu_offset[host] + local

    def _nvswitch(self, host: int) -> int:
        return self._nvswitch_base + host

    def _top(self, t: int) -> int:
        return self._top_base + t

    def construct_topology(self, topo_input: TopologyParams) -> None:
        n = self._total
        nvswitch_cap = 1800.0 / self.chunk_size
        nvswitch_alpha = 0.35 * pow(10, -6)
        network_alpha = 0.7 * pow(10, -6)

        edges = {}  # (i, j) -> (capacity, alpha), added symmetrically below
        for h, (num_gpus, uplinks) in enumerate(self.HOST_SPECS):
            nv = self._nvswitch(h)
            for local in range(num_gpus):
                edges[(self._gpu(h, local), nv)] = (nvswitch_cap, nvswitch_alpha)
            for local, (top_local, cap) in uplinks.items():
                edges[(self._gpu(h, local), self._top(top_local))] = (cap / self.chunk_size, network_alpha)
        # Top mesh.
        for a in range(self.NUM_TOP):
            for b in range(a + 1, self.NUM_TOP):
                edges[(self._top(a), self._top(b))] = (self.TOP_TOP_CAP / self.chunk_size, network_alpha)

        self.capacity = [[0.0] * n for _ in range(n)]
        self.alpha = [[-1.0] * n for _ in range(n)]
        for (i, j), (cap, alpha) in edges.items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = alpha
            self.alpha[j][i] = alpha

        # Deliberately no equivalent_node_indices: the coarse graph is asymmetric.
        self.equivalent_node_indices = []

    def set_switch_indicies(self) -> None:
        self.switch_indices = (
            [self._nvswitch(h) for h in range(len(self.HOST_SPECS))]
            + [self._top(t) for t in range(self.NUM_TOP)]
        )

    def build_hierarchy(self) -> None:
        self.cells = []
        for h, (num_gpus, uplinks) in enumerate(self.HOST_SPECS):
            gpus = [self._gpu(h, local) for local in range(num_gpus)]
            nv = self._nvswitch(h)
            boundary = {}
            for local, (top_local, _cap) in uplinks.items():
                boundary.setdefault(self._top(top_local), []).append(self._gpu(h, local))
            self.cells.append(Cell(
                members=gpus + [nv],
                gpus=gpus,
                internal_switches=[nv],
                boundary=boundary,
            ))
