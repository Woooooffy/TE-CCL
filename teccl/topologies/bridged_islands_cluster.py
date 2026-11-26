"""
The forcing fixture for CELL RELAY -- a coarse path that store-and-forwards through an
intermediate CELL. See teccl/hierarchy/cell_relay_design.md.

No other topology in this repo forces transit. Rail-optimized, hetero and the two-pod variants all
reach every peer through switches, so every coarse path is `U -> switch(es) -> V` and the whole
question stays dormant. That is why the gap sat unmodelled: it is unreachable by construction on
every input the solver has ever been run on.
"""
from teccl.hierarchy.cell import Cell
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class BridgedIslandsCluster(Topology):
    """
    Two switch islands with NO switch-to-switch link, bridged only by one dual-homed host.

        Host A --- T0        T1 --- Host C
                    \\        /
                     \\      /
                      Host B          (dual-homed: one uplink into each island)

    T0 and T1 are NOT connected. So the ONLY route from A to C is

        A -> T0 -> B -> T1 -> C

    which store-and-forwards through cell B -- a host that neither produces nor wants the data.
    That is the transit case exactly: the coarse path splits into a leg filed under (A, B) and a
    leg filed under (B, C), both belonging to the logical flow (A, C). Every cell here is a transit
    cell for some pair (B for A<->C; A and C are not, but they are the endpoints that make B one),
    so the fixture forces the path rather than merely permitting it.

    Fine node indexing (11 nodes), 2 GPUs per host:
        GPUs:        A 0-1,  B 2-3,  C 4-5
        NVSwitches:  A 6,    B 7,    C 8
        Top switch:  T0 9,   T1 10

    Coarse abstraction -> 3 host cells (A=0, B=1, C=2) + 2 switches (T0, T1) = 5 coarse nodes:
        A->T0 100, B->T0 100, B->T1 100, C->T1 100. No T0<->T1 edge.

    Two variants, and the difference between them is the whole point (design §6):

      * BridgedIslandsCluster (this class) -- B's two uplinks are owned by the SAME GPU (g2).
        The transit is CO-LOCATED: the bytes land on the GPU that also owns the outgoing uplink,
        so forwarding costs no intra-cell work and the one-coarse-epoch dwell the coarse solve
        already models (arrival + 1) is enough. This is the variant the identity-resolution work
        carries end to end.

      * BridgedIslandsSplitCluster -- B's uplinks are owned by DIFFERENT GPUs (g2 into T0, g3 into
        T1). Forwarding now needs an intra-cell hop g2 -> g3 between the two legs, which needs
        arrival + 2, and the coarse solve only budgets arrival + 1. That is the no-bridging-GPU
        case of §6: it is INFEASIBLE until the coarse formulation learns a per-cell forwarding
        dwell, which is deliberately out of scope here. It is kept as a fixture for the failure
        message -- the failure is a true statement about the topology, not a bug to route around.
    """

    # (num_gpus, [(local_gpu_index, top_switch_local_id, uplink_capacity_GBs), ...]).
    # A list rather than a dict keyed by local index: the co-located variant needs the SAME GPU to
    # appear twice, once per island, which is precisely what makes it co-located.
    HOST_SPECS = [
        (2, [(0, 0, 100.0)]),                    # Host A -- island 0 only
        (2, [(0, 0, 100.0), (0, 1, 100.0)]),     # Host B -- BRIDGE, both uplinks on g2
        (2, [(0, 1, 100.0)]),                    # Host C -- island 1 only
    ]
    NUM_TOP = 2

    def __init__(self, topo_input: TopologyParams):
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

        edges = {}
        for h, (num_gpus, uplinks) in enumerate(self.HOST_SPECS):
            nv = self._nvswitch(h)
            for local in range(num_gpus):
                edges[(self._gpu(h, local), nv)] = (nvswitch_cap, nvswitch_alpha)
            for local, top_local, cap in uplinks:
                edges[(self._gpu(h, local), self._top(top_local))] = (cap / self.chunk_size,
                                                                     network_alpha)
        # Deliberately NO top-to-top edge: that absence is what forces transit through host B.

        self.capacity = [[0.0] * n for _ in range(n)]
        self.alpha = [[-1.0] * n for _ in range(n)]
        for (i, j), (cap, alpha) in edges.items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = alpha
            self.alpha[j][i] = alpha

        # A and C are structurally interchangeable (one host on each island, same GPU count, same
        # uplink), so the coarse graph DOES have an automorphism swapping them. It is a
        # source-permuting symmetry, not a relay-twin one, so it is not what
        # equivalent_node_indices declares -- leave it empty, as HeteroTaperedCluster does.
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
            for local, top_local, _cap in uplinks:
                boundary.setdefault(self._top(top_local), []).append(self._gpu(h, local))
            self.cells.append(Cell(
                members=gpus + [nv],
                gpus=gpus,
                internal_switches=[nv],
                boundary=boundary,
            ))


class BridgedIslandsSplitCluster(BridgedIslandsCluster):
    """BridgedIslandsCluster with the bridge host's two uplinks on DIFFERENT GPUs.

    The only change is where B's second uplink hangs: g3 instead of g2. That single edge is the
    difference between a transit that costs nothing and one that cannot be implemented at all
    (design §6) -- data landing on g2 from island 0 must cross the NVSwitch to g3 before it can
    leave on island 1, and the coarse solve budgets no epoch for that hop.

    Expected to FAIL until the coarse per-cell forwarding dwell lands. See the class docstring of
    BridgedIslandsCluster.
    """

    HOST_SPECS = [
        (2, [(0, 0, 100.0)]),                    # Host A
        (2, [(0, 0, 100.0), (1, 1, 100.0)]),     # Host B -- BRIDGE, uplinks split across g2/g3
        (2, [(0, 1, 100.0)]),                    # Host C
    ]
