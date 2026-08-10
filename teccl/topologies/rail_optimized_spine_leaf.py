from teccl.hierarchy.cell import Cell
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class RailOptimizedSpineLeaf(Topology):
    """
    32-node / 256-GPU rail-optimized, single-plane, spine-leaf tree.

    Intra-node (per node): 8 GPUs behind one NVSwitch, 1800 GB/s per GPU.
    Rail-optimized network: 8 leaf switches (one per rail) and 4 spine
    switches. Each 64-port switch runs at 400 Gbps/port.
        - Leaf r connects to the rail-r GPU of every one of the 32 nodes
          => 32 leaf-GPU links @ 400 Gbps (uses 32 of the leaf's 64 ports).
        - The leaf's remaining 32 ports go to the 4 spines, 8 ports each,
          modeled as a single 8x400 Gbps = 3200 Gbps (400 GB/s) link per
          leaf-spine pair. Every leaf connects to every spine (full mesh),
          which fills all 64 ports of each spine (8 leaves x 8 ports).

    Node indexing (300 nodes total):
        GPU(node n, rail r) = n * 8 + r          for n in [0,32), r in [0,8)  -> [0,256)
        NVSwitch(node n)    = 256 + n             for n in [0,32)              -> [256,288)
        Leaf(rail r)        = 288 + r             for r in [0,8)               -> [288,296)
        Spine(s)            = 296 + s             for s in [0,4)               -> [296,300)
    """

    NUM_NODES = 32          # physical servers
    GPUS_PER_NODE = 8       # == number of rails
    NUM_LEAF = 8            # one leaf switch per rail
    NUM_SPINE = 4

    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)
        self.node_per_chassis = self.GPUS_PER_NODE

    # --- index helpers -----------------------------------------------------
    def _gpu(self, node: int, rail: int) -> int:
        return node * self.GPUS_PER_NODE + rail

    def _nvswitch(self, node: int) -> int:
        return self.NUM_NODES * self.GPUS_PER_NODE + node

    def _leaf(self, rail: int) -> int:
        return self.NUM_NODES * self.GPUS_PER_NODE + self.NUM_NODES + rail

    def _spine(self, s: int) -> int:
        return (self.NUM_NODES * self.GPUS_PER_NODE + self.NUM_NODES
                + self.NUM_LEAF + s)

    def construct_topology(self, topo_input: TopologyParams):
        num_gpus = self.NUM_NODES * self.GPUS_PER_NODE
        total_nodes = num_gpus + self.NUM_NODES + self.NUM_LEAF + self.NUM_SPINE

        # Capacities in GB/s (per the codebase convention, divided by chunk_size).
        # 400 Gbps = 50 GB/s; 8 x 400 Gbps = 3200 Gbps = 400 GB/s.
        nvswitch_cap = 1800 / self.chunk_size          # 1800 GB/s per GPU
        leaf_gpu_cap = 50 / self.chunk_size            # 400 Gbps
        leaf_spine_cap = 400 / self.chunk_size         # 8 x 400 Gbps = 3200 Gbps

        nvswitch_alpha = 0.35 * pow(10, -6)            # intra-node (NVLink-like)
        network_alpha = 0.7 * pow(10, -6)              # leaf-GPU and leaf-spine

        edges = {}
        # Intra-node: every GPU <-> its node's NVSwitch.
        for n in range(self.NUM_NODES):
            for r in range(self.GPUS_PER_NODE):
                edges[(self._gpu(n, r), self._nvswitch(n))] = (nvswitch_cap, nvswitch_alpha)

        # Rail-optimized leaf-GPU: leaf r <-> rail-r GPU of every node.
        for n in range(self.NUM_NODES):
            for r in range(self.GPUS_PER_NODE):
                edges[(self._gpu(n, r), self._leaf(r))] = (leaf_gpu_cap, network_alpha)

        # Leaf-spine full mesh: every leaf <-> every spine.
        for r in range(self.NUM_LEAF):
            for s in range(self.NUM_SPINE):
                edges[(self._leaf(r), self._spine(s))] = (leaf_spine_cap, network_alpha)

        self.capacity = [[0.0] * total_nodes for _ in range(total_nodes)]
        self.alpha = [[-1.0] * total_nodes for _ in range(total_nodes)]
        for (i, j), (cap, alpha) in edges.items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = alpha
            self.alpha[j][i] = alpha

        # The 4 spines are topological twins: each connects to all 8 leaves
        # with identical link weights. (No two GPUs, NVSwitches, or leaves are
        # twins, since each connects to a distinct set of neighbors.)
        self.equivalent_node_indices = [
            [self._spine(s) for s in range(self.NUM_SPINE)]
        ]

    def set_switch_indicies(self) -> None:
        # NVSwitches, leaf switches, and spine switches are all forwarding
        # switches (no compute / no collective demand of their own).
        self.switch_indices = (
            [self._nvswitch(n) for n in range(self.NUM_NODES)]
            + [self._leaf(r) for r in range(self.NUM_LEAF)]
            + [self._spine(s) for s in range(self.NUM_SPINE)]
        )

    def default_programmable_switch_indices(self):
        # Only the network fabric (leaf + spine) is programmed per route. The per-node NVSwitch
        # is a self-routing crossbar: the solver routes GPU->NVSwitch->GPU hops through it, but
        # no forwarding entry is ever installed on it, so it is excluded from the emitted table.
        return ([self._leaf(r) for r in range(self.NUM_LEAF)]
                + [self._spine(s) for s in range(self.NUM_SPINE)])

    def build_hierarchy(self) -> None:
        # One cell per host: its 8 rail GPUs plus the NVSwitch behind them collapse into a
        # single coarse (data-bearing) node. The NVSwitch is internal and dropped from the
        # coarse graph. The rail constraint -- GPU r reaches only leaf r -- is captured by
        # `boundary`: the coarse host<->leaf(r) link is physically owned by gpu(n, r).
        # The 8 leaf and 4 spine switches are NOT collapsed; they stay as coarse switch
        # nodes, so the coarse topology is {32 hosts} + {8 leaves} + {4 spines} = 44 nodes.
        self.cells = []
        for n in range(self.NUM_NODES):
            gpus = [self._gpu(n, r) for r in range(self.GPUS_PER_NODE)]
            nvswitch = self._nvswitch(n)
            boundary = {
                self._leaf(r): [self._gpu(n, r)] for r in range(self.GPUS_PER_NODE)
            }
            self.cells.append(Cell(
                members=gpus + [nvswitch],
                gpus=gpus,
                internal_switches=[nvswitch],
                boundary=boundary,
            ))
