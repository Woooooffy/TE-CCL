from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class FatTreePodSingleSpineFatUplink(Topology):
    """
    Single pod of a fat-tree with only 1 spine switch (like
    FatTreePodSingleSpine), but the leaf-spine uplinks run at double
    switch_link_capacity. Each edge switch fans in 2 GPU downlinks at full
    switch_link_capacity (2 x cap = 2*cap down) and has 1 spine uplink at
    2 x switch_link_capacity (2*cap up), so cross-pod traffic is 1:1
    non-blocking at every edge switch -- the same non-oversubscribed
    provisioning as FatTreePod, but achieved with a single fat spine uplink
    instead of two parallel leaf-spine paths through two spines.

        GPU_A0 --\\                    /-- GPU_B0
                  EdgeA ==== Spine ==== EdgeB
        GPU_A1 --/                    \\-- GPU_B1

    (=== marks the 2x-capacity leaf-spine uplinks.)

    Node indices: 0=GPU_A0, 1=GPU_A1, 2=GPU_B0, 3=GPU_B1,
                  4=EdgeA, 5=EdgeB, 6=Spine
    """

    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)
        self.node_per_chassis = 4

    def construct_topology(self, topo_input: TopologyParams):
        num_nodes = 7
        switch_link_capacity = 50 / self.chunk_size
        switch_link_alpha = 0.7 * pow(10, -6)

        edges = {
            (0, 4): (switch_link_capacity, switch_link_alpha),      # GPU_A0 - EdgeA
            (1, 4): (switch_link_capacity, switch_link_alpha),      # GPU_A1 - EdgeA
            (2, 5): (switch_link_capacity, switch_link_alpha),      # GPU_B0 - EdgeB
            (3, 5): (switch_link_capacity, switch_link_alpha),      # GPU_B1 - EdgeB
            (4, 6): (2 * switch_link_capacity, switch_link_alpha),  # EdgeA - Spine (fat uplink)
            (5, 6): (2 * switch_link_capacity, switch_link_alpha),  # EdgeB - Spine (fat uplink)
        }

        self.capacity = [[0.0] * num_nodes for _ in range(num_nodes)]
        self.alpha = [[-1.0] * num_nodes for _ in range(num_nodes)]
        for (i, j), (cap, alpha) in edges.items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = alpha
            self.alpha[j][i] = alpha

        # GPU_A0 (0) / GPU_A1 (1) and GPU_B0 (2) / GPU_B1 (3) are pairwise
        # topological twins. Spine (6) has no twin now that there's only one.
        self.equivalent_node_indices = [[0, 1], [2, 3]]

    def set_switch_indicies(self) -> None:
        self.switch_indices = [4, 5, 6]
