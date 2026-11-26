from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class FatTreePodSingleSpine(Topology):
    """
    Single pod of a fat-tree, 4 ports per switch, but with only 1 spine
    switch instead of 2. Each edge switch still fans in 2 GPU downlinks at
    full switch_link_capacity, but now has only 1 spine uplink at the same
    capacity instead of 2, so cross-pod traffic is oversubscribed 2:1 at
    every edge switch and forced to fan in through the single spine switch:
    real incast at the top of the tree (unlike FatTreePod, where 2 spines
    keep the tree non-blocking).

        GPU_A0 --\\                  /-- GPU_B0
                  EdgeA --- Spine ---  EdgeB
        GPU_A1 --/                  \\-- GPU_B1

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
            (0, 4): (switch_link_capacity, switch_link_alpha),  # GPU_A0 - EdgeA
            (1, 4): (switch_link_capacity, switch_link_alpha),  # GPU_A1 - EdgeA
            (2, 5): (switch_link_capacity, switch_link_alpha),  # GPU_B0 - EdgeB
            (3, 5): (switch_link_capacity, switch_link_alpha),  # GPU_B1 - EdgeB
            (4, 6): (switch_link_capacity, switch_link_alpha),  # EdgeA - Spine
            (5, 6): (switch_link_capacity, switch_link_alpha),  # EdgeB - Spine
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
