from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class OddPod(Topology):
    """
    Single pod of a fat-tree, except each edge switch has 3 GPUs instead of 2.

        GPU_A0 --\\                          /-- GPU_B0
        GPU_A1 --- EdgeA --- Spine1, Spine2 --- EdgeB --- GPU_B1
        GPU_A2 --/                          \\-- GPU_B2

    Node indices: 0=GPU_A0, 1=GPU_A1, 2=GPU_A2, 3=GPU_B0, 4=GPU_B1, 5=GPU_B2,
                  6=EdgeA, 7=EdgeB, 8=Spine1, 9=Spine2
    """

    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)
        self.node_per_chassis = 6

    def construct_topology(self, topo_input: TopologyParams):
        num_nodes = 10
        switch_link_capacity = 50 / self.chunk_size
        switch_link_alpha = 0.7 * pow(10, -6)

        edges = {
            (0, 6): (switch_link_capacity, switch_link_alpha),  # GPU_A0 - EdgeA
            (1, 6): (switch_link_capacity, switch_link_alpha),  # GPU_A1 - EdgeA
            (2, 6): (switch_link_capacity, switch_link_alpha),  # GPU_A2 - EdgeA
            (3, 7): (switch_link_capacity, switch_link_alpha),  # GPU_B0 - EdgeB
            (4, 7): (switch_link_capacity, switch_link_alpha),  # GPU_B1 - EdgeB
            (5, 7): (switch_link_capacity, switch_link_alpha),  # GPU_B2 - EdgeB
            (6, 8): (switch_link_capacity, switch_link_alpha),  # EdgeA - Spine1
            (6, 9): (switch_link_capacity, switch_link_alpha),  # EdgeA - Spine2
            (7, 8): (switch_link_capacity, switch_link_alpha),  # EdgeB - Spine1
            (7, 9): (switch_link_capacity, switch_link_alpha),  # EdgeB - Spine2
        }

        self.capacity = [[0.0] * num_nodes for _ in range(num_nodes)]
        self.alpha = [[-1.0] * num_nodes for _ in range(num_nodes)]
        for (i, j), (cap, alpha) in edges.items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = alpha
            self.alpha[j][i] = alpha

    def set_switch_indicies(self) -> None:
        self.switch_indices = [6, 7, 8, 9]
