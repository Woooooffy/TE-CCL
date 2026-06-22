from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class Star(Topology):
    """
    3 GPUs connected to a single switch, no direct GPU-GPU links.

        GPU0 --\\
        GPU1 --- Switch
        GPU2 --/

    Node indices: 0=GPU0, 1=GPU1, 2=GPU2, 3=Switch
    """

    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)
        self.node_per_chassis = 3

    def construct_topology(self, topo_input: TopologyParams):
        num_nodes = 4
        switch_link_capacity = 50 / self.chunk_size
        switch_link_alpha = 0.7 * pow(10, -6)

        edges = {
            (0, 3): (switch_link_capacity, switch_link_alpha),  # GPU0 - Switch
            (1, 3): (switch_link_capacity, switch_link_alpha),  # GPU1 - Switch
            (2, 3): (switch_link_capacity, switch_link_alpha),  # GPU2 - Switch
        }

        self.capacity = [[0.0] * num_nodes for _ in range(num_nodes)]
        self.alpha = [[-1.0] * num_nodes for _ in range(num_nodes)]
        for (i, j), (cap, alpha) in edges.items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = alpha
            self.alpha[j][i] = alpha

    def set_switch_indicies(self) -> None:
        self.switch_indices = [3]
