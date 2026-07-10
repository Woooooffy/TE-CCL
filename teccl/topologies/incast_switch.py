from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class IncastSwitch(Topology):
    """
    4 fast GPUs and 1 slow GPU behind a single switch. The switch's link to
    the slow GPU has 1/4 the capacity of each fast GPU's link, so any chunk
    that needs to cross into or out of the slow GPU is bottlenecked at the
    switch: all four fast GPUs can push a chunk into the switch in the same
    epoch, but the switch can only drain one chunk per OVERSUBSCRIPTION
    epochs toward the slow GPU (and vice versa).

        GPU0 --\\
        GPU1 --- \\
        GPU2 ---- Switch === GPU5 (slow link)
        GPU3 --/

    Node indices: 0-3=fast GPUs, 4=Switch, 5=slow GPU
    """

    OVERSUBSCRIPTION = 4  # fast link capacity is this many times the slow link's

    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)
        self.node_per_chassis = 5

    def construct_topology(self, topo_input: TopologyParams):
        num_nodes = 6
        fast_link_capacity = 100 / self.chunk_size
        slow_link_capacity = fast_link_capacity / self.OVERSUBSCRIPTION
        switch_link_alpha = 0.7 * pow(10, -6)

        edges = {
            (0, 4): (fast_link_capacity, switch_link_alpha),  # GPU0 - Switch
            (1, 4): (fast_link_capacity, switch_link_alpha),  # GPU1 - Switch
            (2, 4): (fast_link_capacity, switch_link_alpha),  # GPU2 - Switch
            (3, 4): (fast_link_capacity, switch_link_alpha),  # GPU3 - Switch
            (5, 4): (slow_link_capacity, switch_link_alpha),  # GPU5(slow) - Switch, oversubscribed
        }

        self.capacity = [[0.0] * num_nodes for _ in range(num_nodes)]
        self.alpha = [[-1.0] * num_nodes for _ in range(num_nodes)]
        for (i, j), (cap, alpha) in edges.items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = alpha
            self.alpha[j][i] = alpha

        # GPU0-GPU3 are topological twins (identical neighbor set/link weight).
        self.equivalent_node_indices = [[0, 1, 2, 3]]

    def set_switch_indicies(self) -> None:
        self.switch_indices = [4]
