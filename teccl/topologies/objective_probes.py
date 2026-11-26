"""
Three minimal two-GPU topologies used to probe an objective's routing PREFERENCE.

Each one gives GPU0 and GPU1 two alternative ways to reach each other, so an AllGather on it
has more than one optimal-makespan schedule and the choice between them is made purely by the
objective. They are deliberately tiny (4-5 nodes, 1 chunk) so a solve is instant and the
resulting flows can be read off by eye.

    DirectVsSwitch     GPU0 =========== GPU1        (direct, 0 switch hops)
                       GPU0 --- S0 ---- GPU1        (1 switch hop)

    OneVsTwoSwitch     GPU0 --- S0 ---- GPU1        (1 switch hop)
                       GPU0 -- S1 - S2 - GPU1       (2 switch hops)

    SlowSwitchVsRelay  GPU0 -.- S0 -.-- GPU1        (1 switch hop, SLOW links)
                       GPU0 === G2 ==== GPU1        (relay through a passive GPU, fast links)

All links are zero-alpha and, except where a test explicitly slows one down, carry exactly one
chunk per epoch, so latency alone cannot separate the alternatives.

See teccl/examples/objective_lexicographic_test.py for what each is expected to show.
"""
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology

# One chunk per epoch on a normal link (paired with epoch_duration = 1.0 in the tests).
LINK_CAPACITY = 1.0
NO_ALPHA = 0.0


class _ProbeTopology(Topology):
    """Shared plumbing: build capacity/alpha from an undirected edge dict."""

    EDGES = {}          # {(i, j): capacity}
    NUM_NODES = 0
    SWITCHES = []

    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)
        self.node_per_chassis = self.NUM_NODES

    def construct_topology(self, topo_input: TopologyParams) -> None:
        n = self.NUM_NODES
        self.capacity = [[0.0] * n for _ in range(n)]
        self.alpha = [[-1.0] * n for _ in range(n)]
        for (i, j), cap in self.edges().items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = NO_ALPHA
            self.alpha[j][i] = NO_ALPHA

    def edges(self):
        return self.EDGES

    def set_switch_indicies(self) -> None:
        self.switch_indices = list(self.SWITCHES)


class DirectVsSwitch(_ProbeTopology):
    """
    GPU0 and GPU1 have BOTH a direct link and a one-switch path, all links equal bandwidth.

        GPU0 ============ GPU1
          \\              /
           +---- S2 ----+

    Node indices: 0=GPU0, 1=GPU1, 2=Switch
    """
    NUM_NODES = 3
    SWITCHES = [2]
    EDGES = {
        (0, 1): LINK_CAPACITY,   # direct
        (0, 2): LINK_CAPACITY,   # GPU0 - switch
        (1, 2): LINK_CAPACITY,   # GPU1 - switch
    }


class OneVsTwoSwitch(_ProbeTopology):
    """
    GPU0 and GPU1 have no direct link, but two switch paths of different chain length, all
    links equal bandwidth.

        GPU0 ---- S2 ---- GPU1               (chain length 1)
        GPU0 -- S3 -- S4 -- GPU1             (chain length 2)

    Node indices: 0=GPU0, 1=GPU1, 2=Switch(short), 3/4=Switches(long)
    """
    NUM_NODES = 5
    SWITCHES = [2, 3, 4]
    EDGES = {
        (0, 2): LINK_CAPACITY,   # GPU0 - short switch
        (1, 2): LINK_CAPACITY,   # short switch - GPU1
        (0, 3): LINK_CAPACITY,   # GPU0 - long switch A
        (3, 4): LINK_CAPACITY,   # long switch A - long switch B
        (1, 4): LINK_CAPACITY,   # long switch B - GPU1
    }


class SlowSwitchVsRelay(_ProbeTopology):
    """
    GPU0 and GPU1 have a SLOW one-switch path and a FAST path that relays through GPU2, a
    passive GPU (present and able to forward, but sourcing/sinking no demand of its own).

        GPU0 -- S3 -- GPU1        switch links at SWITCH_CAPACITY (slow by default)
        GPU0 == G2 == GPU1        relay links at LINK_CAPACITY

    Node indices: 0=GPU0, 1=GPU1, 2=passive GPU relay, 3=Switch

    SWITCH_CAPACITY is a class attribute so a test can re-run the same shape with the switch
    path made just as fast as the relay path (see objective_lexicographic_test).
    """
    NUM_NODES = 4
    SWITCHES = [3]
    SWITCH_CAPACITY = LINK_CAPACITY / 4   # 4 epochs per chunk vs the relay's 1 per hop
    PASSIVE = (2,)

    def edges(self):
        return {
            (0, 3): self.SWITCH_CAPACITY,   # GPU0 - switch  (slow)
            (1, 3): self.SWITCH_CAPACITY,   # switch - GPU1  (slow)
            (0, 2): LINK_CAPACITY,          # GPU0 - relay GPU (fast)
            (1, 2): LINK_CAPACITY,          # relay GPU - GPU1 (fast)
        }


class FastSwitchVsRelay(SlowSwitchVsRelay):
    """
    SlowSwitchVsRelay with the switch path sped up to match the relay path, so the two routes
    take the same number of epochs and only the relay penalty separates them.
    """
    SWITCH_CAPACITY = LINK_CAPACITY
