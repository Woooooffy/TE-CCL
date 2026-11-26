"""
A THREE-level topology: cluster -> racks -> hosts. The first topology whose hierarchy is deeper than
the two levels the solver was originally built for.

Every existing hierarchical topology is exactly two levels (a network of hosts, each host an
NVSwitch), so the recursion had no way to be exercised: the "second level" was always the last one.
This one exists to make the recursive call real and locally testable, and its dimensions are chosen
for that purpose rather than to model a particular machine.

    top switch  ---- 3 racks
      rack r    = rack switch + 3 hosts
      host h    = NVSwitch + 2 GPUs

    GPU 0 of each host uplinks to its rack switch; GPU 0 of host 0 in each rack uplinks to the
    top switch. Everything else reaches the outside world through a relay.

THE KEY PROPERTY, and the reason the dimensions are what they are: **every level's graph collapses
to a crossbar.**

    level 0 (cluster): abstract() collapses each rack -> {3 rack-nodes, 1 top switch}   crossbar
    level 1 (rack):    abstract() collapses each host -> {3 host-nodes, 1 rack switch}  crossbar
    level 2 (host):    2 GPUs on 1 NVSwitch                                             crossbar

So the whole three-level solve takes the memoized closed-form row of the base-case dispatch table at
every level and needs no Gurobi at all -- which is what makes `hierarchy_recursion_test` a local
test rather than a remote one. That is a property of this topology, NOT of the recursion: a level
whose coarse graph has two switches falls through to a real formulation exactly as it should.

The narrow uplinks are the other deliberate choice. One uplink per host and one per rack means most
data cannot leave where it lives, so egress staging and ingress fan-out are FORCED at both interior
levels -- which is precisely the machinery a symmetric topology never triggers, and the reason
HeteroTaperedCluster had to be built for the two-level case.
"""
from typing import List

from teccl.hierarchy.cell import Cell
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class NestedCluster(Topology):
    NUM_RACKS = 3
    HOSTS_PER_RACK = 3
    GPUS_PER_HOST = 2

    # Bandwidths, deliberately tiered so each level is genuinely faster than the one above it --
    # the premise the whole band construction rests on ("the inner fabric hides under the outer").
    NVLINK_BW = 400.0     # gpu <-> its host NVSwitch
    RACK_BW = 100.0       # host gateway gpu <-> rack switch
    TOP_BW = 25.0         # rack gateway gpu <-> top switch
    ALPHA = 0.0

    def __init__(self, topo_input: TopologyParams) -> None:
        super().__init__(topo_input)

    # -- index layout -----------------------------------------------------------------------
    # GPUs first (dense, so demand builders see a clean participant block), then the per-host
    # NVSwitches, then the per-rack switches, then the single top switch.
    def _num_hosts(self) -> int:
        return self.NUM_RACKS * self.HOSTS_PER_RACK

    def _num_gpus(self) -> int:
        return self._num_hosts() * self.GPUS_PER_HOST

    def _gpu(self, rack: int, host: int, local: int) -> int:
        return ((rack * self.HOSTS_PER_RACK) + host) * self.GPUS_PER_HOST + local

    def _nvswitch(self, rack: int, host: int) -> int:
        return self._num_gpus() + (rack * self.HOSTS_PER_RACK) + host

    def _rack_switch(self, rack: int) -> int:
        return self._num_gpus() + self._num_hosts() + rack

    def _top(self) -> int:
        return self._num_gpus() + self._num_hosts() + self.NUM_RACKS

    def construct_topology(self, topo_input: TopologyParams) -> None:
        n = self._num_gpus() + self._num_hosts() + self.NUM_RACKS + 1
        self.capacity = [[0.0] * n for _ in range(n)]
        self.alpha = [[-1.0] * n for _ in range(n)]

        def link(i: int, j: int, bw: float) -> None:
            self.capacity[i][j] = self.capacity[j][i] = bw
            self.alpha[i][j] = self.alpha[j][i] = self.ALPHA

        for r in range(self.NUM_RACKS):
            for h in range(self.HOSTS_PER_RACK):
                nv = self._nvswitch(r, h)
                for g in range(self.GPUS_PER_HOST):
                    link(self._gpu(r, h, g), nv, self.NVLINK_BW)
                # One uplink per host: only local GPU 0 can reach the rack switch, so anything
                # another GPU needs to send out of the host must be staged onto it first.
                link(self._gpu(r, h, 0), self._rack_switch(r), self.RACK_BW)
            # One uplink per rack, owned by host 0's gateway GPU. Same forcing, one level up.
            link(self._gpu(r, 0, 0), self._top(), self.TOP_BW)

        self.equivalent_node_indices = []
        self.node_per_chassis = self.GPUS_PER_HOST

    def set_switch_indicies(self) -> None:
        self.switch_indices = (
            [self._nvswitch(r, h) for r in range(self.NUM_RACKS)
             for h in range(self.HOSTS_PER_RACK)]
            + [self._rack_switch(r) for r in range(self.NUM_RACKS)]
            + [self._top()]
        )

    def build_hierarchy(self) -> None:
        """One cell per RACK, each declaring its hosts as `subcells`.

        The rack cell owns everything inside the rack -- all its GPUs, all its host NVSwitches, and
        the rack switch -- because `abstract()` drops a cell's internal edges, and every one of
        those is a link the rack's OWN level is responsible for scheduling. `subcells` is what tells
        `solve_level` that the rack is itself a hierarchical problem rather than a bottom cell, so
        it recurses instead of handing the rack straight to the crossbar solver.
        """
        self.cells = []
        for r in range(self.NUM_RACKS):
            gpus: List[int] = []
            internal: List[int] = [self._rack_switch(r)]
            subcells: List[Cell] = []
            for h in range(self.HOSTS_PER_RACK):
                host_gpus = [self._gpu(r, h, g) for g in range(self.GPUS_PER_HOST)]
                nv = self._nvswitch(r, h)
                gpus += host_gpus
                internal.append(nv)
                subcells.append(Cell(
                    members=host_gpus + [nv],
                    gpus=host_gpus,
                    internal_switches=[nv],
                    # Inside the rack, the host's one external neighbor is the rack switch.
                    boundary={self._rack_switch(r): [self._gpu(r, h, 0)]},
                ))
            self.cells.append(Cell(
                members=gpus + internal,
                gpus=gpus,
                internal_switches=internal,
                # Outside the rack, its one external neighbor is the top switch.
                boundary={self._top(): [self._gpu(r, 0, 0)]},
                subcells=subcells,
            ))
