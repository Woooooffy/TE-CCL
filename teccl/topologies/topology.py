from abc import ABC, abstractmethod
from itertools import product
from typing import List

from teccl.hierarchy.cell import Cell
from teccl.input_data import *


class Topology(ABC):
    @abstractmethod
    def __init__(self, topo_input: TopologyParams):
        self.chassis = topo_input.chassis
        self.chunk_size = topo_input.chunk_size
        self.capacity: List[List[float]] = []
        self.alpha: List[List[float]] = []
        self.switch_indices: List[int] = []
        self.equivalent_node_indices: List[List[int]] = []
        # Optional hierarchy: groups of fine nodes (e.g. an 8-GPU + NVSwitch host) that a
        # hierarchical solve collapses into one coarse node. Empty => flat solve (default);
        # build_hierarchy() is a no-op unless a subclass overrides it, so every existing
        # topology is unaffected. See teccl/hierarchy and the hierarchical_solver_design note.
        self.cells: List[Cell] = []
        self.epoch_duration_fast_link = 0.0
        self.epoch_duration_slow_link = 0.0
        self.node_per_chassis = 0
        self.side_length = topo_input.side_length # Only for Mesh and Torus topology
        self.construct_topology(topo_input)
        assert len(
            self.capacity) > 0, "Link capacities not set in the construct_topology function"
        assert len(
            self.alpha) > 0, "Link alphas not set in the construct_topology function "
        self.get_epoch_duration_fast_link()
        self.get_epoch_duration_slow_link()
        self.set_switch_indicies()
        self.passive_indices: List[int] = list(topo_input.passive_node_indices)
        assert not set(self.passive_indices) & set(self.switch_indices), \
            "A node cannot be both a switch and a passive forwarding node"
        self.programmable_switch_indices: List[int] = (
            list(topo_input.programmable_switch_indices)
            if topo_input.programmable_switch_indices is not None
            else self.default_programmable_switch_indices())
        assert set(self.programmable_switch_indices) <= set(self.switch_indices), \
            "programmable_switch_indices must be a subset of switch_indices"
        self.build_hierarchy()

    @abstractmethod
    def construct_topology(self, topo_input: TopologyParams) -> None:
        pass

    @abstractmethod
    def set_switch_indicies(self) -> None:
        pass

    def default_programmable_switch_indices(self) -> List[int]:
        """
        Which of self.switch_indices are externally programmable, i.e. which ones ncclize may
        emit a forwarding table for. Default: all of them.

        A subclass overrides this when only some of its switches take an external program. The
        canonical split is network switches (leaf/spine, programmed per route) vs intra-node
        NVSwitches, which route on their own and must be left out of the emitted table even
        though the solver freely routes through them. Overridable per-instance by
        TopologyParams.programmable_switch_indices. Called after set_switch_indicies().
        """
        return list(self.switch_indices)

    def build_hierarchy(self) -> None:
        """
        Populate self.cells for a hierarchical solve. Default: no hierarchy (flat solve).

        A subclass overrides this to declare its coarse structure in FINE indices (see
        teccl.hierarchy.Cell). Called at the end of __init__, after switch_indices and
        passive_indices are set, so a cell may reference them.
        """
        pass

    def compute_pairwise_hop_distance(self) -> None:
        INF = float("inf")
        hop_distance = []
        for i, row in enumerate(self.capacity):
            dist_row = []
            for j, c in enumerate(row):
                if c > 0:
                    dist_row.append(1)
                else:
                    dist_row.append(INF)
            hop_distance.append(dist_row)
        n = len(self.capacity)
        for k, i, j in product(range(n), repeat=3):
            hop_distance[i][j] = min(
                hop_distance[i][j], hop_distance[i][k] + hop_distance[k][j])
        self.hop_distances = hop_distance

    def get_max_hop_distance(self) -> int:
        self.compute_pairwise_hop_distance()
        return max([max(filter(lambda x: x != float("inf"), row)) for row in self.hop_distances])

    def get_largest_time_chunk(self) -> float:
        max_time = 0
        for i, crow in enumerate(self.capacity):
            for j, c in enumerate(crow):
                if c > 0:
                    time_for_chunk = (1.0 / c) + self.alpha[i][j]
                    max_time = max(max_time, time_for_chunk)
        assert max_time > 0, "Max time chunk is 0"
        return max_time

    def get_min_alpha(self) -> float:
        return min([min(filter(lambda x: x >= 0, row)) for row in self.alpha])

    def get_epoch_duration_fast_link(self) -> float:
        if self.epoch_duration_fast_link != 0:
            return self.epoch_duration_fast_link
        self.epoch_duration_fast_link = max(
            [max([x for x in self.capacity[i] if x != 0]) for i in range(len(self.capacity))])
        self.epoch_duration_fast_link = (1.0 / self.epoch_duration_fast_link)
        return self.epoch_duration_fast_link

    def get_epoch_duration_slow_link(self) -> float:
        if self.epoch_duration_slow_link != 0:
            return self.epoch_duration_slow_link
        self.epoch_duration_slow_link = min(
            [min([x for x in self.capacity[i] if x != 0]) for i in range(len(self.capacity))])
        self.epoch_duration_slow_link = (1.0 / self.epoch_duration_slow_link)
        return self.epoch_duration_slow_link
