from abc import ABC, abstractmethod
from collections import Counter
from itertools import product
from typing import Dict, List, Optional, Tuple

from teccl.hierarchy.cell import Cell
from teccl.input_data import *


class Topology(ABC):
    """A fine-grained network: nodes, link capacities, and the physical ports realizing them.

    PORTS COME IN TWO LAYERS, and keeping them apart is what makes the model both packable and
    physically honest:

      * a CABLE INDEX, 0..ports[i][j]-1, names one of the parallel links realizing edge (i, j).
        It is LINK-local and always meaningful, on every edge of every topology. Leaving `u` on
        cable `c` of (u, v) IS arriving at `v` on cable `c` of the same edge, which is what lets
        the post-solve packing (teccl/ncclize/port_split.py) follow a flow across hops.

      * a PORT NUMBER, 0..radix-1, names a socket on a specific node. It is NODE-local, and the
        map from connection to port number is `port_map`. This is the thing real hardware has:
        a 64-port leaf switch whose ports 0-31 take GPU downlinks and 32-63 take spine uplinks.

    Only PROGRAMMABLE switches carry a port map, because the emitted forwarding table is the
    only consumer of a port number and it covers exactly those switches. Every other node --
    GPUs, self-routing NVSwitches -- has `physical_port() is None`, which states that nothing
    here numbered its sockets rather than pretending something did.
    """

    @abstractmethod
    def __init__(self, topo_input: TopologyParams):
        self.chassis = topo_input.chassis
        self.chunk_size = topo_input.chunk_size
        self.capacity: List[List[float]] = []
        self.alpha: List[List[float]] = []
        # How many PARALLEL PHYSICAL PORTS realize each modeled link -- the CABLE COUNT of
        # edge (i, j). `capacity` stays the AGGREGATE, so the solver is unaffected: it never
        # reads this. A post-solve pass (teccl/ncclize/port_split.py) uses it to place each
        # flow on one cable of `capacity[i][j] / ports[i][j]`. Left empty => one cable per
        # link. See the class docstring for how a cable index relates to a PORT NUMBER.
        self.ports: List[List[int]] = []
        # node -> {(neighbor, cable): port number on THIS node}. Real hardware numbers the
        # ports of a switch 0..radix-1 and plugs one cable into each; this is that plugging
        # record. Built by _build_port_map() at the end of __init__, and populated ONLY for
        # programmable switches -- see the class docstring.
        self.port_map: Dict[int, Dict[Tuple[int, int], int]] = {}
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
        self._check_ports()
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
        self._build_port_map()
        self.build_hierarchy()

    def _check_ports(self) -> None:
        """Validate an optional `ports` matrix, or leave it empty (== one port everywhere).

        Checked here rather than in each construct_topology so a subclass only has to set the
        entries it splits. The equal-width restriction is real: `port_split` divides the
        aggregate evenly, so an odd split would have to be expressed as a list of per-port
        capacities instead of a count.
        """
        if not self.ports:
            return
        n = len(self.capacity)
        assert len(self.ports) == n and all(len(r) == n for r in self.ports), \
            "ports must be the same shape as capacity"
        for i in range(n):
            for j in range(n):
                p = self.ports[i][j]
                assert p >= 1 and int(p) == p, f"ports[{i}][{j}] = {p} must be a positive int"
                assert self.ports[i][j] == self.ports[j][i], \
                    f"ports must be symmetric; ports[{i}][{j}] != ports[{j}][{i}]"
                if p > 1:
                    assert self.capacity[i][j] > 0, \
                        f"ports[{i}][{j}] = {p} on an unused link"

    def port_count(self, i: int, j: int) -> int:
        """Cables realizing link (i, j) -- so also the ports it consumes at EACH end. 1 by
        default."""
        return self.ports[i][j] if self.ports else 1

    def port_capacity(self, i: int, j: int) -> float:
        """Bandwidth of ONE port of link (i, j), in the same units as `capacity`."""
        return self.capacity[i][j] / self.port_count(i, j)

    # --- physical port numbering ------------------------------------------------------------
    def neighbors(self, node: int) -> List[int]:
        """Nodes `node` has a link to, ascending. The graph is symmetric, so direction-free."""
        return [j for j, c in enumerate(self.capacity[node]) if c > 0]

    def radix(self, node: int) -> Optional[int]:
        """How many ports the hardware at `node` physically has, or None if it does not say.

        Override in a topology that models a real part -- a 64-port switch has 64 ports whether
        or not the model plugs a cable into every one, and saying so is what turns "I ran out of
        ports" into an assertion at construction rather than a silent overrun. None means the
        node is exactly as wide as its cables, which is the honest default for an abstract node.
        """
        return None

    def port_order(self, node: int) -> List[Tuple[int, int]]:
        """The connections of `node` in the order they claim port numbers: port p is entry p.

        Default: neighbors ascending, cables of one neighbor consecutive. That is not arbitrary
        -- it is the order a two-tier fabric is actually cabled, since a leaf's GPUs occupy the
        low node indices and its spines the high ones, so downlinks land on the low ports and
        uplinks on the high ones by themselves.

        Override when the real part pins a specific plug to a specific port. The base class only
        requires that the result be a permutation of the node's connections, and enforces
        exactly that (_build_port_map, _check_port_map), so a wrong override fails at
        construction rather than emitting ports no switch has.
        """
        return [(nbr, cable)
                for nbr in self.neighbors(node)
                for cable in range(self.port_count(node, nbr))]

    def _port_mapped_nodes(self) -> List[int]:
        """Which nodes get a port map. Programmable switches, and only them.

        A port number is a fiction unless something reads it, and the one consumer is the
        emitted forwarding table, which covers exactly the programmable switches. Numbering a
        GPU's NICs or a self-routing NVSwitch's ports would be inventing a fact about hardware
        nobody checks -- so those nodes map to None and say so, rather than carrying numbers
        that were never verified against anything.

        Note what is NOT skipped: a programmable switch numbers ALL of its connections,
        including the ones to unmapped nodes. A leaf's downlink to a GPU is a port on the LEAF,
        which is the end that has to be programmed.
        """
        return sorted(self.programmable_switch_indices)

    def _build_port_map(self) -> None:
        """Assign every connection of every mapped node a port number, then validate.

        Port numbers are POSITIONAL -- port p is entry p of port_order -- so no two connections
        can land on one port by construction. The one thing positional assignment cannot rule
        out is the same connection appearing twice, which the dict would silently collapse into
        a missing one; caught on the list, before the collapse, so the message names the actual
        mistake instead of its shadow.
        """
        self.port_map = {}
        for node in self._port_mapped_nodes():
            order = self.port_order(node)
            seen = Counter(order)
            dupes = sorted(c for c, n in seen.items() if n > 1)
            assert not dupes, \
                f"port_order({node}) plugs {dupes} into more than one port"
            self.port_map[node] = {conn: port for port, conn in enumerate(order)}
        self._check_port_map()

    def _check_port_map(self) -> None:
        """A port map must be a real plugging record, not just a dict.

        Two properties, each catching a different way an override goes wrong: the mapping is
        exactly the node's set of connections (nothing dropped, nothing invented), and every
        port number is one the hardware actually has (nothing overrun). Double-booking is ruled
        out upstream by positional assignment -- see _build_port_map.
        """
        for node, mapping in self.port_map.items():
            expected = {(nbr, cable)
                        for nbr in self.neighbors(node)
                        for cable in range(self.port_count(node, nbr))}
            missing = expected - set(mapping)
            assert not missing, \
                f"port_order({node}) leaves {sorted(missing)} unplugged"
            extra = set(mapping) - expected
            assert not extra, \
                f"port_order({node}) plugs in {sorted(extra)}, which are not connections of it"
            width = self.radix(node)
            if width is not None:
                assert len(mapping) <= width, \
                    f"node {node} needs {len(mapping)} ports but its radix is {width}"
                over = sorted(p for p in mapping.values() if not 0 <= p < width)
                assert not over, \
                    f"node {node} assigns ports {over} outside its radix {width}"

    def physical_port(self, node: int, neighbor: int, cable: int = 0) -> Optional[int]:
        """The port number on `node` that cable `cable` of link (node, neighbor) is plugged into.

        None when `node` carries no port map -- a GPU or a self-routing switch. Callers emitting
        a forwarding table should treat None as "this node is not programmed", which is exactly
        what it means, rather than substituting a default port number.
        """
        mapping = self.port_map.get(node)
        return None if mapping is None else mapping[(neighbor, cable)]

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
