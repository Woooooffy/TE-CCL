from teccl.hierarchy.cell import Cell
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class DualPlaneHeteroCluster(Topology):
    """
    96-GPU heterogeneous cluster on a DUAL-PLANE, two-tier fabric built from 64-port
    400 Gbps switches.

    Hardware being modeled
    ----------------------
      - 4 hosts x 16 GPUs  +  4 hosts x 8 GPUs  = 96 GPUs, each host a full-mesh NVSwitch.
      - Every GPU's NIC is broken out to 2 x 400 Gbps: ONE port into each plane.
        (A single unbroken 400 Gbps port would be the single-plane variant; the breakout
        is what makes two planes possible at all.)
      - Switches are 64-port x 400 Gbps (25.6 T). 96 endpoints per plane exceeds one
        switch, so each plane is its own two-tier leaf/spine fabric.

    Plane structure (identical in both planes)
    ------------------------------------------
      3 leaves, each 32-down / 32-up, and 2 spines:

          Leaf 0 : hosts 0,1        (16 + 16 = 32 GPUs)
          Leaf 1 : hosts 2,3        (16 + 16 = 32 GPUs)
          Leaf 2 : hosts 4,5,6,7    (8 x 4   = 32 GPUs)

      The heterogeneity partitions exactly: 32 GPUs per leaf and NO host is split across
      leaves, so TP/PP traffic that leaves the NVSwitch still lands one hop away. Each
      leaf sends 16 x 400 Gbps to each of the 2 spines (32 uplinks total), so the fabric
      is non-blocking; each spine uses 3 x 16 = 48 of its 64 ports.

      Port budget per plane: 3 leaves x 64 = 192 ports (fully used, 96 down + 96 up),
      2 spines x 48 = 96 of 128 ports used. 5 switches per plane, 10 switches total.

      Those 32 leaf up-ports are DECLARED, not just aggregated. Each leaf<->spine edge
      carries capacity 16 x 400 Gbps but `ports` says 16, so the post-solve split
      (teccl/ncclize/port_split.py) pins each flow to one 400 Gbps cable; without it a flow
      crossing the spine layer would appear to have 800 GB/s available on one edge -- the same
      overstatement the two-plane split exists to avoid on the GPU side.

      They are also NUMBERED, which is what the emitted forwarding table names. The default
      allocation orders a node's connections by neighbor index, and since GPUs hold the low
      indices and spines the high ones, a leaf comes out cabled the way the hardware is:

          leaf port  0-31   the 32 GPU downlinks, one 400 Gbps port each
          leaf port 32-47   the 16 cables to spine 0
          leaf port 48-63   the 16 cables to spine 1

      -- filling the 64-port radix exactly, which `radix()` below declares so that any edit
      making the fabric wider fails at construction instead of silently overrunning a switch.
      A spine numbers the far end of those same cables in ITS own space (3 leaves x 16 = ports
      0-47 of 64); a port number is only ever meaningful relative to the node holding it.

    Why the two planes are modeled as disjoint fabrics
    --------------------------------------------------
    There is deliberately NO link between plane 0 and plane 1 anywhere -- not at the leaf,
    not at the spine. The only place the planes meet is inside a GPU, which owns one port
    in each. That is what makes a plane a real failure domain: losing any single switch
    degrades bandwidth (a leaf failure drops 32 GPUs from 800 to 400 Gbps) but never
    disconnects a GPU. It also means the solver sees two independent 400 Gbps paths per
    GPU rather than one 800 Gbps pipe, which is the physically honest model -- a single
    flow cannot exceed 400 Gbps on one plane.

    NOTE on host injection: this models 800 Gbps per GPU across the two planes. On PCIe
    Gen5 x16 hosts (~400-500 Gbps usable) the second plane is resilience, not throughput;
    lower PORT_GBS or drop to one plane if you want to model the Gen5 ceiling.

    Fine node indexing (114 nodes)
    ------------------------------
        GPUs           0-95     hosts 0-3 own 16 each (0-15, 16-31, 32-47, 48-63),
                                hosts 4-7 own 8 each  (64-71, 72-79, 80-87, 88-95)
        NVSwitches     96-103   one per host
        Leaves         104-109  plane 0: 104,105,106   plane 1: 107,108,109
        Spines         110-113  plane 0: 110,111       plane 1: 112,113

    Coarse abstraction: each host collapses to one data-bearing node (its NVSwitch is
    internal and dropped); leaves and spines stay as coarse switch nodes. That gives an
    8 + 6 + 4 = 18-node coarse graph. A host's coarse uplink to leaf L of plane p has
    capacity (GPUs in host) x 400 Gbps, since every GPU owns a port into it -- 800 GB/s
    for a 16-GPU host, 400 GB/s for an 8-GPU host.
    """

    # (num_gpus, per-GPU NVSwitch bandwidth in GB/s). Both host classes are given the same
    # intra-node rate; if the 16-GPU generation has a faster NVLink domain, change it here.
    HOST_SPECS = [
        (16, 1800.0), (16, 1800.0), (16, 1800.0), (16, 1800.0),
        (8, 1800.0), (8, 1800.0), (8, 1800.0), (8, 1800.0),
    ]
    # host -> which leaf it hangs off, the SAME leaf index in both planes. Identical
    # placement across planes keeps a GPU pair's hop count equal on both planes, so NCCL
    # striping sees one consistent topology instead of two disagreeing ones.
    HOST_LEAF = [0, 0, 1, 1, 2, 2, 2, 2]

    NUM_PLANES = 2
    NUM_LEAF_PER_PLANE = 3
    NUM_SPINE_PER_PLANE = 2
    SWITCH_RADIX = 64            # ports per switch
    PORT_GBS = 50.0              # 400 Gbps, one port per GPU per plane

    NVSWITCH_ALPHA = 0.35 * pow(10, -6)
    NETWORK_ALPHA = 0.7 * pow(10, -6)

    def __init__(self, topo_input: TopologyParams):
        self._gpu_offset = []
        off = 0
        for num_gpus, _ in self.HOST_SPECS:
            self._gpu_offset.append(off)
            off += num_gpus
        self._num_gpus = off
        self._num_hosts = len(self.HOST_SPECS)
        self._nvswitch_base = self._num_gpus
        self._leaf_base = self._num_gpus + self._num_hosts
        self._spine_base = self._leaf_base + self.NUM_PLANES * self.NUM_LEAF_PER_PLANE
        self._total = self._spine_base + self.NUM_PLANES * self.NUM_SPINE_PER_PLANE
        super().__init__(topo_input)
        self.node_per_chassis = self.HOST_SPECS[0][0]

    # --- index helpers -----------------------------------------------------
    def _gpu(self, host: int, local: int) -> int:
        return self._gpu_offset[host] + local

    def _host_gpus(self, host: int):
        return [self._gpu(host, i) for i in range(self.HOST_SPECS[host][0])]

    def _nvswitch(self, host: int) -> int:
        return self._nvswitch_base + host

    # --- placement hook ----------------------------------------------------
    def _gpu_leaf(self, host: int, local: int) -> int:
        """
        Which leaf (same index in every plane) a given GPU hangs off. This is the ONE
        thing DualPlaneHeteroClusterScattered overrides -- everything else about the two
        topologies is shared code, so a solve difference between them is attributable to
        placement and nothing else. Here: whole host on one leaf.
        """
        return self.HOST_LEAF[host]

    def _leaf(self, plane: int, leaf: int) -> int:
        return self._leaf_base + plane * self.NUM_LEAF_PER_PLANE + leaf

    def _spine(self, plane: int, spine: int) -> int:
        return self._spine_base + plane * self.NUM_SPINE_PER_PLANE + spine

    # --- port budget -------------------------------------------------------
    def _leaf_spine_links(self) -> int:
        """
        Parallel 400 Gbps links between one leaf and one spine of the same plane, derived
        from the port budget rather than hard-coded so that editing HOST_LEAF or the radix
        keeps the fabric non-blocking (or fails loudly).
        """
        downlinks = [0] * self.NUM_LEAF_PER_PLANE
        for h in range(self._num_hosts):
            for local in range(self.HOST_SPECS[h][0]):
                downlinks[self._gpu_leaf(h, local)] += 1
        assert sum(downlinks) == self._num_gpus, \
            "every GPU must hang off exactly one leaf per plane"
        assert len(set(downlinks)) == 1, \
            f"leaves are unevenly loaded {downlinks}; a uniform leaf-spine link count " \
            f"would leave the fabric blocking on the fuller leaves"
        down = downlinks[0]
        up = self.SWITCH_RADIX - down
        assert up >= down, \
            f"leaf has {down} downlinks but only {up} uplinks: fabric is oversubscribed"
        assert up % self.NUM_SPINE_PER_PLANE == 0, \
            f"{up} uplinks do not split evenly over {self.NUM_SPINE_PER_PLANE} spines"
        per_spine = up // self.NUM_SPINE_PER_PLANE
        assert self.NUM_LEAF_PER_PLANE * per_spine <= self.SWITCH_RADIX, \
            f"spine needs {self.NUM_LEAF_PER_PLANE * per_spine} ports > radix {self.SWITCH_RADIX}"
        return per_spine

    def construct_topology(self, topo_input: TopologyParams) -> None:
        n = self._total
        per_spine_links = self._leaf_spine_links()
        # Parallel leaf->spine links are collapsed into one aggregated edge, as in
        # RailOptimizedSpineLeaf: 16 x 400 Gbps = 800 GB/s.
        leaf_spine_cap = (per_spine_links * self.PORT_GBS) / self.chunk_size
        gpu_leaf_cap = self.PORT_GBS / self.chunk_size

        edges = {}
        # Intra-host: every GPU <-> its NVSwitch.
        for h, (num_gpus, nvlink_gbs) in enumerate(self.HOST_SPECS):
            nv = self._nvswitch(h)
            for local in range(num_gpus):
                edges[(self._gpu(h, local), nv)] = (nvlink_gbs / self.chunk_size,
                                                    self.NVSWITCH_ALPHA)

        for p in range(self.NUM_PLANES):
            # Host downlinks: one 400 Gbps port per GPU into this plane's leaf.
            for h in range(self._num_hosts):
                for local in range(self.HOST_SPECS[h][0]):
                    leaf = self._leaf(p, self._gpu_leaf(h, local))
                    edges[(self._gpu(h, local), leaf)] = (gpu_leaf_cap, self.NETWORK_ALPHA)
            # Leaf-spine full mesh WITHIN the plane. No cross-plane edge exists.
            for l in range(self.NUM_LEAF_PER_PLANE):
                for s in range(self.NUM_SPINE_PER_PLANE):
                    edges[(self._leaf(p, l), self._spine(p, s))] = (leaf_spine_cap,
                                                                    self.NETWORK_ALPHA)

        self.capacity = [[0.0] * n for _ in range(n)]
        self.alpha = [[-1.0] * n for _ in range(n)]
        for (i, j), (cap, alpha) in edges.items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = alpha
            self.alpha[j][i] = alpha

        # Parallel-port declaration for the leaf uplinks. Each leaf<->spine edge above is an
        # AGGREGATE of `per_spine_links` physical 400 Gbps ports, so what the aggregate hides
        # is precisely the leaf's 32 up-ports: per_spine_links to each of the
        # NUM_SPINE_PER_PLANE spines. Declaring them lets the post-solve split
        # (teccl/ncclize/port_split.py) place each flow on ONE 400 Gbps port instead of
        # letting it spread across the 800 GB/s bundle, which no single flow can do.
        #
        # Structurally invisible: `capacity` is untouched, so every solve is bit-for-bit what
        # it was before -- nothing upstream of port_split.py reads `ports`. The GPU->leaf
        # downlinks are genuinely one port each and stay at the default 1.
        if per_spine_links > 1:
            self.ports = [[1] * n for _ in range(n)]
            for p in range(self.NUM_PLANES):
                for l in range(self.NUM_LEAF_PER_PLANE):
                    for s in range(self.NUM_SPINE_PER_PLANE):
                        i, j = self._leaf(p, l), self._spine(p, s)
                        self.ports[i][j] = self.ports[j][i] = per_spine_links

        # The spines within a plane are topological twins: each connects to all leaves of
        # its own plane with identical weights. Spines of DIFFERENT planes are not twins --
        # they have disjoint neighbor sets, which is exactly the point of the design. Leaves
        # are not twins either, since each serves a different set of hosts.
        self.equivalent_node_indices = [
            [self._spine(p, s) for s in range(self.NUM_SPINE_PER_PLANE)]
            for p in range(self.NUM_PLANES)
        ]

    def set_switch_indicies(self) -> None:
        self.switch_indices = (
            [self._nvswitch(h) for h in range(self._num_hosts)]
            + [self._leaf(p, l) for p in range(self.NUM_PLANES)
               for l in range(self.NUM_LEAF_PER_PLANE)]
            + [self._spine(p, s) for p in range(self.NUM_PLANES)
               for s in range(self.NUM_SPINE_PER_PLANE)]
        )

    def radix(self, node: int) -> int | None:
        """Leaves and spines are real 64-port parts; declare it so running out of ports asserts.

        The NVSwitches and GPUs are left as None -- SWITCH_RADIX is a fact about the network
        switches this class models, and claiming it for a per-host crossbar would be inventing
        one. Nothing reads their width anyway: neither carries a port map.
        """
        if node >= self._leaf_base:
            return self.SWITCH_RADIX
        return None

    def default_programmable_switch_indices(self):
        # Only the leaf/spine fabric is programmed per route; the per-host NVSwitch is a
        # self-routing crossbar and is left out of the emitted table.
        return ([self._leaf(p, l) for p in range(self.NUM_PLANES)
                 for l in range(self.NUM_LEAF_PER_PLANE)]
                + [self._spine(p, s) for p in range(self.NUM_PLANES)
                   for s in range(self.NUM_SPINE_PER_PLANE)])

    def build_hierarchy(self) -> None:
        # One cell per host: its GPUs + its NVSwitch collapse to a single coarse node. Each
        # host is dual-homed, so its boundary names TWO external neighbors -- its leaf in
        # plane 0 and its leaf in plane 1 -- and every GPU in the host owns a port to each.
        self.cells = []
        for h in range(self._num_hosts):
            gpus = self._host_gpus(h)
            nv = self._nvswitch(h)
            boundary = {}
            for local, g in enumerate(gpus):
                for p in range(self.NUM_PLANES):
                    boundary.setdefault(self._leaf(p, self._gpu_leaf(h, local)), []).append(g)
            self.cells.append(Cell(
                members=gpus + [nv],
                gpus=gpus,
                internal_switches=[nv],
                boundary=boundary,
            ))


class DualPlaneHeteroClusterScattered(DualPlaneHeteroCluster):
    """
    Rail-style SCATTERED placement variant of DualPlaneHeteroCluster, built for A/B
    comparison against it. Identical hardware, identical switch count, identical
    capacities, identical port budget -- the ONLY difference is which leaf each GPU
    hangs off (`_gpu_leaf`), so any solve difference is attributable to placement.

    Placement: global round-robin, GPU g -> leaf (g % 3), applied identically in both
    planes. 96 GPUs / 3 leaves divides exactly, so each leaf still gets precisely 32
    downlinks and the fabric stays non-blocking at 32-down / 32-up.

    What scattering costs: the split is RAGGED, because neither host size divides by 3.

        host 0 (16 GPUs) -> 6 / 5 / 5 across leaves 0 / 1 / 2
        host 1 (16 GPUs) -> 5 / 6 / 5
        host 2 (16 GPUs) -> 5 / 5 / 6
        host 3 (16 GPUs) -> 6 / 5 / 5
        host 4 ( 8 GPUs) -> 2 / 3 / 3
        host 5 ( 8 GPUs) -> 3 / 3 / 2
        host 6 ( 8 GPUs) -> 3 / 2 / 3
        host 7 ( 8 GPUs) -> 2 / 3 / 3

    This is why true rail-optimization does not apply here. Rail-optimization pays off
    when GPU i of every host lands on the same switch, so the inter-node phase of an
    allreduce (rank i <-> rank i) stays one hop. A 6/5/5 split admits no such alignment,
    so this variant gets the scattering WITHOUT the rail payoff.

    What scattering buys, and it is real: leaf locality stops being spent on same-host
    pairs that would have used NVLink anyway. Under clustered placement leaf 0 holds
    hosts 0 and 1, so most of its intra-leaf GPU pairs are same-host and never touch the
    leaf at all. Scattering redirects that locality to cross-host pairs. Measured over
    all 3968 cross-host GPU pairs, the fraction that share a leaf (one network hop rather
    than three) goes 22.6% clustered -> 33.3% scattered.

    What scattering costs on failure: losing one leaf takes 5-6 GPUs out of EVERY host,
    leaving eight crippled partial hosts with broken TP groups, versus losing whole hosts
    (and keeping the rest intact and schedulable) under clustered placement. The other
    plane keeps every GPU reachable in both cases, so this is a degradation-shape
    difference, not a connectivity one.

    Which effect dominates is workload-dependent, which is the point of having both:
    solve the same collective on each and compare, rather than arguing from first
    principles.
    """

    def _gpu_leaf(self, host: int, local: int) -> int:
        return self._gpu(host, local) % self.NUM_LEAF_PER_PLANE
