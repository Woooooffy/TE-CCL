from typing import Sequence, Tuple, Type

from teccl.hierarchy.cell import Cell
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class TwoPodRail(Topology):
    """
    16-GPU / 8-node / 2-pod rail-optimized topology, CONTRIVED so that the optimal schedule is
    forced to (1) split a flow across multiple paths and (2) pace some flows below link rate.

    It is small enough (30 nodes) that the FLAT solve is tractable, so it doubles as the
    ground truth the hierarchical stitch can be validated against -- which neither
    RailOptimizedSpineLeaf (too big) nor HeteroTaperedCluster (no flat reference computed)
    provides.

        Pod A = nodes 0-3, Pod B = nodes 4-7. Each node holds 2 GPUs behind one NVSwitch.
        Leaf(pod p, rail r) serves the rail-r GPU of every node in pod p -> 4 leaves.
        Every leaf reaches BOTH spines. Cross-pod traffic must therefore cross a spine; the
        two spines are deliberately UNEQUAL (50 vs 25 GB/s), which is what makes the split
        forced and non-dyadic instead of a symmetric tie.

    Node indexing (30 nodes):
        GPU(node n, rail r) = n * 2 + r        n in [0,8), r in [0,2)   -> [0,16)
        NVSwitch(node n)    = 16 + n           n in [0,8)               -> [16,24)
        Leaf(pod p, rail r) = 24 + p * 2 + r   p in [0,2), r in [0,2)   -> [24,28)
        Spine(s)            = 28 + s           s in [0,2)               -> [28,30)

    WHY THE TWO GPUs SIT BEHIND AN NVSWITCH RATHER THAN A DIRECT LINK
    Physically a direct GPU<->GPU NVLink would be the honest model of a 2-GPU node, and it is what
    this topology had first. It does not work, for a reason that has nothing to do with the fine
    graph: a bottom cell's interior is scheduled by a CLOSED-FORM ROW, and a switchless 2-node
    fabric matches neither row on offer. `crossbar_solve.is_crossbar` requires exactly one switch;
    `ring_solve.ring_topology_order` requires at least three data nodes each with two undirected
    neighbours -- and its >= 3 guard is substantive, not incidental, because in a 2-cycle a node's
    clockwise and counter-clockwise neighbours are the same node over the SAME link, so the
    `bidirectional=True` it would infer would double-count that GPU's egress. With one NVSwitch the
    cell is an ordinary crossbar, exactly as in every other topology here.

    The extra hop costs nothing that this topology measures: at 900 GB/s the two-hop
    GPU->NVSwitch->GPU path is as far from binding as the direct link was, and neither the spine
    cut nor the GPU->leaf cut involves an intra-node link at all. Every number below is unchanged
    from the direct-link version.

    WHY THE SPINES ARE UNEQUAL
    A symmetric pair of spines makes the two spine paths exactly interchangeable: the aggregate
    split is pinned at 50/50 by symmetry alone and the per-pair assignment is free, so the LP
    lands on an arbitrary point of a degenerate face and a wrong router is indistinguishable
    from a right one. At 50/25 the optimal split is 2:1, so an equal-split (ECMP-style) router
    is provably, measurably wrong -- see the regression table below.

    ANALYSIS AT THE DEFAULT DESIGN POINT (AllGather, LP semantics: no switch copy, so every
    destination needs its own delivery). Normalize one GPU's payload to 1 unit; times are in
    units of (1 unit) / (1 GB/s).

        constraint                     load                    capacity        time LB
        spine cut A->B                 8 srcs x 8 dsts = 64    2 x 75 = 150    0.4267  <- binds
        pod-A GPU->leaf egress         64 cross + 48 intra     8 x 50 = 400    0.28
        pod-A leaf->GPU ingress        48 intra + 64 cross     400             0.28
        NVLink (pod A)                 ~32                     4 x 900         0.009

    T* = 0.4267. Per leaf: downlink 4 x 50 = 200 against uplink 50 + 25 = 75, a 2.67:1 taper.

    GOAL 1 -- MULTIPATH. 8 distinct paths per cross-pod pair (2 spines x 2 ingress rails x 2
    egress rails). Both spine links out of every leaf are saturated for the whole makespan:
    leaf0 moves 32 units in 0.4267s = 75 GB/s = exactly S0 + S1. The split is forced and unique
    at 2/3 via spine0, 1/3 via spine1. Getting it wrong is loud:

        optimal 2:1 split          0.4267      --
        equal split (ECMP hash)    16/25 = 0.64    1.50x
        single spine (all spine0)  32/50 = 0.64    1.50x

    Note this dimension is tested at CHUNK granularity, not fluid rate: a load-aware router
    that assigns whole chunks per pair can hit 2:1 to within granularity. Set num_chunks
    divisible by 3 (6 or 12) or the integer path cannot express 2:1 and you will read a
    granularity artifact as a solver bug.

    GOAL 2 -- RATE LIMITING. Switches here are bufferless (LPFormulation pins the buffer
    variable to 0 at every switch), so a leaf's ingress rate must EQUAL its egress rate: a GPU
    cannot burst into a leaf and let it drain later. That pins rates rather than bounding them.
    Per pod-A GPU, on its single 50 GB/s link to its leaf:

        cross-pod stream (8 units)   18.75 GB/s  (= leaf uplink 75 shared by its 4 GPUs)
          via spine0                 12.50
          via spine1                  6.25
        intra-pod stream (6 units)   14.06 GB/s average

    Two concurrent flows at different, both sub-line-rate speeds on one wire, and the cross-pod
    one further decomposes into two per-path rates. None of that is expressible as "send this
    chunk as fast as the link allows".

    WHY GPU_LEAF_BW IS THE SWEPT KNOB
    Which cut binds is set by the ratio U/H, where U = sum(LEAF_SPINE_BW) and H = GPU_LEAF_BW;
    GPU->leaf utilisation at the optimum is 0.4375 * U/H. The two goals want opposite ends:

        H = 50 (default, U/H = 1.5): SPINE-BOUND. T* = 0.4267, GPU->leaf at 66%. Both spine
            links are saturated end to end, so the 2:1 split is strictly forced and the ECMP
            signal is at full 1.50x. The host layer has headroom, so a greedy emitter that
            over-drives local traffic is partly absorbed. This is the MULTIPATH configuration.
        H = 25 (U/H = 3.0): HOST-BOUND (14/H = 0.56 > 0.4267). The GPU->leaf link becomes the
            binding cut and both streams on it are pinned with ZERO slack -- cross-pod 14.29 +
            intra-pod 10.71 = 25.00 = exactly line rate -- so any misallocation inflates the
            makespan immediately. This is the RATE configuration. Note the trade: the leaf
            uplink now runs at 76%, not 100%, so the 2:1 spine split is no longer strictly
            forced. ECMP still loses, at 1.14x, because spine1's 25 GB/s is overrun either way
            -- but the split is no longer the unique optimum, and the margin is thin enough
            that this configuration should NOT be the one a multipath claim rests on.

            H is also chosen to keep the capacity ratios grid-friendly: 25:50:25 = 1:2:1, so
            flow splits land on halves and fifths. A value like 30 gives 6:10:5, whose splits
            land on /75 and /150 -- past reconstruct.MAX_DENOM = 64, which the identity snap
            rejects outright.

    Both cuts can only bind at once at the knife edge U/H = 2.286, which hands the LP a
    degenerate vertex -- so sweep H rather than trying to tune a single value that does both.
    Use `two_pod_rail_variant(gpu_leaf_bw=...)` to build the sweep points.

    HIERARCHY. Cells are declared (one per node) but each has 2 GPUs and 2 uplinks, so uplinks
    == GPUs and NO egress relay is forced: identity resolution comes out trivial, exactly as it
    does for RailOptimizedSpineLeaf. This topology is not a phase-3 stress test; that needs
    HeteroTaperedCluster, or a 4-GPU-per-node variant of this one with still only 2 rails.
    """

    NUM_NODES = 8
    GPUS_PER_NODE = 2       # == number of rails
    PODS = 2
    NODES_PER_POD = NUM_NODES // PODS
    NUM_LEAF = PODS * GPUS_PER_NODE     # one leaf per (pod, rail)
    NUM_SPINE = 2

    # Bandwidths in GB/s (divided by chunk_size below, per the codebase convention).
    NVLINK_BW = 900.0                   # intra-node, per GPU<->NVSwitch link
    GPU_LEAF_BW = 50.0                  # H -- the swept knob; see the class docstring
    LEAF_SPINE_BW = (50.0, 25.0)        # per spine; unequal ON PURPOSE

    # Parallel CABLES per leaf<->spine link. The AGGREGATE stays LEAF_SPINE_BW, so the solver is
    # identical for any value here. Two post-solve things read it: the port split
    # (port_split.py), which places each flow on one cable, and the port map (Topology), which
    # numbers the sockets those cables occupy -- so changing it shifts the port numbers in the
    # emitted forwarding table even though it changes no schedule. See TwoPodRailHostBound and
    # TwoPodRailSplitPorts.
    LEAF_SPINE_PORTS = (1, 1)

    # Every switch in the fabric is the same 8-port part, which is what makes the port budget
    # below checkable rather than assumed. A variant that widens the fabric -- more GPUs per
    # node, or more cables per spine -- must raise this or fail at construction, which is the
    # point: you cannot plug nine cables into an eight-port switch.
    SWITCH_RADIX = 8

    NVLINK_ALPHA = 0.35 * pow(10, -6)
    NETWORK_ALPHA = 0.7 * pow(10, -6)

    def __init__(self, topo_input: TopologyParams):
        assert len(self.LEAF_SPINE_BW) == self.NUM_SPINE, \
            "LEAF_SPINE_BW must give one capacity per spine"
        super().__init__(topo_input)
        self.node_per_chassis = self.GPUS_PER_NODE

    # --- index helpers -----------------------------------------------------
    def _gpu(self, node: int, rail: int) -> int:
        return node * self.GPUS_PER_NODE + rail

    def _num_gpus(self) -> int:
        return self.NUM_NODES * self.GPUS_PER_NODE

    def _pod(self, node: int) -> int:
        return node // self.NODES_PER_POD

    def _nvswitch(self, node: int) -> int:
        return self._num_gpus() + node

    def _leaf(self, pod: int, rail: int) -> int:
        return self._num_gpus() + self.NUM_NODES + pod * self.GPUS_PER_NODE + rail

    def _spine(self, s: int) -> int:
        return self._num_gpus() + self.NUM_NODES + self.NUM_LEAF + s

    def construct_topology(self, topo_input: TopologyParams) -> None:
        total_nodes = (self._num_gpus() + self.NUM_NODES + self.NUM_LEAF + self.NUM_SPINE)

        nvlink_cap = self.NVLINK_BW / self.chunk_size
        gpu_leaf_cap = self.GPU_LEAF_BW / self.chunk_size

        edges = {}  # (i, j) -> (capacity, alpha); mirrored symmetrically below

        # Intra-node: both GPUs hang off the node's NVSwitch, and there is deliberately NO direct
        # GPU<->GPU link. Both halves of that matter to `crossbar_solve.is_crossbar`, which claims
        # this cell's interior: it requires exactly one switch with every data node attached, AND
        # no data-to-data link (one the crossbar schedule would leave unused, making the closed
        # form silently pessimistic). Adding the switch while keeping the direct link would still
        # fail to match. See the class docstring for why a switchless pair matches no row at all.
        for n in range(self.NUM_NODES):
            for r in range(self.GPUS_PER_NODE):
                edges[(self._gpu(n, r), self._nvswitch(n))] = (nvlink_cap, self.NVLINK_ALPHA)

        # Rail-optimized within a pod: GPU(n, r) reaches only leaf(pod(n), r). This is the
        # constraint that makes the rail a real routing decision -- to inject on the other rail
        # a chunk must first cross the node's NVSwitch to its sibling.
        for n in range(self.NUM_NODES):
            for r in range(self.GPUS_PER_NODE):
                edges[(self._gpu(n, r), self._leaf(self._pod(n), r))] = (
                    gpu_leaf_cap, self.NETWORK_ALPHA)

        # Leaf-spine full mesh, with per-spine capacity.
        for leaf in range(self.NUM_LEAF):
            pod, rail = divmod(leaf, self.GPUS_PER_NODE)
            for s in range(self.NUM_SPINE):
                edges[(self._leaf(pod, rail), self._spine(s))] = (
                    self.LEAF_SPINE_BW[s] / self.chunk_size, self.NETWORK_ALPHA)

        self.capacity = [[0.0] * total_nodes for _ in range(total_nodes)]
        self.alpha = [[-1.0] * total_nodes for _ in range(total_nodes)]
        for (i, j), (cap, alpha) in edges.items():
            self.capacity[i][j] = cap
            self.capacity[j][i] = cap
            self.alpha[i][j] = alpha
            self.alpha[j][i] = alpha

        # Parallel-port declaration. Structurally invisible: `capacity` above is untouched, so
        # every solve is bit-for-bit identical whatever LEAF_SPINE_PORTS says.
        if any(p > 1 for p in self.LEAF_SPINE_PORTS):
            assert len(self.LEAF_SPINE_PORTS) == self.NUM_SPINE, \
                "LEAF_SPINE_PORTS must give one port count per spine"
            self.ports = [[1] * total_nodes for _ in range(total_nodes)]
            for leaf in range(self.NUM_LEAF):
                pod, rail = divmod(leaf, self.GPUS_PER_NODE)
                for s in range(self.NUM_SPINE):
                    i, j = self._leaf(pod, rail), self._spine(s)
                    self.ports[i][j] = self.ports[j][i] = self.LEAF_SPINE_PORTS[s]

        # No twins, deliberately. The spines differ in capacity (that asymmetry is the point of
        # the topology) and no two leaves share a neighbor set. If LEAF_SPINE_BW is ever made
        # uniform the spines become twins and should be declared here -- otherwise the LP is
        # handed a symmetric tie with no symmetry breaking, which is the degenerate-smear case.
        self.equivalent_node_indices = []
        if len(set(self.LEAF_SPINE_BW)) == 1:
            self.equivalent_node_indices = [
                [self._spine(s) for s in range(self.NUM_SPINE)]
            ]

    def set_switch_indicies(self) -> None:
        # Per-node NVSwitches plus the network fabric: all three are forwarding switches with no
        # collective demand of their own.
        self.switch_indices = (
            [self._nvswitch(n) for n in range(self.NUM_NODES)]
            + [self._leaf(p, r) for p in range(self.PODS) for r in range(self.GPUS_PER_NODE)]
            + [self._spine(s) for s in range(self.NUM_SPINE)]
        )

    def radix(self, node: int) -> int | None:
        """Leaf and spine are 8-port parts; the NVSwitches and GPUs do not claim a width.

        Declaring it is what turns "this variant needs more ports than the switch has" into an
        assertion at construction. The base configuration leaves a leaf at 6 of 8 and both
        spines at 4 of 8; TwoPodRailHostBound's second spine0 cable fills spine0 exactly. See
        that class for the port table.
        """
        if node >= self._leaf(0, 0):
            return self.SWITCH_RADIX
        return None

    def default_programmable_switch_indices(self):
        # Only the network fabric (leaf + spine) takes an external program. The per-node NVSwitch
        # is a self-routing crossbar: the solver freely routes GPU->NVSwitch->GPU through it, but
        # no forwarding entry is ever installed, so it stays out of the emitted table -- same split
        # as RailOptimizedSpineLeaf.
        return ([self._leaf(p, r) for p in range(self.PODS) for r in range(self.GPUS_PER_NODE)]
                + [self._spine(s) for s in range(self.NUM_SPINE)])

    def build_hierarchy(self) -> None:
        # One cell per node: its 2 GPUs plus the NVSwitch behind them collapse to a single coarse
        # node, and the NVSwitch is internal so it is dropped from the coarse graph. The rail
        # constraint is carried by `boundary`: the coarse host<->leaf(pod, r) link is owned by
        # gpu(n, r). Coarse graph = 8 hosts + 4 leaves + 2 spines = 14 nodes, unchanged by the
        # NVSwitch. The cell's interior is a one-switch crossbar, which is the shape
        # `crossbar_solve.is_crossbar` claims -- see the class docstring.
        self.cells = []
        for n in range(self.NUM_NODES):
            gpus = [self._gpu(n, r) for r in range(self.GPUS_PER_NODE)]
            nvswitch = self._nvswitch(n)
            boundary = {
                self._leaf(self._pod(n), r): [self._gpu(n, r)]
                for r in range(self.GPUS_PER_NODE)
            }
            self.cells.append(Cell(
                members=gpus + [nvswitch],
                gpus=gpus,
                internal_switches=[nvswitch],
                boundary=boundary,
            ))


class TwoPodRailHostBound(TwoPodRail):
    """
    The RATE configuration: identical graph, GPU_LEAF_BW lowered to 25 GB/s.

    This moves the binding cut from the spine layer to the GPU->leaf link (14/H = 0.56 >
    0.4267), which pins BOTH streams on that link with zero slack -- cross-pod 14.29 +
    intra-pod 10.71 = 25.00 = exactly line rate -- so a rate-oblivious emitter cannot hide in
    headroom. It exists as a named class, rather than only as a two_pod_rail_variant() call, so
    it is reachable by name from a JSON input file and from ncclize's --topology.

    CONFIRMED on the coarse solve (job 1498195, LEXICOGRAPHIC objective): every one of the 224
    host-uplink link-epochs runs at exactly its capacity, and the 7-epoch makespan is exactly
    the bound (8 hosts x 7 chunks = 56 sends over 8 chunks/epoch of aggregate host egress).
    Zero host relay, and a spine cut carrying exactly its 32-chunk minimum.

    That total saturation is also this configuration's one hazard: 224 tight equality
    constraints admit an enormous family of equally-optimal compositions, and LEXICOGRAPHIC's
    tier 1 sums FRACTIONAL demand satisfied, so it cannot tell them apart. Left to the barrier
    solver's analytic center that degenerate face comes back as smeared volumes (83 distinct
    values, most off the MAX_DENOM = 64 identity grid) and the identity snap rejects them. Ask
    for a BASIC solution -- GurobiParams method=1 (dual simplex) or crossover=1 -- so the solve
    lands on a vertex, whose minimal support sits on the natural grid. The sample input
    two_pod_rail_hostbound_allgather.json sets both.

    SPINE0 IS TWO CABLES HERE. Its 50 GB/s leaf<->spine0 links are two 25 GB/s cables
    (LEAF_SPINE_PORTS below), which makes the whole fabric uniformly 25 GB/s per port: each leaf
    has 4 downlinks, 2 up-ports to spine0 and 1 to spine1, for the same 75 GB/s of uplink. With
    SWITCH_RADIX = 8 that is a fully specified port budget, and every switch is the same part:

        leaf(pod p, rail r)   port 0-3   the 4 GPU downlinks, one per node of pod p on rail r
                              port 4-5   the 2 cables to spine 0
                              port 6     the 1 cable to spine 1        (7 of 8 used)
        spine 0               port 0-7   4 leaves x 2 cables           (8 of 8 -- exactly full)
        spine 1               port 0-3   4 leaves x 1 cable            (4 of 8 used)

    This is the small case for the whole port model, which is why it is worth writing out: a
    leaf that is a permanent fan-in/fan-out mismatch -- 4 single-cable downlinks against a
    2-cable and a 1-cable uplink -- is exactly the shape an in-port -> out-port identity mapping
    cannot serve (port_split_design.md section 6), and a spine filled to its last port is the
    case where the radix assertion has something to say.

    The declaration is invisible to the SOLVER -- `capacity[leaf][spine0]` is still 50, so every
    solve, every schedule and every number above is unchanged. Two post-solve passes read it:
    port_split.py places each flow on one cable, and the port map numbers the sockets. Carrying
    it on the named class rather than on a separate subclass is deliberate: this name is what
    scheduler.py, ncclize's --topology and the sample inputs resolve, so the whole workflow
    exercises the split end to end instead of a test-only alias doing it.

    For an unsplit A/B on this same graph, build one:

        two_pod_rail_variant("TwoPodRailHostBoundOnePort", gpu_leaf_bw=25.0,
                             leaf_spine_ports=(1, 1))

    Note what does and does not match across that A/B. The XML is byte-identical and so is every
    route, because the schedule never depended on the cable count. The forwarding table's PORT
    NUMBERS do differ, and must: the unsplit leaf plugs one cable into port 4 and reaches spine1
    on port 5, while this one needs ports 4-5 for spine0 and pushes spine1 to port 6. A port
    number is a fact about how the switch is cabled, so cabling it differently moves it.

    See TwoPodRail for the full derivation and for what this configuration gives up (the 2:1
    spine split stops being the unique optimum).
    """
    GPU_LEAF_BW = 25.0
    LEAF_SPINE_PORTS = (2, 1)


def two_pod_rail_variant(name: str = "TwoPodRailVariant",
                         gpu_leaf_bw: float = None,
                         leaf_spine_bw: Sequence[float] = None,
                         nvlink_bw: float = None,
                         leaf_spine_ports: Sequence[int] = None) -> Type[TwoPodRail]:
    """
    Build a TwoPodRail subclass with overridden bandwidths, for sweeping the design point.

    Only the bandwidths vary -- the graph is identical -- so a sweep is a list of subclasses:

        from teccl.topologies.two_pod_rail import TwoPodRail, two_pod_rail_variant

        multipath_cfg = TwoPodRail                                   # H = 50, spine-bound
        rate_cfg      = TwoPodRailHostBound                          # H = 25, host-bound
        other_cfg     = two_pod_rail_variant("TwoPodRail_h40", gpu_leaf_bw=40.0)

    `leaf_spine_ports` is the one knob the SOLVER does not see: it splits a leaf<->spine link
    into that many equal cables for the post-solve port pass, leaving `capacity` alone. It is
    not free of consequences downstream -- it also decides how many sockets those links occupy,
    so it moves the port numbers in the emitted forwarding table and can overrun SWITCH_RADIX.
    Use it to build an unsplit twin of a split configuration for an A/B:

        unsplit_hostbound = two_pod_rail_variant("TwoPodRailHostBoundOnePort",
                                                 gpu_leaf_bw=25.0, leaf_spine_ports=(1, 1))

    See the TwoPodRail docstring for which cut binds at which H and why that decides what the
    configuration actually tests.
    """
    attrs = {}
    if gpu_leaf_bw is not None:
        attrs["GPU_LEAF_BW"] = float(gpu_leaf_bw)
    if leaf_spine_bw is not None:
        attrs["LEAF_SPINE_BW"] = tuple(float(b) for b in leaf_spine_bw)
    if nvlink_bw is not None:
        attrs["NVLINK_BW"] = float(nvlink_bw)
    if leaf_spine_ports is not None:
        attrs["LEAF_SPINE_PORTS"] = tuple(int(p) for p in leaf_spine_ports)
    return type(name, (TwoPodRail,), attrs)


class TwoPodRailSplitPorts(TwoPodRail):
    """`TwoPodRail` (the SPINE-BOUND config, H = 50) with spine0 as two 25 GB/s ports.

    The sibling of `TwoPodRailHostBound` (which splits spine0 too), and the one that exercises
    the port split's packing CHOICES rather than just its arithmetic. Host-bound runs every leaf
    uplink at exactly line rate, so per-port balance is forced by capacity and every fit rule
    and bucket ordering produces the same assignment. Here the binding cut is the spine layer
    and the GPU->leaf links carry slack, so the buckets arriving at a leaf are not all pinned
    and the placement rule has something to decide -- one bucket outgrows a port and is broken
    up, giving 5 combos per leaf->spine0 link against host-bound's 4.

    Kept a separate class rather than folded into `TwoPodRail` so that `TwoPodRail` stays the
    ports-undeclared reference every byte-identical check compares against.
    """
    LEAF_SPINE_PORTS = (2, 1)
