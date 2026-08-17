from typing import Sequence, Tuple, Type

from teccl.hierarchy.cell import Cell
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class TwoPodRail(Topology):
    """
    16-GPU / 8-node / 2-pod rail-optimized topology, CONTRIVED so that the optimal schedule is
    forced to (1) split a flow across multiple paths and (2) pace some flows below link rate.

    It is small enough (22 nodes) that the FLAT solve is tractable, so it doubles as the
    ground truth the hierarchical stitch can be validated against -- which neither
    RailOptimizedSpineLeaf (too big) nor HeteroTaperedCluster (no flat reference computed)
    provides.

        Pod A = nodes 0-3, Pod B = nodes 4-7. Each node holds 2 GPUs joined by a direct NVLink.
        Leaf(pod p, rail r) serves the rail-r GPU of every node in pod p -> 4 leaves.
        Every leaf reaches BOTH spines. Cross-pod traffic must therefore cross a spine; the
        two spines are deliberately UNEQUAL (50 vs 25 GB/s), which is what makes the split
        forced and non-dyadic instead of a symmetric tie.

    Node indexing (22 nodes):
        GPU(node n, rail r) = n * 2 + r        n in [0,8), r in [0,2)   -> [0,16)
        Leaf(pod p, rail r) = 16 + p * 2 + r   p in [0,2), r in [0,2)   -> [16,20)
        Spine(s)            = 20 + s           s in [0,2)               -> [20,22)

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
        H = 30 (U/H = 2.5): HOST-BOUND (14/H = 0.4667 > 0.4267). The GPU->leaf link becomes the
            binding cut and both streams on it are pinned with ZERO slack -- cross-pod 17.14 +
            intra-pod 12.86 = 30.00 = exactly line rate -- so any misallocation inflates the
            makespan immediately. This is the RATE configuration. Note the trade: the leaf
            uplink now runs at 91%, not 100%, so the 2:1 spine split is no longer strictly
            forced (it has ~9% slack). ECMP still loses, at 1.37x, because spine1's 25 GB/s is
            overrun either way -- but the split is no longer the unique optimum.

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
    NVLINK_BW = 900.0                   # intra-node, direct GPU<->GPU
    GPU_LEAF_BW = 50.0                  # H -- the swept knob; see the class docstring
    LEAF_SPINE_BW = (50.0, 25.0)        # per spine; unequal ON PURPOSE

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

    def _leaf(self, pod: int, rail: int) -> int:
        return self._num_gpus() + pod * self.GPUS_PER_NODE + rail

    def _spine(self, s: int) -> int:
        return self._num_gpus() + self.NUM_LEAF + s

    def construct_topology(self, topo_input: TopologyParams) -> None:
        total_nodes = self._num_gpus() + self.NUM_LEAF + self.NUM_SPINE

        nvlink_cap = self.NVLINK_BW / self.chunk_size
        gpu_leaf_cap = self.GPU_LEAF_BW / self.chunk_size

        edges = {}  # (i, j) -> (capacity, alpha); mirrored symmetrically below

        # Intra-node: the two GPUs of a node are joined directly (no NVSwitch -- with 2 GPUs a
        # crossbar would add a hop without adding a routing choice).
        for n in range(self.NUM_NODES):
            edges[(self._gpu(n, 0), self._gpu(n, 1))] = (nvlink_cap, self.NVLINK_ALPHA)

        # Rail-optimized within a pod: GPU(n, r) reaches only leaf(pod(n), r). This is the
        # constraint that makes the rail a real routing decision -- to inject on the other rail
        # a chunk must first cross the NVLink.
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
        # Leaves and spines only. There is no intra-node switch to exclude, so the base class's
        # default_programmable_switch_indices (every switch) is already correct here.
        self.switch_indices = (
            [self._leaf(p, r) for p in range(self.PODS) for r in range(self.GPUS_PER_NODE)]
            + [self._spine(s) for s in range(self.NUM_SPINE)]
        )

    def build_hierarchy(self) -> None:
        # One cell per node: its 2 GPUs collapse to a single coarse node. There is no internal
        # switch, so the cell's internal fabric is the bare NVLink. The rail constraint is
        # carried by `boundary`: the coarse host<->leaf(pod, r) link is owned by gpu(n, r).
        # Coarse graph = 8 hosts + 4 leaves + 2 spines = 14 nodes.
        self.cells = []
        for n in range(self.NUM_NODES):
            gpus = [self._gpu(n, r) for r in range(self.GPUS_PER_NODE)]
            boundary = {
                self._leaf(self._pod(n), r): [self._gpu(n, r)]
                for r in range(self.GPUS_PER_NODE)
            }
            self.cells.append(Cell(
                members=gpus,
                gpus=gpus,
                internal_switches=[],
                boundary=boundary,
            ))


class TwoPodRailHostBound(TwoPodRail):
    """
    The RATE configuration: identical graph, GPU_LEAF_BW lowered to 30 GB/s.

    This moves the binding cut from the spine layer to the GPU->leaf link (14/H = 0.4667 >
    0.4267), which pins BOTH streams on that link with zero slack -- cross-pod 17.14 +
    intra-pod 12.86 = 30.00 = exactly line rate -- so a rate-oblivious emitter cannot hide in
    headroom. It exists as a named class, rather than only as a two_pod_rail_variant() call, so
    it is reachable by name from a JSON input file and from ncclize's --topology.

    See TwoPodRail for the full derivation and for what this configuration gives up (the 2:1
    spine split stops being the unique optimum).
    """
    GPU_LEAF_BW = 30.0


def two_pod_rail_variant(name: str = "TwoPodRailVariant",
                         gpu_leaf_bw: float = None,
                         leaf_spine_bw: Sequence[float] = None,
                         nvlink_bw: float = None) -> Type[TwoPodRail]:
    """
    Build a TwoPodRail subclass with overridden bandwidths, for sweeping the design point.

    Only the bandwidths vary -- the graph is identical -- so a sweep is a list of subclasses:

        from teccl.topologies.two_pod_rail import TwoPodRail, two_pod_rail_variant

        multipath_cfg = TwoPodRail                                   # H = 50, spine-bound
        rate_cfg      = two_pod_rail_variant("TwoPodRail_h30", gpu_leaf_bw=30.0)  # host-bound

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
    return type(name, (TwoPodRail,), attrs)
