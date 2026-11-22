"""Build a TE-CCL Topology from a `.topo` file written in the topology DSL.

The DSL frontend (teccl/topologies/topology-dsl-frontend, a submodule shared with the ns-3
simulator) parses a `.topo` file, evaluates its modules/loops/conditionals, and hands back a FLAT
IR: a node list per type, an ordered list of undirected links each carrying a bandwidth and a
latency, and the module structure they came from. That IR is backend-neutral -- ns-3 turns it into
C++ (its own `ns3codegen` adds one instruction on top), and this module turns it into the matrices
a Topology is made of. Nothing is emitted here; everything stays Python objects.

NODE NUMBERING is class-major, declaration-order within a class: every `gpu` first, then every
`nvswitch`, then every `switch`. That is not an arbitrary convention -- it is the numbering the
hand-written classes already use, so a `.topo` port of one of them lands on the SAME indices
(verified against TwoPodRailHostBound, HeteroTaperedCluster, RailOptimizedSpineLeaf and
FatTreePod in dsl_topology_test.py). Anything that reads a node index -- a schedule, a cell, an
emitted forwarding table -- is therefore comparable between the two.
"""

import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from teccl.hierarchy.cell import Cell
from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "topology-dsl-frontend")

# Bandwidth units -> GB/s, and latency units -> seconds. Decimal (1e3), matching ns-3's
# DataRate: 400 Gbps is 400e9 bits/s = 50 GB/s, which is exactly what the Python classes
# these files port write as `50 / chunk_size`.
_BANDWIDTH_TO_GBPS = {
    "Gbps": 1e9 / 8 / 1e9, "Mbps": 1e6 / 8 / 1e9, "Kbps": 1e3 / 8 / 1e9,
    "GBps": 1.0, "MBps": 1e-3, "KBps": 1e-6,
}
_LATENCY_TO_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3}

# The DSL's node types, in the order their nodes claim global indices.
_TYPE_ORDER = ("gpu", "nvswitch", "switch")


def _load_frontend():
    """Import the DSL frontend, which is a submodule and so not an importable package.

    `codegen` does `from transformer import *`, so `transformer` has to be reachable under that
    bare name -- hence sys.path rather than importlib with a qualified name. This mirrors what the
    ns-3 consumer does (its topology/main.py), so the two stay in step.
    """
    if not os.path.isfile(os.path.join(FRONTEND_DIR, "grammar.lark")):
        raise FileNotFoundError(
            f"{FRONTEND_DIR}/grammar.lark is missing -- the topology-dsl-frontend submodule is "
            f"not checked out. Run: git submodule update --init "
            f"teccl/topologies/topology-dsl-frontend")
    if FRONTEND_DIR not in sys.path:
        sys.path.insert(0, FRONTEND_DIR)
    try:
        from lark import Lark
    except ImportError as e:
        raise ImportError(
            "the topology DSL needs `lark` (pip install lark) -- it is what parses a .topo "
            "file") from e
    from transformer import TopoTransformer
    # The NEUTRAL IR, not ns3codegen: TE-CCL emits nothing, so ns-3's fabric-build instruction
    # would be one more thing to skip over rather than anything to read.
    from codegen import TopologyIR, InstallLink
    return Lark, TopoTransformer, TopologyIR, InstallLink


def build_ir(topo_file: str):
    """Parse `topo_file` and return the frontend's flattened IR (a TopologyIR).

    Everything the DSL can express -- modules, submodule instantiation, loops, conditionals,
    arithmetic over parameters -- has been evaluated by the time this returns. What is left is
    `ir.gpus / .switches / .nvswitches` (name -> per-type index, in declaration order),
    `ir.link_classes` ((latency, bandwidth, mtu, type) -> id), `ir.insns` (the InstallLink list,
    in source order), and the structure the flattening would otherwise erase: `ir.nodes`,
    `ir.instances`, `ir.symmetry_groups`.
    """
    Lark, TopoTransformer, TopologyIR, _ = _load_frontend()
    with open(os.path.join(FRONTEND_DIR, "grammar.lark")) as f:
        parser = Lark(f.read(), parser="lalr")
    with open(topo_file) as f:
        tree = parser.parse(f.read())
    modules = TopoTransformer().transform(tree)
    if "main" not in modules:
        raise ValueError(f"{topo_file} declares no `main` module, so there is nothing to build")
    codegen = TopologyIR(modules)
    codegen.Generate()
    return codegen


def _to_gigabytes_per_second(value, what: str) -> float:
    if not isinstance(value, tuple):
        raise ValueError(
            f"{what} = {value!r} has no unit. Bandwidth must be written with one, e.g. "
            f"400[Gbps] or 50[GBps] -- a bare number would leave bits-vs-bytes to guesswork")
    number, unit = value
    if unit not in _BANDWIDTH_TO_GBPS:
        raise ValueError(f"{what} has unit {unit!r}, which is not a bandwidth unit "
                         f"({', '.join(_BANDWIDTH_TO_GBPS)})")
    return float(number) * _BANDWIDTH_TO_GBPS[unit]


def _to_seconds(value, what: str) -> float:
    if not isinstance(value, tuple):
        raise ValueError(
            f"{what} = {value!r} has no unit. Latency must be written with one, e.g. 700[ns]")
    number, unit = value
    if unit not in _LATENCY_TO_SECONDS:
        raise ValueError(f"{what} has unit {unit!r}, which is not a latency unit "
                         f"({', '.join(_LATENCY_TO_SECONDS)})")
    return float(number) * _LATENCY_TO_SECONDS[unit]


class DslTopology(Topology):
    """A Topology whose graph comes from a `.topo` file rather than from Python.

    Selected by setting TopologyParams.topo_file; `name` stays a free-form label (it is what
    output filenames are built from). See the module docstring for the node numbering.
    """

    def __init__(self, topo_input: TopologyParams):
        if not topo_input.topo_file:
            raise ValueError("DslTopology needs TopologyParams.topo_file (a path to a .topo file)")
        self.topo_file = topo_input.topo_file
        # Built before super().__init__, because the base class calls construct_topology from
        # inside its own __init__ and every field below is already needed by then.
        self._ir = build_ir(self.topo_file)
        self._index_nodes()
        super().__init__(topo_input)
        # GPUs per cell where the file declares cells (every cell must agree, or the number is
        # a fiction); otherwise every GPU. Only the legacy `chassis * node_per_chassis` paths
        # read it, and they already assume uniformity.
        cell_sizes = {len(c.gpus) for c in self.cells}
        self.node_per_chassis = (cell_sizes.pop() if len(cell_sizes) == 1
                                 else len(self.gpu_indices))
        # Passive nodes may come from the file (`node gpu7 type=gpu passive=1;`) as well as from
        # TopologyParams; the union is what the collective sees. Merged after super().__init__,
        # which is where passive_indices is first set.
        declared_passive = [i for i in self.gpu_indices
                            if self._node_attrs(i).get("passive")]
        if declared_passive:
            self.passive_indices = sorted(set(self.passive_indices) | set(declared_passive))
            assert not set(self.passive_indices) & set(self.switch_indices), \
                "A node cannot be both a switch and a passive forwarding node"

    # --- IR -> nodes --------------------------------------------------------------------------
    def _index_nodes(self) -> None:
        """Assign global node indices, class-major then declaration order.

        The frontend already keeps one dict per type in declaration order, so this is a
        concatenation -- but it is the concatenation ORDER that makes a ported .topo land on the
        same indices as the class it ports, so it is stated here rather than left implicit.
        """
        per_type = {"gpu": self._ir.gpus, "nvswitch": self._ir.nvswitches,
                    "switch": self._ir.switches}
        collisions = sorted(
            set(self._ir.gpus) & set(self._ir.switches)
            | set(self._ir.gpus) & set(self._ir.nvswitches)
            | set(self._ir.switches) & set(self._ir.nvswitches))
        assert not collisions, \
            f"{self.topo_file}: {collisions} are declared with more than one node type"

        by_name = {record.name: record for record in self._ir.nodes}
        self.node_names: List[str] = []
        self.node_types: List[str] = []
        self.node_index: Dict[str, int] = {}
        self.node_records: List[object] = []
        for node_type in _TYPE_ORDER:
            # dict order is declaration order; sort by the frontend's own per-class counter
            # rather than trusting that, so the numbering is stated by the IR, not by dict luck.
            for name, _ in sorted(per_type[node_type].items(), key=lambda kv: kv[1]):
                self.node_index[name] = len(self.node_names)
                self.node_names.append(name)
                self.node_types.append(node_type)
                self.node_records.append(by_name[name])
        self.gpu_indices: List[int] = [i for i, t in enumerate(self.node_types) if t == "gpu"]
        self.nvswitch_indices: List[int] = [
            i for i, t in enumerate(self.node_types) if t == "nvswitch"]
        self.network_switch_indices: List[int] = [
            i for i, t in enumerate(self.node_types) if t == "switch"]

    def _links(self) -> List[Tuple[int, int, float, float]]:
        """Every installed link as (i, j, bandwidth GB/s, latency s), in DECLARATION order.

        Declaration order is load-bearing twice over: it is the cable numbering within a
        multi-cable edge, and it is the port numbering on each endpoint (see port_order).
        """
        _, _, _, install_link = _load_frontend()
        by_id = {cid: attrs for attrs, cid in self._ir.link_classes.items()}
        links = []
        for insn in self._ir.insns:
            if not isinstance(insn, install_link):
                continue
            for end in (insn.src, insn.dst):
                if end not in self.node_index:
                    raise ValueError(
                        f"{self.topo_file}: link references node {end!r}, which is not declared")
            i, j = self.node_index[insn.src], self.node_index[insn.dst]
            if i == j:
                raise ValueError(f"{self.topo_file}: {insn.src} is linked to itself")
            latency, bandwidth, _mtu, _type = by_id[insn.link_class]
            links.append((i, j, _to_gigabytes_per_second(bandwidth, f"link {insn.src}-{insn.dst}"),
                          _to_seconds(latency, f"link {insn.src}-{insn.dst}")))
        if not links:
            raise ValueError(f"{self.topo_file}: declares no links")
        return links

    # --- Topology contract --------------------------------------------------------------------
    def construct_topology(self, topo_input: TopologyParams) -> None:
        n = len(self.node_names)
        cables: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
        # (neighbor, cable) per node, in the order the link statements plug them in
        order: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        for i, j, bandwidth, latency in self._links():
            key = (min(i, j), max(i, j))
            cable = len(cables[key])
            cables[key].append((bandwidth, latency))
            order[i].append((j, cable))
            order[j].append((i, cable))
        self._declaration_port_order = dict(order)

        self.capacity = [[0.0] * n for _ in range(n)]
        self.alpha = [[-1.0] * n for _ in range(n)]
        split = False
        for (i, j), parallel in cables.items():
            # A multi-cable edge is modeled as ONE link of the AGGREGATE capacity (the solver
            # never sees the split; see Topology.ports). port_split then divides that aggregate
            # EVENLY, so cables of unequal width would be silently misrepresented -- refuse them
            # here, where the file that wrote them can still be named.
            widths = {c for c, _ in parallel}
            latencies = {a for _, a in parallel}
            assert len(widths) == 1 and len(latencies) == 1, (
                f"{self.topo_file}: the {len(parallel)} parallel links "
                f"{self.node_names[i]}<->{self.node_names[j]} differ in bandwidth/latency "
                f"({sorted(widths)} GB/s, {sorted(latencies)} s). Parallel cables of one edge "
                f"must be identical; write them as a single link of the summed bandwidth "
                f"instead")
            self.capacity[i][j] = self.capacity[j][i] = (
                sum(c for c, _ in parallel) / self.chunk_size)
            self.alpha[i][j] = self.alpha[j][i] = parallel[0][1]
            split = split or len(parallel) > 1

        if split:
            self.ports = [[1] * n for _ in range(n)]
            for (i, j), parallel in cables.items():
                self.ports[i][j] = self.ports[j][i] = len(parallel)

        self.equivalent_node_indices = self._symmetry_groups()

    def set_switch_indicies(self) -> None:
        # Both DSL switch types forward without bearing demand of their own.
        self.switch_indices = sorted(self.nvswitch_indices + self.network_switch_indices)

    def default_programmable_switch_indices(self) -> List[int]:
        # The DSL's `switch` / `nvswitch` split is exactly TE-CCL's programmable / self-routing
        # split: the solver routes through an NVSwitch freely, but no forwarding entry is ever
        # installed on one, so it stays out of the emitted table.
        return list(self.network_switch_indices)

    def port_order(self, node: int) -> List[Tuple[int, int]]:
        """The node's connections in the order the `link` statements plug them in.

        This is the one ordering fact the DSL carries that the base class's default (neighbors
        ascending) has to guess at, and it is the same order ns-3 assigns device indices in --
        so a port number here names the same socket as in the simulator. For a .topo written as
        a port of a Python class the two orders coincide, which the parity test asserts.
        """
        return list(self._declaration_port_order[node])

    def _node_attrs(self, node: int) -> Dict[str, object]:
        """Everything the `node` statement declared, `type` included."""
        return self.node_records[node].attrs

    def radix(self, node: int) -> Optional[int]:
        """The node's port count if the file states one (`node leaf0 type=switch radix=8;`).

        Stating it is what turns "this fabric needs more ports than the part has" into an
        assertion at construction instead of an emitted table with ports no switch owns. None --
        the default, and what a node with no `radix` attribute gets -- means the node is exactly
        as wide as its cables, which is honest rather than invented.
        """
        width = self._node_attrs(node).get("radix")
        if width is None:
            return None
        assert isinstance(width, int) and width > 0, \
            f"{self.node_names[node]}: radix = {width!r} must be a positive integer"
        return width

    # --- hierarchy ----------------------------------------------------------------------------
    def build_hierarchy(self) -> None:
        """Turn the `cell`-marked module instances into Cells, nesting included.

        A cell in the DSL is a module instance the file marked (`use server() as srv0 cell;`) --
        the one construct that adds a level of structure, since loops and conditionals share
        their enclosing scope. Membership is therefore lexical and exact: a node belongs to the
        innermost marked instance that encloses it, and a marked instance inside a marked
        instance is a subcell. Nothing is inferred from names or from the graph.
        """
        instances = getattr(self._ir, "instances", {})
        marked = [rec for rec in instances.values() if rec.is_cell]
        if not marked:
            self.cells = []
            return
        roots = [rec for rec in marked
                 if not any(other is not rec and rec.scope[:len(other.scope)] == other.scope
                            for other in marked)]
        self.cells = [self._build_cell(rec, marked, None) for rec in roots]

    def _nearest_marked_descendants(self, scope: Tuple[str, ...], marked) -> List:
        """The marked instances strictly inside `scope` with no marked instance between.

        Walks by scope, not by parent links, so an UNMARKED instance in between (a `rack` that
        holds two marked hosts, say) does not hide the cells below it.
        """
        inner = [rec for rec in marked
                 if len(rec.scope) > len(scope) and rec.scope[:len(scope)] == scope]
        return [rec for rec in inner
                if not any(other is not rec and len(other.scope) > len(scope)
                           and rec.scope[:len(other.scope)] == other.scope
                           for other in inner)]

    def _build_cell(self, record, marked, enclosing: Optional[set]) -> Cell:
        scope = record.scope
        label = ".".join(scope)
        # Everything lexically inside the instance, including nodes owned by nested cells --
        # Cell.members is the full collapse set, and `subcells` re-declares the nesting.
        own = sorted(
            n for n in range(len(self.node_names))
            if self.node_records[n].scope[:len(scope)] == scope)
        gpus = [n for n in own if self.node_types[n] == "gpu"]
        assert gpus, (f"cell {label} ({record.module}) contains no gpu, so it has no data to "
                      f"bear as a coarse node")
        inside = set(own)
        switches = [n for n in own if self.node_types[n] != "gpu"]
        internal, external_switches = [], []
        for switch in switches:
            (external_switches if any(v not in inside for v in self.neighbors(switch))
             else internal).append(switch)
        # A coarse link has to be owned by a GPU of the cell (abstract() reads boundary only for
        # cell.gpus), so a switch reaching out of the cell would produce a coarse edge no GPU
        # owns. Say so here, naming the switch, rather than letting the coarse graph come out
        # quietly wrong.
        assert not external_switches, (
            f"cell {label}: switch(es) {[self.node_names[s] for s in external_switches]} link "
            f"outside the cell. A cell's external links must be owned by its GPUs -- either "
            f"widen the cell to include the switch's peers, or do not mark this instance")
        # Boundary = the cell's neighbors AT ITS OWN LEVEL. For a top-level cell that is every
        # neighbor outside it; for a subcell it is scoped to its parent, because the level that
        # reads a subcell's boundary is the parent's induced problem and nothing outside the
        # parent exists there. (induce() filters the same way, so an unscoped entry would be
        # dropped rather than misread -- but writing one would still claim a link this level
        # does not own.)
        boundary: Dict[int, List[int]] = defaultdict(list)
        for gpu in gpus:
            for neighbor in self.neighbors(gpu):
                if neighbor not in inside and (enclosing is None or neighbor in enclosing):
                    boundary[neighbor].append(gpu)
        subcells = [self._build_cell(inner, marked, inside)
                    for inner in self._nearest_marked_descendants(scope, marked)]
        return Cell(members=own, gpus=gpus, internal_switches=internal,
                    boundary={k: v for k, v in sorted(boundary.items())}, subcells=subcells)

    def _symmetry_groups(self) -> List[List[int]]:
        """What the file DECLARES symmetric, plus the switch twins the graph shows.

        The two sources answer different questions. Inference can only find nodes that already
        look alike, and it is restricted to switches because a GPU swap permutes the source index
        (see _infer_switch_twins). A `symmetric` statement is the file stating an intent the
        graph cannot -- including a GPU group, which the hand-written classes also declare
        (FatTreePod's [[0,1],[2,3]]) and which the MILP's total-flow symmetry consumes. A
        declared group is still CHECKED against the graph, so a typo is caught here rather than
        silently constraining a solve into infeasibility.
        """
        declared = []
        for names in getattr(self._ir, "symmetry_groups", []):
            group = sorted(self.node_index[name] for name in names)
            self._check_declared_symmetry(group, names)
            declared.append(group)
        groups = {tuple(g) for g in declared} | {tuple(g) for g in self._infer_switch_twins()}
        return sorted((list(g) for g in groups), key=lambda g: g[0])

    def _check_declared_symmetry(self, group: List[int], names: List[str]) -> None:
        """A declared group must be interchangeable in the graph as well as in intent.

        The test is the same neighborhood fingerprint the inference uses, with the GROUP MATES
        masked out of each key: twins that are linked to each other (a pair of spines with a
        mesh link between them) have deliberately different raw neighborhoods, and masking is
        what lets the check see past that instead of rejecting a correct declaration.
        """
        n = len(self.node_names)
        mates = set(group)

        def masked_key(i: int):
            return tuple(sorted((j, self.capacity[i][j]) for j in range(n)
                                if self.capacity[i][j] > 0 and j not in mates))

        keys = {masked_key(i) for i in group}
        assert len(keys) == 1, (
            f"{self.topo_file}: `symmetric {' '.join(names)}` names nodes that are not "
            f"interchangeable -- their neighborhoods differ:\n" + "\n".join(
                f"  {self.node_names[i]}: " + ", ".join(
                    f"{self.node_names[j]}@{c}" for j, c in masked_key(i))
                for i in group))
        types = {self.node_types[i] for i in group}
        assert len(types) == 1, (
            f"{self.topo_file}: `symmetric {' '.join(names)}` mixes node types {sorted(types)}")

    def _infer_switch_twins(self) -> List[List[int]]:
        """Groups of interchangeable switches, by identical-neighborhood fingerprint.

        Same detector and same restriction as teccl.hierarchy.abstract: nodes whose sorted
        weighted neighborhoods match are twins because swapping them is a graph automorphism,
        and only SWITCHES may be reported -- a GPU swap permutes the source index, which is a
        demand-gated symmetry and not what equivalent_node_indices means. Detects only
        identical-neighborhood twins, and does not mask group-mates out of the key, so twins
        that are adjacent to each other are missed rather than mis-grouped.
        """
        n = len(self.node_names)
        switches = set(self.nvswitch_indices) | set(self.network_switch_indices)

        def key(i: int) -> Tuple:
            return tuple(sorted((j, self.capacity[i][j])
                                for j in range(n) if self.capacity[i][j] > 0))

        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for i in sorted(switches):
            groups[key(i)].append(i)
        return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: g[0])
