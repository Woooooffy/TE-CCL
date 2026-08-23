# Topology

Add your own topologies in this module following the same format.
## Topologies from a `.topo` file

A topology can come from a file instead of a Python class. Point `TopologyParams.topo_file` at a
`.topo` written in the topology DSL and `DslTopology` (`dsl_topology.py`) builds the whole thing
-- capacity, alpha, parallel ports, the port map, switch/programmable sets, symmetry groups and
the hierarchy cells -- from the DSL frontend's flattened IR (`topology-dsl-frontend/`, a submodule
shared with the ns-3 simulator, so one file describes both the simulated and the solved fabric).
`name` stays a free-form label: it is what the output filenames are built from.

```json
"TopologyParams": {
    "name": "MyCluster",
    "topo_file": "teccl/topologies/topology-dsl-frontend/examples/nested_cluster.topo",
    "chunk_size": 1
}
```

Node numbering is class-major, declaration order within a class: every `gpu`, then every
`nvswitch`, then every `switch`. That is the numbering the hand-written classes already use, so a
`.topo` port of one of them lands on the same indices -- which `dsl_topology_test.py` asserts
element-wise against `TwoPodRailHostBound`, `HeteroTaperedCluster`, `RailOptimizedSpineLeaf`,
`FatTreePod` and `NestedCluster` (the last one three levels deep, cells and subcells included).

Two DSL constructs exist for the solver rather than the simulator, and ns-3 ignores both:

  * `use server() as srv0 cell;` marks an instance a HIERARCHY CELL -- the group of fine nodes a
    hierarchical solve collapses into one coarse node. Markers nest, so a marked instance inside a
    marked instance is a subcell.
  * `symmetric spine0 spine1;` declares nodes interchangeable. Declared groups are checked against
    the graph, and are unioned with the switch twins `DslTopology` infers on its own (the same
    identical-neighborhood detector, and the same switches-only restriction, as
    `teccl/hierarchy/abstract.py`).

Per-node attributes are read where TE-CCL has a use for them: `radix=8` bounds a switch's port
count (an over-subscribed fabric then fails at construction rather than emitting ports no switch
has), and `passive=1` marks a GPU that forwards but bears no demand.

Requires `lark`. Run the tests with `conda run -n teccl python teccl/topologies/dsl_topology_test.py`.
