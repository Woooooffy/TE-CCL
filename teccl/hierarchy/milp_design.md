# Hierarchical solver on the MILP formulation

Plan for letting a hierarchy level be solved by the chunk-level MILP
(`teccl/solvers/allgather.py`, `AllGatherFormulation`, which serves allgather and
alltoall) instead of the LP. Branch `hierarchical-milp`, cut from
`cell-relay-transit` @ af74489.

> **Status (2026-10-05): M0–M6 BUILT** on this branch. M7's local half is done; its remote
> half (anything over the restricted Gurobi licence) and M8 are not. The LP path is byte-identical:
> a snapshot of the hetero AG/A2A replays, the bridged transit resolution and both NestedCluster
> solves was diffed after every refactor. The whole pre-existing suite is green. New tests:
> `examples/hierarchy_milp_forward_dwell_test.py` (M5, real Gurobi on 5 coarse nodes) and
> `examples/hierarchy_milp_test.py` (M1–M6 Gurobi-free, plus end to end on the bridged pair).
> End to end, through ncclize `check_implements`, at Q = 1:
> * BridgedIslands allgather (`holder`) and alltoall (`cell`), co-located;
> * BridgedIslandsSplit, the same two, with RELAYED transit (the split bridge used to be
>   unimplementable);
> * HeteroTapered allgather (`cell`): 0.2044 s, against 0.2022 s LP-hierarchical and 0.22 s flat
>   MILP.
>
> Everything else on hetero exceeds the 2000-variable licence.
>
> **Where the build departs from the plan below:**
> * **Extraction bug, found and fixed (pre-existing, flat MILP too).** The no-copy switch rule in
>   `dfs_remove_unnecessary_flows` deleted a switch's in-flow after the first traced path, so every
>   later demand sharing that prefix through a relay GPU was silently skipped. That is exactly the
>   "cell wants and relays" case. It is now a claims map (in-flow → the one out-flow it feeds).
>   A/B on identical solves: flat output unchanged wherever no demand had been skipped.
> * **Leg epochs are normalized.** The MILP's cut-through credits a switch→GPU hop in the epoch
>   the chunk is already buffered, so `arrival_epoch = landing − 1`
>   (`AllGatherFormulation.hop_timing`). That is the LP's convention, so `send ≥ arrival + 1`
>   (co-located) and `+ 2` (relayed) mean the same on both paths.
> * **Tree placement is a separate function** (`reconstruct._place_tree_route`, chosen by
>   `_place_decisions` when a `placed_legs` memo is passed), not a mode inside
>   `_place_downstream_legs`. The LP path stays untouched by construction. Shared legs are
>   canonicalized in `extract_trees` (legs carry their route's origin, which otherwise made a
>   shared leg compare unequal). Step B drops the duplicate pieces under `copy_aware=True`.
> * **One integral MIP per source cell** (scipy/HiGHS); cells are solved in order against a
>   shared egress ledger.
> * **The transit guard was narrowed.** `flat_schedule.assert_native_ownership` now accepts a
>   non-source staging relay that carries a `release_band` (a modelled transit). It still rejects
>   one without.
>
> **Known gaps:**
> * ncclize reports some paced sends as unpinned (e.g. the first sends after a staging
>   prologue). The LP-hierarchical replay shows the same class (5/13 on hetero allgather), so it
>   is pre-existing stitch/ncclize work, not MILP-specific.
> * Ledgers for multi-epoch legs charge `1/occupancy` per epoch, an approximation that the
>   windowed `_assert_rate_within_capacity` backstops. SLOWEST_LINK avoids it (occupancy 1).
> * The LP half of cell-relay step 5 is still open.
>
> **Update 2026-10-12, switch copy:**
> * The switch-copy gate at MILP levels is REMOVED; `switch_copy` is honored.
> * The rule everything follows is `reconstruct.send_group`: one chunk's legs that leave a cell
>   toward the same switch in the same epoch are ONE send, because the MILP's flow on a link in
>   an epoch is one binary. Such legs share a gateway, are charged once by the integral
>   assignment, the forwarding ledger, `_assert_rate_within_capacity` and
>   `flat_schedule.assert_link_capacity` (copy-aware levels only; the LP still counts per piece).
>   `IdentityResolution.copy_aware` carries the semantics to the stitch.
> * The MTZ switch-acyclicity constraint (folded from `forbid-same-epoch-switch-cycles`) stops
>   copy + cut-through from conjuring chunks out of same-epoch switch loops.
> * Verified: hetero allgather with copy runs end to end through `check_implements`, with 3
>   coarse fan-outs; an independent link count is within capacity when a fan-out is counted once.
> * OPEN (step 2): the flat JSON still has one record per branch, and ncclize lowers each to its
>   own send, so a fan-out puts its uplink at 200% on hardware. This is identical to the flat
>   MILP's output, so one output-representation fix covers both.
>
> The original 2026-10-03 plan follows unchanged. Where it and the bullets above disagree, the
> bullets win.

## 0. Why do this, and what not to expect

The case for the MILP rests on four things. Makespan is not one of them.

1. **Q = 1 at every MILP level.** An integral solve never splits a chunk, so the
   level spends nothing from the `refinement_from_root` budget. The rational snap
   and `MAX_DENOM` never come into play. This is the only known way out of
   coarse-LP degeneracy. The scattered dual-plane alltoall needs `MAX_DENOM`
   ~1875+, which the per-instance override cannot reach; the MILP has no
   denominators at all.
2. **Copy semantics.** A cell node can send what it holds to several neighbours,
   because `node_constraint_helper` uses `max_` over outgoing flows. A cell
   that both wants an identity and relays it therefore pulls it once. The LP
   pulls it twice (the deferred 4.5%-of-paced-volume issue in
   `teccl_ncclize_ingress_gate_gap`).
3. **Identity resolution stays, but it becomes integral.** It cannot be dropped
   (§2.2). Every MILP tree carries a whole number of chunks, so the resolution
   is an assignment of whole identities to whole trees. The snap,
   `_subdivision_factor` and `MAX_DENOM` never get involved.
4. **Copy-bound collectives.** Broadcast and gather have egress fan-out that the
   LP structurally cannot price.

**Not expected:** a makespan win on ingress-bound allgather. LP and MILP tie
there, as already measured.

**Not expected to scale:** rail-256 alltoall. §4 sizes the model. The target is
mid-size coarse graphs (hetero, dual-plane, bridged), plus copy-heavy
collectives.

## 1. What happens today with `formulation=MILP, hierarchical=True`

It fails, at three independent points, and in this order:

| # | where | what |
| --- | --- | --- |
| F1 | `solve.py:379` `coarsify_demand` | The coarse demand is a weighted single slot (values like 4 or 6). The MILP's `total_demand_sat` is a {0,1} var (`allgather.py:99`) pinned `== demand` (`:139`), so the model is **structurally infeasible**. It surfaces as "Unable to find a solution in the initial feasible search". Confirmed 2026-07-31 on HeteroTaperedCluster. |
| F2 | `solve.py:288` `gurobi_level_solver` | Reads `best.per_chunk_flow_paths`, which is an LP attribute, and hard-codes `preserves_identity=False`. That routes MILP output into `assign_identities_free`, the LP's step A. |
| F3 | `solve.py:476` `_lower` | The preserving branch asserts that the routing came from a memoized row (`_memoized_row(coarse)`). Nothing lets a *formulation* claim identity. |

There is also a latent trap. `_resolve_formulation` defaults ALLGATHER to
**MILP**, and `solve_hierarchy` never overrides it. So a hierarchical allgather
config with no explicit `formulation` lands on F1 today. M0 fixes that.

What already works and needs no change: `CoarseSolution.preserves_identity`,
`assign_identities_preserving` (single-leg), the `Q == 1` assertion at
`solve.py:394`, all of step B, `_place_downstream_legs`, `_commit_egress`,
`_pick_ingress`, bands, flatten and the flat stitch. The seam is already in the
right place. The MILP has to plug into it.

## 2. Design decisions

### 2.1 Granularity: one `milp_chunking` policy picks the bundle size `b`

The MILP's chunk is a bundle of `b` fine identities. The extremes are known:

* one chunk per fine identity is the rail allgather (256 chunk types) that
  timed out after 2 h with zero incumbents;
* one chunk per cell (the old "option b, bundle rides the tree") solves, but it
  forces every identity of a cell through a single path, which gives up
  multipath on rail fabrics.

Neither is a sensible default for every topology, so granularity is a
per-level option. Like `formulation` and `epoch_type`, it is scalar or a list by
depth:

| `milp_chunking` | one MILP chunk = | `b` | rescale? |
| --- | --- | --- | --- |
| `"fine"` | one fine identity | 1 | never |
| `"holder"` (default) | one holder GPU's identities for one destination-cell set | that class's size (allgather: `num_chunks`; alltoall: `\|V\|·num_chunks`) | when `b > 1` |
| `"cell"` | one cell's identities for one destination-cell set | that class's size | when `b > 1` |
| `int` | `b` identities of one holder GPU | as given | when `b > 1` |

`"cell"` is rarely the right choice, because it voids multipath on any cell
with more than one uplink. It is there for tractability on large coarse graphs.
The integer form is the custom spec, and it covers everything between `"fine"`
and `"holder"`, such as half a GPU's payload for more pipelining. A callable
key (identity, holder, dest cells → class) is easy to expose through the
Python API later, but not through JSON.

**One mechanism behind all four.** A policy is just `(class key, b)`. The key
is `(holder GPU, dest-cell set)` for everything except `"cell"`, which uses
`(cell, dest-cell set)`. `b` must divide every class size, and that gets
asserted. Where a policy names "one class = one chunk" but the class sizes
differ, `b = gcd(class sizes)`, and a warning names how many chunks each class
got. Example: hetero alltoall under `"holder"`, whose classes are `4·nc` and
`6·nc`. The integer form is checked, not adjusted: a `b` that does not divide
a holder class fails loud and lists the class.

**The MILP sees COUNTS, not bundles.** The coarse level cannot tell two chunks
apart if they share a source cell and a destination-cell set. So the tensor is
`coarse[U][V][c] ∈ {0,1}`, with `|class| / b` columns per `(U, dest-cell set)`.
The policy only chooses `b`. Which identities share a chunk, and which chunk
rides which tree, are decided after the solve by the identity resolution
(§2.2), whose egress tier already prefers single-holder sets. The anchor
`sum_classes (|class|/b) * b * |dest cells| == sum coarsify_demand` is asserted.

**Rescale.** Under `"fine"` (and whenever `b = 1`) the coarse topology keeps
the fine chunk and nothing is rescaled. Otherwise `CoarseTopology.rescale_to_chunk(b)`
plus `scale.coarsen(b)` put the level in a `b`-identity chunk, or the MILP
would believe a bundle crosses a link in one fine-chunk time. That is the
topology/scale half of `set_level_chunk`. Its demand half (`rescale_demand`)
does not apply, because the tensor is built in bundles already. `set_level_chunk`
is split accordingly (§6). Compared with the LP, `"holder"` also drops the
cross-cell GCD. Hetero allgather gets `g = gcd(4,4,6) = 2` on the LP, a unit
that is no host's payload. Under `"holder"`, `b = num_chunks`, one GPU's payload.

**Model size** is driven by columns per source cell. That is #identities under
`"fine"` and #holders × #dest-set classes under `"holder"`, so `num_chunks` stops
multiplying the model there. Cells with fewer columns get zero-padded chunk
columns. The MILP restricts sources (`self.sources`) but not chunks, so the
padding costs variables on uneven hetero levels. Renumbering per source is an
M8 item, not a blocker.

### 2.2 The MILP's chunk labels are arbitrary, so identity resolution stays

The abstraction collapses a cell's GPUs into one node, so the coarse MILP
cannot see *which* GPU egresses. Take two interchangeable chunks of cell U
(same destination-cell set). One leaves on gpu0's uplink, the other on gpu1's.
The MILP prices "gpu1's data on gpu0's uplink, gpu0's data on gpu1's" exactly
the same as the uncrossed assignment, even though the crossed one costs two
egress-stage relays. The same blindness applies at ingress: which landing GPU
receives an identity versus which GPUs in V want it (for alltoall, every
identity in a class can want a different GPU). Reading the identity straight
off the MILP's chunk index would ship that arbitrariness.

So step A for a MILP level is a real resolution, with one structural difference
from the LP's:

> **Assign identities to TREES (whole), with integral volumes.** The MILP
> output is a multiset of anonymous trees per class, each carrying `b`
> identities. Variables are `x[identity, tree] ∈ {0,1}` within a class, with
> `sum_t x = 1` per identity and `sum_i x = b` per tree. Cost per
> (identity, tree) = egress relays (holder ≠ the gateway of each first leg out
> of U; a tree may have several first legs, since U can copy) ≫ ingress relays
> (landing GPU of each leg into V ∉ the identity's targets in V) ≫ epoch
> tie-break. These are the same lexicographic tiers as `_solve_assignment`.

* **Integral by construction, where it can be.** Without parallel uplinks
  each source cell's problem is a transportation problem (identity supply 1,
  tree capacity `b`). That is totally unimodular, so its LP optimum is already
  integral. **No Q, no snap.** This is the part the MILP actually saves, and it
  is why the LP path cannot do the same: its slot capacities are fractional
  coarse flow.
* **Parallel fine uplinks: solve it as a small MIP.** When a first leg's coarse
  link aggregates several fine uplinks, an identity also picks the leg's
  gateway. Use `y[i,t] ∈ {0,1}` for identity → tree, and `z[i,t,l,g] ∈ {0,1}` for
  gateway `g` on first leg `l` of tree `t`, with `sum_g z[i,t,l,g] = y[i,t]`.
  Costs sit on `z`, with a per-(gateway, epoch) capacity from fine link
  capacity × Δ in identity units. Two things differ from the LP's `_Slot`:
  * Its proportional `capacity` share does not work here. `b = 1` over two equal
    uplinks gives two slots of 0.5, which is integrally infeasible. Gateway
    capacity is therefore a shared per-epoch row.
  * That row couples every class and tree of the source cell, so the solve runs
    **per source cell**, not per `(U, V)` pair.

  The instances stay tiny (one cell's identities). Solve them with
  `scipy.optimize.milp` (HiGHS, scipy 1.18 in the `teccl` env) and assert every
  value is 0/1. Never refine.
* **Downstream legs stay identity-independent.** At a transit cell the bundle
  is already whole on its landing GPU. Out-leg gateways are therefore placed by
  `_place_downstream_legs` (co-location first), as in `cell_relay_design.md`
  §3.2, and are not priced in the assignment.

### 2.3 Reading the routing as trees

The MILP keeps one copy per chunk and multicasts it. Its solution, per chunk,
is an arborescence over cells. Its edges are **legs**, maximal switch runs
cell → sw… → cell, the same unit `_extract_routes` uses. The per-destination
paths that `dfs_remove_unnecessary_flows` traces share prefixes, so emitting
them as independent routes would ship a shared leg once per destination.
Extraction therefore dedups legs on
`(chunk, src_cell, dst_cell, via_switches, send_epoch)` into a tree.

Validation runs per tree. Fail loud on every one of these:

* each cell is entered at most once;
* every non-root leg leaves a cell no earlier than `arrival + dwell` (§2.5);
* every destination cell of the chunk's class is reached.

The last check matters because `dfs_remove_unnecessary_flows` **skips** a demand
it cannot trace (`allgather.py:822`, a phantom same-epoch switch cycle) and only
logs a warning. That is tolerable for a flat run. In the hierarchy it is a
silently missing delivery, so the adapter must turn it into an error.

### 2.4 `switch_copy=False` at every hierarchical MILP level *(superseded 2026-10-12: switch copy is honored, see status)*

A switch fan-out shows up as one uplink send and several deliveries. The stitch
emits merged per-(sender, receiver) records `via switches`, so it would charge
the uplink once per receiver, and real NCCL has no switch multicast without
SHARP. The LP already requires `switch_copy=False`. The level solver forces it
and warns if the user asked otherwise. Copy at GPU/cell nodes (§0.2) is
unaffected.

### 2.5 Transit is the COMMON case, so forwarding dwell moves onto the critical path

With the LP, cell relay is an edge case (`cell_relay_design.md`). With the MILP
it is the point: allgather trees route through cells that also want the data.
`cell_relay_design.md` §6 then applies directly:

* co-located (one GPU owns the in-link and the out-link) works with a dwell
  of 1, which is what the MILP's store-and-forward buffer already gives;
* relayed (landing GPU ≠ gateway) needs a dwell of 2. Without it the case is
  infeasible, and M4 fails loud on it.

In the MILP the dwell is **easier** than in the LP, and it can be **exact per
link pair** instead of the per-cell 1-or-2 that `cell_relay_design.md` §6
settles for. Two facts make that work:

* the buffer is an explicit binary;
* its update `buffer[k] = buffer[k-1] + sum_h arrival_h(k)` admits at most one
  arrival, so "the in-link the chunk came on" is well defined.

For a non-source cell `i` and out-link `i → j`:

    flow[s][i][j][c][k] <= buffer[s][i][c][k-1] + sum_{h in B(i,j)} arrival_{h->i}(k)

Here `B(i,j)` is the set of in-neighbours `h` such that some GPU of `i` owns
both the `h → i` and the `i → j` links. It comes from `abstract()`'s
`boundary_gpu` and is carried on `CoarseTopology`. `arrival_{h->i}(k)` is the
exact term the buffer constraint already sums. The constraint allows two cases:

* forwarding in the arrival epoch over a bridged pair, which is co-located and
  costs no intra work;
* forwarding after the cell has held the chunk for a full epoch, which leaves
  room for the intra relay.

`total_demand_sat` still reads `buffer[k+1]`, so consumption is never delayed.
That is the separation §6 asks for, with no re-indexing. The constraint adds no
variables, and adds one row per `(s, c, non-source cell, out-link, epoch)`, the
same order as the flow variables on cell out-links.

Residual: if one in-link has several owner GPUs and the cell copies to two
out-links in the arrival epoch, the per-pair test can pass while no *single*
landing GPU bridges both. That needs landing-GPU variables to model exactly.
It is not proposed. M4 placement fails loud on it, and no current topology has
that shape.

### 2.6 Pacing must use the leg's OCCUPANCY, not one epoch

`_piece_rate` paces a piece to fill one coarse epoch. That is right for the LP,
whose capacity is continuous. On a link with `capacity * Δ < 1 chunk`, the MILP
spends `beta + 1` epochs per chunk (`get_beta_num_back`), so one-epoch pacing
overstates the rate by `beta + 1`. Each `_CoarsePiece` gets
`occupancy = max over its hops of (beta + 1)`, alpha excluded because that is
latency, not occupancy. `_piece_rate` then divides by `occupancy * Δ`.
`_assert_rate_within_capacity` already catches the bug if this is skipped. The
ncclize P2 gates take `finish = start + volume/rate`, so multi-epoch sends come
out right with no ncclize change.

Related, and already **accepted** for flat runs: cut-through
(`switch_pipeline=True`) is optimistic on non-uniform bandwidth. Coarse graphs
*are* non-uniform (tapered uplinks against aggregated spine links). Recommended
default at hierarchical MILP levels: `epoch_type=SLOWEST_LINK` (beta = 0
everywhere, so the optimism vanishes and occupancy is 1) through the existing
per-depth list. Compare it against FASTEST_LINK + `switch_pipeline=False` in M7.

### 2.7 Per-level formulation

`formulation` gets the same scalar-or-list-by-depth form as `epoch_type`
(`resolve_formulation(instance, depth)`, next to `resolve_epoch_policy`). Mixed
stacks then need nothing special. LP root over MILP rack, or MILP root over
crossbar hosts, both just work, because step A is picked by
`preserves_identity` and never by solver type. Default for `hierarchical=True`:
**LP at every depth**. That keeps every existing config byte-identical and
closes the §1 trap.

## 3. Milestones

Each milestone ends green on the existing Gurobi-free suite. M1, M2, M4 and M6
are also testable Gurobi-free against a stubbed MILP solution.

**M0 — guard rails (small).** `solve_hierarchy` resolves the formulation per
depth with an LP default (§2.7). Until M3 lands, `formulation=MILP` at a level
that reaches `gurobi_level_solver` raises a clear `NotImplementedError` that
names this doc, instead of failing on F1.

**M1 — identity-preserving coarse demand.** New
`abstract.coarsify_demand_preserving(fine_tensor, mapping, relabel, holders,
policy) -> (coarse_tensor, classes, b)`, where
`classes[(U, c)] = (dest-cell set, member identities)` records which class each
MILP chunk column belongs to. Composition is NOT fixed here (§2.2). The function
sits next to `coarsify_demand` and is driven by the same per-identity loop as
`identity_sets`, so the anchor (§2.1) is checked against `coarsify_demand`
itself. It lands after refactors R1 and R2 (§6), which must leave the LP path
byte-identical on the Gurobi-free suite. Tests, one per policy:
* `"fine"`: `b = 1`, no rescale;
* `"holder"`: hetero 4/4/6 allgather (4/4/6 chunks), and rail allgather at
  `num_chunks=4`, where the chunk count must not change and `b = 4` with a
  rescale;
* `"holder"` on hetero alltoall: the gcd fallback plus its warning;
* `"cell"`;
* integer `b`: one that divides, and one that must fail loud.

**M2 — MILP tree extraction.** (a) In `dfs_remove_unnecessary_flows`, keep the
traced tuple paths (`self.per_demand_paths[(s,d,c)] = my_path`) next to the
strings, and return the skipped-demand list. That is additive, and flat output
stays byte-identical. (b) New `reconstruct._extract_trees(milp, mapping,
bundles)`: switch-run walk (shared with `_extract_routes`, so factor out the
leg builder), leg dedup into trees, the §2.3 validation, and `occupancy` per
leg (§2.6). Gurobi-free tests on handcrafted `(s,i,j,c,k)` sets: a shared
prefix, a relay through a wanting cell, a missing delivery that must raise, and
a slow link with beta = 1.

**M3 — plumbing.** `CoarseSolution` gains an optional `trees`, the anonymous
per-class tree multiset from M2, plus `classes`. `_lower` gets a third step-A
branch (`assign_identities_integral`, M4) chosen when `trees` is present. The
memo-row branch stays as it is, which fixes F3. `preserves_identity` is
split per R7 (§6), because it conflates "identity is read off" with "no Q". A
MILP level has the second property but not the first. `_solve_recursive` picks the coarsifier by the level's
formulation. `gurobi_level_solver` builds the MILP-level `CoarseSolution` and
forces `switch_copy=False` (§2.4). That fixes F1 and F2.

**M4 — integral identity resolution over trees (§2.2).** New
`reconstruct.assign_identities_integral(trees, classes, holder, targets, ...)`,
producing `_Assignment`s so step B is unchanged. It is mostly assembled from
R3–R6 (§6):

* per class, solve the identity→tree transportation problem with the
  lexicographic costs in §2.2. Where a first leg's coarse link aggregates
  parallel fine uplinks, the slot is `(tree, gateway)` with a per-(gateway,
  epoch) capacity, solved as a small integral MIP. Assert every `x` is 0/1;
  never refine;
* the solved `(tree, gateway)` slot fixes the first-leg egress and records it
  in the **egress ledger** (`_commit_egress`). Today's preserving path has no
  egress capacity check, which was safe only because crossbar levels have one
  gateway;
* each cell the tree enters: the landing GPU is a slot choice when ingress is
  priced, else `_pick_ingress` against the ingress ledger. That GPU becomes the
  holder for the cell's own wanters (an ingress distribution) and for every
  out-leg (`_place_downstream_legs`, co-location first);
* relayed transit with dwell 1: fail loud, naming M5.

Reuse `_solve_assignment`'s cost construction where it fits. Its LP solve and
snap do not apply. Tests: the crossed gpu0/gpu1 case must come out uncrossed,
with zero egress stages, and an alltoall class whose trees land on different V
GPUs must match identities to their own targets.

Step B is unchanged, and the `Q == 1` assert holds. The first end-to-end target
is a Gurobi-free stub: a hand-written MILP solution on HeteroTapered coarse,
taken through the stitch to `check_implements()`.

**M5 — forwarding dwell in the MILP (§2.5).** Per-pair bridging sets `B(i,j)` in `abstract()`, the
one constraint family in `node_constraint_helper` for non-source cell nodes,
and dwell-aware validation in M2. This closes cell-relay "step 5" for the MILP.
The LP half stays open and is a separate change.

**M6 — occupancy pacing (§2.6).** `_CoarsePiece.occupancy` and the `_piece_rate`
change. The LP keeps occupancy = 1, so LP output is byte-identical. Test: a
beta = 1 coarse link must pass `_assert_rate_within_capacity` and `check_network_pacing`.

**M7 — validation runs.** Local runs under the restricted Gurobi license
(2000 vars) on the smallest bridged and hetero coarse instances that fit. Size
them first with the §4 formula. Remote runs: HeteroTapered AG + A2A, the
dual-plane clustered and scattered allgather/alltoall pairs, LP-hierarchical vs
MILP-hierarchical vs flat. Report makespan, Q, op count and solve time. The
scattered alltoall is the headline: LP needs Q > 64, MILP must show Q = 1.
DES-simulate every XML, because the pacing report is not transitive
(`ncclize_pacing_report_not_transitive`).

**M8 — later / optional.**
* MILP as the base case for a bottom cell with no closed form
  (`solve.py:344`, the `NotImplementedError`). Its per-identity flows convert to
  `IntraFlow`s directly, with no step A.
* MIP start from the LP's routes, quantized to whole bundles. That gives a
  tractability lever plus a guaranteed incumbent under the time limit.
* `add_symmetry_constraints` on twin cells.
* Per-source chunk renumbering to drop the padding (§2.1).

## 4. Tractability: what to expect

Model size ≈ `|sources| × |links| × chunks_per_source × epochs` binaries.
Per-level numbers under `"holder"` bundling:

| level | sources | chunks/source | note |
| --- | --- | --- | --- |
| hetero 4/4/6 AG | 3 cells | 4–6 | small; should solve locally or near it |
| rail-32 AG | 32 | 8 (`"fine"`: 8·nc) | the known 2 h / no-incumbent instance (that was `"fine"`, nc=8); `"holder"` at nc=1 is the same size, so it needs `"cell"` (1 chunk, ~26 min optimal) or an M8 MIP start |
| rail-32 A2A | 32 | 8 × 31 = 248 | out of reach; stay on LP |
| dual-plane coarse (18 nodes) A2A | ~ cells | holders × dest cells | the case worth measuring (§0.1) |

Levers, cheapest first: `"cell"` bundling, the per-level time limit with the
incumbent accepted (`SolCount > 0` already gates extraction), SLOWEST_LINK
epochs (fewer epochs), then the M8 MIP start and symmetry.

## 5. Risks and open questions

* **Bundles force co-travel.** A tree's `b` identities share every leg and
  epoch. Under `"holder"` the resolution can always pick a single-holder set,
  so this costs nothing beyond the resolution's own suboptimality. Under
  `"cell"` mixed-holder sets are forced and add egress staging. That is visible
  as `egress_stage` volume, and M7 should report it.
* **The resolution is decoupled from the solve.** The MILP minimizes coarse
  makespan blind to relay cost. The resolution then minimizes relays given the
  trees. A tree set that is optimal for the MILP can still force relays that a
  different, equally fast tree set would avoid. This is the same seam the LP
  path has, and it is accepted for the same reason (intra fabric headroom).
  Revisit only if M7 shows relay volume pushing bands past `m`.
* **Tree validation is new ground.** Copy-aware trees have never gone through
  `LevelDemand.from_demands`' chain collapse. A cell that relays *and* wants an
  identity produces an ingress distribution plus an egress stage from the same
  landing GPU. Expected to collapse cleanly (one holder, union of wanters), but
  untested. Add it as an M4 test.
* **Phantom switch cycles** (open formulation gap) become fatal in the
  hierarchy instead of a skipped demand (§2.3). If they show up on coarse
  graphs, `switch_pipeline=False` at MILP levels is the escape hatch.
* **Decision for the user:** whether M5 is in scope before the first remote
  run. Without M5, any topology whose transit cells lack a bridging GPU fails
  loud at M4. (Chunking policies settled 2026-10-04: `"fine"`, `"holder"`
  (default), `"cell"`, integer `b`.)

## 6. Reuse map

The MILP path is meant to be mostly the LP path's machinery, re-entered at a
different seam. Refactors R1–R7 come first. Each is behaviour-preserving on the
LP path and checked byte-identical against the Gurobi-free suite before any
MILP code uses it.

**Reused unchanged**

* `abstract()`, `lift_demand`, `HierarchyMapping` (`boundary_gpu`,
  `coarse_passthrough`), `CoarseTopology.rescale_to_chunk`, `ChunkScale`.
* Step B, all of it: `build_child_problems` → `_snap_volumes` (a no-op on whole
  volumes), `_subdivision_factor` (returns 1), `_emit_refined`,
  `_coalesce_egress`, `_emit_self_distribution`, `_dedup_deliveries`,
  `_assert_rate_within_capacity`.
* Placement primitives: `_place_downstream_legs`, `_commit_egress`,
  `_bridging_gpus`, `_ingress_epoch_capacity`, `_egress_epoch_capacity`.
* Data types: `_CoarsePiece`, `_CoarseRoute`, `_Assignment` (`leg_gpus`,
  `holder`), `IntraCellDemand`, `LevelDemand`, `Subproblem`.
* The recursion (`solve_level`, `_solve_recursive`), bands, flatten,
  `flat_schedule`. The MILP formulation itself, apart from M2a and M5.

**Refactor to share** (R = lands before the MILP code that needs it)

| # | today | becomes | used by |
| --- | --- | --- | --- |
| R1 | `coarsify_demand` and `identity_sets` each walk every `(s, ci)` → destination cells | `_classify_identities(fine_demand, mapping, relabel)` yields `(identity, src cell, {dest cell: wanters})`; both become thin folds over it | LP coarsify, `identity_sets`, M1 |
| R2 | `set_level_chunk` = GCD + topology/scale rescale + demand rescale, fused | `rescale_level(coarse, scale, g)` (topology + scale) split from `rescale_demand`; `set_level_chunk` keeps its signature and calls both | LP (unchanged), M1 with `g = b` |
| R3 | `_extract_routes` walks switch runs inline over LP tuples `(…, i, j, …, vol, k)` | `_legs_of_path(hops, switch_indices, passthrough, origin)` over normalized hops `(sender, receiver, volume, epoch, occupancy)`; LP and MILP adapters normalize their own tuples | `_extract_routes`, M2 `_extract_trees` |
| R4 | `_build_slots` enumerates gateways, checks the passthrough, and computes proportional shares in one loop | `_leg_gateways(leg, cell, mapping, topo) -> [(gpu, cap)]` and `_leg_ingress_candidates(leg, cell, mapping)`; `_build_slots` keeps only the proportional share | LP slots, M4 `(tree, gateway)` columns + per-epoch rows |
| R5 | `_solve_assignment` builds the lexicographic cost vector and calls `linprog` | `_assignment_costs(identities, native, targets, columns)` returns the vector (same W_EGRESS / W_INGRESS / W_EPOCH tiers); the LP solve stays where it is | LP, M4 (`scipy.optimize.milp` with integrality) |
| R6 | `assign_identities_free`'s tail: decisions → transit-first ordering → `_place_downstream_legs` → `_Assignment` | `_place_decisions(decisions, …) -> List[_Assignment]` | LP, M4, and `assign_identities_preserving` (whose single-leg loop becomes a call to it) |
| R7 | `CoarseSolution.preserves_identity` means two things at once | `integral` (gates the `Q == 1` assert) + where the routing comes from (memo row / `trees` / LP paths) | `_lower` dispatch (M3) |

**Two changes inside reused code, because the MILP needs whole volumes:**

* `_pick_ingress` may **split** a landing across several fine downlinks. That is
  legitimate for the LP, but on the MILP path it re-creates fractions and so
  Q > 1. It gets an `integral` mode: the whole identity lands on one GPU, or it
  fails loud. Where ingress is tight, price the landing in the M4 MIP instead.
* `_place_decisions` places each `(U, V)` route on its own. A tree's
  per-destination routes share prefix legs, so placing them independently would
  place and ledger-charge a shared leg once per destination. The fix is a
  per-`(identity, leg)` placement memo, so a shared leg is placed once and later
  routes start from its landing GPU. It is a no-op on the LP path, where routes
  never share a leg.

**Genuinely new:** M1's policy resolution (on R1/R2), the MILP's traced-path
retention plus skipped-demand list (M2a), `_extract_trees`' dedup and validation
(on R3), the integral assignment's constraint rows (on R4/R5), the forwarding-
dwell constraint (M5), `occupancy` in `_piece_rate` (M6), and the per-depth
`resolve_formulation` / `resolve_milp_chunking` next to `resolve_epoch_policy`.
