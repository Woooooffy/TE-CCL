# Switch copy as a first-class schedule fact

Status: IMPLEMENTED 2026-10-13 on branch `switch-copy-send` (worktree `../TE-CCL-switch-copy-send`,
off the pace-class commit `ccfd120` of `milp-bottleneck-pipeline`), uncommitted. Phases 0-5 of §8
are in; §12 records where the implementation differs from the plan below, and how it was checked.
Scope: from the MILP solve through the stitch. The MSCCL XML and switch-forwarding-JSON emitters
are OUT of scope; this design ends at the in-memory schedule they will read.

## 1. The problem

A MILP solved with `switch_copy=True` may send a chunk into a switch once and have the switch send
it out on several links. The solved flow variables encode this exactly. Every stage after the solve
then loses that structure and has to rebuild it:

| Stage | Representation | How it gets the copy back |
|---|---|---|
| `dfs_remove_unnecessary_flows` | one path per demand, shared prefixes duplicated | (this is where it is lost) |
| `extract_trees` | `_CoarseRoute` per dest cell | canonical leg objects + a cell-entered-once check |
| `_solve_integral_assignment`, `_place_tree_route` | legs | `send_group = (cell, egress switch, epoch)`, `placed_legs` memo |
| `build_child_problems` | `ResolvedPiece` per leg | `dict.fromkeys` dedup, `copy_aware` flag |
| `_assert_rate_within_capacity`, `assert_link_capacity` | per-leg records | `one_copy_per_link` key matching |
| `pieces_as_flows` (3+ levels) | `IntraFlow` | nothing: only the first switch is kept, so the copy is gone |

Each rebuild is exact only because of invariants nobody states. **Pace classes break one of them.**
A flow tuple is now an INTEGER count of copies, each with its own pace class. So two legs with the
same `(cell, egress switch, epoch)` can be **two transmissions**. `send_group` would charge them
once and give them one gateway. The bounds (`class_copies`, `send_bound` on the pace-class branch):
- with switch copy on, at most one copy per class, so concurrent copies exist only ACROSS classes,
  and at a one-class level (`SLOWEST_LINK`) every send is binary again;
- with switch copy off, up to `max_copies(s, c)` (the number of nodes demanding the chunk) per class.

This was not hypothetical. The HeteroTaperedCluster copy run had a 2-copy send before the
`class_copies` bound existed; a guard in `extract_trees` (`flow_copies == 1`) now fails loudly on
any multi-copy send until this design replaces `send_group`. The bottom-level crossbar scheduler
also hard-codes `switch_copy=False`.

## 2. The design in one sentence

**The MILP extraction outputs a list of `Send`s, where each `Send` is one physical transmission
(one stream, at one pace class) leaving a non-switch node, fanning out through switches to one or
more receivers. That same type is what every later stage carries, in its own index space and time
units.**

Unicast is the case of a single branch. No-copy is the case where every send has a single branch.
The LP produces single-branch sends. Nothing downstream asks "is this level copy-aware?" any more.

## 3. Data structures

New module `teccl/sends.py`. It is neutral, so solvers, hierarchy and ncclize can all import it
without a cycle.

```python
@dataclass(frozen=True)
class Branch:
    receiver: int                  # non-switch node the copy lands on
    via: Tuple[int, ...]           # switches from sender to receiver, in order; () = direct link
    ready: int                     # first epoch the receiver HOLDS it (holder's units)

@dataclass(frozen=True)
class Send:
    identity: Tuple[int, int]      # (source, chunk) in the holder's labelling
    sender: int                    # non-switch node that transmits
    epoch: int                     # epoch the head of the transmission starts (holder's units)
    span: int                      # pace class + 1: epochs EVERY link of the send is held
    fanout: Tuple[Branch, ...]     # >= 1 branch, sorted by receiver
    volume: float = 1.0
    kind: str = "network"          # network | egress_stage | ingress_distribution | ...
```

Invariants, asserted once in `Send.__post_init__`:
- every branch with a non-empty `via` has the same `via[0]` (one physical link out of the sender);
- the branches' `via` prefixes form a tree: no switch is reached by two different upstream prefixes;
- at most one branch per receiver.

**One timing rule for every link.** A send holds each link of its tree for `span` epochs, starting
when its head reaches that link: `epoch + Σ alpha` of the links before it. On the LP, and on any
link with zero propagation, that is simply `[epoch, epoch + span)`. A branch's `ready` is
`epoch + Σ alpha(path) + span`. The MILP's own capacity rows charge exactly this (a class-b stream
uses `1/(b+1)` of the link for `b+1` epochs on every link of the traversal), so downstream checks
and the solve agree by construction.

Helpers in the same module. They are derived views; nothing else stores them.
- `links(send) -> Set[(u, v)]`: the union of tree edges. Each is charged once, at `1/span` per
  epoch over the send's window on it.
- `switch_out(send) -> Dict[switch, Tuple[next_node, ...]]`: the fan-out at each switch. This is
  what the switch-forwarding JSON will later serialize as a list of output ports.
- `receivers(send) -> Tuple[int, ...]`: what the XML will later serialize as the destination set.
- `delivery_index(sends) -> Dict[(identity, node), (Send, Branch)]`: for each node, the earliest
  branch delivering each identity to it. This is the forest's parent pointer.

### Why these two types and nothing more

- **A branch list rather than an edge set.** Every consumer works per receiver (landing, ingress
  fan-out, the stitch's causality chains), and a receiver's path is right there on its branch. The
  edge set and the per-switch port lists are one helper away.
- **`ready` rather than `arrival_epoch`.** It is the only time any consumer uses: band release,
  the forward dwell, and the stitch's "holds" check. It is also exactly the MILP buffer's
  semantics. It replaces three conventions ("lands during", `arrival + 1`, `completion`) with one.
  It is derivable from `epoch`, `span` and alpha, but consumers below the solver do not have the
  alpha table, so it is stored.
- **One `span` per send, not per link.** Pace classes make this exact rather than an
  approximation. Switches conserve flow per class, so a traversal keeps one class end to end, and
  the class is at least every link's beta: its bottleneck. Under switch copy, a switch's class-b
  outputs are fed only by its class-b input, so **every branch of one multicast carries the same
  class**. A slow branch slows the whole multicast, which is the physical behaviour (backpressure
  on the shared input). This replaces the per-link `Send.span` / `Branch.span` split from the
  2026-10-12 revision. That split only existed because the old formulation charged links
  independently.
- **`span` rather than `rate`.** A rate is `volume * bytes_per_chunk / (span * epoch_duration)`.
  That is exactly `_piece_rate` today, and it is now the bottleneck rate. Whether a send is paced
  follows from `kind` (network sends are paced; intra flows are not).
- **No "parent" field.** A send's parent is `delivery_index[(identity, send.sender)]`. Each node
  receives an identity once on a committed forest, so the index is a function.

### What it replaces

| Today | After |
|---|---|
| `per_demand_paths` + `per_demand_classes`, `required_flows` (as primary output) | `sends`; per-demand paths and classes become a derived view |
| `claims` (no-copy map, now sets bounded by `flow_copies`) | stream pairing in the forest builder (§4) |
| `Trees`, `extract_trees`, canonical legs, the tree check | the level's `sends`, grouped by identity |
| `send_group`, `_first_legs`, the `placed_legs` memo, `("send", …)` group keys | one `Send` = one stream. This also fixes the multi-copy undercount (§1). |
| `ResolvedPiece` | `Send` (fine GPU ids, coarse epochs) |
| `DeliveryRecord` | `Send` (fine GPU ids, absolute fine epochs, `kind` = phase) |
| `IdentityResolution.copy_aware`, `one_copy_per_link`, `dict.fromkeys(pieces)` | deleted: `links(send)` is the union by construction |
| `_CoarsePiece.occupancy` = max hop occupancy | `Send.span`; within one traversal every hop now has the same occupancy anyway |
| `IntraFlow.receiver`, `IntraFlow.via_switch` | `IntraFlow.fanout: Tuple[Branch, ...]` |

`IntraFlow` keeps its own type because its time is a two-part `(band, local_round)`, which is what
`rebase` folds. It shares the fan-out representation and nothing else. The LP's internal step-A
types (`_CoarsePiece`, `_CoarseRoute`, `_Slot`, `_Assignment`) are untouched; only their output
becomes single-branch `Send`s.

## 4. Extraction in `allgather.py`

This builds on the pace-class trace, which already returns `(flow, class)` pairs
(`get_viable_predecessors`, `_dfs_trace_to_source`). It keeps the backtracking DFS, the
(node, epoch) cycle guard and the budget. What changes is that traces are **committed into one
forest of streams** rather than recorded independently.

**The unit is a stream, not a flow tuple.** A solved flow `f` carries `flow_classes(f) = {b: n_b}`
copies. Its streams are `(f, b, n)` for `n < n_b`; an unpaced link (GPU–GPU) has the single stream
`(f, beta, 0)`. A stream is one physical transmission. With switch copy on, `n_b <= 1`
(`class_copies`), so a stream is just `(f, b)`; the copy index only matters with copy off.
Head-anchored timing means every stream of a traversal is indexed at the head's epoch plus
cumulative alpha.

State while building:
```python
parent:    Dict[Stream, Optional[Stream]]      # committed stream -> the stream it continues; None at source
children:  Dict[Stream, List[Stream]]          # inverse, for grouping
delivered: Dict[(s, c, node), Stream]          # non-switch node -> the committed stream into it
```

**Trace each demand** `(s, d, c)` in a deterministic order (by met epoch, then s, d, c):
1. If `(s, c, d)` is already in `delivered`, the demand is already on the forest as a relay. Done.
2. Otherwise run `_dfs_trace_to_source` with two changes:
   - **Join instead of recursing.** When a candidate predecessor's sender is a non-switch node
     already in `delivered`, or a candidate stream is already committed, end the path there. Its
     ancestry is already known.
   - **Prefer committed candidates**, then earliest epoch. This shares as much as the solution
     allows and makes the result deterministic.
3. **Stream pairing at a switch** (replaces `claims`). A class-b out-stream is fed by a class-b
   in-stream of the same chunk (per-class conservation guarantees one exists).
   - Copy on: one in-stream may feed any number of out-streams. Prefer the already-committed one.
   - Copy off: each in-stream feeds exactly one out-stream; take the first unpaired.
4. **Commit the path:** set `parent`/`children`, and set `delivered` for every stream that lands on a
   non-switch node.
5. Untraceable demands go to `self.untraced_demands`, as today.

**Group into sends.** Every committed stream whose sender is a non-switch node starts one `Send`.
Walk `children` forward through switches. Each stream that lands on a non-switch node becomes
`Branch(receiver, via, ready = k + landing offset)`, with the offset from `hop_timing(j, i, b)`.
`Send.span = b + 1` (one class for the whole tree, asserted). `epoch` is the head stream's k.

**Outputs on the formulation:**
- `self.sends: List[Send]`, sorted by (epoch, sender, identity). This is the authoritative result.
- `self.untraced_demands`, unchanged.
- `self.required_flows` = all committed flow tuples, so the `get_schedule()` contract is unchanged.
- `per_demand_paths`, `per_demand_classes` and the `7-Flows` / `8-Chunk paths` strings are a
  projection over `delivery_index(self.sends)`. The flat MILP output keeps its format until the XML
  phase.

`diagnose_switch_phantom_cycles` and the MTZ constraints are unaffected.

## 5. A MILP level in the hierarchy

This replaces `extract_trees`, `assign_identities_integral`'s placement, `_place_tree_route` and the
copy-aware half of `build_child_problems` with one walk over the level's sends. It is the same code
at every depth; `relabel` and `holder_of` carry the depth-specific translation, as today.

**5.1 Read the solution** (`gurobi_level_solver`). It returns
`CoarseSolution(step_a="integral", sends=best.sends, milp_demand=...)`. The level's sends are in
coarse ids: cells and coarse switches. Group them by identity (cell, column) and keep epoch order.
Epoch order is topological, because a node forwards only after it holds the chunk.

**5.2 Coverage.** Use `delivery_index` on the column's sends. Every demanded (column, dest cell)
must have an entry, or the level fails loudly. The tree check disappears; it is a `Send` invariant.

**5.3 Assignment MIP.** Same formulation, re-keyed:
- `z[d, c, S, g]` for each **first send** `S` (a send whose sender is the column's source cell).
  Two copies on one coarse link are two first sends with two gateway choices.
- Capacity rows: `1/S.span` per epoch over `[S.epoch, S.epoch + S.span)`.
- The ingress-miss cost is per branch of the column whose receiver cell wants `d`.

**5.4 Place one identity on its column.** Walk the sends in epoch order, keeping
`at: Dict[cell, gpu]`, the GPU in each cell that holds `d`, starting with `{source cell: holder}`.
For each send `S` out of cell `U`:

- **Choose the sender GPU `g`.**
  - First send: the MIP's gateway.
  - Forward: `at[U]` if it owns the link toward `S`'s first switch and has room (co-located).
    Otherwise a gateway with room, provided `S.epoch >= ready_in(U) + 1`, which is the dwell that
    `forward_dwell_constraints` guaranteed. Otherwise fail loudly (the existing errors).
- **If `g != at[U]`**, emit an `egress_stage` from `at[U]` to `g`. Its deadline is `S.epoch`. Its
  release is `ready_in(U)` for a forward, and it is native (the prologue) for a first send.
- **Charge** `g`'s uplink `1/S.span` per epoch over the send's window.
- **For each branch**, choose the landing GPU in receiver cell `V`:
  - If `V` has outgoing sends in this column, prefer a GPU that bridges all of them, with egress
    room (decision 1).
  - Otherwise prefer a GPU in `targets[(d, V)]`.
  - Charge its downlink `1/S.span` per epoch over the branch's window, `[ready - span, ready)`.
  - Set `at[V] = landing`. If `V` wants `d`, emit an `ingress_distribution` from the landing GPU
    to `targets[(d, V)]`, released at `branch.ready`.
- **Emit** `Send(d, g, S.epoch, S.span, fanout=fine branches)`. Each fine branch is
  `Branch(landing, passthrough(via), ready)`.

Each send is placed exactly once per identity, so no memo is needed and nothing has to be
deduplicated afterwards.

**5.5 Shared tail, unchanged:** `_coalesce_egress`, `_emit_self_distribution`, `_dedup_deliveries`,
and the rate check. The rate check charges every link in `links(send)` at the send's rate over its
window, with no flag. That covers the fine spine links too, which today go unchecked after lowering.

The result is `IdentityResolution(pieces: List[Send], intra_demands, scale, subdivision=1)`. The LP
path produces the same type, so the recursion below `_lower` does not know which solver ran.

## 6. Switch copy at the fine (cell) layer

- **Per-level setting.** `InstanceParams.switch_copy: bool | List[bool]` with
  `resolve_switch_copy(instance, depth)`, mirroring `resolve_formulation`.
  `gurobi_level_solver` writes the resolved scalar into the per-level `ui`. `_solve_base` passes it
  to `schedule_cell` instead of the hard-coded `False`.
- **If a bottom cell is ever solved by the MILP** (today that path raises `NotImplementedError` in
  `_solve_base`), its output is the same `sends`. It needs no new representation.
- **Crossbar row with copy** (`crossbar_solve`):
  - `_Job.dst` becomes `dsts: Tuple[int, ...]`.
  - `_to_jobs`: under copy, a fan-out demand becomes one multicast job instead of N direct edges
    or a binomial tree. Jobs with the same (identity, src) and release merge their `dsts`. A merged
    job is hard if any part is hard, with the tightest deadline.
  - `_schedule_band`: a job fits a round when the source's egress **and every** destination's
    ingress have room. It charges each once.
  - The output is one `IntraFlow` whose `fanout` has a branch per destination, all with
    `via=(nvswitch,)` and `ready=local_round + span`.
  - `_coalesce_subchunks` and `_assert_ports` key on `dsts`.
- **Ring row.** It has no switch to copy at. It emits single-branch `fanout`s and ignores the flag.

Copying inside a cell assumes NVSwitch multicast (NVLS, Hopper and later), hence a per-level switch
that defaults to off below the root (decision 3).

## 7. Through the recursion and the stitch

**Folding a child level up** (`pieces_as_flows` + `rebase`, 3+ levels). A child's `Send` becomes an
`IntraFlow` with the **same `fanout`**:
- the receivers and the full `via` path are kept;
- `ready` maps to the parent's round axis as `(ready + 1) * w`, consistent with where `rebase`
  places the send;
- its node ids go through `to_this_levels_indices` (receivers and every `via` entry).

This closes today's three-level loss.

**The stitch** (`flat_schedule`). The input is the root's `pieces: List[Send]` plus every
`IntraFlow`. The output is `records: List[Send]` on the absolute fine axis, plus the info dict.

- `build_records`:
  - An intra flow becomes one `Send` per identity it carries:
    `epoch = W + m·band + local_round`, `kind = f.kind`, branch `ready` mapped the same way.
  - A piece becomes a `Send` with `epoch = W + m·e`, `span = m·span`, and each branch's
    `ready → W + m·ready`.
- `assert_link_capacity`: for each network send, charge every link in `links(send)` over the send's
  window. A shared link is counted once because it appears once in the set.
- `back_trace`: `holds` and `via` are `delivery_index(records)`. A chain is a list of
  `(Send, Branch)`. The causality check is unchanged: the sender must hold the identity by
  `send.epoch`.
- Pruning: a send is emitted with only the branches some chain uses. Unused branches are dropped
  (`replace(send, fanout=used)`), and a send with no branches left is dropped.
- `check_network_pacing`: one entry per emitted network send. Today it counts one per branch, so a
  switch copy produces duplicate entries.
- `serialize`: until the XML phase, project each `(Send, Branch)` into the existing string grammar,
  so ncclize is unchanged. This projection is the **only** remaining place copy information is
  dropped, and it is the boundary the XML / switch-JSON work replaces. Those emitters read
  `records` directly: `receivers(send)` gives the destination set, `switch_out(send)` the port
  lists.

`solve_hierarchical` returns `records` alongside the info dict (it already returns `_records`
internally).

## 8. Implementation order

Each phase leaves every existing test passing.

0. **Land pace classes** (`milp-bottleneck-pipeline`) into `hierarchical-milp`, with the objective
   decision in §11 settled, since it changes hierarchy makespans.
1. **`sends.py` + `allgather.py` stream-forest extraction (§4).**
   `per_demand_paths` / `per_demand_classes` and the strings are derived from the forest. The flat
   output must match phase 0's on every no-copy, single-copy case. It may differ only where the old
   independent traces disagreed.
2. **Hierarchy MILP lowering from sends (§5).**
   Delete `extract_trees`, `_first_legs`, `send_group`, `_place_tree_route`, the copy-aware
   `build_child_problems` branch and `copy_aware`. `ResolvedPiece` becomes `Send` (the LP path
   emits single-branch sends).
3. **Stitch on `Send` (§7).** Delete `DeliveryRecord` and `one_copy_per_link`.
4. **`IntraFlow.fanout` + `pieces_as_flows` (§7).**
5. **Per-level `switch_copy` + crossbar multicast jobs (§6).**

## 9. Tests

- **`milp_trace_test.py`** (as re-timed on the pace-class branch).
  - `FANOUT_FLOWS` gives **one** send with two branches.
  - No-copy gives single-branch sends and one untraced demand.
  - Phantom gives both demands untraced.
  - A **multi-copy** fixture (two copies of one chunk on one GPU→switch link, no-copy switch) gives
    **two** sends with the same sender, epoch and first link.
  - A forest-invariant checker: each (identity, non-switch node) is delivered at most once; one
    class per send; each send satisfies its invariants.
  - Every delivery is checked against the independent physical checker,
    `milp_pipeline_check.py` (from the pace-class branch), which is the independent timing
    reference the 2026-10-12 brief asked for.
- **`hierarchy_milp_test.py`.**
  - `test_switch_copy_send_charged_once` is rewritten on sends: one send per identity with two
    branches, one gateway, one ledger charge.
  - New: the multi-copy case at a coarse level gives two first sends, two gateway decisions and two
    ledger charges. Today's `send_group` charges once.
  - The hetero copy end-to-end case: three coarse multi-branch sends; six resolved sends with two
    branches each, one staging demand per send; the rate check passes without a flag.
- **New: copy plus transit.** A column whose relay cell forwards with a switch copy.
- **New: three levels with a copying middle MILP level.** The fan-out survives `rebase`, and the
  stitch charges the shared links once.
- **New: crossbar copy.** A fan-out becomes one `IntraFlow` with N receivers. The round count drops
  to ≈ 1 per sub-chunk, and port asserts hold.
- **New: non-uniform level.** One hierarchical case with a level at `beta > 0`, so classes are
  exercised end to end. Every hierarchy test today runs `SLOWEST_LINK`, where every class is 0.

## 10. Decisions

1. **Bridging with several outgoing sends. DECIDED (2026-10-12).** Land on a GPU that bridges all
   of them; each remaining send relays inside the cell if its dwell allows, otherwise fail.
2. **`span` when links differ in speed. DECIDED (2026-10-13): one span per send = pace class + 1**,
   applied to every link of its tree. This supersedes the 2026-10-12 per-link split, which modelled
   the old independent-window formulation. Under pace classes it would only ever carry the same
   number twice.
3. **NVSwitch multicast at the cell layer. DECIDED (2026-10-12):** per-level `switch_copy`, default
   off below the root (it needs NVLS-class hardware).
4. **AllToAll.** There is no separate `alltoall.py` on this line; both collectives use
   `AllGatherFormulation`. Alltoall chunks have one destination, so `max_copies = 1` and every send
   has one branch.

## 11. Impact of pace classes on the hierarchy (assessed 2026-10-13)

These are findings from running both trees side by side. Each is either a decision for the user or
a constraint on this design.

- **The BridgedIslandsSplitCluster alltoall "regression" (0.1228 → 0.1628 s) is an objective
  tie, not a timing error.**
  - On both trees the coarse MILP solves with horizon 4 to the same optimal objective (−1800).
  - The new tree's optimum happens to finish in 4 coarse epochs instead of 3. It delivers three
    demands an epoch earlier and the relayed demand `(2 → 0)` an epoch later.
  - With the horizon forced to 3, the new model finds a 3-epoch schedule and reproduces the old
    0.12278 s exactly.
  - The PAPER objective rewards the SUM of per-demand completion. Nothing rewards the makespan, so
    it is free to trade the last demand's time for earlier ones.
  - The allgather improvements (0.04167 → 0.04111 s) are plausibly the same mechanism in the other
    direction.
  - **Decision needed:** add a makespan tier ahead of latency for hierarchy levels (the
    `LEXICOGRAPHIC` tiers in `hierarchical_objective_tiers` have none), or tighten the horizon.
    Until then, hierarchy makespans are tie-break-sensitive.
- **Multi-copy sends make `send_group` unsound** (§1). Observed: the HeteroTaperedCluster copy run
  produced a 2-copy send `(0, 0, 4, 0, 0)` before te-ccl-67 added the `class_copies` bound (a
  same-class second copy is waste under switch copy). After that fix, multi-copy sends arise only
  with switch copy off, or across classes at a `beta > 0` level. Both are reachable at an allgather
  level, and the `extract_trees` guard (`flow_copies == 1`, now on the pace-class branch) turns
  them into a loud failure until phase 2 of this design replaces `send_group`.
- **Brief issue I4 (aggregated coarse capacity) is unchanged and now more visible.** A coarse level
  with `beta > 0` gets pace classes on AGGREGATE capacity. Its class says nothing about the single
  fine link an identity actually rides, so a lowering can still hit the gateway capacity rows as an
  infeasibility. All hierarchy tests run `SLOWEST_LINK` (classes all 0), so this is untested.
- **Unaffected:** `extract_trees` now reads `per_demand_classes` and `hop_timing(j, i, pace)`, and
  leg arrival epochs are unchanged on uniform links (checked by hand on the split case).
  `use_one_less_epoch` becoming a no-op only affects the ITERATIVE solution method. The hierarchy
  tests all use ONE_SHOT, so an ITERATIVE hierarchical run is unchecked.

## 12. As implemented (2026-10-13)

**Where it lives.** `teccl/sends.py` (`Branch`, `Send`, `first_link`, `links`, `link_windows`,
`rate`, `receivers`, `switch_out`, `delivery_index`). Extraction: `AllGatherFormulation.
_dfs_trace_to_source` / `_commit` / `_sends_of_forest` -> `self.sends`. MILP level:
`milp_routing.level_sends` / `_solve_integral_assignment` / `_place_column` / `lower_integral`,
sharing `reconstruct.finish_resolution` with the LP's `build_child_problems`. Stitch:
`flat_schedule` records are `Send`s. Crossbar: `_Job.copies_to`.

**Differences from the plan above.**
- **Forest bookkeeping is `(flow, class) -> copies committed`** (`forest.used`), a stream being
  `(flow, class, n)`. Under copy, the trace JOINS a committed switch input. Without copy, an input
  whose copies are all committed is skipped. `stream_copies` reads a send's per-class count
  (`flow_classes` reports 1 for a single-class multi-copy send, so it is not used for this).
- **`link_windows`** gives the first link `[epoch, epoch + span)`, each branch's last link
  `[ready - span, ready)`, and intermediate links the first link's window. That is exact while
  propagation rounds to 0 epochs, which holds at every hierarchy level. The rate check
  (`_assert_rate_within_capacity`) and the stitch's `assert_link_capacity` charge every link in it,
  so the fine spine links are now checked too.
- **`IntraFlow`** keeps its two-part time and gets `fanout`. A branch's `ready` is unused on an intra
  flow (every branch lands as the transfer finishes). `receiver` / `via_switch` remain as read-only
  properties for point-to-point flows and raise on a multicast. That is what keeps the ring row
  and the crossbar internals unchanged.
- **`InstanceParams.switch_copy`** is `bool | list` with default `(True, False)`: copy at the root,
  not below. `resolve_switch_copy(instance, depth)` resolves it. `AllGatherFormulation.switch_copy`
  holds the level's resolved value, and `_solve_base` passes `level_switch_copy` to the closed-form
  rows. The hierarchical sample inputs that said `"switch_copy": true` now say `[true, false]`, so
  their output is unchanged.
- **Hierarchical MILP inputs and tests use objective 6** (LEXICOGRAPHIC, with te-ccl-67's
  makespan tier), which settles the §11 objective tie.

**Verification.**
- All 18 test files pass.
- The LP-path tests print byte-identical logs to the base commit.
- The flat schedule dicts of four hierarchical MILP end-to-end runs (BridgedIslands allgather,
  BridgedIslands alltoall, split alltoall, hetero switch copy at the root) are IDENTICAL to the
  base commit. The refactor changes the representation, not the schedule: the hetero copy case
  emits 22 resolved sends where it had 28 pieces, the 6 switch copies being one send with two
  branches.
- New tests:
  - `milp_trace_test`: `validate_sends`; [2] one send with two branches; [3b] two copies on one
    link = two sends;
  - `hierarchy_milp_test`: [4c] one fine send with two branches; [4d] a child's multicast survives
    rebase and the stitch; [4e] a crossbar fan-out under copy is one multicast transfer;
  - [6] end to end: switch copy at every level (0.2022 s against 0.2033 s at the root only).

**Still open.**
- The output format: `serialize` writes one string per branch, so ncclize still sees N sends per
  multicast. That is the XML / switch-JSON phase.
- A three-level run with a copying middle MILP level has only a unit test ([4d]); no topology
  exercises it end to end.
- `pieces_as_flows` folds a child send in with span 1 round regardless of the child's span (this
  predates the design).
