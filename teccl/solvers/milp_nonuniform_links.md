# MILP timing on non-uniform links: investigation brief

Status: OPEN, written 2026-10-12 as a hand-off to a separate session. Nothing here has been
changed in code yet. All line numbers are against the `hierarchical-milp` worktree
(`../TE-CCL-hierarchical-milp`) at commit `8d214b0`.

## Why this is being investigated now

On 2026-09-24 one symptom of this problem (cut-through latency anchored to the first hop) was
**accepted as a known limitation**. The reasoning was that the topologies in use are mostly
uniform-bandwidth, and the affected sample inputs are validated by simulation rather than by the
solver's reported makespan.

The switch-copy redesign (`teccl/hierarchy/switch_copy_design.md`) now has to carry the MILP's
per-link timing downstream explicitly. Each `Send` has a `span` for its first link and each
`Branch` has a `span` for its last link, and those spans drive pacing rates, uplink accounting and
capacity checks across every hierarchy level. That design is **on hold** until this investigation
answers one question:

> Is the MILP's per-link timing model correct enough on non-uniform fabrics to be carried
> downstream as-is? If not, what model should replace it?

## What the session should deliver

1. **An exposure audit** (§5, E1): which inputs in use actually hit this, at the flat level and at
   every hierarchy level.
2. **A ground-truth timing model** for a cut-through path and for a switch multicast over links of
   different speeds, with the assumptions written down (chunk size vs switch buffer,
   backpressure).
3. **Measured gaps**: MILP-reported vs ground truth, both for latency (§3 I1) and for link
   occupancy (§3 I2), on small instances built to isolate each.
4. **A recommendation**: keep and document, or which fix (§6). State what it means for
   `Send.span` / `Branch.span` in the switch-copy design.

## 1. How the MILP models time

`AllGatherFormulation` (`teccl/solvers/allgather.py`) is the only MILP. Both allgather and alltoall
dispatch to it (`teccl/scheduler.py:163-184`). Older notes refer to an `alltoall.py`; that file does
not exist on this branch.

**Per-link constants** (`teccl/solvers/base_formulation.py:124-147`):
- `epoch_capacity(i,j) = capacity[i][j] * epoch_duration`, in chunks per epoch.
- `beta(i,j) = max(0, ceil(1 / epoch_capacity) - 1)`: extra epochs a chunk needs to serialize on
  the link. It is 0 iff the link carries at least one whole chunk per epoch.
- `alpha(i,j) = ceil(alpha / epoch)` if `alpha / epoch > alpha_threshold`, else 0: propagation
  delay in epochs.

**Variables:** `flow[s][i][j][c][k]` is binary: chunk (s,c) starts on link i→j in epoch k.
`buffer[s][i][c][k]` is binary: the non-switch node i holds the chunk at the START of epoch k.
Switches have no buffer.

**Rate: the capacity window** (`allgather.py:381-399`). For each link and epoch k,
`Σ_{l=0..beta} Σ_{s,c} flow[...][k-l] <= (beta+1) * epoch_capacity`. Each link is charged
**independently**, with its own window. (`if k - beta < 0: continue` is correct; see the
2026-09-24 note.)

**Latency: when a hop's chunk becomes usable downstream.** There are two places:
- Into a non-switch buffer (`_arrival_flow`, `allgather.py:159-199`):
  - from a GPU, or from a switch when `switch_pipeline=False`: the flow at
    `k - alpha - 1 - beta` lands in `buffer[k]`;
  - from a switch with `switch_pipeline=True` (SWITCH_GPU): the flow at `k - alpha` lands in
    `buffer[k]`. There is **no `+1` and no `beta`** ("switch second link is ignored to remove
    double accounting").
- A switch's send rule (`node_constraint_helper`, `allgather.py:261-283`): a switch may send in
  epoch k what arrived on
  - a SWITCH_SWITCH in-link with pipelining: flow at `k - alpha` (**no `+1`, no `beta`**);
  - any other in-link (GPU_SWITCH, or unpipelined): flow at `k - alpha - 1 - beta`.

`hop_timing` (`allgather.py:143-157`) and `get_viable_predecessors` (`:790-826`) re-derive the same
offsets for extraction. So does `use_one_less_epoch` and the variable pruning in
`initialize_variables` (`:63-80`, `extra_epochs`). Any fix has to change all of them together.

**Net effect for a pipelined path** g →(L1) T0 →(L2) T1 →(L3) h, where g sends in epoch k:
- T0 can forward from epoch `k + alpha1 + 1 + beta1`;
- T1 from that plus `alpha2`;
- h holds the chunk from that plus `alpha3`.

Delivery therefore takes **`1 + beta(L1) + Σ alpha`**. Meanwhile each link Li is occupied for its
own `beta(Li) + 1` epochs in the capacity window, starting at that hop's own epoch.

**The LP is immune.** Flow is continuous and splits across epochs, so it has no `beta` machinery,
and its per-epoch row `capacity · epoch` is exact. **The A\* variant** (`allgather_astar.py`)
inherits the MILP rules. Its `compute_floyd_warshall` (`base_formulation.py:149-178`) uses
**store-and-forward** per-hop distances, `Σ 1/(c·Δ)`, which are inconsistent with the cut-through
rules above. That is a side note, worth a line in the findings.

## 2. When it matters

Exactly when some link a chunk traverses has `beta > 0`, i.e. carries less than one chunk per
epoch, **and** the links on that path differ.
- `epoch_type = SLOWEST_LINK` makes every `beta` 0. All hierarchical MILP tests and the hetero
  samples use it (`hierarchy_milp_test.py:298`, `hetero_hier_allgather_milp_cell.json:23`).
- A uniform-bandwidth topology makes every `beta` equal.
- Exposure comes from `FASTEST_LINK`, from pinned epochs (`epoch_type: 3`), and from fabrics that
  mix port speeds. Known exposed inputs: `teccl/examples/sample_inputs/mini_*_partial_*.json`. All
  8 carry a `_known_limitation` key that states this.

## 3. The issues, separated

### I1. Latency is anchored to the first hop, not the bottleneck (KNOWN, measured)

- **Credited:** `1 + beta(first hop)`.
- **Physical cut-through:** `ceil(chunk / min bandwidth along the path)` =
  `1 + max(beta over the path)`.
- **Measured** on `mini_1gpu_1nic_alltoall_partial_1A_milp.json`: 100 Gbps NICs, 50 Gbps
  leaf–spine uplinks, 1 GB chunks, pinned 0.08 s epoch, so `beta = 1` on the 50 Gbps links.
  - the MILP reports **0.08 s**;
  - its own chosen path needs **0.16 s**;
  - `switch_pipeline=false` reports **0.48 s**;
  - the LP reports a legitimate 0.08 s by splitting across both spines.

  How it was diagnosed: the fabric legs launch at epoch 1, while
  `total_demand_sat[k=0] == buffer[k=1]` credits delivery at the end of epoch 0.
- Comments at `allgather.py:176-196` and `:268-273` describe it.

### I2. Each link holds the chunk independently, which assumes the switch can buffer the speed mismatch (SUSPECTED, not measured)

The capacity window charges L1 for `beta(L1)+1` epochs and L2 for `beta(L2)+1`. If L1 is fast
(beta 0) and L2 slow (beta 1), the MILP considers L1 free again in the next epoch while L2 is still
draining the same chunk.

That is only physical if T0 can buffer the remainder. With a chunk much larger than the switch's
buffer (chunks are MB–GB; switch buffers are tens of MB, shared), link-level flow control should
backpressure the sender to the bottleneck rate. **L1 is then held for the bottleneck time too**, so
the **rate** is also under-charged on the fast link, not only the latency.

The 2026-09-24 note says "the window correctly charges the chunk's RATE; only its LATENCY is
under-charged". That holds per link. It does not hold if backpressure couples the links on a path.
This needs a constructed instance (E3).

### I3. Discretization artefacts (KNOWN, measured on `incast_switch`)

`beta` rounds up with `ceil` and has no threshold. An effective `floor(epoch_capacity)` on links
carrying fractions of a chunk per epoch makes `epochs_required` **non-monotonic in `num_chunks`**.
Measured: n=8 needs 32 epochs and n=16 needs 22, at an oversubscription ratio of 4.05. The exact
closed form and sweep are in the `solver_switch_epoch_store_and_forward` memory note
(`teccl/topologies/incast_switch.py`, `teccl/examples/sweep_num_chunks.py`). It interacts with I1:
the "free last hop" discount shrinks as n grows.

### I4. Hierarchy: a coarse link's capacity is the sum of fine links (NEW, not measured)

`abstract()` builds a coarse link's capacity by **summing** the fine links behind it
(`teccl/hierarchy/abstract.py:169-176`), and `rescale_level` then divides capacities by the bundle
size `b` (`abstract.py:103-104`). So a coarse `beta` is computed on aggregate capacity, while one
identity physically rides **one** fine link.

Example: a coarse link backed by two fine uplinks at 0.5 chunk/epoch each has
`epoch_capacity = 1` and `beta = 0`, but any single identity needs 2 epochs on its fine link.
Today this surfaces downstream as an infeasibility, either
- "integral identity assignment infeasible" (the gateway capacity rows in
  `milp_routing._solve_integral_assignment`), or
- an `_assert_rate_within_capacity` failure (`reconstruct.py:1569`).

The coarse solve never sees it. Each level has its own epoch policy (`resolve_epoch_policy`,
`level_depth`), so the audit has to run per level.

### I5. Switch multicast over branches of different speeds (NEW, consequence of I1 + I2)

With `switch_copy=True`, one send into a switch fans out to several branches. The MILP treats every
branch link independently, exactly as in I2. Physically, a replicated stream is typically limited
by its slowest output port, through backpressure on the shared input. So one slow branch would slow
every branch, including the fast ones, whose MILP `ready` is then optimistic too. The switch-copy
design must know which model holds.

### I6. Downstream consumers currently disagree with the MILP (KNOWN, code-read)

The hierarchy turns MILP hops into legs with `occupancy = max(hop occupancy over the leg)`
(`teccl/hierarchy/reconstruct.py:385`). That value is used for **both** the sender's uplink and the
landing downlink, and for the pacing rate (`_piece_rate`, `reconstruct.py:1544`). So downstream
paces a leg at its path bottleneck, while the MILP charged the uplink only its own window.

Under switch copy, branches with different worst hops give one shared uplink two different values,
and the code takes whichever it sees first (`members[0]` in the assignment MIP; the first piece in
`one_copy_per_link`). Every run so far has had occupancy 1 everywhere, so none of this has ever
been exercised.

## 4. Questions to answer

1. What is the right physical model for a cut-through traversal and a switch multicast over
   heterogeneous links, for the chunk sizes and fabrics in use? Pick between "switch absorbs the
   mismatch" (the MILP today) and "bottleneck-paced end to end", and state the regime where each
   holds.
2. How large are the I1 and I2 gaps on the exposed inputs (makespan, and link oversubscription
   when realised)?
3. Does any hierarchy level currently run with `beta > 0`? Is I4 a latent infeasibility or a live
   one?
4. Is `SLOWEST_LINK`, or refining the chunk so every `beta` is 0 (fix B), an acceptable blanket
   answer? What does it cost in model size and makespan resolution?
5. Which fix (§6) should be adopted? Does it make every hop of a send carry the same span? If so,
   the switch-copy design's two span fields collapse to one value.

## 5. Suggested experiments

- **E1. Exposure audit (no Gurobi).** For every input under `teccl/examples/sample_inputs/`, and
  for every hierarchy level (run `abstract` + `rescale_level` and apply that level's epoch policy),
  compute `epoch_capacity` and `beta` per link. Report:
  - links with `beta > 0`;
  - paths or switch neighbourhoods whose links have **different** `beta`;
  - at coarse levels, coarse links whose fine constituents have `beta > 0` even though the
    aggregate does not (I4).
- **E2. Reproduce I1.** Re-run `mini_1gpu_1nic_alltoall_partial_1A_milp.json` (expect 0.08 s
  reported vs 0.16 s physical). Then apply fix C as a 2-line patch (keep `beta` in both pipelined
  branches) and record the result. The earlier estimate for C was 0.24 s.
- **E3. Construct I2.** A minimal topology: GPU g with a fast uplink to switch T, and two receivers,
  h1 behind a slow T→h1 link (beta 1) and h2 behind a fast one. Use two chunks from g, with the
  epoch set so the slow link has beta 1. Check whether the MILP sends chunk 1 to h1 in epoch 0 and
  chunk 2 to h2 in epoch 1 on the same uplink. Under backpressure that is infeasible.
- **E4. Independent bottleneck evaluator.** Given extracted per-demand paths (or `Send`s once the
  switch-copy design lands), recompute timing under the bottleneck model:
  - a transmission occupies every link of its path from start until
    `start + chunk / min_bw + Σ alpha`;
  - transmissions sharing a link serialize.

  This is an event-driven replay, and it gives a ground-truth-ish makespan to compare with
  `3-Epochs_Required` without ns-3. A previous session's XML simulator (`sim_xml.py`, a scratchpad
  file, probably gone) did something similar from the XML and the switch-routing JSON.
  `milp_trace_test.validate_paths` is **not** an independent check here: it re-derives timing with
  the same offset rules as the MILP.
- **E5. Hierarchy.** Run `hierarchy_milp_test.test_end_to_end`'s HeteroTaperedCluster cases with
  `epoch_type=FASTEST_LINK` at the coarse level and see which of I4 / I6 fires.

## 6. Fix options (from the 2026-09-24 note, ordered by cost)

- **A. `beta_check()`** next to `alpha_check()` in `base_formulation.set_epoch_duration`. When
  `switch_pipeline` is on, assert or auto-coarsen so that min `epoch_capacity >= 1`. Free and
  sound; costs time resolution.
- **B. Refine the chunk instead of coarsening the epoch:** `chunk_size <= min_bw · epoch`, the
  `ChunkScale` idiom the hierarchy already uses. Keeps the fine clock, makes beta 0 by
  construction; costs model size.
- **C. Keep `beta` in both pipelined branches** (2 lines). Charges the SUM of betas, so it is sound
  but conservative. It is exact when only one leg on a path is narrow. It fixes I1's optimism. It
  does not address I2's coupling.
- **D. Exact bottleneck pacing.** Model the fabric between GPU pairs as one pipelined virtual
  channel with delay `1 + beta_bottleneck`, from a widest-path Floyd-Warshall over the switch-only
  subgraph, and keep per-link capacity rows for contention. Charging the bottleneck needs the path
  identity, which a per-link flow formulation does not carry. That is why C is the cheap sound
  option and D the faithful one. To also cover I2 and I5, the per-link windows would need to hold
  every link of a traversal for the bottleneck window.

Constraints any fix must respect:
- uniform-bandwidth and `SLOWEST_LINK` results must not change (they are immune today);
- the LP must stay untouched;
- every place that re-derives the offsets must change together: `_arrival_flow`, the switch send
  rule, `hop_timing`, `get_viable_predecessors`, `initialize_variables` pruning,
  `use_one_less_epoch`, the objective's windows (`allgather.py:555-558`), and the A\* variant.

## 7. Pointers

- Memory notes (in `~/.claude/projects/-Users-jacquelinewang-Desktop-TE-CCL/memory/`):
  `solver_switch_epoch_store_and_forward.md` (history, measurements, fix menu; its `alltoall.py`
  sections are stale), `switch_link_binary_cost_heterogeneous_bw.md`,
  `epoch_duration_fastest_link_not_required_for_correctness.md`, `switch_copy_send_design.md`.
- The design waiting on this: `teccl/hierarchy/switch_copy_design.md` (§3 "Spans are per LINK",
  §10 decision 2).
- Tests: `teccl/solvers/milp_trace_test.py`, `teccl/examples/hierarchy_milp_test.py`. Run with
  `PYTHONPATH=. conda run -n teccl python <file>` from the worktree root.
