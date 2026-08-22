# Post-solve port splitting at flow granularity

Realize one modeled link of bandwidth `B` as `P` parallel ports of `B/P`, assigning each
**flow** -- the routed path the solve already chose -- to one port on each link it crosses.

The solver, its capacity model, and the schedule's volumes / epochs / rates are untouched.
This is a relabeling of `via switches`, nothing more. Non-goals: changing routes, changing
the makespan, teaching the solver about ports.

## 0. Why this is not the solver's job

`capacity[i][j]` stays the AGGREGATE bandwidth, so every solve is bit-for-bit unchanged. The
solver's feasibility statement (`sum(rate) <= B` on a link, per epoch) is strictly weaker than
the realized one (`sum(rate) <= B/P` on each port), and the gap is exactly what this pass
closes. It closes it without inflating the makespan whenever the per-port packing succeeds --
which is a property of the flow-size vectors, not of the routing.

## 1. Port declaration

`Topology.ports[i][j]`, defaulting to 1 on every used edge, alongside `capacity`. Per-port
bandwidth is `capacity[i][j] / ports[i][j]`. `TwoPodRailHostBoundSplitPorts` sets
`ports[leaf][spine0] = 2`, making the fabric uniformly 25 GB/s per port at unchanged aggregate.

Unequal-width ports would need `ports` to hold a list of capacities instead of a count. Not
built; nothing needs it.

## 2. Ordering: sweep by HOP INDEX

Process hop 0 of every flow, then hop 1, then hop 2, ... Within a hop, every out-link is an
independent problem, because all of its inputs were decided at strictly earlier hops.

**Why hop order and not link order.** The port a flow occupies at hop `k` is decided by its own
hop `k-1`: leaving `u` on port `p` of link `(u,v)` IS arriving at `v` on port `p`. That
dependency lives on `(flow, hop)` pairs and always runs forward in `k`, so it is a disjoint
union of finite chains -- acyclic by construction. Aggregating it to the LINK level is what
creates cycles (a directed cycle in the node graph realized by consecutive hops), and link
order is therefore not always available. Hop order always is.

There is no cross-flow dependency: a relay GPU consumes the data and re-sends it as a new
record whose hop 0 originates in memory, with no inbound port to inherit.

**Price.** A link used at several hop indices is packed INCREMENTALLY across passes rather than
jointly: pass `k` fills into the residual left by passes `< k`, and heavy-first is only
available within a pass. Capacity stays hard-enforced (residuals carry forward), but the
packing is greedier. Measured on `two_pod_rail_hostbound_allgather_fast_epoch_flat.json`: 16 of
80 links are used at more than one hop index (all `leaf->GPU`, at hops 1 and 3 -- intra-pod vs
cross-pod delivery), and all 16 are single-port, so nothing there carries a decision. All 8
links that would be split are single-hop-index. The incremental case is reachable but untested
here; `HeteroTaperedCluster` is where it would first bite.

**Guard.** Assert each record's switch path is simple (no repeated node), so hop index is
strictly increasing along a record. A hairpin `...u->v->u...` inside one record would break the
ordering argument, and it is already an upstream invariant -- `flat_schedule` raises on a cycle
while tracing a demand back to its source.

## 3. The local problem at one (link, hop)

Fully local. Both ends of every port pair are the same switch, and a switch crossbar imposes
nothing across them, so there is no coupling between a flow's choice at one hop and the next.

> **Given** out-link `b` with `P` ports of per-epoch capacity `C(e)`, a residual `used[q][e]`
> carried in from earlier hops, and flows `F` at this hop each with a fixed in-port label
> `rho(f)` and a per-epoch load vector `n_f(e)`:
>
> **choose** `q(f)` such that `for all e, q:  used[q][e] + sum_{q(f)=q} n_f(e) <= C(e)`
>
> **minimizing** (1) flows split across ports, then (2) distinct `(rho(f), q(f))` combos.

Load spread across ports is deliberately NOT an objective. It was in an earlier draft as a
proxy for downstream feasibility, and that proxy is wrong: whether a downstream link can pack
its flows depends only on its own flow set and capacities, both fixed by the solve. In-port
labels enter only the bucket heuristic and the combo count. Optimizing for spread costs real
splits (see section 4) and buys nothing.

### It is VECTOR bin packing

Capacity is per epoch and epochs are independent pools -- a send is epoch-aligned and its rate
fills exactly one epoch, so a piece consumes only its own epoch's capacity. But one port
decision covers a flow's whole lifetime. So the cost is a vector and the decision is a scalar:
`d` simultaneous knapsacks over shared variables, `d` = number of epochs.

That is not a technicality. On the reference schedule the five profile classes are not
proportional, and two pairs are perfectly disjoint in time (`(0,0,0,6,0,3)` never coexists with
`(3,3,3,0,3,0)` or with `(9,9,9,0,9,0)`), so they can share a port for free. Consequences:

* **Order by PEAK relative load, not total.** A port is bound by its worst epoch. Measured
  inversion: `(0,0,0,6,0,3)` is 5th of 5 by total (9 pieces) but 3rd by peak (6); `(3,3,3,0,3,0)`
  is 4th by total (12) but 5th by peak (3).
* **Fit is a max over epochs, not a sum**, evaluated per candidate port -- the binding epoch
  differs per port as packing proceeds. `max_e (used[q][e] + n_f(e)) / C(e)`. This gives
  complementarity for free: co-locating `(9,9,9,0,9,0)` with `(0,0,0,6,0,3)` scores 9/48, two
  9-profiles score 18/48.
* **Keep `d` small.** Work on the coarse epoch grid the flows are paced to (8 here, not the
  12145 fine epochs), then drop epochs where the link is not near capacity -- they constrain
  nothing. `L0->S0` has 5 binding epochs, not 8.

## 4. The algorithm

> **Bucket flows by in-port. Process buckets heavy-first by peak relative load. Place into the
> FULLEST port that still fits (best-fit).**

1. `R[rho] = flows at this hop arriving on in-port rho`.
2. Sort buckets by `max_e (sum_{f in R} n_f(e)) / C(e)`, descending.
3. Place the whole bucket on the fullest port that still fits -> 1 combo. Ties broken by
   `rho.port % P`, then lowest index, for determinism across runs.
4. If the bucket does not fit whole, split it AT FLOW GRANULARITY: sort its flows by peak
   descending and place each by the same rule.
5. A flow that fits nowhere escapes to a piece-level split (section 5).

### Why best-fit and not emptiest-fit

Emptiest-fit is worst-fit: it fragments, leaving every port partially full so nothing large
fits later. That is fatal across hops, where a later pass needs whole ports. Two half-capacity
buckets at hop 1 land on ports 7 and 8 under emptiest-fit and a full-port flow at hop 3 must
then be split; under best-fit they compact onto port 7 and the hop-3 flow takes port 8 intact.

Compaction also wins on the combo objective, on both terms: fewer splits (each split bucket is
+1 combo), and fewer ACTIVE in-ports downstream, which lowers the combo floor
`max(active_in, active_out)`.

The honest counter-case: compaction concentrates a bucket, and a concentrated in-port bucket
can exceed a downstream out-port's capacity and be forced to split there. That is
combos-vs-combos, not combos-vs-feasibility, and it is self-limiting at equal port widths,
where a compacted in-port holds exactly one port's worth and fits exactly one out-port.

### Why heavy-first is still needed

Best-fit chooses well only among ports that ALREADY fit; it cannot undo having partly filled
every port before a large item arrives. Two ports of 10 and items {4,4,6,6}: arrival order
4,4,6,6 forces a split, decreasing order 6,6,4,4 does not. Measured on random instances shaped
like this one (sparse epoch profiles, in-port buckets), mean forced splits:

    utilisation  ports | best-first-heavy | best-arbitrary
           0.70      4 |             1.14 |           1.14
           0.85      2 |             0.49 |           0.75
           0.95      2 |            12.78 |          20.02
           0.95      4 |            19.20 |          28.14

~30% fewer splits at 0.95, nothing at 0.70. These links run at 100%, so that is the regime.
Sorting is free here: at a given (link, hop) every bucket is in hand at once, so there is no
arrival order to be online against.

`first-fit` by lowest index scores within noise of best-fit (12.03 vs 12.78 at 0.95/2 ports,
20.95 vs 19.20 at 0.95/4) and is a one-line swap if determinism ever matters more than the
marginal packing.

## 5. Fallback

Flow granularity is bin packing and can fail even when the aggregate fits. Pieces are atomic
and identical, so PIECE granularity is always feasible when `sum <= P*C`. So: attempt flow
granularity, and where a flow genuinely does not fit, split its pieces across ports, log the
flow, and accept +1 path key on that `(src, dst)` edge. Never inflate the makespan instead: a
split costs one channel against a cap of 32, an overloaded port costs the makespan.

Expected on the reference schedule: ZERO splits.

## 6. Emission -- how the port reaches the program

The port is folded into the path key AT PARSE TIME (`parse_flows` / `parse_flows_lp` take a
`port_qualify` callable), not rewritten afterwards. That is load-bearing: `parse_flows_lp` also
builds `paced_sends` from the path key, and the pacing gate manifest is keyed on
`(step, gpu, peer, path_key)`. Rewriting the keys after parsing would leave the gates keyed on
the unqualified form and `_realize_pacing_gates` would silently match nothing.

The qualified key is `(switches, ports)`, with one port per HOP -- one longer than `switches`,
since hop i is the link INTO switch i. `qualify_path_key` returns the key UNCHANGED when the
route touches no multi-port link, so a topology that declares no ports (every topology today)
produces byte-identical output through every consumer. `unqualify_path_key` is the one place
that knows the shape.

Two consumers need it, and both get it for free from the key:

* **Channels.** Allocated per `(srcGPU, dstGPU)` EDGE
  (`_allocate_channels_match_topology`: `chan = path_idx * link + rr`, `path_indices` a dict per
  edge). Each flow keeps exactly one port-qualified tuple, so the per-edge key count is
  unchanged: measured 104 edges with 1 key, 64 with 2, against `max_channels = 32`, identical
  before and after. Only a SPLIT flow adds one. This is why in-port affinity is a tiebreak and
  not an objective -- it was never buying channels.
* **Forwarding.** A flow id is a bijection with `(src, dst, path_key)`, so two flows down the
  same switch sequence on different ports now get different flow ids and therefore different
  entries. `build_switch_routes` emits `next_hop_port` on each entry: for a programmable switch
  at original index i, that is `ports[i+1]`, the port of the hop LEAVING it. The original index
  is kept while filtering non-programmable switches, because chaining past a self-routing switch
  changes the next hop, not the wire the packet leaves on. The key is added only when the route
  carries ports, so an unsplit table is byte-identical.

## 7. Validation

* Per-PORT per-epoch capacity assert (the port-granular sibling of
  `flat_schedule.assert_link_capacity`).
* Makespan unchanged; per-`(src,dst)`-edge distinct-path-key count unchanged except by logged
  splits.
* Regression on `two_pod_rail_hostbound_allgather_fast_epoch_flat.json`: exact 48/48 on all 8
  directed spine0 links in coarse epochs 0,1,2,3,5 and 24/24 in epoch 6; zero flow splits; 4
  combos per link, at the floor `max(4, 2)`.
* Unit tests for the two rule choices, since the reference schedule cannot discriminate them
  (at 100% utilization all fit rules and both orderings give identical output there):
  the fragmentation case (best-fit vs emptiest-fit across hops) and the {4,4,6,6} case
  (heavy-first vs arbitrary).
* Negative control: a random per-`(flow, link)` port hash is 0/20000 feasible, median max port
  load 72/48 = **1.50x** makespan inflation. The pass must produce <= 48.
* Emission: channels per edge, route count and pacing-gate count all unchanged, and every
  `next_hop_port` in the forwarding table matches its route's own port tuple.

## 8. Build order

1. DONE -- `ports` on `Topology` + `TwoPodRailHostBoundSplitPorts`, defaults 1 everywhere.
   `capacity` and `alpha` verified byte-identical against the unsplit class.
2. DONE -- `port_split.py`: hop sweep + local solver, plus `occupancy_grid` so the packing grid
   is derived from the schedule rather than supplied.
3. DONE (go/no-go) -- port-granular capacity assert. Reference schedule: exact 25/25 GB/s per
   port on all 8 spine0 links, 0 flow splits, 4 combos per link (the floor).
4. DONE -- port folded into the path key at parse time. Channels per edge `{1: 104, 2: 64}`,
   route count and gate count all unchanged; forwarding table gains `next_hop_port` and changes
   in no other way.
5. DONE -- `two_pod_rail_allgather_flat` is the spine-bound (`GPU_LEAF_BW = 50`) instance and IS
   paced, so no new solve was needed. It is the only case that exercises the packing: with
   slack on the GPU->leaf links one ingress bucket outgrows a 25 GB/s port and is broken up at
   flow granularity, giving 5 combos on each leaf->spine0 link -- the transportation bound
   `r + c - 1 = 4 + 2 - 1` -- against 4 in the host-bound config. Still zero flow splits.
   All six paced two_pod_rail schedules (both collectives, both configs, both epoch
   granularities) split with zero flow splits and exact per-port balance.

## 9. What is NOT covered

The fit rule and the bucket ordering are still only discriminated by the synthetic tests. The
spine-bound instance exercises the bucket-split path, but every spine0 link in every schedule
here runs at exactly 100%, so per-port balance is forced by capacity and no fit rule can do
worse. A topology with sustained partial utilisation on a multi-port link -- or a link used at
several hop indices AND declaring more than one port, which none of these have -- is what would
first distinguish them on real data.
