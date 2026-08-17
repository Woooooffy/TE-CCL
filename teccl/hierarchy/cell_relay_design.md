# Identity resolution for CELL RELAY (host transit)

Design note for the gap documented in `reconstruct._origin_diagnosis` and
`bands.band_of`: a coarse path that store-and-forwards through an intermediate
CELL, `A -> sw -> B -> sw -> C`, where `B` is a transit host that neither
produces nor wants the data.

## 1. Why it breaks today

Two independent assumptions, both in `assign_identities_free`:

* `_extract_pieces` walks each coarse path and emits one `_CoarsePiece` per
  maximal switch-run, keyed by its PHYSICAL endpoints. A transit path yields
  legs filed under `(A,B)` and `(B,C)`, both carrying `origin=(A,C)`. The leg
  is the unit of assignment, so the two halves of one delivery are never
  related to each other.
* `identity_sets` is demand-driven: `B` wants none of `A`'s data, so
  `id_sets[(A,B)]` does not exist and `id_sets[(A,C)]` has no pieces.

Result is fail-loud, in one of two places: `no coarse pieces for demanded pair
(A,C)`, or the `total_cap == len(identities)` balance assert on `(A,B)` (which
now carries foreign volume).

The key observation is that **the leg is the wrong unit**. Everything below
follows from changing the unit of assignment from a leg to a whole coarse
route, keyed by its LOGICAL origin.

## 2. The reformulation: assign identities to ROUTES

```python
@dataclass(frozen=True)
class _CoarseRoute:
    origin: Tuple[int, int]        # (logical source cell, logical dest cell)
    legs: Tuple[_CoarsePiece, ...] # >=1, chained: legs[i].dst_cell == legs[i+1].src_cell
    volume: float
```

`_extract_pieces` becomes `_extract_routes`: the switch-run walk it already
does is per `each_path`, and successive runs of ONE path are exactly the legs
of one route — the chaining is free, no reassembly heuristic needed. Legs of a
path share a volume; take `min` and assert agreement within `EPS`.

Routes are keyed by `origin`, not by physical endpoints. That single change
makes the demand anchor hold again:

    sum(route.volume for route in routes_by_pair[(U,V)]) == coarse[U][V] == |ID(U,V)|

and it means **`identity_sets` needs no change at all**. This is worth stating
explicitly, because the previously sketched fix was to "track transit
identities in identity_sets"; doing that would have put identities into
`id_sets[(A,B)]` that no GPU in `B` wants, breaking the
`len(id_sets[(U,V)]) == coarse[U][V]` mirror of `coarsify_demand` that is the
correctness anchor of this whole module. Route-keying gets the same result
without touching the demand side.

Non-transit topologies are unaffected: every route has exactly one leg and the
program is character-for-character the one that runs today.

## 3. Slots: one gateway choice per leg

A slot today is `(piece, egress gateway)` with

    capacity = piece.volume * cap_g / cap_sum

and the proportional split of EVERY piece is what makes the per-(U,V)
decomposition sound (per-epoch load on gateway `g` is bounded by coarse
feasibility, by construction, without a global constraint).

Generalize a slot to `(route, gateway tuple)` — one egress gateway per leg —
with

    capacity = route.volume * prod over legs (cap_{g_leg} / cap_sum_leg)

This is the only generalization that preserves the soundness argument: summing
the product over all combinations containing `g` at leg `i` gives back exactly
`route.volume * cap_g / cap_sum` at that leg, so the per-fine-uplink per-epoch
bound holds at EVERY leg, not just the first. Column count is
`prod |gateways(leg)|`; guard it with a budget (e.g. 64 columns per route) and
raise rather than silently degrade — single-homed and dual-homed cells, which
is everything realistic, stay at 1–4.

`ingress_candidates` becomes per-leg as well; `_pick_ingress` runs once per leg
against the same global ledger, so a transit cell's downlink is budgeted
exactly like a destination cell's.

## 4. Cost function: transit relay is HARD, and joins tier 1

Current tiers: (1) egress relay at the source cell, big-M; (2) ingress relay at
the destination, weight 1; (3) epoch preference for relayed identities, tiny.

A transit cell adds one more forced-work term: if the GPU that LANDS leg `i` is
not the GPU that EGRESSES leg `i+1`, an intra-cell relay inside `B` is
required, and that relay is HARD — a network send is waiting on it, exactly
like `egress_stage` at the source. So it belongs in **tier 1**, summed with the
source egress term:

    tier1(slot, d) = W_EGRESS * ( [g_0 != holder(d)]
                                + sum over i of [gateways(leg i+1) disjoint from
                                                 ingress_candidates(leg i)] )

Priced against the CANDIDATE SETS, not against a chosen GPU — the same
technique tier 2 already uses, because `_pick_ingress` runs after the LP. Then
`_pick_ingress` realizes what the LP priced, by generalizing its notion of a
preferred landing GPU:

* at the final cell: prefer a GPU in `targets[(identity, V)]` (today's rule);
* at a transit cell: prefer the next leg's egress gateway.

One parameter change — pass `preferred: Set[int]` instead of consulting
`target_gpus` internally — and the transit relay disappears entirely whenever
the boundary permits it. On a bridged-island topology where one host owns both
uplinks, transit costs zero intra work.

At a transit cell this preference is in fact a REQUIREMENT under a default
coarse solve, for timing reasons — see §6, which qualifies this section.

`W_EGRESS = nD + 1` must become `W_EGRESS = (nD + 1) * (max legs)` (or simply
scale by the max relay count per route) so tier 1 still strictly dominates the
sum of all tier-2 and tier-3 costs.

## 5. Representing the transit relay: `egress_stage` + an explicit release

The transit relay is not a new kind. `kind` is set by the cell's ROLE for that
piece, and `B`'s role is "sender of the outgoing leg" — that is `egress_stage`,
with `src_gpu` = the landing GPU rather than the native holder:

```python
IntraCellDemand(cell=B, kind="egress_stage", identity=sub,
                src_gpu=ingress_gpu(leg i), dst_gpus=(egress_gpu(leg i+1),),
                volume=1.0, deadline_epoch=leg[i+1].send_epoch,
                release_band=leg[i].arrival_epoch + 1, hard=True)
```

The one thing the current record cannot express is that the data is not there
yet. Today `IntraCellDemand.deadline_epoch` doubles as a release for
`ingress_distribution` (`bands.release_of` returns `deadline_epoch + 1`) and as
a deadline for `egress_stage`. A transit demand needs BOTH numbers, so:

* add `release_band: Optional[int] = None` and `hard: Optional[bool] = None` to
  `IntraCellDemand`;
* `bands.release_of` returns `demand.release_band` when set, else today's
  kind-based rule;
* `bands.assign_bands` already reads `hard` off the demand via `getattr` and
  its docstring already anticipates this caller — no change needed there;
* `crossbar_solve._to_jobs` hardcodes `release=PROLOGUE_BAND` for
  `egress_stage` (line ~345). It must call `bands.release_of` instead. This is
  load-bearing: left as is, the child level would schedule the forwarding relay
  in the prologue, before the data has arrived.
* `_coalesce_egress` merges `egress_stage` by `(cell, identity, src, dst)`
  keeping the MIN deadline; it must also keep the MAX release, and raise if the
  merged window is empty.

Downstream needs nothing else. Each leg is already its own `ResolvedPiece`, so
`_assert_rate_within_capacity`, `pieces_as_flows` and the flat stitch see two
ordinary network flows whose ordering is enforced by the intervening demand's
band. `LevelDemand.from_demands` also already handles it: at `B` the identity's
only source is the landing GPU and it is nobody's destination, so the
chain-collapse picks it as holder.

## 6. Timing: co-location is REQUIRED, not preferred

The coarse solve already models a cell as a store-and-forward host. At a
non-switch node, `lp_formulation` (the `midFC` constraint, ~line 239) lets flow
arriving in epoch `k` leave in epoch `k+1`: a one-coarse-epoch dwell, applied
to exactly the traffic a transit cell forwards. So

    leg[i+1].send_epoch >= leg[i].arrival_epoch + 1

holds by construction, and needs no check.

The gap is narrower than a missing dwell, and it is an ABSTRACTION artifact.
That +1 models a host that receives and re-sends **on the same port**. The
abstraction collapsed a cell's GPUs into one coarse node, so when the landing
GPU and the outgoing gateway differ, the intra-cell hop between them is
invisible to the coarse level and unbudgeted. With sends at the leading edge of
a band, a piece landing at the end of epoch `k` is ready in band `k+1` and can
only feed a send from epoch `k+2`. So:

| transit | needs | coarse solve gives | verdict |
| --- | --- | --- | --- |
| co-located (one GPU owns both links) | arrival + 1 | arrival + 1 | feasible, zero intra work |
| relayed (landing GPU != gateway) | arrival + 2 | arrival + 1 | INFEASIBLE |

This inverts §4's framing. Eliminating the transit relay by co-locating ingress
and egress is not a preference the lexicographic objective expresses — under a
default coarse solve it is the **only feasible option**, so it is a hard
constraint on `_pick_ingress` at a transit cell: land the piece on a GPU that
also owns the outgoing uplink, or fail. Capacity may make that impossible (all
transit volume forced onto the bridging GPUs, exceeding their per-epoch
downlink); the ingress ledger already fails loud on it, and that failure is a
true statement about the topology.

The tier-1 transit term in §4 therefore prices something that can no longer be
chosen freely. Keep it — it still steers the assignment toward routes whose
transit happens to be co-located when several are available — but the emitted
`egress_stage` at a transit cell (§5) is reachable only under the two-epoch
dwell below.

### Cells with no bridging GPU

A host whose two NICs hang off different GPUs — the realistic bridged-host case
— has no co-located transit at all, and no amount of work in identity
resolution fixes it. The coarse level must budget the second epoch, and it can
know which cells need it BEFORE solving: `abstract()` has `boundary_gpu`, so
for each cell and each (in-neighbor, out-neighbor) pair it can ask whether some
GPU owns both links. Declare a per-cell forwarding dwell (1 if every pair
bridges, else 2 — per-node is slightly conservative versus per-pair and much
easier to express) and have the coarse formulation use it for FORWARDED flow.

Implementation subtlety worth stating up front: the arrival term in `midFC`
feeds three things at once — the buffer, the outflow, and `consumed_at_k`.
Shifting the arrival index by one to delay forwarding would also delay
CONSUMPTION at every destination cell, a pessimism for the whole solve rather
than for transit. So forwarding must be separated from consumption (a distinct
forwardable term, or a second constraint), not implemented by re-indexing the
existing one.

### The alternative, and why not

The other way out is to relax the leading-edge rule so a relay can run in the
first few rounds of band `k+1` and feed a send in that same band — tempting,
since the intra fabric is far faster than the network and a coarse epoch is
sized to the network. It does not work without chunk-level pipelining: the
piece is paced across the whole epoch (`_piece_rate`), so its first bytes must
be ready at the band's leading edge whatever the relay costs. It would also
give up the invariant that makes each level's job a plain makespan problem
(`bands.py`). Rejected, but this is the direction to revisit if transit becomes
common enough that the extra epoch is worth pipelining for.

## 7. Checks to add

* per route: legs chain (`dst_cell == next.src_cell`), epochs strictly
  increase, no cell repeats — an LP flow decomposition can emit a cycle, and a
  route that revisits a cell must fail loud rather than deadlock a child level.
* per `(U,V)`: `sum(route.volume) == |ID(U,V)|` (the existing balance assert,
  now over routes; `_origin_diagnosis` becomes a debugging aid rather than a
  gap explainer and can keep its DISPLACED/FOREIGN reporting for genuinely
  malformed extractions).
* the existing replay-test invariants extend unchanged, since delivery coverage
  already computes "required = max over demands, delivered = total inbound", so
  a two-leg chain counts correctly.

## 8. Test plan

No current topology forces transit — rail and hetero both reach every peer
through switches, which is why this has stayed dormant. A forcing fixture is
needed:

* **`BridgedIslandsCluster`**: two switch islands with NO switch-to-switch
  link, bridged only by one dual-homed host `B`. Then `A -> C` has no route
  except `A -> T0 -> B -> T1 -> C`, and every cell is a transit cell for some
  pair. Two variants: one GPU owning both uplinks (co-located transit, zero
  intra work) and two distinct boundary GPUs (relay FORCED).
  **The co-located variant is the one this design can carry end to end.** The
  forced variant is blocked on the per-cell forwarding dwell of §6 — without it
  the coarse solve emits a transit the fine level provably cannot implement, so
  it is a test of the failure message until that lands, not of the relay path.
* Gurobi-free unit fixtures in `hierarchy_identity_resolution_test.py`:
  hand-built two-leg routes asserting (i) the transit `egress_stage` is emitted
  with the right release/deadline pair, (ii) zero transit relays in the
  co-located variant, (iii) tier-1 dominance when a transit relay trades
  against an ingress saving.
* extend `hierarchy_pipeline_replay_test.py` with a transit route so the
  cross-stage invariants (coverage, capacity both ends, rounds <= m) are
  checked end to end.

## 9. Change list

| file | change |
| --- | --- |
| `reconstruct.py` | `_CoarseRoute`; `_extract_pieces` -> `_extract_routes` keyed by origin; `_build_slots` over gateway tuples with product capacity + column budget; `_solve_assignment` tier-1 sums transit relays, `W_EGRESS` rescaled; `_pick_ingress` takes `preferred`; `_Assignment` holds a route and emits one `ResolvedPiece` per leg plus the transit demands; `_coalesce_egress` merges releases; route validity checks |
| `bands.py` | `release_of` honours an explicit `release_band` |
| `crossbar_solve.py` | `_to_jobs` uses `bands.release_of` instead of hardcoding `PROLOGUE_BAND` for `egress_stage` |
| `topologies/` | `BridgedIslandsCluster` fixture (two variants) |
| tests | transit fixtures in the resolution and pipeline-replay tests |
| `abstract.py` | per-cell forwarding dwell derived from `boundary_gpu` (§6) — only for the no-bridging-GPU case |
| `lp_formulation.py` | honour that dwell for FORWARDED flow, without delaying `consumed_at_k` (§6) — same |

Sequencing: §5 (record + release plumbing) and §8's co-located fixture are
independent of §2–§4 and can land first, each with its own test; the route
reformulation is one commit after that, and is a no-op on every existing
topology. Together those cover co-located transit completely. The last two rows
are a separate, larger piece of work in the coarse formulation and are the only
thing standing between this design and the forced-relay case — they should not
be bundled in.
