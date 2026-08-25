# Identity resolution for CELL RELAY (host transit)

Design note for the gap documented in `reconstruct._origin_diagnosis` and
`bands.band_of`: a coarse path that store-and-forwards through an intermediate
CELL, `A -> sw -> B -> sw -> C`, where `B` is a transit host that neither
produces nor wants the data.

> **Revision note.** §3 and §4 were rewritten. The earlier draft assigned
> identities to `(route, gateway TUPLE)` slots, pricing every leg's gateway
> inside one LP. That is rejected here: the downstream gateway choice is
> identity-INDEPENDENT (§6), so putting it in the identity LP pays a
> combinatorial product for nothing. The unit of assignment still becomes the
> route (§2 stands, and is load-bearing), but the LP still sees ONE gateway per
> slot and the downstream legs are placed by a forward pass afterwards. §5, §6
> and §7 are unchanged and are common to both variants.

## 1. Why it breaks today

Two independent assumptions, both in `assign_identities_free`:

* `_extract_pieces` (`reconstruct.py:219`) walks each coarse path and emits one
  `_CoarsePiece` per maximal switch-run, keyed by its PHYSICAL endpoints. A
  transit path yields legs filed under `(A,B)` and `(B,C)`, both carrying
  `origin=(A,C)`. The leg is the unit of assignment, so the two halves of one
  delivery are never related to each other.
* `identity_sets` is demand-driven: `B` wants none of `A`'s data, so
  `id_sets[(A,B)]` does not exist and `id_sets[(A,C)]` has no pieces.

Result is fail-loud, in one of two places: `no coarse pieces for demanded pair
(A,C)`, or the `total_cap == len(identities)` balance assert
(`reconstruct.py:1130`) on `(A,B)` (which now carries foreign volume).

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

### 2.1 The chaining is EXACT, not reconstructed

This is the fact the whole design rests on, so it is worth pinning down.
`per_chunk_flow_paths[(s,d,c)]` entries are WHOLE source-to-dest paths:
`dig_to_source` (`solvers/lp_formulation.py:812`) back-traces from the
destination all the way to the source, and files one complete path at
`lp_formulation.py:872-878`. `_extract_pieces`'s switch-run walk
(`reconstruct.py:246`, the `while start < len(chunk_path)` loop) then iterates
successive runs of ONE such path and drops the relation between them three
lines after having it.

So "which arriving leg at `B` feeds which departing leg at `B`" is not missing
information to be recovered — it is information currently being discarded.
`_extract_pieces` becomes `_extract_routes` and simply stops discarding it: the
chaining is free, no reassembly heuristic and no matching pass. Legs of a path
share a volume (`dig_to_source` decomposes at the path bottleneck); take `min`
and assert agreement within `EPS`.

**Rejected alternative — deriving transit identities by matching.** The
tempting cheaper-looking route is to leave `_extract_pieces` alone, resolve only
the identities a cell natively holds, and then at each transit cell `B` match
what landed against what departs. It is worse on three counts, all avoidable:

* it is a MATCHING, not a derivation — every identity in the pool `ID(A,C)` is
  interchangeable at `B`, so something must choose;
* it takes on a temporal proof obligation the route form never incurs: an
  identity arriving in epoch `k` can only feed a send in epoch `>= k+1`.
  Feasibility does hold (the coarse LP's per-commodity per-epoch conservation is
  exactly the cumulative staircase condition, so earliest-arrival-first greedy
  realizes it) but that argument has to be made and tested;
* it fights `_emit_refined` (`reconstruct.py:985`). Sub-chunk indices are
  allocated per `(identity, dst_cell)` and the `cursor == q` partition check is
  this module's integrity anchor. A transit chain needs the SAME `sub` index on
  both legs. With routes, one `_Assignment` emits one `ResolvedPiece` per leg
  sharing that index and the check is untouched; with independently assigned
  legs the matching would have to run post-refinement at whole-sub-chunk
  granularity, or chain indices across separately refined groups.

### 2.2 Why origin-keying is what fixes the anchor

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

## 3. Slots: still ONE gateway, chosen on the FIRST leg

A slot today is `(piece, egress gateway)` with

    capacity = piece.volume * cap_g / cap_sum

and the proportional split of EVERY piece is what makes the per-(U,V)
decomposition sound (`_Slot`'s docstring, `reconstruct.py:305`): per-epoch load
on gateway `g` is bounded by coarse feasibility, by construction, without a
global constraint.

**Keep that.** A slot becomes `(route, first-leg egress gateway)` with the
capacity formula unchanged — it reads `legs[0]` where it used to read the only
leg. LP width is unchanged, `_build_slots` (`reconstruct.py:325`) keeps its
tested soundness argument, and there is no column budget to guard.

### 3.1 Why not gateway tuples

The generalization to `(route, gateway tuple)` with
`capacity = route.volume * prod over legs (cap_{g_leg} / cap_sum_leg)` is sound
— summing the product over all combinations containing `g` at leg `i` gives
back exactly `route.volume * cap_g / cap_sum` at that leg — but it costs a
`prod |gateways(leg)|` column count needing a budget-and-raise, and it replaces
an accounting argument with an averaging one.

It buys nothing, and §6 is why: whether leg `i`'s landing GPU co-locates with
leg `i+1`'s egress gateway is a property of THE PIECE AND THE TOPOLOGY, not of
which chunk rides it. Every identity on that route pays the same transit cost.
Pricing an identity-independent decision inside the identity LP is a
combinatorial product spent on a term that is constant across the rows it
would discriminate between.

### 3.2 Downstream legs: a forward pass against an EGRESS ledger

After the LP has assigned identities to `(route, first gateway)`, walk each
route's remaining legs in order and place them per PIECE:

* leg `i+1`'s egress gateway: prefer the GPU that landed leg `i` — co-location,
  which §6 shows is a hard requirement under a default coarse solve, not a
  preference;
* leg `i`'s landing GPU: `_pick_ingress` (`reconstruct.py:455`) with
  `preferred = {leg i+1's egress gateway}` at a transit cell, and today's
  `target_gpus` rule at the final cell. One parameter change — pass
  `preferred: Set[int]` instead of consulting `target_gpus` internally.

Budget that placement against an **egress ledger mirroring the existing
`ingress_ledger`** (`reconstruct.py:1093`): same key shape
`(gpu, neighbor, epoch)`, same running-room comparison, same best-fit tie-break,
same amplified `ingress_tol`. This is the honest version of what §3.1's product
formula approximates — the product bounds the expected per-epoch load, the
ledger accounts for the actual one — and it reuses machinery that is already
written, already tested, and already carries the float-noise handling this
chain needs.

### 3.3 What this decomposition gives up

Exactly one thing, and it is worth writing into the `assign_identities_free`
docstring: when a commodity `(A,C)` SPLITS across routes with different
downstream ingress costs at `C`, the first-leg LP chooses identities blind to
that difference.

When a commodity uses a single route shape it is exactly optimal — the whole
pool `ID(A,C)` transits `B`, so the final-cell landing preference is fully
recoverable by the forward pass. And the loss is confined to tier 2, which is
SOFT (see §4). It can never cost feasibility, only an avoidable intra-cell hop
at the destination.

## 4. Cost function: one line changes

Current tiers in `_solve_assignment` (`reconstruct.py:360`): (1) egress relay at
the source cell, big-M; (2) ingress relay at the destination, weight 1; (3)
epoch preference for relayed identities, tiny.

Under §3 the only change is **which leg tier 2 reads**:

* tier 1 is the source-cell egress relay, `legs[0].egress_gpu != holder(d)` —
  unchanged, and `W_EGRESS = nD + 1` needs no rescaling, because no per-leg
  transit term is being summed into it;
* tier 2 is priced against `legs[-1].ingress_candidates` rather than the only
  leg's — the route knows its last leg, so this is a one-line change. It stays
  priced against the CANDIDATE SET, not a chosen GPU, exactly as today, because
  `_pick_ingress` still runs after the LP;
* tier 3 is unchanged.

The transit relay does NOT appear in the objective. §6 makes it infeasible
rather than costly, so it is a constraint on the §3.2 forward pass ("land on a
GPU that also owns the outgoing uplink, or fail"), not a term to trade against
ingress savings.

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
`ingress_distribution` (`bands.release_of`, `bands.py:75`, returns
`deadline_epoch + 1`) and as a deadline for `egress_stage`. A transit demand
needs BOTH numbers, so:

* add `release_band: Optional[int] = None` and `hard: Optional[bool] = None` to
  `IntraCellDemand`;
* `bands.release_of` returns `demand.release_band` when set, else today's
  kind-based rule;
* `bands.assign_bands` already reads `hard` off the demand via `getattr` and
  its docstring already anticipates this caller — no change needed there;
* `crossbar_solve._to_jobs` hardcodes `release=PROLOGUE_BAND` for
  `egress_stage` (`crossbar_solve.py:345`, and the kind test at `:356`). It must
  call `bands.release_of` instead. This is load-bearing: left as is, the child
  level would schedule the forwarding relay in the prologue, before the data has
  arrived.
* `_coalesce_egress` (`reconstruct.py:1303`) merges `egress_stage` by
  `(cell, identity, src, dst)` keeping the MIN deadline; it must also keep the
  MAX release, and raise if the merged window is empty.

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

This is what makes the transit relay a CONSTRAINT rather than a cost, and it is
why §3.2 places downstream legs by co-location and §4 leaves the objective
alone. Capacity may make co-location impossible (all transit volume forced onto
the bridging GPUs, exceeding their per-epoch downlink); the ingress ledger
already fails loud on it, and that failure is a true statement about the
topology. The `egress_stage` at a transit cell (§5) is therefore reachable only
under the two-epoch dwell below.

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
  increase, **no cell repeats**. Prioritize this one. `dig_to_source` carries a
  path-local cycle-avoidance band-aid rather than a real time-expanded flow
  decomposition, so a cyclic path is a live possibility, and a route that
  revisits a cell must fail loud rather than deadlock a child level.
* per `(U,V)`: `sum(route.volume) == |ID(U,V)|` (the existing balance assert at
  `reconstruct.py:1130`, now over routes; `_origin_diagnosis` becomes a
  debugging aid rather than a gap explainer and can keep its DISPLACED/FOREIGN
  reporting for genuinely malformed extractions).
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
  co-located variant, (iii) both legs of one route carry the SAME sub-chunk
  index out of `_emit_refined` and the `cursor == q` partition check still
  holds, (iv) the §3.2 egress ledger refuses a second route whose co-located
  gateway is already saturated in that epoch, rather than silently
  oversubscribing it.
* extend `hierarchy_pipeline_replay_test.py` with a transit route so the
  cross-stage invariants (coverage, capacity both ends, rounds <= m) are
  checked end to end.

## 9. Change list

| file | change |
| --- | --- |
| `reconstruct.py` | `_CoarseRoute`; `_extract_pieces` -> `_extract_routes` keyed by origin (stop discarding the path's own chaining, §2.1); `_build_slots` reads `legs[0]`, capacity formula UNCHANGED; `_solve_assignment` tier 2 reads `legs[-1].ingress_candidates`, weights unchanged; `_pick_ingress` takes `preferred`; forward pass placing legs 1..n against a new egress ledger; `_Assignment` holds a route and emits one `ResolvedPiece` per leg (all sharing the sub-chunk index) plus the transit demands; `_coalesce_egress` merges releases; route validity checks |
| `bands.py` | `release_of` honours an explicit `release_band` |
| `crossbar_solve.py` | `_to_jobs` uses `bands.release_of` instead of hardcoding `PROLOGUE_BAND` for `egress_stage` |
| `topologies/` | `BridgedIslandsCluster` fixture (two variants) |
| tests | transit fixtures in the resolution and pipeline-replay tests |
| `abstract.py` | per-cell forwarding dwell derived from `boundary_gpu` (§6) — only for the no-bridging-GPU case |
| `lp_formulation.py` | honour that dwell for FORWARDED flow, without delaying `consumed_at_k` (§6) — same |

`assign_identities_preserving` needs a mechanical touch only: it builds
`_Assignment`s directly, so it constructs single-leg routes. `make_piece` is
unaffected; a one-leg route around it keeps that path character-for-character
what it is today.

Sequencing:

1. §5 (record + release plumbing) — independent of everything else, own test.
2. §8's co-located `BridgedIslandsCluster` fixture, asserting the current
   failure message. Pins the gap before changing it.
3. §2 route-keyed extraction + §7 checks. A no-op on every existing topology
   (single-leg routes), so it lands green on its own.
4. §3.2 forward pass + egress ledger, and §4's one-line tier-2 move. This is
   what closes the co-located case.
5. The last two rows of the table — the coarse forwarding dwell — are a separate,
   larger piece of work in the coarse formulation and are the only thing standing
   between this design and the forced-relay case. They should not be bundled in.
