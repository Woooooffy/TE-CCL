# Multicast in the XML and the switch-forwarding JSON

Status: IMPLEMENTED 2026-10-14 on branch `switch-copy-send` (uncommitted). Decisions taken: the
receiver list in `<tb send>`, switch JSON format 2 for every entry, flow ids for paced (network)
flows only. §10 records the as-built details and how it was checked.

## 1. What goes wrong today

The hetero switch-copy run (`HeteroTaperedCluster`, allgather, `milp_chunking=cell`, copy at the
root). GPU 4 sends identity (4, 0) once into switch 17 at fine epoch 75, and 17 copies it to GPUs 0
and 8. The stitch holds this as ONE `Send` with two branches. What ncclize emits:

```xml
<gpu id="4">
  <tb id="5" send="0" recv="-1" chan="0">
    <step s="0" type="s" srcbuf="o" srcoff="4" dstbuf="o" dstoff="4" cnt="1" mscclflowid="0" rate="50.0"/> ...
  <tb id="8" send="8" recv="-1" chan="0">
    <step s="0" type="s" srcbuf="o" srcoff="4" dstbuf="o" dstoff="4" cnt="1" mscclflowid="7" rate="50.0"/> ...
```
```json
"3": {"0": {"port": 0, "next_hop_type": "gpu", "next_hop": 0, "src_gpu": 4, "dst_gpu": 0},
      "7": {"port": 2, "next_hop_type": "gpu", "next_hop": 8, "src_gpu": 4, "dst_gpu": 8}}
```

That is two transmissions at 50 GB/s each on GPU 4's 50 GB/s uplink: 100 GB/s, twice what the solve
planned. The cause is the string grammar ("8-Chunk paths" / "7-Flows" name one receiver per hop):
`flat_schedule.serialize` has to write one line per branch, and ncclize builds one send per line.

## 2. The design in one paragraph

Producers write their sends into a new schedule section, `"10-Sends"`, as structured records with a
branch list. ncclize builds its sends from that section when it is present. A send with several
receivers becomes ONE send op in a threadblock whose `send` attribute lists every receiver, and one
ordinary recv op on each receiver. They share one `mscclflowid`, which names the multicast tree.
The switch JSON gives that flow id one entry per switch on the tree, with a LIST of output ports.
A unicast send is the one-receiver case of all of this, and its XML is byte-identical to today's.

## 3. Producer side: the `"10-Sends"` section

One record per `Send`, in the schedule's own epoch units:

```json
"10-Sends": [
  {"chunk": 0, "from": 4, "sender": 4, "epoch": 75, "span": 72, "rate": 50.0, "volume": 1,
   "branches": [{"to": 0, "via": [17], "ready": 147},
                {"to": 8, "via": [17], "ready": 147}]},
  {"chunk": 0, "from": 4, "sender": 0, "epoch": 147, "span": 1, "rate": null, "volume": 1,
   "branches": [{"to": 1, "via": [14], "ready": 148}]}
]
```

- `chunk` / `from`: the chunk label and source GPU, exactly as in the "8-Chunk paths" keys, so
  ncclize decodes addresses the same way for both sections (src-major, or alltoall's dst-major,
  which can only ever have one branch).
- `epoch`, `span`, `ready`: when the head starts, how many epochs every link of the send is held,
  and when each receiver holds the data. ncclize reads the pacing window as `(epoch, epoch + span)`
  directly instead of re-deriving it from a rate.
- `rate`: GB/s for a paced (network) send; `null` for an unpaced one (intra-cell). The level that
  solved the send owns its rate, as today.
- `volume`: always 1. The section is only written by producers whose sends carry whole
  (sub-)chunks. That covers the hierarchical stitch, which refines every volume to 1, and the
  flat MILP.

Who writes it:
- `flat_schedule.serialize`: from `records`, which already hold every send with its fan-out.
- `AllGatherFormulation.dfs_remove_unnecessary_flows`: from `self.sends`, with
  `rate = chunk_size / (span * epoch_duration)`. Today the flat MILP XML paces every op at
  `chunk_size / epoch_duration`, which under pace classes is wrong for every class > 0 (a class-1
  send is emitted twice as fast as solved). Per-send rates fix that as a side effect.
- The flat LP does NOT write it. Its fractional multipath volumes need the per-path piece ranges
  `parse_flows_lp` assigns, and the LP never copies. It keeps the legacy path.

"7-Flows" and "8-Chunk paths" stay as they are: they are what people read. ncclize still reads the
"8-Chunk paths" KEYS for the demand set (participants, met epochs), but takes the transmissions
only from "10-Sends" when that section exists.

## 4. Inside ncclize

**4.1 A route is a unicast path or a tree.** Today an op's `path_key` is the switch tuple of a
unicast route (possibly port-qualified). A multicast send gets a `Tree` key instead:

```python
@dataclass(frozen=True)
class Tree:
    branches: Tuple[Tuple[int, Tuple[int, ...]], ...]   # (receiver dense rank, raw switch path), sorted
```

Every helper that deconstructs a route key learns the one extra case:
- `_send_uplink`: the tree's shared first switch;
- `_recv_downlink`: per receiver, its branch's last switch;
- `_route_links`: the union of the branches' links;
- `build_switch_routes`: the tree, below.

Nothing else needs to tell the two kinds apart. A one-branch send is written as an ordinary unicast
key, never as a one-branch `Tree`, so every unicast key is unchanged.

**4.2 `parse_sends` (teccl_ncclize).** This is the counterpart of `parse_flows_lp` for a schedule
with "10-Sends", returning the same tuple so `build_algorithm` changes only in which parser it
calls:
- **taccl steps:** one `(chunk_id, sender, receiver)` per BRANCH. `check_implements` then
  verifies a multicast exactly as it verifies N unicasts with the same data, and every receiver is
  covered.
- **`flow_path_keys[(step, chunk_id, sender, receiver)]`:** the send's route key, which is a
  `Tree` for a multicast. This is how the tree reaches the op builder. It is carried, not inferred.
- **`flow_rates`:** the record's rate, for paced sends.
- **`paced_sends`:** keyed `(step, sender, peer, route)`, where `peer` is the receiver for a
  unicast and the sorted receiver tuple for a multicast, with window `(epoch, epoch + span)`.

**4.3 Ops (taccl_ncclize.ncclize).**
- Sends are grouped by `(src, route)` instead of `(src, dst, route)` when the route is a `Tree`.
  The group's receivers are the tree's.
- `make_intervals` merges runs that are contiguous at the source AND at every receiver.
- Each interval makes ONE send op with `peer = receivers` (a tuple), and one recv op per receiver
  (`peer = src`, that receiver's own dst buffer and offset).
- The op set the channel allocator sees is `[send, recv_1, ..., recv_N]`.
- Data dependencies are unchanged: the send reads the source once, and each recv writes its own
  receiver.

**4.4 Channels: one route per connection.** An MSCCL connection is `(channel, peer)` on a GPU, an
independent FIFO. The property this design keeps is:

> Two different routes, and therefore two different flow ids, never share a connection: on every
> edge `(src, dst)` they use disjoint channels. Consequently every threadblock carries the ops of
> exactly one route.

A route here is the canonical route of 4.6 (a unicast path or a tree). Unpaced intra-cell routes
are routes too; they just carry no id. So the property covers them as well, but only paced routes
show up as distinct `mscclflowid`s.

It holds by construction, because the allocator partitions channels by the SAME canonical route
object the flow id is assigned from (`route_flow_ids` and the allocator use one key):

- **Rule.** Every op set (`[send, recv_1..recv_N]`; a unicast has N = 1) is assigned channels as
  follows. A route that already owns channels reuses them, round-robin over its replicas. A new
  route takes the first `replicas` channel ids that NO other route owns on ANY of its edges
  `(src, r_i)`. `replicas` is the minimum `link` over those edges.
- **Different routes on one edge get disjoint channels.** Unicast vs unicast, unicast vs tree, and
  tree vs tree all fall under the same rule, so a tree never shares a channel with a unicast
  running along one of its branches, or with another tree through the same receiver.
- **One route may use several channels** (its replicas, as today). That never breaks the property,
  which is about different routes.
- **Threadblocks** stay one per `(gpu, direction, peer, channel)`. With each `(edge, channel)`
  owned by one route, the sender's threadblock and every receiver's recv threadblock each hold one
  route's ops. A multicast threadblock (`send="0,8"`) holds only its tree.
- **Unicast numbering is unchanged where no tree is involved.** On an edge no tree crosses, the
  rule reduces to today's `path_index * link + rr`: the k-th route on the edge gets channels
  `k*link .. k*link + link - 1`. A schedule without multicast therefore keeps every channel id. On
  an edge a tree crosses, routes seen after the tree skip the channels it holds.
- **The channel limit is checked on the id assigned**, not the count per edge. A tree must find a
  channel id free on all of its edges, which can be higher than any single edge needs on its own.
  `MAXCHANNELS` (`max_channels`, 32) bounds the channel IDS a GPU's threadblocks reference, so the
  check becomes `chan < max_channels` on every assignment, failing with the edges and routes
  involved.

Scope: the property is per connection, which is what the runtime orders. Two routes on DIFFERENT
edges (other peers, or other senders) may carry the same channel number. They are on different
connections, so nothing is shared. Making channel ids globally unique per flow would cap a GPU at
32 flows in total and buy no ordering guarantee.

**4.5 Threadblocks and XML.** Threadblocks stay one per `(gpu, direction, peer, channel)`, with
`peer` now possibly a receiver tuple. That makes a multicast threadblock exactly one multicast
connection. In the XML:
- `<tb send="0,8" recv="-1" chan="0">`: `send` is a comma-separated, sorted list of ranks when the
  threadblock multicasts; a single rank otherwise (unchanged). A stock runtime fails to parse the
  list. That is intended: emitting N unicasts is the bug of §1, so it must not happen silently.
- The send step keeps `srcbuf/srcoff/cnt/rate/mscclflowid/depid/netdepid/remotedep` as today.
  `dstbuf/dstoff` on a send step are informational (each receiver's recv step carries its own).
  For a multicast they are written when every receiver agrees, and as `-1` otherwise. For
  allgather they agree whenever the receivers keep the chunk in their output buffer (every
  participant does; a passive relay holding it in scratch may not).
- Recv steps are unchanged. Each receiver's threadblock is `recv=<sender>` on the same channel,
  carrying the tree's `mscclflowid`.
- `rate` is the one stream's rate (50 GB/s in §1), not a sum over branches.

**4.6 Flow ids are paths.** `mscclflowid` is a bijection with the PHYSICAL ROUTE a transmission
takes through the network, so a switch can forward by flow id alone:

- **Unicast:** the route is the raw node sequence `(src, switch_1, ..., switch_k, dst)`, with the
  cable taken on each hop when a link is port-split (the port-qualified path key). This is today's
  `(src, dst, path_key)` key, written out as what it means.
- **Multicast:** the route is the TREE `(src, {(dst_i, switches_i)})`. That is the source plus the
  set of root-to-leaf paths, canonicalized as the branches sorted by receiver, with cables when
  qualified. Two transmissions get the same id iff they leave the same GPU and reach the same
  receivers over the same switches and cables.

So, in both directions:
- one route ↔ one id, across the whole schedule: every epoch, chunk, op and threadblock on a route
  shares its id, and no two routes share one;
- a tree is its own route: a multicast tree and a unicast path that runs along one of its branches
  get DIFFERENT ids, and so do two trees that differ in one branch. This is required, not
  incidental: the id is what a switch forwards on, and the tree's switch replicates where the
  unicast's does not;
- `--no-multicast` expands a tree into its branch paths, which take the unicast ids of those paths,
  shared with any genuine unicast on them.

**Scope:** ids are emitted for PACED transmissions, as today. Those are the network sends a
programmable switch forwards. An unpaced intra-cell hop crosses an NVSwitch that is never
programmed from this table, and carries no id (a multicast one included). If ids are wanted there
too, the rule above is unchanged; only the "paced" filter goes.

**Where it is assigned:** `ncclize` keeps one `route -> id` map (`route_flow_ids`, keyed by the
canonical route above instead of `(src, dst, path_key)`), handing out dense ids in first-seen
order. The send op and EVERY recv op of the transmission carry that id. Each `flow_manifest` record
carries the route itself, so the switch JSON is built from the same object the id was assigned to.
Nothing downstream re-derives a route from an id or an id from a route.

**4.7 Pacing gates.** These are unchanged in logic, with keys generalized as in 4.2:
- A multicast send is ONE entry in the SEND pool (it finishes once, at `epoch + span`).
- It is one arrival per receiver in the RECV / REMOTE pools; the mirror key for receiver `r` is
  `(step, r, src, tree)`.
- `_realize_pacing_gates` indexes send ops by `(step, gpu, peer, route)` with `peer` the receiver
  tuple, and recv ops by their ordinary mirror keys, so both resolve.
- `unpinned_sends` mirrors once per receiver.
- `_widen_rates_for_tb_serialization` needs no change: it groups by `(step, route)` inside one
  threadblock.

## 5. The switch JSON, format 2

Two sections: the routes themselves, keyed by flow id (the bijection of 4.6, written down), and
each programmable switch's forwarding entry per flow id, with lists where a multicast has several
values. Flow 0 is the §1 copy, with its real ports; flow 1 is an illustrative unicast:

```json
{"format": 2,
 "switch_id_map": {"0": 14, "1": 15, "2": 16, "3": 17, "4": 18},
 "flows": {
   "0": {"src_gpu": 4, "branches": [{"dst_gpu": 0, "switches": [17]},
                                    {"dst_gpu": 8, "switches": [17]}]},
   "1": {"src_gpu": 5, "branches": [{"dst_gpu": 1, "switches": [17, 18]}]}},
 "switches": {
   "3": {"0": {"ports": [0, 2],
               "next_hops": [{"type": "gpu", "id": 0}, {"type": "gpu", "id": 8}]},
         "1": {"ports": [3], "next_hops": [{"type": "switch", "id": 4}]}}}}
```

- `flows`: the route of every flow id, in raw switch ids, plus `cables` per branch when a link is
  port-split. This makes "id ↔ path" explicit and checkable, and is what a reader looks up to learn
  what an id means. A unicast route is a one-branch entry.
- `ports`: this switch's output ports for the flow, from `teccl.sends.switch_out` (the set of next
  nodes after this switch, over all branches). A switch the tree passes through without branching
  has one port. **A runtime replicates a packet to every listed port.** `ports` is still the only
  load-bearing key.
- `next_hops`: debug, as today, in list form. The endpoints that used to be repeated on every
  entry (`src_gpu`, `dst_gpu`) now live once in `flows`.
- Non-programmable (self-routing) switches are still skipped, with the existing chain-past check.
  It is extended with one new rule: a tree may not BRANCH at a skipped switch, because nothing
  would tell that switch to replicate. Fail at generation. Inter-cell trees branch only at network
  switches today, and intra-cell (NVSwitch) multicasts are unpaced, so they get no flow id and no
  entry at all.
- Port-split (multi-port) links: a multicast route over a link with `port_count > 1` is refused for
  now. No topology declares one yet, and splitting a tree's pieces across cables needs its own
  rule.

Unicast entries change shape too: `port` becomes `ports: [p]`. One format with one rule is simpler
than two shapes keyed on the fan-out, and the top-level `"format": 2` lets a reader reject an old
file instead of misreading it.

## 6. Checks

- **`check_implements`**: unchanged. It sees every branch as a delivery (4.2).
- **XML self-check**, new and cheap: every multicast send step has exactly one recv step on each
  listed receiver, on the same channel, at the same position in that `(peer, channel)` FIFO. This
  extends the existing FIFO-pairing invariant to N receivers.
- **One route per connection**, new, at generation, replacing today's warning (taccl_ncclize,
  "switch paths ... only got N distinct channel(s)", which only looks within one step): every
  threadblock's ops carry one route (one `path_key`, hence at most one `mscclflowid`). This is
  exactly 4.4's property, checked on the emitted structure rather than on the allocator's
  bookkeeping, so a future change to either cannot silently break it.
- **Flow-id bijection check**, new, at generation: the `route -> id` and `id -> route` maps are
  inverse (no route got two ids, no id names two routes). Every op carrying an id is on that id's
  route: its sender is the route's source, and a recv's GPU is one of its receivers.
- **Switch-table check**, new: for every flow id, unicast or multicast, following `ports` from the
  source's first switch reproduces exactly that id's entry in `flows`. Every listed switch is
  reached, in order, every receiver is reached once, and nothing else is.
- **Link-load check from the XML**, new: per sender per step, the sum of send-step rates per
  uplink must fit within the uplink's capacity. On §1 it fails today (100 against 50) and passes
  after. This is the regression test that matters, and it needs no simulator.

## 7. Compatibility

- **Schedules without switch copy:** must give byte-identical XML through `parse_sends` and
  through the legacy parsers. That is the A/B gate of step 2, over the existing sample schedules.
- **Old schedule files (no "10-Sends"):** the legacy parsers run unchanged.
- **`--no-multicast`:** expands every `Tree` into per-branch unicast routes before op building.
  That is today's output, for a runtime that cannot multicast yet. It is explicit and opt-in,
  because it oversubscribes the sender's uplink.
- **The switch JSON is format 2 for every schedule.** A consumer of format 1 must be updated
  (`port` → `ports[0]`).

## 8. Implementation order

1. **`"10-Sends"` writers:** `flat_schedule.serialize` and the flat MILP. Additive; no consumer
   changes.
2. **`parse_sends` + `Tree` + the route helpers.** A/B: unicast XML byte-identical to the legacy
   parsers on every sample.
3. **Ops, channels, threadblocks, the XML `send` list** (4.3-4.6), with the one-route-per-
   connection assert. Test on the §1 case: GPU 4 has one threadblock `send="0,8"`, the link-load
   check passes, and no threadblock mixes routes. Add a unit case where a tree and a unicast along
   one of its branches share an edge: they must get different channels.
4. **Pacing keys** (4.7). `pacing_gates_test` gets a multicast case: one send-pool entry, two
   arrivals.
5. **Switch JSON format 2** (§5) with the `flows` section, the bijection check and the table
   check.
6. **`--no-multicast`** and the XML self-check.

## 9. Decisions to make

1. **`send` as a rank list in `<tb>`** (proposed), or keep `send` single and add a new
   `mcast="0,8"` attribute. The list is the direct reading of "the destination is a set" and fails
   loudly on a runtime without multicast. The extra attribute would be silently ignored by one,
   which reproduces §1.
2. **Switch JSON format 2 for everything** (proposed), or keep format 1's `port` for unicast
   entries and add `ports` only for multicast ones.
3. **Who consumes multicast**: the MSCCL runtime fork (one stream per multicast connection; NVLS
   inside a node), the simulator, or both. The format above is runtime-neutral, but a runtime must
   at least read the `send` list before any multicast XML can run. The `--no-multicast` path
   covers the gap.

## 10. As implemented (2026-10-14)

**Where it lives.**
- `teccl/ncclize/routes.py`: `Tree` and the route helpers (`first_hop`, `last_hop`,
  `route_links`, `receivers_of`).
- `teccl_ncclize.parse_sends`, dispatched from `build_algorithm` when `"10-Sends"` is present,
  with `multicast=` / `--no-multicast`.
- `taccl_ncclize`:
  - multicast grouping, `make_multicast_intervals`, one send op plus N recv ops per op set;
  - the first-fit channel allocator over all edges of a route;
  - tuple-peer threadblocks and the `send="r1,r2"` XML;
  - `_assert_one_route_per_threadblock` and `_assert_multicast_pairing`.
- `teccl_ncclize.build_switch_routes` (format 2 with `flows`, the bijection check, the
  no-branching-at-a-skipped-switch rule) and `check_switch_routes`, run by the CLI.
- Producers: `flat_schedule.serialize` and `AllGatherFormulation` write `"10-Sends"`.

**Details settled during implementation.**
- **A record's `chunk` label is its DEMAND's label.** An alltoall label encodes the demand's
  final destination, not a relay hop's receiver. The stitch takes it from the first chain that
  uses the send, and `parse_sends` reads the destination slot off the label.
- **Records are written in first-use order over the demand chains,** so a schedule without copy
  expands to exactly the hops, in the order, `parse_flows_lp` reads off "8-Chunk paths".
- **Remote (P4) gates follow the legacy parser of the same format:** on for the LP/hierarchical
  format, off for the flat MILP, which never had them.
- **ncclize XML was nondeterministic before this change.** Dependency lists were `list(set(ops))`,
  and ops hash by identity, so which dependency became `depid` and which a nop varied between
  runs: 2 distinct XMLs in 4 runs of the hetero LP replay. They are now deduplicated in order. This
  was needed for any byte-level A/B.

**Verification.**
- All 19 test files pass. The new one is `teccl/ncclize/multicast_test.py`, covering:
  - one send threadblock to both receivers;
  - a tree and a unicast along one of its branches on different ids and channels;
  - format 2;
  - the fallback;
  - the hetero end-to-end run with an uplink-load check read from the XML.
- **A/B, legacy parsers against `parse_sends` on the same schedule dict:** BYTE-IDENTICAL XML for
  both LP stitch replays (hetero allgather, alltoall) and for every no-copy hierarchical MILP
  case. Copy cases and the flat MILP have identical ops. In the flat MILP, the legacy order comes
  from the set-built "7-Flows", and the gate sets pin exactly the same sends (both report zero
  unpinned) but break ties between equally timed producers differently.
- **The CLI on legacy schedule files without `"10-Sends"`** (`fat_tree_pod_schedule_2chunks`,
  `fat_tree_pod_alltoall_schedule`): byte-identical XML to the base commit.
- **Hetero switch copy:**
  - every uplink is within capacity;
  - `--no-multicast` reproduces the old 100 GB/s on GPU 4's 50 GB/s uplink;
  - switch 17's entry for GPU 4's tree is `ports: [0, 2]`.

**Not done.** A runtime or simulator that reads the `send` list. The XML is runtime-neutral, and a
stock MSCCL runtime fails to parse it by design.
