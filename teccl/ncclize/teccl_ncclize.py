#!/usr/bin/env python3
"""
Convert a TE-CCL schedule JSON (as produced by teccl/scheduler.py, e.g.
teccl/examples/schedules/ndv2_schedule.json) into the NCCL/MSCCL XML format,
by injecting it into TACCL's Algorithm representation and running TACCL's
own ncclize().

Usage:
    python teccl_ncclize.py SCHEDULE.json -o OUTPUT.xml

Requires taccl package (https://github.com/microsoft/taccl)'s ncclize dependencies
(lxml, z3-solver).
"""
import argparse
import bisect
import json
import math
import os
import re
import sys
from collections import defaultdict
from fractions import Fraction

# Rationalize each solver volume to a fraction with denominator <= MAX_DENOM
# before taking the lcm. This is the "error factor" that keeps float noise in
# the LP output (e.g. 0.3333333) from producing an absurd denominator; LP path
# splits are simple fractions (halves/thirds/quarters).
MAX_DENOM = 64
# Hard cap on the global subdivision factor M. alltoall(N).chunk_up(M) makes
# N*N*M chunks and check_implements allocates an O(N^3 * M) state array, so an
# unbounded M would blow up. Raise clearly instead.
MAX_M = 128

try:                                  # sibling module; this file runs both as a script
    from port_split import unqualify_path_key                            # noqa: E402
except ImportError:                   # ... and as part of the teccl package
    from teccl.ncclize.port_split import unqualify_path_key

FLOW_RE = re.compile(
    r'Chunk (\d+) from (\d+) traveled over (\d+)->(\d+)'
    r'(?:\s+with volume\s+(?P<volume>[\d.]+))?'
    r'\s+in epoch (?P<epoch>\d+)'
    r'(?:\s+at rate\s+(?P<rate>[\d.eE+-]+))?'
    r'(?:\s+via switches\s+(?P<switches>[\d\s\->]+))?$'
)

# Matches an "8-Chunk paths" demand key, e.g.
# "Demand at 3 for chunk 0 from 0 met by epoch 3".
DEMAND_RE = re.compile(
    r'Demand at (\d+) for chunk (\d+) from (\d+) met by epoch (\d+)$'
)

# Matches one entry of an "8-Chunk paths" demand's path list, e.g.
# "0->3 in epoch 0 via switches 4 -> 7 -> 5", or the LP form
# "0->3 with volume 0.5 in epoch 0 via switches 4->6->5". Same grammar as
# FLOW_RE's suffix, minus the "Chunk C from O traveled over " prefix. The
# "with volume V" token is optional so both the older allgather MILP format
# (no volume) and the LP/alltoall format parse with one regex.
#
# "at rate R" (GB/s) is likewise optional. A rate is emitted by the solver level
# that produced the flow -- it knows its own epoch duration and capacity model --
# rather than being re-derived here from a single global epoch duration. A flow
# without one is deliberately unpaced (e.g. the hierarchical solver's intra-cell
# NVLink hops, which carry ordering but are not pinned to the epoch grid).
PATH_SEGMENT_RE = re.compile(
    r'(\d+)->(\d+)'
    r'(?:\s+with volume\s+(?P<volume>[\d.]+))?'
    r'\s+in epoch (?P<epoch>\d+)'
    r'(?:\s+at rate\s+(?P<rate>[\d.eE+-]+))?'
    r'(?:\s+via switches\s+(?P<switches>[\d\s\->]+))?$'
)


def _parse_switch_path(switches):
    """Turn a FLOW_RE 'switches' group into a hashable path key.

    None (no 'via switches' suffix, i.e. a direct hop) and a populated tuple
    are intentionally distinct keys, so two chunks on the same (src, dst)
    edge that took different switch paths -- or one direct and one relayed --
    are never treated as interchangeable.
    """
    if switches is None:
        return None
    return tuple(int(s.strip()) for s in switches.split('->'))


def parse_flows(schedule, port_qualify=None):
    """Group TE-CCL's '7-Flows' entries by epoch, remapping GPU ids to a dense
    0-indexed range.

    TE-CCL's node ids are raw 0-indexed topology indices that may include
    switch nodes at arbitrary positions (e.g. index 0 for NDv2, the last index
    for Star). Switch hops never appear as the 'A->B' endpoints in 7-Flows --
    TE-CCL's own flow-merging collapses them into a trailing "via switches"
    annotation -- so the set of ids that *do* appear as an origin/src/dst is
    exactly the set of real GPUs, and we remap that set to 0..N-1 instead of
    assuming any fixed offset.

    Returns (num_nodes, num_subchunks, steps_in_order, flow_path_keys,
    switch_rank_map, sorted_epochs, flow_completion_epochs, gpu_rank_map):
    - steps_in_order is a list of lists of (global_chunk_id, src, dst)
      0-indexed tuples, one list per non-empty epoch, in increasing epoch
      order.
    - sorted_epochs is the parallel list of raw epoch numbers (steps_in_order[i]
      is raw epoch sorted_epochs[i]). Exposed so downstream code can recover the
      true epoch axis -- including gaps where a whole epoch is globally empty
      and thus dropped from steps_in_order.
    - flow_completion_epochs maps (step_idx, global_chunk_id, src, dst) -> the
      raw epoch by which dst actually holds the chunk. For a flow relayed through
      switches, "in epoch N" in a 7-Flows line is only the *start* epoch (see
      chunk_flow_path_to_string() in teccl/solvers/allgather.py), so a relayed
      flow's true completion can be several epochs later; it is read from the
      "8-Chunk paths" section's "met by epoch E" rather than assumed to be one
      epoch per hop, which only holds at zero link latency. Used to place a recv
      at its arrival epoch in the per-GPU debug view -- diagnostics only, it does
      not influence the emitted operation order.
    - flow_path_keys maps (step_idx, global_chunk_id, src, dst) -> path key,
      where step_idx is the index into steps_in_order (not the raw epoch
      number) and the path key is whatever _parse_switch_path() returned for
      that flow (a tuple of raw switch ids, or None for a direct hop). Used
      downstream to avoid merging chunks that took different switch paths
      into a single send op.
    - switch_rank_map maps each raw switch id appearing in any path key to a
      dense 0-indexed id, the same way gpu_rank_map does for GPU ids.
    - gpu_rank_map maps each raw GPU id to its dense id. Returned rather than
      discarded because the switch forwarding table has to name the PORT a route
      leaves a switch on, and the last hop's port is the one facing the
      destination GPU -- which is a port lookup on a RAW node id, so the dense
      id in the manifest is not enough to find it.
    """
    parsed = []
    raw_ids = set()
    switch_raw_ids = set()
    max_subchunk = 0

    for line in schedule['7-Flows']:
        m = FLOW_RE.match(line)
        if not m:
            raise ValueError(f'Could not parse flow line: {line!r}')
        subchunk, origin, src, dst = (int(x) for x in m.group(1, 2, 3, 4))
        epoch = int(m.group('epoch'))
        path_key = _parse_switch_path(m.group('switches'))
        raw_ids.update((origin, src, dst))
        if path_key:
            switch_raw_ids.update(path_key)
        # Port qualification happens HERE, not on the way out, so that every downstream key
        # built from a path key -- flow_path_keys, paced_sends and therefore the pacing gates --
        # agrees. Rewriting flow_path_keys after the fact would leave the gate manifest keyed on
        # the unqualified form and silently drop every gate.
        if port_qualify is not None:
            path_key = port_qualify(src, dst, path_key, origin, subchunk, epoch)
        max_subchunk = max(max_subchunk, subchunk)
        parsed.append((epoch, subchunk, origin, src, dst, path_key))

    rank_map = {raw: idx for idx, raw in enumerate(sorted(raw_ids))}
    switch_rank_map = {raw: idx for idx, raw in enumerate(sorted(switch_raw_ids))}
    num_nodes = len(rank_map)
    num_subchunks = max_subchunk + 1

    by_epoch = defaultdict(list)
    for epoch, subchunk, origin, src, dst, _ in parsed:
        chunk_id = rank_map[origin] * num_subchunks + subchunk
        by_epoch[epoch].append((chunk_id, rank_map[src], rank_map[dst]))

    sorted_epochs = sorted(by_epoch)
    steps_in_order = [by_epoch[epoch] for epoch in sorted_epochs]

    epoch_to_step_idx = {epoch: idx for idx, epoch in enumerate(sorted_epochs)}
    flow_path_keys = {}
    for epoch, subchunk, origin, src, dst, path_key in parsed:
        chunk_id = rank_map[origin] * num_subchunks + subchunk
        step_idx = epoch_to_step_idx[epoch]
        flow_path_keys[(step_idx, chunk_id, rank_map[src], rank_map[dst])] = path_key

    # Derive each flow's true completion epoch from "8-Chunk paths", keyed
    # the same way as flow_path_keys so both can be looked up together
    # downstream. Every demand's path list's *last* segment is the hop that
    # actually lands the chunk at the demand's destination -- and, since
    # this is an allgather, every relay GPU also has its own separate
    # demand entry for the same chunk (it needs the chunk for itself too),
    # so "last segment of every demand" covers every hop appearing in
    # 7-Flows, regardless of relay-chain length.
    # flow_completion_epochs keeps the raw completion epoch. It lets the per-GPU
    # debug view place each recv at the epoch the chunk actually *arrives*,
    # rather than the epoch its flow started -- a switch-relayed recv can land
    # several epochs after it was sent, and can even exceed the last flow-start
    # epoch.
    flow_completion_epochs = {}
    for demand_key, path_segments in schedule['8-Chunk paths'].items():
        dm = DEMAND_RE.match(demand_key)
        if not dm:
            raise ValueError(f'Could not parse demand key: {demand_key!r}')
        dst_raw, subchunk, origin_raw, completion_epoch = (
            int(x) for x in dm.group(1, 2, 3, 4))

        last_segment = path_segments[-1]
        sm = PATH_SEGMENT_RE.match(last_segment)
        if not sm:
            raise ValueError(f'Could not parse path segment: {last_segment!r}')
        hop_src_raw, hop_dst_raw = (int(x) for x in sm.group(1, 2))
        hop_epoch = int(sm.group('epoch'))

        if hop_dst_raw != dst_raw:
            raise ValueError(
                f'"8-Chunk paths" entry {demand_key!r} has a last path segment '
                f'{last_segment!r} landing at {hop_dst_raw}, expected {dst_raw}')
        for raw_id in (origin_raw, hop_src_raw, hop_dst_raw):
            if raw_id not in rank_map:
                raise ValueError(
                    f'Unknown GPU id {raw_id} in "8-Chunk paths" entry {demand_key!r}')
        if hop_epoch not in epoch_to_step_idx:
            raise ValueError(
                f'Epoch {hop_epoch} in "8-Chunk paths" entry {demand_key!r} '
                f'does not appear in "7-Flows"')

        chunk_id = rank_map[origin_raw] * num_subchunks + subchunk
        step_idx = epoch_to_step_idx[hop_epoch]
        flow_completion_epochs[
            (step_idx, chunk_id, rank_map[hop_src_raw], rank_map[hop_dst_raw])
        ] = completion_epoch

    return (num_nodes, num_subchunks, steps_in_order, flow_path_keys,
            switch_rank_map, sorted_epochs, flow_completion_epochs, rank_map)


def _segment_volume(match):
    """Volume fraction from a FLOW_RE / PATH_SEGMENT_RE match, defaulting to 1.0
    when the 'with volume' token is absent (older allgather MILP format). A
    default of 1.0 makes the subdivision factor collapse to 1 for those
    schedules, so their handling is unchanged."""
    v = match.group('volume')
    return 1.0 if v is None else float(v)


def detect_collective(schedule):
    """Return the collective a schedule solves: 'alltoall', 'allgather',
    'gather' or 'broadcast'.

    This is the collective *identity* (which pre/postconditions and chunk
    addressing apply), independent of the *format* the schedule is written in
    (see is_lp_format). Prefers the explicit "0-Collective" field the solvers
    now emit. For schedules predating that field, falls back to the structural
    divergence of "8-Chunk paths": the LP generator emits a *nested* value per
    demand (a list of paths, each a list of [epoch, segment] pairs) and only
    ever solved alltoall back then, while the allgather MILP emits a *flat* list
    of segment strings.
    """
    explicit = schedule.get('0-Collective')
    if explicit:
        return explicit
    for paths in schedule['8-Chunk paths'].values():
        if not paths:
            continue
        return 'alltoall' if isinstance(paths[0], list) else 'allgather'
    return 'allgather'


def is_lp_format(schedule):
    """Whether "8-Chunk paths" is in the LP (continuous-flow) format rather than
    the MILP format.

    The parser axis is the FORMULATION, not the collective: the LP emits a
    *nested* per-demand path structure (a list of paths, each a list of
    [epoch, segment] pairs) with fractional volumes, while the MILP emits a
    *flat* list of segment strings. Those two formats happened to correlate with
    the collective only because scheduler._resolve_formulation defaults
    ALLGATHER -> MILP and everything else -> LP; the hierarchical path breaks
    that correlation by emitting an LP-format allgather.

    So this reads the explicit "0-Formulation" field (Formulation.LP/MILP, see
    teccl/input_data.py) and falls back to the structural nested-vs-flat sniff
    only for schedules that predate the field. The taccl collective that gets
    built is a separate axis and keys on detect_collective().
    """
    explicit = schedule.get('0-Formulation')
    if explicit:
        name = str(explicit).rsplit('.', 1)[-1].upper()   # 'LP' | 'Formulation.LP'
        if name not in ('LP', 'MILP'):
            raise ValueError(
                f"Unknown '0-Formulation' value {explicit!r}; expected 'LP' or 'MILP'.")
        return name == 'LP'
    for paths in schedule['8-Chunk paths'].values():
        if not paths:
            continue
        return isinstance(paths[0], list)
    return False


def _compute_subdivision(schedule):
    """Global subdivision factor M so every flow's fractional volume becomes an
    integer number of sub-chunks.

    Each distinct volume is rationalized to a bounded-denominator fraction (see
    MAX_DENOM) and M is the lcm of those denominators, so v * M is an integer
    for every v. Capped at MAX_M to avoid an intractable chunk_up() expansion.
    Old allgather schedules (all volumes default to 1.0) yield M = 1.
    """
    M = 1
    denoms = set()
    for paths in schedule['8-Chunk paths'].values():
        for path in paths:
            for _epoch, segment in path:
                m = PATH_SEGMENT_RE.match(segment)
                if not m:
                    raise ValueError(f'Could not parse path segment: {segment!r}')
                frac = Fraction(_segment_volume(m)).limit_denominator(MAX_DENOM)
                d = frac.denominator
                if d not in denoms:
                    denoms.add(d)
                    M = M * d // math.gcd(M, d)
                    if M > MAX_M:
                        raise ValueError(
                            f'Volume denominators {sorted(denoms)} require a '
                            f'subdivision factor M > MAX_M={MAX_M}; refusing to '
                            f'expand chunks that finely.')
    return M


def _send_uplink(key):
    """The physical outgoing PORT a send op contends for, from its key
    (step, src, dst, path_key).

    A send leaves its source GPU over that GPU's uplink to the first switch on its
    route, so the contended link is (src, first_switch); two sends from one GPU that
    enter the fabric at DIFFERENT first switches use different uplinks and do not
    compete. A send with no switch route (path_key is None -- a direct hop) is keyed
    by (src, None, 0); those all share one bucket, which is the safe conservative
    grouping.

    The key must be DECONSTRUCTED, not indexed. A port-qualified key (port_split) is
    ((switches...), (ports...)), so path_key[0] on one of those is the whole switch
    tuple rather than the first switch -- which would give every distinct ROUTE its
    own contention pool and stop sends that really do share an uplink from pacing
    against each other. unqualify_path_key is the one place that knows the shape.

    When cables ARE present the first hop's cable joins the identity, because that
    is the physically contended thing: two sends leaving one GPU on different cables
    of a split uplink do not compete. On an unsplit link the cable is 0 for every
    send, so the grouping is exactly what it was.

    Note this is the CABLE index and not a port number (Topology.port_map). It has
    to be: hop 0 leaves a GPU, and a GPU carries no port map, so every cable of a
    split uplink would collapse to one None and the pool would over-serialize.
    """
    _step, src, _dst, path_key = key
    switches, cables = unqualify_path_key(path_key)
    first_switch = switches[0] if switches else None
    first_cable = cables[0] if cables else 0
    return (src, first_switch, first_cable)


def _finish_before_start_gates(paced_sends):
    """Derive pacing gate edges from per-send link-occupancy windows.

    `paced_sends` maps a send op key (step_idx, src, dst, path_key) to its
    (start_epoch, finish_epoch) window in FINE epochs, where finish = start +
    volume/rate (how long the send occupies its outgoing link). Only flows the
    producing level chose to PACE appear here, so every candidate below is pinned to
    that level's epoch grid; unpaced intra-level hops carry ordering, not time, and
    must never become a clock.

    Each send S is gated on the latest paced event that lands at or before S starts,
    drawn from two pools:

      SEND pool (P2)  -- sends contending for the same physical uplink (see
        _send_uplink), timed by their link-occupancy FINISH: the send that frees the
        uplink is what lets S onto the wire.
      RECV pool (P3)  -- paced deliveries INTO S's source GPU, timed by their ARRIVAL.
        A send at epoch k whose uplink was idle through k-1 has no send to wait on;
        a recv landing exactly at k is then the only rate-paced clock tick available,
        and gating on it pins S to k. This is what closes the residual the stitch
        reports (flat_schedule.check_network_pacing) for a GPU that goes quiet on an
        uplink for an epoch but is still receiving.

    A recv is a paced send observed from the far end, so the RECV pool needs no new
    data: a send (step, src, dst, path_key) finishing at t IS a delivery at `dst`
    arriving at t, and the receiving op's mirror key is (step, dst, src, path_key).

    Deriving the edge from finish/arrival-vs-start rather than step ORDER is what makes
    it correct for two cases a plain sort gets wrong: two sends in the SAME epoch never
    gate each other (a same-epoch P has finish > start_S, so it is not a candidate),
    and a multi-epoch send does not serialize a later send that runs alongside its
    tail on freed capacity (that P is excluded; the send that truly frees the link is
    picked). Grouping is per physical uplink (src, first_switch), so a multi-uplink
    GPU paces each of its uplinks independently.

    Ties prefer the SEND pool. A send-sourced gate is free -- it rides netdepid/netdeps
    and costs no XML step -- while a recv-sourced one is a depid/deps edge that may cost
    an expansion nop, so the recv pool is only reached for a send the uplink cannot pin.

    Returns a list of (consumer_key, producer_key, kind) edges, kind being 'send' or
    'recv'. The kind picks the RUNTIME CARRIER and cannot be inferred from the keys:
    a recv key (step, dst, src, path_key) is indistinguishable from the key of a real
    send on the reverse edge at the same step, which bidirectional traffic produces.
    A 'send' edge orders one send behind another ON THE WIRE and is carried by
    netdepid/netdeps, which the proxy thread enforces against the NIC. A 'recv' edge
    is discharged the moment the proxy marks the recv slot filled -- exactly the event
    depid/deps already encodes and the GPU kernel already waits on -- so it rides that
    path instead; see taccl_ncclize._realize_pacing_gates.
    """
    from collections import defaultdict as _dd
    by_link = _dd(list)
    arrivals = _dd(list)   # receiving gpu -> [(arrival_epoch, recv mirror key)]
    for key, (start, finish) in paced_sends.items():
        step, src, dst, path_key = key
        by_link[_send_uplink(key)].append((start, finish, key))
        arrivals[dst].append((finish, (step, dst, src, path_key)))

    edges = []
    for sends in by_link.values():
        for cstart, _cfin, ckey in sends:
            csrc = ckey[1]
            # (time, kind_rank, key), maximised: the latest-landing eligible producer. The rank
            # breaks a tie at equal time toward the SEND pool, so it ranks ABOVE recv here.
            best = None
            for _pstart, pfin, pkey in sends:
                if pkey == ckey:
                    continue
                if pfin <= cstart and (best is None or (pfin, 1) > (best[0], best[1])):
                    best = (pfin, 1, pkey)
            for arrival, rkey in arrivals.get(csrc, ()):
                if arrival <= cstart and (best is None or (arrival, 0) > (best[0], best[1])):
                    best = (arrival, 0, rkey)
            if best is not None:
                edges.append((ckey, best[2], 'send' if best[1] == 1 else 'recv'))
    return edges


def parse_flows_lp(schedule, collective_name, port_qualify=None):
    """Parse an LP schedule (any collective) the way parse_flows() does for the
    allgather MILP, but generate sends from the nested "8-Chunk paths" so each
    chunk's multipath decomposition can be split into disjoint integer sub-chunk
    pieces.

    The schedule carries S sub-chunks per demand; each is further subdivided into
    M volume pieces (M from _compute_subdivision) so fractional volumes become
    integer piece counts. The chunk_up factor is S * M, and piece p of a demand's
    sub-chunk sc gets

        chunk_id = base * (S * M) + sc * M + p

    where `base` is the base chunk index the taccl collective (before chunk_up)
    assigns to that demand's data. chunk_up fans each base address a out to
    a * (S*M) + local, all inheriting a's pre/postcondition ranks, so `base` must
    match the collective's own indexing or check_implements() will raise. The
    per-collective base index (and how the chunk label c decodes into a
    sub-chunk) is:

      - DST-MAJOR, alltoall(N): base = dst_dense * N + src_dense,
        c = sc * N + dst_dense (precondition rank == index % N is the source,
        postcondition rank == index // N is the destination).
      - SRC-MAJOR, allgather(N) / broadcast(N, root) / gather(N, root):
        base = src_dense, c = sc. The destination is NOT packed into the label,
        either because there is only one (gather's root) or because every GPU is
        one (the replicating collectives).

    The dst-major/src-major split is the collective's own chunk ADDRESSING, not
    the schedule format (which keys on the formulation, see is_lp_format).

    Replicating collectives work because src-major labelling makes them work:
    two demands whose paths share a physical hop compute the *same* chunk_id
    there, so the hop is one send that both demands read, and ncclize's
    grouped_sends (a set) dedupes the duplicate. That holds as long as each
    demand's paths carry the whole chunk, which the hierarchical solver's
    sub-chunk refinement guarantees (every volume is exactly 1.0). LABEL_RE-style
    consistency is asserted below rather than assumed, so a fractional
    replicating schedule fails loudly instead of emitting a wrong XML.

    Returns the parse_flows() 8-tuple plus flow_rates and a trailing root_dense
    (the dense rank of the root for rooted collectives, else None). The second
    element is the chunk_up factor S * M.
    """
    if collective_name not in ('alltoall', 'gather', 'allgather', 'broadcast'):
        raise ValueError(f"Unknown LP collective {collective_name!r}")
    dst_major = collective_name == 'alltoall'
    rooted = collective_name in ('gather', 'broadcast')

    root_raw = schedule.get('0-Root')
    if rooted and root_raw is None:
        raise ValueError(
            f"{collective_name} schedule is missing the '0-Root' field needed to "
            f"identify the root GPU; re-run the solver to emit it.")

    M = _compute_subdivision(schedule)

    # First pass: collect the id universe and a structured per-demand view.
    raw_ids = set()
    switch_raw_ids = set()
    demands = []  # (dst_raw, chunk, src_raw, met_epoch, [(volume_fraction, hops), ...])
    for demand_key, paths in schedule['8-Chunk paths'].items():
        dm = DEMAND_RE.match(demand_key)
        if not dm:
            raise ValueError(f'Could not parse demand key: {demand_key!r}')
        dst_raw, chunk, src_raw, met_epoch = (int(x) for x in dm.group(1, 2, 3, 4))
        raw_ids.update((dst_raw, src_raw))

        path_records = []
        for path in paths:
            hops = []      # (hop_epoch, hop_src_raw, hop_dst_raw, path_key, rate)
            volumes = []
            for _epoch, segment in path:
                sm = PATH_SEGMENT_RE.match(segment)
                if not sm:
                    raise ValueError(f'Could not parse path segment: {segment!r}')
                hop_src_raw, hop_dst_raw = int(sm.group(1)), int(sm.group(2))
                hop_epoch = int(sm.group('epoch'))
                path_key = _parse_switch_path(sm.group('switches'))
                rate = sm.group('rate')
                raw_ids.update((hop_src_raw, hop_dst_raw))
                if path_key:
                    switch_raw_ids.update(path_key)
                if port_qualify is not None:      # see the note in parse_flows()
                    path_key = port_qualify(hop_src_raw, hop_dst_raw, path_key,
                                            src_raw, chunk, hop_epoch)
                volumes.append(_segment_volume(sm))
                hops.append((hop_epoch, hop_src_raw, hop_dst_raw, path_key,
                             None if rate is None else float(rate)))

            # Flow is conserved along a route, so every hop of one path carries
            # the same volume; that shared value is the fraction of the chunk
            # this path delivers.
            v0 = volumes[0]
            for v in volumes[1:]:
                if abs(v - v0) > 1e-6:
                    raise ValueError(
                        f'Path in {demand_key!r} has inconsistent per-hop '
                        f'volumes {volumes}.')
            hops.sort(key=lambda h: h[0])  # order hops by (start) epoch
            path_records.append(
                (Fraction(v0).limit_denominator(MAX_DENOM), hops))
        demands.append((dst_raw, chunk, src_raw, met_epoch, path_records))

    rank_map = {raw: idx for idx, raw in enumerate(sorted(raw_ids))}
    switch_rank_map = {raw: idx for idx, raw in enumerate(sorted(switch_raw_ids))}
    num_nodes = len(rank_map)

    root_dense = None
    if rooted:
        if root_raw not in rank_map:
            raise ValueError(
                f"{collective_name} root {root_raw} never appears as a flow endpoint "
                f"in the schedule; cannot map it to a dense rank.")
        root_dense = rank_map[root_raw]

    # S = number of sub-chunks per demand, decoded from the chunk label:
    #   dst-major packs it above the dense destination (c = subchunk * N + dest),
    #   src-major uses the label directly (c = subchunk).
    # The chunk_up factor is S * M (S real sub-chunks x M volume pieces each).
    if dst_major:
        num_subchunks = max(chunk for _, chunk, _, _, _ in demands) // num_nodes + 1
    else:
        num_subchunks = max(chunk for _, chunk, _, _, _ in demands) + 1
    factor = num_subchunks * M

    # Second pass: assign disjoint piece ranges per demand and emit sends.
    # epoch -> {(chunk_id, src_dense, dst_dense): None}, a dict used as an
    # insertion-ordered set. Dedupe is needed because src-major addressing makes N
    # replicating demands label one shared hop identically (that is the point); it
    # must preserve first-seen order so flow-id assignment, which walks these in
    # order, is unchanged for schedules that have no duplicates to remove.
    by_epoch = defaultdict(dict)
    raw_flows = []                 # (epoch, chunk_id, src, dst, path_key, completion, rate)
    # (epoch, src, dst, path_key, base) -> {demand_key: the piece ranges that demand
    # puts on the hop}. The invariant is NOT "one label per hop": one hop legitimately
    # carries many chunks (different `base`s) in an epoch, and one demand's several
    # multipath routes may each cross the same hop with a different slice. What must
    # hold is that two DIFFERENT demands for the SAME data (same base) crossing the
    # same hop slice it identically -- otherwise the same bytes are labelled two ways.
    # Under dst-major no two demands ever share a base, so this is vacuous; under
    # src-major that sharing is the norm (see the docstring) and the dedupe below
    # collapses it to a single send. With whole-chunk volumes every range is [0, 1),
    # so this is the guard for a future fractional replicating schedule.
    hop_pieces = defaultdict(lambda: defaultdict(set))
    for dst_raw, chunk, src_raw, met_epoch, path_records in demands:
        dst_dense = rank_map[dst_raw]
        src_dense = rank_map[src_raw]
        if dst_major:
            subchunk = chunk // num_nodes
            # Chunk labels index the destination by its *dense* GPU rank (0..N-1),
            # which differs from the raw node id when switches occupy interior
            # indices (e.g. incast, where GPU ids skip the switch).
            if chunk % num_nodes != dst_dense:
                raise ValueError(
                    f'Alltoall chunk label {chunk} implies dense destination '
                    f'{chunk % num_nodes} but demand is at raw {dst_raw} '
                    f'(dense {dst_dense}).')
            base = (dst_dense * num_nodes + src_dense) * factor + subchunk * M
        else:
            # src-major: the base index is the source; the destination is not part
            # of the label (there is only one, or every GPU is one).
            subchunk = chunk
            if rooted and dst_dense != root_dense:
                raise ValueError(
                    f'{collective_name} demand is at dense rank {dst_dense} (raw '
                    f'{dst_raw}) but the root is dense {root_dense} (raw {root_raw}); '
                    f'every {collective_name} demand must terminate at the root.')
            base = src_dense * factor + subchunk * M
        cumulative = Fraction(0)
        for v_frac, hops in path_records:
            lo = cumulative * M
            cumulative += v_frac
            hi = cumulative * M
            if lo.denominator != 1 or hi.denominator != 1:
                raise ValueError(
                    f'Non-integral piece range {lo}..{hi} (M={M}) for demand '
                    f'dest {dst_raw} src {src_raw}; volume {v_frac} does not '
                    f'divide the subdivision.')
            lo, hi = int(lo), int(hi)
            for h_idx, (hop_epoch, hsrc_raw, hdst_raw, path_key, rate) in enumerate(hops):
                hsrc, hdst = rank_map[hsrc_raw], rank_map[hdst_raw]
                # A relay forwards only after it has received, so the next hop's
                # start epoch is this hop's completion; the last hop completes
                # when the demand is met.
                if h_idx + 1 < len(hops):
                    completion_epoch = hops[h_idx + 1][0]
                else:
                    completion_epoch = met_epoch
                hop_pieces[(hop_epoch, hsrc, hdst, path_key, base)][
                    (dst_raw, chunk, src_raw)].add((lo, hi))
                for p in range(lo, hi):
                    chunk_id = base + p
                    by_epoch[hop_epoch][(chunk_id, hsrc, hdst)] = None
                    raw_flows.append(
                        (hop_epoch, chunk_id, hsrc, hdst, path_key, completion_epoch, rate))
        if cumulative != 1:
            raise ValueError(
                f'Paths for demand dest {dst_raw} src {src_raw} carry total '
                f'volume {cumulative}, expected exactly 1.')

    for (hop_epoch, hsrc, hdst, _pk, base), by_demand in sorted(hop_pieces.items()):
        if len(by_demand) < 2:
            continue
        slices = {frozenset(v) for v in by_demand.values()}
        if len(slices) > 1:
            offenders = sorted(by_demand)[:2]
            raise ValueError(
                f'Hop {hsrc}->{hdst} (dense) in epoch {hop_epoch} carries different '
                f'pieces of chunk base {base} for demands {offenders[0]} and '
                f'{offenders[1]}: {sorted(by_demand[offenders[0]])} vs '
                f'{sorted(by_demand[offenders[1]])}. The same bytes would be labelled '
                f'two ways. (Only reachable for a replicating collective whose demands '
                f'do not carry whole chunks -- refine the schedule so every volume '
                f'is 1.0.)')

    sorted_epochs = sorted(by_epoch)
    steps_in_order = [list(by_epoch[epoch]) for epoch in sorted_epochs]
    epoch_to_step_idx = {epoch: idx for idx, epoch in enumerate(sorted_epochs)}

    flow_path_keys = {}
    flow_completion_epochs = {}
    flow_rates = {}
    # Per-send link-occupancy window, keyed at OP granularity (one send op per
    # (step, src, dst, path_key); its many chunk pieces share it). start is the
    # send's fine start epoch; finish = start + volume/rate is when the SENDER is
    # done pushing and the uplink frees -- deliberately NOT completion_epoch, which
    # is the data's ARRIVAL (later, includes relay latency) and is the wrong clock
    # for uplink pacing. Under the level's fill-one-epoch rate rule the duration is
    # exactly m; a sub-rate (multi-epoch) send yields a proportionally larger finish,
    # which is what makes the gate robust to that case.
    chunk_size = schedule.get('9-Chunk_Size', 1.0)
    delta = schedule['1-Epoch_Duration']
    paced_sends = {}   # (step_idx, src, dst, path_key) -> (start_fine, finish_fine)
    for hop_epoch, chunk_id, src, dst, path_key, completion_epoch, rate in raw_flows:
        step_idx = epoch_to_step_idx[hop_epoch]
        key = (step_idx, chunk_id, src, dst)
        flow_path_keys[key] = path_key
        flow_completion_epochs[key] = completion_epoch
        if rate is not None:
            prev = flow_rates.setdefault(key, rate)
            if abs(prev - rate) > 1e-9:
                raise ValueError(
                    f'Flow {src}->{dst} chunk {chunk_id} at step {step_idx} carries '
                    f'two different rates ({prev} and {rate}).')
            duration = max(1, round(chunk_size / (M * rate * delta)))
            paced_sends[(step_idx, src, dst, path_key)] = (hop_epoch, hop_epoch + duration)

    pacing_gates = _finish_before_start_gates(paced_sends)

    return (num_nodes, factor, steps_in_order, flow_path_keys, switch_rank_map,
            sorted_epochs, flow_completion_epochs, flow_rates, root_dense,
            pacing_gates, rank_map)


def build_switch_routes(flow_manifest, switch_rank_map, programmable_switches=None,
                        physical_port=None, gpu_rank_map=None):
    """Build a per-switch flow_id -> next-hop forwarding table.

    flow_manifest is the list of {'flow_id', 'step', 'src', 'dst', 'path_key'}
    records recorded by ncclize(). Since a flow_id is a bijection with a
    physical route (src, dst, path_key), the same flow_id can appear in several
    records (one per epoch/chunk that used the route); they all describe the
    identical forwarding, so building the table is idempotent per flow_id. The
    'step' (epoch) field is intentionally NOT carried into the table: forwarding
    is a property of the route, not of when it is used, so a route-level entry
    has no single meaningful epoch.

    The table covers the PACED (network) routes only, because ncclize() emits a
    flow id only for those. That is the intended scope, not a gap: an unpaced hop
    is an intra-cell one crossing an NVSwitch, which routes on its own and is
    never programmed from this table. A hierarchical schedule therefore yields
    entries for its inter-cell switches and none for the per-cell NVSwitches.

    programmable_switches, when given, is the set of RAW switch node ids that
    accept an external forwarding program (Topology.programmable_switch_indices).
    Only those get a table; every other switch on a route is treated as a
    transparent self-routing hop and skipped, so a route
    gpu -> nvswitch -> leaf -> spine -> leaf -> nvswitch -> gpu yields entries on
    the three network switches only, chained leaf -> spine -> leaf -> dst gpu as
    if the NVSwitches were not there. That is the correct program for such a
    fabric: the NVSwitch delivers by its own addressing, so the last programmable
    switch's next hop is the destination GPU. A route whose switches are all
    non-programmable contributes nothing. None => every switch is programmable
    (the previous behaviour).

    THE ENTRY IS A FLOW ID -> EGRESS PORT MAPPING. 'port' is the only key a
    runtime needs and the only one that is load-bearing: a switch that sees a
    packet tagged with this flow id sends it out of that physical port, and
    every other key in the entry is there to be read by a human. 'port' is the
    port number on THIS switch, in its own 0..radix-1 numbering
    (Topology.port_map), of the cable the route leaves on -- not a link-local
    index and not the next hop's port.

    physical_port(node, neighbor, cable) supplies those numbers, normally
    Topology.physical_port. Without it (no --topology) the entries carry no
    'port' at all rather than a guessed one, since a port number that was not
    read off a topology is a fiction. gpu_rank_map (raw GPU id -> dense) is
    needed alongside it because the LAST hop's port faces the destination GPU,
    and a port lookup is on raw node ids while the manifest carries dense ones.

    The remaining keys are debug: next_hop / next_hop_type name where the packet
    is going in the same dense 0-indexed numbering used elsewhere (GPU ids
    matching the main XML's, switch ids ranked over the switches this table
    covers), with next_hop_type ('switch' or 'gpu') disambiguating the two since
    they are independently 0-indexed and can otherwise collide numerically;
    src_gpu / dst_gpu name the route's endpoints. A reader can check any 'port'
    against them; a runtime never has to.

    Returns {'switch_id_map': {switch_id_str: raw_node_id},
             'switches': {switch_id_str: {flow_id_str: {...}}}}. switch_id_map
    records the dense id -> raw fine node id correspondence, which is no longer
    inferable from the schedule alone once the table covers a subset of the
    switches.
    """
    # Rank over the switches this table actually covers, so its ids stay dense
    # 0..k-1 rather than inheriting holes where a filtered-out switch sat.
    if programmable_switches is None:
        rank_map = dict(switch_rank_map)
    else:
        programmable_switches = set(programmable_switches)
        rank_map = {raw: idx for idx, raw in
                    enumerate(sorted(s for s in switch_rank_map
                                     if s in programmable_switches))}

    raw_gpu = ({dense: raw for raw, dense in gpu_rank_map.items()}
               if gpu_rank_map is not None else None)

    routes = defaultdict(dict)
    for record in flow_manifest:
        raw_key, cables = unqualify_path_key(record['path_key'])
        if not raw_key:
            continue  # direct GPU-GPU link, no switch hop involved
        # Keep each surviving switch's ORIGINAL index. Its egress port is a property of the
        # RAW wire leaving it, which is unaffected by whether the switches after it were
        # filtered out as self-routing: chaining past a skipped switch changes where the packet
        # ends up, not which socket it left this one by.
        kept = [(i, rank_map[s]) for i, s in enumerate(raw_key) if s in rank_map]
        if not kept:
            continue  # entirely self-routing (e.g. an intra-node NVSwitch hop)
        switch_path = [dense for _, dense in kept]
        flow_id, src, dst = (
            record['flow_id'], record['src'], record['dst'])
        # The raw node sequence the wires actually follow. `dst` is dense, so the last hop's
        # far end has to be translated back before it can be looked up as a port.
        raw_dst = None if raw_gpu is None else raw_gpu.get(dst)
        raw_path = (raw_key + (raw_dst,)) if raw_dst is not None else None
        for i, (orig_idx, switch) in enumerate(kept):
            is_last = i == len(switch_path) - 1
            next_hop_type = 'gpu' if is_last else 'switch'
            next_hop = dst if is_last else switch_path[i + 1]
            entry = {}
            if physical_port is not None and raw_path is not None:
                # `cables` carries one entry per HOP, so hop orig_idx+1 is the one LEAVING this
                # switch; an unqualified key touches no multi-port link, so every cable is 0.
                cable = cables[orig_idx + 1] if cables is not None else 0
                port = physical_port(raw_key[orig_idx], raw_path[orig_idx + 1], cable)
                if port is not None:
                    entry['port'] = port
            entry.update({
                'next_hop_type': next_hop_type,
                'next_hop': next_hop,
                'src_gpu': src,
                'dst_gpu': dst,
            })
            routes[switch][flow_id] = entry

    return {
        'switch_id_map': {str(dense): raw for raw, dense in sorted(
            rank_map.items(), key=lambda kv: kv[1])},
        'switches': {
            str(switch): {str(flow_id): entry for flow_id, entry in flows.items()}
            for switch, flows in sorted(routes.items())
        }
    }


class TeCCLTopology:
    """Minimal topology shim satisfying the interface ncclize()/Algorithm need.

    `links[dst][src]` is built from the schedule -- the max number of times a given src->dst edge
    is used within any single epoch (counted on the raw per-chunk list, BEFORE make_intervals
    merges contiguous chunks into one op). Historically that integer did DOUBLE DUTY: the channel
    allocator round-robins each edge's flows across `link(src,dst)` channels
    (_allocate_channels_match_topology), AND Algorithm.make_implementation checks it as a bandwidth
    capacity (bandwidth_constraints -> `util <= bw * rounds`).

    That coupling both OVER-ALLOCATES channels (the pre-merge concurrency count, not the physical
    link parallelism) and pins the channel count to the bandwidth demand. But the bandwidth check
    is TAUTOLOGICAL here: `bw` is set to the max-over-epochs concurrency and `util` is that same
    concurrency with rounds=1, so `util <= bw*1` holds by construction and catches nothing. The
    REAL capacity guarantees live in TE-CCL's own rate-based asserts against the real fine topology
    (reconstruct._assert_rate_within_capacity, stitch.assert_link_capacity), which model the
    rate-paced runtime the round-based taccl check cannot.

    So in PHYSICAL mode (a real fine Topology is fed in, `physical_replicas=True`) we:
      * report `link(src,dst) = 1` for each used edge -- the physical point-to-point parallelism
        (TE-CCL topologies model bandwidth, not multi-rail counts), so channels = one per distinct
        switch path, no per-chunk multiplier; same-path flows sharing a channel stay correctly
        ordered (same route => departure order == arrival order), so this is correct, not just
        smaller; and
      * emit NO bandwidth_constraints, skipping the tautological self-check (TE-CCL already
        verified real capacity upstream). This is what lets `link=1` stand without tripping
        `util <= bw*rounds`.
    Default (no real topology, e.g. a flat single-level schedule) keeps the schedule-inferred
    counts and the (harmless, tautological) check, so those paths are byte-for-byte unchanged.
    """

    def __init__(self, name, num_nodes, steps, physical_replicas=False):
        self.name = name
        self._num_nodes = num_nodes
        self.physical_replicas = physical_replicas
        self.switches = []  # TE-CCL's switch hops are not modeled as separate
                             # topology elements; see note in parse_flows().

        # links[dst][src] = per-epoch concurrency on that edge (used edges are those with > 0).
        self.links = [[0] * num_nodes for _ in range(num_nodes)]
        for sends in steps:
            per_epoch_count = defaultdict(int)
            for _, src, dst in sends:
                per_epoch_count[(src, dst)] += 1
            for (src, dst), count in per_epoch_count.items():
                self.links[dst][src] = max(self.links[dst][src], count)

    def num_nodes(self):
        return self._num_nodes

    def link(self, src, dst):
        used = self.links[dst][src]
        if self.physical_replicas:
            # One physical link per used edge; bandwidth is handled by rate-pacing + TE-CCL's own
            # capacity asserts, not by this count.
            return 1 if used > 0 else 0
        return used

    def bandwidth_constraints(self):
        if self.physical_replicas:
            # Tautological here and redundant with TE-CCL's rate-based capacity asserts; skipping it
            # is what lets the physical link count (1) stand instead of the concurrency count.
            return
        for dst, dst_links in enumerate(self.links):
            for src, lk in enumerate(dst_links):
                if lk > 0:
                    yield ([src], [dst], lk, f'{src}->{dst}')


def _build_collective(collective_name, num_nodes, root_dense):
    """The taccl collective for a schedule's collective identity.

    Kept separate from the parser so both format branches dispatch the same way:
    the format (LP vs MILP) and the collective identity are independent axes, and
    the MILP branch's old hardcoded allgather() was an artifact of
    scheduler._resolve_formulation defaulting ALLGATHER -> MILP.
    """
    from taccl_collectives import allgather, alltoall, broadcast, gather

    if collective_name == 'alltoall':
        return alltoall(num_nodes)
    if collective_name == 'allgather':
        return allgather(num_nodes)
    if collective_name in ('gather', 'broadcast'):
        if root_dense is None:
            raise ValueError(
                f"{collective_name} needs a root; the schedule has no '0-Root' field.")
        ctor = gather if collective_name == 'gather' else broadcast
        return ctor(num_nodes, root_dense)
    raise NotImplementedError(
        f"No taccl collective mapping for {collective_name!r}")


def _build_port_qualifier(schedule, topology, lp_format):
    """Run the post-solve port split and return (qualify, assignment), or (None, None).

    Returns None unless the topology actually declares a multi-port link, so every topology
    today -- none of which do -- takes exactly the path it takes now, with no port machinery in
    the emitted output at all.

    The split is computed once, here, from the schedule and the real fine topology, and the
    resulting per-(flow, link) port is folded into the path key at parse time. That makes the
    port part of a route's IDENTITY, which is what the two consumers need: the channel allocator
    partitions each edge's flows by path key, and the flow id is a bijection with (src, dst,
    path key), so two flows down the same switch sequence on different ports get different flow
    ids and therefore different forwarding entries.

    The capacity assert runs before anything is emitted: if the split cannot fit the schedule
    onto real ports, that is a fact about the topology and the solve, and it should stop the
    emission rather than silently produce a program that overruns a port.
    """
    if topology is None or not getattr(topology, 'ports', None):
        return None, None
    try:                                  # sibling module; this file runs both as a script
        from port_split import (Flow, assign_ports, assert_port_capacity,   # noqa: E402
                                flow_loads, occupancy_grid)
    except ImportError:                   # ... and as part of the teccl package
        from teccl.ncclize.port_split import (Flow, assign_ports, assert_port_capacity,
                                              flow_loads, occupancy_grid)

    subdivision = _compute_subdivision(schedule) if lp_format else 1
    loads = flow_loads(schedule, occupancy_grid(schedule, subdivision))
    assignment = assign_ports(loads, topology.port_count, topology.port_capacity)
    assert_port_capacity(loads, assignment, topology.port_count, topology.port_capacity)

    def qualify(src_raw, dst_raw, path_key, origin, chunk, epoch):
        """The path key for one send: its SUBFLOW's key, or the route unchanged.

        The piece address only selects WHICH subflow -- it is never part of the key. A flow the
        packing kept whole has one subflow, so its key is the same for every piece and the
        `(channel, peer)` connection stays single; only a partitioned flow resolves to more than
        one, and those are genuinely different routes on different wires.

        Returned unchanged when the route touches no multi-port link, so a topology declaring no
        ports emits byte-identical keys.
        """
        flow = Flow(src_raw, tuple(path_key or ()), dst_raw)
        if not any(topology.port_count(*h) > 1 for h in flow.hops()):
            return path_key
        try:
            return assignment.subflow_of(flow, origin, chunk, epoch).key
        except KeyError:
            raise AssertionError(
                f"chunk {chunk} from {origin} in epoch {epoch} on route {flow} appears in the "
                f"schedule section this parser reads but not in '7-Flows', which the port "
                f"split was computed from; the two sections disagree about what exists")

    return qualify, assignment


def build_algorithm(schedule, name='teccl', topology=None):
    """Build the taccl Algorithm from a TE-CCL schedule.

    `topology` is the real fine Topology, when the caller has it (the hierarchical driver passes
    the object; the CLI constructs it from --topology). Supplying it switches TeCCLTopology into
    PHYSICAL mode: one channel replica per used link and no (tautological) taccl bandwidth check --
    the fix for the channel over-allocation. Omitted (None) keeps the schedule-inferred behaviour
    every flat single-level schedule relies on.
    """
    from taccl_algorithm import Algorithm, Step
    from taccl_instance import Instance
    from helpers import build_gpu_epoch_view

    # The collective *identity* (which taccl collective to build) and the
    # schedule *format* (which parser to run) are independent: the LP now solves
    # several collectives in one nested format, while the allgather MILP has its
    # own flat format.
    collective_name = detect_collective(schedule)
    lp_format = is_lp_format(schedule)
    port_qualify, _port_assignment = _build_port_qualifier(schedule, topology, lp_format)

    if lp_format:
        (num_nodes, factor, steps_in_order, flow_path_keys, switch_rank_map,
         sorted_epochs, flow_completion_epochs, flow_rates, root_dense,
         pacing_gates, gpu_rank_map) = parse_flows_lp(schedule, collective_name, port_qualify)
        collective = _build_collective(collective_name, num_nodes, root_dense)
    else:
        # parse_flows labels chunks src-major (rank_map[origin] * S + subchunk), so
        # it cannot express a collective whose chunk identity includes the
        # destination. In practice the MILP only ever solves allgather; fail loudly
        # rather than mislabel if that ever changes.
        if collective_name == 'alltoall':
            raise NotImplementedError(
                "MILP-format schedules use src-major chunk labels, which cannot "
                "represent alltoall's destination-major chunk identity.")
        (num_nodes, factor, steps_in_order, flow_path_keys, switch_rank_map,
         sorted_epochs, flow_completion_epochs, gpu_rank_map) = parse_flows(schedule, port_qualify)
        flow_rates = {}
        collective = _build_collective(collective_name, num_nodes,
                                       schedule.get('0-Root'))
        # Flat single-level schedule: every send occupies its epoch for exactly one
        # epoch, so the link-occupancy finish is start+1 for all of them. Same
        # finish-before-start rule as the LP path -> gate the first send of each epoch
        # on the previous epoch's send, never same-epoch sends.
        flat_sends = {}
        for step_idx, sends in enumerate(steps_in_order):
            start = sorted_epochs[step_idx]
            for chunk_id, src, dst in sends:
                flat_sends[(step_idx, src, dst, None)] = (start, start + 1)
        pacing_gates = _finish_before_start_gates(flat_sends)

    gpu_epoch_view = build_gpu_epoch_view(
        steps_in_order, sorted_epochs, num_nodes, flow_completion_epochs)

    # `factor` is the chunk_up() multiplier. For the allgather MILP it is
    # num_subchunks (each source's data pre-split into that many real chunks by
    # the solver). For the LP it is S * M: S real sub-chunks per demand times the
    # M-way volume subdivision this converter introduced to make fractional
    # volumes integral.
    if factor > 1:
        collective = collective.chunk_up(factor)

    if topology is not None:
        # A real fine topology was supplied: sanity-check it against the schedule, then run
        # TeCCLTopology in physical mode (one channel per link, no tautological bandwidth check).
        # The schedule's GPUs are exactly the non-switch nodes of the fine topology; a mismatch
        # means the wrong --topology was passed, which would silently mis-scale channels.
        real_gpus = len(topology.capacity) - len(topology.switch_indices)
        if num_nodes != real_gpus:
            raise ValueError(
                f"topology {type(topology).__name__} has {real_gpus} non-switch GPU(s) but the "
                f"schedule has {num_nodes}; wrong topology for this schedule.")
        topology = TeCCLTopology(name, num_nodes, steps_in_order, physical_replicas=True)
    else:
        topology = TeCCLTopology(name, num_nodes, steps_in_order)

    steps = [Step(1, sends) for sends in steps_in_order]

    instance = Instance(steps=len(steps), extra_rounds=0, chunks=factor)

    algo = Algorithm.make_implementation(
        collective, topology, instance, steps, cont=False, suffix='-teccl')

    # Physical send rate (GB/s) per unit piece sent, so a merged op of `cnt` pieces
    # gets rate = cnt * piece_rate.
    #
    # PREFERRED: a per-flow rate carried in the schedule ("at rate R" on a segment).
    # The level of the solve that produced a flow is the only place that knows its
    # own epoch duration and capacity model, so it computes the rate and emits it;
    # this function must not re-derive one. A hierarchical schedule mixes flows from
    # several levels -- its epoch axis is the FINEST level's, so a single global
    # rate derived from "1-Epoch_Duration" would be wrong for every other level --
    # and flows a level chose not to pace carry no rate at all.
    #
    # LEGACY fallback, for flat single-level schedules with no per-flow rate: every
    # "chunk"/"sub-chunk" is one chunk_size, and the LP converter split each into M
    # volume pieces of chunk_size/M, so volume v -> round(v*M) pieces gives
    # v*chunk_size/epoch back. In the allgather MILP format each "Chunk C from S"
    # piece is already a full chunk_size (no extra 1/M division).
    if flow_rates:
        piece_rate = flow_rates
    else:
        epoch_duration = schedule['1-Epoch_Duration']
        chunk_size = schedule.get('9-Chunk_Size', 1.0)
        if lp_format:
            piece_rate = chunk_size / (_compute_subdivision(schedule) * epoch_duration)
        else:
            piece_rate = chunk_size / epoch_duration

    return (algo, flow_path_keys, switch_rank_map, gpu_epoch_view, piece_rate,
            pacing_gates, gpu_rank_map)


def load_topology(name, chunk_size=1.0):
    """Construct a real fine Topology by class name, for --topology.

    The fine topologies are all constructible from just their name + chunk_size (the structure is
    fixed in the class); chunk_size does not affect the link structure the channel allocator reads,
    so the default is fine. The hierarchical driver, which already holds the Topology object, should
    pass it to build_algorithm directly instead of going through here.
    """
    # This module is usually run as a script with teccl/ncclize on sys.path (for the taccl_* imports),
    # so the `teccl` package root is not importable yet -- add the repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from teccl.input_data import TopologyParams
    from teccl.topologies.topology import Topology

    # name -> "module:ClassName" for the topologies that can appear at the ncclize boundary.
    registry = {
        'HeteroTaperedCluster': 'hetero_tapered_cluster:HeteroTaperedCluster',
        'DualPlaneHeteroCluster': 'dual_plane_hetero_cluster:DualPlaneHeteroCluster',
        'DualPlaneHeteroClusterScattered': 'dual_plane_hetero_cluster:DualPlaneHeteroClusterScattered',
        'RailOptimizedSpineLeaf': 'rail_optimized_spine_leaf:RailOptimizedSpineLeaf',
        'TwoPodRail': 'two_pod_rail:TwoPodRail',
        'TwoPodRailHostBound': 'two_pod_rail:TwoPodRailHostBound',
        'TwoPodRailSplitPorts': 'two_pod_rail:TwoPodRailSplitPorts',
        'NestedCluster': 'nested_cluster:NestedCluster',
        'FatTreePod': 'fat_tree_pod:FatTreePod',
        'FatTreePodSingleSpine': 'fat_tree_pod_single_spine:FatTreePodSingleSpine',
        'DGX1': 'dgx1:DGX1',
        'DGX2': 'dgx2:DGX2',
        'NDv2': 'ndv2:NDv2',
        'Mesh': 'mesh:Mesh',
        'Star': 'star:Star',
        'IncastSwitch': 'incast_switch:IncastSwitch',
    }
    if name not in registry:
        raise ValueError(
            f"unknown --topology {name!r}; known: {', '.join(sorted(registry))}. "
            f"(Or call build_algorithm(schedule, topology=<Topology instance>) directly.)")
    module_name, class_name = registry[name].split(':')
    import importlib
    module = importlib.import_module(f'teccl.topologies.{module_name}')
    cls = getattr(module, class_name)
    topo = cls(TopologyParams(name=name, chunk_size=chunk_size))
    assert isinstance(topo, Topology)
    return topo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule', help='TE-CCL schedule JSON file')
    p.add_argument('-o', '--output', required=True, help='output XML file')
    p.add_argument('--topology', default=None,
                    help='name of the real fine Topology this schedule runs on (e.g. '
                         'HeteroTaperedCluster). When given, the channel allocator uses physical '
                         '(one-per-link) channel counts and the tautological taccl bandwidth check '
                         'is skipped (TE-CCL verifies real capacity upstream). Omit for a flat '
                         'single-level teccl solve.')
    p.add_argument('--instances', type=int, default=1)
    p.add_argument('--scale-remote', type=int, default=1)
    p.add_argument('--switch-routing-output', default=None,
                    help='optional path to write per-switch flow_id -> next-hop routing table as JSON')
    p.add_argument('--programmable-switches', default=None,
                    help='comma-separated RAW switch node ids to emit forwarding entries for; '
                         'switches outside the set are treated as transparent self-routing hops. '
                         'Defaults to the --topology class\'s programmable_switch_indices (e.g. '
                         'RailOptimizedSpineLeaf: leaf+spine only, no NVSwitches), or to every '
                         'switch when no --topology is given.')
    p.add_argument('--epoch-debug-output', default=None,
                    help='optional path to write a human-readable per-GPU, '
                         'per-epoch schedule dump. The realizability feasibility '
                         'check (and its warnings) runs regardless of this flag.')
    p.add_argument('--no-rate', action='store_true',
                    help='omit the per-op rate attribute from the emitted XML. '
                         'The schedule is computed identically; only the final '
                         'written output differs, to enable regression testing '
                         'against outputs that predate the rate field.')
    p.add_argument('--hierarchical', action='store_true',
                    help='this schedule came from the hierarchical (multi-level) solver, so the '
                         'flat-axis realizability feasibility check does not apply -- its epoch '
                         'axis interleaves per-level grids and the network layer is checked '
                         'per-layer in the stitch instead. Omit for an old-style flat single-level '
                         'teccl solve (MILP or LP), where the flat-axis check is run.')
    args = p.parse_args()

    from taccl_ncclize import ncclize, ChannelPolicy
    from helpers import (check_epoch_ordering_feasibility,
                         warn_epoch_ordering_violations, write_gpu_epoch_debug)

    with open(args.schedule) as f:
        schedule = json.load(f)

    real_topology = load_topology(args.topology) if args.topology else None
    (algo, flow_path_keys, switch_rank_map,
     gpu_epoch_view, piece_rate, pacing_gates, gpu_rank_map) = build_algorithm(
        schedule, topology=real_topology)

    # Send pacing is enforced INSIDE ncclize by realizing the pacing_gates manifest (per-flow
    # finish-before-start edges derived here in teccl_ncclize), so there is no XML post-pass.
    #
    # The flat-axis feasibility check only makes sense for a SINGLE-LEVEL (flat) schedule, where the
    # whole schedule shares one epoch grid, so "the preceding epoch has a send" genuinely means "the
    # send is paced to its epoch". A HIERARCHICAL (multi-level) schedule interleaves levels with
    # different epoch lengths on one fine axis -- a coarse network send legitimately sits m fine
    # epochs after the previous one -- so this check would false-positive on that intended sparsity;
    # the network layer's realizability is reported per-layer by the stitch
    # (teccl.hierarchy.flat_schedule.check_network_pacing) at solve time instead.
    #
    # This is a property of HOW the schedule was solved (flat vs hierarchical), NOT of the schedule
    # FORMAT (MILP-flat vs LP-nested): a flat single-level solve using the LP formulation is still
    # flat and still wants the check. So the caller states it explicitly with --hierarchical rather
    # than us inferring it from is_lp_format.
    if args.hierarchical:
        violations = None
    else:
        violations = check_epoch_ordering_feasibility(gpu_epoch_view)
        warn_epoch_ordering_violations(violations)

    # flow_manifest drives switch routing (one entry per route).
    flow_manifest = []

    xml = ncclize(
        algo,
        channel_policy=ChannelPolicy.MatchTopology,
        old_format=True,
        use_scratch=True,
        instances=args.instances,
        scale_remote=args.scale_remote,
        flow_path_keys=flow_path_keys,
        flow_manifest=flow_manifest,
        piece_rate=piece_rate,
        pacing_gates=pacing_gates,
        logging=True,
    )

    if args.no_rate:
        # Strip the per-op rate attribute from the already-serialized XML so the
        # schedule computation stays untouched; only the written output changes.
        xml = re.sub(r'\s+rate="[^"]*"', '', xml)

    with open(args.output, 'w') as f:
        f.write(xml)
    print(f'Wrote {args.output}')

    if args.epoch_debug_output:
        write_gpu_epoch_debug(args.epoch_debug_output, gpu_epoch_view,
                              violations, source=args.schedule)
        print(f'Wrote {args.epoch_debug_output}')

    if args.switch_routing_output:
        # Explicit --programmable-switches wins; otherwise the topology declares it. With no
        # topology at all there is nothing to filter by, so every switch stays programmable.
        if args.programmable_switches is not None:
            programmable = {int(s) for s in args.programmable_switches.split(',') if s.strip()}
        elif real_topology is not None:
            programmable = set(real_topology.programmable_switch_indices)
        else:
            programmable = None
        # The topology is what knows a port number; without --topology the table is emitted
        # with its debug fields and no 'port', which is the honest output for "nobody said".
        routes = build_switch_routes(
            flow_manifest, switch_rank_map, programmable,
            physical_port=(real_topology.physical_port if real_topology is not None else None),
            gpu_rank_map=gpu_rank_map)
        with open(args.switch_routing_output, 'w') as f:
            json.dump(routes, f, indent=2)
        print(f'Wrote {args.switch_routing_output}')


if __name__ == '__main__':
    main()
