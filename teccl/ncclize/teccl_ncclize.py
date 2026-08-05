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


def parse_flows(schedule):
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
    switch_rank_map, flow_completion_steps, sorted_epochs,
    flow_completion_epochs):
    - steps_in_order is a list of lists of (global_chunk_id, src, dst)
      0-indexed tuples, one list per non-empty epoch, in increasing epoch
      order.
    - sorted_epochs is the parallel list of raw epoch numbers (steps_in_order[i]
      is raw epoch sorted_epochs[i]). Exposed so downstream code can recover the
      true epoch axis -- including gaps where a whole epoch is globally empty
      and thus dropped from steps_in_order.
    - flow_completion_epochs maps (step_idx, global_chunk_id, src, dst) -> the
      raw epoch by which dst actually holds the chunk, the raw-epoch counterpart
      of flow_completion_steps. Used to place a recv at its arrival epoch in the
      per-GPU debug view (a relayed recv can land later than, and even after the
      last, flow-start epoch).
    - flow_path_keys maps (step_idx, global_chunk_id, src, dst) -> path key,
      where step_idx is the index into steps_in_order (not the raw epoch
      number) and the path key is whatever _parse_switch_path() returned for
      that flow (a tuple of raw switch ids, or None for a direct hop). Used
      downstream to avoid merging chunks that took different switch paths
      into a single send op.
    - switch_rank_map maps each raw switch id appearing in any path key to a
      dense 0-indexed id, the same way rank_map does for GPU ids.
    - flow_completion_steps maps (step_idx, global_chunk_id, src, dst) -> a
      dense step-domain value (comparable to step_idx) representing the true
      epoch by which dst actually has the data, as opposed to step_idx which
      only reflects the epoch the transfer started. For a flow relayed
      through switches, "in epoch N" in a 7-Flows line is only the *start*
      epoch (see chunk_flow_path_to_string() in teccl/solvers/allgather.py);
      each switch hop costs at least one additional epoch of store-and-
      forward latency, so a relayed flow's true completion can be several
      epochs later than step_idx implies. That true completion is recorded
      separately, per (destination, chunk, origin) demand, in the schedule's
      "8-Chunk paths" section as "met by epoch E". This is derived from that
      section rather than assumed to be "1 epoch per hop", since that only
      holds when the topology's link-latency parameters are zero.

      Known limitation: this value only corrects ordering *across* threadblock
      groups (e.g. an unrelated send no longer waits behind a switch-relayed
      recv it shares a threadblock with). It does not, and cannot, reorder
      multiple recvs that already share one (gpu, peer, channel) -- switches
      and per-flow paths are invisible to TeCCLTopology (see its docstring),
      so two flows collapsed onto the same virtual src-dst edge keep whatever
      relative order their raw start epochs gave them, regardless of whether
      one of them actually completes later than the other because it took a
      longer/more congested switch path. That's a pre-existing gap in this
      topology abstraction, not something introduced or fixable by this
      value -- properly addressing it would require ncclize to reason about
      per-flow path/congestion information when deciding intra-channel order,
      which it does not do today.
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
    # flow_completion_epochs mirrors flow_completion_steps but keeps the *raw*
    # completion epoch (not the dense step-domain value). It lets the per-GPU
    # debug view place each recv at the epoch the chunk actually *arrives*,
    # rather than the epoch its flow started -- a switch-relayed recv can land
    # several epochs after it was sent, and raw completion epochs (unlike the
    # bisected dense value) can even exceed the last flow-start epoch.
    flow_completion_steps = {}
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
        completion_step = bisect.bisect_left(sorted_epochs, completion_epoch)
        flow_completion_steps[
            (step_idx, chunk_id, rank_map[hop_src_raw], rank_map[hop_dst_raw])
        ] = completion_step
        flow_completion_epochs[
            (step_idx, chunk_id, rank_map[hop_src_raw], rank_map[hop_dst_raw])
        ] = completion_epoch

    return (num_nodes, num_subchunks, steps_in_order, flow_path_keys,
            switch_rank_map, flow_completion_steps, sorted_epochs,
            flow_completion_epochs)


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


def parse_flows_lp(schedule, collective_name):
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
    flow_completion_steps = {}
    flow_completion_epochs = {}
    flow_rates = {}
    for hop_epoch, chunk_id, src, dst, path_key, completion_epoch, rate in raw_flows:
        step_idx = epoch_to_step_idx[hop_epoch]
        key = (step_idx, chunk_id, src, dst)
        flow_path_keys[key] = path_key
        flow_completion_steps[key] = bisect.bisect_left(sorted_epochs, completion_epoch)
        flow_completion_epochs[key] = completion_epoch
        if rate is not None:
            prev = flow_rates.setdefault(key, rate)
            if abs(prev - rate) > 1e-9:
                raise ValueError(
                    f'Flow {src}->{dst} chunk {chunk_id} at step {step_idx} carries '
                    f'two different rates ({prev} and {rate}).')

    return (num_nodes, factor, steps_in_order, flow_path_keys, switch_rank_map,
            flow_completion_steps, sorted_epochs, flow_completion_epochs,
            flow_rates, root_dense)


def build_switch_routes(flow_manifest, switch_rank_map):
    """Build a per-switch flow_id -> next-hop forwarding table.

    flow_manifest is the list of {'flow_id', 'step', 'src', 'dst', 'path_key'}
    records recorded by ncclize(). Since a flow_id is a bijection with a
    physical route (src, dst, path_key), the same flow_id can appear in several
    records (one per epoch/chunk that used the route); they all describe the
    identical forwarding, so building the table is idempotent per flow_id. The
    'step' (epoch) field is intentionally NOT carried into the table: forwarding
    is a property of the route, not of when it is used, so a route-level entry
    has no single meaningful epoch.

    Returns {'switches': {switch_id_str: {flow_id_str: {...}}}}, with switch
    and GPU ids both in the same dense 0-indexed numbering used elsewhere
    (GPU ids matching the main XML's, switch ids via switch_rank_map) and an
    explicit next_hop_type ('switch' or 'gpu') disambiguating the two, since
    they are independently 0-indexed and can otherwise collide numerically.
    """
    routes = defaultdict(dict)
    for record in flow_manifest:
        path_key = record['path_key']
        if not path_key:
            continue  # direct GPU-GPU link, no switch hop involved
        switch_path = tuple(switch_rank_map[s] for s in path_key)
        flow_id, src, dst = (
            record['flow_id'], record['src'], record['dst'])
        for i, switch in enumerate(switch_path):
            is_last = i == len(switch_path) - 1
            next_hop_type = 'gpu' if is_last else 'switch'
            next_hop = dst if is_last else switch_path[i + 1]
            routes[switch][flow_id] = {
                'next_hop_type': next_hop_type,
                'next_hop': next_hop,
                'src_gpu': src,
                'dst_gpu': dst,
            }

    return {
        'switches': {
            str(switch): {str(flow_id): entry for flow_id, entry in flows.items()}
            for switch, flows in sorted(routes.items())
        }
    }


class TeCCLTopology:
    """Minimal topology shim satisfying the interface ncclize()/Algorithm need.

    Link capacities are derived directly from the schedule itself (the max
    number of times a given src->dst edge is used within any single epoch),
    rather than reconstructed from TE-CCL's internal NDv2/DGX/etc. link model,
    so the result is always consistent with whatever schedule is fed in.
    """

    def __init__(self, name, num_nodes, steps):
        self.name = name
        self._num_nodes = num_nodes
        self.switches = []  # TE-CCL's switch hops are not modeled as separate
                             # topology elements; see note in parse_flows().

        # links[dst][src] = number of channels available on that edge
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
        return self.links[dst][src]

    def bandwidth_constraints(self):
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


def build_algorithm(schedule, name='teccl'):
    from taccl_algorithm import Algorithm, Step
    from taccl_instance import Instance
    from helpers import build_gpu_epoch_view

    # The collective *identity* (which taccl collective to build) and the
    # schedule *format* (which parser to run) are independent: the LP now solves
    # several collectives in one nested format, while the allgather MILP has its
    # own flat format.
    collective_name = detect_collective(schedule)
    lp_format = is_lp_format(schedule)

    if lp_format:
        (num_nodes, factor, steps_in_order, flow_path_keys, switch_rank_map,
         flow_completion_steps, sorted_epochs, flow_completion_epochs,
         flow_rates, root_dense) = parse_flows_lp(schedule, collective_name)
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
         flow_completion_steps, sorted_epochs, flow_completion_epochs) = \
            parse_flows(schedule)
        flow_rates = {}
        collective = _build_collective(collective_name, num_nodes,
                                       schedule.get('0-Root'))

    gpu_epoch_view = build_gpu_epoch_view(
        steps_in_order, sorted_epochs, num_nodes, flow_completion_epochs)

    # `factor` is the chunk_up() multiplier. For the allgather MILP it is
    # num_subchunks (each source's data pre-split into that many real chunks by
    # the solver). For the LP it is S * M: S real sub-chunks per demand times the
    # M-way volume subdivision this converter introduced to make fractional
    # volumes integral.
    if factor > 1:
        collective = collective.chunk_up(factor)

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

    return (algo, flow_path_keys, switch_rank_map, flow_completion_steps,
            gpu_epoch_view, piece_rate)


def enforce_send_epoch_ordering(xml_str, send_epoch_manifest):
    """Post-process MSCCL XML to serialize same-GPU sends by epoch.

    TACCL's ncclize() only generates depid/deps links for data-flow reasons:
    a send depends on the recv that previously wrote the chunk into the GPU's
    buffer. Sends of a GPU's *own* chunk (initialised by <copy>, never tracked
    in the writers table) therefore get depid='-1', so all such sends to
    different peers start simultaneously regardless of epoch.

    In TE-CCL's fat-tree model every GPU shares a single uplink to its leaf
    switch across all epochs. Sending in multiple epochs simultaneously
    overloads that uplink. This function adds the missing deps so that an
    epoch-N send waits for the epoch-(N-1) send on the same GPU to complete.

    Only sends with depid=='-1' are modified. Relay sends already carry a
    data-flow dep from ncclize (on the recv that wrote the chunk being
    forwarded) and are implicitly epoch-ordered through that chain.

    depid/deps semantics (msccl_interpreter.h / taccl_ncclize.py):
      depid  = block_rbid of the dependency op (TB id on the *same* GPU)
      deps   = op.idx of the dependency op (= 's' attribute in the XML)
    hasdep=1 on the target op enables the flag write in the MSCCL runtime.

    send_epoch_manifest is ncclize()'s list of per-send-op records
    {'gpu', 'tb', 's', 'epoch'}, populated at XML emission. The epoch is looked
    up by an op's final (gpu, tb, s) location rather than by its mscclflowid,
    because a flow id is now a bijection with a route and so spans multiple
    epochs (see the flow_id assignment in ncclize()).
    """
    import xml.etree.ElementTree as ET

    # (gpu, tb, s) -> epoch index (= index into steps_in_order)
    epoch_by_op = {(r['gpu'], r['tb'], r['s']): r['epoch']
                   for r in send_epoch_manifest}

    root = ET.fromstring(xml_str)

    for gpu_elem in root.findall('gpu'):
        gpu_id = int(gpu_elem.get('id'))
        # Collect every send op: (epoch, tb_rbid, s_idx, step_elem)
        sends = []
        for tb_elem in gpu_elem.findall('tb'):
            tb_rbid = int(tb_elem.get('id'))
            for step_elem in tb_elem.findall('step'):
                if step_elem.get('type') != 's':
                    continue
                s_idx = int(step_elem.get('s'))
                epoch = epoch_by_op.get((gpu_id, tb_rbid, s_idx))
                if epoch is None:
                    continue
                sends.append((epoch, tb_rbid, s_idx, step_elem))

        if len(sends) <= 1:
            continue

        # Stable sort by (epoch, s_idx) gives a consistent serialisation order.
        sends.sort(key=lambda x: (x[0], x[2]))

        for i in range(1, len(sends)):
            curr_epoch, curr_tb, _, curr_elem = sends[i]
            prev_epoch, prev_tb, prev_s, prev_elem = sends[i - 1]

            if curr_epoch <= prev_epoch:
                continue  # same epoch: concurrent sends on different links are fine
            if curr_tb == prev_tb:
                continue  # same TB: sequential step ordering already handles this
            if curr_elem.get('depid') != '-1':
                continue  # ncclize already added a data-flow dep (relay send)

            curr_elem.set('depid', str(prev_tb))
            curr_elem.set('deps', str(prev_s))
            # Mark the dep target so the runtime writes its completion flag.
            if prev_elem.get('hasdep') == '0':
                prev_elem.set('hasdep', '1')

    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule', help='TE-CCL schedule JSON file')
    p.add_argument('-o', '--output', required=True, help='output XML file')
    p.add_argument('--instances', type=int, default=1)
    p.add_argument('--scale-remote', type=int, default=1)
    p.add_argument('--switch-routing-output', default=None,
                    help='optional path to write per-switch flow_id -> next-hop routing table as JSON')
    p.add_argument('--epoch-debug-output', default=None,
                    help='optional path to write a human-readable per-GPU, '
                         'per-epoch schedule dump. The realizability feasibility '
                         'check (and its warnings) runs regardless of this flag.')
    p.add_argument('--no-rate', action='store_true',
                    help='omit the per-op rate attribute from the emitted XML. '
                         'The schedule is computed identically; only the final '
                         'written output differs, to enable regression testing '
                         'against outputs that predate the rate field.')
    args = p.parse_args()

    from taccl_ncclize import ncclize, ChannelPolicy
    from helpers import (check_epoch_ordering_feasibility,
                         warn_epoch_ordering_violations, write_gpu_epoch_debug)

    with open(args.schedule) as f:
        schedule = json.load(f)

    (algo, flow_path_keys, switch_rank_map, flow_completion_steps,
     gpu_epoch_view, piece_rate) = build_algorithm(schedule)

    # Always run the feasibility check and surface any realizability warnings;
    # writing the human-readable dump is independent and opt-in.
    violations = check_epoch_ordering_feasibility(gpu_epoch_view)
    warn_epoch_ordering_violations(violations)

    # flow_manifest drives switch routing (one entry per route); the per-op
    # send_epoch_manifest drives epoch ordering.
    flow_manifest = []
    send_epoch_manifest = []

    xml = ncclize(
        algo,
        channel_policy=ChannelPolicy.MatchTopology,
        old_format=True,
        use_scratch=True,
        instances=args.instances,
        scale_remote=args.scale_remote,
        flow_path_keys=flow_path_keys,
        flow_manifest=flow_manifest,
        flow_completion_steps=flow_completion_steps,
        piece_rate=piece_rate,
        send_epoch_manifest=send_epoch_manifest,
        logging=True,
    )

    xml = enforce_send_epoch_ordering(xml, send_epoch_manifest)

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
        routes = build_switch_routes(flow_manifest, switch_rank_map)
        with open(args.switch_routing_output, 'w') as f:
            json.dump(routes, f, indent=2)
        print(f'Wrote {args.switch_routing_output}')


if __name__ == '__main__':
    main()
