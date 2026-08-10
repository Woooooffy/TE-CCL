# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from lxml import etree as ET
from collections import defaultdict
from dataclasses import dataclass, field, replace
import math
import threading, queue
from enum import Enum
from z3 import *

@dataclass
class _Gpu:
    copies: list
    inputs: dict
    outputs: dict
    input_chunks: int
    output_chunks: int
    scratch: dict = field(default_factory=dict)
    threadbloks: list = field(default_factory=list)

    def scratch_size(self):
        return max((idx for addr, idx in self.scratch.items()), default=-1) + 1

@dataclass
class _Threadblock:
    channel: int
    rbid: int = None
    send: int = -1
    recv: int = -1
    steps: list = field(default_factory=list)
    # The steps may expand into multiple operations here
    ops: list = field(default_factory=list)

@dataclass
class _Copy:
    input_offset: int
    output_offset: int

@dataclass
class _Op:
    gpu: int
    peer: int
    step: int
    is_send: bool
    op_type: str
    src_buffer: str
    src_offset: int
    dst_buffer: str
    dst_offset: int
    cnt: int
    depends: list
    block_rbid: int = None
    # idx is the NCCL XML step index, which may not be the same as the algorithm step index
    idx: int = None
    has_dependence: bool = False
    mscclflowid: int = None
    # Hashable key identifying which physical path (e.g. switch route) this op's
    # transfer takes; None means "no path information" (every caller except
    # teccl_ncclize.py's flow_path_keys, and any direct/non-switch hop there).
    # Carried through to channel assignment so that _allocate_channels_match_topology
    # can keep different physical paths on the same (src,dst) edge off of the same
    # channel -- see that function's docstring for why this matters.
    path_key: object = None
    # Physical transmission rate (GB/s) for ONE piece of this op, so the emitted
    # rate is cnt * piece_rate. None means "unpaced": either no rate information was
    # supplied at all, or the level of the solve that produced this flow chose not
    # to pace it (e.g. an intra-node NVLink hop that carries ordering but is not
    # pinned to the epoch grid).
    piece_rate: float = None

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

# Poor hack
is_reduce = False

def _analyze_liveness(gpus, algorithm):
    # Initialize liveness intervals for buffers on each GPU
    input_livenesses = {rank: [[(-1,-1)] for _ in range(gpu.input_chunks)] for rank, gpu in gpus.items()}
    output_livenesses = {rank: [[(math.inf,math.inf)] for _ in range(gpu.output_chunks)] for rank, gpu in gpus.items()}
    scratch_livenesses = {rank: [[(math.inf,-1)] for addr, idx in gpu.scratch.items()] for rank, gpu in gpus.items()}

    # For copies reserve the index in the output buffer from the very beginning
    for rank, gpu in gpus.items():
        for copy in gpu.copies:
            output_livenesses[rank][copy.output_offset] = [(-1,math.inf)]

    def update_liveness(rank, addr, step_idx):
        gpu = gpus[rank]
        # Find the relevant buffer and livenesses for the address
        if addr in gpu.inputs:
            buffer = gpu.inputs
            liveness = input_livenesses[rank]
        elif addr in gpu.outputs:
            buffer = gpu.outputs
            liveness = output_livenesses[rank]
        elif addr in gpu.scratch:
            buffer = gpu.scratch
            liveness = scratch_livenesses[rank]
        else:
            raise RuntimeError(f'Address {addr} not found in any buffer of rank {rank}.')

        # Expand the interval to include the step
        idx = buffer[addr]
        start, end = liveness[idx][0]
        liveness[idx][0] = (min(start, step_idx), max(end, step_idx))

    # For each step of the algorithm, update liveness intervals for all buffers
    for step_idx, step in enumerate(algorithm.steps):
        if len(step.sends[0]) == 5:
            for addr, src, dst, _, _ in step.sends:
                update_liveness(src, addr, step_idx)
                update_liveness(dst, addr, step_idx)
        elif len(step.sends[0]) == 6:
            for addr, src, dst, _, _, _ in step.sends:
                update_liveness(src, addr, step_idx)
                update_liveness(dst, addr, step_idx)
        else:
            for addr, src, dst in step.sends:
                update_liveness(src, addr, step_idx)
                update_liveness(dst, addr, step_idx)

    return (input_livenesses, output_livenesses, scratch_livenesses)

def _remap_scratch_into_input_output(liveness, gpus, logging):
    '''
    This function solves and applies a static mapping for scratch buffer indices to input/output buffers that minimizes
    scratch buffer usage for each GPU. The solving is done per GPU using the Z3 SMT solver.
    '''
    input_livenesses, output_livenesses, scratch_livenesses = liveness

    if logging:
        print('Remapping scratch into input/output...')

    def conflict(b1, b2):
        # Check if any of the intervals in lists b1 and b2 overlap
        return any(s1 <= e2 and s2 <= e1 for s1, e1 in b1 for s2, e2 in b2)

    print('Optimizing scratch mapping on all GPUs: ', end='', flush=True)
    # Handle each GPU separately
    for rank, gpu in gpus.items():
        ctx = Context()
        s = Solver(ctx=ctx)

        def remap(idx):
            # Choose for each scratch index a new index in one of the buffers
            # The index space has the input buffer from 0 to input_chunks-1,
            # the output buffer from input_chunks to output_chunks-1,
            # and the scratch buffer for any indices past that.
            return Int(f'{idx}_remap', ctx=ctx)

        # This variable limits the maximum index, in effect the size of the scratch buffer
        idx_end = Int(f'idx_end', ctx=ctx)

        for scratch_idx, scratch_liveness in enumerate(scratch_livenesses[rank]):
            # Block any input indices that conflict with the scratch index
            for input_idx, liveness in enumerate(input_livenesses[rank]):
                if conflict(scratch_liveness, liveness):
                    s.add(remap(scratch_idx) != input_idx)
            # Block any output indices that conflict with the scratch index
            for output_idx, liveness in enumerate(output_livenesses[rank]):
                if conflict(scratch_liveness, liveness):
                    s.add(remap(scratch_idx) != output_idx + gpu.input_chunks)
            # Block remapping conflicting scratch indices to the same input/output indices
            for other_idx, liveness in enumerate(scratch_livenesses[rank]):
                if other_idx != scratch_idx and conflict(liveness, scratch_liveness):
                    s.add(remap(scratch_idx) != remap(other_idx))
            # Require all indices to fit in the allowed buffer space
            s.add(remap(scratch_idx) >= 0)
            s.add(remap(scratch_idx) < idx_end)

        no_memory = gpu.input_chunks + gpu.output_chunks

        q = queue.Queue()
        def optimize(q):
            # Iterate the memory limit down to find a mapping that minimizes scratch usage
            for memory in range(no_memory + gpu.scratch_size(), no_memory - 1, -1):
                if s.check(idx_end == memory) == sat:
                    # Remember the model for the best solution
                    try:
                        m = s.model()
                        new_idxs = {addr: m[remap(old_idx)].as_long() for addr, old_idx in gpu.scratch.items()}
                        q.put(new_idxs)
                    except Z3Exception:
                        # This can happen when the solver is interrupted
                        return
                else:
                    return
        t = threading.Thread(target=optimize, args=(q,))
        t.start()
        t.join(1)
        ctx.interrupt()

        new_idxs = None
        while not q.empty():
            new_idxs = q.get()

        if new_idxs != None:
            print('.', end='', flush=True)
            # Apply the model to remap the scratch indices
            new_scratch = {}
            new_scratch_livenesses = [[] for addr, idx in gpu.scratch.items()]
            for addr, old_idx in gpu.scratch.items():
                new_idx = new_idxs[addr]
                # Figure out which buffer the index is in
                if new_idx < gpu.input_chunks:
                    tgt_buffer = gpu.inputs
                    tgt_idx = new_idx
                    tgt_liveness = input_livenesses[rank][tgt_idx]
                elif new_idx < gpu.input_chunks + gpu.output_chunks:
                    tgt_buffer = gpu.outputs
                    tgt_idx = new_idx - gpu.input_chunks
                    tgt_liveness = output_livenesses[rank][tgt_idx]
                else:
                    tgt_buffer = new_scratch
                    tgt_idx = new_idx - gpu.input_chunks - gpu.output_chunks
                    tgt_liveness = new_scratch_livenesses[tgt_idx]

                # Check that the remapping doesn't conflict with any existing mappings
                liveness = scratch_livenesses[rank][old_idx]
                assert not conflict(tgt_liveness, liveness)
                tgt_liveness.extend(liveness)

                # Remap the scratch index to the new index in the target buffer
                tgt_buffer[addr] = tgt_idx
            gpu.scratch = new_scratch
        else:
            print('x', end='', flush=True)
    else:
        print()

    if logging:
        max_scratch_overhead = max(gpu.scratch_size() / (gpu.input_chunks + gpu.output_chunks) for gpu in gpus.values())
        print(f'Maximum scratch overhead is {max_scratch_overhead * 100:.0f}%')

def _allocate_channels_max_concurrency(op_sets, logging):
    # This function solves a coloring problem to ops to a minimal set of channels
    ctx = Context()

    def chan(idx):
        return Int(f'chan_{idx}', ctx=ctx)
    max_channels = Int('max_channels', ctx=ctx)

    constraints = []

    # Add basic constraints and find conflicting sets of operations
    conflict_groups = defaultdict(set)
    for idx, op_set in enumerate(op_sets):
        for op in op_set:
            # Two operations conflict if they use the same src-dst edge on the same step
            conflict_groups[(op.gpu, op.is_send, op.peer, op.step)].add(idx)
        constraints.append(chan(idx) >= 0)
        constraints.append(chan(idx) < max_channels)

    # Require channels within the conflict groups to be disjoint
    for grp in conflict_groups.values():
        constraints.append(Distinct([chan(idx) for idx in grp]))

    opt = Optimize(ctx=ctx)
    opt.add(constraints)
    opt.minimize(max_channels)

    t = threading.Thread(target=opt.check)
    t.start()
    t.join(1)
    main_ctx().interrupt()
    t.join()

    try:
        model = opt.model()
    except Z3Exception:
        # TODO: This altenate process does not guarantee that channels are contiguous
        s = Solver(ctx=ctx)
        s.add(constraints)
        s.check()
        model = s.model()

    if logging:
        print(f'Using up to {model[max_channels].as_long()} channels')

    # Group the operations by which channels they use
    ops_by_channel = defaultdict(list)
    for idx, op_set in enumerate(op_sets):
        ops = ops_by_channel[model[chan(idx)].as_long()]
        ops.extend(op_set)

    return ops_by_channel

def _is_relay_link(topology, src, dst):
    if "copies" in topology.name:
        num_copies = topology.name.split(",")[1].strip(")")
        copies = int(num_copies[7:])
    else:
        copies = 1
    num_local = len(topology.links) // copies
    if src // num_local != dst // num_local:
        return True
    return False

def _allocate_channels_match_topology(op_sets, topology, instances, scale_remote, logging, max_channels=32):
    '''
    Assign each op_set (a [send_op, recv_op] pair) a channel, round-robining within each
    (src, dst) edge's replica count (topology.link(src,dst), scaled by instances/scale_remote)
    exactly as before -- *except* now partitioned first by op.path_key, so that two flows
    sharing an edge but taking different physical paths (e.g. different switch routes in a
    fat tree, distinguished via mscclflowid upstream) can never land on the same channel.

    Why this matters (see ncclize_switch_path_ordering_gap project memory and
    msccl_step_ordering_correctness sibling-project memory, both dated 2026-07-06): an
    MSCCL connection is scoped exactly to (channelId, peer) -- each channel+peer pair is
    its own independent step-counter/FIFO (mscclSetupConnections in msccl_setup.cc), so
    different channels between the same GPU pair have *zero* ordering dependency on each
    other. Multipath schedules can have a later-issued chunk physically arrive before an
    earlier one (different switch paths, different congestion/hop-count), and this
    topology/ncclize layer has no reliable way to know the true relative completion order
    of two flows sharing one channel (TeCCLTopology never models switches at all -- see
    its own docstring). Rather than try to reorder ops on a shared channel, this function
    avoids the problem structurally: two flows that took different paths are *never* placed
    on the same channel, so there is no intra-channel relative order to get right in the
    first place. This is sufficient (not just a heuristic) because flows sharing one
    path_key inherently can't be reordered relative to each other -- same physical route
    means departure order == completion order -- so keeping same-path flows on shared
    channels (round-robinned across replicas exactly as before) remains correct.

    This handles WHICH CHANNEL a flow lands on. The complementary question -- that the two
    ends of one (peer, channel) agree on the ORDER of the operations they exchange, since
    the connection is a FIFO -- is handled by giving each (gpu, direction, peer, channel)
    its own threadblock and keeping send and recv at the same .step; see the threadblock
    construction in ncclize() below.

    max_channels caps the channel ids used per edge (num_distinct_path_keys * link); this
    is a real hardware/runtime limit (MAXCHANNELS in NCCL/MSCCL, 32 as of the vendored
    msccl-executor-nccl fork) on how many channels a single GPU's threadblocks may
    reference in total, so it's enforced here rather than left to fail confusingly
    downstream.
    '''
    if len(topology.switches) > 0 and logging:
        print('Warning: Switches in the topology are ignored for the channel policy MatchTopology.')
    # print(topology)
    ops_by_channel = defaultdict(list)
    # (src, dst) -> {path_key: first-seen index}; index 0 is always assigned to whichever
    # path_key is seen first for that edge, so an edge with only path_key=None (every
    # non-teccl caller, and any teccl edge with no switch-path diversity) always gets
    # index 0 for it -- channel = 0 * link + rr == rr, identical to pre-existing behavior.
    path_index_by_edge = defaultdict(dict)
    next_channel = defaultdict(lambda: 0)
    for op_set in op_sets:
        send = op_set[0]
        assert send.op_type == 's'
        src = send.gpu
        dst = send.peer
        path_key = send.path_key

        link = topology.link(src,dst) * instances
        global is_reduce
        if is_reduce: # and ("DGX1" in topology.name or "DGX2RFix" in topology.name):
            if link == 0:
                print(f"link {src}->{dst} was 0. Making it {topology.link(dst,src)}")
                topology.links[dst][src] = topology.links[src][dst]
                link = topology.link(src,dst)
                # topology.link(src,dst) = topology.link(dst,src)
                assert link > 0
        else:
            #if link <= 0:
            #    print("INVALID SEND:", src, "->", dst)
            #    print("Topology links:", topology.links)
            assert link > 0, 'Encountered send on non-existent link'
        if _is_relay_link(topology, src, dst):
            link = link * scale_remote

        path_indices = path_index_by_edge[(src, dst)]
        if path_key not in path_indices:
            path_indices[path_key] = len(path_indices)
        path_idx = path_indices[path_key]

        edge_channels_needed = len(path_indices) * link
        if edge_channels_needed > max_channels:
            raise ValueError(
                f'Edge {src}->{dst} needs {len(path_indices)} distinct path(s) x {link} '
                f'replica(s) = {edge_channels_needed} channels, exceeding max_channels='
                f'{max_channels}. Reduce the number of distinct switch paths routed over '
                f'this edge, or raise max_channels if the target NCCL/MSCCL build supports '
                f'more channels.')

        chan = path_idx * link + next_channel[(src, dst, path_key)]
        ops_by_channel[chan].extend(op_set)
        next_channel[(src, dst, path_key)] = (next_channel[(src, dst, path_key)] + 1) % link

    return ops_by_channel

class ChannelPolicy(Enum):
    One = 'One'
    MaxConcurrency = 'MaxConcurrency'
    MatchTopology = 'MatchTopology'

    def __str__(self):
        return self.value


def _realize_pacing_gates(gpus, pacing_gates):
    """Wire the caller's send-pacing gate manifest into ``op.depends``.

    ``pacing_gates`` is a list of ``(consumer_key, producer_key)`` edges, each key a
    ``(step, src, dst, path_key)`` send-op identity, computed by teccl_ncclize from per-send
    link-occupancy windows (finish-before-start; see ``_finish_before_start_gates`` there). The
    pacing POLICY -- which send waits on which, from rate and topology -- lives with the level that
    produced the schedule; this function only REALIZES it, because only here do ops have the
    threadblock ids and post-nop-expansion step indices a ``depid``/``deps`` edge needs.

    A key can map to several ops when ``make_intervals`` splits one epoch's transfer into
    non-contiguous offset runs; those share a threadblock and run in step order, so gating a
    consumer on the producer key's LAST op (all its bytes done) suffices, and every consumer op of
    the key inherits the gate. The gate is appended as an extra dependency; the nop-expansion pass
    below realizes any dependency past the first and drops same-threadblock ones (threadblock step
    order already serializes those). Every edge points to a strictly-earlier-finishing send, so the
    added edges cannot form a cycle. Requires ``op.block_rbid`` to be assigned already.
    """
    key_to_ops = defaultdict(list)
    for gpu in gpus.values():
        for tb in gpu.threadblocks:
            for op in tb.steps:
                if op.is_send:
                    key_to_ops[(op.step, op.gpu, op.peer, op.path_key)].append(op)
    for consumer_key, producer_key in pacing_gates:
        producers = key_to_ops.get(producer_key)
        consumers = key_to_ops.get(consumer_key)
        if not producers or not consumers:
            continue
        producer = producers[-1]  # last in threadblock order: its completion covers the key
        for op in consumers:
            op.depends = list(op.depends) + [producer]


def ncclize(algorithm, remap_scratch = None, channel_policy=ChannelPolicy.MatchTopology, pretty_print = True, old_format=False, use_scratch=False, merge_contiguous=True, instances=1, scale_remote=1, combine_contig=False, aid_IB_contig=False, prefix="", logging=False, flow_path_keys=None, flow_manifest=None, piece_rate=None, send_epoch_manifest=None, pacing_gates=None, max_channels=32):
    '''
    Generate the XML format used by the NCCL SCCL backend.

    Sends are split into send/recv operations and grouped by the rank executing them. Within each rank operations are
    grouped under <threadblock/> tags, which handle 1) a single peer, 2) a single type of operation, and 3) at most one
    operation per each step of the algorithm. Additional threadblocks are created as necessary to meet these
    constraints.

    Each send operation is mapped from the abstract addresses used by the synthesized algorithm to offsets into three
    named buffers, "input", "output" and "scratch", based on whether the address appears in a particular rank's
    precondition, postcondition or neither. For addresses that would be in both the input and output buffers <copy/>
    tags are created to mark an initial transfer to the output buffer and only the output buffer mapping is kept.

    flow_path_keys, if given, maps (step_idx, addr, src, dst) -> a hashable path key. Addresses sharing an edge within
    a step are only merged into one send op if they share the same path key, so e.g. chunks that traveled via
    different switch paths are kept as separate ops (and thus get distinct mscclflowids). Under channel_policy=
    MatchTopology, these path keys also keep same-edge, different-path flows off of the same channel entirely --
    see _allocate_channels_match_topology()'s docstring for why that's the correct fix for multipath ordering
    (as opposed to reordering ops on a shared channel). None preserves prior behavior (no disambiguation).


    max_channels caps how many channel ids a single edge's path_key partitioning may use under channel_policy=
    MatchTopology (default 32, matching NCCL/MSCCL's MAXCHANNELS hardware limit); see
    _allocate_channels_match_topology().

    piece_rate is the physical transmission rate (GB/s) of ONE piece; an op of cnt pieces emits cnt * piece_rate.
    It may be a scalar (pace every op identically -- the flat, single-level case) or a dict keyed like
    flow_path_keys, (step_idx, addr, src, dst) -> rate, supplying a per-flow rate. The per-flow form exists because
    a hierarchical schedule interleaves flows solved at different levels, each with its own epoch duration and
    capacity model: the level that produced a flow computes its rate, and a flow it chose not to pace is simply
    absent from the map and emits no rate attribute. None disables the attribute entirely.

    piece_rate also GATES the mscclflowid attribute and the flow_manifest, since both describe a network send
    and a rate is the marker for one (see the flow-id assignment below). A caller that supplies no piece_rate at
    all therefore gets no flow ids either -- which is the right answer for the callers that pass neither
    flow_path_keys nor flow_manifest, as nothing downstream of them reads the id.
    '''

    if algorithm.is_pipelined():
        raise ValueError('Pipelining is not supported.')

    if remap_scratch is None:
        if algorithm.instance.extra_memory != None:
            remap_scratch = True
            if logging:
                print('Turning scratch remapping on to honor the memory limit set in the instance.')
        else:
            remap_scratch = False

    # Create GPUs, their address to buffer mappings and possible copies
    gpus = {}
    for rank in algorithm.ranks():
        outputs = {}
        if rank in algorithm.output_map:
            outputs.update({ addr: idx for idx, addr in enumerate(sorted(algorithm.output_map[rank])) })
        inputs = {}
        copies = []
        if rank in algorithm.input_map:
            for idx, addr in enumerate(sorted(algorithm.input_map[rank])):
                if addr in outputs:
                    copies.append(_Copy(idx, outputs[addr]))
                else:
                    inputs[addr] = idx
        gpus[rank] = _Gpu(copies, inputs, outputs, len(inputs) + len(copies), len(outputs))

    # Create scratch buffer mappings if necessary
    def allocate_scratch(gpu, addr):
        if not (addr in gpu.inputs or addr in gpu.outputs or addr in gpu.scratch):
            offset = len(gpu.scratch)
            gpu.scratch[addr] = offset

    if aid_IB_contig:
        # first add scratch for relay sends only
        for step in algorithm.steps:
        # for s, cnt1 in sorted(list(enumerate(longest_relay)), key=lambda x:x[1], reverse=True):
            # step = algorithm.steps[s]
            if len(step.sends[0]) == 5:
                for addr, src, dst, _, _ in step.sends:
                    if _is_relay_link(algorithm.topology,src,dst):
                        allocate_scratch(gpus[src], addr)
                        allocate_scratch(gpus[dst], addr)
            elif len(step.sends[0]) == 6:
                for addr, src, dst, _, _, _ in step.sends:
                    if _is_relay_link(algorithm.topology,src,dst):
                        allocate_scratch(gpus[src], addr)
                        allocate_scratch(gpus[dst], addr)
            else:
                for addr, src, dst in step.sends:
                    if _is_relay_link(algorithm.topology,src,dst):
                        allocate_scratch(gpus[src], addr)
                        allocate_scratch(gpus[dst], addr)

    # next add for remaining steps
    for step in algorithm.steps:
        if len(step.sends[0]) == 5:
            for addr, src, dst, _, _ in step.sends:
                allocate_scratch(gpus[src], addr)
                allocate_scratch(gpus[dst], addr)
        elif len(step.sends[0]) == 6:
            for addr, src, dst, _, _, _ in step.sends:
                allocate_scratch(gpus[src], addr)
                allocate_scratch(gpus[dst], addr)
        else:
            for addr, src, dst in step.sends:
                allocate_scratch(gpus[src], addr)
                allocate_scratch(gpus[dst], addr)

    # Analyze liveness of indices in buffers and remap scratch into input/output as possible
    if remap_scratch:
        liveness = _analyze_liveness(gpus, algorithm)
        _remap_scratch_into_input_output(liveness, gpus, logging)


    def get_buffer_and_offset(gpu, addr):
        # Map an address to one of the named buffers
        if addr in gpu.inputs:
            return 'i', gpu.inputs[addr]
        elif addr in gpu.outputs:
            return 'o', gpu.outputs[addr]
        elif addr in gpu.scratch:
            return 's', gpu.scratch[addr]
        else:
            raise RuntimeError('Address is not mapped to a buffer')

    def make_intervals(src, dst, addrs_set):
        if len(addrs_set) == 0:
            return

        buffs_and_offs = []
        for addr in addrs_set:
            srcbuff, srcoff = get_buffer_and_offset(gpus[src], addr)
            dstbuff, dstoff = get_buffer_and_offset(gpus[dst], addr)
            buffs_and_offs.append((srcbuff, srcoff, dstbuff, dstoff))

        if merge_contiguous:
            # Sort sends by both buffers and offsets and merge sends into larger intervals when both the source and
            # destination are contiguous.
            buffs_and_offs.sort()
            start = prev = buffs_and_offs[0]

            def make_interval(a,b):
                cnt = b[1] - a[1] + 1
                assert cnt == b[3] - a[3] + 1, 'Source and destination count mismatch'
                return (a[0], a[1], a[2], a[3], cnt)

            for x in buffs_and_offs[1:]:
                if x[0] == prev[0] and x[1] == prev[1] + 1 and x[2] == prev[2] and x[3] == prev[3] + 1:
                    # Merge into previous interval if buffers match and the new offsets are at the end of the interval
                    prev = x
                else:
                    # Yield the previous interval and start a new one
                    yield make_interval(start, prev)
                    start = prev = x
            # Yield the last interval
            yield make_interval(start, prev)
        else:
            # Just yield size 1 intervals if merging is disabled
            for srcbuff, srcoff, dstbuff, dstoff in buffs_and_offs:
                yield (srcbuff, srcoff, dstbuff, dstoff, 1)

    # Turn all steps of the algorithm into operations
    op_sets = []
    # (src, dst, path_key) route -> its flow id. A flow id is a bijection with a
    # physical route, so every send/recv on that route (across all epochs and
    # chunks) shares one id (see the flow_id assignment below). Only PACED routes
    # are entered here; see the assignment for why.
    route_flow_ids = {}
    # (src, dst, path_key) route -> whether its ops carry a rate. A route must be
    # uniformly paced or uniformly unpaced for the flow-id restriction below to be
    # well defined; this records the first answer so the rest can be checked.
    route_paced = {}
    # Track the latest op that wrote to each buffer index
    writers = defaultdict(list)
    # Track all the reads since the last write to each buffer index
    readers = defaultdict(list)
    for step_idx, step in enumerate(algorithm.steps):
        new_writers = defaultdict(list)
        new_readers = defaultdict(list)

        # Group sent addresses by edge
        grouped_sends = defaultdict(set)
        # (src, dst, path_key) -> per-piece rate, when piece_rate is a per-flow map.
        group_rate = {}
        if len(step.sends[0]) == 5:
            for addr, src, dst, t, l in step.sends:
                if combine_contig:
                    grouped_sends[(src,dst)].add(addr)
                else:
                    grouped_sends[(src,dst,t,l)].add(addr)
        elif len(step.sends[0]) == 6:
            for addr, src, dst, t, l, redop in step.sends:
                if combine_contig:
                    grouped_sends[(src,dst)].add(addr)
                else:
                    grouped_sends[(src,dst,t,l,redop)].add(addr)
        else:
            for addr, src, dst in step.sends:
                path_key = flow_path_keys.get((step_idx, addr, src, dst)) if flow_path_keys else None
                grouped_sends[(src, dst, path_key)].add(addr)
            if isinstance(piece_rate, dict):
                # A merged op emits ONE rate for all cnt of its pieces, so every piece
                # merged into it must have been paced identically. Assert that rather
                # than picking one: a level that paces its flows non-uniformly would
                # otherwise have its schedule silently misrepresented here.
                for (src, dst, path_key), addrs in grouped_sends.items():
                    vals = {piece_rate.get((step_idx, addr, src, dst)) for addr in addrs}
                    assert len(vals) == 1, (
                        f"pieces merged into one op ({src}->{dst} at step {step_idx}, "
                        f"path {path_key}) carry different rates {sorted(vals, key=str)}; "
                        f"a merged op can only emit a single rate")
                    group_rate[(src, dst, path_key)] = vals.pop()

        # Combine sends into intervals and create multiple instances if necessary
        sends = []
        if combine_contig or len(step.sends[0])<5:
            for key, addrs in grouped_sends.items():
                src, dst = key[0], key[1]
                path_key = key[2] if len(key) > 2 else None
                for src_buf, src_off, dst_buf, dst_off, cnt in make_intervals(src, dst, addrs):
                    for i in range(instances):
                        new_src_off = src_off * instances + i * cnt
                        new_dst_off = dst_off * instances + i * cnt
                        send = (src, dst, src_buf, new_src_off, dst_buf, new_dst_off, cnt, path_key)
                        sends.append(send)
        elif len(step.sends[0])==6:
            for (src,dst,t,l,redop) in sorted(grouped_sends, key=lambda x: x[2]):
                addrs = grouped_sends[(src,dst,t,l,redop)]
                for src_buf, src_off, dst_buf, dst_off, cnt in make_intervals(src, dst, addrs):
                    for i in range(instances):
                        new_src_off = src_off * instances + i * cnt
                        new_dst_off = dst_off * instances + i * cnt
                        send = (src, dst, src_buf, new_src_off, dst_buf, new_dst_off, cnt, redop, None)
                        sends.append(send)
        else:
            for (src, dst,t,l) in sorted(grouped_sends, key=lambda x: x[2]):
                addrs = grouped_sends[(src,dst,t,l)]
                for src_buf, src_off, dst_buf, dst_off, cnt in make_intervals(src, dst, addrs):
                    for i in range(instances):
                        new_src_off = src_off * instances + i * cnt
                        new_dst_off = dst_off * instances + i * cnt
                        send = (src, dst, src_buf, new_src_off, dst_buf, new_dst_off, cnt, None)
                        sends.append(send)
        # Perform dependency tracking and create _Op instances
        global is_reduce
        for send in sends:
            redop = None
            if len(send) == 8:
                src, dst, src_buf, src_off, dst_buf, dst_off, cnt, path_key = send
            else:
                src, dst, src_buf, src_off, dst_buf, dst_off, cnt, redop, path_key = send
            read_keys = [(src,src_buf,src_off+i) for i in range(cnt)]
            # A send must wait for the previous recv (if any) to finish
            send_depends = list(set(d for k in read_keys for d in writers[k]))

            write_keys = [(dst,dst_buf,dst_off+i) for i in range(cnt)]
            # A receive must wait for both the previous recv and any previous sends to finish
            recv_depends = list(set(d for deps in (readers, writers) for k in write_keys for d in deps[k]))

            send_op = _Op(src, dst, step_idx, True, 's', src_buf, src_off, dst_buf, dst_off, cnt, send_depends)
            if redop is None:
                recv_op = _Op(dst, src, step_idx, False, 'r', src_buf, src_off, dst_buf, dst_off, cnt, recv_depends)
            else:
                assert redop == 'rrc'
                is_reduce = True
                recv_op = _Op(dst, src, step_idx, False, redop, src_buf, src_off, dst_buf, dst_off, cnt, recv_depends)

            # Carry the path key through to channel assignment (see
            # _allocate_channels_match_topology()'s docstring): None for every caller
            # except teccl_ncclize.py's flow_path_keys on a switch-relayed hop.
            send_op.path_key = path_key
            recv_op.path_key = path_key

            # Per-piece transmission rate. A scalar piece_rate paces every op the same
            # (the flat single-level case); a dict is a per-flow rate supplied by the
            # level of the solve that produced each flow, and a flow absent from it is
            # deliberately unpaced.
            rate = (group_rate.get((src, dst, path_key)) if isinstance(piece_rate, dict)
                    else piece_rate)
            send_op.piece_rate = rate
            recv_op.piece_rate = rate

            # send_op.step and recv_op.step are DELIBERATELY EQUAL (both step_idx). Threadblocks
            # are one per (gpu, direction, peer, channel) and their ops are sorted by .step, so
            # equal steps make the two ends of a connection order their operations identically:
            # ops_by_channel receives [send_op, recv_op] together per op_set, so the sender's and
            # the receiver's threadblocks see the same op_sets in the same sequence, and a stable
            # sort on equal keys preserves it -- ties included. That is the invariant the runtime
            # requires of a (peer, channel) FIFO.
            #
            # The recv's .step used to be advanced to its true completion epoch, to stop a
            # switch-relayed recv from stalling an unrelated send packed into the same
            # threadblock. Threadblocks are no longer packed (see below), so there is no unrelated
            # send to stall, and advancing it would only reintroduce the ordering mismatch.

            # The flow id is a bijection with the physical route (src -> switches
            # -> dst), keyed by (src, dst, path_key): every send/recv on that
            # route, in any epoch and for any chunk, shares one id. This gives the
            # switch forwarding table exactly one entry per route. Per-op epoch
            # ordering is recovered from send_epoch_manifest (populated at XML
            # emission), not from the flow id, precisely because one flow id now
            # spans multiple epochs.
            #
            # Flow ids are emitted ONLY for PACED (rate-bearing) ops, for the same
            # reason the rate is: both are properties of a NETWORK send. A flow id
            # exists to let a programmable inter-node switch forward by route, and
            # a rate exists to hold that send to the epoch grid the level solved
            # on. An intra-cell hop has neither -- it crosses an NVSwitch that
            # forwards on its own and it was scheduled for ORDER, not for pacing
            # (see reconstruct._piece_rate and flat_schedule._segment) -- so
            # tagging it would put a phantom entry in the forwarding table for a
            # switch that is never programmed from it.
            #
            # Both invariants survive the restriction:
            #  * flow id <-> route stays a bijection, now over the paced routes
            #    only: route_flow_ids is still keyed by route and still hands out
            #    one dense id per distinct route.
            #  * distinct paths still land on distinct channels: channel
            #    allocation partitions each edge by op.path_key, never by flow id
            #    (see _allocate_channels_match_topology), and path_key is set on
            #    every op regardless of pacing.
            # What the restriction does require is that a route not be paced in
            # one epoch and unpaced in another -- that would tag only part of the
            # route's traffic and leave the switch with packets it has an entry
            # for but cannot match. Nothing structurally forbids it, so assert it.
            route = (src, dst, path_key)
            paced = rate is not None
            prev_paced = route_paced.setdefault(route, paced)
            assert prev_paced == paced, (
                f"route {src}->{dst} (path {path_key}) is paced in some epochs and "
                f"unpaced in others; a route must be uniformly paced, since its "
                f"flow id is emitted for the whole route or not at all")

            if paced:
                flow_id = route_flow_ids.setdefault(route, len(route_flow_ids))
                send_op.mscclflowid = flow_id
                recv_op.mscclflowid = flow_id

                if flow_manifest is not None:
                    flow_manifest.append({
                        'flow_id': flow_id,
                        'step': step_idx,
                        'src': src,
                        'dst': dst,
                        'path_key': path_key,
                    })

            # Record the send and receive as a set of operations that must happen on the same channel
            # if src_off == 0 or src_off == 1:
            op_sets.append([send_op, recv_op])

            # Mark writers and readers to be added for the next step
            for k in write_keys:
                new_writers[k].append(recv_op)
            for k in read_keys:
                new_readers[k].append(send_op)
        # Writes cut the dependency to both previous writes and reads
        for key, deps in new_writers.items():
            if key in new_readers:
                gpu, buf, off = key
                if "phasewise" in prefix:
                    print("key", key)
                    print("readers", new_readers[key])
                    print("writers", new_writers[key])
                    dep_send_op = new_readers[key][0]
                    old_recv_op = new_writers[key][0]
                    assert old_recv_op.op_type == 'rrc'
                    deplist = old_recv_op.depends
                    deplist.append(dep_send_op)
                    new_recv_op = _Op(old_recv_op.gpu, old_recv_op.peer, old_recv_op.step, False, old_recv_op.op_type, old_recv_op.src_buffer, old_recv_op.src_offset, old_recv_op.dst_buffer, old_recv_op.dst_offset, old_recv_op.cnt, deplist)
                    new_recv_op.mscclflowid = old_recv_op.mscclflowid
                    for i, op_set in enumerate(op_sets):
                        if op_set[1] == old_recv_op:
                            op_sets[i][1] = new_recv_op
                    new_writers[key][0] = new_recv_op
                    print(f'Encountered receive and send on the same buffer index on step {step_idx + 1} (gpu={gpu}, buf={buf}, off={off})')
                    # print('but added deps')
                else:
                    raise RuntimeError(f'Encountered receive and send on the same buffer index on step {step_idx + 1} (gpu={gpu}, buf={buf}, off={off})\nAre you running a phasewise algo? Add prefix="_phasewise"')
            writers[key] = deps
            readers[key] = []
        # Reads get added to any previous reads
        for key, deps in new_readers.items():
            readers[key].extend(deps)

    # Fixup everything to match the instanced sends when multiple instances are generated
    if instances > 1:
        for gpu in gpus.values():
            # Create instances copies of the copies.
            new_copies = []
            for copy in gpu.copies:
                for i in range(instances):
                    new_copy = _Copy(copy.input_offset * instances + i, copy.output_offset * instances + i)
                    new_copies.append(new_copy)
            gpu.copies = new_copies

            # Multiply the other metadata with instances
            def expand_mappings(mappings):
                return { addr * instances + i: idx * instances + i for addr, idx in mappings.items() for i in range(instances) }
            gpu.inputs = expand_mappings(gpu.inputs)
            gpu.outputs = expand_mappings(gpu.outputs)
            gpu.input_chunks *= instances
            gpu.output_chunks *= instances
            gpu.scratch = expand_mappings(gpu.scratch)

    # Allocate channels and group operations by channel
    if channel_policy == ChannelPolicy.One:
        ops_by_channel = {0: [op for op_set in op_sets for op in op_set]}
    elif channel_policy == ChannelPolicy.MaxConcurrency:
        ops_by_channel = _allocate_channels_max_concurrency(op_sets, logging)
    elif channel_policy == ChannelPolicy.MatchTopology:
        ops_by_channel = _allocate_channels_match_topology(op_sets, algorithm.topology, instances, scale_remote, logging, max_channels)
    else:
        assert False, 'Unhandled channel policy'

    if flow_path_keys:
        # Sanity check: ops that were kept separate because they take distinct
        # physical paths (distinct path_keys, e.g. different switch routes) are
        # only guaranteed to land in separate threadblocks if they also land on
        # distinct channels here -- threadblocks are grouped by (gpu, is_send,
        # peer, channel) below, so two same-step ops sharing a channel get merged
        # into one threadblock regardless of taking different paths. This can
        # happen if channel_policy doesn't separate same-step/same-edge ops (e.g.
        # ChannelPolicy.One).
        #
        # Grouped by path_key rather than by mscclflowid: the flow id is now
        # emitted only for paced ops (see its assignment above), but an UNPACED op
        # is separated by its path_key just the same and needs the same guarantee,
        # so keying on the flow id would silently stop checking intra-cell hops.
        # path_key is what the allocator actually partitions on, so this also
        # states the guard in the terms of the thing it is guarding.
        #
        # Under channel_policy=MatchTopology this should now be structurally
        # unreachable: _allocate_channels_match_topology() partitions every edge by
        # op.path_key first, so ops with different path_keys always land in disjoint
        # channel sub-ranges. If this warning ever fires under MatchTopology, that's a
        # bug in that partitioning (e.g. a path_key not making it onto the op), not an
        # inherent limitation -- kept here as a cheap regression guard, and because
        # it's still a real limitation for other channel policies (e.g. One).
        path_chans = defaultdict(set)
        for chan, chan_ops in ops_by_channel.items():
            for op in chan_ops:
                path_chans[(op.gpu, op.is_send, op.peer, op.step)].add((op.path_key, chan))
        for (gpu, is_send, peer, step), path_chan_pairs in path_chans.items():
            path_keys = {p for p, _ in path_chan_pairs}
            chans = {c for _, c in path_chan_pairs}
            if len(chans) < len(path_keys):
                print(f'Warning: switch paths {sorted(path_keys, key=str)} on gpu={gpu} '
                      f'is_send={is_send} peer={peer} step={step} only got {len(chans)} distinct '
                      f'channel(s) under channel_policy={channel_policy} -- they will be forced '
                      f'into the same threadblock, losing the intended parallelism between '
                      f'switch paths.')

    # Group by which operations need to be in the same threadblock, then give each group its OWN
    # threadblock: one per (gpu, direction, peer, channel).
    #
    # That quadruple is exactly an MSCCL connection -- a connection is scoped to (channelId, peer)
    # and each one is an independent FIFO with its own step counter (mscclSetupConnections in
    # msccl_setup.cc) -- so a threadblock now serializes precisely the set of operations that the
    # runtime already serializes, and nothing else.
    #
    # Threadblocks used to be PACKED: several groups shared one threadblock whenever their peers
    # and steps did not collide, minimizing threadblock count. That packing is what made
    # flow_completion_steps necessary, because it put unrelated operations -- a send to one peer
    # and a switch-relayed recv from another -- into one step-ordered list, where the recv could
    # stall the send behind it. Correcting the recv's .step to its true completion epoch fixed that
    # stall, but at the cost of ordering recvs by ARRIVAL while sends stayed ordered by DEPARTURE,
    # which breaks the invariant the runtime actually requires: for one (peer, channel) the two
    # ends must agree on order, or the FIFO pairs a send with the wrong recv. That is silent data
    # corruption, not a stall. Measured before this change: 3 of the 32 example schedules emitted
    # a mismatched pairing, one of them on every single (peer, channel) group it had.
    #
    # Splitting removes the cause instead of compensating for it: with no mixed threadblock there
    # is no unrelated operation to stall, so no completion correction is needed, and send and recv
    # keep the same .step and therefore the same order (see the sort below). The cost is more
    # threadblocks -- measured +64% on the hetero allgather, 107 -> 176, max 20 per GPU -- with an
    # unchanged step count, since the operations themselves are the same ones.
    tb_groups = defaultdict(list)
    for chan, chan_ops in ops_by_channel.items():
        for op in chan_ops:
            tb_groups[(op.gpu, op.is_send, op.peer, chan)].append(op)

    tbs_by_gpu_chan = defaultdict(lambda: defaultdict(list))
    for (rank, is_send, peer, chan), grp in tb_groups.items():
        tb = _Threadblock(chan)
        if is_send:
            tb.send = peer
        else:
            tb.recv = peer
        tb.steps.extend(grp)
        tbs_by_gpu_chan[rank][chan].append(tb)

    # Sort threadblocks in each GPU by peers and then the channel
    # This is important as in NCCL threadblocks using the same NVLink concurrently should be close together
    for rank, gpu in gpus.items():
        gpu.threadblocks = sorted([tb for tbs in tbs_by_gpu_chan[rank].values() for tb in tbs],
            key=lambda tb: (tb.send, tb.recv, tb.channel))
        for i, tb in enumerate(gpu.threadblocks):
            tb.rbid = i

    # Do some additional postprocessing of operations:
    # - Expand operations with extra dependencies with no-ops
    # - Mark the index of each operation taking any extra no-ops into account
    # - Record the threadblock rbids for each operation
    all_ops = []
    # assign rbid
    for rank, gpu in gpus.items():
        for tb in gpu.threadblocks:
            tb.steps.sort(key=lambda op: op.step)
            for op in tb.steps:
                op.block_rbid = tb.rbid

    # Realize the caller's send-pacing gates (finish-before-start edges) into op.depends BEFORE nop
    # expansion, so each added gate becomes an ordinary extra dependency.
    if pacing_gates:
        _realize_pacing_gates(gpus, pacing_gates)

    for rank, gpu in gpus.items():
        for tb in gpu.threadblocks:
            tb.steps.sort(key=lambda op: op.step)
            for op in tb.steps:
                # Expand extra dependencies into nop operations
                # Filter out dependencies within the same threadblock first, because that way nop doesn't get used when not needed
                op.depends = list(filter(lambda d: d.block_rbid != op.block_rbid, op.depends))
                if len(op.depends) > 1:
                    extra_deps = op.depends[1:]
                    op.depends = op.depends[:1]
                    first_step = op.step
                    for i, dep in enumerate(extra_deps):
                        tb.ops.append(_Op(op.gpu, None, op.step, False, 'nop', None, None, None, None, 0, [dep]))
                        tb.ops[-1].idx = len(tb.ops) - 1
                tb.ops.append(op)
                tb.ops[-1].idx = len(tb.ops) - 1
            for op in tb.ops:
                op.block_rbid = tb.rbid
            all_ops.extend(tb.ops)

    for op in all_ops:
        if len(op.depends):
            if op.depends[0].block_rbid is None:
                print("this op", len(op.depends), op)
                print("None depends:",op.depends[0])


    # Mark all ops that have a dependence on them
    for op in all_ops:
        for dep in op.depends:
            dep.has_dependence = True

    # Generate the XML structure
    algo_elem = ET.Element('algo')
    algo_elem.set('name', "taccl")
    algo_elem.set('nchannels', str(1 + max(max(tb.channel for tb in gpu.threadblocks) for gpu in gpus.values())))
    if old_format:
        algo_elem.set('nchunksperloop', str(max(max(gpu.input_chunks, gpu.output_chunks) for gpu in gpus.values())))
        algo_elem.set('proto', "Simple")
        algo_elem.set('maxBytes', str(2**63 - 1))
        algo_elem.set('minBytes', "0")
        if "Allgather" in algorithm.name:
            algo_elem.set('coll', "allgather")
            algo_elem.set('inplace', "1")
        elif "Alltoall" in algorithm.name:
            algo_elem.set('coll', "alltoall")
            algo_elem.set('inplace', "0")
        elif "Allreduce" in algorithm.name:
            algo_elem.set('coll', "allreduce")
            algo_elem.set('inplace', "1")
        elif "ReduceScatter" in algorithm.name:
            algo_elem.set('coll', "reduce_scatter")
            algo_elem.set('inplace', "1")
        algo_elem.set('outofplace', "0" if algo_elem.get('inplace') == "1" else "1")
        algo_elem.set('redop', "nop")
        algo_elem.set('ngpus', str(len(gpus)))
    for rank, gpu in gpus.items():
        gpu_elem = ET.SubElement(algo_elem, 'gpu')
        gpu_elem.set('id', str(rank))
        gpu_elem.set('i_chunks', str(gpu.input_chunks))
        gpu_elem.set('o_chunks', str(gpu.output_chunks))
        gpu_elem.set('s_chunks', str(gpu.scratch_size()))
        for copy in gpu.copies:
            copy_elem = ET.SubElement(gpu_elem, 'copy')
            copy_elem.set('i_off', str(copy.input_offset))
            copy_elem.set('o_off', str(copy.output_offset))
        for tb in gpu.threadblocks:
            tb_elem = ET.SubElement(gpu_elem, 'tb')
            tb_elem.set('id', str(tb.rbid))
            tb_elem.set('send', str(tb.send))
            tb_elem.set('recv', str(tb.recv))
            tb_elem.set('chan', str(tb.channel))
            for op in tb.ops:
                op_elem = ET.SubElement(tb_elem, 'op' if not old_format else 'step')
                op_elem.set('step' if not old_format else 's', str(op.idx))
                op_elem.set('type', op.op_type)

                # The NCCL backend currently wants scratch at the end of output
                if not use_scratch:
                    if op.src_buffer == 's':
                        op.src_buffer = 'o'
                        op.src_offset += gpu.output_chunks
                    if op.dst_buffer == 's':
                        op.dst_buffer = 'o'
                        op.dst_offset += gpu.output_chunks

                if old_format:
                    if op.src_buffer is not None:
                        op_elem.set('srcbuf', op.src_buffer)
                        op_elem.set('srcoff', str(op.src_offset))
                    else:
                        op_elem.set('srcbuf', 'i')
                        op_elem.set('srcoff', '-1')
                    if op.dst_buffer is not None:
                        op_elem.set('dstbuf', op.dst_buffer)
                        op_elem.set('dstoff', str(op.dst_offset))
                    else:
                        op_elem.set('dstbuf', 'o')
                        op_elem.set('dstoff', '-1')
                else:
                    if op.is_send:
                        if op.src_buffer is not None:
                            op_elem.set('buf', op.src_buffer)
                            op_elem.set('off', str(op.src_offset))
                    else:
                        if op.dst_buffer is not None:
                            op_elem.set('buf', op.dst_buffer)
                            op_elem.set('off', str(op.dst_offset))
                if op.cnt > 1 or old_format:
                    op_elem.set('cnt', str(op.cnt))
                assert len(op.depends) <= 1
                if len(op.depends) == 1:
                    # These were previously guarded by a `make_none` flag that was only ever
                    # assigned False, so the guards were unreachable and the values below were
                    # always emitted. Assert instead of silently writing depid="None", which the
                    # runtime would parse as garbage.
                    dep = op.depends[0]
                    assert dep.block_rbid is not None and dep.idx is not None, (
                        f'dependency of {op.op_type} on gpu {op.gpu} was never assigned a '
                        f'threadblock/index: rbid={dep.block_rbid} idx={dep.idx}')
                    op_elem.set('depid', str(dep.block_rbid))
                    op_elem.set('deps', str(dep.idx))
                elif old_format:
                    op_elem.set('depid', '-1')
                    op_elem.set('deps', '-1')
                if op.has_dependence:
                    op_elem.set('hasdep', '1')
                elif old_format:
                    op_elem.set('hasdep', '0')
                if op.mscclflowid is not None:
                    op_elem.set('mscclflowid', str(op.mscclflowid))
                # Physical send rate (GB/s): op.cnt pieces at op.piece_rate each.
                # Merges only combine chunks within a single step, so an op never
                # spans epochs and its rate is well-defined. op.piece_rate is None
                # for a deliberately unpaced flow, which emits no attribute at all.
                if op.piece_rate is not None:
                    op_elem.set('rate', str(op.cnt * op.piece_rate))
                # Optional per-send epoch record, keyed by final XML location (gpu, tb, s).
                # Epoch ordering is now enforced in-band by _realize_pacing_gates (extra op.depends
                # before nop expansion), so this manifest no longer drives it; it is retained only
                # as an informational hook for callers that want a per-op epoch map.
                if send_epoch_manifest is not None and op.is_send:
                    send_epoch_manifest.append(
                        {'gpu': rank, 'tb': tb.rbid, 's': op.idx, 'epoch': op.step})

    if pretty_print:
        ET.indent(algo_elem, space='  ')
    return ET.tostring(algo_elem, encoding='unicode')
