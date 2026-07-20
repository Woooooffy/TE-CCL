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
    its own docstring -- and even where true completion times are available via
    flow_completion_steps, they're only precise up to the send-epoch dense scale, a
    deliberate, separately-documented tradeoff). Rather than reorder ops on a shared
    channel using that imprecise data, this function avoids the problem structurally: two
    flows that took different paths are *never* placed on the same channel, so there is no
    intra-channel relative order to get right in the first place. This is sufficient (not
    just a heuristic) because flows sharing one path_key inherently can't be reordered
    relative to each other -- same physical route means departure order == completion
    order -- so keeping same-path flows on shared channels (round-robinned across replicas
    exactly as before) remains correct.

    Note this only fixes *intra-channel* wire order (which recv/send ends up in which
    array slot within one threadblock's own (gpu,peer,chan) group). It does NOT replace
    flow_completion_steps, which fixes a different hazard: TACCL's threadblock-packing
    reuses one TB across *different* peers when their .step values don't collide (see the
    eligibility check in ncclize(), a few hundred lines below), and once two peers'
    ops share a TB, ALL of that TB's ops -- regardless of peer -- get serialized together
    by .step. If a switch-relayed recv's .step were left at its (too-early) send-start
    epoch instead of its corrected true-completion epoch, an unrelated send to a
    completely different peer could get sorted after it and stall behind a relay it has no
    real data dependency on. That cross-peer TB-packing hazard is orthogonal to path
    diversity on a single edge (it's about threadblock-count minimization across peers,
    not about multipath at all), so flow_completion_steps's correction stays necessary
    here regardless of the path-aware channel assignment below.

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

def ncclize(algorithm, remap_scratch = None, channel_policy=ChannelPolicy.MatchTopology, pretty_print = True, old_format=False, use_scratch=False, merge_contiguous=True, instances=1, scale_remote=1, combine_contig=False, aid_IB_contig=False, prefix="", logging=False, flow_path_keys=None, flow_manifest=None, flow_completion_steps=None, piece_rate=None, send_epoch_manifest=None, max_channels=32):
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

    flow_completion_steps, if given, maps (step_idx, addr, src, dst) -> a dense step-domain value, comparable to
    step_idx, representing the epoch by which dst truly has the data (as opposed to step_idx, which is only the
    epoch the transfer started). Used to correct the .step of the corresponding recv op in place, so that
    threadblock-internal step ordering and threadblock-reuse eligibility reflect true data availability rather than
    merely transfer start time -- relevant when a transfer is relayed (e.g. through switches) and so completes
    later than it started. None preserves prior behavior (recv ops keep their send-side step_idx). This remains
    necessary alongside flow_path_keys/path-aware channels above -- it fixes a different hazard (an unrelated op
    sharing a threadblock, across peers, with a switch-relayed recv), not the intra-channel multipath ordering
    that path-aware channel assignment fixes structurally.

    max_channels caps how many channel ids a single edge's path_key partitioning may use under channel_policy=
    MatchTopology (default 32, matching NCCL/MSCCL's MAXCHANNELS hardware limit); see
    _allocate_channels_match_topology().
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
    # chunks) shares one id (see the flow_id assignment below).
    route_flow_ids = {}
    # Track the latest op that wrote to each buffer index
    writers = defaultdict(list)
    # Track all the reads since the last write to each buffer index
    readers = defaultdict(list)
    for step_idx, step in enumerate(algorithm.steps):
        new_writers = defaultdict(list)
        new_readers = defaultdict(list)

        # Group sent addresses by edge
        grouped_sends = defaultdict(set)
        # Maps (src, dst, path_key) -> corrected recv .step (see flow_completion_steps
        # docstring above); only populated in the 3-tuple step.sends branch below, since
        # that's the only shape teccl_ncclize.py ever produces.
        group_completion_step = {}
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
            if flow_completion_steps:
                for (src, dst, path_key), addrs in grouped_sends.items():
                    vals = [flow_completion_steps.get((step_idx, addr, src, dst)) for addr in addrs]
                    vals = [v for v in vals if v is not None]
                    # max(): if merge_contiguous combined several chunk-ids into one op,
                    # the op isn't truly ready until the last of its constituents arrives.
                    if vals:
                        group_completion_step[(src, dst, path_key)] = max(vals)

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

            # Correct the recv op's .step to when dst truly has the data, rather than
            # when the transfer started, for relayed transfers (see flow_completion_steps
            # docstring above). send_op.step is left as the true start epoch, which is
            # already correct for a send.
            recv_op.step = group_completion_step.get((src, dst, path_key), recv_op.step)

            # The flow id is a bijection with the physical route (src -> switches
            # -> dst), keyed by (src, dst, path_key): every send/recv on that
            # route, in any epoch and for any chunk, shares one id. This gives the
            # switch forwarding table and channel assignment exactly one entry per
            # route. Per-op epoch ordering is recovered from send_epoch_manifest
            # (populated at XML emission), not from the flow id, precisely because
            # one flow id now spans multiple epochs.
            route = (src, dst, path_key)
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
        # Sanity check: ops that were kept separate because they carry distinct
        # mscclflowids (e.g. different switch paths) are only guaranteed to land
        # in separate threadblocks if they also land on distinct channels here --
        # threadblocks are grouped by (gpu, is_send, peer, channel) below, so two
        # same-step ops sharing a channel get merged into one threadblock
        # regardless of having different flow ids. This can happen if channel_policy
        # doesn't separate same-step/same-edge ops (e.g. ChannelPolicy.One).
        #
        # Under channel_policy=MatchTopology this should now be structurally
        # unreachable: _allocate_channels_match_topology() partitions every edge by
        # op.path_key first, so ops with different path_keys always land in disjoint
        # channel sub-ranges. If this warning ever fires under MatchTopology, that's a
        # bug in that partitioning (e.g. a path_key not making it onto the op), not an
        # inherent limitation -- kept here as a cheap regression guard, and because
        # it's still a real limitation for other channel policies (e.g. One).
        flow_chans = defaultdict(set)
        for chan, chan_ops in ops_by_channel.items():
            for op in chan_ops:
                if op.mscclflowid is not None:
                    flow_chans[(op.gpu, op.is_send, op.peer, op.step)].add((op.mscclflowid, chan))
        for (gpu, is_send, peer, step), flowid_chan_pairs in flow_chans.items():
            flow_ids = {f for f, _ in flowid_chan_pairs}
            chans = {c for _, c in flowid_chan_pairs}
            if len(chans) < len(flow_ids):
                print(f'Warning: mscclflowids {sorted(flow_ids)} on gpu={gpu} is_send={is_send} '
                      f'peer={peer} step={step} only got {len(chans)} distinct channel(s) under '
                      f'channel_policy={channel_policy} -- they will be forced into the same '
                      f'threadblock, losing the intended parallelism between switch paths.')

    # Group by which operations need to be in the same threadblock
    tb_groups = defaultdict(list)
    for chan, chan_ops in ops_by_channel.items():
        for op in chan_ops:
            tb_groups[(op.gpu, op.is_send, op.peer, chan)].append(op)

    tbs_by_gpu_chan = defaultdict(lambda: defaultdict(list))
    # For each group find or create a threadblock to add them to
    for key, grp in tb_groups.items():
        rank, is_send, peer, chan = key
        make_none = False
        tbs = tbs_by_gpu_chan[rank][chan]
        for tb in tbs:
            tb_peer = tb.send if is_send else tb.recv
            # An existing threadblock can be reused if:
            # - Either the relevant peer is not set yet or the peer is the same
            # - No operations already in the threadblock execute in the same step
            if tb_peer == -1 or tb_peer == peer:
                if all(not any(op1.step == op2.step for op2 in grp) for op1 in tb.steps):
                    break
        else:
            # No existing threadblock was suitble, so create a new one
            tb = _Threadblock(chan)
            tbs.append(tb)
        # Ensure the peer is set correctly
        if is_send:
            assert tb.send == -1 or tb.send == peer
            tb.send = peer
        else:
            assert tb.recv == -1 or tb.recv == peer
            tb.recv = peer
        tb.steps.extend(grp)

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
                    if make_none and op.depends[0].block_rbid is None:
                        op_elem.set('depid', '-1')
                    else:
                        op_elem.set('depid', str(op.depends[0].block_rbid))
                    if make_none and op.depends[0].idx is None:
                        op_elem.set('deps', '-1')
                    else:
                        op_elem.set('deps', str(op.depends[0].idx))
                elif old_format:
                    op_elem.set('depid', '-1')
                    op_elem.set('deps', '-1')
                if op.has_dependence:
                    op_elem.set('hasdep', '1')
                elif old_format:
                    op_elem.set('hasdep', '0')
                if op.mscclflowid is not None:
                    op_elem.set('mscclflowid', str(op.mscclflowid))
                # Physical send rate (GB/s): op.cnt pieces at piece_rate each.
                # Merges only combine chunks within a single step, so an op
                # never spans epochs and its rate is well-defined.
                if piece_rate is not None:
                    op_elem.set('rate', str(op.cnt * piece_rate))
                # Record each send op's epoch, keyed by its final XML location
                # (gpu, tb, s). Epoch ordering can no longer read the epoch off
                # the flow id (one route-based id spans epochs), so it uses this.
                if send_epoch_manifest is not None and op.is_send:
                    send_epoch_manifest.append(
                        {'gpu': rank, 'tb': tb.rbid, 's': op.idx, 'epoch': op.step})

    if pretty_print:
        ET.indent(algo_elem, space='  ')
    return ET.tostring(algo_elem, encoding='unicode')
