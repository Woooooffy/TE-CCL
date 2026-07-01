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
import json
import re
import sys
from collections import defaultdict

FLOW_RE = re.compile(
    r'Chunk (\d+) from (\d+) traveled over (\d+)->(\d+) in epoch (\d+)'
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
    switch_rank_map):
    - steps_in_order is a list of lists of (global_chunk_id, src, dst)
      0-indexed tuples, one list per non-empty epoch, in increasing epoch
      order.
    - flow_path_keys maps (step_idx, global_chunk_id, src, dst) -> path key,
      where step_idx is the index into steps_in_order (not the raw epoch
      number) and the path key is whatever _parse_switch_path() returned for
      that flow (a tuple of raw switch ids, or None for a direct hop). Used
      downstream to avoid merging chunks that took different switch paths
      into a single send op.
    - switch_rank_map maps each raw switch id appearing in any path key to a
      dense 0-indexed id, the same way rank_map does for GPU ids.
    """
    parsed = []
    raw_ids = set()
    switch_raw_ids = set()
    max_subchunk = 0

    for line in schedule['7-Flows']:
        m = FLOW_RE.match(line)
        if not m:
            raise ValueError(f'Could not parse flow line: {line!r}')
        subchunk, origin, src, dst, epoch = (
            int(x) for x in m.group(1, 2, 3, 4, 5))
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

    return num_nodes, num_subchunks, steps_in_order, flow_path_keys, switch_rank_map


def build_switch_routes(flow_manifest, switch_rank_map):
    """Build a per-switch flow_id -> next-hop forwarding table.

    flow_manifest is the list of {'flow_id', 'step', 'src', 'dst', 'path_key'}
    records recorded by ncclize() (one per emitted send op). Since ncclize()
    only merges chunks sharing the same path_key into one op (see
    flow_path_keys), each flow_id here has exactly one, unambiguous path.

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
        flow_id, src, dst, step = (
            record['flow_id'], record['src'], record['dst'], record['step'])
        for i, switch in enumerate(switch_path):
            is_last = i == len(switch_path) - 1
            next_hop_type = 'gpu' if is_last else 'switch'
            next_hop = dst if is_last else switch_path[i + 1]
            routes[switch][flow_id] = {
                'next_hop_type': next_hop_type,
                'next_hop': next_hop,
                'src_gpu': src,
                'dst_gpu': dst,
                'step': step,
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


def build_algorithm(schedule, name='teccl'):
    from taccl_algorithm import Algorithm, Step
    from taccl_collectives import allgather
    from taccl_instance import Instance

    num_nodes, num_subchunks, steps_in_order, flow_path_keys, switch_rank_map = parse_flows(schedule)

    collective = allgather(num_nodes)
    if num_subchunks > 1:
        collective = collective.chunk_up(num_subchunks)

    topology = TeCCLTopology(name, num_nodes, steps_in_order)

    steps = [Step(1, sends) for sends in steps_in_order]

    instance = Instance(steps=len(steps), extra_rounds=0, chunks=num_subchunks)

    algo = Algorithm.make_implementation(
        collective, topology, instance, steps, cont=False, suffix='-teccl')
    return algo, flow_path_keys, switch_rank_map


def enforce_send_epoch_ordering(xml_str, flow_manifest):
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
    """
    import xml.etree.ElementTree as ET

    # flow_id -> epoch index (= index into steps_in_order)
    flow_epoch = {r['flow_id']: r['step'] for r in flow_manifest}

    root = ET.fromstring(xml_str)

    for gpu_elem in root.findall('gpu'):
        # Collect every send op: (epoch, tb_rbid, s_idx, step_elem)
        sends = []
        for tb_elem in gpu_elem.findall('tb'):
            tb_rbid = int(tb_elem.get('id'))
            for step_elem in tb_elem.findall('step'):
                if step_elem.get('type') != 's':
                    continue
                fid_str = step_elem.get('mscclflowid')
                if fid_str is None:
                    continue
                epoch = flow_epoch.get(int(fid_str))
                if epoch is None:
                    continue
                sends.append((epoch, tb_rbid, int(step_elem.get('s')), step_elem))

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
    args = p.parse_args()

    from taccl_ncclize import ncclize, ChannelPolicy

    with open(args.schedule) as f:
        schedule = json.load(f)

    algo, flow_path_keys, switch_rank_map = build_algorithm(schedule)

    # Always collect flow_manifest: needed for epoch ordering and optional routing.
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
        logging=True,
    )

    xml = enforce_send_epoch_ordering(xml, flow_manifest)

    with open(args.output, 'w') as f:
        f.write(xml)
    print(f'Wrote {args.output}')

    if args.switch_routing_output:
        routes = build_switch_routes(flow_manifest, switch_rank_map)
        with open(args.switch_routing_output, 'w') as f:
            json.dump(routes, f, indent=2)
        print(f'Wrote {args.switch_routing_output}')


if __name__ == '__main__':
    main()
