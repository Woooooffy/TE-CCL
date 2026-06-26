#!/usr/bin/env python3
"""
Convert a TE-CCL schedule JSON (as produced by teccl/scheduler.py, e.g.
teccl/examples/schedules/ndv2_schedule.json) into the NCCL/MSCCL XML format,
by injecting it into TACCL's Algorithm representation and running TACCL's
own ncclize().

Usage:
    python teccl_ncclize.py SCHEDULE.json -o OUTPUT.xml --taccl-path /path/to/taccl

Requires the taccl package (https://github.com/microsoft/taccl) to be importable
(pass --taccl-path, or have it on PYTHONPATH), plus its ncclize dependencies
(lxml, z3-solver).
"""
import argparse
import json
import re
import sys
from collections import defaultdict

FLOW_RE = re.compile(
    r'Chunk (\d+) from (\d+) traveled over (\d+)->(\d+) in epoch (\d+)'
)


def parse_flows(schedule):
    """Group TE-CCL's '7-Flows' entries by epoch, remapping GPU ids to a dense
    0-indexed range.

    TE-CCL's node ids are raw 0-indexed topology indices that may include
    switch nodes at arbitrary positions (e.g. index 0 for NDv2, the last index
    for Star). Switch hops never appear as the 'A->B' endpoints in 7-Flows --
    TE-CCL's own flow-merging collapses them into a trailing "via switches"
    annotation (ignored here) -- so the set of ids that *do* appear as an
    origin/src/dst is exactly the set of real GPUs, and we remap that set to
    0..N-1 instead of assuming any fixed offset.

    Returns (num_nodes, num_subchunks, steps_in_order) where steps_in_order is a
    list of lists of (global_chunk_id, src, dst) 0-indexed tuples, one list per
    non-empty epoch, in increasing epoch order.
    """
    parsed = []
    raw_ids = set()
    max_subchunk = 0

    for line in schedule['7-Flows']:
        m = FLOW_RE.match(line)
        if not m:
            raise ValueError(f'Could not parse flow line: {line!r}')
        subchunk, origin, src, dst, epoch = (int(x) for x in m.groups())
        raw_ids.update((origin, src, dst))
        max_subchunk = max(max_subchunk, subchunk)
        parsed.append((epoch, subchunk, origin, src, dst))

    rank_map = {raw: idx for idx, raw in enumerate(sorted(raw_ids))}
    num_nodes = len(rank_map)
    num_subchunks = max_subchunk + 1

    by_epoch = defaultdict(list)
    for epoch, subchunk, origin, src, dst in parsed:
        chunk_id = rank_map[origin] * num_subchunks + subchunk
        by_epoch[epoch].append((chunk_id, rank_map[src], rank_map[dst]))

    steps_in_order = [by_epoch[epoch] for epoch in sorted(by_epoch)]

    return num_nodes, num_subchunks, steps_in_order


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

    num_nodes, num_subchunks, steps_in_order = parse_flows(schedule)

    collective = allgather(num_nodes)
    if num_subchunks > 1:
        collective = collective.chunk_up(num_subchunks)

    topology = TeCCLTopology(name, num_nodes, steps_in_order)

    steps = [Step(1, sends) for sends in steps_in_order]

    instance = Instance(steps=len(steps), extra_rounds=0, chunks=num_subchunks)

    return Algorithm.make_implementation(
        collective, topology, instance, steps, cont=False, suffix='-teccl')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule', help='TE-CCL schedule JSON file')
    p.add_argument('-o', '--output', required=True, help='output XML file')
    p.add_argument('--instances', type=int, default=1)
    p.add_argument('--scale-remote', type=int, default=1)
    args = p.parse_args()

    if args.taccl_path:
        sys.path.insert(0, args.taccl_path)

    from taccl_ncclize import ncclize, ChannelPolicy

    with open(args.schedule) as f:
        schedule = json.load(f)

    algo = build_algorithm(schedule)

    xml = ncclize(
        algo,
        channel_policy=ChannelPolicy.MatchTopology,
        old_format=True,
        use_scratch=True,
        instances=args.instances,
        scale_remote=args.scale_remote,
        logging=True,
    )

    with open(args.output, 'w') as f:
        f.write(xml)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
