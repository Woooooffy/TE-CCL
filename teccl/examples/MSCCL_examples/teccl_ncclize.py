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
    """Group TE-CCL's '7-Flows' entries by epoch, converting to 0-indexed ranks.

    Returns (num_nodes, num_subchunks, steps_in_order) where steps_in_order is a
    list of lists of (global_chunk_id, src, dst) 0-indexed tuples, one list per
    non-empty epoch, in increasing epoch order.
    """
    by_epoch = defaultdict(list)
    max_node = 0
    max_subchunk = 0

    for line in schedule['7-Flows']:
        m = FLOW_RE.match(line)
        if not m:
            raise ValueError(f'Could not parse flow line: {line!r}')
        subchunk, origin, src, dst, epoch = (int(x) for x in m.groups())
        max_node = max(max_node, origin, src, dst)
        max_subchunk = max(max_subchunk, subchunk)
        # 0-index everything; "via switches" routing is metadata only and is
        # dropped here -- it doesn't change which GPU-to-GPU sends happen.
        by_epoch[epoch].append((subchunk, origin - 1, src - 1, dst - 1))

    num_nodes = max_node  # GPUs are 1..N, so N == max_node
    num_subchunks = max_subchunk + 1

    steps_in_order = []
    for epoch in sorted(by_epoch):
        sends = []
        for subchunk, origin, src, dst in by_epoch[epoch]:
            chunk_id = origin * num_subchunks + subchunk
            sends.append((chunk_id, src, dst))
        steps_in_order.append(sends)

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
    from taccl.algorithm import Algorithm, Step
    from taccl.collectives import allgather
    from taccl.instance import Instance

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
    p.add_argument('schedule', help='TE-CCL schedule JSON file')
    p.add_argument('-o', '--output', required=True, help='output XML file')
    p.add_argument('--taccl-path', default=None,
                    help='path to a cloned taccl repo, if not already importable')
    p.add_argument('--instances', type=int, default=1)
    p.add_argument('--scale-remote', type=int, default=1)
    args = p.parse_args()

    if args.taccl_path:
        sys.path.insert(0, args.taccl_path)

    from taccl.ncclize import ncclize, ChannelPolicy

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
