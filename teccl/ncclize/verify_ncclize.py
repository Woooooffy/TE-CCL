#!/usr/bin/env python3
"""
Regenerate every schedule under teccl/examples/schedules/ through
teccl_ncclize.py and sanity-check the resulting XML.

There is no automated test suite for this pipeline, so this script is the
closest thing to a regression check: it doesn't validate the *algorithm* is
correct, only that ncclize() ran to completion and produced structurally
sound XML -- well-formed, with every depid/deps reference resolving to a real
threadblock/step on the same GPU, and with both ends of every (peer, channel)
connection agreeing on operation order (see check_connection_fifo, a failure
class that is invisible in the XML and shows up only as corrupt data at
runtime).

Usage:
    python verify_ncclize.py [SCHEDULE.json ...]

With no arguments, every *.json file in teccl/examples/schedules/ is checked.
"""
import argparse
import collections
import glob
import os
import sys

from lxml import etree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teccl_ncclize import build_algorithm
import json


def generate_xml(schedule_path, hierarchical=False):
    with open(schedule_path) as f:
        schedule = json.load(f)

    from taccl_ncclize import ncclize, ChannelPolicy
    from helpers import (check_epoch_ordering_feasibility,
                         warn_epoch_ordering_violations)

    (algo, flow_path_keys, switch_rank_map,
     gpu_epoch_view, piece_rate, pacing_gates) = build_algorithm(schedule)
    # The flat-axis feasibility check only applies to a SINGLE-LEVEL (flat) solve; a hierarchical
    # multi-level schedule interleaves per-level epoch grids, so its network-layer pacing is checked
    # per-layer in the stitch instead. This is about how it was SOLVED (flat vs hierarchical), not
    # the schedule FORMAT (a flat LP solve still wants the check), so the caller states it.
    if not hierarchical:
        warn_epoch_ordering_violations(check_epoch_ordering_feasibility(gpu_epoch_view))
    flow_manifest = []
    # Send pacing is realized inside ncclize from the pacing_gates manifest; no XML post-pass.
    return ncclize(
        algo,
        channel_policy=ChannelPolicy.MatchTopology,
        old_format=True,
        use_scratch=True,
        instances=1,
        scale_remote=1,
        flow_path_keys=flow_path_keys,
        flow_manifest=flow_manifest,
        piece_rate=piece_rate,
        pacing_gates=pacing_gates,
        logging=False,
    )


def check_depid_deps(root):
    """Every depid!=-1 must reference an existing <tb id=...> on the same
    <gpu>, and deps must reference a step s= value that actually exists in
    that target tb. Returns a list of error strings (empty if all good)."""
    errors = []
    for gpu_elem in root.findall('gpu'):
        tbs_by_id = {}
        for tb_elem in gpu_elem.findall('tb'):
            tb_id = int(tb_elem.get('id'))
            steps = {int(s.get('s')) for s in tb_elem.findall('step')}
            tbs_by_id[tb_id] = steps

        for tb_elem in gpu_elem.findall('tb'):
            for step_elem in tb_elem.findall('step'):
                depid = int(step_elem.get('depid'))
                deps = int(step_elem.get('deps'))
                if depid == -1:
                    continue
                if depid not in tbs_by_id:
                    errors.append(
                        f'gpu={gpu_elem.get("id")} tb={tb_elem.get("id")} '
                        f's={step_elem.get("s")}: depid={depid} does not exist')
                    continue
                if deps not in tbs_by_id[depid]:
                    errors.append(
                        f'gpu={gpu_elem.get("id")} tb={tb_elem.get("id")} '
                        f's={step_elem.get("s")}: deps={deps} not found in tb={depid}')
    return errors


def check_connection_fifo(root):
    """The two ends of every (peer, channel) must agree on operation order.

    An MSCCL connection is scoped to (channelId, peer) and is a FIFO: the receiver matches
    its n-th recv against the sender's n-th send on that connection, with no key beyond
    position. So for every ordered pair, the sequence of transfers the sender emits must be
    identical to the sequence the receiver expects. If they diverge, the runtime pairs a
    send with the wrong recv and writes the wrong bytes into the wrong buffer -- silent
    corruption, with nothing in the XML itself that looks wrong.

    This once happened for real: threadblocks used to be packed across peers, which forced
    recvs to be ordered by their true arrival epoch while sends stayed ordered by departure,
    and 3 of the example schedules emitted mismatched pairings. Threadblocks are now one per
    (gpu, direction, peer, channel) with send and recv sharing a .step, which makes the
    orders identical by construction -- this check is what keeps it that way.

    A transfer is identified by its buffer coordinates, which sender and receiver both
    record on their respective ops. Returns a list of error strings (empty if all good).
    """
    sends = collections.defaultdict(list)
    recvs = collections.defaultdict(list)
    for gpu_elem in root.findall('gpu'):
        gpu = int(gpu_elem.get('id'))
        for tb_elem in gpu_elem.findall('tb'):
            chan = int(tb_elem.get('chan'))
            send_peer, recv_peer = int(tb_elem.get('send')), int(tb_elem.get('recv'))
            for step_elem in tb_elem.findall('step'):
                kind = step_elem.get('type')
                key = tuple(step_elem.get(a) for a in
                            ('srcbuf', 'srcoff', 'dstbuf', 'dstoff', 'cnt'))
                if kind == 's':
                    sends[(gpu, send_peer, chan)].append(key)
                elif kind in ('r', 'rrc'):
                    recvs[(gpu, recv_peer, chan)].append(key)

    errors = []
    for (gpu, peer, chan), sent in sorted(sends.items()):
        received = recvs.get((peer, gpu, chan))
        if received is None:
            errors.append(f'gpu={gpu} sends to peer={peer} on chan={chan}, but peer has no '
                          f'matching recv threadblock on that channel')
        elif received != sent:
            first = next((i for i, (a, b) in enumerate(zip(sent, received)) if a != b),
                         min(len(sent), len(received)))
            errors.append(
                f'gpu={gpu} -> peer={peer} chan={chan}: send/recv order diverges at '
                f'position {first} (sender has {len(sent)} op(s), receiver {len(received)}); '
                f'sender {sent[first] if first < len(sent) else "<end>"} vs receiver '
                f'{received[first] if first < len(received) else "<end>"}')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('schedules', nargs='*',
                    help='schedule JSON files to check; defaults to all of '
                         'teccl/examples/schedules/*.json')
    p.add_argument('--hierarchical', action='store_true',
                    help='treat the given schedule(s) as hierarchical multi-level solves and skip '
                         'the flat-axis feasibility warnings (see generate_xml). The default '
                         'example suite is all flat, so omit it there.')
    args = p.parse_args()

    schedules = args.schedules
    if not schedules:
        here = os.path.dirname(os.path.abspath(__file__))
        schedules = sorted(glob.glob(os.path.join(here, '..', 'examples', 'schedules', '*.json')))

    failures = 0
    for schedule_path in schedules:
        name = os.path.basename(schedule_path)
        try:
            xml_str = generate_xml(schedule_path, hierarchical=args.hierarchical)
        except Exception as e:
            print(f'[FAIL] {name}: exception during generation: {e}')
            failures += 1
            continue

        try:
            root = ET.fromstring(xml_str.encode('utf-8'))
        except ET.XMLSyntaxError as e:
            print(f'[FAIL] {name}: XML did not parse: {e}')
            failures += 1
            continue

        errors = check_depid_deps(root)
        if errors:
            print(f'[FAIL] {name}: {len(errors)} depid/deps integrity error(s)')
            for err in errors[:5]:
                print(f'    {err}')
            failures += 1
            continue

        errors = check_connection_fifo(root)
        if errors:
            print(f'[FAIL] {name}: {len(errors)} connection FIFO ordering error(s)')
            for err in errors[:5]:
                print(f'    {err}')
            failures += 1
            continue

        print(f'[OK]   {name}')

    if failures:
        print(f'\n{failures} of {len(schedules)} schedule(s) failed.')
        sys.exit(1)
    print(f'\nAll {len(schedules)} schedule(s) passed.')


if __name__ == '__main__':
    main()
