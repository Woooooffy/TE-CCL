#!/usr/bin/env python3
"""
Regenerate every schedule under teccl/examples/schedules/ through
teccl_ncclize.py and sanity-check the resulting XML.

There is no automated test suite for this pipeline, so this script is the
closest thing to a regression check: it doesn't validate the *algorithm* is
correct, only that ncclize() ran to completion and produced structurally
sound XML (well-formed, with every depid/deps reference resolving to a real
threadblock/step on the same GPU).

Usage:
    python verify_ncclize.py [SCHEDULE.json ...]

With no arguments, every *.json file in teccl/examples/schedules/ is checked.
"""
import argparse
import glob
import os
import sys

from lxml import etree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teccl_ncclize import build_algorithm, enforce_send_epoch_ordering
import json


def generate_xml(schedule_path):
    with open(schedule_path) as f:
        schedule = json.load(f)

    from taccl_ncclize import ncclize, ChannelPolicy

    algo, flow_path_keys, switch_rank_map, flow_completion_steps = build_algorithm(schedule)
    flow_manifest = []
    xml = ncclize(
        algo,
        channel_policy=ChannelPolicy.MatchTopology,
        old_format=True,
        use_scratch=True,
        instances=1,
        scale_remote=1,
        flow_path_keys=flow_path_keys,
        flow_manifest=flow_manifest,
        flow_completion_steps=flow_completion_steps,
        logging=False,
    )
    return enforce_send_epoch_ordering(xml, flow_manifest)


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('schedules', nargs='*',
                    help='schedule JSON files to check; defaults to all of '
                         'teccl/examples/schedules/*.json')
    args = p.parse_args()

    schedules = args.schedules
    if not schedules:
        here = os.path.dirname(os.path.abspath(__file__))
        schedules = sorted(glob.glob(os.path.join(here, '..', 'examples', 'schedules', '*.json')))

    failures = 0
    for schedule_path in schedules:
        name = os.path.basename(schedule_path)
        try:
            xml_str = generate_xml(schedule_path)
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

        print(f'[OK]   {name}')

    if failures:
        print(f'\n{failures} of {len(schedules)} schedule(s) failed.')
        sys.exit(1)
    print(f'\nAll {len(schedules)} schedule(s) passed.')


if __name__ == '__main__':
    main()
