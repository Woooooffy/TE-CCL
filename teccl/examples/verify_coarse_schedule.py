"""
Verify a coarse rail-optimized AllGather schedule (LP or MILP) for correct, expected behavior.

Checks, from the schedule JSON alone:
  - every (src, dst, chunk) allgather demand is delivered in full (volume ~1.0);
  - per-source egress volume (copy-free LP => one copy per (dest, chunk); multicast MILP => far
    less, since leaves/spines replicate);
  - which switches carry traffic (leaves vs spines) and whether flow volumes are integral or
    fractionally split across rails;
  - epochs required vs the copy-free unicast bandwidth lower bound.

Usage (no Gurobi needed):
    python -m teccl.examples.verify_coarse_schedule Schedules/coarse_rail_lp_8chunks.json

The rail-optimized coarse ids are hosts 0..31, leaves 32..39, spines 40..43.
"""
import json
import re
import sys
from collections import Counter, defaultdict

FLOW_RE = re.compile(
    r'Chunk (\d+) from (\d+) traveled over (\d+)->(\d+) with volume ([\d.eE+-]+) '
    r'in epoch (\d+)(?: via switches (.*))?')
DEMAND_RE = re.compile(r'Demand at (\d+) for chunk (\d+) from (\d+) met by epoch (\d+)')

LEAF_IDS = set(range(32, 40))
SPINE_IDS = set(range(40, 44))


def verify(path: str) -> None:
    d = json.load(open(path))
    print(f"=== {path} ===")
    for k in ("0-Collective", "1-Epoch_Duration", "3-Epochs_Required",
              "4-Collective_Finish_Time", "5-Algo_Bandwidth", "Solver_Time"):
        if k in d:
            print(f"  {k}: {d[k]}")

    flows = d.get("7-Flows", [])
    per_demand = defaultdict(float)      # (src, dst, chunk) -> delivered volume (final hop into dst)
    src_egress = defaultdict(float)      # src -> total volume leaving src (first hop out of a host)
    chunks_of_src = defaultdict(set)
    switches = Counter()
    vols = []
    epochs = []
    hosts = set()
    for f in flows:
        m = FLOW_RE.match(f)
        if not m:
            print("  UNPARSED FLOW:", f)
            continue
        chunk, src, a, b, vol, e, sws = m.groups()
        chunk, src, a, b, vol, e = int(chunk), int(src), int(a), int(b), float(vol), int(e)
        vols.append(vol)
        epochs.append(e)
        chunks_of_src[src].add(chunk)
        hosts.update((a, b))  # host endpoints (switches appear only in the 'via' annotation)
        if sws:
            for s in sws.split('->'):
                switches[int(s)] += 1
        # a->b endpoints are hosts; b is the delivery target for (src, chunk).
        per_demand[(src, b, chunk)] += vol
        src_egress[src] += vol

    hosts = {h for h in hosts if h < 32}
    num_hosts = len(hosts)
    chunks_per_host = max((len(cs) for cs in chunks_of_src.values()), default=0)

    print("\n  -- delivery completeness --")
    print(f"  distinct sources: {len(chunks_of_src)}, chunks/source: {chunks_per_host}, hosts seen: {num_hosts}")
    expected_demands = num_hosts * (num_hosts - 1) * chunks_per_host
    delivered_full = sum(1 for v in per_demand.values() if abs(v - 1.0) < 1e-4)
    print(f"  (src,dst,chunk) demands present: {len(per_demand)} (expect {expected_demands})")
    print(f"  delivered in full (~1.0): {delivered_full} / {len(per_demand)}")
    if per_demand:
        print(f"  delivered volume min/max: {min(per_demand.values()):.6f} / {max(per_demand.values()):.6f}")

    print("\n  -- egress / routing --")
    if src_egress:
        eg = list(src_egress.values())
        print(f"  per-source egress volume min/max: {min(eg):.4f} / {max(eg):.4f}")
        unicast_egress = (num_hosts - 1) * chunks_per_host
        print(f"  copy-free unicast egress would be {unicast_egress} per source "
              f"({'MATCHES => no multicast' if abs(max(eg) - unicast_egress) < 1e-3 else 'LESS => multicast/copy in use'})")
    used_leaves = sorted(s for s in switches if s in LEAF_IDS)
    used_spines = sorted(s for s in switches if s in SPINE_IDS)
    print(f"  leaves carrying traffic: {used_leaves}")
    print(f"  spines carrying traffic: {used_spines}  ({'none' if not used_spines else 'SPINE PATHS USED'})")

    print("\n  -- flow volume structure --")
    vc = Counter(round(v, 6) for v in vols)
    print(f"  distinct flow volumes: {dict(sorted(vc.items()))}")
    print(f"  epoch histogram (flow starts): {sorted(Counter(epochs).items())}")

    print("\n  -- epoch / bandwidth bound --")
    # Copy-free bound: each host egresses (num_hosts-1)*chunks_per_host copies; egress capacity
    # is 8 rails * 1 chunk/epoch = 8 chunks/epoch (rail 50 GB/s * 0.02 epoch = 1 chunk).
    import math
    rails = 8
    bound = math.ceil((num_hosts - 1) * chunks_per_host / rails) if num_hosts else 0
    print(f"  copy-free unicast epoch lower bound = ceil((#hosts-1)*chunks/{rails} rails) = {bound}")
    print(f"  schedule Epochs_Required = {d.get('3-Epochs_Required')} "
          f"({'== bound (unicast-optimal)' if d.get('3-Epochs_Required') == bound else 'BELOW bound => multicast, or ABOVE => suboptimal'})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m teccl.examples.verify_coarse_schedule <schedule.json> [<schedule.json> ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        verify(p)
        print()
