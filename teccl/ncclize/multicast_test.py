"""Multicast in the XML and the switch-forwarding JSON (multicast_output_design.md).

Gurobi-free:
  [1] a switch copy is ONE send threadblock listing every receiver (send="1,2"), matched by a recv
      threadblock on each receiver on the same channel, all sharing the tree's mscclflowid
  [2] flow ids are a bijection with routes: a tree and a unicast along one of its branches get
      DIFFERENT ids and land on DIFFERENT channels of the edge they share
  [3] switch JSON format 2: a `flows` table (id -> route), and the branching switch's entry lists
      both next hops; check_switch_routes reproduces every route
  [4] --no-multicast is the pre-multicast output: one send per receiver, one id per branch path
Gurobi:
  [5] the hetero switch-copy run: GPU 4's copy is one send at 50 GB/s (was two, 100 GB/s on a 50
      GB/s uplink), and the uplink load read back from the XML fits every uplink; the switch table
      gives that flow two ports on switch 17

Run from the repo root:
    PYTHONPATH=. conda run -n teccl python teccl/ncclize/multicast_test.py
"""
import collections
import contextlib
import io
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lxml import etree as ET                                             # noqa: E402

from taccl_ncclize import ChannelPolicy, ncclize                         # noqa: E402
from teccl_ncclize import build_algorithm, build_switch_routes, check_switch_routes  # noqa: E402


def emit(info, multicast=True, topology=None, programmable=None):
    """(xml root, switch routes, epoch manifest) for one schedule, as the CLI produces them."""
    with contextlib.redirect_stdout(io.StringIO()):
        algo, fpk, srm, _view, rate, gates, grm = build_algorithm(
            info, topology=topology, multicast=multicast)
        manifest, epochs = [], []
        xml = ncclize(algo, channel_policy=ChannelPolicy.MatchTopology, old_format=True,
                      use_scratch=True, flow_path_keys=fpk, flow_manifest=manifest,
                      piece_rate=rate, pacing_gates=gates, send_epoch_manifest=epochs)
    routes = build_switch_routes(
        manifest, srm, programmable,
        physical_port=topology.physical_port if topology is not None else None,
        gpu_rank_map=grm, topology=topology)
    check_switch_routes(routes)
    return ET.fromstring(xml), routes, epochs


def tbs(root, gpu):
    return [tb for g in root.findall('gpu') if g.get('id') == str(gpu) for tb in g.findall('tb')]


def flow_ids(tb):
    """The flow ids a threadblock's transfers carry (dependency-expansion nops carry none)."""
    return {s.get('mscclflowid') for s in tb.findall('step') if s.get('type') != 'nop'}


# A 3-GPU broadcast from GPU 0 through one switch (node 10). Chunk 0 is ONE send copied by the
# switch to GPUs 1 and 2; chunk 1 goes to each of them as its own unicast send, over the same
# switch -- so edge 0->1 carries the tree and a unicast along one of its branches.
def broadcast_schedule():
    sends = [
        {"chunk": 0, "from": 0, "sender": 0, "epoch": 0, "span": 1, "rate": 1.0, "volume": 1,
         "branches": [{"to": 1, "via": [10], "ready": 1}, {"to": 2, "via": [10], "ready": 1}]},
        {"chunk": 1, "from": 0, "sender": 0, "epoch": 1, "span": 1, "rate": 1.0, "volume": 1,
         "branches": [{"to": 1, "via": [10], "ready": 2}]},
        {"chunk": 1, "from": 0, "sender": 0, "epoch": 2, "span": 1, "rate": 1.0, "volume": 1,
         "branches": [{"to": 2, "via": [10], "ready": 3}]},
    ]
    demands = {f"Demand at {t} for chunk {c} from 0 met by epoch {e}": []
               for t, c, e in ((1, 0, 1), (2, 0, 1), (1, 1, 2), (2, 1, 3))}
    return {"0-Collective": "broadcast", "0-Formulation": "LP", "0-Root": 0,
            "1-Epoch_Duration": 1.0, "9-Chunk_Size": 1.0, "7-Flows": [],
            "8-Chunk paths": demands, "10-Sends": sends}


def test_one_send_many_receivers():
    root, _routes, _ = emit(broadcast_schedule())
    send_tbs = {tb.get('send'): tb for tb in tbs(root, 0) if tb.get('send') != '-1'}
    assert '1,2' in send_tbs, f"the copy must be one threadblock to both receivers: {list(send_tbs)}"
    mc = send_tbs['1,2']
    [mc_id] = flow_ids(mc)
    for r in (1, 2):
        recv = [tb for tb in tbs(root, r) if tb.get('recv') == '0' and tb.get('chan') == mc.get('chan')]
        assert len(recv) == 1 and flow_ids(recv[0]) == {mc_id}, (r, [dict(t.attrib) for t in recv])
    print("  [1] switch copy: one send threadblock send=\"1,2\", a recv on each receiver, "
          "same channel, one shared flow id")


def test_tree_and_unicast_are_different_routes():
    root, routes, _ = emit(broadcast_schedule())
    send_tbs = {tb.get('send'): tb for tb in tbs(root, 0) if tb.get('send') != '-1'}
    tree, uni = send_tbs['1,2'], send_tbs['1']
    assert flow_ids(tree) != flow_ids(uni), "a tree and a unicast along its branch share an id"
    assert tree.get('chan') != uni.get('chan'), (
        "a tree and a unicast on the same edge 0->1 must land on different channels")
    # every route has exactly one id, and every id one route
    assert len(routes['flows']) == 3, routes['flows']
    print("  [2] tree and unicast along its branch: different flow ids, different channels "
          f"(chan {tree.get('chan')} vs {uni.get('chan')}); 3 routes, 3 ids")


def test_switch_json_format_2():
    _root, routes, _ = emit(broadcast_schedule())
    assert routes['format'] == 2
    trees = [fid for fid, f in routes['flows'].items() if len(f['branches']) == 2]
    assert len(trees) == 1, routes['flows']
    entry = routes['switches']['0'][trees[0]]
    assert entry['next_hops'] == [{'type': 'gpu', 'id': 1}, {'type': 'gpu', 'id': 2}], entry
    print("  [3] switch JSON format 2: flows table + the switch replicates the tree to both "
          "receivers; check_switch_routes reproduces every route")


def test_no_multicast_fallback():
    root, routes, _ = emit(broadcast_schedule(), multicast=False)
    sends = sorted(tb.get('send') for tb in tbs(root, 0) if tb.get('send') != '-1')
    assert all(',' not in s for s in sends), sends
    assert all(len(f['branches']) == 1 for f in routes['flows'].values()), routes['flows']
    # the copy's branches take the unicast paths' ids, shared with the genuine unicasts on them
    assert len(routes['flows']) == 2, routes['flows']
    print("  [4] --no-multicast: one send per receiver, one id per branch path (2 paths, 2 ids)")


def uplink_load(root, routes, epochs, topology):
    """Per (gpu, uplink switch, start epoch), the rate on the uplink read back from the XML: an op's
    rate and flow id from the XML, its first switch from the flows table. Threadblocks run
    concurrently, but a threadblock runs its ops strictly in order, so it contributes its PEAK op
    rate in an epoch, never a sum (several ops of one flow serialized in one threadblock each emit
    the group's total rate -- _widen_rates_for_tb_serialization)."""
    first_switch = {int(fid): f['branches'][0]['switches'][0]
                    for fid, f in routes['flows'].items() if f['branches'][0]['switches']}
    at = {(e['gpu'], e['tb'], e['s']): e['epoch'] for e in epochs}
    per_tb = collections.defaultdict(float)
    for g in root.findall('gpu'):
        for tb in g.findall('tb'):
            for st in tb.findall('step'):
                if st.get('type') != 's' or st.get('mscclflowid') is None:
                    continue
                gpu = int(g.get('id'))
                key = (gpu, first_switch[int(st.get('mscclflowid'))],
                       at[(gpu, int(tb.get('id')), int(st.get('s')))], tb.get('id'))
                per_tb[key] = max(per_tb[key], float(st.get('rate')))
    load = collections.defaultdict(float)
    for (gpu, sw, epoch, _tb), r in per_tb.items():
        load[(gpu, sw, epoch)] += r
    return {k: (v, topology.capacity[k[0]][k[1]]) for k, v in load.items()}


def test_hetero_switch_copy():
    from teccl.hierarchy.solve import solve_hierarchical
    from teccl.input_data import (Collective, EpochType, Formulation, InstanceParams,
                                  ObjectiveType, SolutionMethod, TopologyParams, UserInputParams)
    from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster

    ui = UserInputParams()
    ui.instance = InstanceParams(
        collective=Collective.ALLGATHER, formulation=Formulation.MILP, num_chunks=1,
        epoch_type=EpochType.SLOWEST_LINK, objective_type=ObjectiveType.LEXICOGRAPHIC,
        solution_method=SolutionMethod.ONE_SHOT, switch_copy=[True, False],
        milp_chunking="cell", schedule_output_file=os.path.join(tempfile.mkdtemp(), "c.json"))
    ui.gurobi.output_flag = 0
    topo = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    with contextlib.redirect_stdout(io.StringIO()):
        info, _ = solve_hierarchical(topo, ui, Collective.ALLGATHER, 1)

    programmable = topo.programmable_switch_indices
    root, routes, epochs = emit(info, topology=topo, programmable=programmable)
    mc = [tb for tb in tbs(root, 4) if tb.get('send') == '0,8']
    assert mc, "GPU 4's switch copy must be one threadblock to GPUs 0 and 8"
    over = {k: v for k, v in uplink_load(root, routes, epochs, topo).items() if v[0] > v[1] + 1e-6}
    assert not over, f"uplinks over capacity: {over}"
    [fid] = flow_ids(mc[0])
    sw17 = next(d for d, raw in routes['switch_id_map'].items() if raw == 17)
    assert len(routes['switches'][sw17][fid]['ports']) == 2, routes['switches'][sw17][fid]

    # the pre-multicast output, for contrast: the same copy as two sends overruns GPU 4's uplink
    root_u, routes_u, epochs_u = emit(info, multicast=False, topology=topo,
                                      programmable=programmable)
    over_u = {k: v for k, v in uplink_load(root_u, routes_u, epochs_u, topo).items()
              if v[0] > v[1] + 1e-6}
    assert any(k[0] == 4 for k in over_u), over_u
    worst = max(over_u.values(), key=lambda v: v[0] / v[1])
    multicast_tbs = sum(1 for g in root.findall('gpu') for tb in g.findall('tb')
                        if re.fullmatch(r'\d+(,\d+)+', tb.get('send')))
    print(f"  [5] hetero switch copy: {multicast_tbs} multicast threadblocks, every uplink within "
          f"capacity, 2 ports on switch 17 (--no-multicast: {len(over_u)} uplink-epochs over, "
          f"worst {worst[0]:g} GB/s on a {worst[1]:g} GB/s link)")


if __name__ == "__main__":
    print("multicast output tests")
    test_one_send_many_receivers()
    test_tree_and_unicast_are_different_routes()
    test_switch_json_format_2()
    test_no_multicast_fallback()
    if "--no-gurobi" not in sys.argv:
        test_hetero_switch_copy()
    print("multicast output tests OK")
