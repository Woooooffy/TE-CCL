"""Solution -> path tracing for the chunk-level MILP (AllGatherFormulation.dfs_remove_unnecessary_flows),
with and without switch copy.

The paths are checked by `validate_paths`, an INDEPENDENT re-derivation of what a realizable path
is. It shares no code with the tracer: it re-reads the formulation's timing rules from the link
primitives (alpha/beta/link type/switch_pipeline) rather than calling the trace's predecessor
logic. Per path:

    endpoints     first hop leaves the source, last hop reaches the demanding node, hops chain
    real          every hop is a flow the solver actually set
    causal        a GPU sends only once the chunk is in its buffer; a SWITCH has no buffer, so it
                  sends in exactly the epoch its inbound hop makes the chunk available there
    delivered     the chunk lands by the epoch the solver reports the demand met
and across the union of a chunk's paths, when switch_copy is OFF: each flow into a switch feeds at
most as many distinct flows out of it as it carried copies (a no-copy switch forwards one copy
per copy received).

`validate_sends` checks the tracer's authoritative output, `sends` (teccl/sends.py), as a forest:
every (identity, node) is delivered by exactly one branch, every sender already holds what it
sends, every demand is delivered on time, and without switch copy every send has one branch.

Gurobi-free (fixtures replayed into a model-less formulation):
  [0] the validator itself rejects forged bad paths (sourceless, late switch hop, no-copy fan-out)
  [1] phantom: a chunk that appears from a same-epoch T0 <-> T1 loop is REFUSED, not traced
  [2] switch copy: one switch in-flow feeding two out-flows traces BOTH demands
  [3] no copy: the same flows trace only ONE of them -- the other is a second copy that the
      switch never received
  [3b] no copy, but B sent TWO copies into T0: both demands traced, as two sends with the same
      sender, epoch and first link (two transmissions, not one)
Gurobi (size-limited license is enough; SKIPPED if the model does not fit):
  [4] real solves, switch_copy on and off, every demand traced and every path valid -- including
      the coarse hetero instance whose switch-copy solve used to deliver 2 demands via a phantom
      (fixed by switch_acyclicity_constraints, which the trace correctly refused to paper over)

Run from the repo root:
    PYTHONPATH=. conda run -n teccl python teccl/solvers/milp_trace_test.py
"""
import io
import contextlib
import logging
import os
import sys
import tempfile
from collections import defaultdict

from teccl.input_data import (Collective, EpochType, Formulation, InstanceParams, ObjectiveType,
                              SolutionMethod, TopologyParams, UserInputParams)

logging.disable(logging.WARNING)


# ------------------------------------------------------------------------------------------------
# The independent validator
# ------------------------------------------------------------------------------------------------
def _landing_offset(f, hop, pace):
    """Epochs from a send `hop` = (s, j, i, c, k) to the chunk being usable at i -- re-derived from
    the formulation's rules (node_constraint_helper), not read off the tracer.

    With switch_pipeline, every link touching a switch is part of a cut-through traversal paced
    at its class b (`pace`, which the solver must actually have sent the hop in): into a switch,
    forwardable in epoch k + alpha; into a GPU, in the buffer from k + alpha + 1 + b. Any other hop
    (GPU -> GPU, or every hop of a store-and-forward fabric): in the buffer, or forwardable, from
    k + alpha + 1 + beta."""
    _s, j, i, _c, _k = hop
    alpha, beta = f.get_alpha_num_back(j, i), f.get_beta_num_back(j, i)
    switches = set(f.topology.switch_indices)
    if not (f.user_input.instance.switch_pipeline and (i in switches or j in switches)):
        return alpha + 1 + beta
    assert pace in f.flow_classes(*hop), f"{hop}: traced in class {pace}, not one it was sent in"
    assert pace >= beta, f"{hop}: class {pace} below the link's own beta {beta}"
    return alpha if i in switches else alpha + 1 + pace


def validate_paths(f, raw_flows, demand_met):
    """Every problem with the traced paths, as strings; [] means all valid."""
    problems = []
    raw = set(raw_flows)
    switches = set(f.topology.switch_indices)
    switch_out_of_in = defaultdict(set)          # (s, c, in-flow into a switch) -> out-flows
    for (s, d, c), path in f.per_demand_paths.items():
        paces = f.per_demand_classes.get((s, d, c))
        if paces is None or len(paces) != len(path):   # a forged path (test [0]): its own classes
            paces = [min(f.flow_classes(*h)) for h in path]
        paces = list(reversed(paces))
        hops = list(reversed(path))              # traced destination-first
        tag = f"demand s={s} d={d} c={c}"
        if not hops:
            problems.append(f"{tag}: empty path")
            continue
        if hops[0][1] != s or hops[-1][2] != d:
            problems.append(f"{tag}: runs {hops[0][1]} -> {hops[-1][2]}")
        held = 0                                  # epoch the chunk is usable at the current node
        for n, h in enumerate(hops):
            (_s, i, j, _c, k) = h
            if h not in raw:
                problems.append(f"{tag}: hop {h} is not a solver flow")
            if n and hops[n - 1][2] != i:
                problems.append(f"{tag}: hop {n} leaves {i}, previous reached {hops[n - 1][2]}")
            if i in switches:
                if k != held:
                    problems.append(f"{tag}: switch {i} sends in epoch {k} but the chunk is "
                                    f"there only in epoch {held} (a switch has no buffer)")
                if n and f.user_input.instance.switch_copy is False:
                    switch_out_of_in[(s, c, hops[n - 1])].add(h)
            elif k < held:
                problems.append(f"{tag}: gpu {i} sends in epoch {k}, holds the chunk from {held}")
            held = k + _landing_offset(f, h, paces[n])
        met = demand_met.get((s, d, c))
        if met is not None and held > met + 1:
            problems.append(f"{tag}: lands in epoch {held} but is reported met by {met}")
    for (s, c, in_flow), outs in switch_out_of_in.items():
        if len(outs) > f.flow_copies(*in_flow):
            problems.append(f"no-copy switch {in_flow[2]} forwards chunk ({s}, {c}) received on "
                            f"{in_flow} to {len(outs)} places {sorted(outs)}")
    return problems


def validate_sends(f, demand_met):
    """Every problem with `f.sends` as a forest of transmissions, as strings; [] means valid."""
    from teccl.sends import delivery_index

    problems = []
    seen = defaultdict(int)
    for send in f.sends:
        for b in send.fanout:
            seen[(send.identity, b.receiver)] += 1
        if not f.switch_copy and len(send.fanout) > 1:
            problems.append(f"no-copy send {send} has {len(send.fanout)} branches")
    problems += [f"{key} delivered {n} times" for key, n in seen.items() if n > 1]
    index = delivery_index(f.sends)
    for send in f.sends:
        if send.sender != send.identity[0]:
            held = index.get((send.identity, send.sender))
            if held is None or held[1].ready > send.epoch:
                problems.append(f"{send.sender} sends {send.identity} in epoch {send.epoch} but "
                                f"holds it from {held[1].ready if held else None}")
    for (s, d, c), path in f.per_demand_paths.items():
        delivery = index.get(((s, c), d))
        if delivery is None:
            problems.append(f"demand {(s, d, c)} has a path but no send delivers it")
        elif delivery[1].ready > demand_met[(s, d, c)] + 1:
            problems.append(f"demand {(s, d, c)} lands in {delivery[1].ready}, met by "
                            f"{demand_met[(s, d, c)]}")
    return problems


# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------
def _hetero_coarse():
    """HeteroTaperedCluster's coarse graph at 'cell' chunking: cells A=0, B=1, C=2, switches
    T0=3, T1=4 with a T0 <-> T1 link -- the switch mesh where the phantom lives."""
    from teccl.hierarchy.abstract import abstract, rescale_level
    from teccl.hierarchy.milp_routing import coarsify_demand_preserving
    from teccl.solvers.demand import build_demand
    from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster

    fine = HeteroTaperedCluster(TopologyParams(name="HeteroTaperedCluster", chunk_size=1))
    coarse, mapping = abstract(fine)
    pd = coarsify_demand_preserving(build_demand(Collective.ALLGATHER, fine, 1), mapping, "cell")
    rescale_level(coarse, pd.b)
    coarse.demand_override = pd.tensor
    return coarse


def _ui(switch_copy, objective=ObjectiveType.PAPER, num_epochs=-1):
    ui = UserInputParams()
    ui.instance = InstanceParams(
        collective=Collective.ALLGATHER, formulation=Formulation.MILP, num_chunks=1,
        epoch_type=EpochType.SLOWEST_LINK, objective_type=objective,
        solution_method=SolutionMethod.ONE_SHOT, switch_copy=switch_copy, num_epochs=num_epochs,
        schedule_output_file=os.path.join(tempfile.mkdtemp(), "s.json"))
    ui.gurobi.output_flag = 0
    return ui


def _replay(topology, switch_copy, flows, demand_met, buffers, copies=None):
    """A model-less AllGatherFormulation whose 'solution' is the given flows (each carrying one
    copy unless `copies` says otherwise)."""
    from teccl.solvers.allgather import AllGatherFormulation

    f = AllGatherFormulation(_ui(switch_copy), topology)
    if copies:
        f.flow_copies = lambda *h: copies.get(h, 1)
    f.get_flows_buffer_demand = lambda: (list(flows), dict(demand_met), dict(buffers),
                                         max(demand_met.values()))
    f.find_demand_satisfied_k = lambda: max(demand_met.values())
    f.sources = [n for n in f.nodes if f.demand[n].any()]     # set by initialize_variables
    f.dfs_remove_unnecessary_flows(astar=False)
    return f


# The real raw solution of source cell B's chunk 0 from the coarse hetero switch-copy solve,
# before switch_acyclicity_constraints, re-timed to the pace-class rules (a switch forwards in the
# epoch its traversal starts, so every epoch is one earlier than as recorded): B only ever sends at
# epochs 1 and 2, yet at epoch 0 the chunk is circulating T0 <-> T1 and copied out to A and C,
# both "met" at epoch 0.
PHANTOM_FLOWS = [(1, 3, 2, 0, 0), (1, 3, 4, 0, 0), (1, 4, 0, 0, 0), (1, 4, 3, 0, 0),
                 (1, 1, 3, 0, 1), (1, 1, 3, 0, 2)]
PHANTOM_MET = {(1, 0, 0): 0, (1, 2, 0): 0}
PHANTOM_BUF = {(1, 0, 0): 1, (1, 1, 0): 0, (1, 2, 0): 1}

# B's chunk 1 from the same solve, grounded: B -> T0 in epoch 0, and T0 sends it on TWICE in the
# same epoch (cut-through) -- to C, and to T1 (then A). One switch in-flow, two out-flows: a
# switch copy.
FANOUT_FLOWS = [(1, 1, 3, 1, 0), (1, 3, 2, 1, 0), (1, 3, 4, 1, 0), (1, 4, 0, 1, 0)]
FANOUT_MET = {(1, 0, 1): 0, (1, 2, 1): 0}
FANOUT_BUF = {(1, 1, 1): 0, (1, 0, 1): 1, (1, 2, 1): 1}


def test_validator_rejects_bad_paths():
    """The validator is only evidence if it can fail: forge paths the tracer must never emit."""
    f = _replay(_hetero_coarse(), True, FANOUT_FLOWS, FANOUT_MET, FANOUT_BUF)
    good = dict(f.per_demand_paths)
    # (a) the phantom delivery, as if traced: starts in the T0<->T1 loop, never at the source
    f.per_demand_paths = {(1, 0, 0): [(1, 4, 0, 0, 1), (1, 3, 4, 0, 1)]}
    assert any("runs" in p for p in validate_paths(f, PHANTOM_FLOWS, PHANTOM_MET))
    # (b) a switch sending an epoch after the chunk passed through it (switches hold nothing)
    late = [(1, 3, 2, 1, 2), (1, 1, 3, 1, 0)]
    f.per_demand_paths = {(1, 2, 1): late}
    assert any("no buffer" in p for p in validate_paths(f, FANOUT_FLOWS + late[:1], FANOUT_MET))
    # (c) the copy fan-out, judged as if switch_copy were OFF
    f.per_demand_paths = good
    f.user_input.instance.switch_copy = False
    assert any("no-copy switch" in p for p in validate_paths(f, FANOUT_FLOWS, FANOUT_MET))
    print("  [0] validator rejects a sourceless path, a late switch hop, and a no-copy fan-out")


def test_phantom_refused():
    f = _replay(_hetero_coarse(), True, PHANTOM_FLOWS, PHANTOM_MET, PHANTOM_BUF)
    assert sorted(f.untraced_demands) == [(1, 0, 0), (1, 2, 0)], f.untraced_demands
    assert not f.per_demand_paths, f.per_demand_paths
    print("  [1] phantom (chunk out of a same-epoch T0<->T1 loop) refused: both demands untraced")


def test_switch_copy_fanout():
    f = _replay(_hetero_coarse(), True, FANOUT_FLOWS, FANOUT_MET, FANOUT_BUF)
    assert f.untraced_demands == [], f.untraced_demands
    assert set(f.per_demand_paths) == {(1, 0, 1), (1, 2, 1)}, f.per_demand_paths
    shared = (1, 1, 3, 1, 0)
    assert all(p[-1] == shared for p in f.per_demand_paths.values()), f.per_demand_paths
    problems = validate_paths(f, FANOUT_FLOWS, FANOUT_MET) + validate_sends(f, FANOUT_MET)
    assert not problems, problems
    [send] = f.sends
    assert (send.sender, send.epoch) == (1, 0), send
    assert [(b.receiver, b.via) for b in send.fanout] == [(0, (3, 4)), (2, (3,))], send
    print("  [2] switch copy: one B->T0 send feeds T0->C and T0->T1->A; both demands traced, valid "
          "-- as ONE send with two branches")


def test_no_copy_single_copy():
    f = _replay(_hetero_coarse(), False, FANOUT_FLOWS, FANOUT_MET, FANOUT_BUF)
    assert len(f.untraced_demands) == 1 and len(f.per_demand_paths) == 1, (
        f"a no-copy switch forwards ONE copy per copy received: {f.untraced_demands}, "
        f"{f.per_demand_paths}")
    problems = validate_paths(f, FANOUT_FLOWS, FANOUT_MET) + validate_sends(f, FANOUT_MET)
    assert not problems, problems
    assert len(f.sends) == 1 and len(f.sends[0].fanout) == 1, f.sends
    print("  [3] no copy: the same flows trace exactly one demand; the second copy never existed")


def test_no_copy_two_copies():
    """A paced send is a count of copies: B -> T0 carrying TWO lets a no-copy T0 feed both C and
    T1 (then A). That is two transmissions on one link in one epoch, and must come out as two
    sends -- one gateway and one uplink charge for both would undercount the link."""
    f = _replay(_hetero_coarse(), False, FANOUT_FLOWS, FANOUT_MET, FANOUT_BUF,
                copies={(1, 1, 3, 1, 0): 2})
    assert f.untraced_demands == [], f.untraced_demands
    problems = validate_paths(f, FANOUT_FLOWS, FANOUT_MET) + validate_sends(f, FANOUT_MET)
    assert not problems, problems
    assert len(f.sends) == 2, f.sends
    assert len({(x.sender, x.epoch, x.fanout[0].via[0]) for x in f.sends}) == 1, f.sends
    assert sorted(x.fanout[0].receiver for x in f.sends) == [0, 2], f.sends
    print("  [3b] no copy, two copies into T0: both demands traced, as two sends on one link")


# ------------------------------------------------------------------------------------------------
# Real solves
# ------------------------------------------------------------------------------------------------
def _instances():
    from teccl.hierarchy.abstract import abstract
    from teccl.topologies.bridged_islands_cluster import BridgedIslandsSplitCluster
    from teccl.topologies.dsl_topology import DslTopology

    def mini():
        return DslTopology(TopologyParams(
            name="Mini1GPU1NIC", chunk_size=1,
            topo_file="teccl/topologies/topology-dsl-frontend/examples/mini_1gpu_1nic.topo"))

    def bridged():
        from teccl.hierarchy.abstract import rescale_level
        from teccl.hierarchy.milp_routing import coarsify_demand_preserving
        from teccl.solvers.demand import build_demand
        fine = BridgedIslandsSplitCluster(TopologyParams(name="BridgedIslandsSplit", chunk_size=1))
        coarse, mapping = abstract(fine)
        pd = coarsify_demand_preserving(build_demand(Collective.ALLGATHER, fine, 1), mapping,
                                        "holder")
        coarse.demand_override = pd.tensor
        return coarse

    for copy in (True, False):
        for obj in (ObjectiveType.PAPER, ObjectiveType.TOTAL_DEMAND):
            yield f"coarse hetero / cell / copy={copy} / {obj.name}", _hetero_coarse, copy, obj
        yield f"coarse bridged-split / holder / copy={copy}", bridged, copy, ObjectiveType.PAPER
        yield f"flat mini_1gpu_1nic / copy={copy}", mini, copy, ObjectiveType.PAPER


def test_real_solves():
    from teccl.scheduler import solve_on_topology

    for name, build, copy, obj in _instances():
        topo = build()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                b = solve_on_topology(_ui(copy, obj), topo).best_solver
                b.get_schedule()
        except Exception as e:
            if "size-limited license" in str(e):
                print(f"  [4] SKIPPED {name}: model too large for this Gurobi license")
                continue
            raise
        raw, met, _buf, _k = b.get_flows_buffer_demand()
        demanded = {(s, d, c) for s in b.nodes for d in b.nodes for c in b.chunks
                    if b.demand[s][d][c]}
        assert b.untraced_demands == [], f"{name}: untraced {b.untraced_demands}"
        assert set(b.per_demand_paths) == demanded, (
            f"{name}: traced {len(b.per_demand_paths)} of {len(demanded)} demands")
        problems = validate_paths(b, raw, met) + validate_sends(b, met)
        assert not problems, f"{name}: {problems[:5]}"
        fanouts = sum(1 for outs in _switch_fanouts(b).values() if len(outs) > 1)
        multicast = sum(1 for x in b.sends if len(x.fanout) > 1)
        print(f"  [4] {name}: {len(demanded)}/{len(demanded)} demands traced, all paths valid, "
              f"{fanouts} switch fan-out(s) used, {len(b.sends)} sends ({multicast} multicast), "
              f"{b.find_demand_satisfied_k() + 1} epochs")


def _switch_fanouts(f):
    """(s, c, in-flow into a switch) -> the distinct out-flows the traced paths take from it."""
    switches = set(f.topology.switch_indices)
    out = defaultdict(set)
    for (s, _d, c), path in f.per_demand_paths.items():
        hops = list(reversed(path))
        for prev, h in zip(hops, hops[1:]):
            if h[1] in switches:
                out[(s, c, prev)].add(h)
    return out


if __name__ == "__main__":
    print("MILP solution -> path tracing")
    test_validator_rejects_bad_paths()
    test_phantom_refused()
    test_switch_copy_fanout()
    test_no_copy_single_copy()
    test_no_copy_two_copies()
    if "--no-gurobi" not in sys.argv:
        test_real_solves()
    print("MILP trace tests OK")
