"""Physical check of chunk-level MILP schedules on cut-through switch fabrics.

The MILP (AllGatherFormulation) times a cut-through traversal with pace classes. This module checks
a solved schedule against a model of the hardware that shares NO code with the formulation's
timing rules -- it reads only the traced paths, the epoch length and the raw link capacities and
latencies:

    a STREAM is one copy of a chunk a GPU sends into the fabric and everything it cut-throughs
        to: a traced segment (GPU -> switches -> GPU, or a direct GPU -> GPU hop), plus, under
        switch copy, every same-class branch its send feeds. It may start no earlier than the
        head's epoch, k * epoch, and a GPU holds nothing it has not fully received
        (store-and-forward). A send never feeds more streams than the copies it carried.
    a switch stores nothing, so all links of a stream carry it at the same instant (one rate
        profile per stream, shared by every link and every branch) -- hence never faster than its
        slowest link. Streams sharing a link share its capacity.
    a stream's DEADLINE is what the schedule claims: the end of the epoch a demand is reported
        met in, or the start of the next segment that forwards the chunk from the GPU it lands on.

and then reports, per schedule:
    earliest landing    each demand alone at its bottleneck rate, no contention: a necessary
                        condition; `late` lists demands whose claimed time beats even this, and
                        realized_s is the latest of these landings
    capacity factor     lambda*, the least factor every link capacity must be scaled by for SOME
                        rate allocation to meet every stream's release and deadline (a small LP over the
                        intervals between releases and deadlines -- exact for fluid sharing, since
                        within an interval a constant rate is as good as any profile). <= 1 means
                        the claimed timing is physically achievable
    causality           a segment starting before the chunk could have landed at its sender
    lower bound         max(widest-path latency, every directed cut's volume / capacity), the
                        fastest ANY schedule can finish, even one splitting chunks over paths;
                        and the WHOLE-CHUNK bound, the same with every chunk copy crossing a cut
                        on a single link (_whole_chunk_time) -- the fastest any MILP schedule can
                        finish, and what lb_epochs reports

A schedule is physically sound iff lambda* <= 1 and nothing is late or acausal. It is optimal on the
epoch grid when its epoch count equals ceil(lower bound / epoch). Propagation latencies the solver
rounds to zero (below alpha_threshold) are taken as zero; the LP check asserts that all are.

Usage, from the repo root (the .topo paths are relative); the code under test is whatever teccl
PYTHONPATH selects, so the same file checks the pre-fix tree as well:
    PYTHONPATH=<tree> conda run -n teccl python teccl/solvers/milp_pipeline_check.py <scenario>
        solve, check, print the report
    ... milp_pipeline_check.py solve <scenario> <artifact.json>    solve only, save the schedule
    ... milp_pipeline_check.py check <artifact.json>               report on a saved schedule
With no argument it lists the scenarios.
"""
import contextlib
import io
import json
import logging
import math
import os
import sys
import tempfile
from collections import defaultdict

TOL = 1e-9

DSL = "teccl/topologies/topology-dsl-frontend/examples/"
TEST = "teccl/topologies/dsl_test_files/"


# ------------------------------------------------------------------------------------------------
# The independent physical model
# ------------------------------------------------------------------------------------------------
def _segments(path, switches):
    """A traced path (destination-first hop tuples) as source-first GPU-to-GPU segments."""
    segs, cur = [], []
    for hop in reversed(path):
        cur.append(hop)
        if hop[2] not in switches:
            segs.append(tuple(cur))
            cur = []
    assert not cur, f"path {path} ends at a switch"
    return segs


def realize(topology, epoch, paths, demand_met, alpha_threshold=0.0, classes=None,
            switch_copy=False, copies=None):
    """Replay traced paths on the physical model; see the module docstring.

    Segments are grouped into STREAMS by what the solver claims they share. With switch copy, the
    branches a send feeds in one pace class are one replicated stream (key: first hop, class);
    without it every distinct segment is its own stream (key: segment, class). `classes` holds
    the traced class of every hop (per_demand_classes; None for the pre-fix code, where every send
    is one stream), `copies` how many copies each send carried, which caps the streams sharing it.

    A latency below alpha_threshold epochs counts as 0, the solver's documented rounding of
    negligible propagation (get_alpha_num_back) -- otherwise every 700 ns hop would read as late."""
    cap = topology.capacity
    alpha = [[a if a / epoch > alpha_threshold else 0.0 for a in row] for row in topology.alpha]
    switches = set(topology.switch_indices)
    trees = defaultdict(set)                      # stream -> every hop it feeds
    segs_of = {}
    for key, path in paths.items():
        segs = _segments(path, switches)
        paces = list(reversed(classes[key])) if classes else [None] * len(path)
        keys, n = [], 0
        for seg in segs:
            stream = ((seg[0], paces[n]) if switch_copy else (seg, paces[n]))
            n += len(seg)
            trees[stream].update(seg)
            keys.append(stream)
        segs_of[key] = list(zip(segs, keys))

    problems, landing = [], {}
    trans = {}
    per_send = defaultdict(set)
    for stream, hops in trees.items():
        first = stream[0] if switch_copy else stream[0][0]
        per_send[first].add(stream)
        rate = min(cap[h[1]][h[2]] for h in hops)
        duration = 1.0 / rate
        start = {first[1]: first[4] * epoch}      # when the head reaches each node
        todo = set(hops)
        while todo:
            ready = {h for h in todo if h[1] in start}
            if not ready:
                problems.append(f"stream {stream}: hops {sorted(todo)} are disconnected")
                break
            for h in ready:
                if h[2] in start:
                    problems.append(f"stream {stream}: node {h[2]} entered twice")
                start[h[2]] = start[h[1]] + alpha[h[1]][h[2]]
            todo -= ready
        trans[stream] = dict(head=first, pace=stream[1], rate=rate, duration=duration,
                             links=sorted({h[1:3] for h in hops}))
        for node, t in start.items():
            if node not in switches and node != first[1]:
                landing[(stream, node)] = t + duration
    for first, streams in per_send.items():
        have = (copies or {}).get(first, 1)
        if len(streams) > have:
            problems.append(f"send {first} carries {have} cop(ies) but feeds {len(streams)} streams")

    late, causality, delivered = [], [], {}
    for (s, d, c), segs in segs_of.items():
        held = 0.0                                # the source holds its own chunk from t = 0
        for seg, stream in segs:
            t0 = seg[0][4] * epoch
            if t0 + TOL < held:
                causality.append(f"demand {(s, d, c)}: {seg[0][1]} sends at {t0:.6g}s, chunk "
                                 f"lands there at {held:.6g}s")
            held = landing[(stream, seg[-1][2])]
        delivered[(s, d, c)] = held
        met = demand_met.get((s, d, c))
        if met is not None and held > (met + 1) * epoch * (1 + 1e-6) + TOL:
            late.append(f"demand {(s, d, c)}: lands {held:.6g}s, reported met by epoch {met} "
                        f"(ends {(met + 1) * epoch:.6g}s)")

    # deadline of each stream: the earliest time any consumer of it needs it complete
    deadline = {}
    for (s, d, c), segs in segs_of.items():
        met = demand_met.get((s, d, c))
        for n, (seg, stream) in enumerate(segs):
            due = (segs[n + 1][0][0][4] * epoch if n + 1 < len(segs)
                   else (math.inf if met is None else (met + 1) * epoch))
            deadline[stream] = min(deadline.get(stream, math.inf), due)
    assert all(a == 0.0 for row in alpha for a in row), "capacity LP assumes zero latency"
    lam = capacity_factor(cap, trans, deadline, epoch)
    return dict(
        realized=max(delivered.values()) if delivered else 0.0, capacity_factor=lam,
        late=late, causality=causality, problems=problems,
        transmissions=trans, delivered=delivered)


def capacity_factor(cap, trans, deadline, epoch):
    """lambda*: min lambda such that every stream moves its one chunk inside
    [head epoch * epoch, deadline] with every link's summed rate <= lambda * capacity.
    trans: stream -> {"head": its first hop (s, i, j, c, k), "links": [(i, j), ...]}."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    window = {f: (round(t["head"][4] * epoch, 12), round(deadline[f], 12)) for f, t in trans.items()}
    if any(b <= a for a, b in window.values()):
        return math.inf
    points = sorted({t for w in window.values() for t in w})
    spans = list(zip(points, points[1:]))
    var = {}                                      # (transmission, interval) -> column
    for f, (a, b) in window.items():
        for n, (lo, hi) in enumerate(spans):
            if lo >= a and hi <= b:
                var[(f, n)] = len(var)
    lam = len(var)
    rows = defaultdict(list)                      # (link, interval) -> columns
    for (f, n), col in var.items():
        for link in trans[f]["links"]:
            rows[(link, n)].append(col)
    a_ub = lil_matrix((len(rows), lam + 1))
    for r, ((link, n), cols) in enumerate(rows.items()):
        for col in cols:
            a_ub[r, col] = 1.0
        lo, hi = spans[n]
        a_ub[r, lam] = -cap[link[0]][link[1]] * (hi - lo)
    a_eq = lil_matrix((len(trans), lam + 1))
    for r, f in enumerate(trans):
        for n in range(len(spans)):
            if (f, n) in var:
                a_eq[r, var[(f, n)]] = 1.0
    cost = np.zeros(lam + 1)
    cost[lam] = 1.0
    res = linprog(cost, A_ub=a_ub.tocsr(), b_ub=np.zeros(len(rows)), A_eq=a_eq.tocsr(),
                  b_eq=np.ones(len(trans)), bounds=[(0, None)] * (lam + 1), method="highs")
    assert res.status == 0, res.message
    return float(res.x[lam])


def lower_bound(topology, demanded):
    """Seconds no schedule can beat: widest-path latency and every directed cut (see docstring).
    demanded: the (s, d, c) demands."""
    cap = topology.capacity
    n = len(cap)
    wanted = defaultdict(set)                     # chunk (s, c) -> nodes that demand it
    for s, d, c in demanded:
        wanted[(s, c)].add(d)

    # widest path: the largest bottleneck rate over every s -> d route, any node as transit
    width = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            width[i][j] = cap[i][j] if cap[i][j] > 0 else 0.0
        width[i][i] = math.inf
    for m in range(n):
        for i in range(n):
            for j in range(n):
                width[i][j] = max(width[i][j], min(width[i][m], width[m][j]))
    latency = max((1.0 / width[s][d] for (s, _c), ds in wanted.items() for d in ds), default=0.0)

    cut, cut_set, whole, whole_set = 0.0, None, 0.0, None
    if n <= 18:
        for mask in range(1, (1 << n) - 1):
            inside = [(mask >> v) & 1 for v in range(n)]
            volume = sum(1 for (s, _c), ds in wanted.items()
                         if inside[s] and any(not inside[d] for d in ds))
            if not volume:
                continue
            links = [cap[i][j] for i in range(n) if inside[i]
                     for j in range(n) if not inside[j] and cap[i][j] > 0]
            if volume / sum(links) > cut:
                cut, cut_set = volume / sum(links), [v for v in range(n) if inside[v]]
            t = _whole_chunk_time(links, volume)
            if t > whole:
                whole, whole_set = t, [v for v in range(n) if inside[v]]
    return dict(latency=latency, cut=cut, cut_set=cut_set, bound=max(latency, cut),
                whole_cut=whole, whole_cut_set=whole_set, whole_bound=max(latency, whole))


def _whole_chunk_time(links, volume):
    """Least T in which `volume` whole chunks can cross a cut whose links have these capacities,
    when a chunk cannot be split across links (the MILP's single path per copy): the smallest T
    with sum over links of floor(capacity * T) >= volume. A link carries any number of chunks
    concurrently, but n of them take n / capacity on it in total."""
    candidates = sorted({m / c for c in links for m in range(1, volume + 1)})
    for t in candidates:
        if sum(math.floor(c * t + 1e-9) for c in links) >= volume:
            return t
    return math.inf


# ------------------------------------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------------------------------------
def _sc(topo, collective="alltoall", epoch=0.08, epochs=8, passive=(), copy=False):
    return dict(topo=topo, collective=collective, epoch=epoch, epochs=epochs, passive=passive,
                copy=copy)


SCENARIOS = {
    # EXPOSED: some traversal crosses a link slower than its first link, at an epoch shorter than
    # a chunk on it.
    "chain_a2a_0.08": _sc(TEST + "pace_chain.topo", epochs=4),
    "mini1_1A_a2a_0.08": _sc(DSL + "mini_1gpu_1nic.topo", passive=(1, 3)),
    "mini1_1A_a2a_0.04": _sc(DSL + "mini_1gpu_1nic.topo", epoch=0.04, epochs=12, passive=(1, 3)),
    "mini1_a2a_0.08": _sc(DSL + "mini_1gpu_1nic.topo", epochs=10),
    "mini1_ag_0.08": _sc(DSL + "mini_1gpu_1nic.topo", collective="allgather", epochs=10),
    "mini1_ag_copy_0.08": _sc(DSL + "mini_1gpu_1nic.topo", collective="allgather", epochs=10,
                              copy=True),
    "mini2g1n_2B_a2a_0.08": _sc(DSL + "mini_2gpu_1nic.topo", epochs=12, passive=(1, 4, 6, 7)),
    "mini2g1n_2C_a2a_0.08": _sc(DSL + "mini_2gpu_1nic.topo", epochs=18, passive=(1, 7)),
    "mini2g1n_a2a_0.08": _sc(DSL + "mini_2gpu_1nic.topo", epochs=17),
    # CONTROLS: every traversal's first link is its slowest (or beta is 0 everywhere).
    "sfs_a2a_0.08": _sc(TEST + "pace_slow_fast_slow.topo", epochs=4),
    "mini1_a2a_0.16": _sc(DSL + "mini_1gpu_1nic.topo", epoch=0.16, epochs=6),
    "mini1_ag_copy_0.16": _sc(DSL + "mini_1gpu_1nic.topo", collective="allgather", epoch=0.16,
                              epochs=6, copy=True),
    "mini2g1n_a2a_0.16": _sc(DSL + "mini_2gpu_1nic.topo", epoch=0.16, epochs=12),
    "mini2g2n_a2a_0.08": _sc(DSL + "mini_2gpu_2nic.topo", epochs=12),
    "mini2g2n_a2a_0.04": _sc(DSL + "mini_2gpu_2nic.topo", epoch=0.04, epochs=17),
    "mini2g2n_1perhost_a2a_0.04": _sc(DSL + "mini_2gpu_2nic.topo", epoch=0.04, epochs=12,
                                      passive=(1, 3, 5, 7)),
    "mini2g2n_3A_a2a_0.04": _sc(DSL + "mini_2gpu_2nic.topo", epoch=0.04, epochs=14,
                                passive=(1, 7)),
    "mini2g2n_ag_copy_0.04": _sc(DSL + "mini_2gpu_2nic.topo", collective="allgather", epoch=0.04,
                                 epochs=14, copy=True),
}


def _topology(sc, name):
    from teccl.input_data import TopologyParams
    from teccl.topologies.dsl_topology import DslTopology
    return DslTopology(TopologyParams(name=name, chunk_size=1, topo_file=sc["topo"],
                                      passive_node_indices=tuple(sc["passive"])))


def solve(name):
    """Solve a scenario with whatever teccl is on the path; return the schedule as plain data."""
    from teccl.input_data import (Collective, EpochType, Formulation, InstanceParams,
                                  ObjectiveType, SolutionMethod, TopologyParams, UserInputParams)
    from teccl.scheduler import solve_on_topology
    import teccl.solvers.allgather as code

    sc = SCENARIOS[name]
    topo = _topology(sc, name)
    ui = UserInputParams()
    ui.topology = TopologyParams(name=name, chunk_size=1, topo_file=sc["topo"],
                                 passive_node_indices=tuple(sc["passive"]))
    ui.instance = InstanceParams(
        collective=Collective[sc["collective"].upper()], formulation=Formulation.MILP,
        num_chunks=1, epoch_type=EpochType.USER_INPUT, epoch_duration=sc["epoch"],
        num_epochs=sc["epochs"], objective_type=ObjectiveType.TOTAL_DEMAND,
        solution_method=SolutionMethod.ONE_SHOT, switch_copy=sc["copy"], switch_pipeline=True,
        schedule_output_file=os.path.join(tempfile.mkdtemp(), "s.json"))
    ui.gurobi.output_flag = 0
    ui.gurobi.mip_gap = 0
    # hours; a run stopped by it is still judged -- an incumbent at the lower bound is optimal
    ui.gurobi.time_limit = float(os.environ.get("PIPELINE_TIME_LIMIT_H", ui.gurobi.time_limit))
    with contextlib.redirect_stdout(io.StringIO()):
        solver = solve_on_topology(ui, topo)
    out = dict(scenario=name, code=os.path.dirname(os.path.dirname(os.path.dirname(code.__file__))),
               horizon=sc["epochs"])
    b = getattr(solver, "best_solver", None)
    if b is None:
        return dict(out, status="no schedule")
    b.get_schedule()
    _raw, met, _buf, _k = b.get_flows_buffer_demand()
    hops = {h for path in b.per_demand_paths.values() for h in path}
    return dict(
        out, status=int(b.model.Status), mip_gap=b.model.MIPGap, runtime_s=round(b.model.Runtime, 2),
        epoch=b.epoch_duration, epochs=b.find_demand_satisfied_k() + 1,
        alpha_threshold=b.user_input.instance.alpha_threshold,
        demanded=[[s, d, c] for s in b.nodes for d in b.nodes for c in b.chunks if b.demand[s][d][c]],
        untraced=[list(x) for x in b.untraced_demands],
        paths=[[list(k), [list(h) for h in path]] for k, path in b.per_demand_paths.items()],
        met=[[list(k), v] for k, v in met.items()],
        switch_copy=b.switch_copy,
        classes=([[list(k), v] for k, v in b.per_demand_classes.items()]
                 if hasattr(b, "per_demand_classes") else None),
        copies=([[list(h), b.flow_copies(*h)] for h in sorted(hops)]
                if hasattr(b, "flow_copies") else None))


def check(art):
    """The physical report for one solved schedule (output of solve)."""
    if "epochs" not in art:
        return art
    name, eps = art["scenario"], art["epoch"]
    topo = _topology(SCENARIOS[name], name)
    switches = set(topo.switch_indices)
    paths = {tuple(k): [tuple(h) for h in p] for k, p in art["paths"]}
    met = {tuple(k): v for k, v in art["met"]}
    classes = {tuple(k): v for k, v in art["classes"]} if art.get("classes") else None
    copies = {tuple(h): n for h, n in art["copies"]} if art.get("copies") else None
    real = realize(topo, eps, paths, met, art["alpha_threshold"], classes,
                   art.get("switch_copy", SCENARIOS[name]["copy"]), copies)
    lb = lower_bound(topo, [tuple(x) for x in art["demanded"]])
    mismatch, lines = [], []
    for t in sorted(real["transmissions"].values(), key=lambda t: (t["head"][4], t["head"])):
        first, pace = t["head"], t["pace"]
        need = max(0, math.ceil(1.0 / (t["rate"] * eps) - 1e-9) - 1)
        if pace is not None and pace < need:
            mismatch.append(f"{first}: solver class {pace}, physics needs {need}")
        to = "/".join(str(j) for _i, j in t["links"] if j not in switches)
        lines.append(f"{first[1]}->{to} chunk({first[0]},{first[3]}) head ep {first[4]}: "
                     f"{t['duration'] / eps:.3g} ep at {t['rate'] * eps:.3g} chunk/ep"
                     + (f", class {pace}" if pace is not None else "")
                     + (f", send carries {copies[first]} copies" if copies and copies.get(first, 1) > 1 else ""))
    return dict(
        scenario=name, code=os.path.basename(art["code"]), status=art["status"],
        mip_gap=art["mip_gap"], runtime_s=art["runtime_s"], horizon=art["horizon"], epoch=eps,
        epochs=art["epochs"], reported_s=round(art["epochs"] * eps, 9),
        realized_s=round(real["realized"], 9), capacity_factor=round(real["capacity_factor"], 6),
        lower_bound_s=round(lb["bound"], 9), lb_fluid_epochs=math.ceil(lb["bound"] / eps - 1e-9),
        lb_latency_s=round(lb["latency"], 9), lb_cut_s=round(lb["cut"], 9),
        lb_cut_set=lb["cut_set"], whole_bound_s=round(lb["whole_bound"], 9),
        lb_epochs=math.ceil(lb["whole_bound"] / eps - 1e-9), whole_cut_set=lb["whole_cut_set"], demands=len(art["demanded"]), traced=len(paths),
        untraced=len(art["untraced"]), late=real["late"], causality=real["causality"],
        problems=real["problems"], class_mismatch=mismatch, transmissions=lines)


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    if len(sys.argv) < 2:
        print("\n".join(SCENARIOS))
    elif sys.argv[1] == "solve":            # solve <scenario> <artifact.json>
        with open(sys.argv[3], "w") as f:
            json.dump(solve(sys.argv[2]), f)
    elif sys.argv[1] == "check":            # check <artifact.json>
        print(json.dumps(check(json.load(open(sys.argv[2]))), indent=1, default=str))
    else:                                   # <scenario>: solve and check
        print(json.dumps(check(json.loads(json.dumps(solve(sys.argv[1])))), indent=1, default=str))
