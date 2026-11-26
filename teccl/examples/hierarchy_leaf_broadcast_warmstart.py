"""
A MIP start for a hierarchical MILP allgather whose root (coarse) level is a leaf-switch fabric: every
cell links to the same set of leaf switches, and every leaf links to every cell (the rail-optimized
fabrics, single and dual plane).

The MILP on those levels is large (rail_optimized_256gpu at milp_chunking=2: 1.7M variables, 1.1M
constraints) and Gurobi found no incumbent in 4 h. But a good schedule is known in closed form:

    leaf broadcast   column (s, c) goes s -> leaf L -> every other cell in ONE epoch, the leaf
                     copying it (switch_copy) on its cut-through traversal. A leaf broadcasts one
                     column per epoch -- two would need two chunks on some downlink -- so the
                     columns are dealt round-robin over the leaves, (s * C + c) mod |leaves|, and
                     each leaf runs its share back to back.

Its makespan is ceil(#columns / #leaves) epochs, against the ingress bound
ceil((#cells - 1) * C / in_capacity); on rail_optimized_256gpu at b = 2 both are 16, so the start is
already makespan-optimal and the solve only improves the lower tiers.

This builds the root level's model exactly as the run does (same input, same code, so the variable
names match), fills EVERY variable from that schedule -- buffers, demand indicators, the max_()
auxiliaries and the makespan tier are simulated off the model's own terms, not re-derived -- checks
every bound, integrality, linear row and max_() constraint against it, and writes a Gurobi .mst.
Point InstanceParams.warmstart at the file (read in AllGatherFormulation.encode_problem, ONE_SHOT).

    PYTHONPATH=. python teccl/examples/hierarchy_leaf_broadcast_warmstart.py \
        -i teccl/examples/sample_inputs/rail_optimized_256gpu_hierarchical_allgather_milp_b2_ws.json

Writes warmstarts/<hierarchy_prefix>.mst unless -o is given. Regenerate it whenever the model
changes (input, num_epochs, formulation code): a start whose names no longer match is ignored.
"""
import argparse
import copy
import math
import pathlib
import sys
import time
from collections import defaultdict

import gurobipy as gp
import numpy as np
from gurobipy import GRB

import teccl.scheduler as scheduler
from teccl.cli.solve import make_handle_solve
from teccl.scheduler import TECCLSolver


class _Built(Exception):
    """Raised in place of optimize() once the root level's model is built."""

    def __init__(self, formulation):
        self.formulation = formulation


def build_root_model(input_path: str):
    """The root level's AllGatherFormulation, model built and never optimized."""
    def capture(ui, topo):
        solver = TECCLSolver.__new__(TECCLSolver)
        solver.user_input, solver.topology_obj = ui, topo
        ui = copy.deepcopy(ui)
        # The input names the .mst this script is about to write; encode_problem would try to read
        # it (and fail on a fresh checkout) before optimize. The start does not change variable
        # names, so the model built without it is the one the run builds.
        ui.instance.warmstart = ""
        formulation = solver.get_solver(ui, topo)

        def stop(_model):
            raise _Built(formulation)
        optimize, gp.Model.optimize = gp.Model.optimize, stop
        try:
            formulation.encode_problem()
        finally:
            gp.Model.optimize = optimize

    scheduler.solve_on_topology = capture
    parser = argparse.ArgumentParser()
    handler = make_handle_solve(parser.add_subparsers(dest="command"))
    try:
        handler(parser.parse_args(["solve", "-i", input_path]), "solve")
    except _Built as built:
        built.formulation.model.update()
        return built.formulation
    raise RuntimeError(f"{input_path} reached no Gurobi level; is it a hierarchical MILP input?")


def leaf_broadcast(f):
    """(s, c) -> (leaf, epoch) for every demanded column, or raise if the fabric is not leaf-shaped."""
    cap = f.topology.capacity
    switches = set(f.topology.switch_indices)
    cells = [n for n in f.nodes if n not in switches]
    leaves = sorted(v for v in switches if any(cap[h][v] > 0 for h in cells))
    for v in leaves:
        missing = [h for h in cells if not (cap[h][v] > 0 and cap[v][h] > 0)]
        if missing:
            raise ValueError(f"leaf {v} does not link both ways to cells {missing}: not a leaf fabric")
        if any(cap[v][h] * f.epoch_duration < 1 - 1e-9 for h in cells):
            raise ValueError(f"leaf {v} has a cell link below 1 chunk/epoch; the schedule assumes 1")
    columns = {s: [c for c in f.chunks if any(f.demand[s][d][c] for d in f.nodes)]
               for s in f.sources}
    slot = defaultdict(int)
    plan = {}
    for s in sorted(columns):
        if len(columns[s]) > len(leaves):
            raise ValueError(f"cell {s} has {len(columns[s])} columns but only {len(leaves)} leaves")
        for n, c in enumerate(columns[s]):
            leaf = leaves[(s * len(columns[s]) + n) % len(leaves)]
            plan[(s, c)] = (leaf, slot[leaf])
            slot[leaf] += 1
    makespan = max(slot.values())
    if makespan > f.num_epochs:
        raise ValueError(f"leaf broadcast needs {makespan} epochs but num_epochs is {f.num_epochs}")
    return plan, makespan, len(leaves)


def _value(expr, vals) -> float:
    if expr is None:
        return 0.0
    if isinstance(expr, gp.Var):
        return vals[expr.index]
    return expr.getConstant() + sum(expr.getCoeff(t) * vals[expr.getVar(t).index]
                                    for t in range(expr.size()))


def fill(f, plan):
    """Every variable's value under the leaf-broadcast schedule, indexed by Var.index."""
    model = f.model
    variables = model.getVars()
    vals = np.zeros(len(variables))
    switches = set(f.topology.switch_indices)

    def put(var, v):
        if not isinstance(var, gp.Var):
            raise AssertionError(f"schedule uses a send the model did not create: {var}")
        vals[var.index] = v

    for (s, c), (leaf, k) in plan.items():
        put(f.flow[s][s][leaf][c][k], 1)
        for d in f.nodes:
            if d != s and f.demand[s][d][c]:
                put(f.flow[s][leaf][d][c][k], 1)

    # Buffers and demand indicators read the model's own arrival terms, so the timing they encode
    # (pace classes, alpha) is the model's, not a re-derivation of it.
    E = f.num_epochs
    for s in f.sources:
        for c in f.chunks:
            for i in f.nodes:
                if i in switches:
                    continue
                arrivals = [j for j in f.nodes if f.topology.capacity[j][i] > 0]
                held = 1.0 if i == s else 0.0
                for k in range(E):
                    if k > 0:
                        held += sum(_value(f._arrival_flow(s, j, i, c, k), vals) for j in arrivals)
                    put(f.buffer[s][i][c][k], held)
                if not f.demand[s][i][c]:
                    continue
                for k in range(E):
                    met = _value(f.buffer[s][i][c][k], vals) + sum(
                        _value(f._arrival_flow(s, j, i, c, k + 1), vals) for j in arrivals)
                    put(f.total_demand_sat[s][i][c][k], met)

    demanded = [(s, d, c) for s in f.sources for d in f.nodes for c in f.chunks if f.demand[s][d][c]]
    for k in range(E):
        var = model.getVarByName(f"makespan_met_{k}")
        if var is not None:
            vals[var.index] = float(all(
                vals[f.total_demand_sat[s][d][c][k].index] > 0.5 for s, d, c in demanded))

    for v in variables:
        if v.VarName.startswith("switch_order_"):
            vals[v.index] = v.LB          # no switch -> switch send, so every ordering row is slack

    # max_() auxiliaries: resultant = max(operands), operands are flows already set.
    for gc in model.getGenConstrs():
        if gc.GenConstrType != GRB.GENCONSTR_MAX:
            raise AssertionError(f"unexpected general constraint type {gc.GenConstrType}")
        res, operands, constant = model.getGenConstrMax(gc)
        vals[res.index] = max([vals[o.index] for o in operands] +
                              ([constant] if constant > -GRB.INFINITY else []))
    return vals


def verify(f, vals, tol=1e-6):
    """Every bound, integrality, linear row and max_() constraint, under `vals`. Raises on any miss."""
    model = f.model
    variables = model.getVars()
    lb = np.array(model.getAttr("LB", variables))
    ub = np.array(model.getAttr("UB", variables))
    vtype = model.getAttr("VType", variables)
    bad = [variables[i].VarName for i in np.nonzero((vals < lb - tol) | (vals > ub + tol))[0][:5]]
    if bad:
        raise AssertionError(f"start breaks variable bounds, e.g. {bad}")
    bad = [v.VarName for v, t in zip(variables, vtype)
           if t != GRB.CONTINUOUS and abs(vals[v.index] - round(vals[v.index])) > tol][:5]
    if bad:
        raise AssertionError(f"start is fractional on integer variables, e.g. {bad}")

    constrs = model.getConstrs()
    lhs = model.getA() @ vals
    rhs = np.array(model.getAttr("RHS", constrs))
    sense = np.array(model.getAttr("Sense", constrs))
    viol = np.where(sense == "<", lhs - rhs, np.where(sense == ">", rhs - lhs, np.abs(lhs - rhs)))
    worst = np.argsort(-viol)[:5]
    if viol[worst[0]] > tol:
        raise AssertionError("start violates linear rows: " + ", ".join(
            f"{constrs[i].ConstrName} (by {viol[i]:g})" for i in worst if viol[i] > tol))

    for gc in model.getGenConstrs():
        res, operands, constant = model.getGenConstrMax(gc)
        want = max([vals[o.index] for o in operands] +
                   ([constant] if constant > -GRB.INFINITY else []))
        if abs(vals[res.index] - want) > tol:
            raise AssertionError(f"start violates {gc.GenConstrName}")
    return len(constrs), model.NumGenConstrs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-i", "--input", required=True, help="hierarchical MILP allgather input JSON")
    ap.add_argument("-o", "--output", help="the .mst to write (default warmstarts/<prefix>.mst)")
    args = ap.parse_args()

    t = time.time()
    f = build_root_model(args.input)
    print(f"built root level: {f.model.NumVars} vars, {f.model.NumConstrs} rows, "
          f"{f.num_epochs} epochs of {f.epoch_duration} ({time.time() - t:.0f}s)")
    plan, makespan, num_leaves = leaf_broadcast(f)
    cells = len(f.nodes) - len(f.topology.switch_indices)
    per_cell = max(sum(1 for (s, _c) in plan if s == x) for x in f.sources)
    bound = math.ceil((cells - 1) * per_cell / num_leaves)
    print(f"leaf broadcast: {len(plan)} columns over {num_leaves} leaves -> makespan {makespan} "
          f"epochs (ingress bound {bound})")

    vals = fill(f, plan)
    rows, gens = verify(f, vals)
    print(f"verified: bounds, integrality, {rows} linear rows, {gens} max_() constraints")

    prefix = f.user_input.instance.hierarchy_prefix or pathlib.Path(args.input).stem
    out = pathlib.Path(args.output or f"warmstarts/{prefix}.mst")
    out.parent.mkdir(parents=True, exist_ok=True)
    names = f.model.getAttr("VarName", f.model.getVars())
    with open(out, "w") as fh:
        fh.write(f"# leaf-broadcast MIP start for {args.input}: makespan {makespan} epochs\n")
        for name, v in zip(names, vals):
            fh.write(f"{name} {v:.10g}\n")
    print(f"wrote {out} ({len(names)} values)")


if __name__ == "__main__":
    sys.exit(main())
