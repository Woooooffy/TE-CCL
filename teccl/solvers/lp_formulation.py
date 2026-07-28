import copy
import logging
import time
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from teccl.gurobi_env import get_gurobi_env
from teccl.input_data import *
from teccl.solvers.base_formulation import BaseFormulation
from teccl.topologies.topology import Topology


class LPFormulation(BaseFormulation):
    """
    Collective-agnostic continuous-flow LP solver.

    This formulation satisfies an *arbitrary* demand matrix (`self.demand`, produced by the
    demand generator in BaseFormulation) — it never branches on the collective type. A new
    collective is added simply by supplying its demand generator; the solver body here is
    reused unchanged. AllToAll is the canonical user, and AllGather can also be routed here
    (only meaningful with switch copy disabled: the LP aggregates flow per source and bounds
    a node's outgoing flow by its incoming+buffered flow, so it has no replication/copy).
    """

    def __init__(self, user_input: UserInputParams, topology: Topology) -> None:
        super().__init__(user_input, topology)
        self.solver_name = f"{self.user_input.instance.collective.name}_LP"

    def initialize_variables(self) -> None:
        """
          flow(s, i, j, k) : the fraction of data from source s that goes over link (i, j) in epoch k
          buffer(s, i, k) : the fraction of data from source s that is in the buffer at node i at epoch k
          total_demand_sat(s, i, k) : the amount of demand satisfied at node i till epoch k
          consumed_at_k(s, i, k) : fraction of demand coming from source s that is satisfied at node i in epoch k
        """

        logging.debug("starting to initialize the variables for the LP")
        start_time = time.time()

        # compute the total demand across all nodes to set upper bound on flow variable.
        self.all_demand = 0
        for s, d in product(self.nodes, self.nodes):
            for c in range(len(self.demand[s][d])):
                self.all_demand += self.demand[s][d][c]

        # calculate the demand at each node to set better variable limits.
        # total_demand_at_s : total volume of data that node s needs to send
        # demand_at_i : total volume of data node s needs to send to node i
        # (computed before creating variables so we can restrict them to the
        # demand-bearing sources below.)
        self.total_demand_at_s = defaultdict(float)
        self.demand_at_i = defaultdict(float)
        for s in self.nodes:
            node_demand = 0
            for d in self.nodes:
                for c in self.chunks:
                    node_demand += self.demand[s][d][c]
                    self.demand_at_i[(s, d)] += self.demand[s][d][c]
            self.total_demand_at_s[s] = node_demand

        # Only demand-bearing sources originate any flow. Every flow / buffer /
        # consume / demand-sat variable and every constraint is indexed by a
        # source s (the ORIGIN of the data); for a zero-demand source (switches,
        # passive nodes, and any GPU with no demand) flow[s][.][.] is provably 0 in
        # every feasible solution -- nothing is injected at s and nothing consumes
        # s's data -- so building those variables/constraints only inflates the
        # model without changing the feasible set or optimum. Restricting the
        # *source* dimension to self.sources shrinks the variable / row / nonzero
        # counts by num_nodes / #demand_sources -- ~13x for the 300-node /
        # 23-active-GPU MoE topologies, which is what pushed the feasible-search LP
        # (47.7M cols / 140M nonzeros) over the memory limit and OOM-killed it in
        # presolve. Relaying is unaffected: a zero-demand node still forwards and
        # buffers an active source's data as an intermediate i/j/n hop in
        # flow[active_source][i][j] (the node/link dimension stays all num_nodes),
        # so ONLY the source dimension is restricted here.
        self.sources = [s for s in self.nodes if self.total_demand_at_s[s] > 0]

        # initialize flow variables.
        # Sparse container: the dense num_nodes^3 x num_epochs list this used to be
        # (2.16B cells at 300 nodes / 80 epochs -> ~17 GB for the numpy array alone,
        # far more once .tolist()'d -> OOM-killed before Gurobi ever starts) is almost
        # entirely zeros; only real links ever carry a variable. A nested defaultdict
        # makes a missing (s, i, j, k) read as 0.0 -- exactly the old np.zeros default,
        # so every LinExpr.add() downstream is unchanged -- while storing only the
        # real-link entries that are actually created below.
        self.flow = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))

        for i, j in product(self.nodes, self.nodes):
            # Skip non-links. Non-links have capacity 0 in these topologies (only
            # explicit edges are set positive), and every downstream read of flow
            # guards on capacity > 0, so `<= 0` here keeps variable creation
            # consistent with use and keeps the container sparse (real links only).
            if self.topology.capacity[i][j] <= 0:
                continue
            for s in self.sources:
                for k in self.epochs:
                    self.flow[s][i][j][k] = self.model.addVar(
                        0, self.all_demand, vtype=GRB.CONTINUOUS, name='f_%d_%d_%d_%d' % (s, i, j, k))
        logging.debug(
            f"Time for F (flows) initialization: {time.time() - start_time}")

        start_time = time.time()

        # Dense lists indexed [source][node][epoch]; only the self.sources rows are
        # populated with variables, the rest stay 0.0 (matching the old np.zeros
        # default) and are never read, since every constraint loop below is also
        # restricted to self.sources.
        self.buffer = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_epochs)).tolist()
        self.consumed_at_k = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_epochs)).tolist()
        self.total_demand_sat = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_epochs)).tolist()

        # initialize remaining variables (source dimension restricted to self.sources)
        for s, i, k in product(self.sources, self.nodes, self.epochs):
            self.buffer[s][i][k] = self.model.addVar(
                0, self.total_demand_at_s[s], vtype=GRB.CONTINUOUS, name='B_%d_%d_%d' % (s, i, k))
            self.consumed_at_k[s][i][k] = self.model.addVar(
                0, self.demand_at_i[(s, i)], vtype=GRB.CONTINUOUS, name='T_%d_%d_%d' % (s, i, k))
            self.total_demand_sat[s][i][k] = self.model.addVar(
                0, self.demand_at_i[(s, i)], vtype=GRB.CONTINUOUS, name='t_%d_%d_%d' % (s, i, k))

    def destination_constraints(self) -> None:
        """
        add constraints to the model to account for how much of the demand is met at the end
        of each epoch.
        """
        # s ranges only over demand-bearing sources (see initialize_variables):
        # a zero-demand source's demand is 0 everywhere, so these constraints are
        # vacuous for it and its total_demand_sat/consumed vars are not created.
        for s, d, k in product(self.sources, self.nodes, self.epochs):
            consume_constr = gp.LinExpr(0.0)
            # the demand met so far is the sum of all the chunk the node has consumed so far.
            if k > 0:
                consume_constr.add(self.total_demand_sat[s][d][k - 1])
            consume_constr.add(self.consumed_at_k[s][d][k])

            self.model.addConstr(
                consume_constr == self.total_demand_sat[s][d][k], name='total_demand_sat_%d_%d_%d' % (s, d, k))

            if k <  self.num_epochs - 1:
                # do not consume more than the demand.
                self.model.addConstr(consume_constr <= self.demand_at_i[(
                    s, d)], name='demand_constraint_%d_%d_%d' % (s, d, k))
            else:
                self.model.addConstr(consume_constr == self.demand_at_i[(
                    s, d)], name='full_demand_satisfiablility_%d_%d_%d' % (s, d, k))

    def switch_ingress_cut_through(self) -> bool:
        """
        Whether a switch's ingress hops are modeled as cut-through, i.e. the switch relays
        without paying the store-and-forward "+1" epoch. In this LP a pipelined switch
        fabric makes both ingress hop types cut-through: switch->switch and gpu->switch
        (the final switch->gpu egress leg is already propagation-only by construction).
        Gated by the single switch_pipeline flag.
        """
        return self.user_input.instance.switch_pipeline

    def node_constraint_helper(self, s: int, n: int, k: int) -> None:
        """
            Adds constraints on the buffers and adds flow conservation constraints.
            s : source node
            n : current node for which we are adding constraints
            k : epoch for which we are adding the constraints
        """

        if n in self.topology.switch_indices and self.switch_ingress_cut_through():
            # Cut-through switch: it relays a chunk in the SAME epoch the chunk becomes
            # available (sent to it at k - alpha), with no store-and-forward "+1" gap.
            # Crucially this egress is allowed in epoch 0 as well, so a source GPU feeding
            # the fabric at epoch 0 is relayed through the switch(es) within epoch 0 -- the
            # store-and-forward path below would instead force switch egress in epoch 0 to
            # zero, wasting the first epoch for every fabric path. Serialization is charged
            # only by the per-epoch capacity limit, never as a per-hop latency. This fully
            # handles the switch (it stores nothing and never consumes), so we return.
            self.model.addConstr(
                self.buffer[s][n][k] == 0, name=f'switch_constraint-source_{s}-node_{n}-epoch_{k}')
            switch_fc = gp.LinExpr(0.0)
            for j in self.nodes:
                if self.topology.capacity[j][n] > 0:
                    alpha_num_back = self.get_alpha_num_back(j, n)
                    if k - alpha_num_back >= 0:
                        switch_fc.add(self.flow[s][j][n][k - alpha_num_back])
                if self.topology.capacity[n][j] > 0:
                    switch_fc.add(self.flow[s][n][j][k], -1)
            # egress in epoch k <= ingress that becomes available in epoch k.
            self.model.addConstr(
                switch_fc >= 0, name=f"cutthrough-switch-epoch_{k}-node_{n}-source_{s}")
            return

        # we need to account for the case where the number of epochs is 1 separately.
        if self.num_epochs > 1:
            if k == 0 and n != s:
                # nothing can go out of the node since there is nothing in the buffer of non-source nodes.
                buffer_constr = gp.LinExpr(0.0)
                buffer_constr.add(self.buffer[s][n][k])
                for j in self.nodes:
                    if self.topology.capacity[n][j] > 0:
                        buffer_constr.add(self.flow[s][n][j][k])
                self.model.addConstr(
                    buffer_constr == 0, name=f"Initial_flow_{s}-node_{n}-epoch_{k}")
            elif k == 0 and n == s:
                # the source node's buffer contains all the chunks s wants to send.
                # it can have flow at epoch 0 which is bounded by the contents of the buffer.
                buffer_constr = gp.LinExpr(0.0)
                buffer_constr.add(self.buffer[s][n][k])
                for j in self.nodes:
                    if self.topology.capacity[n][j] > 0:
                        buffer_constr.add(self.flow[s][n][j][k])
                self.model.addConstr(
                    buffer_constr == self.total_demand_at_s[n], name=f'EX_buffer-source_{s}-node_{n}')
        else:
            # if the Kmax is 1 then the middle constraints will interfere if we dont do this.
            if n != s:
                buffer_constr = gp.LinExpr(0.0)
                buffer_constr.add(self.buffer[s][n][k])
                self.model.addConstr(
                    buffer_constr == 0, f'Initial_flow_{s}-node_{n}-epoch_{k}')
            else:
                buffer_constr = gp.LinExpr(0.0)
                buffer_constr.add(self.buffer[s][n][k])
                self.model.addConstr(
                    buffer_constr == self.total_demand_at_s[n], name=f'Inital_flow_{s}-node_{n}')

        if n in self.topology.switch_indices:
            # if the node is a switch then we need to add a constraint that the buffer is 0.
            switch_buffer = gp.LinExpr(0.0)
            switch_buffer.add(self.buffer[s][n][k])
            self.model.addConstr(
                switch_buffer == 0, name=f'switch_constraint-source_{s}-node_{n}-epoch_{k}')

        # next we implement flow conservation constraints.
        if k + 1 < self.num_epochs and (n not in self.topology.switch_indices):
            flow_conservation = gp.LinExpr(0.0)
            flow_conservation.add(self.buffer[s][n][k])
            for j in self.nodes:
                if self.topology.capacity[j][n] > 0:
                    alpha_num_back = self.get_alpha_num_back(j, n)
                    if k - alpha_num_back >= 0:
                        flow_conservation.add(
                            self.flow[s][j][n][k - alpha_num_back])
                if self.topology.capacity[n][j] > 0:
                    flow_conservation.add(self.flow[s][n][j][k + 1], -1)
            flow_conservation.add(self.buffer[s][n][k + 1], -1)
            flow_conservation.add(self.consumed_at_k[s][n][k], -1)
            #  Buffer at the beginning of epoch k + flows that reach by end of epoch k >=
            #       buffer at the beginning of epoch (k + 1) + flows going out of the node in epoch (k + 1) + chunks portion consumed by the end of epoch k
            # TODO: this allows for flows getting dropped, we might want to just make it equality.
            self.model.addConstr(flow_conservation >= 0,
                                 name=f"midFC-epoch_{k}-node_{n}-source_{s}")

        elif k + 1 < self.num_epochs and n in self.topology.switch_indices:
            # Store-and-forward switch (cut-through switches are handled/returned above):
            # the switch must fully receive (ingress available by epoch k) before it can
            # forward (egress at k+1) -- the "+1" store-and-forward epoch.
            flow_conservation = gp.LinExpr(0.0)
            for j in self.nodes:
                if self.topology.capacity[j][n] > 0:
                    alpha_num_back = self.get_alpha_num_back(j, n)
                    if k - alpha_num_back >= 0:
                        flow_conservation.add(
                            self.flow[s][j][n][k - alpha_num_back])
                if self.topology.capacity[n][j] > 0:
                    flow_conservation.add(self.flow[s][n][j][k + 1], -1)
            # TODO: once again flow drop can happen here.
            self.model.addConstr(flow_conservation >= 0, name=f"midFC-switch-epoch_{k}-node_{n}-source_{s}")

        # last epoch flow conservation constraints.
        if k + 1 == self.num_epochs:
            if n not in self.topology.switch_indices and s != n:
                incoming = gp.LinExpr(0.0)
                for j in self.nodes:
                    if self.topology.capacity[j][n] > 0:
                        alpha_num_back = self.get_alpha_num_back(j, n)
                        if k - alpha_num_back >= 0:
                            incoming.add(self.flow[s][j]
                                         [n][k - alpha_num_back])
                incoming.add(self.consumed_at_k[s][n][k], -1)
                # All the flows that reach the node by the end of the last epoch should be consumed by the node as they can't be forwarded further.
                self.model.addConstr(
                    incoming >= 0, name=f'I_epoch_{k}-node_{n}-source_{s}')
            elif s == n or k == 0:
                flow_conservation = gp.LinExpr(0.0)
                flow_conservation.add(self.buffer[s][n][k])
                for j in self.nodes:
                    if self.topology.capacity[n][j] > 0:
                        flow_conservation.add(self.flow[s][n][j][k], -1)
                self.model.addConstr(flow_conservation >= 0,
                                     name=f"FC_epoch_{k}-node_{n}-source_{s}")

    def node_constraints(self) -> None:
        # n ranges over ALL nodes (any node can relay/buffer an active source's
        # data), while s ranges only over demand-bearing sources -- flow[s][.][.]
        # is provably 0 for a zero-demand source, so its conservation constraints
        # are vacuous. This is the source-only restriction: relaying through
        # zero-demand nodes is preserved via the full n (and i/j) range.
        for n, s, k in product(self.nodes, self.sources, self.epochs):
            self.node_constraint_helper(s, n, k)

    def capacity_constraints(self) -> None:
        """
        encodes capacity constraints.

        The RHS (capacity * epoch_duration) is the ONLY coefficient in the whole
        model that changes when just epoch_duration changes. We keep a handle to
        every capacity constraint in self._cap_constrs so update_epoch_duration()
        can rescale those RHS values in place (and warm-start) instead of
        rebuilding the model -- see build_model()/update_epoch_duration().
        """

        # Uncomment this if you want to impose buffer limits.
        # if self.buffer_limit_ >= 0:
        #     for i in self.nodes_:
        #         for k in self.epochs_:
        #             buffer_constr = gp.LinExpr(0.0)
        #             for s in self.nodes_:
        #                 if s == i:
        #                     continue
        #                 buffer_constr.add(self.B_[s][i][k])
        #             self.model_.addConstr(
        #                 buffer_constr <= self.buffer_limit_, name=f"buffer_limit_constr_{i}_{k}")

        for i, j, k in product(self.nodes, self.nodes, self.epochs):
            if self.topology.capacity[i][j] <= 0:
                continue
            cap_constr = gp.LinExpr(0.0)
            # Only demand-bearing sources carry flow; a zero-demand source's
            # flow[s][i][j][k] is provably 0, so it contributes nothing to the
            # link's load. Summing over self.sources keeps the capacity constraint
            # identical while cutting its nonzero count from num_nodes to #sources.
            for s in self.sources:
                cap_constr.add(self.flow[s][i][j][k])
            self._cap_constrs[(i, j, k)] = self.model.addConstr(
                cap_constr <= (self.topology.capacity[i][j] * self.epoch_duration), name=f"cap_constr_link_{i}-{j}-{k}")

    def objective_formulation(self, objective_type: ObjectiveType = ObjectiveType.PAPER) -> gp.LinExpr:
        """
        returns the objective for the optimization.
        I've only implemented the paper and the total demand objective here.
        """
        objective = gp.LinExpr(0.0)
        multiplier = pow(10, 2)

        if objective_type == ObjectiveType.TOTAL_DEMAND:
            for k in self.epochs:
                tmp = gp.LinExpr(0.0)
                for s, d in product(self.nodes, self.nodes):
                    tmp.add(self.total_demand_sat[s][d][k])
 
                self.aux_var.append(self.model.addVar(-self.all_demand, 1,
                                    vtype=GRB.CONTINUOUS, name='aux_var_boj_%d'% k))
                tmp.add(self.all_demand - 1, -1)
                self.model.addConstr(self.aux_var[len(
                    self.aux_var) - 1] == tmp, "obj1_s_%d" % k)
                self.aux_var.append(self.model.addVar(
                    0, 1, vtype=GRB.CONTINUOUS, name='aux_var_obj2_%d' % k))
                self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(
                    [self.aux_var[len(self.aux_var) - 2], 0]), name='obj2_k_%d' % k)

                objective.add(
                    self.aux_var[len(self.aux_var) - 1], (-multiplier))
                for i in self.nodes:
                    if self.topology.capacity[i][d] <= 0:
                        continue
                    objective.add(
                        self.flow[s][i][d][k], 10 * (pow(10, -1) / (self.num_epochs + 1)) * 2)
        # Experimental objective that has been removed.
        # elif objective_type == ObjectiveType.EXPERIMENTAL:
        #     objective = gp.LinExpr(0.0)
        #     for s, d, k in product(self.nodes, self.nodes, self.epochs):
        #         objective.add(
        #             self.total_demand_sat[s][d][k], -1 * multiplier * pow(0.1, (k + 1)))

        else:
            assert objective_type == ObjectiveType.PAPER, "wrong objective type"
            objective = gp.LinExpr(0.0)
            for s, d, k in product(self.nodes, self.nodes, self.epochs):
                objective.add(
                    self.total_demand_sat[s][d][k], -1 * multiplier * pow(10, -1) / (k + 1))
                for i in self.nodes:
                    if self.topology.capacity[i][d] <= 0:
                        continue
                    objective.add(
                        self.flow[s][i][d][k], 10 * (pow(10, -1) / (self.num_epochs + 1)) * 2)
        return objective

    def _compute_alpha_signature(self) -> Tuple[int, ...]:
        """
        The per-link alpha_num_back values are the ONLY thing that changes the
        *structure* (not just a coefficient) of the model when epoch_duration
        changes: they shift which epoch index k - alpha_num_back each
        flow-conservation term refers to. Everything else -- the variables, the
        objective, the destination constraints, and the capacity-constraint
        incidence (only its RHS scales) -- is invariant in epoch_duration.

        This returns a hashable signature of those values (read at the current
        self.epoch_duration) so update_epoch_duration() can tell whether the node
        (flow-conservation) constraints can be reused as-is (signature unchanged)
        or must be rebuilt (signature changed). For typical topologies alpha is
        tiny relative to epoch_duration, so every alpha_num_back is 0 across the
        whole feasible search and this signature never changes -> the node
        constraints are reused every iteration.
        """
        return tuple(
            self.get_alpha_num_back(i, j)
            for i in self.nodes for j in self.nodes
            if self.topology.capacity[i][j] > 0
        )

    def build_model(self) -> None:
        """
        Build the full model from scratch for the current self.epoch_duration.

        Records the capacity-constraint handles (self._cap_constrs) and the alpha
        signature (self._alpha_signature) so a subsequent epoch_duration change can
        be applied incrementally via update_epoch_duration() -- reusing every
        variable and constraint and warm-starting Gurobi -- instead of rebuilding.
        """
        setup_start = time.time()
        self.model = gp.Model('LP', env=get_gurobi_env())
        # These accumulate model objects, so reset them on every (re)build.
        self.aux_var = []
        self._cap_constrs = {}
        self.initialize_variables()
        self.destination_constraints()
        self.node_constraints()
        self.capacity_constraints()
        self.model.setObjective(self.objective_formulation(self.user_input.instance.objective_type))
        self._alpha_signature = self._compute_alpha_signature()

        self._log_file = f'Logs/{self.solver_name}_{self.user_input.topology.name}_{self.num_nodes}-nodes_' \
            f'{self.num_chunks}-chunks_{self.num_epochs}-epochs_{self.epoch_duration}-epoch_duration_{self.user_input.instance.objective_type}'

        if self.user_input.gurobi.output_flag == 1 or self.user_input.instance.debug:
            if self.user_input.gurobi.log_file:
                self._log_file += self.user_input.gurobi.log_file
            self.model.setParam("LogFile", self._log_file + ".log")
            self.model.Params.LogToConsole = 0
            # if self.user_input.instance.debug:
            #     self.model.write(self._log_file + '.lp')

        self.set_gurobi_params()
        self._model_built = True
        logging.debug(f'Total time for build {time.time() - setup_start}')

    def update_epoch_duration(self, epoch_duration: float) -> None:
        """
        Refresh the model for a new epoch_duration WITHOUT rebuilding it: reuse all
        variables/objective/constraints and warm-start Gurobi from the previous
        solve's basis. Only two things depend on epoch_duration:
          - capacity-constraint RHS (capacity * epoch_duration): rescaled in place.
          - flow-conservation structure via alpha_num_back: unchanged for typical
            topologies (alpha << epoch_duration); if the alpha signature does change
            we fall back to a full rebuild (correct, and rare).

        The feasible search calls this ~10x with num_epochs held fixed, so this
        turns ~10 full builds + cold solves into 1 build + ~10 warm re-solves.
        """
        self.epoch_duration = epoch_duration
        if not getattr(self, "_model_built", False):
            # First call: nothing to reuse yet.
            self.build_model()
            return
        if self._compute_alpha_signature() != self._alpha_signature:
            # Structural change in the flow-conservation constraints -> rebuild.
            self.build_model()
            return
        # Fast path: only the capacity RHS moved. Rescale in place and warm-start.
        for (i, j, k), constr in self._cap_constrs.items():
            constr.RHS = self.topology.capacity[i][j] * self.epoch_duration
        self.model.update()

    def solve_model(self) -> int:
        """
        Optimize the already-built model (see build_model / update_epoch_duration)
        and return its Gurobi status. Separated from build_model so the feasible
        search can rebuild once and re-solve many times.
        """
        logging.debug(f'Epoch duration {self.epoch_duration}')
        logging.debug(f'Starting model optimization {self._log_file}')

        solve_start = time.time()
        self.model.optimize()
        logging.debug(
            f'Finished model optimization {self._log_file} in {time.time() - solve_start}')

        if self.model.Status != GRB.OPTIMAL:
            logging.warning(
                f"Not_Optimal_{self.solver_name}_Status-{self.model.Status}_{self.user_input.topology.name}_"
                f"{self.num_nodes}-nodes_{self.num_chunks}-chunks_{self.num_epochs}-epochs_{self.epoch_duration}-epoch_duration"
            )
            if self.user_input.instance.debug and self.model.SolCount > 0:
                logging.debug(
                    f"Epoch at the end of which all demands are satisfied: {self.find_demand_satisfied_k() + 1}")
            # compute an Irreducible Inconsistent Subsystem
            # https://www.gurobi.com/documentation/10.0/refman/py_model_computeiis.html
            # else:
            # self.model.computeIIS()
            # self.model.write(self._log_file + '_unsat.ilp')
            return self.model.Status

        if self.user_input.instance.debug:
            logging.debug(
                f"Epoch at the end of which all demands are satisfied: {self.find_demand_satisfied_k() + 1}")
        return self.model.Status

    def encode_problem(self, use_one_less_epoch=False) -> int:
        """Build the model from scratch and solve it (one-shot / iterative paths)."""
        self.build_model()
        return self.solve_model()

    def get_flows_and_consumes(self) -> Tuple[List[Tuple[int, int, int, float, int]], Dict]:
        """
        this function returns the list of all the flows that the optimization assigned positive value to
        and the time-series for each destination of when it consumed part of its demand.
        """
        consumed = {}
        full_flow_list = []
        for v in self.model.getVars():
            if 'f_' in v.varName and v.x != 0.0:
                components = v.varName.split('_')
                _, s, i, j, k = components
                full_flow_list.append((int(s), int(i), int(j), round(v.x,6), int(k)))
            if 'T_' in v.varName and v.x > 0:
                components = v.varName.split('_')
                _, s, d, k = components
                if int(d) not in consumed:
                    consumed[int(d)] = []
                consumed[int(d)].append((int(s), int(k), round(v.x,6)))
        return (full_flow_list, consumed)

    def account_for_consume(self, consume: float, source: int, destination: int, i: int, j: int, k: int, paths: Dict) -> Dict:
        """
        converts the aggregated flow into per-flow form.
        """
        path = {}
        for c in self.chunks:
            if c not in path:
                path[c] = []
            if self.demand_copy[source][destination][c] > 0:
                if consume > self.demand_copy[source][destination][c]:
                    self.per_chunk_flows[source][i][j][c][k] = self.demand_copy[source][destination][c]
                    consume -= self.demand_copy[source][destination][c]
                    if c not in paths.keys():
                        path[c] = (source, i, j, c,
                                   self.demand_copy[source][destination][c], k)
                    else:
                        path[c] += (source, i, j, c,
                                    self.demand_copy[source][destination][c], k)
                    if source == i:
                        self.demand_copy[source][destination][c] = 0
                else:
                    assert consume > 0
                    self.per_chunk_flows[source][i][j][c][k] = consume
                    if c not in paths.keys():
                        path[c] = (source, i, j, c, consume, k)
                    else:
                        path[c] += (source, i, j, c, consume, k)
                    if source == i:
                        self.demand_copy[source][destination][c] -= consume
                    consume = 0
                    break
            if consume == 0:
                break
        for c in path.keys():
            if c not in paths:
                paths[c] = []
            paths[c] += [path[c]]
        consume = round(consume, 5)
        if consume != 0:
            print(f"source ={source}, destination={destination}, consume={consume}")
        if consume <= 1e-6:
            consume = 0
        assert consume == 0
        return paths

    def check_if_viable(self, hop: int, dest: int, step : int, instance: Tuple[int, int, int, int, float, int]) -> bool:
        """
        checks if the timing of the instnace is correct with respect to where we are at.
        """
        #enable the commented code if we make it such that the second link from the switch is instantaneous.
        if hop == dest: # or instance[1] in self.topology.switch_indicies:
            return (instance[-1] <= step - self.get_alpha_num_back(instance[1], hop))
        elif hop in self.topology.switch_indices:
            # Mirror the solver's switch flow-conservation: a cut-through ingress hop has
            # no store-and-forward "+1" epoch, so the switch relayed in the same epoch it
            # received (only propagation). Applies to switch->switch and GPU->switch ingress
            # per their respective flags (see switch_ingress_cut_through).
            if self.switch_ingress_cut_through():
                return (instance[-1] == step - self.get_alpha_num_back(instance[1], hop))
            return (instance[-1] == step - 1 - self.get_alpha_num_back(instance[1], hop))
        else:
            return (instance[-1] <= step - 1 - self.get_alpha_num_back(instance[1], hop))
        

    @staticmethod
    def _find_cycle(adj: Dict[int, List[int]]) -> List[int]:
        """
        Return a directed cycle in `adj` as a node list [n0, ..., n0], or None if the
        graph is acyclic. Iterative-free recursive DFS is fine: each (source, epoch)
        subgraph has at most num_nodes vertices.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)  # defaults to WHITE
        stack: List[int] = []

        def visit(u: int):
            color[u] = GRAY
            stack.append(u)
            for v in adj.get(u, []):
                if color[v] == GRAY:
                    # found a back edge: the cycle is the tail of the stack from v.
                    return stack[stack.index(v):] + [v]
                if color[v] == WHITE:
                    found = visit(v)
                    if found is not None:
                        return found
            stack.pop()
            color[u] = BLACK
            return None

        for node in list(adj.keys()):
            if color[node] == WHITE:
                found = visit(node)
                if found is not None:
                    return found
        return None

    def cancel_flow_cycles(self, full_flow_list: List[Tuple[int, int, int, float, int]]
                           ) -> List[Tuple[int, int, int, float, int]]:
        """
        Cancel pure circulations from the solved flow list. The time-expanded flow graph
        is a DAG across epochs (GPU forwarding strictly advances the epoch), so every
        directed cycle is confined to a single (source, epoch) -- these arise from
        cut-through switches relaying within one epoch and from LP degeneracy. A cycle
        adds zero to every node's net divergence, so cancelling it preserves all buffers,
        consumes and deliveries; it only removes loops that dig_to_source() would otherwise
        follow forever. Each entry is (source, sender, receiver, volume, epoch).
        """
        EPS = 1e-9
        groups: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = defaultdict(dict)
        for (s, i, j, vol, k) in full_flow_list:
            edges = groups[(s, k)]
            edges[(i, j)] = edges.get((i, j), 0.0) + vol

        for edges in groups.values():
            while True:
                adj: Dict[int, List[int]] = defaultdict(list)
                for (i, j), vol in edges.items():
                    if vol > EPS:
                        adj[i].append(j)
                cycle = self._find_cycle(adj)
                if cycle is None:
                    break
                cyc_edges = list(zip(cycle[:-1], cycle[1:]))
                bottleneck = min(edges[e] for e in cyc_edges)
                for e in cyc_edges:
                    edges[e] -= bottleneck
                    if edges[e] <= EPS:
                        edges[e] = 0.0

        new_list = []
        for (s, k), edges in groups.items():
            for (i, j), vol in edges.items():
                if vol > EPS:
                    new_list.append((s, i, j, vol, k))
        return new_list

    def dig_to_source(self, hop: int, traffic: List[Tuple[int, int, int, int, float, int]],
                      consumed: Dict[int, Tuple[int, int, float]], source: int, step: int, dest: int,
                      volume: float, path: List[Tuple[int, int, int, float, int]], paths: Dict[int, List[Tuple[int, int, int, int, float, int]]] = {}) -> List[Tuple[int, int, int, int, float, int]]:
        """
        does DFS to trace back the path of a chunk to the source.
        """
        # find the nodes that sent the traffic to the node we are currently at.
        

        this_hop_consumed_volume = 0
        previous_hops = [x for x in traffic if x[2] == hop and (self.check_if_viable(hop, dest, step, x))]
        previous_hops = sorted(previous_hops, key=lambda x: x[-1], reverse = True)
        old_paths = paths
        for each_previous_hop in previous_hops:
            paths = old_paths
            if volume == 0:
                break
            found = False
            for i in range(len(traffic)):
                if traffic[i] == each_previous_hop:
                    consume = min(traffic[i][3], volume)
                    stored_traffic = traffic[i]
                    found = True
                    break
            assert found, "the previous hop not found in traffic list, must be a bug"
            
            step = each_previous_hop[-1]
            new_path = copy.deepcopy(path)
            new_path += [(source, each_previous_hop[1], hop, consume, step)]
            
            if consume <= 1e-5:
                continue
            traffic, consumed_volume = self.dig_to_source(
                    each_previous_hop[1], traffic, consumed, source, step, dest, consume, new_path, paths)
            
            for i in range(len(traffic)):
                if traffic[i][0] == stored_traffic[0]:
                    if traffic[i][1] == stored_traffic[1]:
                        if traffic[i][2] == stored_traffic[2]:
                                if traffic[i][4] == stored_traffic[4]:
                                    index = i
                                    break

            this_hop_consumed_volume += consumed_volume
            if consumed_volume < volume:
                volume = volume - consumed_volume
            else:
                volume = 0
           
            traffic[index] = (traffic[index][0], traffic[index][1], traffic[index]
                                  [2], traffic[index][3] - consumed_volume, traffic[index][4])
            if traffic[index][3] == 0:
                traffic = [x for x in traffic if x != traffic[index]]
        if len(previous_hops) == 0:
            assert source == hop
            this_hop_consumed_volume = min([x[3] for x in path])
            for i in range(len(path)):
                paths = self.account_for_consume(
                    this_hop_consumed_volume , source, dest, path[i][1], path[i][2], path[i][-1], paths)
            for c in paths.keys():
                if (source, dest, c) not in self.per_chunk_flow_paths.keys():
                    paths[c] = [x for x in paths[c] if len(x) > 0]
                    if len(paths[c]) == 0:
                        continue
                    self.per_chunk_flow_paths[(source, dest, c)] = [paths[c]]
                    paths[c] = []
                else:
                    self.per_chunk_flow_paths[(source, dest, c)] += [paths[c]]
                    paths[c] = []

        return traffic, this_hop_consumed_volume

    def get_per_chunk_flows(self) -> Dict:
        per_chunk_flow_list = {}
        # Iterate only the populated (s, i, j, c, k) entries of the sparse
        # per_chunk_flows container (a nested defaultdict) rather than the dense
        # num_nodes^3 x chunks x epochs product -- almost all of that product is
        # zero, and materializing/scanning it at 300 nodes is what OOM-killed /
        # hung schedule extraction.
        for s, s_map in self.per_chunk_flows.items():
            for i, i_map in s_map.items():
                for j, j_map in i_map.items():
                    for c, c_map in j_map.items():
                        for k, vol in c_map.items():
                            if vol > 0:
                                per_chunk_flow_list.setdefault(k, []).append(
                                    (int(s), int(i), int(j), int(c), vol, int(k)))
        # Preserve the old dense-product emission order (sorted by s, i, j, c
        # within each epoch) so the produced schedule is byte-identical to the
        # previous version for the same solver solution.
        for k in per_chunk_flow_list:
            per_chunk_flow_list[k].sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        return per_chunk_flow_list

    def chunk_flow_paths_to_string(self) -> Dict:
        """
        convert the flow paths in the per_chunk_flow_paths dictionary into their string version.
        """
        chunk_flow_paths = {}
        for s, d, c in self.per_chunk_flow_paths.keys():
            chunk_flow_paths[(s, d, c)] = []
            for each_path in self.per_chunk_flow_paths[(s, d, c)]:
                each_path = [x for x in each_path if len(x) != 0]
                if len(each_path) == 0:
                    continue
                each_path = sorted(each_path, key=lambda x: x[-1])
                path = []
                start = 0
                next = 0
                chunk_path = each_path[::-1]
                chunk_path = sorted(chunk_path, key=lambda x: x[-1])
                chunk_path = [x for x in chunk_path if round(x[4],6) >0]
                while start < len(chunk_path):
                    accumulated_flows = [chunk_path[start]]
                    while chunk_path[next][2] in self.topology.switch_indices:
                        next += 1
                        accumulated_flows.append(chunk_path[next])
                    start_node = accumulated_flows[0][1]
                    end_node = accumulated_flows[-1][2]
                    sending_epoch = accumulated_flows[0][5]
                    volume = accumulated_flows[0][4]
                    switches = "->".join([str(x[2])
                                         for x in accumulated_flows[:-1]])
                    if switches:
                        path.append(
                            (sending_epoch, f'{start_node}->{end_node} with volume {volume} in epoch {sending_epoch} via switches {switches}'))
                    else:
                        path.append(
                            (sending_epoch, f'{start_node}->{end_node} with volume {volume} in epoch {sending_epoch}'))
                    start = next = next + 1
                chunk_flow_paths[(s, d, c)].append(path)
            # Order a demand's multipath branches by their start epoch so the
            # emitted "8-Chunk paths" is canonical. teccl_ncclize's
            # parse_flows_alltoall assigns sub-chunk piece indices in path-list
            # order, so sorting here makes piece 0 correspond to the earliest
            # epoch (the pieces are interchangeable, so this is purely for a
            # readable, deterministic piece<->epoch correspondence).
            chunk_flow_paths[(s, d, c)].sort(
                key=lambda p: min(epoch for epoch, _ in p))
        return chunk_flow_paths

    def get_flow_schedule(self) -> Tuple[List, Dict]:
        full_flow_list, consumed = self.get_flows_and_consumes()

        # ===== TEMP DEBUG: raw continuous solver output (pre per-chunk decomposition) =====
        # full_flow_list entries: (source_s, sender_i, receiver_j, volume, epoch_k)
        #   == the solved value of variable f_s_i_j_k = flow[s][i][j][k].
        #   Volume is CONTINUOUS and AGGREGATED over all chunks that share source s;
        #   there is no chunk index here yet. dig_to_source() below splits it per chunk.
        print("\n" + "=" * 78)
        print("RAW LP FLOW VARS  f_s_i_j_k = flow[source s][link i->j][epoch k]")
        print("(continuous volume, summed over all of source s's chunks; not yet per-chunk)")
        print("=" * 78)
        for s in sorted({x[0] for x in full_flow_list}):
            rows = sorted([x for x in full_flow_list if x[0] == s],
                          key=lambda x: (x[4], x[1], x[2]))  # by epoch, then link
            print(f"\n-- source {s} --")
            for (_s, i, j, vol, k) in rows:
                itag = "SW" if i in self.topology.switch_indices else "gpu"
                jtag = "SW" if j in self.topology.switch_indices else "gpu"
                print(f"   epoch {k}:  {i:>2}({itag}) -> {j:>2}({jtag})   volume = {vol}")
        print("\n" + "-" * 78)
        print("RAW CONSUMED VARS  T_s_d_k = consumed_at_k[source s][dest d][epoch k]")
        print("(volume of source s's demand absorbed AT destination d during epoch k)")
        print("-" * 78)
        for d in sorted(consumed.keys()):
            for (s, k, vol) in sorted(consumed[d], key=lambda x: (x[0], x[1])):
                print(f"   dest {d:>2}:  from source {s:>2}  epoch {k}  consumed = {vol}")
        print("=" * 78 + "\n")
        # ===== END TEMP DEBUG =====

        # Remove pure circulations from the solved flow before decomposing it into paths.
        # The LP can add a same-epoch loop (e.g. a GPU sending a chunk into a cut-through
        # switch and receiving it right back in the same epoch) that satisfies flow
        # conservation but delivers nothing. Such a cycle would make dig_to_source() walk
        # it forever (RecursionError). Cancelling it changes no buffer/consume/delivery.
        full_flow_list = self.cancel_flow_cycles(full_flow_list)

        # Sparse per-chunk flow container (see the flow-cube note in
        # initialize_variables). The dense num_nodes^3 x num_chunks x num_epochs
        # array this used to be is ~60 GB at 300 nodes / 23 chunks and is almost
        # entirely zero -- only the handful of (s, i, j, c, k) pieces the path
        # tracer assigns are nonzero. A 5-level nested defaultdict stores just
        # those; writes below are unchanged (per_chunk_flows[s][i][j][c][k] = vol),
        # and get_per_chunk_flows() iterates the populated keys instead of the
        # dense product (which was also far too large/slow to scan).
        self.per_chunk_flows = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
        self.per_chunk_flow_paths = {}
        self.demand_copy = np.array(copy.deepcopy(self.demand), dtype=float)

        sources = set([x[0] for x in full_flow_list])
        destinations = self.nodes

        for each_source in sources:
            # traffic : flows sent on a link for a particular source
            traffic = [x for x in full_flow_list if x[0] == each_source]
            traffic = sorted(traffic, key=lambda x: x[-1])
            
            for each_dest in destinations:
                if self.demand_at_i[(each_source, each_dest)] == 0:
                    continue


                # all the steps in which a particular destination consumed hunks for a source
                steps = set(x[1]
                            for x in consumed[each_dest] if x[0] == each_source and x[2] > 1e-7)
                steps = sorted(list(steps))
                for each_step in steps:
                    # at a given step at a destination for a source there is only one
                    # consumed value so the volume list is a singleton
                    # volume represents the total amount consumed across all the incoming links
                    volume = [x[2] for x in consumed[each_dest]
                              if x[0] == each_source and x[1] == each_step][0]
                    hop = each_dest
                    traffic, _ = self.dig_to_source(
                        hop, traffic, consumed, each_source, each_step, each_dest, volume, [])
                    
            traffic = [x for x in traffic if x[3] > 1e-5]

            if len(traffic) != 0:
                logging.warning(
                    "there is a potential bug! there is traffic unaccounted for")
                print(traffic)
                assert 0, "potential bug, check code"

        per_chunk_flows = self.get_per_chunk_flows()
        chunk_str_paths = self.chunk_flow_paths_to_string()
        chunk_paths = {}
        flows_str = set()
        for (s, d, c) in self.per_chunk_flow_paths.keys():
            paths = self.per_chunk_flow_paths[(s, d, c)]
            k = max([x[-1] for x in [y[0] for y in paths if len(y[0]) > 0]])
            chunk_paths[f"Demand at {d} for chunk {c} from {s} met by epoch {k}"] = chunk_str_paths[(
                s, d, c)]
            for multipath in chunk_str_paths[(s, d, c)]:
                for epoch, path in multipath:
                    flows_str.add((epoch, f'Chunk {c} from {s} traveled over {path}'))

        required_flows = []
        Kmax = self.find_demand_satisfied_k()
        for k in range(Kmax):
            if k in per_chunk_flows.keys():
                required_flows += per_chunk_flows[k]
        flow_str_info = {}
        # Emit the actual collective this LP solved (the LP is collective-agnostic and now
        # serves alltoall/allgather/gather/broadcast), so downstream tools (e.g. ncclize)
        # build the right collective instead of assuming alltoall.
        collective = self.user_input.instance.collective
        flow_str_info["0-Collective"] = collective.name.lower()
        if collective in (Collective.GATHER, Collective.BROADCAST):
            # Rooted collectives need the root GPU index to reconstruct the collective.
            flow_str_info["0-Root"] = self.user_input.instance.root
        flow_str_info["1-Epoch_Duration"] = self.epoch_duration
        flow_str_info["2-Expected_Epoch_Duration"] = self.expected_epoch_duration
        flow_str_info["3-Epochs_Required"] = self.find_demand_satisfied_k() + 1
        flow_str_info["4-Collective_Finish_Time"] = flow_str_info["1-Epoch_Duration"] * flow_str_info["3-Epochs_Required"]
        flow_str_info["5-Algo_Bandwidth"] = self.topology.node_per_chassis * self.topology.chunk_size * self.num_chunks * self.topology.chassis / flow_str_info["4-Collective_Finish_Time"]
        flows_str = sorted(list(flows_str), key=lambda x: x[0])
        flow_str_info['7-Flows'] = [x[1] for x in flows_str]
        flow_str_info['8-Chunk paths'] = chunk_paths
        flow_str_info["9-Chunk_Size"] = self.topology.chunk_size
        return required_flows, flow_str_info

    def find_demand_satisfied_k(self) -> int:
        """
        returns the total number of epochs we need to satisfy the demand.
        """
        satisfied_epochs = {}
        for v in self.model.getVars():
            if 'T_' in v.varName and v.x > 0:
                components = v.varName.split('_')
                _, s, i, k = components
                if (s, i) in satisfied_epochs:
                    if satisfied_epochs[(s, i)] < int(k):
                        satisfied_epochs[(s, i)] = int(k)
                else:
                    satisfied_epochs[(s, i)] = int(k)
        return max(satisfied_epochs.values())

    def get_schedule(self) -> Tuple[List, Dict]:
        if self.model.SolCount > 0:
            return self.get_flow_schedule()
        else:
            return [], {}
