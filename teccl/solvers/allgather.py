from collections import defaultdict
import logging
import math
import time
from itertools import product
from types import SimpleNamespace
from typing import Dict, List, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from teccl.gurobi_env import get_gurobi_env
from teccl.input_data import *
from teccl.sends import Branch, Send
from teccl.solvers.base_formulation import BaseFormulation
from teccl.topologies.topology import Topology


class AllGatherFormulation(BaseFormulation):
    def __init__(self, user_input: UserInputParams, topology: Topology) -> None:
        super().__init__(user_input, topology)
        # The chunk-level integer program is demand-tensor driven and never branches on the
        # collective: it just satisfies self.demand, which teccl.solvers.demand builds per
        # collective. The class keeps its AllGather name for the callers that import it, but the
        # solver/log/model labels follow the collective actually being solved so an alltoall run
        # does not overwrite an allgather log.
        self.collective_name = user_input.instance.collective.name.lower()
        # Whether this level's switches copy (InstanceParams.switch_copy may be per level).
        self.switch_copy = resolve_switch_copy(user_input.instance, user_input.instance.level_depth)
        self.solver_name = f"{user_input.instance.collective.name.title()}_MILP"
        self.required_flows = []
        self.flows_str_info = {}
        # Re-derived in initialize_variables; set here too so a model-less instance (a replayed
        # solution, as in milp_trace_test) can still answer _paced / flow_class.
        self.pace = defaultdict(lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(dict))))
        self._flow_ub = {}
        self._init_pace_levels()

    # ------------------------------------------------------------------------------------------
    # Pipelined (cut-through) switch timing: PACE CLASSES
    #
    # With switch_pipeline on, a chunk a GPU sends into the switch fabric streams through every
    # switch to the next GPU without being stored -- one TRAVERSAL. Physically the whole traversal
    # runs at its BOTTLENECK rate: a switch has no room to absorb a chunk-sized speed mismatch, so
    # backpressure paces every link of it, fast ones included, to the slowest link's rate, and the
    # chunk lands when the slowest link has carried it. In epochs that is: every link of the
    # traversal busy from the sender's epoch k (plus propagation) for b + 1 epochs, and the chunk
    # in the receiver's buffer from k + alpha + 1 + b, where b = max beta over the traversal.
    #
    # The flow formulation is per link, so it does not know a traversal's links. Instead every
    # flow on a link with a switch at either end carries a pace class b (pace[s][i][j][c][k] maps
    # b -> its var, the copies sent in that class (max_copies); flow[...] is their sum), allowed
    # only for b >= beta(i, j), and switches
    # conserve flow PER CLASS. A traversal therefore keeps one class end to end, that class is at
    # least the beta of every link it crosses, and the solver picks the smallest one there is no
    # reason to exceed: the bottleneck. A class-b flow occupies 1/(b+1) chunk per epoch on its link
    # for b + 1 epochs (capacity_constraints), and lands b + 1 epochs after it starts
    # (_paced_landing). A link whose own beta is b carrying class b is charged exactly the old
    # beta window, so uniform-bandwidth fabrics and SLOWEST_LINK epochs (beta 0 everywhere) keep a
    # single class and the old model, re-timed (see below).
    #
    # This replaces the old rule, which made the first GPU->switch hop store-and-forward (the
    # switch sent from k + alpha + 1 + beta(first hop)) and every later hop free. That credited a
    # traversal with its FIRST link's beta instead of its slowest link's, so a slow link after a
    # fast one was crossed in less time than it physically takes, and the fast link was freed
    # while the slow one was still draining the same chunk. On uniform links the two models are
    # the same schedule shifted in time: the old one indexed a switch's out-links 1 + beta epochs
    # after the head, this one indexes them at the head. The old one therefore could not use the
    # horizon's last epoch for a switch traversal (use_one_less_epoch existed to compensate).
    # ------------------------------------------------------------------------------------------
    def _paced(self, i: int, j: int) -> bool:
        """Whether link (i, j) is part of a cut-through traversal, i.e. carries pace classes."""
        return self.user_input.instance.switch_pipeline and (
            i in self._switches or j in self._switches)

    def pace_classes(self, i: int, j: int) -> List[int]:
        """The pace classes link (i, j) may carry: every bottleneck class at least its own beta."""
        beta = self.get_beta_num_back(i, j)
        return [b for b in self.pace_levels if b >= beta]

    def _pace(self, s: int, i: int, j: int, c: int, k: int) -> Dict[int, gp.Var]:
        """class -> var for the send of chunk (s, c) on paced link (i, j) in epoch k ({} if none)."""
        if k < 0:
            return {}
        return self.pace[s][i][j][c].get(k, {})

    def _paced_landing(self, s: int, j: int, i: int, c: int, e: int) -> gp.LinExpr:
        """The sends on paced link (j, i) into non-switch node i whose traversal has finished
        landing chunk (s, c) by the END of epoch e: a class-b send started in epoch e - alpha - b."""
        alpha = self.get_alpha_num_back(j, i)
        landing = gp.LinExpr(0.0)
        for b in self.pace_classes(j, i):
            var = self._pace(s, j, i, c, e - alpha - b).get(b)
            if var is not None:
                landing.add(var)
        return landing

    def flow_classes(self, s: int, i: int, j: int, c: int, k: int) -> Dict[int, int]:
        """class -> copies of a solved send (s, i, j, c, k). A send may carry several copies
        (max_copies), possibly of different classes; an unpaced link carries its own beta."""
        classes = self._pace(s, i, j, c, k) if self._paced(i, j) else {}
        if not classes:
            return {self.get_beta_num_back(i, j): 1}
        if len(classes) == 1:
            return {next(iter(classes)): 1}
        return {b: int(round(var.X)) for b, var in classes.items() if var.X > 0.5}

    def flow_class(self, s: int, i: int, j: int, c: int, k: int) -> int:
        """The pace class of a solved send that carries one class (every send of a chunk with a
        single destination, and every send under SLOWEST_LINK epochs)."""
        classes = self.flow_classes(s, i, j, c, k)
        if len(classes) != 1:
            raise ValueError(f"send {(s, i, j, c, k)} carries classes {classes}; "
                             f"read the traced class from per_demand_classes instead")
        return next(iter(classes))

    def _init_pace_levels(self) -> None:
        self._switches = set(self.topology.switch_indices)
        self.pace_levels = sorted({
            self.get_beta_num_back(i, j) for i, j in product(self.nodes, self.nodes)
            if self.topology.capacity[i][j] > 0 and self._paced(i, j)})

    def max_copies(self, s: int, c: int) -> int:
        """How many copies of chunk (s, c) one paced send may carry at once: one per node that
        demands it.

        A paced send is a stream, not a slot: a fast link can carry two class-1 streams of the same
        chunk side by side, each at half rate. That is how a chunk reaches two receivers through a
        switch that cannot copy (switch_copy off) -- e.g. a GPU sending its own data to a far leaf
        and to its leaf neighbour at once over one NIC -- and with a slow class it is the only way
        to use the half of the link the first stream leaves free. Capping a send at 1 would forbid
        it and cost epochs no hardware pays. A chunk wanted by one node (every alltoall chunk) is
        capped at 1, so those models are unchanged."""
        return max(1, int(sum(1 for d in self.nodes if self.demand[s][d][c])))

    def class_copies(self, s: int, c: int) -> int:
        """Copies of chunk (s, c) one paced send may carry in ONE pace class. With switch copy it
        is 1: a same-class copy downstream of a copying switch is that switch's job, and a GPU
        needs a chunk once, so a second same-class copy on one link only spends its capacity.
        Concurrent copies then exist only across classes. Without switch copy, max_copies."""
        return 1 if self.switch_copy else self.max_copies(s, c)

    def send_bound(self, s: int, i: int, j: int, c: int, k: int) -> int:
        """Upper bound of flow[s][i][j][c][k]: how many copies that send may carry (1 if none)."""
        return self._flow_ub.get((s, i, j, c, k), 1)

    def flow_copies(self, s: int, i: int, j: int, c: int, k: int) -> int:
        """How many copies a solved send carries (1 for a model-less, replayed solution)."""
        flow = self.flow[s][i][j][c][k] if hasattr(self, "flow") else None
        return int(round(flow.X)) if isinstance(flow, gp.Var) else 1

    def _add_paced_link_vars(self, i: int, j: int) -> None:
        """flow (and, when there is a choice, per-class pace) vars for paced link (i, j)."""
        alpha = self.get_alpha_num_back(i, j)
        classes = self.pace_classes(i, j)
        for s, c, k in product(self.sources, self.chunks, self.epochs):
            if s == j:
                continue
            # A class-b traversal through this link lands no earlier than the end of epoch
            # k + alpha + b; one that cannot land inside the horizon is never useful.
            live = [b for b in classes if k + alpha + b <= self.num_epochs - 1]
            if not live:
                continue
            per_class = self.class_copies(s, c)
            copies = min(self.max_copies(s, c), per_class * len(live))
            flow = self.model.addVar(
                0, copies, vtype=GRB.INTEGER, name='flow_%d_%d_%d_%d_%d' % (s, i, j, c, k))
            self.flow[s][i][j][c][k] = flow
            if copies > 1:
                self._flow_ub[(s, i, j, c, k)] = copies
            if len(live) == 1:
                self.pace[s][i][j][c][k] = {live[0]: flow}
                continue
            pace = {b: self.model.addVar(0, per_class, vtype=GRB.INTEGER,
                                         name='pace_%d_%d_%d_%d_%d_%d' % (s, i, j, c, k, b))
                    for b in live}
            self.model.addConstr(flow == gp.quicksum(pace.values()),
                                 name='pace_sum_%d_%d_%d_%d_%d' % (s, i, j, c, k))
            self.pace[s][i][j][c][k] = pace

    def initialize_variables(self) -> None:
        """
            flow (s, i, j, c, k) : Whether chunk (s,c) is going on link (i,j) in epoch k.
            buffer (s, i, c, k) : Whether the chunk (s, c) is in the buffer at node i at beginning of epoch k.
            total_demand_sat (s, i, c, k):  Whether the demand for chunk (s,c) is satisfied at node i by the end of epoch k.
        """
        logging.debug(f'Initializing flow variables')
        time_start = time.time()
        # Demand-bearing sources: the GPUs that actually originate a chunk (in
        # allgather, every active GPU sources its own data; switches and passive
        # nodes source nothing). Every flow/buffer/demand variable is indexed by a
        # source s; for a zero-demand source flow[s][.] is provably 0 (its chunk
        # never exists), so restricting the source dimension to self.sources
        # removes only structurally-zero variables/constraints -- no change to the
        # feasible set or optimum -- and shrinks the model by num_nodes/#sources.
        # Relaying is preserved: an idle node still forwards an active source's
        # chunk as an intermediate i/j hop in flow[active_source][i][j] (the node
        # dimension stays all num_nodes). (AStarFormulation overrides this to keep
        # all nodes as sources -- see its initialize_variables.)
        self.sources = [s for s in self.nodes if self.demand[s].any()]
        # Sparse container: the dense num_nodes^3 x num_chunks x num_epochs list this
        # used to be is almost entirely zeros -- only real links carry a variable, and
        # even on real links many (s, c, k) entries are skipped below (s == j,
        # not_necessary_k). A 5-level nested defaultdict makes a missing
        # (s, i, j, c, k) read as 0.0 -- exactly the old np.zeros default -- so every
        # LinExpr.add()/gp.max_() downstream (which already received those 0.0's for
        # skipped entries) is unchanged, while only the created entries are stored.
        self.flow = defaultdict(lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(float)))))
        # pace[s][i][j][c][k] -> {class: var}, paced links only (see the pace-class note above).
        self.pace = defaultdict(lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(dict))))
        self._flow_ub = {}
        self._init_pace_levels()

        for i, j, in product(self.nodes, self.nodes):
            if self.topology.capacity[i][j] <= 0:
                continue
            if self._paced(i, j):
                self._add_paced_link_vars(i, j)
                continue
            alpha_num_back = self.get_alpha_num_back(i, j)
            link_type = self.get_link_type(i, j)
            beta_num_back = self.get_beta_num_back(i, j)
            extra_epochs = alpha_num_back + beta_num_back
            not_necessary_k = [self.num_epochs -
                               i - 1 for i in range(extra_epochs)]
            for s, c, k in product(self.sources, self.chunks, self.epochs):
                if s == j:
                    # Node j never need to receive its chunks from others
                    continue
                if link_type == self.LinkType.GPU_SWITCH and k in not_necessary_k:
                    continue
                if link_type == self.LinkType.GPU_GPU and k in not_necessary_k:
                    # For GPU-GPU link, we don't need to send the chunk in the last epochs as the chunk will not reach the destination.
                    continue
                # We leave the SWITCH_GPU link as it is as it links to the buffers on the GPUs and can not be set arbitarily.
                self.flow[s][i][j][c][k] = self.model.addVar(
                    0, 1, vtype=GRB.INTEGER, name='flow_%d_%d_%d_%d_%d' % (s, i, j, c, k))
        logging.debug(
            f'Finished initializing flow variables in {time.time()-time_start}')

        logging.debug(f'Initializing buffer and demand_sat variables')
        time_start = time.time()
        self.buffer = np.zeros((self.num_nodes, self.num_nodes,
                                self.num_chunks, self.num_epochs)).tolist()

        self.total_demand_sat = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_chunks, self.num_epochs)).tolist()

        for s, i, c, k in product(self.sources, self.nodes, self.chunks, self.epochs):
            self.buffer[s][i][c][k] = self.model.addVar(
                0, 1, vtype=GRB.INTEGER, name='buffer_%d_%d_%d_%d' % (s, i, c, k))
            if self.demand[s][i][c]:
                self.total_demand_sat[s][i][c][k] = self.model.addVar(
                    0, 1, vtype=GRB.INTEGER, name='total_demand_%d_%d_%d_%d' % (s, i, c, k))
        logging.debug(
            f'Finished initializing buffer and demand_sat variables in {time.time()- time_start}')

    def destination_constraints(self, astar_multiple_rounds: bool = False) -> None:
        """
            Adds constraints to the model for calculating the demand satisfied at each node after each epoch.
            When astar_multiple_rounds is False, it adds constraints that demand must be satisfied by the end of the last epoch.
        """
        logging.debug(f'Adding destination constraints')
        start = time.time()
        for s, d, c, k in product(self.nodes, self.nodes, self.chunks, self.epochs):
            if not self.demand[s][d][c]:
                continue
            if k + 1 < self.num_epochs:
                # The demand satsified by the end of the kth epoch is equal to the
                #   buffer at the beginning of k+1th epoch
                self.model.addConstr(self.total_demand_sat[s][d][c][k] == self.buffer[
                    s][d][c][k + 1], name="dest_total_s_%d_d_%d_c_%d_k_%d" % (s, d, c, k))
            else:
                # k = self.num_epochs - 1
                # For the last epoch B[K+1] does not exist
                # We consider the buffer at the beginning of last round and the flows in the last round that add to the buffer
                dem_sat_constr = gp.LinExpr(0.0)
                dem_sat_constr.add(self.buffer[s][d][c][k])
                for i in self.nodes:
                    if self.topology.capacity[i][d] > 0:
                        if self._paced(i, d):
                            dem_sat_constr.add(self._paced_landing(s, i, d, c, k))
                            continue
                        alpha_num_back = self.get_alpha_num_back(i, d)
                        beta_num_back = self.get_beta_num_back(i, d)
                        if k - alpha_num_back - beta_num_back >= 0:
                            dem_sat_constr.add(
                                self.flow[s][i][d][c][k - alpha_num_back - beta_num_back])
                self.model.addConstr(dem_sat_constr == self.total_demand_sat[
                    s][d][c][k], name="dest_total_last_s_%d_d_%d_c_%d_k_%d" % (s, d, c, k))
                if not astar_multiple_rounds:
                    # At the end of the last epoch, the demand must be satisfied if it is not an A* round
                    self.model.addConstr(self.total_demand_sat[
                        s][d][c][k] == self.demand[s][d][c], name="dest_final_s_%d_d_%d_c_%d_k_%d" % (s, d, c, k))
        logging.debug(
            f'Finished adding destination constraints in {time.time() - start}')

    def hop_timing(self, j: int, i: int, pace: int = None) -> Tuple[int, int]:
        """
            (landing offset, occupancy) of one send on link (j, i), for consumers that read the
            schedule back (the hierarchical solver's tree extraction).

            landing offset: a send in epoch k is usable at i from epoch k + offset -- in a
                non-switch i's buffer (the inverse of _arrival_flow, so the two cannot disagree; the
                epoch during which the bytes LAND is therefore k + offset - 1), or, at a switch on a
                cut-through traversal, forwarded in epoch k + offset.
            occupancy: how many epochs the send holds the link: its pace class + 1 on a paced link,
                else beta + 1 (get_beta_num_back).

            pace: the send's pace class (flow_class); defaults to the link's own beta, the class it
            carries whenever it is its traversal's bottleneck.
        """
        alpha_num_back = self.get_alpha_num_back(j, i)
        beta_num_back = self.get_beta_num_back(j, i)
        if not self._paced(j, i):
            return alpha_num_back + 1 + beta_num_back, beta_num_back + 1
        if pace is None:
            pace = beta_num_back
        if i in self._switches:
            return alpha_num_back, pace + 1
        return alpha_num_back + 1 + pace, pace + 1

    def _arrival_flow(self, s: int, j: int, i: int, c: int, k: int):
        """
            The flow on link (j, i) that lands chunk (s, c) in non-switch node i's buffer at the
            BEGINNING of epoch k, or None if no send on that link can land by then. This is the
            per-in-link arrival term of the buffer recurrence
                buffer[k] = buffer[k-1] + sum_j arrival(j -> i, k)
            factored out so forward_dwell_constraints reads the identical term.
        """
        if self._paced(j, i):
            # landed by the end of epoch k - 1, i.e. in the buffer at the beginning of epoch k
            landing = self._paced_landing(s, j, i, c, k - 1)
            return landing if landing.size() else None
        alpha_num_back = self.get_alpha_num_back(j, i)
        beta_num_back = self.get_beta_num_back(j, i)
        if k - alpha_num_back - 1 - beta_num_back >= 0:
            # -1 is required as Buffer[i] stands for buffer at the "beginning" of epoch i
            return self.flow[s][j][i][c][k - alpha_num_back - 1 - beta_num_back]
        return None

    def node_constraint_helper(self, i: int, s: int, c: int, k: int, previous_round_buffers: List[List[int]]) -> None:
        """
            Adds contraints to model how the buffer changes with each epoch.
            Adds constraints to model what flows can arise from a node given the buffers.
        """
        buffer_constr = gp.LinExpr(0.0)
        if k == 0 and i != s:
            # The node i is not the source and the epoch is the first epoch.
            buffer_constr.add(self.buffer[s][i][c][k])
            from_last_round = 0
            if i not in self.topology.switch_indices and previous_round_buffers:
                # If the node is not a switch then it can have the chunk from the previous round into the buffer.
                # For a switch, buffer is always zero; the previous round flows are accounted in the flow constraints below.
                from_last_round = previous_round_buffers[s][i][c][0]
            self.model.addConstr(
                buffer_constr == from_last_round, name="node_first_buffer_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
        elif k == 0 and i == s:
            # max request for chunk C that originates from node s.
            dem_c = max([self.demand[s][d][c] for d in self.nodes])
            buffer_constr.add(self.buffer[s][s][c][0])
            # we have the chunks that need to be sent at s.
            self.model.addConstr(
                buffer_constr == dem_c, name="node_initial_buffer_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
        elif i not in self.topology.switch_indices:
            # constraints on the buffer contents for non-switches for all epochs other than first (k > 0) epoch
            buffer_constr.add(self.buffer[s][i][c][k - 1])
            for j in self.nodes:
                if self.topology.capacity[j][i] > 0:
                    arrival = self._arrival_flow(s, j, i, c, k)
                    if arrival is not None:
                        buffer_constr.add(arrival)
            if previous_round_buffers and k < len(previous_round_buffers[s][i][c]):
                if previous_round_buffers[s][i][c][k - 1] == 0 and previous_round_buffers[s][i][c][k] == 1:
                    # The chunk sent in the previous round arrived by the beginning of epoch k
                    # If previous k-1 is positive, it means the chunk has arrived before.
                    buffer_constr.add(previous_round_buffers[s][i][c][k])
            self.model.addConstr(buffer_constr == self.buffer[
                s][i][c][k], name="node_buffer_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
        elif i in self.topology.switch_indices:
            buffer_constr.add(self.buffer[s][i][c][k])
            self.model.addConstr(
                buffer_constr == 0, name="node_switch_buffer_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))

        # constraint on what can get out of the node in the current epoch
        if i not in self.topology.switch_indices:
            # For non-switch nodes, the buffer already acounts for previous rounds and flows.
            # A node can send in an epoch what is in the buffer at the start of the epoch.
            # Since a node can (copy) send the chunk from its buffer to multiple neighbors, we account for that
            #  using the max_ function. The maximum that a node can send to any of its neighbors is the buffer.
            # A send that may carry several copies at once (send_bound > 1, paced links only) is
            # bounded by copies * buffer instead: one held chunk feeds any number of copies.
            single = []
            for v in range(self.num_nodes):
                if self.topology.capacity[i][v] <= 0:
                    continue
                copies = self.send_bound(s, i, v, c, k)
                if copies > 1:
                    self.model.addConstr(
                        self.flow[s][i][v][c][k] <= copies * self.buffer[s][i][c][k],
                        name="node_copies_s_%d_i_%d_v_%d_c_%d_k_%d" % (s, i, v, c, k))
                    continue
                single.append(self.flow[s][i][v][c][k])
            if single:
                self.aux_var.append(self.model.addVar(
                    0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='aux_var_%d_%d_%d_%d' % (s, i, c, k)))
                self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(single), name="node_aux_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))

                node_constr = gp.LinExpr(0.0)
                node_constr.add(self.buffer[s][i][c][k])
                self.model.addConstr(node_constr >= self.aux_var[
                    len(self.aux_var) - 1], name="node_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
        elif self.user_input.instance.switch_pipeline:
            # Cut-through switch: it forwards, in the SAME class, what reaches it in this epoch.
            # Conservation per pace class is what keeps a traversal at one class end to end, and
            # hence paced at its slowest link (see the pace-class note at the top of the class).
            # Every in-link counts at k - alpha, the GPU->switch one included: the traversal's
            # links all start in the sender's epoch. Epoch 0 is no exception -- a source GPU's
            # epoch-0 send is relayed in epoch 0. (previous_round_buffers is A*-only and A* keeps
            # its own timing rules, so it is not threaded through here.)
            inflow = defaultdict(lambda: gp.LinExpr(0.0))
            for j in self.nodes:
                if self.topology.capacity[j][i] > 0:
                    alpha_num_back = self.get_alpha_num_back(j, i)
                    for b, var in self._pace(s, j, i, c, k - alpha_num_back).items():
                        inflow[b].add(var)
            outflow = defaultdict(list)
            for v in self.nodes:
                if self.topology.capacity[i][v] > 0:
                    for b, var in self._pace(s, i, v, c, k).items():
                        outflow[b].append(var)
            for b, outs in sorted(outflow.items()):
                self._switch_egress_constr(inflow[b], outs, "s_%d_i_%d_c_%d_k_%d_b_%d" % (s, i, c, k, b))
        else:
            # Store-and-forward switch: it sends what fully arrived in earlier epochs and rounds.
            switch_node_constr = gp.LinExpr(0.0)
            if k > 0:
                for j in range(self.num_nodes):
                    if self.topology.capacity[j][i] > 0:
                        alpha_num_back = self.get_alpha_num_back(j, i)
                        beta_num_back = self.get_beta_num_back(j, i)
                        if k - alpha_num_back - 1 - beta_num_back >= 0:
                            switch_node_constr.add(
                                self.flow[s][j][i][c][k - alpha_num_back - 1 - beta_num_back])

            if previous_round_buffers and k < len(previous_round_buffers[s][i][c]):
                switch_node_constr.add(previous_round_buffers[s][i][c][k])

            self._switch_egress_constr(
                switch_node_constr,
                [self.flow[s][i][v][c][k] for v in range(self.num_nodes) if self.topology.capacity[i][v] > 0],
                "s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))

    def _switch_egress_constr(self, inflow: gp.LinExpr, outs: List, tag: str) -> None:
        """What a switch may send in one epoch given what reaches it (`inflow`)."""
        if not self.switch_copy:
            '''
                switch no-copy version. For every outgoing chunk there has to be a corresponding incoming chunk.
            '''
            switch_outgoing_links = gp.LinExpr(0.0)
            for out in outs:
                switch_outgoing_links.add(out)
            # We allow the switch to drop the chunks with ">=" as we found it to be faster.
            # If we use "==" then in allgather case, it will result in no unnecessary flows.
            self.model.addConstr(inflow >= switch_outgoing_links, name="switch_flow_" + tag)
        else:
            '''
                switch copy version. The switch can copy the incoming chunk to multiple outgoing links.
            '''
            self.aux_var.append(self.model.addVar(
                0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='aux_var_switch_' + tag))
            self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(outs),
                                 name="node_switch_flow_" + tag)
            # We allow the switch to drop the chunks with ">=" as we found it to be faster.
            # If we use "==" then in allgather case, it will result in no unnecessary flows, which is necessary for the symmetric case.
            if self.user_input.instance.symmetry:
                self.model.addConstr(inflow == self.aux_var[len(self.aux_var) - 1],
                                     name="switch_flow_" + tag)
            else:
                self.model.addConstr(inflow >= self.aux_var[len(self.aux_var) - 1],
                                     name="switch_flow_" + tag)

    def node_constraints(self, previous_buffers=None) -> None:
        logging.debug(f'Adding node buffer and flow constraints')
        start = time.time()
        # s ranges only over demand-bearing sources; i (the node whose buffer/flow
        # conservation we write) stays all nodes, so any node can still relay/buffer
        # an active source's chunk. flow/buffer for a zero-demand source are provably
        # 0 and were not created, so their constraints would be vacuous.
        for i, s, c, k in product(self.nodes, self.sources, self.chunks, self.epochs):
            self.node_constraint_helper(i, s, c, k, previous_buffers)
        logging.debug(
            f'Finished adding node buffer and flow constraints in {time.time() - start}')

    def forward_dwell_constraints(self) -> None:
        """
            Forwarding dwell at a collapsed CELL node (hierarchical levels only).

            A coarse cell is many GPUs folded into one node, and the node's store-and-forward buffer
            charges one epoch between arrival and re-send. That is exact when the GPU the chunk lands
            on also owns the outgoing link (co-located). When it does not, the bytes must first cross
            the cell's internal fabric to the gateway, and since a send is paced across its whole
            epoch, that relay needs an epoch of its own. So for a non-source cell i and out-link
            i -> j:

                flow[s][i][j][c][k] <= buffer[s][i][c][k-1] + sum_{h in B(i,j)} arrival(h -> i, k)

            B(i,j) = the in-neighbours h whose link into i shares an owning GPU with i -> j
            (topology.forward_bridges, built by hierarchy.abstract). Forwarding is therefore allowed
            in the arrival epoch only over a bridged pair, and otherwise only after the chunk has
            sat in the cell for a full epoch. Consumption is untouched: total_demand_sat still
            reads buffer[k+1], so a cell that WANTS the chunk is never delayed by this.

            Exact per pair, not a per-cell 1-or-2 dwell, because the buffer is binary and its
            recurrence admits at most one arrival -- "the in-link the chunk came on" is well defined.
            A pair whose B(i,j) is every in-neighbour is skipped: there the row is implied by the
            buffer recurrence itself. Topologies without `forward_bridges` (every flat run) get none.
        """
        bridges = getattr(self.topology, "forward_bridges", None)
        if not bridges:
            return
        start = time.time()
        added = 0
        for (i, j), bridged in sorted(bridges.items()):
            if self.topology.capacity[i][j] <= 0:
                continue
            in_nbrs = {h for h in self.nodes if self.topology.capacity[h][i] > 0}
            if in_nbrs <= set(bridged):
                continue
            for s, c, k in product(self.sources, self.chunks, self.epochs):
                if s == i or k == 0:
                    # the source's own data is native (nothing is forwarded), and at k = 0 a
                    # non-source buffer is empty, so the node constraint already forbids the send.
                    continue
                out = self.flow[s][i][j][c][k]
                if not isinstance(out, gp.Var):
                    continue
                rhs = gp.LinExpr(self.buffer[s][i][c][k - 1])
                for h in sorted(bridged & in_nbrs):
                    arrival = self._arrival_flow(s, h, i, c, k)
                    if arrival is not None:
                        rhs.add(arrival)
                self.model.addConstr(
                    out <= rhs, name="fwd_dwell_s_%d_i_%d_j_%d_c_%d_k_%d" % (s, i, j, c, k))
                added += 1
        logging.debug(f'Added {added} forward-dwell constraints in {time.time() - start}')

    def capacity_constraints(self) -> None:
        logging.debug(f'Adding capacity constraints')
        start = time.time()
        for i, j, k in product(self.nodes, self.nodes, self.epochs):
            if self.topology.capacity[i][j] <= 0:
                continue
            cap_constr = gp.LinExpr(0.0)
            epoch_capacity = self.topology.capacity[i][j] * self.epoch_duration
            beta_num_back = max(0, math.ceil(1 / epoch_capacity) - 1)
            if k - beta_num_back < 0:
                continue
            if self._paced(i, j):
                # Epoch k's load: every class-b send started in k-b..k streams 1/(b+1) chunk through
                # the link in epoch k. Scaled by beta + 1, a link's own class (b = beta) reads
                # exactly as the window row below; a slower class takes a smaller share for longer.
                # The rows for k < beta that the guard above skips are implied by row beta, since
                # every class's window reaches back to epoch 0 there.
                for b in self.pace_classes(i, j):
                    for l in range(b + 1):
                        for s, c in product(self.sources, self.chunks):
                            var = self._pace(s, i, j, c, k - l).get(b)
                            if var is not None:
                                cap_constr.add(var, (beta_num_back + 1) / (b + 1))
                self.model.addConstr(
                    cap_constr <= ((beta_num_back + 1) * epoch_capacity), name="capacity_i_%d_j_%d_k_%d" % (i, j, k))
                continue
            for l in range(beta_num_back + 1):
                # Only demand-bearing sources carry flow; a zero-demand source's
                # flow is 0 and contributes nothing to the link load.
                for s, c in product(self.sources, self.chunks):
                    if k - l >= 0:
                        cap_constr.add(self.flow[s][i][j][c][k - l])
            self.model.addConstr(
                cap_constr <= ((beta_num_back + 1) * epoch_capacity), name="capacity_i_%d_j_%d_k_%d" % (i, j, k))
        logging.debug(
            f'Finished adding capacity constraints in {time.time() - start}')

    def switch_acyclicity_constraints(self) -> None:
        """
            Forbid same-epoch switch cut-through CYCLES of any length. Only needed when
            switches BOTH cut through AND copy (see the guard below).

            The switch flow constraint (node_constraint_helper, switch branch) is pure
            flow *conservation*, and a cut-through leg relays within the SAME epoch
            (alpha_num_back == 0). Conservation is satisfied by a circulation with zero
            external injection, so the solver can route chunk (s, c) around a same-epoch
            switch loop (e.g. T0->T1->T0) that no GPU ever fed. Under switch_copy the egress
            is `max(outgoing)`, so that one circulating unit can ALSO be copied to a real
            delivery -- the loop then satisfies a demand that is physically unrealizable
            (nothing produces the chunk at any switch within the epoch). See
            diagnose_switch_phantom_cycles / dfs_remove_unnecessary_flows.

            Fix: an MTZ ordering per (source, chunk, epoch). Give each cut-through-fabric
            switch v a continuous potential u[s][v][c][k] in [1, N] and require, for every
            switch->switch flow that is 1, u[a] < u[b]. A directed cycle would need a value
            strictly less than itself, so no same-epoch switch cycle of ANY length can exist,
            while a legitimate same-epoch multi-hop cut-through chain (GPU->a->b->GPU') is
            still allowed -- it just has strictly increasing potential. A chunk genuinely
            sent one direction never needs to complete a same-epoch loop, so this removes no
            realizable optimum. Potentials are created lazily, only for (s, c, k) that carry
            a switch->switch flow, keeping the added model small.
        """
        # Needed only when switches BOTH cut through and copy:
        #  - without cut-through (switch_pipeline), a switch->switch hop advances the epoch,
        #    so no same-epoch cycle can form at all;
        #  - without copy, the switch egress is pure sum-conservation (Sum out <= Sum in,
        #    like the LP): a same-epoch circulation cancels and CANNOT feed a delivery, so it
        #    is a benign/removable waste loop that the backtracking extractor routes around
        #    (and prunes) -- no constraint required.
        # Only the copy (max-egress) case lets a circulation manufacture an ungrounded
        # delivery, which is the physically-unrealizable phantom this constraint forbids.
        if not (self.user_input.instance.switch_pipeline and self.switch_copy):
            return
        switches = set(self.topology.switch_indices)
        ss_links = [(a, b) for a in switches for b in switches
                    if a != b and self.topology.capacity[a][b] > 0]
        fabric = sorted({v for link in ss_links for v in link})
        if len(fabric) < 2:
            # No switch<->switch pair -> no same-epoch switch cycle is possible.
            return
        logging.debug('Adding switch acyclicity (MTZ) constraints')
        start = time.time()
        N = len(fabric)
        order = {}

        def order_var(s: int, v: int, c: int, k: int) -> gp.Var:
            key = (s, v, c, k)
            if key not in order:
                order[key] = self.model.addVar(
                    1, N, vtype=GRB.CONTINUOUS,
                    name='switch_order_%d_%d_%d_%d' % (s, v, c, k))
            return order[key]

        for s, c, k in product(self.sources, self.chunks, self.epochs):
            for (a, b) in ss_links:
                f = self.flow[s][a][b][c][k]
                if not isinstance(f, gp.Var):
                    continue
                copies = self.send_bound(s, a, b, c, k)
                if copies > 1:
                    # a send that may carry several copies (max_copies): order on whether it is
                    # used at all, or two copies would make the row infeasible outright
                    used = self.model.addVar(vtype=GRB.BINARY,
                                             name='switch_used_%d_%d_%d_%d_%d' % (s, a, b, c, k))
                    self.model.addConstr(f <= copies * used,
                                         name="switch_used_s_%d_a_%d_b_%d_c_%d_k_%d" % (s, a, b, c, k))
                    f = used
                # f == 1  =>  u[a] - u[b] <= -1  (u[a] < u[b]); f == 0 leaves it slack.
                self.model.addConstr(
                    order_var(s, a, c, k) - order_var(s, b, c, k) + N * f <= N - 1,
                    name="switch_acyc_s_%d_a_%d_b_%d_c_%d_k_%d" % (s, a, b, c, k))
        logging.debug(
            f'Finished adding switch acyclicity constraints in {time.time() - start}')

    def use_one_less_epoch(self) -> None:
        """
           No-op, kept for the iterative driver (scheduler.get_schedules) that asks for it.

           It used to force every demand to be met one epoch before the horizon when switches cut
           through, because a cut-through switch's out-links were indexed one epoch after the
           chunk's arrival was credited, so a traversal landing in the last epoch needed a send in
           an epoch past the horizon. Pace classes index a traversal at its head and credit it when
           its last byte lands (see the pace-class note at the top of the class), so a horizon of K
           epochs now holds every schedule that finishes in K and nothing needs to be held back.
        """
        return

    def add_symmetry_constraints(self) -> None:
        """
            Constraints to make the solution symmetric.
            If node i and j are symmetric, then the total outflow /inflow from i and j should be the same.
        """
        logging.debug(f'Adding symmetry constraints')
        start = time.time()
        for symmetry_nodes in self.topology.equivalent_node_indices:
            i = symmetry_nodes[0]
            for j in symmetry_nodes[1:]:
                flow_equality_out_constr = gp.LinExpr(0.0)
                flow_equality_in_constr = gp.LinExpr(0.0)
                for s, l, c, k in product(self.sources, self.nodes, self.chunks, self.epochs):
                    flow_equality_out_constr.add(self.flow[s][i][l][c][k])
                    flow_equality_out_constr.add(self.flow[s][j][l][c][k], -1)
                    flow_equality_in_constr.add(self.flow[s][l][i][c][k])
                    flow_equality_in_constr.add(self.flow[s][l][j][c][k], -1)
                self.model.addConstr(
                    flow_equality_out_constr == 0, name="symmetry_out_%d_%d" % (i, j))
                self.model.addConstr(
                    flow_equality_in_constr == 0, name="symmetry_in_%d_%d" % (i, j))
        logging.debug(
            f'Finished adding symmetry constraints in {time.time() - start}')

    def objective_formulation(self, objective_type: ObjectiveType = ObjectiveType.PAPER) -> gp.LinExpr:
        logging.debug(f'Adding objective')
        start = time.time()
        # Gurobi minimizes the objective
        objective = gp.LinExpr(0.0)
        mulitplier = 100
        self.epoch_used = []

        demand_total = 0
        for s, d, c in product(self.nodes, self.nodes, self.chunks):
            if not self.demand[s][d][c]:
                continue
            demand_total += 1

        if objective_type == ObjectiveType.BINARY_USED_EPOCHS:
            # Minimize the number of epochs used using a binary variable for each epoch
            for k in self.epochs:
                self.epoch_used.append(self.model.addVar(
                    vtype=GRB.BINARY, name=f'epoch_used_{k}'))
                if len(self.epoch_used) > 1:
                    # If epoch k is not used then all the subsequent epochs also should not be used.
                    self.model.addConstr(
                        self.epoch_used[-2] >= self.epoch_used[-1], name=f'epoch_used_{k}-{k-1}')
                tmp = gp.LinExpr(0.0)
                # An epoch is used if there is some flow in that epoch
                # "some flow" --> the flow variables could all be zero, but the epoch can be used if the flow started
                # in a prior epoch and is still going through
                for i, j in product(self.nodes, self.nodes):
                    if self.topology.capacity[i][j] <= 0 or i in self.topology.switch_indices:
                        continue
                    if self._paced(i, j):
                        # a class-b traversal keeps its first link busy for b + 1 epochs
                        for b in self.pace_classes(i, j):
                            for l in range(b + 1):
                                for s, c in product(self.sources, self.chunks):
                                    var = self._pace(s, i, j, c, k - l).get(b)
                                    if var is not None:
                                        tmp.add(var)
                        continue
                    beta_num_back = self.get_beta_num_back(i, j)
                    if k - beta_num_back < 0:
                        continue
                    for l in range(beta_num_back + 1):
                        for s, c in product(self.sources, self.chunks):
                            if k - l >= 0:
                                tmp.add(self.flow[s][i][j][c][k - l])
                self.model.addConstr(
                    self.epoch_used[-1] * demand_total >= tmp, name=f'flows_sum_{k}')
                objective.add(self.epoch_used[-1])

        if objective_type in (ObjectiveType.TOTAL_DEMAND, ObjectiveType.TOTAL_DEMAND_MIN_SWITCH_HOPS):
            # Objective based on total demand: Σ max((Σtotal_demand_k - (Demand-1)), 0)
            # aux_var_obj1 = tmp = Σtotal_demand_k - (Demand-1)
            # aux_var_obj2 = max(aux_var_obj1, 0)
            for k in self.epochs:
                demand_total = 0
                tmp = gp.LinExpr(0.0)
                for s, d, c in product(self.nodes, self.nodes, self.chunks):
                    if not self.demand[s][d][c]:
                        continue
                    demand_total += 1
                    tmp.add(self.total_demand_sat[s][d][c][k])
                self.aux_var.append(self.model.addVar(
                    -demand_total, 1, vtype=GRB.CONTINUOUS, name='aux_var_obj1_%d_%d_%d_%d' % (s, d, c, k)))
                demand_total = demand_total - 1
                tmp.add(demand_total, -1)
                self.model.addConstr(self.aux_var[len(
                    self.aux_var) - 1] == tmp, name="obj1_s_%d_i_%d_c_%d_k_%d" % (s, d, c, k))
                self.aux_var.append(self.model.addVar(
                    0, 1, vtype=GRB.CONTINUOUS, name='aux_var_obj2_%d_%d_%d_%d' % (s, d, c, k)))
                self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(
                    [self.aux_var[len(self.aux_var) - 2], 0]), name="obj2_s_%d_i_%d_c_%d_k_%d" % (s, d, c, k))
                objective.add(
                    self.aux_var[len(self.aux_var) - 1], (-mulitplier))

        if objective_type == ObjectiveType.TOTAL_DEMAND_MIN_SWITCH_HOPS:
            # Tie-breaker layered on the TOTAL_DEMAND reward above: a tiny positive cost on
            # every unit of flow crossing a switch->switch link (leaf<->spine here). The
            # completion reward moves the objective by ~mulitplier per epoch, so with GAMMA
            # this small the solver never delays completion to save hops; it only decides
            # *among equal-makespan solutions*, where it now prefers fewer switch relays
            # (e.g. a direct leaf hop over a spine detour). A direct same-rail cross-node
            # path uses zero switch->switch links; a spine detour uses two, so it is strictly
            # penalized. Only the handful of leaf-spine links carry a term, keeping the added
            # objective size small.
            GAMMA = 1e-4
            switch_switch_links = [
                (i, j)
                for i in self.topology.switch_indices
                for j in self.topology.switch_indices
                if self.topology.capacity[i][j] > 0
            ]
            for s, c, k in product(self.sources, self.chunks, self.epochs):
                for (i, j) in switch_switch_links:
                    objective.add(self.flow[s][i][j][c][k], GAMMA)

        if objective_type == ObjectiveType.PAPER:
            # Objective based on incremental progress
            # sum_end = self.num_nodes * 2 + self.num_chunks + self.num_epochs
            for s, d, c, k in product(self.nodes, self.nodes, self.chunks, self.epochs):
                if not self.demand[s][d][c]:
                    continue
                # d + 1 to create priority
                # my_contribution = 1 + ((s + d + c + k) / sum_end)
                # objective.add(self.total_demand_sat[
                #     s][d][c][k], -mulitplier * my_contribution * (self.user_input.instance.epsilon / (k + 1)))
                objective.add(self.total_demand_sat[s][d][c][k], -mulitplier)
                """
                  To disallow unnecessary flows, we can give a positive contribution to the objective for each flow.
                  This however affects the solver time significantly and unnecessary flows are pruned in post processing.
                """
                # for i in self.nodes:
                #     if self.topology.capacity[i][d] <= 0:
                #         continue
                #     # my_gamma_cont = (s + d + c + k + i) / (sum_end + self.num_nodes)
                #     objective.add(self.flow[s][i][d][c][k], mulitplier * 0.2)
        logging.debug(f'Finished adding objective in {time.time() - start}')
        return objective

    def hierarchical_objective_tiers(self) -> List[Tuple[str, gp.LinExpr, float]]:
        """
        The tiers of ObjectiveType.LEXICOGRAPHIC (highest priority first), the MILP
        counterpart of LPFormulation.hierarchical_objective_tiers -- same quantities, summed
        over the extra chunk index this formulation carries:

          tier 0, makespan:     the epoch by which every demand is met (makespan_tier). Without
                                it the latency tier alone, a SUM over demands, can tie a schedule
                                that finishes an epoch later -- some demands land earlier, a relay
                                lands later -- whenever the horizon leaves room.
          tier 1, latency:      -sum over (s, d, c, k) of total_demand_sat, i.e. every chunk
                                pays one unit for every epoch it has not landed yet.
          tier 2, host relay:   flow leaving a host that did not originate there (s != n).
          tier 3, switch chain: flow entering a switch, one unit per hop, so the term is the
                                length of the switch chain a unit traverses: 0 for a direct
                                link, 1 for gpu->switch->gpu, 3 for leaf->spine->leaf.

        Only variables that were actually created are read; the sparse flow container returns
        0.0 for the (s, i, j, c, k) combinations initialize_variables skipped, which add into
        a LinExpr as a constant exactly as the single-objective code above does.
        """
        instance = self.user_input.instance

        demanded = [(s, d, c) for s, d, c in product(self.sources, self.nodes, self.chunks)
                    if self.demand[s][d][c]]
        makespan = self.makespan_tier(
            lambda k: (self.total_demand_sat[s][d][c][k] for s, d, c in demanded))

        latency = gp.LinExpr(0.0)
        for s, d, c, k in product(self.sources, self.nodes, self.chunks, self.epochs):
            if not self.demand[s][d][c]:
                continue
            latency.add(self.total_demand_sat[s][d][c][k], -1.0)

        relay = gp.LinExpr(0.0)
        for n in self.host_indices():
            for j in self.nodes:
                if self.topology.capacity[n][j] <= 0:
                    continue
                for s in self.sources:
                    if s == n:
                        continue
                    for c, k in product(self.chunks, self.epochs):
                        relay.add(self.flow[s][n][j][c][k], 1.0)

        switch_chain = gp.LinExpr(0.0)
        for (i, j) in self.switch_ingress_links():
            for s, c, k in product(self.sources, self.chunks, self.epochs):
                switch_chain.add(self.flow[s][i][j][c][k], 1.0)

        return [
            ("makespan", makespan, instance.objective_makespan_rel_tol),
            ("latency", latency, instance.objective_latency_rel_tol),
            ("host_relay", relay, instance.objective_relay_rel_tol),
            ("switch_chain", switch_chain, 0.0),
        ]

    def encode_problem(self, use_one_less_epoch: bool = False, previous_buffers: List[List[int]] = []) -> int:
        setup_start = time.time()
        self.model = gp.Model(self.solver_name, env=get_gurobi_env())
        self.initialize_variables()
        self.destination_constraints()
        self.node_constraints(previous_buffers)
        self.forward_dwell_constraints()
        self.capacity_constraints()
        self.switch_acyclicity_constraints()
        if use_one_less_epoch:
            self.use_one_less_epoch()
        if self.user_input.instance.symmetry:
            self.add_symmetry_constraints()
        self.apply_objective()

        log_file = f'Logs/{self.solver_name}_{self.user_input.topology.name}_{self.num_nodes}-nodes_' \
            f'{self.num_chunks}-chunks_{self.num_epochs}-epochs_{self.epoch_duration}-epochduration_{self.user_input.instance.objective_type}'

        if self.user_input.gurobi.output_flag == 1 or self.user_input.instance.debug:
            if self.user_input.gurobi.log_file:
                log_file += self.user_input.gurobi.log_file
            self.model.setParam("LogFile", log_file + ".log")
            self.model.Params.LogToConsole = 0
            # if self.user_input.instance.debug:
            #     self.model.write(log_file + ".lp")

        self.set_gurobi_params()
        setup_end = time.time()

        logging.debug(f'Total time for setup {setup_end - setup_start}')
        logging.debug(f'Epoch duration {self.epoch_duration}')
        logging.debug(f'Starting model optimization {log_file}')

        # warm start option
        # self.model.update()
        # error = self.model.read("<path to .sol file>")

        if self.user_input.instance.warmstart and self.user_input.instance.solution_method == SolutionMethod.ONE_SHOT:
            self.model.update()
            self.model.read(self.user_input.instance.warmstart)

        solve_start = time.time()
        self.model.optimize()
        solve_end = time.time()
        logging.debug(
            f'Finished model optimization {log_file} in {solve_end - solve_start} ')

        if self.model.Status != GRB.OPTIMAL:
            logging.warning(
                f"Not_Optimal_{self.solver_name}_Status-{self.model.Status}_{self.user_input.topology.name}_"
                f"{self.num_nodes}-nodes_{self.num_chunks}-chunks_{self.num_epochs}-epochs_{self.epoch_duration}-epochduration")
            if self.user_input.instance.debug and self.model.SolCount > 0:
                logging.debug(
                    f"Epoch at the end of which all demands are satisfied: {self.find_demand_satisfied_k() + 1}")
                # self.model.write(log_file + '.sol')
            # compute an Irreducible Inconsistent Subsystem
            # https://www.gurobi.com/documentation/10.0/refman/py_model_computeiis.html
            # else:
            #     self.model.computeIIS()
            #     self.model.write(log_file + '_unsat.ilp')
            return self.model.Status

        if self.user_input.instance.debug:
            # self.model.write(log_file + '.sol')
            logging.debug(
                f"Epoch at the end of which all demands are satisfied: {self.find_demand_satisfied_k() + 1}")
        return self.model.Status

    def get_flows_buffer_demand(self) -> Tuple[List[Tuple[int, int, int, int, int]], Dict[Tuple[int, int, int], int], Dict[Tuple[int, int, int], int], int]:
        """
            After the model is solved, this function extracts the non-zero flows, non-zero buffers,
                the epoch at which each demand is met and the epoch in which final demand is met.
        """
        demand_met_epoch = {}
        list_of_sends = []
        buffers = {}
        for v in self.model.getVars():
            # Gurobi is not setting flows to be perfectly zero, so we check to ensure it is over 0.9.
            if 'flow' in v.varName and v.x > 0.9:
                if 'future' in v.varName:
                    continue
                components = v.varName.split('_')
                _, s_t, i, j, c, k = components
                list_of_sends.append(
                    (int(s_t), int(i), int(j), int(c), int(k)))
            if 'total_demand_' in v.varName and v.x > 0.9:
                components = v.varName.split('_')
                _, _, s, i, c, k = components
                s, i, c = int(s), int(i), int(c)
                if (s, i, c) in demand_met_epoch:
                    if demand_met_epoch[(s, i, c)] > int(k):
                        demand_met_epoch[(s, i, c)] = int(k)
                else:
                    demand_met_epoch[(s, i, c)] = int(k)
            if 'buffer_' in v.varName and v.x > 0.9:
                if 'buffer_ahead_' in v.varName:
                    continue
                components = v.varName.split('_')
                _, s, i, c, k = components
                s, i, c = int(s), int(i), int(c)
                if (s, i, c) in buffers:
                    if buffers[(s, i, c)] > int(k):
                        buffers[(s, i, c)] = int(k)
                else:
                    buffers[(s, i, c)] = int(k)
        correct_k = max(demand_met_epoch.values())
        return (list_of_sends, demand_met_epoch, buffers, correct_k)

    def get_viable_predecessors(self, flows: Set[Tuple[int, int, int, int, int]], s: int, d: int, c: int, k: int, pred_memo: Dict, pace: int = None) -> List[Tuple[Tuple[int, int, int, int, int], int]]:
        """
            Return every (flow, pace class) that could have delivered chunk (s, c) to node d by
            epoch k,
            i.e. all (s, n, d, c, k') in `flows` where n->d with the epoch offset implied by
            the link type (propagation alpha, per-hop beta, and cut-through switch legs).
            Sorted by send epoch (earliest first) so the DFS in _dfs_trace_to_source tries
            the earliest-arriving predecessor first (mirrors the old greedy min-epoch pick).

            For a switch d on a cut-through traversal, k is one before the epoch d forwards in, and
            `pace` is the class it forwards: only a same-class send reaching d in that epoch fed it.

            Memoized on (s, d, c, k, pace): the result is a pure function of `flows` and the query,
            independent of any back-trace path, so caching is sound. Which inputs the forest
            still has free is decided AFTER this lookup (in _dfs_trace_to_source), never by
            mutating `flows`, so the memo never goes stale.
        """
        key = (s, d, c, k, pace)
        cached = pred_memo.get(key)
        if cached is not None:
            return cached
        viable_flows = []
        # Loop over the neighbors to find every feasible incoming flow.
        for n in self.nodes:
            if self.topology.capacity[n][d] <= 0:
                continue
            alpha_num_back = self.get_alpha_num_back(n, d)
            if not self._paced(n, d):
                beta_num_back = self.get_beta_num_back(n, d)
                expected_flow = (s, n, d, c, k - alpha_num_back - beta_num_back)
                if expected_flow in flows:
                    viable_flows.append((expected_flow, beta_num_back))
            elif d in self._switches:
                # same-class send reaching the switch in the epoch it forwards (k + 1)
                expected_flow = (s, n, d, c, k + 1 - alpha_num_back)
                if expected_flow in flows and pace in self.flow_classes(*expected_flow):
                    viable_flows.append((expected_flow, pace))
            else:
                # end of a traversal: a class-b send finished landing in epoch k
                for b in self.pace_classes(n, d):
                    expected_flow = (s, n, d, c, k - alpha_num_back - b)
                    if expected_flow in flows and b in self.flow_classes(*expected_flow):
                        viable_flows.append((expected_flow, b))
        viable_flows.sort(key=lambda x: x[0][4])
        pred_memo[key] = viable_flows
        return viable_flows

    def stream_copies(self, f: Tuple[int, int, int, int, int], b: int) -> int:
        """How many class-b copies the solved send f carries. A flow var counts copies (pace
        classes made it integer), and a send split over several classes counts each in its own
        pace var; a send with one live class keeps its count in the flow var itself."""
        classes = self._pace(*f) if self._paced(f[1], f[2]) else {}
        if len(classes) > 1:
            var = classes.get(b)
            return int(round(var.X)) if var is not None else 0
        return self.flow_copies(*f)

    def _dfs_trace_to_source(self, flows: Set[Tuple[int, int, int, int, int]], s: int, node: int, c: int, k: int,
                             visited: frozenset, pred_memo: Dict, buffers: Dict, budget: List[int],
                             forest: SimpleNamespace, out_flow: Tuple[int, int, int, int, int] = None,
                             out_pace: int = None):
        """
            Trace how `node` received chunk (s, c) by epoch k back to what is already known --
            the source, or the forest of streams committed for earlier demands -- returning
            (hops, anchor), or None if there is no acyclic route.

              hops:   [(flow, pace class), ...] the NEW streams, destination-first;
              anchor: the committed stream the last of them continues (a switch input it is copied
                      from, or the delivery of the GPU it leaves), None when it leaves the source.

            A stream is one physical transmission, `(flow, class, copy)`: a flow var counts
            copies (stream_copies), so one flow tuple can be several. `out_pace` is the class of
            `out_flow`, the switch output this call traces the input of: a switch forwards in the
            class it received. At a switch the forest decides which inputs are free:
              * switch copy: an input already on the forest is JOINED -- the switch replicates it,
                so a second output reuses it rather than tracing another transmission;
              * no copy: each copy of an input feeds one output, so an input whose copies are all
                committed is not a candidate (this replaced the per-flow `claims` map).
            A GPU already on the forest is a join too: it holds the chunk, and the stream that
            delivered it is the parent of whatever it sends.

            This is a depth-first search WITH BACKTRACKING: when a switch has several viable
            predecessors (common under switch_copy) the greedy min-epoch pick may dead-end at a
            cut-through top-switch mesh even though a real predecessor reaches the source; on a
            dead end we try the next predecessor instead of dropping the whole demand. Committed
            inputs are tried first, so the forest shares as much as the solution allows.

            `visited` holds the (node, send_epoch) vertices already on the current path. In
            the time-expanded graph GPU forwarding strictly advances the epoch, so the only
            cycles are same-epoch cut-through loops; excluding a predecessor whose
            (node, send_epoch) is already visited breaks exactly those zero-time loops while
            still allowing a node revisited at a DIFFERENT epoch (not a cycle). This also
            bounds recursion: every step consumes a distinct (node, epoch), so depth is at
            most num_nodes * num_epochs; `budget` caps total node expansions to keep a
            pathological (exponential) search from running away.
        """
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        copy = self.switch_copy
        pace = out_pace if out_flow is not None and self._paced(out_flow[1], out_flow[2]) else None
        candidates = self.get_viable_predecessors(flows, s, node, c, k, pred_memo, pace)
        if out_flow is not None and copy:
            candidates = sorted(candidates, key=lambda fb: forest.used.get(fb, 0) == 0)
        for f, f_pace in candidates:
            pred = f[1]
            if (pred, f[4]) in visited:
                # picking f would revisit a same-epoch vertex -> a zero-time cycle; skip it.
                continue
            if out_flow is not None:
                used = forest.used.get((f, f_pace), 0)
                if copy and used:
                    return [], (f, f_pace, 0)
                if not copy and used >= self.stream_copies(f, f_pace):
                    continue
            if pred == s:
                # reached the chunk source node.
                return [(f, f_pace)], None
            if pred not in self.topology.switch_indices:
                held = forest.delivered.get((s, c, pred))
                if held is not None:
                    return [(f, f_pace)], held
                # A GPU relays out of its buffer; the epoch it received the chunk is given by
                # buffer. Without a buffer var this branch can't be continued -> try the next.
                if (s, pred, c) not in buffers:
                    continue
                nk = buffers[(s, pred, c)] - 1
            else:
                # A switch's upstream arrival epoch is one before its own outgoing send epoch;
                # get_viable_predecessors resolves the exact offset per incoming link type.
                nk = f[4] - 1
            sub = self._dfs_trace_to_source(
                flows, s, pred, c, nk, visited | {(pred, f[4])}, pred_memo, buffers, budget, forest,
                out_flow=f if pred in self.topology.switch_indices else None, out_pace=f_pace)
            if sub is not None:
                return [(f, f_pace)] + sub[0], sub[1]
        return None

    def _commit(self, forest: SimpleNamespace, s: int, c: int, hops: List, anchor) -> None:
        """Add a traced path's new streams to the forest, source side first."""
        parent = anchor
        for f, b in reversed(hops):
            stream = (f, b, forest.used.get((f, b), 0))
            forest.used[(f, b)] = stream[2] + 1
            forest.parent[stream] = parent
            if parent is not None:
                forest.children[parent].append(stream)
            if f[2] not in self.topology.switch_indices:
                forest.delivered[(s, c, f[2])] = stream
            parent = stream

    def _sends_of_forest(self, forest: SimpleNamespace) -> List[Send]:
        """One Send per committed stream leaving a non-switch node, with every stream it reaches
        through switches as its tree. Under cut-through a switch forwards in the class it received,
        so the whole tree is one class and one span; a store-and-forward fabric (switch_pipeline
        off) holds each hop at its own beta and the send is charged its slowest."""
        sends = []
        for head in forest.parent:
            f, b, _ = head
            if f[1] in self.topology.switch_indices:
                continue
            branches, classes = [], set()
            stack = [(head, ())]
            while stack:
                stream, via = stack.pop()
                g, gb, _n = stream
                classes.add(gb)
                if g[2] in self.topology.switch_indices:
                    stack.extend((child, via + (g[2],)) for child in forest.children[stream])
                else:
                    branches.append(Branch(g[2], via, g[4] + self.hop_timing(g[1], g[2], gb)[0]))
            if self.user_input.instance.switch_pipeline and len(classes) != 1:
                raise AssertionError(f"send {head} spans pace classes {classes}; a cut-through "
                                     f"traversal keeps one class end to end")
            sends.append(Send(identity=(f[0], f[3]), sender=f[1], epoch=f[4],
                              fanout=tuple(branches), span=max(classes) + 1))
        sends.sort(key=lambda x: (x.epoch, x.sender, x.identity, x.fanout))
        return sends

    def chunk_flow_path_to_string(self, chunk_path: List[Tuple[int, int, int, int, int]]) -> Tuple[int, List[str]]:
        """
            Convert the input flow path to a string merging the flows that go through the switches.
        """
        path = []
        start = 0
        next = 0
        chunk_path = chunk_path[::-1]
        while start < len(chunk_path):
            accumulated_flows = [chunk_path[start]]
            # Find the flows that goes through the switches.
            while chunk_path[next][2] in self.topology.switch_indices:
                next += 1
                accumulated_flows.append(chunk_path[next])
            start_node = accumulated_flows[0][1]
            end_node = accumulated_flows[-1][2]
            sending_epoch = accumulated_flows[0][4]
            switches = " -> ".join([str(x[2]) for x in accumulated_flows[:-1]])
            if switches:
                path.append(
                    (sending_epoch, f'{start_node}->{end_node} in epoch {sending_epoch} via switches {switches}'))
            else:
                path.append(
                    (sending_epoch, f'{start_node}->{end_node} in epoch {sending_epoch}'))
            start = next = next + 1
        return path

    def dfs_remove_unnecessary_flows(self, astar: bool) -> Tuple[List[Tuple[int, int, int, int, int]], Dict]:
        """
            This function removes the unnecessary flows from the list of flows.
            Traces the path of the chunk from the destination to the source accumulating the flows required to satisfy the demand.
            All the flows that are not required are removed.
        """
        flows, demand_met_epoch, buffers, _ = self.get_flows_buffer_demand()
        logging.debug(f'Number of flows before pruning: {len(flows)}')
        if astar:
            # In an A* round, we can't say whether a flow is required or not until the end of all the rounds as
            #  the node receiving the chunk may be an intermediate node in the path to the destination in the later rounds.
            flows.sort(key=lambda x: x[4])
            flows_str_info = {}
            flows_str_info["1-Epoch_Duration"] = self.epoch_duration
            flows_str_info["2-Expected_Epoch_Duration"] = self.expected_epoch_duration
            flows_str_info["3-Epochs_Required"] = self.find_demand_satisfied_k() + 1
            flows_str_info["4-Collective_Finish_Time"] = flows_str_info["1-Epoch_Duration"] * flows_str_info["3-Epochs_Required"]
            # node_per_chassis * chassis assumes every chassis contributes the same GPU
            # count; on heterogeneous topologies (e.g. HeteroTaperedCluster) that undercounts
            # the real number of demand-bearing sources. Count actual sources from self.demand
            # instead -- correct for any topology, and unaffected by AStar's self.sources
            # override (self.demand itself is never overridden), so it holds here even though
            # this branch runs for AStarFormulation instances too.
            num_data_sources = sum(1 for s in self.nodes if self.demand[s].any())
            flows_str_info["5-Algo_Bandwidth"] = num_data_sources * self.topology.chunk_size * self.num_chunks / flows_str_info["4-Collective_Finish_Time"]
            flows_str_info['7-Flows'] = [
                f"Chunk {c} from {s} traveled over {i}->{j} in epoch {k}" for s, i, j, c, k in flows]
            return flows, flows_str_info
        required_flows = set()
        required_flows_str = set()
        flows = set(flows)
        chunk_paths = {}
        demand_met_str = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float)))
        # (s, d, c, k) -> [(s, i, d, c, k'), ...]. Memoizes the viable incoming flows that
        # could have delivered chunk (s, c) to d by epoch k (a pure function of `flows`).
        pred_memo = {}
        # Every demand's trace is committed into ONE forest of streams (stream = (flow, pace class,
        # copy), one physical transmission), so a later trace joins what is already there instead
        # of describing the same transmission again: a GPU that holds the chunk, or -- under
        # switch copy -- a switch input that is already being replicated.
        #   parent:    stream -> the stream it continues (None when it leaves the source)
        #   children:  the inverse, for grouping the forest into sends
        #   used:      (flow, class) -> streams of it committed so far
        #   delivered: (s, c, non-switch node) -> the stream that lands the chunk there
        forest = SimpleNamespace(parent={}, children=defaultdict(list), used={}, delivered={})
        # The traced paths as flow tuples, keyed like the demand, and the demands no acyclic path
        # was found for. The paths are a per-demand VIEW of the forest (walk a demand's delivery
        # up to the source); `self.sends` below is the forest itself.
        self.per_demand_paths = {}
        # (s, d, c) -> the pace class of each hop of per_demand_paths[(s, d, c)], aligned with it
        self.per_demand_classes = {}
        self.untraced_demands = []
        for s, d, c in product(self.nodes, self.nodes, self.chunks):
            if not self.demand[s][d][c]:
                # If the node d did not request the chunk s c, then we can skip it.
                # If d is an intermediate node in the path to the destination, then the flows to d are
                #   accounted when we consider the flows to the destination.
                continue
            if (s, d, c) not in demand_met_epoch:
                logging.warning(f"Demand not satisfied for s_{s}, d_{d}, c_{c}")
                continue
            demand_met_str[f"GPU {d}"][f"GPU {s}"][f"Chunk {c}"] = self.epoch_duration * (
                demand_met_epoch[(s, d, c)] + 1)
            # We use demand_met as the last step may not have a buffer variable.
            demand_met_k = demand_met_epoch[(s, d, c)]
            # Trace the chunk's delivery back to the source with a backtracking DFS. It follows
            # only real hops and, keying its visited set on (node, epoch), never follows a
            # same-epoch cut-through cycle backedge -- that edge, never traced, is then dropped
            # as an unnecessary flow (the whole point of this pass). Backtracking matters under
            # switch_copy: a switch (esp. a cut-through top-switch mesh) can have several viable
            # predecessors, and the earliest one may dead-end while another reaches the source;
            # on a dead end the DFS backs up and tries the next instead of dropping the demand.
            # A budget bounds the (worst-case exponential) search. This supersedes the earlier
            # unsound whole-cycle removal: with binary flows a legit forward edge and a wasteful
            # reverse edge form one 2-cycle, so removing the cycle would delete the legit edge.
            if (s, c, d) not in forest.delivered:
                budget = [50 * len(self.nodes) * (self.num_epochs + 1)]
                traced = self._dfs_trace_to_source(
                    flows, s, d, c, demand_met_k, frozenset(), pred_memo, buffers, budget, forest)
                if traced is None:
                    # No acyclic route to source: the solver delivers this demand only via a
                    # same-epoch circulation (a phantom cut-through switch cycle -- a formulation
                    # artifact), or the search budget was exhausted. Skip this demand instead of
                    # hanging/crashing; flows collected for other demands are unaffected.
                    logging.warning(
                        f"Could not trace demand s_{s} d_{d} c_{c} back to source "
                        f"(no acyclic path -- likely a same-epoch switch circulation); skipping.")
                    self.untraced_demands.append((s, d, c))
                    continue
                self._commit(forest, s, c, *traced)
            my_path, classes = [], []
            stream = forest.delivered[(s, c, d)]
            while stream is not None:
                my_path.append(stream[0])
                classes.append(stream[1])
                stream = forest.parent[stream]
            required_flows.update(my_path)
            self.per_demand_paths[(s, d, c)] = my_path
            self.per_demand_classes[(s, d, c)] = classes
            chunk_str_path = self.chunk_flow_path_to_string(my_path)
            chunk_paths[f"Demand at {d} for chunk {c} from {s} met by epoch {demand_met_k}"] = [
                x[1] for x in chunk_str_path]
            for epoch, cpath in chunk_str_path:
                required_flows_str.add(
                    (epoch, f"Chunk {c} from {s} traveled over {cpath}"))

        # The authoritative result: one Send per physical transmission (teccl/sends.py).
        self.sends = self._sends_of_forest(forest)
        logging.debug(f'Number of flows after pruning: {len(required_flows)}')
        required_flows = list(required_flows)
        required_flows.sort(key=lambda x: x[4])
        required_flows_str = list(required_flows_str)
        required_flows_str.sort(key=lambda x: x[0])
        flows_str_info = {}
        # The collective this MILP actually solved (allgather or alltoall), so ncclize builds the
        # right collective instead of assuming allgather; "0-Formulation" is emitted alongside it
        # because the two are independent axes downstream (teccl_ncclize.is_lp_format) and the
        # structural nested-vs-flat sniff is only a fallback for schedules predating the field.
        flows_str_info["0-Collective"] = self.collective_name
        flows_str_info["0-Formulation"] = "MILP"
        if self.user_input.instance.collective in (Collective.GATHER, Collective.BROADCAST):
            flows_str_info["0-Root"] = self.user_input.instance.root
        flows_str_info["1-Epoch_Duration"] = self.epoch_duration
        flows_str_info["2-Expected_Epoch_Duration"] = self.expected_epoch_duration
        flows_str_info["3-Epochs_Required"] = self.find_demand_satisfied_k() + 1
        flows_str_info["4-Collective_Finish_Time"] = flows_str_info["1-Epoch_Duration"] * flows_str_info["3-Epochs_Required"]
        # See the astar branch above: node_per_chassis * chassis assumes uniform GPUs-per-
        # chassis, which breaks on heterogeneous topologies. len(self.sources) is exactly the
        # demand-bearing source count here (self.sources is not overridden on this path).
        flows_str_info["5-Algo_Bandwidth"] = len(self.sources) * self.topology.chunk_size * self.num_chunks / flows_str_info["4-Collective_Finish_Time"]
        flows_str_info["6-Demand_Met"] = demand_met_str
        flows_str_info['7-Flows'] = [x[1] for x in required_flows_str]
        flows_str_info['8-Chunk paths'] = chunk_paths
        flows_str_info["9-Chunk_Size"] = self.topology.chunk_size
        # The transmissions themselves (teccl_ncclize.parse_sends): a switch copy is ONE record
        # with a branch per receiver, and each send is paced by its own pace class -- held for
        # `span` epochs at chunk / (span * epoch) -- rather than at one rate for every op.
        flows_str_info["10-Sends"] = [
            {"chunk": x.identity[1], "from": x.identity[0], "sender": x.sender, "epoch": x.epoch,
             "span": x.span, "rate": self.topology.chunk_size / (x.span * self.epoch_duration),
             "volume": x.volume,
             "branches": [{"to": b.receiver, "via": list(b.via), "ready": b.ready}
                          for b in x.fanout]}
            for x in self.sends]
        return required_flows, flows_str_info

    def find_demand_satisfied_k(self):
        satisfied_epochs = {}
        for v in self.model.getVars():
            if 'total_demand_' in v.varName and v.x > 0.0:
                components = v.varName.split('_')
                _, _, s, i, c, k = components
                if (s, i, c) in satisfied_epochs:
                    if satisfied_epochs[(s, i, c)] > int(k):
                        satisfied_epochs[(s, i, c)] = int(k)
                else:
                    satisfied_epochs[(s, i, c)] = int(k)
        for s, i, c in product(self.nodes, self.nodes, self.chunks):
            if self.demand[s][i][c]:
                if (str(s), str(i), str(c)) not in satisfied_epochs:
                    logging.warning(
                        f"Demand not satisfied for s_{s}, i_{i}, c_{c}")
        return max(satisfied_epochs.values())

    def diagnose_switch_phantom_cycles(self) -> None:
        """
            Post-solve diagnostic that PROVES/locates same-epoch switch phantom cycles in the
            raw solved flow (not the pruned schedule). For each (source, chunk, epoch) it finds
            switch<->switch links carrying the chunk BOTH directions in the same epoch
            (flow[s][a][b][c][k]==1 AND flow[s][b][a][c][k]==1) -- a circulation the switch
            conservation constraint (out==in) satisfies with no GPU grounding -- and, for each
            such source, prints whether/when the chunk ever actually entered the switch fabric
            via a GPU->switch egress. "NO GPU->switch egress" is the smoking gun: the mesh flow
            for that source is conjured out of nothing. Reads model vars only; safe to call
            once a solution exists.
        """
        switches = set(self.topology.switch_indices)
        gpu_to_switch = defaultdict(list)          # (s,c) -> [(gpu, switch, k)]
        ss_edges = defaultdict(set)                # (s,c,k) -> {(a,b) switch->switch}
        for v in self.model.getVars():
            if not v.varName.startswith('flow_') or 'future' in v.varName or v.x <= 0.9:
                continue
            parts = v.varName.split('_')
            if len(parts) != 6:
                continue
            _, s, i, j, c, k = (int(p) if idx else p for idx, p in enumerate(parts))
            if i not in switches and j in switches:
                gpu_to_switch[(s, c)].append((i, j, k))
            elif i in switches and j in switches:
                ss_edges[(s, c, k)].add((i, j))

        phantom_sources = set()
        cycle_count = 0
        for (s, c, k), edges in sorted(ss_edges.items()):
            for (a, b) in edges:
                if a < b and (b, a) in edges:
                    cycle_count += 1
                    phantom_sources.add((s, c))
                    logging.warning(
                        f"[PHANTOM] same-epoch switch cycle: source {s} chunk {c} epoch {k}: "
                        f"flow[{a}->{b}]=1 AND flow[{b}->{a}]=1 (conservation satisfied circularly)")
        for (s, c) in sorted(phantom_sources):
            egress = sorted(gpu_to_switch.get((s, c), []), key=lambda x: x[2])
            if egress:
                logging.warning(
                    f"[PHANTOM] source {s} chunk {c} GPU->switch egress (only real way into the mesh): "
                    + ", ".join(f"{g}->{sw}@ep{k}" for (g, sw, k) in egress))
            else:
                logging.warning(
                    f"[PHANTOM] source {s} chunk {c}: NO GPU->switch egress ANYWHERE -> the mesh flow "
                    f"is pure circulation, conjured with no real origin at the source GPU")
        logging.warning(
            f"[PHANTOM] summary: {cycle_count} same-epoch switch 2-cycle(s) across "
            f"{len(phantom_sources)} (source,chunk) pair(s)")

    def get_schedule(self) -> Tuple[List[Tuple[int, int, int, int, int]], Dict]:
        if self.model.SolCount > 0:
            if self.user_input.instance.debug:
                # Phantom cycles are made infeasible by switch_acyclicity_constraints when
                # switch_pipeline+switch_copy are on, so this is normally a no-op; run it only
                # under debug to confirm/locate any (e.g. if that constraint is disabled).
                self.diagnose_switch_phantom_cycles()
            if not self.required_flows:
                self.required_flows, self.flows_str_info = self.dfs_remove_unnecessary_flows(
                    astar=False)
            return self.required_flows, self.flows_str_info
        else:
            return [], {}

    def write_schedule_to_file(self, schedule: List[Tuple[int, int, int, int, int]], filename: str):
        with open(filename, 'w') as f:
            for s, i, j, c, k in schedule:
                f.write(
                    f'Chunk {c} from {s} traveled over {i}->{j} at epoch {k}\n')
