from collections import defaultdict
import logging
import math
import time
from itertools import product
from typing import Dict, List, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from teccl.gurobi_env import get_gurobi_env
from teccl.input_data import *
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
        self.solver_name = f"{user_input.instance.collective.name.title()}_MILP"
        self.required_flows = []
        self.flows_str_info = {}

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

        for i, j, in product(self.nodes, self.nodes):
            if self.topology.capacity[i][j] <= 0:
                continue
            alpha_num_back = self.get_alpha_num_back(i, j)
            link_type = self.get_link_type(i, j)
            beta_num_back = self.get_beta_num_back(i, j)
            extra_epochs = alpha_num_back + beta_num_back
            if link_type == self.LinkType.GPU_SWITCH and self.user_input.instance.switch_pipeline:
                # For GPU-Switch link, we don't need to send the chunk in the one more extra epoch too as the switch to GPU link is instantaneous and we need another epoch to account for this case.
                extra_epochs += 1
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
                        alpha_num_back = self.get_alpha_num_back(i, d)
                        link_type = self.get_link_type(i, d)
                        if link_type != self.LinkType.SWITCH_GPU or not self.user_input.instance.switch_pipeline:
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

    def hop_timing(self, j: int, i: int) -> Tuple[int, int]:
        """
            (landing offset, occupancy) of one send on link (j, i), for consumers that read the
            schedule back (the hierarchical solver's tree extraction).

            landing offset: a send in epoch k is in non-switch node i's buffer from epoch k + offset
                -- the inverse of _arrival_flow, so the two cannot disagree. The epoch during which
                the bytes LAND is therefore k + offset - 1.
            occupancy: how many epochs the send holds the link, beta + 1 (get_beta_num_back).
        """
        alpha_num_back = self.get_alpha_num_back(j, i)
        beta_num_back = self.get_beta_num_back(j, i)
        if self.get_link_type(j, i) != self.LinkType.SWITCH_GPU or not self.user_input.instance.switch_pipeline:
            return alpha_num_back + 1 + beta_num_back, beta_num_back + 1
        return alpha_num_back, beta_num_back + 1

    def _arrival_flow(self, s: int, j: int, i: int, c: int, k: int):
        """
            The flow on link (j, i) that lands chunk (s, c) in non-switch node i's buffer at the
            BEGINNING of epoch k, or None if no send on that link can land by then. This is the
            per-in-link arrival term of the buffer recurrence
                buffer[k] = buffer[k-1] + sum_j arrival(j -> i, k)
            factored out so forward_dwell_constraints reads the identical term.
        """
        alpha_num_back = self.get_alpha_num_back(j, i)
        link_type = self.get_link_type(j, i)
        if link_type != self.LinkType.SWITCH_GPU or not self.user_input.instance.switch_pipeline:
            beta_num_back = self.get_beta_num_back(j, i)
            if k - alpha_num_back - 1 - beta_num_back >= 0:
                # -1 is required as Buffer[i] stands for buffer at the "beginning" of epoch i
                return self.flow[s][j][i][c][k - alpha_num_back - 1 - beta_num_back]
            return None
                # switch second link is ignored to remove double accounting of transmission delay
                #
                # KNOWN LIMITATION (non-uniform link bandwidths + sub-chunk epoch). Dropping
                # beta_num_back here, and in the SWITCH_SWITCH branch of the switch send rule
                # below, charges the whole cut-through traversal the cost of its FIRST
                # GPU->switch hop: the credited delivery time is 1 + beta(first hop), whereas
                # the physical cut-through time is ceil(chunk / min bandwidth ALONG THE PATH)
                # = 1 + max(beta over the legs used). The two agree whenever every link
                # carries at least one whole chunk per epoch (beta == 0 everywhere), which is
                # what epoch_type SLOWEST_LINK gives you and what uniform-bandwidth
                # topologies give you for free. They diverge when a fast ingress port feeds
                # narrower fabric links AND the epoch is shorter than a chunk on those links:
                # the solve then reports a makespan SHORTER than any single link on its own
                # chosen path can deliver -- optimistic, not infeasible. Measured on
                # mini_1gpu_1nic at a 0.08 s epoch with 1 GB chunks (100Gbps NICs, 50Gbps
                # uplinks): 0.08 s reported, 0.16 s physical, 0.48 s for the same run with
                # switch_pipeline=False. Charging the max needs the path identity, which this
                # per-link flow formulation does not carry; charging the SUM (keep
                # beta_num_back in both pipelined branches) would be sound but conservative.
                # ACCEPTED as-is: the routing is unaffected, and the affected inputs
                # (teccl/examples/sample_inputs/mini_*_partial_*.json) say in their
                # _known_limitation that their MILP timings are not to be quoted.
        if k - alpha_num_back >= 0:
            return self.flow[s][j][i][c][k - alpha_num_back]
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
            self.aux_var.append(self.model.addVar(
                0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='aux_var_%d_%d_%d_%d' % (s, i, c, k)))
            self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(
                [self.flow[s][i][v][c][k] for v in range(self.num_nodes) if self.topology.capacity[i][v] > 0]), name="node_aux_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))

            node_constr = gp.LinExpr(0.0)
            node_constr.add(self.buffer[s][i][c][k])
            self.model.addConstr(node_constr >= self.aux_var[
                len(self.aux_var) - 1], name="node_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
        else:
            # Switch can send whatever it got in the last epochs and round
            switch_node_constr = gp.LinExpr(0.0)
            if k > 0:
                for j in range(self.num_nodes):
                    if self.topology.capacity[j][i] > 0:
                        alpha_num_back = self.get_alpha_num_back(j, i)
                        link_type = self.get_link_type(j, i)
                        if link_type == self.LinkType.SWITCH_SWITCH and self.user_input.instance.switch_pipeline:
                            # Chained switches act as a single cut-through fabric: the full
                            # store-and-forward cost was already paid once at the first
                            # GPU->switch hop, so a switch-to-switch hop only adds propagation delay.
                            # KNOWN LIMITATION: "paid once at the first hop" is the wrong quantity
                            # when link bandwidths differ -- the bottleneck leg, not the first leg,
                            # paces a cut-through traversal. See the long note on the SWITCH_GPU
                            # buffer credit in node_constraint_helper.
                            if k - alpha_num_back >= 0:
                                switch_node_constr.add(
                                    self.flow[s][j][i][c][k - alpha_num_back])
                        else:
                            beta_num_back = self.get_beta_num_back(j, i)
                            if k - alpha_num_back - 1 - beta_num_back >= 0:
                                switch_node_constr.add(
                                    self.flow[s][j][i][c][k - alpha_num_back - 1 - beta_num_back])

            if previous_round_buffers and k < len(previous_round_buffers[s][i][c]):
                switch_node_constr.add(previous_round_buffers[s][i][c][k])

            if not self.user_input.instance.switch_copy:
                '''
                    switch no-copy version. For every outgoing chunk there has to be a corresponding incoming chunk.
                '''
                switch_outgoing_links = gp.LinExpr(0.0)
                for v in range(self.num_nodes):
                    if self.topology.capacity[i][v] > 0:
                        switch_outgoing_links.add(self.flow[s][i][v][c][k])
                # We allow the switch to drop the chunks with ">=" as we found it to be faster.
                # If we use "==" then in allgather case, it will result in no unnecessary flows.
                self.model.addConstr(switch_node_constr >= switch_outgoing_links,
                                     name="switch_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
            else:
                '''
                    switch copy version. The switch can copy the incoming chunk to multiple outgoing links.
                '''
                self.aux_var.append(self.model.addVar(
                    0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='aux_var_%d_%d_%d_%d' % (s, i, c, k)))
                self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(
                    [self.flow[s][i][v][c][k] for v in range(self.num_nodes) if self.topology.capacity[i][v] > 0]), name="node_switch_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
                # We allow the switch to drop the chunks with ">=" as we found it to be faster.
                # If we use "==" then in allgather case, it will result in no unnecessary flows, which is necessary for the symmetric case.
                if self.user_input.instance.symmetry:
                    self.model.addConstr(switch_node_constr == self.aux_var[len(self.aux_var) - 1],
                                         name="switch_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
                else:
                    self.model.addConstr(switch_node_constr >= self.aux_var[len(self.aux_var) - 1],
                                         name="switch_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))

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
        if not (self.user_input.instance.switch_pipeline and self.user_input.instance.switch_copy):
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
                # f == 1  =>  u[a] - u[b] <= -1  (u[a] < u[b]); f == 0 leaves it slack.
                self.model.addConstr(
                    order_var(s, a, c, k) - order_var(s, b, c, k) + N * f <= N - 1,
                    name="switch_acyc_s_%d_a_%d_b_%d_c_%d_k_%d" % (s, a, b, c, k))
        logging.debug(
            f'Finished adding switch acyclicity constraints in {time.time() - start}')

    def use_one_less_epoch(self) -> None:
        """
           Constraints to find the solution using one less epoch than the input number of epochs.
           This is required for the case when the switch to gpu link is off (instantaneous transfer).
           We have flow_K to know which node n is receiving the chunk from the switch, but the demand at
           node n is taken to be satisifed in the previous epoch K-1.
           For the iterative search, the solver needs extra epochs to find the solution at K-1, but it can
           also find the solution at K epochs as we stop the search when the solution count is 1.
        """
        if not self.user_input.instance.switch_pipeline:
            return
        logging.debug(
            f'Adding demand sat for switch to gpu link constraints to limit max epochs to 1 less than input')
        start = time.time()
        self.aux_var.append(self.model.addVar(
            0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='aux_var_switch_gpu'))
        required_demand_sats = []
        for s, d, c in product(self.nodes, self.nodes, self.chunks):
            if not self.demand[s][d][c]:
                continue
            required_demand_sats.append(
                self.total_demand_sat[s][d][c][self.num_epochs - 2])

        self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.min_(
            required_demand_sats), name="min_total_demand_sat_switch_gpu")
        self.model.addConstr(1 == self.aux_var[len(
            self.aux_var) - 1], name="min_total_demand_sat_switch_gpu_1")
        logging.debug(
            f'Finished adding demand sat for switch to gpu link constraints in {time.time() - start}')

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
        The three tiers of ObjectiveType.LEXICOGRAPHIC (highest priority first), the MILP
        counterpart of LPFormulation.hierarchical_objective_tiers -- same three quantities,
        summed over the extra chunk index this formulation carries:

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

    def get_viable_predecessors(self, flows: Set[Tuple[int, int, int, int, int]], s: int, d: int, c: int, k: int, pred_memo: Dict) -> List[Tuple[int, int, int, int, int]]:
        """
            Return every flow that could have delivered chunk (s, c) to node d by epoch k,
            i.e. all (s, n, d, c, k') in `flows` where n->d with the epoch offset implied by
            the link type (propagation alpha, per-hop beta, and cut-through switch legs).
            Sorted by send epoch (earliest first) so the DFS in _dfs_trace_to_source tries
            the earliest-arriving predecessor first (mirrors the old greedy min-epoch pick).

            Memoized on (s, d, c, k): the result is a pure function of `flows` and the query,
            independent of any back-trace path, so caching is sound. The no-copy switch rule
            is applied AFTER this lookup (as `claims` in _dfs_trace_to_source), never by
            mutating `flows`, so the memo never goes stale.
        """
        key = (s, d, c, k)
        cached = pred_memo.get(key)
        if cached is not None:
            return cached
        viable_flows = []
        # Loop over the neighbors to find every feasible incoming flow.
        for n in self.nodes:
            if self.topology.capacity[n][d] <= 0:
                continue
            alpha_num_back = self.get_alpha_num_back(n, d)
            beta_num_back = -1
            link_type = self.get_link_type(n, d)
            cutthrough = (
                self.user_input.instance.switch_pipeline and
                link_type in (self.LinkType.SWITCH_GPU, self.LinkType.SWITCH_SWITCH)
            )
            if not cutthrough:
                beta_num_back = self.get_beta_num_back(n, d)
            expected_flow = (s, n, d, c, k - alpha_num_back - beta_num_back)
            if expected_flow in flows:
                viable_flows.append(expected_flow)
        viable_flows.sort(key=lambda x: x[4])
        pred_memo[key] = viable_flows
        return viable_flows

    def _dfs_trace_to_source(self, flows: Set[Tuple[int, int, int, int, int]], s: int, node: int, c: int, k: int,
                             visited: frozenset, pred_memo: Dict, buffers: Dict, budget: List[int],
                             claims: Dict = None, out_flow: Tuple[int, int, int, int, int] = None
                             ) -> List[Tuple[int, int, int, int, int]]:
        """
            Trace how `node` received chunk (s, c) by epoch k back to the source, returning
            the list of flows [flow_into_node, ..., flow_from_source] or None if there is no
            acyclic route. This is a depth-first search WITH BACKTRACKING: when a switch has
            several viable predecessors (common under switch_copy) the greedy min-epoch pick
            may dead-end at a cut-through top-switch mesh even though a real predecessor
            reaches the source; on a dead end we try the next predecessor instead of dropping
            the whole demand.

            `visited` holds the (node, send_epoch) vertices already on the current path. In
            the time-expanded graph GPU forwarding strictly advances the epoch, so the only
            cycles are same-epoch cut-through loops; excluding a predecessor whose
            (node, send_epoch) is already visited breaks exactly those zero-time loops while
            still allowing a node revisited at a DIFFERENT epoch (not a cycle). This also
            bounds recursion: every step consumes a distinct (node, epoch), so depth is at
            most num_nodes * num_epochs; `budget` caps total node expansions to keep a
            pathological (exponential) search from running away.

            `claims` (no-copy switches only) maps a flow INTO a switch to the one flow OUT of the
            switch it already feeds on a committed path; `out_flow` is the switch-outgoing flow
            this call is tracing for, set when `node` is a switch. An incoming flow claimed by a
            DIFFERENT outgoing flow is unavailable -- a no-copy switch cannot fan one copy out
            twice -- but one claimed by the SAME outgoing flow is a shared path prefix and is
            reused freely (e.g. A -> T0 -> B committed for B, then reused by C's path through B).
        """
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        for f in self.get_viable_predecessors(flows, s, node, c, k, pred_memo):
            pred = f[1]
            if (pred, f[4]) in visited:
                # picking f would revisit a same-epoch vertex -> a zero-time cycle; skip it.
                continue
            if claims is not None and out_flow is not None and claims.get(f, out_flow) != out_flow:
                # this switch's incoming copy already feeds a different outgoing hop.
                continue
            if pred == s:
                # reached the chunk source node.
                return [f]
            if pred not in self.topology.switch_indices:
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
                flows, s, pred, c, nk, visited | {(pred, f[4])}, pred_memo, buffers, budget,
                claims=claims,
                out_flow=f if pred in self.topology.switch_indices else None)
            if sub is not None:
                return [f] + sub
        return None

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
        # No-copy switches: flow into a switch -> the one flow out of it that it feeds on a
        # committed path (see _dfs_trace_to_source). None under switch_copy, where a switch may
        # fan a copy out to any number of receivers.
        claims = None if self.user_input.instance.switch_copy else {}
        # The traced paths as flow tuples, keyed like the demand, and the demands no acyclic path
        # was found for. Both are for consumers that need more than the strings below -- the
        # hierarchical solver reads the routing off the paths, and must treat an untraced demand
        # as a missing delivery rather than a warning.
        self.per_demand_paths = {}
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
            budget = [50 * len(self.nodes) * (self.num_epochs + 1)]
            my_path = self._dfs_trace_to_source(
                flows, s, d, c, demand_met_k, frozenset(), pred_memo, buffers, budget,
                claims=claims)
            if my_path is None:
                # No acyclic route to source: the solver delivers this demand only via a
                # same-epoch circulation (a phantom cut-through switch cycle -- a formulation
                # artifact), or the search budget was exhausted. Skip this demand instead of
                # hanging/crashing; flows collected for other demands are unaffected.
                logging.warning(
                    f"Could not trace demand s_{s} d_{d} c_{c} back to source "
                    f"(no acyclic path -- likely a same-epoch switch circulation); skipping.")
                self.untraced_demands.append((s, d, c))
                continue
            for f in my_path:
                required_flows.add(f)
            self.per_demand_paths[(s, d, c)] = list(my_path)
            # No-copy: a switch relays a single copy, so a committed path CLAIMS each flow into a
            # switch for the one outgoing flow it fed. Later traces may reuse the same in -> out
            # pair (a shared prefix, e.g. through a relay GPU that also wants the chunk) but not
            # branch the copy elsewhere. This used to delete the incoming flow outright, which
            # also forbade the shared prefix and silently dropped every demand routed through a
            # relay that had already been served. Claims do not change `flows`, so pred_memo
            # stays valid.
            if claims is not None:
                for out_f, in_f in zip(my_path, my_path[1:]):
                    if in_f[2] in self.topology.switch_indices:
                        claims.setdefault(in_f, out_f)
            chunk_str_path = self.chunk_flow_path_to_string(my_path)
            chunk_paths[f"Demand at {d} for chunk {c} from {s} met by epoch {demand_met_k}"] = [
                x[1] for x in chunk_str_path]
            for epoch, cpath in chunk_str_path:
                required_flows_str.add(
                    (epoch, f"Chunk {c} from {s} traveled over {cpath}"))

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
