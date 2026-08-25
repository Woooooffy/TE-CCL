import logging
import math
from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from teccl.gurobi_env import get_gurobi_env
from teccl.input_data import *
from teccl.solvers.demand import build_demand
from teccl.topologies.topology import Topology


class BaseFormulation(ABC):
    @abstractmethod
    def __init__(self, user_input: UserInputParams, topology: Topology) -> None:
        """
            Base class for all the solvers which creates the model and the variables common across solvers.
            Sets the epoch duration and the number of epochs using the user input and the topology.
        """
        self.user_input = user_input
        self.solver_name = ""
        self.topology = topology
        self.model = gp.Model('Base', env=get_gurobi_env())

        collective = user_input.instance.collective
        if collective in (Collective.GATHER, Collective.BROADCAST):
            self._validate_root(user_input.instance.root)

        # Hierarchical coarse solve: abstract() attaches a precomputed COARSE demand matrix
        # (coarsify_demand of the fine demand) to the coarse topology. When present, satisfy it
        # directly so the coarse solve is collective-agnostic -- it routes the aggregated
        # inter-cell volumes instead of regenerating a per-chunk demand for the collapsed graph.
        # Otherwise build the fine demand for the requested collective (single source of truth
        # in teccl.solvers.demand).
        demand_override = getattr(self.topology, "demand_override", None)
        if demand_override is not None:
            self.demand = np.asarray(demand_override, dtype=np.int32)
        else:
            self.demand = build_demand(
                collective, self.topology, user_input.instance.num_chunks,
                user_input.instance.root)

        self.num_nodes = len(self.topology.capacity)
        # num_chunks tracks the demand tensor's chunk axis exactly. For the ordinary path this
        # equals the requested num_chunks; an injected coarse demand uses a single weighted
        # slot, so num_chunks == 1 there.
        self.num_chunks = self.demand.shape[2]

        self.nodes = list(range(self.num_nodes))
        self.chunks = list(range(self.num_chunks))
        self.aux_var: List[dict] = []

        self.set_epoch_duration()
        self.set_num_epochs()

    def set_epoch_duration(self) -> None:
        """
            Sets the epoch duration based on the user input and the topology.
            If the user input is not -1, then the epoch duration is set to the user input.
            If the user input is -1, then the epoch duration is set to the fastest or slowest link in the topology,
                which means the time it takes for one chunk to go on the fastest or slowest link.
        """
        def alpha_check():
            """
                Checks if the alpha_epoch_duration_ratio_max is too large in which case the epoch duration is increased
                    to avoid large models. (Solution quality is not affected much as alpha dominates the collective finish time).           
            """
            min_alpha = self.topology.get_min_alpha()
            alpha_epoch_ratio = min_alpha / self.epoch_duration
            if alpha_epoch_ratio > self.user_input.instance.alpha_epoch_duration_ratio_max:
                logging.warning(f"alpha_epoch_ratio is too large. "
                                f"alpha_epoch_ratio = {alpha_epoch_ratio}. "
                                f"changing epoch_duration to alpha_epoch_ratio in the user input = {self.user_input.instance.alpha_epoch_duration_ratio_max}")
                self.epoch_duration = min_alpha / \
                    self.user_input.instance.alpha_epoch_duration_ratio_max
        # Both fields may be per-level lists; resolve_epoch_policy picks this level's entry (see
        # InstanceParams.level_depth). For a scalar -- every flat solve and every pre-existing
        # input file -- this returns the value unchanged, so nothing below this line changed.
        epoch_type, user_epoch_duration = resolve_epoch_policy(
            self.user_input.instance, self.user_input.instance.level_depth)
        if user_epoch_duration != -1:
            self.epoch_duration = user_epoch_duration
        # Read through the GETTERS, not the cached attributes. Topology.__init__ populates both
        # eagerly, so for a topology that is never re-expressed the two are the same value -- but a
        # level of the hierarchical solve rescales its capacity matrix after construction
        # (CoarseTopology.rescale_to_chunk), and only the getter recomputes. Reading the attribute
        # there yielded 0 and tripped the assert below with a message about the epoch multiplier,
        # which was not the problem.
        elif epoch_type == EpochType.FASTEST_LINK:
            self.epoch_duration = self.topology.get_epoch_duration_fast_link() * \
                self.user_input.instance.epoch_multiplier
            assert self.epoch_duration > 0, (
                f"non-positive epoch duration {self.epoch_duration}: fastest-link epoch is "
                f"{self.topology.get_epoch_duration_fast_link()} and epoch_multiplier is "
                f"{self.user_input.instance.epoch_multiplier}; both must be positive")
        elif epoch_type == EpochType.SLOWEST_LINK:
            self.epoch_duration = self.topology.get_epoch_duration_slow_link() * \
                self.user_input.instance.epoch_multiplier
            assert self.epoch_duration > 0, (
                f"non-positive epoch duration {self.epoch_duration}: slowest-link epoch is "
                f"{self.topology.get_epoch_duration_slow_link()} and epoch_multiplier is "
                f"{self.user_input.instance.epoch_multiplier}; both must be positive")
        else:
            raise ValueError(
                f"level_depth={self.user_input.instance.level_depth} resolved to epoch_type "
                f"{epoch_type} but its epoch_duration is -1 (unset). With the per-level list form, "
                f"the two must line up entry by entry: a level asking for USER_INPUT needs a real "
                f"duration at the SAME index of epoch_duration.")
        
        self.expected_epoch_duration = self.epoch_duration
        alpha_check()

    def set_num_epochs(self, epochs=100) -> None:
        if self.user_input.instance.num_epochs != -1:
            assert self.user_input.instance.num_epochs > 0, "Number of epochs in the user input is not positive"
            self.num_epochs = self.user_input.instance.num_epochs
        else:
            self.num_epochs = epochs
        self.epochs = list(range(self.num_epochs))

    def get_alpha_num_back(self, i: int, j: int) -> int:
        """
            The number of epochs required for a chunk to travel on the link (i,j) taking
            into account the link propagation delay.
            If the link has a propogation delay of 1 us, and the epoch is 0.5 us, then the chunk
            takes 2 additional epochs to reach j.
        """
        link_alpha = self.topology.alpha[i][j]
        if (link_alpha / self.epoch_duration) > self.user_input.instance.alpha_threshold:
            return math.ceil(
                link_alpha / self.epoch_duration)
        else:
            return 0

    def get_beta_num_back(self, i: int, j: int) -> int:
        """
            The number of extra epochs required for a chunk to travel on the link (i,j).
            If the link capacity is >= 1 chunk/sec, then there is no beta_num_back as the chunk can travel in one epoch.
            If the link capacity is < 1 chunk/sec, then the chunk needs to travel in multiple epochs.
            For example, if the link capacity is 0.5 chunk/sec, then the chunk needs to travel in 2 epochs.
            Since we account for one epoch implicity, this function returns 2-1 = 1 as the extra epoch.
        """
        epoch_capacity = self.topology.capacity[i][j] * self.epoch_duration
        return max(0, math.ceil(1 / epoch_capacity) - 1)

    def compute_floyd_warshall(self) -> None:
        """
            Computes the shortest path between all pairs of nodes in the network,
            where the distance between two nodes is the number of epochs it
            takes for a chunk to travel between them.
        """
        INF = float("inf")
        # epoch_distance is the number of epochs it takes for a chunk to cross the link
        epoch_distance = []
        for i, row in enumerate(self.topology.capacity):
            dist_row = []
            for j, c in enumerate(row):
                if c > 0:
                    epochs_for_one_chunk = 1 / (c * self.epoch_duration)
                    alpha_epochs = 0
                    if (self.topology.alpha[i][j] / self.epoch_duration) > self.user_input.instance.alpha_threshold:
                        alpha_epochs = math.ceil(
                            self.topology.alpha[i][j] / self.epoch_duration)
                    total_epochs = epochs_for_one_chunk + alpha_epochs
                    dist_row.append(total_epochs)
                else:
                    dist_row.append(INF)
            epoch_distance.append(dist_row)
        n = len(self.topology.capacity)

        for k, i, j in product(range(n), repeat=3):
            epoch_distance[i][j] = min(
                epoch_distance[i][j], epoch_distance[i][k] + epoch_distance[k][j])

        self.floyd_warshall = epoch_distance

    class LinkType(Enum):
        GPU_GPU = 1
        SWITCH_GPU = 2
        GPU_SWITCH = 3
        SWITCH_SWITCH = 4

    def get_link_type(self, i, j) -> LinkType:
        if i not in self.topology.switch_indices and j not in self.topology.switch_indices:
            return self.LinkType.GPU_GPU
        elif i not in self.topology.switch_indices and j in self.topology.switch_indices:
            return self.LinkType.GPU_SWITCH
        elif i in self.topology.switch_indices and j in self.topology.switch_indices:
            return self.LinkType.SWITCH_SWITCH
        elif i in self.topology.switch_indices and j not in self.topology.switch_indices:
            return self.LinkType.SWITCH_GPU
        else:
            raise ValueError("Invalid link type")

    def _validate_root(self, root: int) -> None:
        """
            Validates that the configured root of a rooted collective (GATHER/BROADCAST) is a
            real GPU that participates in the collective (not out of range, not a switch, not a
            passive forwarding-only node).
        """
        gpus = len(self.topology.capacity)
        if root < 0 or root >= gpus:
            raise ValueError(f"root {root} is out of range for a topology with {gpus} nodes")
        if root in self.topology.switch_indices:
            raise ValueError(f"root {root} is a switch index and cannot source/sink demand")
        if root in self.topology.passive_indices:
            raise ValueError(f"root {root} is a passive index and cannot source/sink demand")

    # ---------------------------------------------------------------------------------
    # Objective wiring
    # ---------------------------------------------------------------------------------

    def host_indices(self) -> List[int]:
        """
            The nodes that are hosts (GPUs): everything that is not a switch. Traffic a host
            carries for a source other than itself is *relay* traffic -- see
            hierarchical_objective_tiers.

            Passive nodes are deliberately INCLUDED. "Passive" only means the node originates
            and consumes no demand of its own; on real hardware it is still a GPU, so relaying
            through it costs the same local copy, kernel launches and link occupancy as
            relaying through any other GPU. Only a switch relays for free (in the sense the
            relay tier cares about), which is why switches are the sole exclusion.
        """
        switches = set(self.topology.switch_indices)
        return [n for n in self.nodes if n not in switches]

    def switch_ingress_links(self) -> List[Tuple[int, int]]:
        """
            Every link that ENTERS a switch -- gpu->switch and switch->switch alike. One unit of
            flow here is one hop of a switch chain, so summing flow over these links measures
            chain LENGTH: a direct gpu->gpu route scores 0, gpu->switch->gpu scores 1, and
            gpu->leaf->spine->leaf->gpu scores 3. (Counting only switch->switch links would
            score the first two identically, and a chain of one switch is still a chain.)
        """
        switches = set(self.topology.switch_indices)
        return [
            (i, j)
            for i in self.nodes
            for j in switches
            if self.topology.capacity[i][j] > 0
        ]

    def hierarchical_objective_tiers(self) -> List[Tuple[str, gp.LinExpr, float]]:
        """
            The tiers of ObjectiveType.LEXICOGRAPHIC, HIGHEST priority first, as
            (name, expression-to-MINIMIZE, relative tolerance the next tiers may degrade it by).
            Implemented per formulation because the flow/demand variables differ in shape
            (the MILP carries a chunk index the LP does not).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement ObjectiveType.LEXICOGRAPHIC")

    def apply_objective(self) -> None:
        """
            Install the objective selected by InstanceParams.objective_type on self.model.

            Every objective except LEXICOGRAPHIC is a single weighted LinExpr built by the
            formulation's objective_formulation(). LEXICOGRAPHIC instead installs one Gurobi
            objective per tier with strictly decreasing priority, so Gurobi optimizes them in
            order and only ever breaks ties of the higher tiers -- no cross-tier weight
            calibration, which is what a single blended objective (e.g.
            TOTAL_DEMAND_MIN_SWITCH_HOPS's GAMMA) has to get right by hand.
        """
        objective_type = self.user_input.instance.objective_type
        if objective_type != ObjectiveType.LEXICOGRAPHIC:
            self.model.setObjective(self.objective_formulation(objective_type))
            return

        tiers = self.hierarchical_objective_tiers()
        self.model.ModelSense = GRB.MINIMIZE
        self.model.NumObj = len(tiers)
        for index, (name, expr, rel_tol) in enumerate(tiers):
            # priority is highest-first; the last tier keeps no slack of its own.
            self.model.setObjectiveN(
                expr, index=index, priority=len(tiers) - index,
                reltol=rel_tol, name=name)
        logging.debug(
            "Lexicographic objective installed: " +
            " > ".join(f"{name} (reltol {rel_tol})" for name, _, rel_tol in tiers))

    def set_gurobi_params(self) -> None:
        self.model.Params.OutputFlag = self.user_input.gurobi.output_flag
        self.model.Params.TimeLimit = self.user_input.gurobi.time_limit * 60 * 60
        self.model.Params.FeasibilityTol = self.user_input.gurobi.feasibility_tol
        self.model.Params.IntFeasTol = self.user_input.gurobi.intfeas_tol
        self.model.Params.OptimalityTol = self.user_input.gurobi.optimality_tol
        self.model.Params.MIPGap = self.user_input.gurobi.mip_gap
        self.model.Params.Crossover = self.user_input.gurobi.crossover
        self.model.Params.MIPFocus = self.user_input.gurobi.mip_focus
        self.model.Params.Method = self.user_input.gurobi.method
        self.model.Params.Heuristics = self.user_input.gurobi.heuristics
        self.model.Params.Presolve = self.user_input.gurobi.presolve
        self.model.Params.SolutionLimit = self.user_input.gurobi.solution_limit
        # self.model.Params.NoRelHeurTime = 1200
        self.model.Params.PreSOS1BigM = 1e6
        # self.model.Params.Cuts = 1
        # self.model_.Params.RINS = 5000
        # self.model_.Params.Threads = 80

    @abstractmethod
    def get_schedule(self) -> None:
        pass
