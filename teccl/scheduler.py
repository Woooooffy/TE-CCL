import copy
import json
import logging
import math
import pathlib
from time import time
from typing import Dict, Tuple, Union

from gurobipy import GRB
from teccl.input_data import *
from teccl.solvers.allgather import AllGatherFormulation
from teccl.solvers.allgather_astar import AStarFormulation
from teccl.solvers.lp_formulation import LPFormulation
from teccl.solvers.base_formulation import BaseFormulation
from teccl.topologies.dgx1 import DGX1
from teccl.topologies.dgx2 import DGX2
from teccl.topologies.ndv2 import NDv2
from teccl.topologies.amd import AMD
from teccl.topologies.mesh import Mesh
from teccl.topologies.fat_tree_pod import FatTreePod
from teccl.topologies.fat_tree_pod_single_spine import FatTreePodSingleSpine
from teccl.topologies.fat_tree_pod_single_spine_fat_uplink import FatTreePodSingleSpineFatUplink
from teccl.topologies.odd_pod import OddPod
from teccl.topologies.star import Star
from teccl.topologies.incast_switch import IncastSwitch
from teccl.topologies.rail_optimized_spine_leaf import RailOptimizedSpineLeaf
from teccl.topologies.bridged_islands_cluster import (BridgedIslandsCluster,
                                                      BridgedIslandsSplitCluster)
from teccl.topologies.hetero_tapered_cluster import HeteroTaperedCluster
from teccl.topologies.dual_plane_hetero_cluster import (
    DualPlaneHeteroCluster, DualPlaneHeteroClusterScattered)
from teccl.topologies.two_pod_rail import TwoPodRail, TwoPodRailHostBound
from teccl.topologies.nested_cluster import NestedCluster
from teccl.topologies.dsl_topology import DslTopology
from teccl.topologies.topology import Topology


def solve_on_topology(user_input: UserInputParams, topology: Topology) -> "TECCLSolver":
    """Run TECCLSolver.solve() against an already-built Topology (bypassing get_topology, which
    only knows the named built-ins, not the CoarseTopology a hierarchical level constructs).
    Returns the TECCLSolver so the caller can reach the solved formulation
    (teccl_solver.best_solver) for post-processing.

    This lives in the core rather than beside the drivers because `hierarchy.solve` calls it on
    every level that needs a real formulation: teccl/examples has no __init__.py, so
    find_packages() never ships it and an INSTALLED teccl could not solve a level at all.
    """
    solver = TECCLSolver.__new__(TECCLSolver)
    solver.user_input = user_input
    solver.topology_obj = topology
    solver.solver = solver.get_solver(copy.deepcopy(user_input), topology)
    solver.solve()
    return solver


class TECCLSolver(object):
    def __init__(self, user_input: UserInputParams):
        self.user_input = user_input
        self.topology_obj = self.get_topology(user_input.topology)
        if user_input.instance.hierarchical:
            # No flat solver: each level builds its own (see solve_hierarchy). Building one here
            # would not just waste a model on the full fine topology -- get_solver's LP AllGather
            # check would reject a switch_copy=True run outright, even though it is the COARSE
            # level, not this one, that the formulation ever sees.
            self._require_hierarchy()
            self.solver = None
        else:
            self.solver = self.get_solver(copy.deepcopy(user_input), self.topology_obj)

    def _require_hierarchy(self) -> None:
        if not self.topology_obj.cells:
            raise ValueError(
                f"instance.hierarchical is set, but topology {self.user_input.topology.name} "
                f"declares no cells, so there is no hierarchy to solve. Override "
                f"Topology.build_hierarchy() (see teccl.hierarchy.Cell) or unset hierarchical.")


    def get_topology(self, topology_params: TopologyParams) -> Topology:
        # A .topo file SELECTS the topology, ahead of the name chain: the DSL builds the graph
        # from the file, so `name` is left free as the label the output filenames use.
        if topology_params.topo_file:
            return DslTopology(topology_params)
        if topology_params.name == "DGX1":
            return DGX1(topology_params)
        elif topology_params.name == "DGX2":
            return DGX2(topology_params)
        elif topology_params.name == "NDv2":
            return NDv2(topology_params)
        elif topology_params.name == "AMD":
            return AMD(topology_params)
        elif topology_params.name == "Mesh":
            return Mesh(topology_params)
        elif topology_params.name == "FatTreePod":
            return FatTreePod(topology_params)
        elif topology_params.name == "FatTreePodSingleSpine":
            return FatTreePodSingleSpine(topology_params)
        elif topology_params.name == "FatTreePodSingleSpineFatUplink":
            return FatTreePodSingleSpineFatUplink(topology_params)
        elif topology_params.name == "OddPod":
            return OddPod(topology_params)
        elif topology_params.name == "Star":
            return Star(topology_params)
        elif topology_params.name == "IncastSwitch":
            return IncastSwitch(topology_params)
        elif topology_params.name == "RailOptimizedSpineLeaf":
            return RailOptimizedSpineLeaf(topology_params)
        elif topology_params.name == "HeteroTaperedCluster":
            return HeteroTaperedCluster(topology_params)
        elif topology_params.name == "BridgedIslandsCluster":
            return BridgedIslandsCluster(topology_params)
        elif topology_params.name == "BridgedIslandsSplitCluster":
            return BridgedIslandsSplitCluster(topology_params)
        elif topology_params.name == "DualPlaneHeteroCluster":
            return DualPlaneHeteroCluster(topology_params)
        elif topology_params.name == "DualPlaneHeteroClusterScattered":
            return DualPlaneHeteroClusterScattered(topology_params)
        elif topology_params.name == "TwoPodRail":
            return TwoPodRail(topology_params)
        elif topology_params.name == "TwoPodRailHostBound":
            return TwoPodRailHostBound(topology_params)
        elif topology_params.name == "NestedCluster":
            return NestedCluster(topology_params)
        else:
            raise NotImplementedError(
                f"Input topology {topology_params.name} not implemented")


    def _resolve_formulation(self, user_input: UserInputParams) -> Formulation:
        """
            Resolves the effective solver formulation. If the user set one explicitly it is
            honored; otherwise the per-collective default is used: ALLGATHER -> MILP (the only
            collective the MILP implements), every other collective -> LP.
        """
        f = user_input.instance.formulation
        if f is not None:
            return f
        return Formulation.MILP if user_input.instance.collective == Collective.ALLGATHER else Formulation.LP

    def get_solver(self, user_input: UserInputParams, topology: Topology) -> BaseFormulation:
        collective = user_input.instance.collective
        formulation = self._resolve_formulation(user_input)

        if formulation == Formulation.MILP:
            if collective != Collective.ALLGATHER:
                raise NotImplementedError(
                    f"MILP formulation is only implemented for ALLGATHER, not {collective}")
            if user_input.instance.objective_type == ObjectiveType.ASTAR:
                return AStarFormulation(user_input, topology)
            return AllGatherFormulation(user_input, topology)

        # LP formulation: collective-agnostic, demand-matrix driven.
        assert formulation == Formulation.LP
        if collective == Collective.ALLGATHER and user_input.instance.switch_copy:
            raise ValueError(
                "AllGather with the LP formulation requires switch_copy=False: the LP aggregates "
                "flow per source and cannot represent switch/GPU copy (replication).")
        if collective == Collective.ALLTOALL:
            # Scale the per-GPU chunk count up to the total number of alltoall
            # chunks (one group of chunks per active GPU). Do this on a copy so
            # we never mutate the caller's instance in place: get_solver is
            # called repeatedly (feasible search / iterative binary search) and
            # an in-place scaling would compound (num_chunks *= num_gpus every
            # call), inflating every subsequent solve.
            lp_input = copy.deepcopy(user_input)
            lp_input.instance.num_chunks = user_input.instance.num_chunks * \
                (len(topology.capacity) - len(topology.switch_indices) - len(topology.passive_indices))
            return LPFormulation(lp_input, topology)
        # AllGather (and future demand-driven collectives): num_chunks passes through unscaled.
        return LPFormulation(user_input, topology)

    def feasible_solution_search(self, user_input: UserInputParams, topology_obj: Topology, final_epoch_duration: float) -> Union[Tuple[float, int], ValueError]:
        """
            Finds a feasible time in which the collective can finish using large epochs with fewer of them.
        """
        is_lp = self._resolve_formulation(user_input) == Formulation.LP
        if is_lp:
            num_epochs = math.ceil(topology_obj.get_max_hop_distance() * 20)
            factor = 100
        else:
            num_epochs = math.ceil(topology_obj.get_max_hop_distance() * 3)
            factor = 1
        # 3 is a good factor to not have too many epochs, but also to take into account the alpha
        max_time_chunk = topology_obj.get_largest_time_chunk()

        collective_time_estimate = num_epochs * max_time_chunk * \
            user_input.instance.num_chunks * len(topology_obj.capacity) / 2

        # binary search on the time estimate
        lower_bound = 0

        upper_bound = collective_time_estimate * factor
        attempts = 0
        feasible_time = upper_bound

        # Across this search only epoch_duration changes; num_epochs is fixed, so the
        # LP model (variables, objective, most constraints) is invariant. Build it
        # once and refresh only the epoch_duration-dependent parts each iteration
        # (LPFormulation.update_epoch_duration) so Gurobi warm-starts instead of
        # rebuilding + cold-solving ~10 large models. (The MILP path is left on the
        # per-iteration rebuild; its structure depends on epoch_duration in more
        # ways -- beta_num_back, switch pipelining -- so it is not reused here.)
        reusable_lp_solver = None
        if is_lp:
            base_input = copy.deepcopy(user_input)
            base_input.gurobi = copy.deepcopy(user_input.gurobi)
            base_input.instance = copy.deepcopy(user_input.instance)
            base_input.gurobi.solution_limit = 1
            base_input.instance.num_epochs = num_epochs
            reusable_lp_solver = self.get_solver(base_input, topology_obj)

        while lower_bound <= upper_bound:
            mid = (upper_bound + lower_bound) / 2
            epoch_duration = mid / num_epochs
            if epoch_duration <= final_epoch_duration and feasible_time != collective_time_estimate * factor:
                break
            # user_input.instance.debug = True
            if is_lp:
                # Reuse the built model; only rescale the epoch_duration-dependent
                # capacity RHS (and rebuild only if the alpha structure changes).
                reusable_lp_solver.update_epoch_duration(epoch_duration)
                solver_inst = reusable_lp_solver
                result = solver_inst.solve_model()
            else:
                new_user_input = copy.deepcopy(user_input)
                new_user_input.gurobi = copy.deepcopy(user_input.gurobi)
                new_user_input.instance = copy.deepcopy(user_input.instance)
                # find some feasible solution
                new_user_input.gurobi.solution_limit = 1
                new_user_input.instance.num_epochs = num_epochs
                new_user_input.instance.epoch_duration = epoch_duration
                solver_inst = self.get_solver(new_user_input, topology_obj)
                result = solver_inst.encode_problem()
            if result != GRB.INFEASIBLE:
                epochs_taken = solver_inst.find_demand_satisfied_k() + 1
                time_taken = epochs_taken * solver_inst.epoch_duration
                upper_bound = time_taken
                feasible_time = min(feasible_time, time_taken)
            else:
                lower_bound = mid
            attempts += 1
            if attempts > 10:
                # Avoids trying too many times and spending time in the initial search.
                break
        if feasible_time != collective_time_estimate * factor:
            return feasible_time, num_epochs
        raise ValueError(
            "Unable to find a solution in the initial feasible search algorithm (try with factor > 1)")


    def get_schedules(self, initial_solver: BaseFormulation, user_input: UserInputParams, topology_obj: Topology) -> Dict:
        """
            Finds the optimal schedule for the collective either directly or iteratively.
            In the direct method, the solver is instantiated to find the optimal solution in the given number of epochs.
            In the iterative method, the solution is found using binary search and in each iteration the solver is instantiated
                to find some feasible solution.
        """
        epoch_result_schedule_solver = {}
        if user_input.instance.solution_method == SolutionMethod.ONE_SHOT:
            # One shot
            result = initial_solver.encode_problem()
            schedule, schedule_json = initial_solver.get_schedule()
            if schedule:
                epochs_taken = initial_solver.find_demand_satisfied_k() + 1
                epoch_result_schedule_solver[epochs_taken] = {"result": result,
                                                            "schedule": (schedule, schedule_json),
                                                            "solver": initial_solver}
        else:
            # Iterative
            user_input.gurobi.solution_limit = 1
            lower_bound = 0
            upper_bound = user_input.instance.num_epochs
            tried_epochs = set()
            while lower_bound <= upper_bound:
                mid = math.ceil((upper_bound + lower_bound) / 2)
                if mid in tried_epochs:
                    break
                tried_epochs.add(mid)
                new_user_input = copy.deepcopy(user_input)
                new_user_input.instance.num_epochs = mid
                solver_inst = self.get_solver(new_user_input, topology_obj)
                solver = solver_inst
                result = solver_inst.encode_problem(use_one_less_epoch=True)
                schedule, schedule_json = solver_inst.get_schedule()
                if schedule:
                    epochs_taken = solver_inst.find_demand_satisfied_k() + 1
                    logging.debug(
                        f"Found a feasible schedule in {epochs_taken} epochs")
                    upper_bound = epochs_taken
                    epoch_result_schedule_solver[epochs_taken] = {"result": result,
                                                                "schedule": (schedule, schedule_json),
                                                                "solver": solver_inst}
                else:
                    lower_bound = mid
        return epoch_result_schedule_solver
    
    def solve_hierarchy(self, start: float) -> Dict:
        """
            Solves the collective LEVEL BY LEVEL instead of as one flat problem, and writes the
            result to the same place the flat path does.

            Everything hierarchy-specific stays inside this method: what comes back from
            solve_hierarchical is an ordinary flat schedule on the fine topology, so the output
            file means exactly what it means in the flat mode and ncclize needs no hierarchy
            awareness. The side artifacts (coarse schedule, identities, intra) land next to it
            under Schedules/ so a bad run can be diagnosed without re-solving.
        """
        from teccl.hierarchy import ring_solve
        from teccl.hierarchy.solve import solve_hierarchical, write_side_outputs

        user_input = self.user_input
        instance = user_input.instance
        topology = self.topology_obj
        self._require_hierarchy()

        if instance.intra_algo:
            # Read through ring_solve.intra_algo() everywhere, so assigning the module global is
            # how a caller selects the base-case algorithm. Validate it here rather than letting
            # it raise from inside the recursion, several levels deep.
            ring_solve.INTRA_ALGO = instance.intra_algo
        algo = ring_solve.intra_algo()

        prefix = instance.hierarchy_prefix or \
            f"{user_input.topology.name}_{instance.collective.name.lower()}"
        if algo != ring_solve.ALGO_CROSSBAR:
            # Tag the non-default algorithm into the artifact names so an A/B cannot silently
            # overwrite its own baseline.
            prefix = f"{prefix}_{algo}"

        # num_chunks means the same thing in both modes: chunks per source per destination on the
        # FINE topology. The flat path scales alltoall up by the participating-GPU count inside
        # get_solver; the hierarchical path reaches build_demand directly, so scale it here.
        num_chunks = instance.num_chunks
        if instance.collective == Collective.ALLTOALL:
            num_chunks *= (len(topology.capacity) - len(topology.switch_indices)
                           - len(topology.passive_indices))

        level_input = copy.deepcopy(user_input)
        # Levels are flat solves; without this the root level would recurse into this method again.
        level_input.instance.hierarchical = False
        # Every level solves through TECCLSolver, and the root level would otherwise write its
        # COARSE schedule over the user's output file -- which has to hold the flat one.
        level_input.instance.schedule_output_file = f"Schedules/{prefix}_coarse.json"

        info, solution = solve_hierarchical(
            topology, level_input, instance.collective, num_chunks,
            prefix=prefix, debug=instance.debug, level_chunk=instance.level_chunk)

        if instance.hierarchy_side_outputs and solution.resolution is not None:
            write_side_outputs(solution.resolution, solution.flows, prefix)

        output_file = instance.schedule_output_file or f"Schedules/{prefix}_flat.json"
        info["Solver_Time"] = time() - start
        pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w+') as f:
            f.write(json.dumps(info, indent=2, sort_keys=True))
        print(f'Schedule written to {output_file}')
        return info

    def solve(self):
        """
            Main function that first finds a feasible collective finish time to estimate the number of epochs required
             if user did not provide it. It then finds the optimal solution and outputs the schedule to the file.
        """
        start = time()
        logs_dir = pathlib.Path("Logs")
        logs_dir.mkdir(exist_ok=True)
        user_input = self.user_input
        solver = self.solver
        if user_input.instance.debug:
            logging.basicConfig(format='%(asctime)s %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG, filename=user_input.instance.debug_output_file)

        if user_input.instance.hierarchical:
            # The level-by-level path owns its own epoch sizing (each level derives its own from
            # its own chunk), so neither the feasible search nor get_schedules applies.
            self.solve_hierarchy(start)
            return

        if user_input.instance.num_epochs == -1:
            # Search for a feasible time to estimate the number of epochs
            feasible_time, _ = self.feasible_solution_search(
                user_input, self.topology_obj, solver.epoch_duration)
            user_input.instance.num_epochs = math.ceil(
                feasible_time / solver.epoch_duration)
            solver.set_num_epochs(user_input.instance.num_epochs)

        epoch_result_schedule_solver = self.get_schedules(
            solver, user_input, self.topology_obj)
        timestamp = int(time())

        if epoch_result_schedule_solver:
            output_file = user_input.instance.schedule_output_file
            best_epochs = min(epoch_result_schedule_solver.keys())
            solver = epoch_result_schedule_solver[best_epochs]["solver"]
            # Expose the solved formulation (per_chunk_flow_paths populated) for hierarchical
            # post-processing, e.g. teccl.hierarchy.reconstruct.resolve_identities.
            self.best_solver = solver
            if user_input.instance.schedule_output_file == "":
                output_file = f'{user_input.topology.name}_{solver.num_nodes}-nodes_{solver.num_chunks}-chunks_{user_input.topology.chunk_size}-chunksize_{solver.solver_name}_{timestamp}.json'
            epoch_result_schedule_solver[best_epochs]["schedule"][1]["Solver_Time"] = time(
            ) - start
            pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w+') as f:
                json_obj = json.dumps(
                    epoch_result_schedule_solver[best_epochs]["schedule"][1], indent=2, sort_keys=True)
                f.write(json_obj)
            print(f'Schedule written to {output_file}')


        else:
            logging.error("No schedule found with the given parameters")


