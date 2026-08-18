from dataclasses import dataclass, field
from enum import Enum


@dataclass
class TopologyParams:
    name: str = "DGX1"
    chassis: int = 1
    chunk_size: float = 1 # in GB
    alpha: tuple = (0 ,0) # (link alpha, switch alpha)
    side_length: int = 4 # Only for Mesh and Torus topology
    passive_node_indices: tuple = ()  # GPU indices present (can forward) but excluded from this collective's demand
    # Switch indices whose forwarding is externally programmable, i.e. the switches ncclize is
    # allowed to emit a routing table for. None => use the topology class's default (see
    # Topology.default_programmable_switch_indices, which permits every switch). Set it to
    # restrict emission to e.g. the network switches only, leaving self-routing fabrics such as
    # intra-node NVSwitches out of the table. Does NOT affect the solve -- routing/capacity still
    # use every switch; it only scopes the emitted switch program.
    programmable_switch_indices: tuple = None

@dataclass
class GurobiParams:
    time_limit: float = 2           # in hrs https://www.gurobi.com/documentation/10.0/refman/timelimit.html 
    feasibility_tol: float = 1e-4   # https://www.gurobi.com/documentation/10.0/refman/feasibilitytol.html
    intfeas_tol: float = 1e-4       # https://www.gurobi.com/documentation/10.0/refman/intfeastol.html
    optimality_tol: float = 1e-4    # https://www.gurobi.com/documentation/10.0/refman/optimalitytol.html
    output_flag: int = 1            # https://www.gurobi.com/documentation/10.0/refman/outputflag.html
    log_file: str = ""              # https://www.gurobi.com/documentation/10.0/refman/logfile.html#parameter:LogFile
    log_to_console: int = 0         # https://www.gurobi.com/documentation/10.0/refman/logtoconsole.html
    mip_gap: float = 1e-4           # https://www.gurobi.com/documentation/10.0/refman/mipgap2.html
    mip_focus: int = 0              # https://www.gurobi.com/documentation/10.0/refman/mipfocus.html    
    crossover: int = -1             # https://www.gurobi.com/documentation/10.0/refman/crossover.html
    method: int = -1                # https://www.gurobi.com/documentation/10.0/refman/method.html
    heuristics: float = 0.05        # https://www.gurobi.com/documentation/9.5/refman/heuristics.html
    presolve: int = -1              # https://www.gurobi.com/documentation/9.5/refman/presolve.html
    solution_limit: int = 2000000 # https://www.gurobi.com/documentation/9.5/refman/solutionlimit.html

class ObjectiveType(Enum):
    """
        Different objective functions for AllGather.
        1 - BINARY_USED_EPOCHS - Uses a binary variable for each epoch and minimizes the number of used epochs.
        2 - TOTAL_DEMAND - Gives a reward starting from the epoch all the demands are met.
        3 - PAPER - For each demand met, gives a reward starting from the epoch the demand is met.
        4 - ASTAR - Motivate the solver to make as much progress towards the goal of satisfying all demands as possible in each epoch.
        5 - TOTAL_DEMAND_MIN_SWITCH_HOPS - Identical reward to TOTAL_DEMAND, plus a tiny per-hop
            penalty on flow crossing switch->switch links (e.g. leaf<->spine). Among solutions
            that finish in the same epoch (the TOTAL_DEMAND term dominates), this makes the
            solver prefer routes with fewer switch relays -- e.g. a direct leaf hop over a spine
            detour. The penalty is scaled far below the completion reward so it acts purely as a
            tie-breaker and never trades completion time for fewer hops.
        6 - LEXICOGRAPHIC - A three-tier hierarchical (lexicographic) objective solved with
            Gurobi's multi-objective support, in strictly decreasing priority:
              tier 1 (latency)      minimize the demand-weighted completion time,
              tier 2 (host relay)   among the tier-1 optima, minimize the volume that a GPU
                                    forwards on behalf of another GPU (relay through a host
                                    burns that host's link bandwidth and its copy engines),
              tier 3 (switch chain) among those, minimize the number of switch hops the volume
                                    takes (flow entering a switch), i.e. prefer the shortest
                                    switch chain: a direct link over a one-switch path, and a
                                    one-switch path over a leaf->spine->leaf detour.
            Unlike option 5 -- which folds the switch-hop term into ONE objective with a hand
            tuned GAMMA and so depends on that constant being small enough -- the tiers here
            are enforced by the solver: a lower tier can never degrade a higher one by more
            than the tolerance you allow it (InstanceParams.objective_latency_rel_tol /
            objective_relay_rel_tol, both 0 by default = pure lexicographic).
    """
    BINARY_USED_EPOCHS = 1
    TOTAL_DEMAND = 2
    PAPER = 3
    ASTAR = 4
    TOTAL_DEMAND_MIN_SWITCH_HOPS = 5
    LEXICOGRAPHIC = 6

class Collective(Enum):
    ALLGATHER = 1
    ALLTOALL = 2
    GATHER = 3    # every (non-root) GPU sends its distinct data to a single root GPU
    BROADCAST = 4 # a single root GPU sends its data to every other GPU

class Formulation(Enum):
    """
        Which solver formulation to use for the collective.
            1 - MILP - The chunk-level integer program (AllGatherFormulation). Supports switch/GPU copy.
            2 - LP   - The continuous per-source flow LP (LPFormulation). Demand-matrix driven and
                       collective-agnostic; used for AllToAll, and usable for AllGather only when
                       switch_copy is disabled (the LP cannot represent copy/replication).
        Leave unset (None) on InstanceParams to use the per-collective default:
            ALLGATHER -> MILP, ALLTOALL -> LP.
    """
    MILP = 1
    LP = 2

class EpochType(Enum):
    """ 
        Epoch_type is used to set the epoch duration. 
            1 - FASTEST_LINK - set epoch duration based on the fastest link (fine-grained epoch duration)
            2 - SLOWEST_LINK - set epoch duration based on the slowest link (coarse-grained epoch duration)
            3 - USER_INPUT - uses the input epoch duration 
    """
    FASTEST_LINK = 1
    SLOWEST_LINK = 2    
    USER_INPUT = 3

class SolutionMethod(Enum):
    """
        1 - One shot - The optimization is run till the time limit is reached or it finds a solution within the specified mip gap
        2 - Iterative - The optimization is run iteratively using binary search to find a solution within limit of num_epochs.
    """
    ONE_SHOT = 1
    ITERATIVE = 2

@dataclass
class InstanceParams:
    collective: Collective = Collective.ALLGATHER
    formulation: Formulation = None # Solver formulation (None = per-collective default: ALLGATHER->MILP, everything else->LP)
    root: int = 0 # Root GPU index for rooted collectives (GATHER destination / BROADCAST source); ignored otherwise
    num_chunks: int = 1 # Number of chunks to be transferred from each node to each other node
    epoch_type: EpochType = EpochType.FASTEST_LINK 
    epoch_duration: float = -1
    epoch_multiplier: int = 1   # Multiplier for epoch duration (helpful for epoch_type != -1)
    num_epochs:int = -1         # Number of epochs to be run (-1 to automatically figure out the number of epochs)
    epsilon: float = pow(10, -1)
    alpha_threshold: float = 0.1 # Link alpha to epoch duration ratio threshold below which alpha is taken as 0
    alpha_epoch_duration_ratio_max: int = 200 # Maximum ratio of alpha to epoch duration (if exceeded, epoch duration is increased)
    switch_copy: bool = True # If True, switch can copy the chunks
    switch_pipeline: bool = True # If True, switches are a cut-through (pipelined) fabric: a chunk relayed through switches only pays propagation delay per hop, not a full store-and-forward serialization epoch. If False, every switch hop is store-and-forward. Which hops are made cut-through depends on the formulation: allgather pipelines the switch->switch and switch->gpu (egress) hops; alltoall pipelines the gpu->switch (ingress) and switch->switch hops. (The complementary leg is already handled by each formulation's structure: the first gpu->switch hop for allgather, the final switch->gpu hop for alltoall.)
    debug: bool = False # If True, prints debug information
    debug_output_file: str = "" # If debug is True, prints debug information to this file
    objective_type: ObjectiveType = ObjectiveType.PAPER # The objective function to be used (3 - The objective function used in the paper)
    # Only used by ObjectiveType.LEXICOGRAPHIC: how much a LOWER priority tier is allowed to
    # degrade a HIGHER one, as a fraction of that tier's optimal value (Gurobi ObjNRelTol).
    # 0.0 == pure lexicographic (a tier is fixed at its exact optimum before the next is
    # optimized). Give latency a small slack (e.g. 0.01) when you would accept 1% more
    # completion time to buy a materially less relay-heavy / shorter-chained routing.
    objective_latency_rel_tol: float = 0.0   # slack on tier 1 (latency) for tiers 2-3
    objective_relay_rel_tol: float = 0.0     # slack on tier 2 (host relay) for tier 3
    solution_method: SolutionMethod = SolutionMethod.ONE_SHOT
    schedule_output_file: str = "" # If not empty, the schedule is written to this file. Default is "Topology-Chunks-chunksize-timestamp.json"
    lower: bool = False # If true will use the lowering code from Meghan to lower the input.
    lower_xml: str = "" # If not empty, the XML is written to this file. Default is "Topology-Chunks-chunksize-timestamp.xml"
    warmstart: str = "" # If not empty, the warmstart file is used to warmstart the optimization.
    symmetry: bool = False # If true, nodes that are given as symmetric are constrainted to have same number of total flows.
    # --- Hierarchical solve (teccl.hierarchy.solve.solve_hierarchical) -------------------------
    # If True, the topology is solved LEVEL BY LEVEL (abstract -> solve -> lower -> recurse) instead
    # of as one flat problem. Requires a topology that declares cells (Topology.build_hierarchy).
    # The output is still an ordinary flat schedule on the fine topology, written to
    # schedule_output_file exactly as in the flat mode, so everything downstream (ncclize) is
    # unchanged. Every other InstanceParams field keeps its meaning and is applied at each level:
    # formulation/objective_type/symmetry/switch_copy configure the per-level solve, and num_chunks
    # is still per source per destination on the FINE topology.
    hierarchical: bool = False
    # Base-case algorithm for a crossbar cell: "crossbar" (default) or "ring". None leaves
    # teccl.hierarchy.ring_solve.INTRA_ALGO alone, i.e. honors $TECCL_INTRA_ALGO.
    intra_algo: str = None
    # Force the root level's chunk unit g instead of taking the GCD of its coarse demands. Set it
    # to 1 to reproduce the pre-coarsening behaviour (the drivers' `nocoarsen` A/B); None = derive.
    level_chunk: int = None
    # Base name for the hierarchy's side artifacts under Schedules/ (_coarse, _identities, _intra).
    # Empty => "{topology name}_{collective}", plus an "_{algo}" tag when the intra-cell algorithm
    # is not the default, so a ring run never overwrites its crossbar baseline.
    hierarchy_prefix: str = ""
    # If True, also write Schedules/{prefix}_{identities,intra}.json: the resolved inter-cell
    # traffic and the assembled sub-level schedule. Nothing consumes them; they are how a bad run
    # is diagnosed without re-running the (expensive) coarse solve.
    hierarchy_side_outputs: bool = False
    
@dataclass
class UserInputParams:
    # Use default_factory so each UserInputParams gets its own params objects.
    # Bare mutable defaults would be shared across all instances (and not copied
    # by copy.deepcopy, since they'd live on the class), which silently couples
    # unrelated solves and compounds in-place mutations like the alltoall
    # num_chunks scaling in the scheduler.
    topology: TopologyParams = field(default_factory=TopologyParams)
    gurobi: GurobiParams = field(default_factory=GurobiParams)
    instance: InstanceParams = field(default_factory=InstanceParams)