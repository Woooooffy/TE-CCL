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
    """
    BINARY_USED_EPOCHS = 1
    TOTAL_DEMAND = 2
    PAPER = 3
    ASTAR = 4
    TOTAL_DEMAND_MIN_SWITCH_HOPS = 5

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
    solution_method: SolutionMethod = SolutionMethod.ONE_SHOT
    schedule_output_file: str = "" # If not empty, the schedule is written to this file. Default is "Topology-Chunks-chunksize-timestamp.json"
    lower: bool = False # If true will use the lowering code from Meghan to lower the input.
    lower_xml: str = "" # If not empty, the XML is written to this file. Default is "Topology-Chunks-chunksize-timestamp.xml"
    warmstart: str = "" # If not empty, the warmstart file is used to warmstart the optimization.
    symmetry: bool = False # If true, nodes that are given as symmetric are constrainted to have same number of total flows. 
    
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