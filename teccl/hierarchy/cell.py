from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Cell:
    """
    A group of fine-grained nodes collapsed into one coarse node.

    A Cell is declared by Topology.build_hierarchy() in FINE node indices only -- it has no
    knowledge of the coarse numbering, which abstract() assigns. For the rail-optimized
    spine-leaf topology a Cell is one host: its 8 GPUs plus the 1 NVSwitch behind them.

    Fields:
        members:           all fine node indices collapsed into this cell (gpus + internal
                           switches). Cells must be disjoint and none may contain a node that
                           has an external (inter-cell) role for another cell.
        gpus:              the data-bearing fine indices in the cell (they source/sink demand
                           and can be an external link's physical endpoint). Order matters: it
                           defines the sub-chunk <-> origin-GPU correspondence used by
                           lift_demand (sub-chunk c of this cell originates on gpus[c]).
        internal_switches: fine switch indices that live entirely inside the cell (e.g. the
                           NVSwitch). These are dropped from the coarse graph; they only
                           matter to the intra-cell (phase 3) reconstruction.
        boundary:          fine external neighbor -> the fine gpu(s) inside this cell that
                           physically own the link to that neighbor. This is the load-bearing
                           port map: in the rail-optimized topology GPU r can only reach leaf
                           r, so boundary is {leaf(r): [gpu(n, r)] for r}. abstract() converts
                           this to coarse ids and cross-checks it against the capacity matrix.
    """
    members: List[int]
    gpus: List[int]
    internal_switches: List[int] = field(default_factory=list)
    boundary: Dict[int, List[int]] = field(default_factory=dict)
