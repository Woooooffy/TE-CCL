"""
The recursion's vocabulary: what one level of the hierarchical solve is handed, and what it returns.

The contract these types make executable is the one teccl/hierarchy/scale.py states in prose: *a
level is (topology, demand, scale); the coarse solve and the intra-cell solve are the same problem;
every level receives INTEGER demands expressed in that level's own chunk unit.* Before these types
existed the contract was asserted in the docstrings and violated in the structure -- the "second
level" was a direct call to the crossbar scheduler, not a recursive call to anything.

Three ideas do all the work.

**Identities are GLOBAL at every depth.** An `Identity` is `(fine GPU, fine chunk)` -- it names the
data, not a position in some level's index space -- so a flow produced eight levels down means the
same thing to the stitch as one produced at the root. This is what makes flatten-on-return sound: a
child hands its parent finished flows, and the only thing the parent has to fix up is the
`(band, local_round)` time coordinate.

**Node indices are LOCAL, and translated at exactly one place.** A `Topology` is dense 0..n-1 by
contract, so a cell's scattered global member ids get renumbered by `subtopology.induce`, which is
the sole owner of the correspondence. `LevelDemand.local_to_global` carries it onward.

**The boundary currency is `List[IntraCellDemand]`, not a tensor.** It is what step B
(`reconstruct.build_child_problems`) already emits, and it is strictly richer: it carries the global
identity, the hard/soft kind, and the deadline that decides the band. The dense `demand[s][t][c]`
tensor is a LAZY ADAPTER (`LevelDemand.from_demands`), built only on the two branches that actually
need one -- a child solved by a real formulation (which wants `topology.demand_override`) and a
child that recurses further (`coarsify_demand` takes a tensor).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from teccl.hierarchy.crossbar_solve import IntraFlow
from teccl.hierarchy.reconstruct import Identity, IntraCellDemand, ResolvedPiece
from teccl.hierarchy.scale import ChunkScale
from teccl.topologies.topology import Topology


@dataclass
class LevelDemand:
    """A level's demand as a dense tensor, plus the table that maps it back to global identities.

    `demand[s][t][c]` is the shape every formulation already consumes (`solvers.demand`), in this
    level's LOCAL node indices. The chunk axis is a re-labelling, not a chunk: slot `c` of local
    source `s` is whatever global identity `identities[(s, c)]` says it is. Without that table the
    tensor is lossy -- two different global identities held by the same GPU are indistinguishable
    once they are just "chunk 0" and "chunk 1".
    """
    demand: np.ndarray
    identities: Dict[Tuple[int, int], Identity]
    local_to_global: List[int]
    global_to_local: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.global_to_local:
            self.global_to_local = {g: i for i, g in enumerate(self.local_to_global)}

    @classmethod
    def from_demands(cls, demands: Sequence[IntraCellDemand], topology) -> "LevelDemand":
        """Build the tensor adapter from the boundary currency.

        The source axis is the GPU that CURRENTLY HOLDS the identity (`src_gpu`), which below the
        root is generally not the identity's native source -- the data has already been relayed, and
        a demand tensor's axis-0 means "who has it", since that is the assumption every formulation
        makes about where a commodity starts. The chunk axis is a fresh per-source enumeration of the
        distinct identities that holder must distribute.
        """
        l2g = list(getattr(topology, "local_to_global", range(len(topology.capacity))))
        g2l = {g: i for i, g in enumerate(l2g)}
        n = len(topology.capacity)

        # ONE HOLDER PER IDENTITY. Several demands can name the same identity with different
        # `src_gpu` -- an egress_stage 1->0 and the ingress fan-out 0->{...} are one chain, and both
        # call themselves a source. A demand tensor cannot express a chain: it says "this node has
        # it, those nodes want it". So collapse the chain to its ROOT (the holder that is nobody
        # else's destination) and the UNION of everything that wants it. That is not a loss of
        # information, it is the correct statement of the level's problem -- how the data actually
        # gets from the root to the wanters is what this level is being asked to decide, and letting
        # a parent's relay structure leak in would both double-count the traffic and pin the level
        # to a route it never chose.
        wanted: Dict[Identity, set] = {}
        srcs: Dict[Identity, set] = {}
        dsts: Dict[Identity, set] = {}
        for d in demands:
            srcs.setdefault(d.identity, set()).add(g2l[d.src_gpu])
            tset = {g2l[t] for t in d.dst_gpus}
            dsts.setdefault(d.identity, set()).update(tset)
            wanted.setdefault(d.identity, set()).update(tset)

        holder_of: Dict[Identity, int] = {}
        for ident, ss in srcs.items():
            roots = sorted(ss - dsts.get(ident, set()))
            holder_of[ident] = roots[0] if roots else min(ss)

        # Deterministic slot order, so two runs (and a memo hit) agree on the labelling.
        per_src: Dict[int, List[Identity]] = {}
        for ident in sorted(holder_of, key=lambda i: (holder_of[i], i)):
            per_src.setdefault(holder_of[ident], []).append(ident)

        slot: Dict[Tuple[int, Identity], int] = {}
        identities: Dict[Tuple[int, int], Identity] = {}
        for s, ids in per_src.items():
            for c, ident in enumerate(ids):
                slot[(s, ident)] = c
                identities[(s, c)] = ident

        num_chunks = max((len(v) for v in per_src.values()), default=1)
        demand = np.zeros((n, n, num_chunks), dtype=np.int32)
        for ident, targets in wanted.items():
            s = holder_of[ident]
            c = slot[(s, ident)]
            for t in targets:
                if t != s:
                    demand[s][t][c] = 1
        return cls(demand=demand, identities=identities, local_to_global=l2g, global_to_local=g2l)

    def relabel(self, local_identity: Identity) -> Identity:
        """Translate a `(local source, slot)` label -- what `identity_sets` reads off the tensor --
        back to the global identity it stands for.

        Every consumer downstream of a level solve MUST go through this. `identity_sets`
        (reconstruct.py) reads identities straight off the demand array, so below the root the
        labels it returns are local slots that happen to have the same shape as an Identity; using
        one unremapped would silently attribute a chunk to the wrong GPU.
        """
        return self.identities.get(local_identity, local_identity)


@dataclass
class Subproblem:
    """One level of the recursion.

    band / budget_rounds are the parent's timing contract, and they are what turn an arbitrary
    solver into a usable level: "schedule this work inside the `budget_rounds` rounds that parent
    band `band` is worth". Both are None/0 at the root, which has no parent to answer to.
    """
    topology: Topology
    demands: List[IntraCellDemand]
    scale: ChunkScale
    depth: int = 0
    band: int = 0
    budget_rounds: Optional[int] = None
    cell_id: int = -1
    # The ROOT's demand, which arrives as a tensor rather than as IntraCellDemands because nothing
    # has resolved anything yet -- there is no parent level to have emitted them. Every level below
    # builds its tensor from `demands` instead (LevelDemand.from_demands), so this is set exactly
    # once, at depth 0.
    root_tensor: Optional[object] = None
    # The LevelDemand a child was built from, kept so the relabelling table survives into the level
    # solve: below the root, tensor coordinates are local slots and must be translated back to
    # global identities before any of them is used as one.
    level_demand: Optional["LevelDemand"] = None

    def is_degenerate(self) -> bool:
        """Nothing to solve: no demand, or nowhere for it to go.

        A single data-bearing node is degenerate rather than a base case -- there is no link to
        schedule on -- and it is the honest terminal for a cell that was collapsed down to one GPU.
        """
        # The root is described by a tensor, not by demands, so an empty `demands` says nothing
        # about it -- checking only `demands` would make every root trivially degenerate.
        if not self.demands and self.root_tensor is None:
            return True
        switches = set(self.topology.switch_indices)
        passive = set(getattr(self.topology, "passive_indices", []))
        data = [i for i in range(len(self.topology.capacity))
                if i not in switches and i not in passive]
        return len(data) <= 1


@dataclass
class LevelSolution:
    """What a level returns to its parent.

    `flows` are ALREADY FLAT in the parent's `(band, local_round)` coordinate -- flatten-on-return.
    A level converts its children's schedules onto its own round axis before returning, so no
    consumer ever sees a nested time coordinate and `stitch.py` stays a two-level flattener however
    deep the recursion actually went.

    `pieces` is the one depth-dependent field. At depth 0 a piece is NETWORK traffic and goes to the
    stitch as such; at depth >= 1 the same piece is an ordinary transfer over some cell's internal
    fabric, so `pieces_as_flows` turns it into an `IntraFlow` and the parent never sees a piece at
    all.

    `scale` is the NET of this whole subtree's refinements, resolved bottom-up: a refinement
    discovered three levels down still has to be paid for at the root, because
    `refinement_from_root` is what reaches ncclize's `chunk_up()` and MAX_M bounds the product over
    every level.
    """
    flows: List[IntraFlow] = field(default_factory=list)
    pieces: List[ResolvedPiece] = field(default_factory=list)
    scale: Optional[ChunkScale] = None
    # This level's own step-B output and epoch length. Only the ROOT's are consumed -- `stitch`
    # needs the resolution (for the network pieces, the scale and the subdivision) and the coarse
    # epoch it was solved against. Carrying them on the solution rather than reaching into the
    # recursion for them is what keeps `solve_level` a pure function of its Subproblem.
    resolution: Optional[object] = None
    epoch_duration: Optional[float] = None

    def pieces_as_flows(self, cell_id: int, band: int) -> List[IntraFlow]:
        """Re-read this level's inter-cell pieces as intra flows of the cell one level up.

        `send_epoch` becomes the round offset because a piece leaving in this level's epoch k is,
        from the parent's point of view, a transfer in round k of the band it granted.
        """
        return [IntraFlow(
            cell=cell_id, identity=p.identity, sender=p.egress_gpu, receiver=p.ingress_gpu,
            via_switch=p.via_switches[0], volume=p.volume, band=band,
            local_round=int(p.send_epoch), kind="sublevel_transfer", hard=False)
            for p in self.pieces]


class CoarseSolution:
    """The protocol `reconstruct._extract_pieces` duck-types, made explicit.

    Any level solver -- the LP, the MILP, a closed-form crossbar -- is consumable by step A as long
    as it can say what flowed where and when. Naming it here is what lets `solve_flat` dispatch
    between solvers without the lowering half caring which one ran. The test suite already relied on
    this seam via a `SimpleNamespace` shim; this is the same thing with a name and a docstring.
    """

    def __init__(self, per_chunk_flow_paths, topology, epoch_duration: float,
                 preserves_identity: bool = False) -> None:
        self.per_chunk_flow_paths = per_chunk_flow_paths
        self.topology = topology
        self.epoch_duration = epoch_duration
        # Which step-A variant this solver's output needs. False = the solution is identity-free and
        # `assign_identities_free` must recover identity from it; True = the solver kept identity and
        # the assignment is read off directly (which is also what keeps Q == 1 at that level).
        self.preserves_identity = preserves_identity
