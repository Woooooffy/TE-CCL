"""
Chunk granularity for one level of the hierarchical (recursive) solve.

The coarse solve and the intra-cell solve are the SAME problem at different levels: given a
topology and a demand set (what volume must go where), produce flows. The governing invariant of
that recursion is that **every level receives integer demands, expressed in that level's own chunk
unit** -- which is achievable only because refinement is free: a chunk is a bookkeeping unit over
contiguous bytes, so subdividing one is always legal.

Fractional volumes are introduced only by a RELAXATION at some level (the coarse LP splitting a
commodity across parallel paths; the abstraction summing several fine links into one coarse link so
that even an integral coarse flow does not decompose integrally) and are absorbed at the boundary
immediately below it, by refining the chunk. `ChunkScale` is what makes that refinement explicit
and auditable instead of a silently redefined `chunk_size`.

`Topology.chunk_size` stays the ROOT value. Everything downstream of a refinement -- the fine
epoch duration, the epoch-per-coarse-epoch factor, a schedule's "9-Chunk_Size", the algorithmic
bandwidth -- must be derived from the live ChunkScale rather than read off the topology, or the
numbers silently desync from the demands they describe.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ChunkScale:
    """The unit in which one level's demand volumes are expressed.

    INVARIANT: bytes_per_chunk * num_chunks is the per-GPU payload and is CONSTANT across levels.
    Refinement trades chunk size for chunk count; it never creates or destroys bytes.

    refinement_from_root accumulates the product of every Q applied so far. It is the factor that
    ultimately reaches ncclize's `chunk_up()`, so ncclize's MAX_M bounds the PRODUCT over all
    levels -- each level's Q spends from one shared budget, consumed only where a relaxation
    actually happened (a memoized NVSwitch level, whose edge-colouring of integer demands stays
    integral, spends nothing).
    """
    bytes_per_chunk: float
    num_chunks: int
    refinement_from_root: int = 1

    @property
    def payload_per_gpu(self) -> float:
        """The conserved quantity. Equal at every level; assert against it after refining."""
        return self.bytes_per_chunk * self.num_chunks

    def refine(self, q: int) -> "ChunkScale":
        """Split each chunk into `q` sub-chunks. The only legal transformation."""
        if q < 1 or int(q) != q:
            raise ValueError(f"refinement factor must be a positive integer, got {q!r}")
        q = int(q)
        if q == 1:
            return self
        return replace(self,
                       bytes_per_chunk=self.bytes_per_chunk / q,
                       num_chunks=self.num_chunks * q,
                       refinement_from_root=self.refinement_from_root * q)

    def epoch_duration(self, max_link_bw: float) -> float:
        """Fine epoch length under the FASTEST_LINK convention: the time for one chunk (at THIS
        level's size) to cross the fastest link. Shrinks as the scale refines, which is why it must
        be derived here rather than written down -- see the module docstring."""
        if max_link_bw <= 0:
            raise ValueError(f"max_link_bw must be positive, got {max_link_bw!r}")
        return self.bytes_per_chunk / max_link_bw

    def assert_conserves(self, other: "ChunkScale", tol: float = 1e-9) -> None:
        """Refinement must leave the per-GPU payload untouched."""
        a, b = self.payload_per_gpu, other.payload_per_gpu
        if abs(a - b) > tol * max(1.0, abs(a)):
            raise AssertionError(
                f"ChunkScale refinement did not conserve payload: {a} -> {b} "
                f"({self} -> {other})")

    def __str__(self) -> str:
        return (f"ChunkScale(bytes={self.bytes_per_chunk:g}, chunks={self.num_chunks}, "
                f"refined x{self.refinement_from_root})")
