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

REFINEMENT AND COARSENING ARE THE SAME MOVE IN OPPOSITE DIRECTIONS. Descending a level (identity
resolution lowering a coarse solution onto fine GPUs) refines; ASCENDING one (abstraction collapsing
cells into coarse nodes) coarsens, because a coarse node's natural unit is its whole payload, not
the fine chunk it happens to be built from. Both are legal for the same reason -- a chunk is a
bookkeeping unit over contiguous bytes -- and both conserve `payload_per_gpu`, so a level boundary
can be crossed in either direction without the byte accounting drifting. `refinement_from_root` is
therefore a RATIO, not a count: coarsening by g credits the budget back, so the net value is the
honest measure of how much relaxation actually happened across the whole recursion, and it is the
net that ncclize's chunk_up() has to pay for.
"""
from dataclasses import dataclass, field, replace
from fractions import Fraction


@dataclass(frozen=True)
class ChunkScale:
    """The unit in which one level's demand volumes are expressed.

    INVARIANT: bytes_per_chunk * num_chunks is the per-GPU payload and is CONSTANT across levels.
    Refinement trades chunk size for chunk count; it never creates or destroys bytes. Coarsening
    trades the other way and conserves it just as exactly.

    `num_chunks` is always "chunks AT THIS LEVEL'S SIZE per FINE GPU", which is why it is a
    Fraction rather than an int: at a coarse level one chunk spans several fine GPUs' payloads, so
    a single fine GPU owns a fraction of one. Rail's 8-GPU host at a coarse chunk of 8 GB gives
    num_chunks = 1/8, and 8 GB * 1/8 = 1 GB is still exactly that GPU's payload.

    refinement_from_root accumulates every Q applied so far DIVIDED BY every g coarsened away. It
    is the factor that ultimately reaches ncclize's `chunk_up()`, so ncclize's MAX_M bounds the NET
    value over all levels -- each level's Q spends from one shared budget and each coarsening
    credits it back, so the budget is consumed only where a relaxation actually happened (a
    memoized NVSwitch level, whose edge-colouring of integer demands stays integral, spends
    nothing; a coarsen-by-8 followed by a clean refine-by-8 nets to 1 and spends nothing either).
    """
    bytes_per_chunk: float
    num_chunks: Fraction
    refinement_from_root: Fraction = field(default_factory=lambda: Fraction(1))

    def __post_init__(self) -> None:
        # Accept ints/floats at the boundary so callers (and the root construction from a demand
        # tensor's chunk axis) need not know this is rational underneath. Frozen dataclass, so the
        # coercion goes through object.__setattr__.
        object.__setattr__(self, "num_chunks", Fraction(self.num_chunks))
        object.__setattr__(self, "refinement_from_root", Fraction(self.refinement_from_root))

    @property
    def payload_per_gpu(self) -> float:
        """The conserved quantity. Equal at every level; assert against it after refining."""
        return float(self.bytes_per_chunk * self.num_chunks)

    def refine(self, q: int) -> "ChunkScale":
        """Split each chunk into `q` sub-chunks. Descends a level (or absorbs a relaxation)."""
        if q < 1 or int(q) != q:
            raise ValueError(f"refinement factor must be a positive integer, got {q!r}")
        q = int(q)
        if q == 1:
            return self
        return replace(self,
                       bytes_per_chunk=self.bytes_per_chunk / q,
                       num_chunks=self.num_chunks * q,
                       refinement_from_root=self.refinement_from_root * q)

    def coarsen(self, g: int) -> "ChunkScale":
        """Fuse `g` chunks into one. The exact inverse of refine, used when ASCENDING a level:
        after abstraction, a coarse node's demands are whole multiples of g fine chunks, so g fine
        chunks are what one coarse chunk should mean. g == 1 is an exact no-op, which is the
        graceful-degradation path when the coarse volumes share no common factor."""
        if g < 1 or int(g) != g:
            raise ValueError(f"coarsening factor must be a positive integer, got {g!r}")
        g = int(g)
        if g == 1:
            return self
        return replace(self,
                       bytes_per_chunk=self.bytes_per_chunk * g,
                       num_chunks=self.num_chunks / g,
                       refinement_from_root=self.refinement_from_root / g)

    def epoch_duration(self, link_bw: float) -> float:
        """Epoch length at THIS level: the time for one chunk (at this level's size) to cross a
        link of `link_bw`. The caller picks which link -- fastest or slowest, per the user's
        EpochType -- and picks it from THIS LEVEL'S OWN link set, which is why the bandwidth is a
        parameter rather than read off a topology here. (For a single-NVSwitch cell every internal
        link is identical, so the two conventions coincide and the choice is moot.) Shrinks as the
        scale refines and grows as it coarsens, which is why it must be derived here rather than
        written down -- see the module docstring."""
        if link_bw <= 0:
            raise ValueError(f"link_bw must be positive, got {link_bw!r}")
        return self.bytes_per_chunk / link_bw

    def to_json(self) -> dict:
        """JSON-safe view. `dataclasses.asdict` leaks raw Fractions, which no JSON encoder can
        write (and which a `default=list` fallback turns into a confusing TypeError rather than an
        obvious one), so serialize the ratios EXPLICITLY: exact as [numerator, denominator] plus a
        float alongside for anything that just wants to eyeball it. Exactness matters here -- these
        are the numbers that say how much refinement budget a level spent."""
        def ratio(f: Fraction) -> dict:
            return {"num": f.numerator, "den": f.denominator, "value": float(f)}
        return {
            "bytes_per_chunk": self.bytes_per_chunk,
            "num_chunks": ratio(self.num_chunks),
            "refinement_from_root": ratio(self.refinement_from_root),
            "payload_per_gpu": self.payload_per_gpu,
        }

    def assert_conserves(self, other: "ChunkScale", tol: float = 1e-9) -> None:
        """Refinement must leave the per-GPU payload untouched."""
        a, b = self.payload_per_gpu, other.payload_per_gpu
        if abs(a - b) > tol * max(1.0, abs(a)):
            raise AssertionError(
                f"ChunkScale refinement did not conserve payload: {a} -> {b} "
                f"({self} -> {other})")

    def __str__(self) -> str:
        # The net refinement is the interesting number, so show it as a plain integer when it is
        # one (the common case) and as a ratio only when a coarsening is still outstanding.
        r = self.refinement_from_root
        net = str(r.numerator) if r.denominator == 1 else f"{r.numerator}/{r.denominator}"
        return (f"ChunkScale(bytes={self.bytes_per_chunk:g}, chunks={self.num_chunks}, "
                f"refined x{net})")
