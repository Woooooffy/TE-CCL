"""
Pure demand-tensor builders, shared by BaseFormulation's demand generators and by the
hierarchical demand coarsification (teccl.hierarchy.abstract.coarsify_demand).

A demand tensor is ``demand[s][t][c]`` = volume of data identity ``(source s, chunk c)`` that
destination ``t`` requires. This is the general "what volume goes from where to where"
representation of a collective; the solvers never branch on collective type, they just satisfy
the tensor. Keeping the builders here -- Gurobi-free -- lets the coarsification and its tests
build demand without constructing a solver model. BaseFormulation just assigns the result.
"""
from collections import defaultdict

import numpy as np

from teccl.input_data import Collective
from teccl.topologies.topology import Topology


def _demand_sources(topology: Topology):
    """GPUs that can source/sink demand: not switch, not passive."""
    n = len(topology.capacity)
    switch = set(topology.switch_indices)
    passive = set(topology.passive_indices)
    return [d for d in range(n) if d not in switch and d not in passive]


def all_gather_demand(topology: Topology, num_chunks: int) -> np.ndarray:
    """AllGather: every participating GPU's chunks are wanted by every other participating GPU.
    Each source's chunk c is a distinct global identity (the source index distinguishes them)."""
    n = len(topology.capacity)
    parts = _demand_sources(topology)
    demand = np.zeros((n, n, num_chunks), dtype=np.int32)
    for s in parts:
        for t in parts:
            if s == t:
                continue
            demand[s][t][:] = 1
    return demand


def all_to_all_demand(topology: Topology, num_chunks: int) -> np.ndarray:
    """AllToAll: each ordered GPU pair (s, t) exchanges distinct data. The chunk index encodes
    the destination (device_chunk_map[t] + c*gpus), so (s, chunk) identifies both endpoints --
    unlike AllGather these identities are per-destination-GPU distinct (no fan-out sharing)."""
    n = len(topology.capacity)
    parts = _demand_sources(topology)
    gpus = len(parts)
    device_chunk_map = defaultdict(int)
    for i, d in enumerate(parts):
        device_chunk_map[d] = i
    demand = np.zeros((n, n, num_chunks), dtype=np.int32)
    for s in parts:
        for t in parts:
            if s == t:
                continue
            for c in range(num_chunks // gpus):
                demand[s][t][device_chunk_map[t] + c * gpus] = 1
    return demand


def gather_demand(topology: Topology, num_chunks: int, root: int) -> np.ndarray:
    """Gather: every participating GPU (except root) sends its own distinct data to root."""
    n = len(topology.capacity)
    parts = _demand_sources(topology)
    demand = np.zeros((n, n, num_chunks), dtype=np.int32)
    for s in parts:
        if s == root:
            continue
        demand[s][root][:] = 1
    return demand


def broadcast_demand(topology: Topology, num_chunks: int, root: int) -> np.ndarray:
    """Broadcast: root sends its data to every other participating GPU (all want the same)."""
    n = len(topology.capacity)
    parts = _demand_sources(topology)
    demand = np.zeros((n, n, num_chunks), dtype=np.int32)
    for t in parts:
        if t == root:
            continue
        demand[root][t][:] = 1
    return demand


def build_demand(collective: Collective, topology: Topology, num_chunks: int,
                 root: int = 0) -> np.ndarray:
    """Dispatch to the builder for `collective`. Single source of truth for the fine demand,
    used by BaseFormulation and by the hierarchical coarsification/driver/tests."""
    if collective == Collective.ALLGATHER:
        return all_gather_demand(topology, num_chunks)
    if collective == Collective.ALLTOALL:
        return all_to_all_demand(topology, num_chunks)
    if collective == Collective.GATHER:
        return gather_demand(topology, num_chunks, root)
    if collective == Collective.BROADCAST:
        return broadcast_demand(topology, num_chunks, root)
    raise ValueError(f"Demand type {collective} is not expected")
