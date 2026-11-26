"""
What a route is, for ncclize: the physical path one transmission takes through the network.

A route is identified by its `path_key` on the ops that use it:

  * a UNICAST route's key is what it always was -- the raw switch sequence (None for a direct
    link), or a port-qualified ((switches...), (cables...)) pair (port_split);
  * a MULTICAST route's key is a `Tree`: the switch path to every receiver of one transmission.

Flow ids (`mscclflowid`) are a bijection with routes, and channels are partitioned by route, so
both the switch table and the channel allocator key on the same object. The helpers below are the
only place that tells the two kinds apart. See multicast_output_design.md.
"""
from dataclasses import dataclass
from typing import Optional, Set, Tuple

try:                                  # sibling module; this file runs both as a script
    from port_split import unqualify_path_key                            # noqa: E402
except ImportError:                   # ... and as part of the teccl package
    from teccl.ncclize.port_split import unqualify_path_key


@dataclass(frozen=True)
class Tree:
    """A multicast route: one transmission from one GPU, replicated by switches to several.

    `branches` is ((receiver, switches), ...), sorted by receiver: `receiver` is the receiving
    GPU's DENSE rank (the ids ops carry) and `switches` the RAW switch path to it. Every branch
    leaves the sender through the same first switch. Two transmissions take the same tree iff they
    leave the same GPU over the same branches -- so (sender, Tree) is the route's identity.
    """
    branches: Tuple[Tuple[int, Tuple[int, ...]], ...]

    def __post_init__(self):
        if len(self.branches) < 2:
            raise ValueError(f"a Tree has at least two branches; a single branch is a unicast "
                             f"route and keeps its unicast key: {self.branches}")
        if len({switches[0] for _r, switches in self.branches if switches}) > 1:
            raise ValueError(f"multicast branches leave on different first switches: "
                             f"{self.branches}")

    @property
    def receivers(self) -> Tuple[int, ...]:
        return tuple(r for r, _s in self.branches)

    def switches_to(self, receiver: int) -> Tuple[int, ...]:
        return dict(self.branches)[receiver]


def is_tree(path_key) -> bool:
    return isinstance(path_key, Tree)


def receivers_of(peer) -> Tuple[int, ...]:
    """The receivers an op's `peer` names: a multicast send's peer is the receiver tuple."""
    return tuple(peer) if isinstance(peer, tuple) else (peer,)


def first_hop(path_key) -> Tuple[Optional[int], int]:
    """(first switch, cable) a route leaves its sender on; (None, 0) for a direct link."""
    if is_tree(path_key):
        return path_key.branches[0][1][0], 0
    switches, cables = unqualify_path_key(path_key)
    return (switches[0] if switches else None), (cables[0] if cables else 0)


def last_hop(path_key, receiver: int) -> Tuple[Optional[int], int]:
    """(last switch, cable) a route lands on `receiver` through."""
    if is_tree(path_key):
        switches = path_key.switches_to(receiver)
        return (switches[-1] if switches else None), 0
    switches, cables = unqualify_path_key(path_key)
    return (switches[-1] if switches else None), (cables[-1] if cables else 0)


def route_links(src: int, peer, path_key) -> Set[Tuple[int, int, int]]:
    """Every directed link a route occupies, as (from, to, cable), once each."""
    if is_tree(path_key):
        out = set()
        for receiver, switches in path_key.branches:
            nodes = [src, *switches, receiver]
            out |= {(u, v, 0) for u, v in zip(nodes, nodes[1:])}
        return out
    switches, cables = unqualify_path_key(path_key)
    nodes = [src, *(switches or ()), peer]
    cables = cables or (0,) * (len(nodes) - 1)
    return {(u, v, c) for u, v, c in zip(nodes, nodes[1:], cables)}
