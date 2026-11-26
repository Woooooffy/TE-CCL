"""
One physical transmission, at every stage of a schedule.

A `Send` is what one non-switch node puts on its link once: a chunk identity, streamed at one pace
for `span` epochs, fanning out through switches to one or more receivers (a `Branch` each). The
same type carries what a MILP chose (in that level's node ids and epochs), what a hierarchy level
resolved onto real GPUs, and what the stitch lays on the absolute fine axis -- only the index space
and the time unit change. Unicast is a send with one branch; a switch copy is a send with several.

Every link of a send is held for the same `span`: a cut-through traversal keeps one pace class end
to end, and a switch replicates a stream at the rate it receives it, so the slowest link of the tree
paces all of it. Rates are not stored -- `volume * bytes_per_chunk / (span * epoch_duration)` is the
rate in whichever units the holder works in.

See teccl/hierarchy/switch_copy_design.md.
"""
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

Identity = Tuple[int, int]
Link = Tuple[int, int]


@dataclass(frozen=True, order=True)
class Branch:
    """Where one copy of a send lands."""
    receiver: int                  # non-switch node the copy lands on
    via: Tuple[int, ...] = ()      # switches from the sender to the receiver, in order; () = direct
    ready: int = 0                 # first epoch the receiver HOLDS the data (the holder's units)


@dataclass(frozen=True)
class Send:
    """One stream from `sender`, starting in `epoch`, holding every link of its tree for `span`."""
    identity: Identity             # (source, chunk) in the holder's labelling
    sender: int                    # non-switch node that transmits
    epoch: int                     # epoch the head of the stream starts (the holder's units)
    fanout: Tuple[Branch, ...]     # >= 1 branch, sorted by receiver
    span: int = 1                  # epochs every link of the send is held (pace class + 1)
    volume: float = 1.0
    kind: str = "network"          # network | egress_stage | ingress_distribution | ...

    def __post_init__(self):
        if not self.fanout:
            raise ValueError(f"send of {self.identity} from {self.sender} has no branch")
        if self.span < 1:
            raise ValueError(f"send of {self.identity} from {self.sender} has span {self.span}")
        fanout = tuple(sorted(self.fanout))
        object.__setattr__(self, "fanout", fanout)
        receivers = [b.receiver for b in fanout]
        if len(set(receivers)) != len(receivers):
            raise ValueError(f"send of {self.identity} from {self.sender} lands twice on one "
                             f"receiver: {receivers}")
        # One physical transmission: every branch leaves on the same first link, and a switch is
        # entered from one place only (the branches' paths form a tree rooted at the sender).
        upstream: Dict[int, int] = {}
        for b in fanout:
            prev = self.sender
            for node in b.via + (b.receiver,):
                if upstream.setdefault(node, prev) != prev:
                    raise ValueError(
                        f"send of {self.identity} from {self.sender} is not one transmission: "
                        f"{node} is reached from both {upstream[node]} and {prev} "
                        f"(branches {fanout})")
                prev = node
        if len({first_hop(self.sender, b) for b in fanout}) > 1:
            raise ValueError(f"send of {self.identity} from {self.sender} leaves on more than one "
                             f"link: {fanout}")


def first_hop(sender: int, branch: Branch) -> Link:
    """The link a branch leaves its sender on."""
    return (sender, branch.via[0] if branch.via else branch.receiver)


def hops(sender: int, branch: Branch) -> List[Link]:
    """The links of one branch, in order."""
    nodes = (sender,) + branch.via + (branch.receiver,)
    return list(zip(nodes, nodes[1:]))


def first_link(send: Send) -> Link:
    """The link every branch of `send` leaves on: the sender's uplink."""
    return first_hop(send.sender, send.fanout[0])


def links(send: Send) -> Set[Link]:
    """Every link the send's tree uses, once each -- what one transmission charges."""
    return {link for b in send.fanout for link in hops(send.sender, b)}


def link_windows(send: Send) -> Dict[Link, range]:
    """The epochs each link of the send is held, in the holder's units.

    A send holds every link for `span` epochs. The first link starts at `send.epoch`; a branch's
    last link ends at `branch.ready` (the receiver holds the data from then on). Links in between
    are given the first link's window. That is exact whenever propagation rounds to zero epochs
    (every hierarchy level and every SLOWEST_LINK solve); with propagation a later link's window
    starts up to its alpha later.
    """
    windows: Dict[Link, range] = {}
    for b in send.fanout:
        path = hops(send.sender, b)
        for n, link in enumerate(path):
            start = b.ready - send.span if n == len(path) - 1 and n > 0 else send.epoch
            windows.setdefault(link, range(start, start + send.span))
    return windows


def rate(send: Send, bytes_per_chunk: float, epoch_duration: float) -> float:
    """The send's pacing rate: its volume spread over the `span` epochs it holds each link. The
    self-clocking rule every level paces by -- a send finishes exactly at an epoch boundary."""
    return send.volume * bytes_per_chunk / (send.span * epoch_duration)


def receivers(send: Send) -> Tuple[int, ...]:
    """The destination set -- what a multicast op names."""
    return tuple(b.receiver for b in send.fanout)


def switch_out(send: Send) -> Dict[int, Tuple[int, ...]]:
    """switch -> the next nodes it forwards this send to: its output ports for this transmission."""
    out: Dict[int, Set[int]] = defaultdict(set)
    for b in send.fanout:
        for u, v in hops(send.sender, b):
            if u != send.sender:
                out[u].add(v)
    return {sw: tuple(sorted(nxt)) for sw, nxt in sorted(out.items())}


def delivery_index(sends: Iterable[Send]) -> Dict[Tuple[Identity, int], Tuple[Send, Branch]]:
    """(identity, node) -> the (send, branch) that delivers it there EARLIEST (ties: the earlier
    send).

    On a forest of sends -- every node receives an identity at most once -- this is the parent
    pointer: the send that delivered `send.identity` to `send.sender` is `index[(identity, sender)]`.
    """
    index: Dict[Tuple[Identity, int], Tuple[Send, Branch]] = {}
    for s in sends:
        for b in s.fanout:
            key = (s.identity, b.receiver)
            prev = index.get(key)
            if prev is None or (b.ready, s.epoch) < (prev[1].ready, prev[0].epoch):
                index[key] = (s, b)
    return index


# Field order of a "10-Sends" record and of its branches as written to a schedule file: who sends
# what and when first, the fan-out last, so a record reads top-down.
SEND_FIELDS = ("sender", "from", "chunk", "epoch", "span", "rate", "volume", "branches")
BRANCH_FIELDS = ("to", "via", "ready")


def schedule_json(info: Dict) -> str:
    """A schedule dict as the text of a schedule file.

    Top-level sections go in the numeric order of their prefix ("2-..." before "10-Sends"; unnumbered
    ones such as Solver_Time last), where json's sort_keys would order them as strings. "10-Sends"
    records and their branches follow SEND_FIELDS / BRANCH_FIELDS; every other section is sorted
    recursively, exactly as sort_keys wrote it before."""
    def section(key: str):
        head = key.split("-", 1)[0]
        return (int(head), key) if head.isdigit() else (float("inf"), key)

    def ordered(record: Dict, fields: Tuple[str, ...]) -> Dict:
        return {**{k: record[k] for k in fields if k in record},
                **{k: record[k] for k in sorted(record) if k not in fields}}

    def sort_all(x):
        if isinstance(x, dict):
            return {k: sort_all(x[k]) for k in sorted(x)}
        if isinstance(x, list):
            return [sort_all(v) for v in x]
        return x

    out = {}
    for key in sorted(info, key=section):
        if key == "10-Sends":
            out[key] = [{**ordered(s, SEND_FIELDS),
                         "branches": [ordered(b, BRANCH_FIELDS) for b in s["branches"]]}
                        for s in info[key]]
        else:
            out[key] = sort_all(info[key])
    return json.dumps(out, indent=2)
