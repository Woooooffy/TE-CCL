"""Unit tests for the stitch's network-layer pacing check (check_network_pacing).

The check reports which paced sends no clock pins to their coarse epoch. It must stay the
mirror of the ncclize gate manifest (teccl_ncclize._finish_before_start_gates, covered by
teccl/ncclize/pacing_gates_test.py) so that what it reports is exactly what that manifest
cannot pin: P2 (one of the GPU's own paced sends completing at k) and P3 (a paced delivery
arriving at the sending GPU at k). Both pools are per GPU, not per uplink.

The function is extracted by AST rather than imported, so these run in a bare env: the
hierarchy package pulls in numpy and gurobipy at import time for everything else.

Run from the repo root:
    python teccl/hierarchy/pacing_check_test.py
"""
import ast, types
from collections import defaultdict
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

src = open('teccl/hierarchy/flat_schedule.py').read()
fn = next(n for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name == 'check_network_pacing')
ns = dict(defaultdict=defaultdict, Dict=Dict, Tuple=Tuple, Optional=Optional, List=List)
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<fn>', 'exec'), ns)
check = ns['check_network_pacing']

@dataclass
class P:
    egress_gpu: int; ingress_gpu: int; via_switches: tuple
    send_epoch: int; volume: float; rate: float
class Res:
    def __init__(self, pieces):
        self.pieces = pieces
        self.scale = types.SimpleNamespace(bytes_per_chunk=1.0)

E = 1.0   # coarse_epoch; volume 1 at rate 1 -> duration 1

def run(name, pieces, expect):
    got = check(Res(pieces), E)
    assert got == expect, f'{name}: expected {expect}, got {got}'
    print(f'  [OK] {name}')

# The delivering gpu7 needs its OWN clock in these fixtures, or it reports a residual of its
# own and drowns out what is under test. A send at epoch 0 (held by P1) chained to one at
# epoch 1 gives it that; the epoch-1 send lands at gpu0 at epoch 2.
def deliverer(via):
    return [P(7, 4, via, 0, 1, 1), P(7, 0, via, 1, 1, 1)]

run('P2 pins a contiguous send',
    [P(0,5,(9,),0,1,1), P(0,5,(9,),1,1,1)], [])

run('no clock at k -> residual',
    [P(0,5,(9,),0,1,1), P(0,5,(9,),2,1,1)], [(0,2)])

run('P3 arrival at k pins the send',
    [P(0,5,(9,),0,1,1), P(0,5,(9,),2,1,1)] + deliverer((9,)), [])

run('P3 arrival at the wrong epoch does not pin',
    [P(0,5,(9,),0,1,1), P(0,5,(9,),3,1,1)] + deliverer((9,)), [(0,3)])

run('P3 pins across a different uplink',
    [P(0,5,(9,),2,1,1)] + deliverer((8,)), [])

# P2 is per GPU: gpu0's send on uplink 8 completes at 2 and pins its send on uplink 9 at 2.
# Grouping P2 per uplink would report (0, 2) here -- the dual-plane/multi-rail gap.
run('P2 pins across the GPU\'s other uplink',
    [P(0,5,(8,),0,1,1), P(0,5,(8,),1,1,1), P(0,6,(9,),2,1,1)], [])

# Same epoch on two uplinks is concurrency, not a clock: neither pins the other, and the
# epoch-2 pair has nothing finishing at 2.
run('P2 same-epoch sends on two uplinks do not pin each other',
    [P(0,5,(8,),0,1,1), P(0,5,(8,),2,1,1), P(0,6,(9,),2,1,1)], [(0,2)])

run('epoch 0 is never a residual', [P(0,5,(9,),0,1,1)], [])

run('unpaced flows supply no clock',
    [P(0,5,(9,),0,1,1), P(0,5,(9,),2,1,1),
     P(7,4,(9,),0,1,1), P(7,0,(9,),1,1,None)], [(0,2)])

run('residual reported once per (gpu, epoch)',
    [P(0,5,(9,),0,1,1), P(0,5,(9,),2,1,1), P(0,6,(8,),2,1,1)], [(0,2)])

print('check_network_pacing tests OK')
