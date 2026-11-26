# Rail 256-GPU hierarchical MILP allgather, milp_chunking=2, warm-started (2026-10-21)

Schedules from two local runs (Apple M2, 8 cores, Gurobi 13.0.2, academic WLS license) on branch
`cell-relay-transit`, worktree `TE-CCL-switch-copy-send`, HEAD `fb36c48` plus UNCOMMITTED changes:

- `InstanceParams.objective_tier_time_limit` (teccl/input_data.py, teccl/solvers/base_formulation.py)
- `teccl/examples/hierarchy_leaf_broadcast_warmstart.py` (generates the MIP start)
- the two b2 inputs (copies in `inputs/`): `num_epochs` 19 -> 16, `objective_tier_time_limit`,
  `warmstart`

These are NOT runs of the solver as-is. Each started from a verified leaf-broadcast MIP start
(`warmstarts/<prefix>.mst`, not archived; regenerate it from the input with the script above). The
solver as-is (no start, 19 epochs, no tier limits) found no incumbent in 4 h on the remote.

| | single plane | dual plane |
|---|---|---|
| time limit (global; tiers makespan / latency / relay / chain) | 4 h; 0.1 / 2.5 / 0.5 / rest | 6 h; 0.1 / 4.0 / 0.5 / rest |
| Gurobi status | 13 (suboptimal: tier limits hit) | 13 |
| makespan tier | 16 epochs (= start; 16 is the ingress bound) | 16 (= start) |
| latency tier | -33728 (= start; bound -34688, gap 2.85%) | -33728 (= start; root LP unfinished at the 4 h cap) |
| host relay tier | 0, optimal | 0, optimal |
| switch chain tier | 128, optimal | 128, optimal |
| coarse finish time | 0.64 | 0.32 |
| flat finish time | 0.6572 | 0.3372 |

Every tier ended at the start's value, so these schedules are, up to Gurobi tie-breaking, the leaf
broadcast itself (not diffed against the start). The flat network-pacing check reported every paced send pinned on both.

`schedules/SHA256SUMS` fingerprints the files as produced. `logs/` holds the full Gurobi logs and the
driver output.
