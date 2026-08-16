# Open-Source ROADEF IRP Solver: State and User Guide

## Mission and validity contract

This project is building a deterministic, open-source replacement for the
proprietary ROADEF/EURO 2016 inventory-routing solver.

Production solver runs must satisfy three constraints:

1. No instance names, customer IDs, copied routes, or manual schedules in
   production policy.
2. A `native-cold-start` run receives only the instance and seed/configuration.
3. Local replay is necessary but insufficient. Published validity requires the
   released V2 checker to accept the exact output XML.

The released checker archive used by the project has SHA-256:

```text
fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a
```

## Current unified-engine status

On 2026-08-16, the connected engine was audited on all 15 Set B instances with
a 1,800-second feasibility budget per instance. Runs stopped when local
feasibility was first reached; the exact XML was then submitted to the released
checker. The audit established **6/15 officially valid native cold starts** and
zero local/official checker disagreements.

| Instance | Days | Customers | Drivers / Trailers | Result | Time | Errors or official LR |
|---|---:|---:|---:|---|---:|---:|
| `V2.13` | 10 | 53 | 5 / 5 | Officially valid | 1 s | LR `0.077477` |
| `V2.14` | 35 | 53 | 5 / 5 | Officially valid | 95 s | LR `0.096214` |
| `V2.16.2` | 10 | 184 | 7 / 4 | Officially valid | 1,762 s | LR `0.029289` |
| `V2.24` | 10 | 32 | 5 / 6 | Officially valid | 3 s | LR `0.025699` |
| `V2.25` | 35 | 32 | 5 / 6 | Officially valid | 16 s | LR `0.033212` |
| `V2.26` | 35 | 32 | 5 / 6 | Officially valid | 16 s | LR `0.044782` |
| `V2.15` | 10 | 134 | 4 / 3 | Near-feasible | 1,804 s | 12 `QS02` |
| `V2.20.2` | 35 | 184 | 7 / 4 | Near-feasible | 1,915 s | 3 `QS02` |
| `V2.21.2` | 35 | 184 | 7 / 4 | Near-feasible | 1,920 s | 6 `QS02` |
| `V2.19` | 35 | 53 | 5 / 5 | Incomplete | 1,823 s | 52 `QS02` |
| `V2.12` | 10 | 324 | 13 / 15 | Constructor failure | 1,848 s | 297 errors |
| `V2.17` | 35 | 134 | 4 / 3 | Constructor failure | 1,828 s | 7,348 errors |
| `V2.18` | 35 | 134 | 4 / 3 | Constructor failure | 1,806 s | 8,020 errors |
| `V2.22` | 21 | 324 | 13 / 15 | Constructor failure | 1,811 s | 6,706 errors |
| `V2.23` | 21 | 324 | 13 / 15 | Constructor failure | 1,811 s | 2,547 errors |

See [UNIFIED_SET_B_30MIN_AUDIT.md](UNIFIED_SET_B_30MIN_AUDIT.md) for the
methodology, violation vectors, interpretation, and next experiments.

Saved reference or repaired artifacts are valuable regression fixtures, but
they do not change the unified cold-start count unless the general entry point
reproduces them from raw instance XML.

## Connected solver architecture

The entry point is `src/vrp_solver/solver/unified_engine.py`.

```text
raw instance
  -> seeded cluster constructions
  -> fingerprinted diagnostic frontier (ViolationVector)
  -> hard full-horizon quantity repair
  -> bounded surgical topology search
       -> pressure blocks and internal/tail insertions
       -> route recombination and ejection
       -> multi-reload candidates
       -> connected resource-block retiming
       -> hard quantity repair and global replay
  -> local feasibility
  -> exact XML serialization
  -> released-checker verification
```

The engine defaults to feasibility-first behavior:

- stop seed generation on the first locally feasible construction;
- skip quantity and topology polishing when already feasible;
- stop repaired-frontier evaluation on feasibility;
- stop surgical search on the first feasible incumbent.

Set `stop_when_feasible=False` only for explicit LR-improvement experiments.
The `SolverResult.valid` field means **locally feasible**, while
`validation_status` makes that scope explicit. Official validity is established
only after saving and checking the exact XML.

## Running the solver

### Python API

```python
from vrp_solver.solver.unified_engine import solve_cold_start
from vrp_solver.xml_io import load_instance, save_solution

instance_path = (
    "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.13.xml"
)
instance = load_instance(instance_path)
result = solve_cold_start(
    instance,
    num_seeds=10,
    time_limit=1_800,
    stop_when_feasible=True,
)
save_solution(result.solution, "my_solution.xml")
print(result.validation_status, result.errors, result.runtime_seconds)
```

### Official verification

```bash
.venv/bin/vrp-solver verify-official \
  roadef_2016_data/set_B/Instances_B_V25-11042016/V2.13.xml \
  my_solution.xml
```

### Full Set B audit

The audit prioritizes the five high-confidence instances, uses three isolated
workers, stops on feasibility, checkpoints after every instance, and invokes
the official checker only for locally feasible outputs.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python3 -u scratch/audit_unified_set_b.py 1800 3
```

Generated audit XML and JSON are written under
`/private/tmp/vrp-unified-set-b-audit/` and do not overwrite repository
benchmark artifacts.

## Development priorities

1. Close `V2.20.2`, `V2.21.2`, and `V2.15`. They have only 3–12 safety errors
   and no physical, call-in, or resource violations.
2. Test Gurobi on the near-feasible group. Quantity repair and shift selection
   already support `ROADEF_SOLVER=gurobi`; connected timing remains HiGHS-only.
3. Treat `V2.19` as a moderate topology-density problem rather than a
   quantity-only repair.
4. Redesign construction for the collapse group using compatibility-aware
   resource reservation, denser multi-stop chains, proactive reloads, and
   periodic exact repair during construction.
5. Add a hard process-level audit timeout. The internal deadline can be
   exceeded by an in-flight exact model.
