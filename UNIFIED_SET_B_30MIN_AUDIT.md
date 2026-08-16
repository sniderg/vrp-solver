# Unified Set B 30-Minute Native Cold-Start Audit

## Scope

Date: 2026-08-16

This audit measured the present validity of the connected unified solver, not
the validity of saved reference, oracle, or native-repair artifacts.

Every run had provenance `native-cold-start`: its only routing input was the raw
instance XML plus deterministic solver configuration. The configuration was:

```text
num_seeds=10
frontier_size=3
search_iterations=64
search_workers=1
time_limit=1800 seconds per instance
stop_when_feasible=True
outer audit workers=3
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python3 -u scratch/audit_unified_set_b.py 1800 3
```

The five high-confidence instances (`V2.13`, `V2.14`, `V2.24`, `V2.25`, and
`V2.26`) were queued first. Locally invalid candidates were not submitted to the
official checker. Locally feasible candidates were serialized and checked using
the released V2 checker archive with SHA-256:

```text
fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a
```

## Results

| Instance | Official | Solver seconds | Shifts | Operations | Seeds | Repairs | Search steps | Final diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `V2.13` | valid | 1.28 | 19 | 86 | 1 | 0 | 0 | LR `0.077477` |
| `V2.14` | valid | 94.56 | 71 | 293 | 4 | 0 | 0 | LR `0.096214` |
| `V2.16.2` | valid | 1,761.75 | 36 | 218 | 10 | 3 | 14 | LR `0.029289` |
| `V2.24` | valid | 2.59 | 31 | 94 | 5 | 0 | 0 | LR `0.025699` |
| `V2.25` | valid | 15.86 | 84 | 246 | 1 | 0 | 0 | LR `0.033212` |
| `V2.26` | valid | 16.01 | 92 | 342 | 1 | 0 | 0 | LR `0.044782` |
| `V2.15` | not run | 1,803.52 | 21 | 83 | 10 | 3 | 27 | 12 `QS02`; safety area 3,268.69 |
| `V2.20.2` | not run | 1,914.83 | 157 | 904 | 10 | 3 | 3 | 3 `QS02`; safety area 8,928.59 |
| `V2.21.2` | not run | 1,920.39 | 159 | 961 | 10 | 3 | 3 | 6 `QS02`; safety area 9,623.39 |
| `V2.19` | not run | 1,823.15 | 65 | 263 | 10 | 3 | 8 | 52 `QS02`; safety area 337,668.83 |
| `V2.12` | not run | 1,847.95 | 77 | 561 | 10 | 3 | 3 | 297 errors; 28 unscheduled |
| `V2.17` | not run | 1,827.97 | 75 | 303 | 10 | 3 | 8 | 7,348 errors; 73 unscheduled |
| `V2.18` | not run | 1,806.29 | 73 | 292 | 10 | 3 | 12 | 8,020 errors; 77 unscheduled |
| `V2.22` | not run | 1,810.56 | 153 | 1,163 | 10 | 3 | 3 | 6,706 errors; 119 unscheduled |
| `V2.23` | not run | 1,810.79 | 163 | 1,127 | 10 | 3 | 3 | 2,547 errors; 82 unscheduled |

Officially valid native cold starts: **6/15**. Local/official disagreements:
**0**.

The wall clock can exceed 1,800 seconds because an exact model already in
flight may return after the orchestration deadline. Observed solver overruns
were under approximately two minutes in this audit. A process-level timeout is
still required for a strict external SLA.

## What separated passes from failures

### Direct-construction successes

`V2.13`, `V2.14`, and `V2.24`–`V2.26` reached feasibility in the constructor.
They required no quantity repair and no topology-search step. These instances
have 32–53 customers and trailer compatibility ratios from 0.883 to 1.000.

Call-ins alone are not the blocker: `V2.24`–`V2.26` each contain 30 orders, but
their small customer set and complete compatibility make resource chaining
easy.

### Search-connected success

`V2.16.2` is the strongest evidence that the connected pipeline adds value. It
used all ten constructions, three exact quantity repairs, and 14 surgical steps
before reaching a solution accepted by the released checker.

### Near-feasible failures

`V2.15`, `V2.20.2`, and `V2.21.2` ended with only 3–12 safety violations and no
physical, call-in, or resource-timing errors. Their smallest demonstrated
blocker is a handful of early deliveries/load-path exchanges. `V2.19` is the
same class at larger magnitude.

### Constructor-collapse failures

`V2.12`, `V2.17`, `V2.18`, `V2.22`, and `V2.23` retained 28–119 unscheduled
customers and large negative-inventory/safety areas. Quantity repair cannot
create the missing route topology. These instances need compatibility-aware
resource reservation, multi-reload route density, and rolling construction
repair rather than a larger terminal LP budget.

## Solver-backend implication

HiGHS returned `Unknown` or per-model time limits on some long-horizon repair
models. This is not feasibility and was correctly rejected. The next controlled
experiment should rerun the near-feasible group with
`ROADEF_SOLVER=gurobi`. Full-horizon quantity repair and shift selection already
support that backend; joint block timing does not.

If Gurobi closes the near-feasible cases, implement HiGHS-first fallback on
unresolved statuses. If it does not, prioritize topology operators rather than
solver substitution.
