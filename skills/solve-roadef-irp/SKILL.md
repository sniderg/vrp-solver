---
name: solve-roadef-irp
description: Design, implement, diagnose, benchmark, and independently validate native cold-start solvers for the ROADEF 2016 inventory-routing problem, especially Set B. Use for route/resource chaining, rolling-horizon or column-generation construction, ALNS and matheuristics, timing or quantity repair, call-in and VMI inventory, driver/trailer/source constraints, solver-backend comparisons, official-checker validity, and Set B qualification.
---

# Solve ROADEF IRP

Treat official validity as the first objective and LR as the objective only
among valid solutions.

## Preserve provenance and proof

Label inputs and outputs as `native-cold-start`, `native-repair`, `oracle`, or
`reference`. Never describe repair of a supplied candidate or imported
oracle/reference topology as a cold start.

Replay every candidate from initial state. Treat malformed data, overfill,
negative stock, safety breach, missed order, trailer discontinuity, illegal
timing, and resource overlap as hard failures. Local zero errors are necessary
but insufficient: publish validity only when the released checker accepts the
exact serialized XML through the fail-closed wrapper.

Record instance, provenance, seed/configuration, command, solver backends,
runtime, checker hash, local violation vector, official verdict, and valid LR.

## Inspect before changing the solver

Locate parsing, XML I/O, replay, validation, official-checker wrapper, solver
entry points, exact models, audit harnesses, tests, and current candidates.
Inspect working-tree changes and preserve unrelated work. Use existing entry
points and validators rather than creating competing versions.

## Classify the failure regime

Use structured violation magnitude and duration, not raw error count alone.
Classify a run before choosing an intervention:

- **Direct-construction feasible:** stop when the task is feasibility; do not
  spend the remaining budget polishing LR unless explicitly requested.
- **Near-feasible:** no physical/resource/call-in failures and a small number or
  area of inventory deficits. Use pressure-band donor exchange, internal/tail
  insertion, reload relocation, joint retiming, and hard quantity repair.
- **Moderate topology deficiency:** coverage exists but deficit area remains
  material. Expand to multi-route densification and fragment exchange.
- **Constructor collapse:** many unscheduled customers, missed orders, or large
  negative/safety areas. Do not rely on terminal quantity repair. Redesign
  construction with compatibility scarcity, dense columns, proactive reloads,
  and rolling lookahead.

## Use the chain-first matheuristic

Read [references/matheuristic.md](references/matheuristic.md) before changing
construction, topology, timing, quantities, acceptance, backend policy, or
qualification.

Use this atomic iteration:

`select connected block -> destroy -> rebuild route/resource chains -> topology screen -> joint timing repair -> hard full-horizon quantity repair -> global replay -> accept/reject`

Include affected driver/trailer predecessors and successors. Quantity-only
repair cannot create absent route topology. A direct emergency shift is a
fallback column, not the default move.

## Price compatibility and future feasibility

Treat usable capacity as compatible driver-trailer-source time, not free
driver-days. Rank demands using first breach, deficit area, window slack,
compatibility scarcity, load-path position, incremental travel, and future
resource opportunity cost.

For long or resource-tight horizons, prefer rolling construction with a
protected prefix and lookahead tail. Penalize terminal states by future demand
and reachability rather than generic tank fill.

## Screen before exact optimization

Before calling an LP/MIP, reject a topology that lacks:

- an active visit before the relevant breach/order deadline;
- sufficient cumulative compatible trailer capacity;
- a reachable compatible source or carried load;
- a legal driver/trailer chain gap including return and rest;
- a relocation path for displaced mandatory demand.

Log candidate funnel counts so time spent in generation, timing, quantity
repair, and replay is distinguishable.

## Treat solver backends as an experiment

Fail closed on `Unknown`, time limit without an accepted incumbent, numerical
failure, or ambiguous status. Do not infer that a stronger backend fixes a
topology problem.

When comparing HiGHS and Gurobi, hold construction, seed, topology sequence,
model, and budget fixed. Record status distributions, feasible-model counts,
violation-vector improvements, and time to official validity. Prefer
HiGHS-first fallback on unresolved models only after an ablation demonstrates
value. Keep the open-source path independently functional.

## Use hard budgets and honest telemetry

An internal deadline may be exceeded by an in-flight exact model. Use a
process-level timeout for a strict SLA and checkpoint after each instance.
Recompute final metrics from the returned solution; do not report stale
constructor fields as final coverage.

Maintain separate feasible and diagnostic archives. Run the official checker
when the first local-feasible XML appears, when the best valid LR improves, and
before publication—not for known-invalid candidates.

## Generalise and report precisely

Promote customer-specific probes into feature-driven operators and regression
tests. Never branch on instance name, customer ID, fixed day, or copied route.
Qualify repeated fixed seeds across call-in-heavy, VMI-heavy,
compatibility-sparse, resource-tight, reload/layover, small, and scale-extreme
regimes.

State exactly whether the result is a mechanism test, locally feasible,
officially valid native repair, or officially valid native cold start. If the
requested level was not reached, name the smallest demonstrated blocker.
