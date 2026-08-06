---
name: solve-roadef-irp
description: Design, implement, diagnose, benchmark, and independently validate native cold-start solvers for the ROADEF 2016 inventory-routing problem, especially Set B. Use for route chaining, ALNS or matheuristics, timing and quantity repair, call-in orders, VMI inventory, resource constraints, Solver.exe comparisons, official-checker validity, or Set B generalisation.
---

# Solve ROADEF IRP

Treat official validity as the first objective and LR as the objective among valid solutions.

## Preserve provenance

Label outputs as `native-cold-start`, `native-repair`, `oracle`, or `reference`. Never describe oracle/reference topology or repair of a supplied candidate as native cold start. Keep oracle routes out of native construction and use them only for aggregate analysis.

## Inspect and validate first

1. Locate parsing, XML I/O, simulation, local validation, official-checker wrapper, solver entry points, and tests.
2. Identify the released checker's fail-closed success condition.
3. Replay the current candidate from initial state, including time zero and horizon boundaries.
4. Record instance, provenance, seed, command, runtime, checker hash, violations, official verdict, and valid LR.

Treat overfill, negative stock, trailer discontinuity, illegal timing, and missed orders as hard failures. Never clamp/delete a violation and call it repaired without full replay. Local zero errors are necessary but insufficient; publish success only when the released checker accepts the exact XML.

## Use the chain-first matheuristic

Read [references/matheuristic.md](references/matheuristic.md) before changing construction, topology, timing, quantities, acceptance, or qualification.

Use:

`select multi-shift block -> destroy -> rebuild route/resource chains -> joint timing repair -> hard full-horizon quantity repair -> replay -> accept/reject`

Do not rely on quantity-only repair or repeated single-customer emergency insertion when topology is deficient. Include driver/trailer predecessor and successor boundaries in atomic block replacement.

## Keep search and proof separate

Use the local checker in the inner loop. Run the official checker when a locally feasible candidate first appears, when the best valid LR improves, and before publication. Keep feasible candidates ordered by LR and diagnostics ordered by structured violation magnitude/duration.

## Demonstrate real progress

Report generated/deduplicated candidates, timing-feasible count, quantity-feasible count, complete violation-vector deltas, shifts, stops, reloads, delivered volume, valid LR, and wall time. A completed run with no improved invariant is evidence, not success.

## Promote probes into mechanics

Use customer-specific probes only to identify a missing move or invariant. Once a reusable mechanism is found, stop manually repairing IDs/timestamps, implement a feature-driven production operator with tests, and resume through the native entry point. Do not spend more than two consecutive experiments micromanaging named customers except for checker disagreement or minimal regression isolation.

Express operators using pressure area, first breach, window slack, resource boundaries, compatibility, load-path position, and incremental travel. Never encode the probe's customer ID, instance name, or fixed minute.

## Generalise and report honestly

Derive policy from continuous instance features and qualify call-in-heavy, VMI-heavy, compatibility-sparse, resource-tight, reload/layover, and scale-extreme Set B regimes with repeated seeds. Require official validity on the agreed corpus before calling the solver general.

State exactly whether the result is a mechanism test, locally feasible, officially valid native repair, or officially valid native cold start. If the requested level was not reached, name the smallest demonstrated blocker.
