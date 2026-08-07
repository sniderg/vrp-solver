---
name: solve-roadef-irp
description: Design, implement, diagnose, benchmark, and independently validate native cold-start solvers for the ROADEF 2016 inventory-routing problem, especially Set B. Use for route chaining, ALNS or matheuristics, timing and quantity repair, call-in orders, VMI inventory, resource constraints, Solver.exe comparisons, official-checker validity, or Set B generalisation.
---

# Solve ROADEF IRP

Treat official validity as the first objective and LR as the objective among valid solutions.

## Run a cold start

Two commands. The first reads only an instance XML and a seed; the second decides validity.

```bash
vrp-solver native-solve <instance.xml> <out.xml> \
  --seed 1 --time-limit 1200 --no-improvement-limit 10000 --restart-rounds 1

vrp-solver verify-official <instance.xml> <out.xml>
```

Substitute `.venv/Scripts/python.exe -m vrp_solver.cli` (add `-u` when redirecting a log, or it stays empty until exit) or `uv run vrp-solver` when `vrp-solver` is not on `PATH`. Requires `roadef_2016_data/Checker_V2.2_07032016.zip`, SHA-256 `fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a`.

Publish only on `official_valid,True`. `local_errors,0` is necessary, never sufficient. Compare `official_logistic_ratio` only among valid solutions. Track `safety_deficit_qm` to see movement when the error count plateaus.

High-leverage flags: `--seed` (vary it — outcomes differ substantially), `--candidates-per-move 120` (beats 32 decisively), `--restart-rounds 1` with a high `--no-improvement-limit` (restart rounds discard progress), and `--idle-cap` (caps mid-route idle waiting; omit to build both an uncapped and a capped seed and keep the better one). Run one process per instance in parallel rather than raising `--workers`; candidate generation dominates. Full runbook: [README.md](../../README.md#run-a-native-cold-start-solve-start-here). Per-instance status: [NATIVE_BENCHMARK_RESULTS.md](../../NATIVE_BENCHMARK_RESULTS.md).

For a whole corpus, use one command rather than a driver script — it runs the identical pipeline per instance in its own process, verifies each output with the released checker, and exits non-zero unless all are valid:

```bash
vrp-solver native-solve-batch <instance-dir> <out-dir> \
  --seed 1 --time-limit 1800 --concurrency 7 --summary-csv <out-dir>/summary.csv
```

To continue a stalled instance, pass its earlier native output to `native-solve --resume-from`; construction is skipped and the search continues from that incumbent. Only ever resume from this pipeline's own output — resuming from a reference or oracle XML makes the result `native-repair`, not a cold start.

## Keep every published result on one CLI path

When a hard instance is finally closed, the recipe that closed it must move into the shipped entry point, not stay in a scratch driver. Two failure modes recur: the entry point's operator schedule silently differs from the script's (so the published command cannot reproduce the published artifact), and continuation exists only as an external loop (so long runs are unreproducible). Fix both by making the entry point rotate operators across restart rounds internally, continuing each round from the prior incumbent, and by exposing checkpoint continuation as a flag. Then re-verify one known-good instance end-to-end through the CLI before claiming reproducibility, and re-run the corpus batch — an aligned pipeline often improves already-valid instances too, so re-hash and re-record their artifacts instead of assuming the old numbers still describe the best output.

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

## Diagnose a stalled constructor by accounting for resource time

When a cold start stalls with many safety breaches, do not theorise about coverage. Measure, per candidate and per produced route:

1. Split in-route elapsed time into travel, setup, and idle waiting. Idle exceeding travel means the constructor is buying delivery size with resource availability.
2. For every tank served after its own no-delivery breach instant, compare that instant against the *start* of the serving shift. `shift_start > breach` is starvation (no shift was even under way, so the fix is cadence or dispatch volume); `shift_start <= breach` is stretching (the route was out but arrived late, so the fix is ordering or route length).
3. Check natural-breach coverage before blaming triage. A constructor that already serves nearly every breaching tank has a timing problem, not a selection problem.

Idle waiting is the usual culprit: waiting for an economically full drop holds a driver and trailer out of the pool while other tanks breach. Cap mid-route idle waiting and price it in candidate scoring so ending a shift competes with extending it. Exempt rest waits (the driver owes that time) and the pre-first-stop wait (the planner already times each departure).

A monotone inventory projection makes an "economic fill level" step always precede the safety-breach step, so clamping the economic deferral to the breach deadline is a no-op. Verify that a proposed guard can actually bind before implementing it.

## Diagnose a stalled *search* by instrumenting candidate generation

A search that plateaus is often not out of ideas — it is generating nothing. Before tuning acceptance or adding operators, count per step: candidates generated, candidates surviving each filter, and steps accepted. Two distinct pathologies look identical from the error count:

- **Empty generation.** Operators return zero candidates and still consume a step. Log the count per operator per step; an operator that repeatedly yields zero is burning budget. Walk its filter chain and count survivors at every stage to find the single binding stage rather than guessing.
- **Unhelpful generation.** Operators return plenty of candidates but none improve. This is a scoring or targeting problem, not a reachability one.

When insertion operators yield nothing, check the chain in this order, because each step refutes cheap theories before expensive ones: is the breaching entity even targeted; does a legal host exist; does the quantity floor admit the needed top-up; can the route be retimed. Verify the retiming model can reproduce each **unmodified** existing route first — if it can, later rejections are real infeasibilities, not modelling blind spots, and adding candidate start times or relaxing bounds will not help.

Distinguish *aggregate* slack from *jointly available* slack. A driver 50% idle and a trailer 50% idle can still admit no new route, because a shift needs both simultaneously, plus the driver's inter-shift separation on each side. Enumerate joint free intervals directly; per-resource utilisation is misleading on its own. Conversely, when large in-shift idle time coexists with a small marginal need, capacity is present but trapped inside committed shifts.

Trapped in-shift idle time is not necessarily reachable from construction policy. Tightening a construction idle cap only changes which routes get built; it cannot compact routes an incumbent already committed. Measuring trapped idle therefore identifies the *quantity* of recoverable capacity, not the mechanism — sweeping construction knobs to chase it produced strictly worse incumbents on an instance whose search had already gone much further. Reclaiming it requires a repair-time move that shortens or re-times committed shifts. Distinguish "capacity exists" from "this knob can reach it" before spending a sweep.

## Treat construction policy knobs as a seed portfolio

A construction parameter that helps most instances usually breaks a few that already close. Measure the seed's error count and safety-deficit quantity-minutes for each setting across the whole corpus, then construct several seeds per run and keep the best rather than shipping one global default. Report which instances each setting helps and hurts; a mean improvement hides regressions on already-solved instances.

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
