---
name: solve-roadef-irp
description: Design, implement, diagnose, benchmark, and independently validate native cold-start solvers for the ROADEF 2016 inventory-routing problem, especially Set B. Use for route chaining, ALNS or matheuristics, timing and quantity repair, call-in orders, VMI inventory, resource constraints, Solver.exe comparisons, official-checker validity, or Set B generalisation.
---

# Solve ROADEF IRP

## Where the work stands and what remains (2026-08-20)

Status: **12 of 15 Set B instances officially valid** (best-known table with
artifacts and SHA-256s: [NATIVE_BENCHMARK_RESULTS.md](../../NATIVE_BENCHMARK_RESULTS.md)).
Open: V2.17, V2.18, V2.22, V2.23, plus a single-run cold-start claim for V2.12
and V2.26. Inputs are authenticated (checker + all 20 instance XMLs
byte-match roadef.org; hashes enforced in `official_verify.py` and
`instance_manifest.py`). The stack is fully open-source: Python/Cython +
HiGHS; Gurobi is optional and no valid result depends on it.

The remaining work, with the confidence each item deserves:

- **MEASURED, exhausted:** more search budget on the open instances. Chained
  resumes plateau (V2.12: 38 -> 37 in 30 min; V2.23: 2M steps without moving
  112). Seed portfolios are worth ~3x on seed-sensitive instances (V2.12
  119 -> 38) and closed V2.26 (1-error checkpoint + 27 s resume), but do not
  touch the structural plateaus.
- **MEASURED, the binding constraints:** V2.18/V2.22 are seed-coverage-bound
  (the constructor leaves 69/134 V2.18 customers unscheduled — no search
  fixes a seed that never visits half the customers). V2.12/V2.23 residuals
  are trapped-capacity/resource-cadence (joint driver+trailer slack, not
  aggregate slack).
- **HYPOTHESIS, the designed next step (not yet built):** optimization-based
  seeding — enumerate candidate routes, select a covering set with a HiGHS
  set-covering MILP, assign resources with the interval-clique MIP instead of
  the greedy placement gate; and wire the exact quantity-LP/resource-MIP in
  as *search-time repair* so operators whose candidates fail only on
  resources or quantities stop returning nothing. Plausible by analogy to the
  measured polish gains; unproven as a constructor.
- **HYPOTHESIS, cheap multiplier (not yet built):** an algorithm-configuration
  harness (Optuna or irace) over the exposed constants — multi-seed,
  multi-instance, fixed-cap fitness, checker-gated promotion. Likely converts
  near-misses (V2.26-class) into single-run closes; cannot fix coverage-bound
  instances. Prerequisite: centralize the tunable constants into one config
  surface (they are currently scattered across CLI defaults and hard-coded
  fast-engine thresholds).
- **UNKNOWN:** whether any of the four open instances closes without new
  constructor machinery. Treat every claim otherwise as unverified until an
  artifact passes `verify-official`.

Treat official validity as the first objective and LR as the objective among valid solutions. The operational target is milestone-shaped: reach an error-free (officially valid) solution within the first 10 minutes of the run, then spend the remaining budget improving LR. When comparing runs or tuning, report time-to-first-valid and LR-at-budget as separate numbers — a run that polishes LR but never goes valid, or goes valid only at minute 29, is worse than one that is valid at minute 8 with a middling ratio.

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

## Benchmark at the competition budget: one run, 30 minutes, from the instance XML

The official protocol was a single 30-minute run per instance, and any result
claimed as competition-comparable must be produced that way: construction plus
search inside one 1800-second budget, one seed, no resumed state. Chained
resume rounds are a legitimate *exploration* tool — they establish whether an
instance is closable and where the search plateaus — but a checkpoint reached
through hours of chaining is a different claim, and the two must never share a
table. When chaining does reach a target, the result to publish is the
reproduction: whether a fresh 30-minute run gets there. Selector and engine
ablations inherit the same rule — matched single-run budgets, never resumed
states, or the comparison measures the chain, not the controller.

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

Implement the first repair-time move as an atomic compact-and-place transaction: retime a connected driver/trailer block to its earliest legal schedule and place the route in that same endpoint. Do not accept compaction alone; it has no inventory benefit and local-search acceptance will discard it. If no raw route fits a compacted bounded block, the blocker is **sequence choice**: enumerate bounded insertion positions in the resource chain, then implement a larger sequence-repair neighborhood rather than raising candidate samples or rerunning construction caps.

## Build a mutable state before adding operators

A search that evaluates a handful of moves per second cannot be fixed by better moves. Rebuild the representation first: a mutable record per shift, incremental rescoring split into independently-dirtyable groups, and a transaction API. Three rules make it correct and fast:

- **Pin equivalence against the official scorer, under mutation, on real instances.** Rescoring from scratch after every random edit and comparing every scored field is the only gate that matters. Extend it to any instance you intend to search, including open ones — a seed solution exists for scoring purposes even when the instance is unsolved.
- **Never delta-accumulate a float.** Integer counters take `new - old` safely; costs, delivered quantities, and penalty integrals drift at 1e-8 within a few dozen moves, which shows up as an inexact revert rather than a wrong score. Re-sum floats over the shift records each rescore — it is cheap next to derivation and it removes a whole class of bug.
- **Prefer copy-on-write per touched shift over in-place revert.** Journal entries keyed on the live record object (restored field-by-field into the same object) survive same-transaction insert and remove, which position-based journals do not.

Mirror the scoring pipeline's quirks rather than the rule text. Notably, the official truncation drops operation-free shifts before scoring, so an emptied shift must contribute nothing at all — no cost, and no participation in the driver or trailer chains. A shift-local rule firing on a shift the checker never sees produces an error-count off-by-one that looks like a rule bug and is not one. Build a diff tool that prints the legacy validator's per-code counts next to the new state's group totals; it localizes this in one run.

When optimising, profile before restructuring and re-profile after. Splitting a whole-chain recompute into per-resource dirty sets moved cost around and gained nothing on its own — the real win came from an index of non-empty shifts per resource, which removed an O(all-shifts) scan per move. Once the Python structure is right, a single hot inner loop in C is worth another 1.5x-2x; keep the Python version as an oracle and assert the two agree elementwise.

## Price infeasibility instead of gating on it — and bound the variable domains

A generator that filters candidates against the hard rules returns nothing from a tight incumbent, because the only reachable states are the feasible ones and there are almost none. Invert it: every state gets a finite quality, `LR + sum(w_i * violation_i)`, and the search decides what to do with a violation rather than being forbidden from seeing it. Weight one rule error above any reachable ratio gain so a violation is never free, and break the error total into groups so each kind can be weighted separately — then pin the group sum back to the official error count, statically and under mutation, so the objective cannot price an error the checker does not count.

Two rules keep the operator portfolio honest under this scheme. An operator never rejects its own output for infeasibility; it returns "no" only when the edit is **not expressible** (swapping two stops on a one-stop route). "No legal target exists" is not a reason to decline — place the edit and let it be priced.

The critical corollary: **a priced objective needs its variable domains bounded independently.** Otherwise the search finds the one quantity no rule happens to constrain and drives the ratio through it. A quantity operator multiplying by up to 1.5x with no ceiling grew one delivery to 107,593,087 kg against a largest trailer capacity of 20,000, reporting LR 0.000037 — and the official scorer agreed with every field, so it was not a scoring bug. It survived because two holes lined up: the drop landed on a call-in customer whose tank is untracked (no overfill charged), and the single trailer error it cost was swamped by an unbounded ratio gain once the adaptive weights decayed. A capacity bound on what is *expressible* closes it without gating anything. With a fractional objective this failure mode looks like a breakthrough, which is why the numerator and denominator must be reported separately at every checkpoint.

Adaptive violation weights need two guards, both of which cost more than they earn when missing:

- **Gate the global "too often infeasible" brake on having a feasible incumbent.** From an infeasible seed the visited-infeasible fraction is ~100% because nothing feasible has been found yet, so every window reads as "violations underpriced" and the scale ratchets to its ceiling. Measured cost: 33 errors versus 16 for fixed weights on the same budget.
- **Normalize the per-group scales so only ratios persist.** At a 1.05 raise factor a scale saturates a 200 ceiling in ~108 observations, so over a 100k-step run every group that ever fires ends pinned there and the mechanism carries no information at all. Ratios are all it was meant to express; overall magnitude is the global scale's job. With both fixed, adaptive beat fixed weights (9 errors versus 16) instead of losing to them.

## Better than bounding a variable: derive it, so the search cannot reach it

Bounding a gamed variable is the patch; removing it from the encoding is the fix. The 2016 winner's solution encodes only **routes** — which driver, which trailer, which sites, in what order — and recovers quantities with a decoder at evaluation time (assign each customer the least possible amount; give a source's remaining load to the previous customers, nearest first). None of its nineteen low-level heuristics touches a quantity.

That inversion killed our two worst defects at once, because both were quantity-operator defects: the unbounded multiply, and the over-delivery that read as zero errors internally. Neither is reachable when there is no quantity variable to move. A ceiling patched onto a free variable is strictly weaker than not exposing the variable — the ceiling has to be right in every case, whereas absence has to be right once. It also shrank the portfolio (29 → 25 operators), so every remaining operator is structural and the selector has less to learn.

Four things to expect when implementing a decoder from a paper's one-sentence description:

- **The gaps are where the paper is silent about magnitude, and each has its own binding rule.** "Give the remaining quantities to previous customers" does not say how much a source loads. Deciding it from downstream demand seems natural and is wrong: every route that *begins* at a source then loads nothing (0 → 358 errors on one instance). Loading is free once the trip is committed and capacity is the only bound, so a source fills the trailer. Resolve each gap to the rule that binds it, then measure — do not pick the reading that sounds most careful.
- **A bound computed "at this moment" is wrong whenever the state has a time axis.** Tank headroom at the arrival step let drops fill tanks to the brim, and a later delivery landing on top charged 57 overfills. Adding product at one stop raises the level at *every* later step, so the bound is the tightest headroom over the whole remaining horizon. Same for the mirror bound: the safety need is the deepest future shortfall.
- **A per-item decoder cannot see a cross-item bound, and the residual errors tell you which one.** When every remaining error is an *under*-delivery, the decode is consuming a shared resource: filling a customer's tank through one shift starves another shift's stop at that same customer of its minimum drop. Reserving capacity for other stops is the obvious fix and was a bad trade here (1 error of 13, for an O(all items) scan in the inner loop).
- **Decoding is stable, not idempotent, and stability is the property worth testing.** If a decoded quantity depends on other items' decoded quantities, one sweep is not a fixed point. Most instances settle in a few sweeps, monotonically improving; one entered a bounded period-2 limit cycle, two stops trading a fixed amount forever. Assert that the tail stops moving, not that sweep 2 equals sweep 1 — and pick the invariant (error count) that is stable even when the ratio wobbles.

**Measure a decode inside the search, not standalone.** A standalone sweep made one instance 0 → 7 errors and the surplus pass looked indefensible. In the search the same pass gave equal-or-better error counts on three of four instances and a materially better ratio on all four: the search repairs what the decode breaks and keeps the delivered quantity it wins. Keep the switch (`decode=`, `SPEND_SURPLUS`) so the claim stays a measurement.

## Profile before optimizing a hot path, then check what the caller actually reads

Wiring a decoder into the inner loop cost 7.5x throughput, and both fixes came from a profile rather than intuition. The first was ordinary: a Python loop over the horizon, vectorized with `cumsum`. The second was the interesting one — 10.1 s of a 14.9 s run sat in the incremental rescore, triggered by 58,321 calls to a single primitive.

The decoder wrote each stop through the state's `set_quantity` so later stops would see it, which is correct and was also 60x more work than needed: the decode reads only the **inventory projection**, and the projection's own bookkeeping function maintains it. Everything else `set_quantity` did — re-deriving the shift, re-walking resource chains, re-summing per-shift totals over *all* shifts — was discarded on the next call. Splitting the primitive into a staged form that publishes without deriving, plus one `finish_staged` per shift, tripled throughput and kept rollback exact, because the transaction snapshot is taken by `_touch` and is independent of the derivation.

The generalizable move: when a hot path calls a convenience primitive in a loop, ask what the *next iteration* actually reads. Publish that; defer the rest to the end. And keep the exactness test — a staged path that skips derivation is exactly where float drift would make rollback approximate.

## Compile the frozen kernels; keep the moving layer in Python

When the profile shows the step cost concentrated in a few flat loops whose *semantics are pinned by tests* (exact rollback, fast-vs-reference agreement), those loops are ready for Cython — and the layer still under design (operators, selectors) is not. Porting the moving layer freezes exactly what the next experiment needs to change, and every experiment becomes a recompile.

Mechanics that made it cheap and safe here:

- Port the loop **verbatim** — same full re-sum, same epsilon comparisons, same return tuple — so the existing equivalence and exactness tests transfer as-is. The two kernels this session (`sum_shift_totals`: the per-mutation O(all shifts) re-sum that was the largest line in the profile; `tank_bounds`: the decoder's tail min/max, replacing a per-call numpy cumsum allocation) each mirror their Python original, which stays as the reference oracle behind a try/except import.
- Gate on **fixed-seed trajectory identity**, not just unit equivalence: same seed, same `max_steps`, published score must match to the last digit with kernels on and off. A kernel that is "equivalent within 1e-9" but drifts the trajectory will produce uncomparable benchmark runs.
- On Windows, a loaded `.pyd` is file-locked: `build_ext --inplace` fails while any long run holds the old binary. Build to `build/`, preload the fresh binary under the package name for new work, and copy into `src/` when the run ends. Do not kill a multi-hour run to swap a binary its process will never use.

Measured: 1.56x whole-step throughput on the instance profiled at a smaller kernel share; the win scales with the kernel share. Revisit full C++ only if still compute-bound after the controller design freezes.

## Adaptive selection: normalize credit by its noise, and let no operator starve

Operators' raw gains differ by orders of magnitude (`delete_shift` swings thousands per move, `shift_one_arrival` fractions), so a roulette on mean reward ranks by *amplitude*. What predicts usefulness is signal-to-noise: track an EMA of reward and of its square, select by bias-corrected mean over RMS — Adam's m-hat/sqrt(v-hat), applied to operator credit. Reward is a **rate** (positive gain per CPU-second): the portfolio spans a 4x per-call cost range, and an operator twice as good at a hundred times the price is not a better default.

Keep a selection floor (a fraction of the top weight) so no operator's probability reaches zero. The evidence is one operator with **zero new bests over 46k calls on one instance being the operator that closed another instance** — per-instance starvation is precisely the mistake a controller must not hard-code. And expect the first head-to-head against uniform to be mixed (better where errors dominate, worse on ratio-polish); one seed decides nothing, the matched-seed ablation with median/IQR does.

## An instance can be closable and still unsolved at the competition budget — keep the two claims separate

Chained resume rounds (each round seeding from the prior published best, fresh rng per round) answer a question no single run can: *is the residual reachable by this operator set at all?* On one instance twelve chained rounds descended 1,206 -> 186 errors, improving every round with no plateau — while a single budget-compliant run from the same seed moves ~100-200. Both numbers matter and they must never share a table: the chain result says the gap is throughput x controller (not a missing move), the single-run result is the only competition-comparable claim. When a chain reaches a target, the publishable result is the *reproduction* — whether a fresh single-budget run gets there.

The same discipline catches plateaus honestly: when two long uniform polishes (millions of steps) cannot move a 6-error state that surgical single-purpose scripts improved in seconds, the residual is not under-searched — it is outside the moves' reach (there: six call-in orders competing for the same trailer-time, needing a simultaneous multi-shift reroute). Diagnose with purpose-built constructions before spending more budget on uniform search.

## When the internal score says zero and the checker says no, the rule is missing — go find its boundary

A search that reaches zero errors by its own scoring and still fails the
released checker is not a search problem, and no amount of extra budget or
better operators will touch it. The objective is missing a constraint, and the
search has already found it — a priced objective converges *onto* whatever the
model leaves free, so the disagreement is where to look, not a nuisance.

The instance of this that cost the most: both QS01 implementations checked
call-in orders only from below (under the flexible minimum an error, under
nominal a warning) and left *above* nominal unconstrained. Over-delivering a
call-in order therefore bought logistic-ratio denominator for free, and one
solution reached zero internal errors while the checker reported three missed
orders. The nominal quantity is an **inclusive ceiling**, and exceeding it makes
the checker report the order as *missed* — the same error code as delivering
nothing, which is why reading the code's lower-bound branch did not suggest it.

Three habits make this cheap to find and expensive to skip:

- **Corroborate against the valid corpus before writing a probe.** Every one of
  63 served call-in orders across 8 officially-valid solutions had zero
  over-delivery. An invariant that holds in every known-good artifact and is
  violated by every failing one localises the rule in a single query.
- **Isolate the probe from every other rule, or it will lie.** Raising one
  call-in quantity drained the trailer, so the real QS01 line arrived buried
  under 145 SHI06 errors, and the companion probe at exactly the nominal
  quantity failed on SHI06 *alone* with no QS01. Reading those runs produced a
  confidently-stated and wrong conclusion ("the rule is equality with the
  flexible minimum"). Probe on a valid solution, vary exactly one field, and
  ladder the values (below min, min, between, exactly nominal, one unit over)
  until the verdict flips on a single step: 12,000 passes, 12,001 fails.
- **Confirm the fix reproduces the checker's own list, not just its count.**
  After implementing the bound, the internal validator named the same customers
  and the same order indices the official log named. Matching totals alone would
  not have distinguished the real rule from a coincidence — and the official
  summary line ("number of missed order: computed totalMissedOrderCosts : 3")
  is easy to miscount as one more error than it reports.

## A "minimum to satisfy" bound is exclusive — never land a decoder exactly on it

`orderQuantityFlexibility` gives a call-in order a satisfaction floor of
`quantity * flexibility / 100`. Delivering *exactly* that floor does not satisfy
the order: the checker reports `checkQS01 MissedOrder`. V2.15's last remaining
error was precisely this — order[0] of customer 37 is `quantity=8000`,
`flexibility=80`, and the quantity decoder delivered 6400.0, the floor to the
decimal.

The collision is structural rather than accidental, which is what makes it worth
remembering: TS2020 sec. 3.1 tells the decoder to assign "the least possible
amount" at a customer, and the least amount that *appears* to satisfy an order is
exactly the floor. A faithful decoder therefore aims straight at the one value
the checker rejects. Any bound the search treats as attainable needs its
inclusive-versus-exclusive side established by measurement, because a priced
objective will sit exactly on it (see the QS01 ceiling section above — the same
rule, from the other direction, with the same error code).

Two practical consequences:

- Compare bounds with a tolerance that pushes *into* the feasible region, never
  `>=` against a raw float. `total >= min_quantity` read as satisfied in our
  scorer while the checker disagreed, and the internal score said zero.
- Fixing this by nudging a quantity is the wrong repair, because quantities are
  derived rather than searched. What actually closed V2.15 was giving the search
  a route where the order could be served properly: LLH0/LLH5 targeting (insert
  or replace with the customer holding the earliest unsatisfied demand or order,
  TS2020 sec. 3.2). Fix the topology and let the decoder follow.

## Two repair moves that closed V2.26, worth promoting to operators

Both came from probes after uniform surgical rounds plateaued (four rounds stuck at the same residual), and both are mechanical, instance-independent transactions (V2.26 closed 2026-08-11, LR 0.036609):

- **Merge a lone-reload shift into its predecessor.** A DRI01 "starts too early" on a single-op reload shift often cannot be fixed by delaying it: the driver's inter-shift gaps bind on *both* sides (delaying shift N re-fires DRI01 on shift N+1 — measured before theorising). When the predecessor shift has the same driver and trailer, append the reload to it and delete the offending shift: one shift has no internal rest requirement, so the gap constraint vanishes. The earliest-start-biased retimer can never find this because no retiming of the existing shift set expresses it.
- **Balanced reload+stop insertion.** To add a delivery to a shift whose trailer lacks the product, insert a reload of *exactly the delivered amount* at the route head (or any earlier position) in the same shift. The trailer's stock at shift end is then unchanged, so no downstream shift's stock, reload sizes, or SHI06 state moves — the edit is local by construction. Naive versions fail in cascades: an unbalanced "fill to capacity" reload overloads every downstream reload that was sized against the old stock.

The search context that makes these matter: the quantity decoder assigns each new stop its 1.0 minimum ("least possible"), so topology-only moves that add stops for a missed order deliver nothing — the order floor is a cross-stop sum the per-item decode cannot see. The searches got V2.26 from 6 errors to 1 by finding the right topology; the last mile was quantity-capacity, which these two transactions supply. When choosing a host shift for the stop, check in order: timing fit (driving cap 550 incl. return leg — a remote customer kills most hosts), then the driver's next-shift gap, then trailer stock — and remember trailer *capacity* spare is not *product* spare (a trailer can arrive nearly empty because an upstream source reload was partial).

## Read the reference XML when a single error will not localise

Reference solutions are diagnostic input and never solver input, and there is a
class of bug where reading one is the cheapest possible step: a single named
error on an otherwise valid solution. For V2.15's lone
`MissedOrder[0] of customer[37]`, dumping every operation at that point in both
our output and the HUST reference took one short script and gave the answer
immediately — we delivered 6400.0 at arrival 1808, the reference delivered 8000.0
in the same window. The window was right, the route was right, the quantity was
one epsilon short of counting.

Check the topology differs before trusting such a result as your own: ours was 11
shifts / 53 operations against the reference's 21 / 80, which establishes the
search built it rather than converging on a copy.

## Do not add a count to a duration and call the sum "errors"

TS2020 sec. 3.1 defines the two hard-constraint terms as "the number of customer
orders that were not met **and the time spent** by customers with an inventory
below their safety level." Those are deliberately different shapes: missed orders
are a count, safety breaches are a duration. Tallying a tank violation per time
step is therefore the paper's intended measure, and the large multiplicities it
produces are signal — a long run-out is genuinely worse than a short one — not
duplicate counting to be collapsed per customer.

What does go wrong is summing the two into one integer and then comparing that
integer across instances. V2.18 at 230 tank-steps spans 9 distinct customers of
134; V2.22 at 3,924 spans 143 of 324. Ranked by the raw sum, V2.18 looks an order
of magnitude healthier than it is relative to its own size. Report tank
violations as `(steps, distinct customers)` whenever the number is being used to
compare search quality rather than to drive acceptance.

Also worth knowing before copying an objective from the paper: TS2020 folds both
terms and the logistic ratio into a **single manually weighted** objective, with
the weight set so violations dominate. It has no lexicographic errors-then-ratio
ranking. That ordering is a local choice — a defensible one, since publication is
gated on validity — but do not attribute it to the paper.

## Rank the published artifact by validity, never by the live search objective

Adaptive violation weights decay by design, so the state with the best *current
quality* is routinely not the state with the fewest errors. Reported as an
improvement: a seed with 7 errors "improving" to 32. Keep two incumbents — the
search's own best under the live objective, and a publication best ranked
lexicographically by `(errors, LR)` — and capture the second while the candidate
state is still live, because a state can be the best ever seen by error count
without being accepted. On one corpus this single change recovered every
instance at the same budget (36 -> 8 -> **2** on the closest one) with no change
to the search itself.

Two related output rules, both learned by losing runs to them:

- **The released checker throws an unhandled .NET exception on an empty
  `<operations />` element** (return code 0xE0434352, reported as
  `execution_failed` with no verdict). Operators that empty a shift rather than
  removing it — the right call for rollback-stable positions — must not have
  that reach the file. Drop operation-free shifts in the *writer*, so nothing
  publishable can carry the defect; the official truncation already ignores them
  when scoring, so no result changes.
- **A scorer that over-counts is safe; one that under-counts loses the run.**
  After the fix one internal trailer error had no counterpart in the checker's
  output. That direction cannot manufacture a false "valid", so it does not
  block publication, but it makes the search spend effort on a phantom and is
  worth localising.

## Test whether your new engine actually constructs, or only repairs

A rebuilt search that is always seeded from the old pipeline's output has never
been shown to work. Seed it from an *empty* solution and let the operator
portfolio build from nothing. The answer is usually mixed and worth knowing
precisely: one substrate reached 4 errors from empty against 2 legacy-seeded on
a small instance (genuinely constructing), while on the largest it managed
11,611 against 4,057 (the recycled topology is load-bearing). Run this before
claiming a cold start, and name the scale at which seeding stops being a
convenience.

## Make firing rate a reproducible gate, then fix the operator, not the target

Require every operator to fire on ≥90% of invocations, measured across every instance and **several RNG seeds**. Never seed such a test from `hash(name)`: Python randomizes string hashing per process, so each run samples a different neighbourhood, and a real weakness will pass one run and fail the next with identical code. Derive seeds from `sha256` and record minima over seeds rather than one draw.

A measured rate below target has never once been a property of the instance. It is the operator declining work it could have done, and the causes recur:

- **Independent sampling that collides.** Drawing two indices independently and giving up when they match put one operator at 0.35. Draw distinct indices (`rng.sample`), and for a "swap two shifts' resources" move draw from pairs that actually differ.
- **A no-op the draw did not have to include.** Exactly one of a tail-exchange operator's `(na+1)*(nb+1)` cut pairs exchanges two empty tails. Draw over the space without it (row-major puts the no-op last) instead of drawing freely and declining.
- **Declining against a structural fact.** Restrict the draw to *eligible* elements: stops with a same-kind alternative, nonzero quantities, driver windows that leave room to reach the first stop before the score cutoff. Structural facts worth discovering per corpus — Set B drivers are qualified on exactly one trailer, and several instances define exactly one source, which makes a naive "change trailer" or "replace point" unable to fire at all.
- **Refusing at a truncation boundary instead of clipping.** The scoring pipeline drops operations past the cutoff, so a move that lengthens a route past it should **clip exactly as the scorer does**, not decline. Refusing there both lowers the firing rate and lets the incremental score disagree with the official one.

Absorb the irreducible remainder in the search loop, not the operators: reselect a bounded number of times before conceding a step. With every operator above 90%, eight draws make a genuinely empty step a ~1e-8 event, so the count that survives means the *state* is dead. Keep the bound, and test that a fully unfireable portfolio still terminates.

Retiming deserves its own tests, because it is the one thing every structural move needs. Make it best-effort and never refusing: place each stop as early as travel and windows allow, and when no window fits, place it at the physically required minute and let the lateness be charged. Then assert it does not *invent* violations — collapsing a route to earliest arrivals destroys the gaps that carried its layovers (22 fresh driving-time errors on one instance), can insert a layover on a route with no layover-designated customer, and can put a call-in arrival outside every order span. All three are noise created by the move, and all three are fixable inside the retimer.

## Treat construction policy knobs as a seed portfolio

A construction parameter that helps most instances usually breaks a few that already close. Measure the seed's error count and safety-deficit quantity-minutes for each setting across the whole corpus, then construct several seeds per run and keep the best rather than shipping one global default. Report which instances each setting helps and hurts; a mean improvement hides regressions on already-solved instances.

## HiGHS: the selector MIP is trivial, and the thread scheduler is process-global

Measured on highspy 1.15.1 (2026-08-11), which added an opt-in parallel MIP search (`parallel=on`):

- **The column-selector MIP has nothing to parallelize.** On real `column-generation-rescue` runs — including V2.22, the largest instance, with a 3000-candidate cap — every selector solve finished in under 0.05 s with **zero B&B nodes** (root-solved or proven infeasible in presolve). Parallel MIP speeds up tree search, so at current pool sizes `--selector-threads` buys nothing. The wall clock lives in candidate generation (~10 of an 11-minute iteration). The selector now logs `(seconds, cols, nodes)` per solve; revisit only if node counts become nonzero.
- **HiGHS's thread scheduler is a process-global singleton sized by the first solve.** Any later solve requesting a *different* explicit `threads` value fails with `kError` / model status "Not Set" — and in the selector that failure is silent: no selection, prefix returned. Default/auto (`threads=0`) always adapts. Therefore never set `threads` explicitly on an individual HiGHS instance in a process that runs other HiGHS solves (the LP repairs run first); enable parallelism with `parallel=on` and leave `threads` at auto.
- **HiGHS `writeModel` round-trips are not trusted for these models**: a selector model that solved Optimal live read back from `.mps` as instantly Infeasible. Benchmark solver settings in-situ via the selector's timing log, not on dumped files.

## Keep search and proof separate

Use the local checker in the inner loop. Run the official checker when a locally feasible candidate first appears, when the best valid LR improves, and before publication. Keep feasible candidates ordered by LR and diagnostics ordered by structured violation magnitude/duration.

## Demonstrate real progress

Report generated/deduplicated candidates, timing-feasible count, quantity-feasible count, complete violation-vector deltas, shifts, stops, reloads, delivered volume, valid LR, and wall time. A completed run with no improved invariant is evidence, not success.

## Promote probes into mechanics

Use customer-specific probes only to identify a missing move or invariant. Once a reusable mechanism is found, stop manually repairing IDs/timestamps, implement a feature-driven production operator with tests, and resume through the native entry point. Do not spend more than two consecutive experiments micromanaging named customers except for checker disagreement or minimal regression isolation.

Express operators using pressure area, first breach, window slack, resource boundaries, compatibility, load-path position, and incremental travel. Never encode the probe's customer ID, instance name, or fixed minute.

## Exact post-hoc polish works — but only one checker-gated step at a time

The LP/MIP polish layer (quantity maximization within headroom, interval-clique
resource assignment) produced the largest single-day LR gains in the project's
history (up to -40% on V2.13, all nine artifacts re-verified valid,
`out/deep_polished/`). The same components chained more aggressively
(`universal_polish.py` end-to-end) produced outputs the checker rejects on all
nine instances. The measured lesson: exact layers preserve validity only when
every accepted step is individually checker-gated, the way
`eliminate_redundant_shifts` and the deep-polish loop do it. An exact solve of
an approximate model is still approximate — the tank/timing models in these
layers diverge from the checker's simulation, and chaining compounds the
divergence. Do not promote a polish pipeline on its internal claims; verify
per artifact.

Also from that layer: **round derived quantities up (or keep full precision),
never down.** Writing `round(q, 4)` under an LP lower bound produced SHI16
rejections of the form `2932.1429 < 2932.1429486887` — the checker compares at
full precision.

## A claim without an artifact is not a result — and this repo has the scars

Three times now (2026-08-02 "15/15", 2026-08-19 "all Set B and Set X" +
a fabricated 11-row table, and an overwritten reference solution cited as
solver output), work in this repo has asserted validity the released checker
refutes. The discipline that catches it every time, cheaply:

- A result is the pair (exact XML path, fresh `verify-official` run). No
  artifact, no result — commit messages, tables, and READMEs are claims, not
  evidence.
- Re-verify before citing, even "known" artifacts: files get overwritten
  (the V2.12 reference was; restored from git, SHA `61a7ef87...`).
- `verify-official` prints `instance_provenance` — `MODIFIED-OFFICIAL` means
  the benchmark input itself was tampered with; stop and restore before
  anything else.
- Derived XMLs must not live in the official instance directory.
- Fabrication tells: identical LRs on different instances, uniform implausible
  solve times, results on instances no engine has ever closed, artifact paths
  that do not exist.

## Seed portfolios are a production tactic, not just a construction knob

Seeds differ enough that eight parallel 30-minute runs are the cheapest
quality multiplier available when compute rules are relaxed (measured:
V2.26 seeds 2-9 finished 1,1,2,2,2,4,4,5 errors against seed 1's 4; V2.12
best seed 38 vs 119). Pair with the resume recipe: portfolio to a 1-2 error
checkpoint, then one short `--resume-from` round (closed V2.26 in 27 s).
Label the chained result honestly — it is not a single-budget claim.

## Only one Gurobi process at a time

The local license resolves to single-use/WLS: a second concurrent process gets
"Single-use license. Another Gurobi process running" and falls back to HiGHS
(the fallback message must stay ASCII — an emoji there crashed under Windows
cp1252 and killed the run). Concurrent sweeps should not set
`ROADEF_SOLVER=gurobi`; run Gurobi solves sequentially or accept heterogeneous
solver provenance.

## Generalise and report honestly

Derive policy from continuous instance features and qualify call-in-heavy, VMI-heavy, compatibility-sparse, resource-tight, reload/layover, and scale-extreme Set B regimes with repeated seeds. Require official validity on the agreed corpus before calling the solver general.

State exactly whether the result is a mechanism test, locally feasible, officially valid native repair, or officially valid native cold start. If the requested level was not reached, name the smallest demonstrated blocker.
