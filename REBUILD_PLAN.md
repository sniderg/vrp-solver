# Search Substrate Rebuild Plan

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` abandoned (record why)

Created 2026-08-10 after reading `vrp-stuff/documents/TS2020.pdf` (Kheiri, the
winning SSHH method) and the ROADEF-2016 winners analysis. This plan replaces
the search substrate. It does not replace parsing, validation, scoring, the
constructor, or the CLI.

---

## 0. Why we are rebuilding

Measured on V2.15 (`out/probe_V2.15.xml`, seed 1, 300 s wall):

| Metric | Measured |
| --- | ---: |
| Search steps completed in 300 s | **9** |
| Steps that generated **zero** candidates | 3 of 9 |
| Share of wall time in candidate generation | **96%** |
| Share in quantity LP repair | 3% |
| Share in scoring | 0.2% |

Per-operator generation cost on one step:

```
create_shift                 134.55 s -> 0 candidates
multiroute_pressure_block     34.81 s -> 0 candidates
recombine_route_blocks        14.96 s -> 47
pressure_band_resource_block   8.84 s -> 3
insert_operation               1.86 s -> 0 candidates
replace_operation_point        3.16 s -> 76
```

Isolating `create_shift`: the raw route builders produce 260 routes in ~1.3 s,
then `_resource_safe_created_candidates` spends **126 s rejecting all 260**.

Kheiri reports first feasible on this same instance (his B-12..B-26 set maps to
our V2.12..V2.26; B-15 = V2.15) at **0.140 s**, and LR 0.026608 at 30 minutes.
We reach 27 errors in 300 s and never become feasible.

### The three architectural inversions (all confirmed in code)

1. **Feasibility is a hard gate, not a penalty.**
   `_resource_safe_created_candidates` returns `[]` when nothing places
   legally, so a search step is consumed with no move at all. Kheiri kept one
   penalized objective, deliberately allowed infeasible intermediate states,
   and therefore never had an empty neighbourhood. Our `_accept_move` does have
   a penalty allowance, but generation upstream already discarded everything.

2. **Expensive-before-cheap.** We run timing MIPs *inside candidate
   generation*. HUST (3rd place) ranked moves with cheap deltas first and sent
   only surviving elites to an exact solver. We inverted the cascade.

3. **No incremental evaluation.** Every candidate is a fresh immutable
   `Solution`, rescored whole. `_score` runs at 229/s on this small instance.
   `project_customer_inventory` still materialises 240 `TankEvent` dataclasses
   per customer per call, while the Cython core underneath it does 304k/s — a
   ~100x wrapper overhead. There are no prefix/suffix route labels and no
   per-customer dirty tracking.

The operator *count* (11) is not the problem; it is inside Kheiri's 15-25
range. The problem is that ours are scheduling subproblems rather than O(1)
transformations.

### Corrections to earlier project notes

`NATIVE_BENCHMARK_RESULTS.md` and the project memory state that V2.15's blocker
is the resource-placement gate finding "zero joint driver+trailer windows".
Re-measured on the current checkpoint: **19 joint free windows >= 300 min
exist**, and the gate rejects all 260 routes regardless. Raw generation now
also returns 0 *before* the gate on the post-search state. The "sequence
choice" theory was built on a state that has since moved.

More telling: our incumbent has 20 shifts and **7,008 idle minutes**; the
supplied V2.15 reference has 31 shifts and **89**. We build few long routes
padded with waiting. That also explains the gate — long padded shifts leave
fragmented resource gaps.

### What is NOT the problem (do not re-litigate)

- Operator count. 11 is enough to start.
- The Cython inventory core. It does 304k projections/s; the Python wrapper
  wastes it.
- HiGHS itself. The LP is 3% of runtime. It is called in the wrong place, not
  too slowly.
- Missing exact optimisation. Kheiri won without MIP in the core search;
  Absi's branch-cut-and-price placed 7th.

### Source-code availability (checked 2026-08-10)

Kheiri's competition solver is **not published**. His publications page links
code for 10 papers but entry [j14] (the Transportation Science IRP paper) has
only the preprint PDF. The hyper-heuristics in his public repos
(`windfarm/Java/SRIE.java`, `dynamic-insertion/SRIE.py`,
`CoSPLib/.../hyper_heuristic.py`, `magic-square`) are all *uniform random*
selection with no transition matrix. SSHH is not in the HyFlex/CHeSC
collections either.

This does not block us: **TS2020.pdf Algorithm 1 gives the complete SSHH loop
in 31 lines of pseudocode**, plus equations (2)-(4) for transition
probability, acceptance-strategy probability, and the threshold `T`. Nothing
needs reverse-engineering.

Reference code that does exist and is worth reading (read-only, different
problem): `github.com/vidalt/HGS-IRP` — real C++ `LocalSearch.cpp`,
`LotSizingSolver.cpp`. Targets the *classical* academic IRP, not ROADEF.
`HUST-Smart/ROADEF2016-IRP-Results` has **no source**, only solution XMLs and
a closed-source `Solver.exe`; their *paper* is the useful artifact.

---

## Ground rules for the whole rebuild

1. **The regression baseline is sacred.** Eight instances are officially valid
   today (see table in step 0.1). The old path stays runnable and default
   until the new one matches or beats all eight. No published artifact is
   overwritten; new outputs get new filenames.
2. **Every step has a numeric gate.** If the gate is not met, do not proceed to
   the next step — go to that step's contingency.
3. **Measure, do not theorise.** This codebase has a documented history of
   plausible theories refuted by measurement (constructor coverage,
   dispatch-before-breach, layover blocking, candidate start-time sparsity).
   Add a diagnostic before adding a mechanism.
4. **Official validity is the only success claim.** `local_errors,0` is
   necessary, never sufficient. Publish only on `official_valid,True`.
5. **Record refutations in this file.** A dead end that is written down is
   worth as much as a win.

### 0.1 Regression baseline (as of 2026-08-10)

Officially valid native cold starts, artifacts confirmed present in `scratch/`:

| Instance | Artifact | LR |
| --- | --- | ---: |
| V2.13 | `scratch/replicate_V2.13_native.xml` | — |
| V2.14 | `scratch/cold_V2.14_cadence.xml` | 0.084934 |
| V2.16.2 | `scratch/cold_V2.16.2_batch.xml` | 0.042634 |
| V2.19 | `scratch/opt_V2.19_native.xml` | 0.096702 |
| V2.20.2 | `scratch/opt_V2.20.2_native.xml` | 0.032622 |
| V2.21.2 | `scratch/opt_V2.21.2_native.xml` | 0.032982 |
| V2.24 | `scratch/replicate_V2.24_native.xml` | — |
| V2.25 | `scratch/opt3_V2.25_native.xml` | 0.035982 |

Open: V2.12, V2.15, V2.17, V2.18, V2.22, V2.23, V2.26 (7 of 15).

Best known open-instance error counts to beat: V2.15 36, V2.26 74,
V2.12 1306, V2.18 1626, V2.23 2523, V2.17 5584, V2.22 6176.

---

## Step 1 — Mutable state + delta evaluator

**Goal:** a solution representation that can be mutated and re-scored
incrementally, instead of rebuilt and rescored whole.

New module: `src/vrp_solver/fast/` (new package, nothing existing modified).

- [x] 1.1 `state.py`: mutable `SearchState` — flat arrays for routes
      (`point`, `arrival`, `departure`, `quantity`), per-shift
      `(driver, trailer, start, end)`, and a per-customer delivery index.
- [x] 1.2 Prefix/suffix route labels: cumulative travel, cumulative time,
      earliest-feasible-arrival forward pass and latest-feasible-departure
      backward pass, so a relocate/swap is checkable from O(1) cached labels.
      *Implemented as the per-shift `cum_sorted` cache plus the `_by_trailer` /
      `_by_driver` membership index; the chain walks read those instead of
      rescanning every shift.*
- [x] 1.3 Per-customer dirty tracking: a delivery change to customer `c`
      reprojects only `c`.
      *Note: `_recompute_customer_tank` reprojects `c` over the whole horizon,
      not from the first changed step. The horizon is <= 240 steps and the loop
      is now C, so the from-step-`k` refinement bought nothing and was dropped.*
- [x] 1.4 `StateScore`: accumulator over the five dirtyable groups.
      *Integer counters take deltas; every float total (`cost`, `delivered`,
      `loaded`, `safety_kg_min`) is fully re-summed each rescore. Delta
      accumulation on floats drifted at 1e-8 and broke exact revert.*
- [x] 1.5 `begin` / `commit` / `rollback` with copy-on-write per touched shift
      (contingency 1C, adopted up front). Revert is exact — the crux.
      *Journal entries key on the live `ShiftRec` object and restore through
      `ShiftRec.restore_from`, so same-transaction insert/remove is safe.*
- [x] 1.6 Equivalence test: for randomised move sequences on every Set B
      instance with an available seed, incremental score ==
      `contest.score_prefix_with_feasibility_tail` recomputed from scratch,
      to 1e-6. `tests/test_fast_state.py`, 10 instances including V2.15 and
      V2.22. `tools/diff_fast_rules.py` localizes any divergence to a rule code.
- [x] 1.7 Round-trip test: `SearchState` -> `Solution` -> `SearchState`
      preserves every field; and apply-then-revert restores identical state
      (60 randomized transactions per instance, structure and score both).
- [x] 1.8 Microbenchmark `tools/bench_moves.py`: moves evaluated/sec on all
      10 seeded instances, next to the legacy full-rescore rate.

**Gate:** >= 10,000 move evaluations/sec on V2.15 and >= 2,000/sec on V2.22,
with step 1.6 equivalence passing. Current effective rate is 0.03 steps/sec.

**Gate met.** Measured on the apply / score / revert cycle (reverse-block move,
4 s per instance, `--seed 1`):

| instance | customers | shifts | pure Python /s | with C tank loop /s | legacy /s | speedup |
|---|---|---|---|---|---|---|
| V2.13   |  53 |  20 |  6672 | 13728 | 527 | 26x |
| V2.14   |  53 |  73 |  4000 |  6832 | 141 | 48x |
| **V2.15** | 134 |  23 |  5920 | **11488** | 229 | 50x |
| V2.16.2 | 184 |  34 |  4640 |  8880 | 169 | 53x |
| V2.19   |  53 |  73 |  4032 |  7120 | 145 | 49x |
| V2.20.2 | 184 | 156 |  2512 |  3936 |  44 | 89x |
| V2.21.2 | 184 | 161 |  2304 |  3648 |  43 | 85x |
| **V2.22** | 324 | 159 |  2416 |  **3728** |  39 | 96x |
| V2.24   |  32 |  28 |  9328 | 12432 | 607 | 20x |
| V2.25   |  32 |  77 |  5408 |  7264 | 209 | 35x |

V2.15 11488 >= 10000 and V2.22 3728 >= 2000. Slowest overall is V2.21.2 at
3648/s, ~26x above the 2000/s floor the gate sets for the largest instance.
Speedup column is the C rate over the legacy rate.

**Contingency 1A — throughput gate missed in pure Python.** Expected;
Python may plateau near 2-5k/s. Move the inner accumulator into
`inventory_fast.pyx` (the extension already builds). Keep the Python version
as the reference oracle for 1.6.

- [x] 1A **taken.** Pure Python plateaued exactly as predicted (5920/s on
      V2.15 against a 10000/s gate; slowest 2304/s). Profiling put 38% of
      tottime in `_recompute_customer_tank`, which was making five separate
      numpy passes over the horizon (`cumsum`, three `count_nonzero`, a masked
      sum). Collapsed into one C loop as
      `inventory_fast.score_customer_row`; `state.py` imports it with a
      pure-Python fallback on `ImportError`. Net 1.5x-2.1x, and the gate clears.
      The Python version is retained as an oracle and asserted against the C
      loop per customer row in
      `test_cython_tank_row_matches_python_oracle`.
      *Build note: `Cython` was absent from the venv (the `.pyd` was stale
      prebuilt). It is already declared in `pyproject.toml` `build-system`;
      installed with `uv pip install Cython`, then
      `.venv/Scripts/python.exe setup.py build_ext --inplace`.*

**Contingency 1B — still short after Cython.** Reduce evaluation *scope*
rather than cost: score only the affected customers and the affected shifts'
cost contribution, and defer the full-horizon check to acceptance time. This
is HUST's cheap-then-exact cascade and is legitimate, but it must be gated by
1.6 run at acceptance points.

**Contingency 1C — revert correctness proves fragile.** Fall back to
copy-on-write per shift (copy only mutated shifts, share the rest). Costs
maybe 2x versus true in-place, still ~1000x better than today. Prefer this
over shipping a subtly wrong revert.

**Contingency 1D — equivalence test cannot be made to pass.** Then the
existing `contest` scorer and the new one disagree about the *problem*, not
just performance. Stop and find out which is right by running the released
checker on a handful of deliberately-perturbed solutions. Do not proceed on an
unexplained mismatch.

---

## Step 2 — Single penalized objective

**Goal:** no move is ever discarded as "infeasible". Every state is scored;
violations cost weight.

- [x] 2.1 `objective.py`: `Objective.quality = LR + sum(w_i * violation_i)`,
      where LR is `cost / max(1, delivered)`. Weights start from the magnitudes
      already encoded in `surgical_search._scalar`.
      *`StateScore` now breaks `feasibility_errors` into the seven groups in
      `objective.GROUPS`, so each violation kind can be weighted separately.
      `test_fast_objective` pins the decomposition's sum back to the
      official-equivalent total, both statically and under mutation — the
      objective is only allowed to price errors the verified scorer counts.*
- [x] 2.2 Adaptive weights: `AdaptiveWeights` raises `w_i` while violation `i`
      persists and decays it while absent. Target 15-35% of visited states
      infeasible (the analysis doc's suggested starting range — to be measured,
      not trusted).
      *The per-group scale and the global infeasible-fraction brake are kept on
      **separate** multipliers. Folding both into one number made the brake
      inert: per-group scales move every observation and travel by
      `factor ** window` across a window, so a once-per-window global nudge
      of 1.05 was three orders of magnitude too small to see. Pinned by
      `test_global_brake_is_not_swamped_by_per_group_drift`.*
- [x] 2.3 Kheiri's acceptance rule (TS2020 eq. 4), as `objective.accept` /
      `objective.acceptance_threshold`: accept if
      `quality(new) <= quality(current)`, or if
      `quality(new) < quality(best) + T*abs(quality(best))` where
      `T = 0.001` while best is infeasible, else
      `T = 0.0001 + 0.01*(1 - t_elapsed/t_limit)`.
      *`abs(quality(best))` rather than `quality(best)`, so the band cannot
      invert if a future weighting admits a negative quality.*
- [x] 2.4 Instrument: `SearchTelemetry` — fraction of steps accepted, fraction
      of visited states infeasible, empty-neighbourhood count, and the LR
      numerator and denominator reported separately (the contingency-2B check).

**Gate:** on V2.15, zero steps with an empty neighbourhood over a 60 s run,
and >= 20% of steps accepted.

**Gate deferred to step 3, by construction.** Both halves of it are properties
of the *neighbourhood*, which does not exist until the operator portfolio is
built: with no operators there are no steps to accept and every step has an
empty neighbourhood. Steps 2.1-2.4 are unit-tested in
`tests/test_fast_objective.py` and this gate is measured at the end of step 3
instead. Recorded here rather than quietly re-scoped.

**Gate met**, measured with `tools/bench_search.py` on V2.15, 60 s, uniform
selection (step 4.1), from the `V2.15_compact_full_master` seed (36 errors,
LR 0.0713):

| seed | steps | steps/s | accept | empty | errors | LR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 434,114 | 7,235 | 23.2% | **0** | 36 -> 10 | 0.0541 |
| 2 | 430,582 | 7,176 | 21.9% | **0** | 36 -> **1** | 0.0438 |
| 3 | 437,585 | 7,293 | 21.4% | **0** | 36 -> 12 | 0.0541 |
| 4 | 489,263 | 8,154 | 19.5% | **0** | 36 -> 8 | 0.0502 |

Empty neighbourhoods are zero and accept sits at 19.5-23.2%, inside the
intended band. Compare step 0 on this instance: **9** steps in 300 s, three of
them generating nothing.

Three things had to change for this, all found by measuring rather than
reasoning:

- [x] 2.6 **The empty-neighbourhood half belongs to the loop, not the
      operators.** `two_opt_star` (6.7% of invocations) and `create_shift`
      (4.1%) decline for real structural reasons and pass the step 3 gate doing
      so. The loop now reselects up to `_EMPTY_RETRIES = 8` times before
      conceding a step; with every operator above 90% that makes a genuinely
      empty step a ~1e-8 event, so the count that remains means the *state* is
      dead. The bound is what keeps a dead state from spinning, and
      `test_a_dead_state_terminates_instead_of_spinning` pins that.
- [x] 2.7 **Two defects in `AdaptiveWeights`, both of which made it worse than
      fixed weights.** Measured on V2.15 over 10 s: adaptive ended on 33 errors
      where a fixed `Objective()` ended on 16.
      1. *The global brake ratcheted to its ceiling and stayed.* From an
         infeasible seed the visited-infeasible fraction is ~100% because no
         feasible state exists yet, so every window read as "violations
         underpriced" and the scale multiplied by 1.5 each time until it hit
         200. Quality reached 2.5e5. The brake is now gated on
         `best_feasible`: the 15-35% band only means something once there is a
         feasible incumbent to return to, and while the best is infeasible the
         search is already spending everything on repair.
      2. *Per-group scales all saturated, so they carried no information.* At
         `raise_factor = 1.05` a scale reaches `max_scale = 200` in ~108
         observations; over a 100k-step run every group that ever fired ended
         pinned at 200. `_normalize` now holds the largest scale at 1 so only
         *ratios* persist, which is all the per-group mechanism was ever
         supposed to express — overall magnitude is the global scale's job.

      With both fixed, adaptive beats fixed on the same seed and budget
      (9 errors versus 16), so step 2.2 earns its place instead of being
      carried on faith.

**Contingency 2B fired, and the instrumentation is what caught it.** Seed 4 of
the first 60 s gate run reported LR **0.000037** — a thousand-fold below
anything on this instance, and it would have read as a breakthrough if the plan
had not insisted on reporting the numerator and denominator separately. The
denominator was 107.6M kg delivered against 24.7k loaded. The official scorer
*agreed* with every field, so this was never a scoring bug:

- **Cause.** `increase_quantity` multiplied a drop by up to 1.5x with no
  ceiling, so repeated application grew it geometrically. One operation reached
  107,593,087 kg against a largest trailer capacity of 20,000.
- **Why the objective did not stop it.** Two independent holes lined up. The
  absurd drop landed on a *call-in* customer, whose tank is not tracked, so no
  overfill was charged; and the single trailer-load error it did cost was
  outweighed by an unbounded LR gain once the adaptive weights decayed.
  Penalizing infeasibility only works if every way of cheating costs something.
- [x] 2.8 **Fix: bound the quantity domain.** A trailer cannot move more than
      its capacity in one drop, and a drop cannot exceed the tank, so
      `_quantity_ceiling` clamps `increase_quantity` and `fill_quantity`. This
      is a bound on what is *expressible*, which is the one thing an operator is
      allowed to refuse — it is not a feasibility gate, and no penalized state
      is rejected by it. Pinned by
      `test_a_quantity_operator_cannot_exceed_trailer_capacity` and
      `test_the_logistic_ratio_denominator_stays_physical`.

The general lesson, which belongs in the skill file: an objective that prices
infeasibility needs its *variable domains* bounded independently. Otherwise the
search will find the one quantity no rule happens to constrain and drive the
ratio through it, and a fractional objective makes that failure look like
success.

**Contingency 2A — search wanders and never returns to feasible.** Weights
too low. Add a feasibility-restoring intensification phase: when best has been
infeasible for N steps, multiply violation weights until feasible, then relax.
- [ ] 2.5 (if needed) implement the ratchet.

**Contingency 2B — the ratio objective is gamed** (delivered quantity inflated
to shrink LR without real benefit). Report numerator and denominator
separately at every checkpoint, as the analysis doc insists. If it happens,
constrain the denominator to only count deliveries the checker will accept.

---

## Step 3 — O(1) operator portfolio

**Goal:** 15-25 genuinely different cheap transformations over `SearchState`.
Derived from TS2020 §3.2 LLH0-LLH18 (one-line descriptions; we design our own
implementations — no source exists to port).

All 29 live in `src/vrp_solver/fast/operators.py` under one contract,
`operator(state, rng, lists) -> bool`, always called inside a `begin()`
transaction. Two rules are documented at the top of that module and are the
whole point of the step: an operator never rejects its own output for
infeasibility, and it returns `False` only when the edit is not *expressible*.

- [x] 3.1 Route-internal: relocate-1, relocate-block, swap-1-1, swap-block,
      2-opt (reverse block), move layover.
- [x] 3.2 Inter-route: relocate between shifts, swap between shifts,
      2-opt*, cross-exchange, merge shifts, split shift.
      *`merge_shifts` and `delete_shift` **empty** a shift instead of removing
      it, so the shift list stays index-stable mid-transaction and the emptied
      shift remains a host for a later insertion.*
- [x] 3.3 Structural: insert customer, insert source, delete operation,
      replace point, create shift, delete shift.
- [x] 3.4 Quantity/inventory: increase quantity, decrease quantity,
      retime shift, shift a single arrival.
      *Plus `fill_quantity`, `minimal_quantity` (contingency 3B's greedy rule
      as a move) and `compact_shift`.*
- [x] 3.5 Resource: change trailer, change driver, swap trailers of two
      shifts, swap drivers of two shifts.
- [x] 3.6 Candidate lists: per-customer nearest-K compatible neighbours,
      K swept at 5/10/20/40% (HUST used ~10%).
      *`retime.CandidateLists`, `k = max(5, min(n-1, round(n*fraction)))` with
      `fraction=0.10` as the default; the floor of 5 keeps small instances from
      degenerating to one candidate. The sweep itself is deferred to step 4,
      where there is a search to measure it with — K only matters through its
      effect on search progress, and 0.10 is HUST's own answer.*
- [x] 3.7 Each operator: a unit test that it produces a *valid mutation* and
      that revert restores state.
      *`tests/test_fast_operators.py`, 161 cases. Three obligations per
      operator, all on real instances: `test_operator_rollback_is_exact`
      (score and structure identical after 40 apply/rollback cycles),
      `test_operator_output_agrees_with_the_official_scorer` (every scored
      field still equals `score_prefix_with_feasibility_tail` after each of 25
      applications, so an operator cannot build a state the search would
      mis-price), and `test_operator_survives_an_empty_state`.*

**Gate:** every operator returns >= 1 candidate on >= 90% of invocations
across all 15 Set B instances. This is the direct fix for today's
zero-candidate steps.

**Gate met, with no exemptions.** Measured over 29 operators x 10 seeded
instances x 8 RNG seeds x 200 invocations. The worst rate anywhere is
`retime_shift` at **0.915** (V2.24), then `create_shift` at 0.935 and
`shift_one_arrival` at 0.975; the other **26 of 29 fire on 100% of
invocations on every instance and seed**. Compare step 0: three of nine legacy
steps generated *zero* candidates, and `create_shift` spent 134 s to return
nothing.

`create_shift`'s exemption has been removed — it was only ever needed because
the operator drew driver windows freely and then discovered the shift would be
invisible to the scorer. It now draws only from starts that leave room for the
drive out, so it clears the target like everything else and the gate is
strictly stronger than as written.

- [x] 3.9 **The gate itself was not reproducible.** The per-operator RNG seeds
      came from `hash(op_name)`, and Python randomizes string hashing per
      process, so every run sampled a different neighbourhood. A real
      `two_opt_star` weakness passed on one run and failed on the next with
      identical code. Seeds are now `sha256`-derived, and the recorded rates
      above are minima over 8 seeds instead of one arbitrary draw. A gate that
      is not reproducible is not a gate.

Getting there took four rounds of diagnosis, all of the same shape — a
measured firing rate below target was never a property of the instance, it was
the operator declining work it could have done:

1. *Self-inflicted rule noise.* Retiming a route to its earliest arrivals
   destroyed the gaps that carried its layovers (22 fresh DRI03 on V2.13),
   invented a layover on a route with no layover customer (LAY02 on V2.24), and
   put a call-in arrival outside every order span (QS03). All three are fixed
   inside `earliest_arrivals`, which now carries the driving cap including the
   return leg, gates layovers on `layover_allowed`, and intersects customer
   windows with order spans. Residual DYN01/QS01 movement is genuine: retiming
   moves deliveries between time steps.
2. *Declines from independent sampling.* Six operators drew two indices
   independently and gave up when they collided — `relocate_block_within` was
   at 0.35 for this reason alone. Fixed with `rng.sample(range(n), 2)`,
   block sizes capped at `n-1`, and `_pick_differing_pair` for the resource
   swaps.
3. *Declines against a structural fact.* `replace_point` sat at 0.73-0.79
   because several Set B instances define exactly **one source**, so a source
   stop (a quarter of all stops on V2.13) has no same-kind alternative;
   `decrease_quantity` sat at 0.855 because 8 of 80 quantities are zero and a
   zero cannot be scaled down. Both now draw only from eligible stops.
4. *Declines on a no-op the draw did not have to include.* Exactly one of
   `two_opt_star`'s `(na+1)*(nb+1)` cut pairs exchanges two empty tails; it now
   draws over the space *without* that pair (row-major, so the no-op is last and
   a draw below the total cannot hit it) and went from 6.7% empty invocations to
   zero. Together with `create_shift`'s window fix, these were the last two
   sources of empty steps in the search loop, so fixing them closed the step 2
   gate at the operator level as well as at the loop level.

**Contingency 3A — an operator is structurally unable to fire** (e.g. create
shift with no legal resource slot). It must still return a *penalized*
candidate rather than nothing: place the shift and let the objective charge for
the resource violation, with repair left to later moves. This is the core
lesson from step 0's inversion #1.
- [x] 3A **taken**, twice, both times adopted up front rather than as a rescue.
      `create_shift` honours only TL03 and the driver window, places one
      customer, and lets every resource-timing violation be priced.
      `change_trailer` had to follow: Set B drivers are qualified on **exactly
      one trailer** (one per driver on V2.13/V2.24/V2.20.2; one for 12 of 13 on
      V2.22), so a TL03-respecting `change_trailer` cannot fire *at all* — 0 in
      40 attempts — which would also freeze `swap_trailers`. It now falls back
      to any other trailer and lets TL03 be charged.
- [x] 3A, third instance: **clip at the cutoff, do not refuse.**
      `contest.truncate_solution` drops operations arriving at or after the
      score cutoff, so a move that lengthens a route past it made the
      incremental score disagree with the official one (`create_shift`
      fast=206 ref=203). `_clip_to_cutoff` now truncates the route tail exactly
      as the scorer does, applied in `_apply_route`, `_apply_timing` and both
      `insert_shift` sites. `retime_shift` and `shift_one_arrival` were
      switched from declining on the cutoff to clipping, which also lifted
      their firing rates.

**Contingency 3B — quantity assignment needs the LP too often.** Implement
Kheiri's greedy rule (TS2020 §3.1): at a customer assign the *least* feasible
amount; at a source give all remaining to the previous customers, nearest
first. O(n) and no solver. Demote HiGHS to periodic incumbent polish only.
- [ ] 3.8 (likely needed regardless) implement the greedy quantity rule.

**Contingency 3C — throughput collapses as operators are added.** Profile per
operator and enforce a per-operator time budget; disable any operator whose
improvement-per-second falls below the portfolio median (the analysis doc's
orchestration policy).

---

## Step 4 — Adaptive selection, staged

**Goal:** establish that each increment of controller complexity earns its
place. Do **not** jump to the HMM.

The loop lives in `src/vrp_solver/fast/search.py`. It is deliberately thin:
what to try lives behind a `Selector`, whether to keep it lives in
`objective.py`. 4.2-4.4 are then swaps at one seam, which is the only way the
ablation in 4.5 can be honest. `Selector.select` returns a *sequence* of
operator ids so SSHH fits the same interface as uniform.

- [x] 4.1 Uniform random selection. Record as the baseline.

**Baseline**, `tools/bench_search.py --limit 30`, seed 1, uniform selection,
from each instance's `bench_moves` seed solution. Errors and LR as
`seed -> final`:

| instance | steps | steps/s | accept | empty | errors | LR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2.13 | 280,044 | 9,335 | 14.3% | 0 | 0 -> 0 | 0.0715 -> **0.0468** |
| V2.14 | 151,819 | 5,061 | 12.3% | 0 | 0 -> 0 | 0.0849 -> **0.0765** |
| V2.15 | 229,707 | 7,657 | 22.9% | 0 | 36 -> **10** | 0.0713 -> 0.0562 |
| V2.16.2 | 218,640 | 7,288 | 15.4% | 0 | 0 -> 0 | 0.0426 -> **0.0281** |
| V2.19 | 151,688 | 5,056 | 10.4% | 0 | 0 -> 0 | 0.0967 -> **0.0878** |
| V2.20.2 | 84,885 | 2,830 | 6.3% | 0 | 0 -> 0 | 0.0326 -> **0.0321** |
| V2.21.2 | 86,455 | 2,882 | 7.8% | 0 | 0 -> 0 | 0.0330 -> **0.0326** |
| V2.22 | 78,996 | 2,633 | 13.3% | 0 | 6176 -> **4959** | 0.0221 |
| V2.24 | 253,367 | 8,446 | 11.4% | 0 | 0 -> 0 | 0.0270 -> **0.0171** |
| V2.25 | 165,393 | 5,513 | 13.0% | 0 | 0 -> 0 | 0.0360 -> **0.0239** |

Eight of the ten seeds were already feasible and stayed feasible while LR fell
20-39% in 30 s. The two infeasible seeds both improved without becoming
feasible. Read this as a substrate result, not a solution result: these are
30 s runs from existing seeds with no controller, and none is checker-verified
yet — step 5 does that. What it establishes is that the search now *moves*,
against 9 steps in 300 s at step 0.

Two honest caveats to carry into 4.5:

- **Accept fraction falls below the step 2 band on the large instances**
  (V2.20.2 6.2%, V2.21.2 7.0%). The band was calibrated on V2.15. On a bigger
  instance one move perturbs a smaller share of the solution, so the same
  relative threshold admits less; whether that needs a per-instance `T` is a
  4.5 question, and it is a plausible reason those two barely moved.
- **V2.22 is the outlier at both ends**: the worst seed (6,176 errors) and the
  lowest throughput (2,646 steps/s). Its 304 `tank_negative_steps` are the
  deepest failure the objective prices, and 30 s of uniform selection cleared
  only a fifth of its errors.
- [ ] 4.2 Reward-per-CPU-second roulette (an operator that improves twice as
      much but costs 100x is not a better default).
- [ ] 4.3 Bandit controller (UCB or Thompson).
- [ ] 4.4 SSHH: transition matrix `T[h_prev][h_cur]` + acceptance-strategy
      matrix `AS[h][0/1]` + solution-parameter matrix `S[h][0/1]`, all
      initialised to 1, incremented on new-best, roulette-selected per
      TS2020 eqs. (2)-(3), following Algorithm 1 exactly.
- [ ] 4.5 Ablation table across 4.1-4.4: >= 20 matched seeds per instance,
      report median/IQR/best/worst and time-to-target, never best-of-N alone.

**Gate:** 4.4 beats 4.1 on median final objective across the corpus. If it
does not, keep the simpler controller and say so.

**Contingency 4A — SSHH shows no gain over 4.2/4.3.** A real, publishable
finding for this codebase: our operators may already be too coarse-grained for
sequence synergy to matter. Keep the best simpler controller, record the
refutation here, and spend the time on step 3 breadth instead.

**Contingency 4B — sequence learning needs more samples than the budget
allows.** Sequence scores need thousands of new-best events. If step 1's
throughput came in at the low end, run 4.4 only on the smaller instances and
report the scale limit honestly.

---

## Interlude — the QS01 nominal ceiling (a missing rule, not a missing move)

Prompted by "are you missing something that prevents us from solving B
instances?". The answer was yes, and it was not search power.

The engine drove V2.15 to **zero errors by its own scoring**, agreed by
`contest` and `rules.py`, while the released checker rejected it with three
`checkQS01 MissedOrder` lines on customers 3 and 37. Both internal QS01
implementations (`rules.py::_validate_service_quality`,
`fast/state.py::_recompute_customer_callin`) checked call-in orders only from
**below**: under the flexible minimum was an error, under nominal a warning,
and *above* nominal was unconstrained. So over-delivering a call-in order
bought logistic-ratio denominator for free, and the search found it — customer
37 got 11,070 kg at each of two windows against a nominal 8,000.

Corroboration before touching code: across all 8 officially-valid solutions,
63 served call-in orders have `over_nominal = 0` in every single one.

**The boundary, measured rather than reasoned.** Two earlier probes were
confounded — the extra load drained the trailer, so QS01 was buried under 145
SHI06 errors, and a companion probe at exactly nominal failed on SHI06 alone
with no QS01 at all. The claim "it must equal `min_quantity`" came from that
confounded run and was wrong. Re-probed on the officially-valid V2.24, order 0
of customer 6 (flexible minimum 9,600, nominal 12,000), varying that one
operation's quantity and verifying each file:

| delivered | official QS01 MissedOrder |
| --------- | ------------------------- |
| 9,600 (min) | 0 |
| 10,800 | 0 |
| 12,000 (nominal) | 0 |
| 12,001 | **1** |

The nominal quantity is an **inclusive ceiling**, and exceeding it makes the
checker report the order as *missed*. Implemented in both scorers as
`delivered > order.quantity + EPSILON -> error`.

Verification that the blind spot is closed: V2.15's previously "0-error"
solution now scores **3 internal errors**, and `validate_solution` names
exactly the orders the official log named — `order 0` of customer 3, `order 0`
and `order 1` of customer 37. (The official summary line "totalMissedOrderCosts
: 3" was earlier miscounted as 4 errors; it is 3.)

Two defects found alongside it:

- **The checker crashes on an operation-free shift.** `<operations />` produces
  `official_status,execution_failed`, return code 0xE0434352 (a .NET unhandled
  exception) — no verdict at all. Operators empty shifts rather than removing
  them so positions survive a rollback, and the official truncation drops
  operation-free shifts before scoring, so nothing internal ever noticed. Fixed
  in `save_solution` (the writer, so nothing publishable can carry it) plus
  `to_solution(drop_empty=True)` for callers that want it earlier.
- **"Best" was best-by-adaptive-quality, which is not publication order.** The
  adaptive weights decay, so a 32-error state out-priced a 7-error one and was
  reported as an improvement — V2.26 "improved" from a 7-error seed to 32.
  `SearchResult` now also carries `published_solution/published_score`, ranked
  lexicographically by `(errors, LR)` and captured while the candidate state is
  still live (a state can be the best ever seen by error count without being
  accepted). Recovered on every open instance at 60 s, seed 1:

| instance | seed | adaptive-best | publication-best |
| -------- | ---- | ------------- | ---------------- |
| V2.12 | 1306 | 135 | **130** |
| V2.15 | 36 | 8 | **2** |
| V2.17 | 5584 | 1298 | **1206** |
| V2.18 | 1626 | 342 | **316** |
| V2.22 | 6176 | 4248 | **4057** |
| V2.23 | 2523 | 1412 | **1320** |
| V2.26 | 7 | 32 (regressed) | **7** |

V2.15 at 2 internal errors verifies through the released checker at **one**
official error (a single missed order on customer 37) — the closest any
artifact has come. The internal scorer counts one extra `trailer_errors` that
the checker does not flag: a disagreement in the *safe* direction (it cannot
manufacture a false "valid"), but it makes the search chase a phantom, so it is
worth localising in step 5.

**Is the engine only a repair pass over legacy topology?** Tested directly by
seeding every open instance from an empty solution (no shifts) and letting the
portfolio construct, 60 s, seed 1:

| instance | from empty | legacy-seeded |
| -------- | ---------- | ------------- |
| V2.15 | 4 | 2 |
| V2.12 | 1269 | 130 |
| V2.17 | 1614 | 1206 |
| V2.18 | 1475 | 316 |
| V2.26 | 582 | 7 |
| V2.22 | 11611 | 4057 |
| V2.23 | 10464 | 1320 |

So the substrate genuinely constructs — V2.15 reaches 4 errors from nothing at
all — but legacy seeding still wins decisively at scale. On the large instances
the recycled topology is load-bearing, and that is a real limitation to name in
step 5 rather than paper over.

**Multi-seed confirmation, 120 s, seeds 1-6.** V2.15 lands on exactly **2**
internal errors in 5 of 6 seeds (LR 0.0526-0.0699) and 4 in the sixth; V2.26
lands on exactly **7** in all 6 (LR 0.0420-0.0462). Both are hard floors, not
seed noise, so the remaining gap is structural and not a budget or luck
question.

Released-checker verdicts on the best of those:

| artifact | internal | official | official errors |
| -------- | -------- | -------- | --------------- |
| `scratch/push_V2.15.xml` | 2 | invalid | **1** `checkQS01 MissedOrder`, customer 37 |
| `scratch/push_V2.26.xml` | 7 | invalid | 8 `checkQS01 MissedOrder` |

V2.15 is one missed order from valid. Both remaining failures are now *entirely*
QS01 — every other rule class is clean — which is a much narrower target than
the mixed violation vectors this rebuild started from, and it says the next move
is a call-in-order-aware one (serve the order at all, within its window, without
exceeding nominal) rather than more search.

Regression tests: `tests/test_callin_nominal_ceiling.py` (14 passed), pinning
the inclusive ceiling, the surviving lower bound, fast/reference agreement
under doubled call-in quantities, and the empty-shift output defect; plus four
in `tests/test_fast_search.py` pinning the published incumbent (never worse
than its seed, never worse than the adaptive best on errors, scores what the
run reported via the official path, carries no empty shift). Full suite **446
passed / 11 failed** — exactly the pre-existing unrelated set.

---

## Interlude 2 — the quantity decoder (removing the variable, not bounding it)

Decision (user, 2026-08-10): *"go with the decoder. trust the winners
description over our own attempts unless we are filling in gaps."*

TS2020 §3.1 encodes a solution as **routes only**. Quantities are not decision
variables; evaluation runs two decoders — timing (earliest feasible, which
`retime` already did) then quantity:

> if the current site is a customer, then we assign the least possible amount of
> product to deliver; otherwise if the site is a source, then all the remaining
> quantities will be given to the previous customer sites, starting from the
> nearest ones.

None of LLH0–LLH18 touches a quantity. That is the point: our four quantity
operators caused the two worst defects of this rebuild (the unbounded multiply
that grew one drop to 107,593,087 kg, and the QS01 over-delivery that read as
zero errors internally while the checker rejected the file). Both are
*structurally unreachable* with no quantity variable to game. A ceiling patched
onto a free variable is strictly weaker than not exposing the variable.

- [x] `src/vrp_solver/fast/decode.py` — the two-pass decoder.
- [x] Removed `increase_quantity`, `decrease_quantity`, `fill_quantity`,
      `minimal_quantity` and `_quantity_ceiling` from the portfolio. **29 → 25
      operators.**
- [x] `SearchState.touched_positions()` + `decode` flag in `run_search`, so only
      the shifts a step touched are re-derived.
- [x] `tests/test_fast_decode.py` — 110 tests.

### The three gaps the paper's sentence leaves

Each resolved toward its own binding rule, and each one measured:

| Gap | Resolution | Evidence it matters |
| --- | --- | --- |
| How much a source loads | Fills the trailer: loading is free once the trip is committed, capacity is the only bound | Deciding it from downstream demand made every leading source load 0 — V2.24 went 0 → 358 errors |
| "Least possible" at a VMI customer | `max(min_operation_quantity, deepest future safety shortfall)`, clipped to the tightest headroom over the whole remaining horizon | Using headroom *at arrival* charged 57 QS02 overfills on V2.24, since a drop raises the level at every later step |
| "Least possible" at a call-in customer | C24 flexible minimum less what is already booked against that same order; ceiling is C23's nominal | The QS01 interlude above |

### Two defects the decoder surfaced, both cross-shift

1. **Stale ceilings made the decode drift.** The surplus pass reused ceilings
   captured during the forward pass, so two stops at one customer each saw a
   bound that ignored the other's push. V2.24: 47 spurious QS02 breaches on the
   second sweep. Fixed by recomputing the bound live per candidate.
2. **Repeated decoding is stable but not idempotent, and that is inherent.** A
   customer's headroom depends on *other shifts'* deliveries, and shifts decode
   in trailer-chain order. Most instances settle exactly (V2.24/V2.13 by sweep
   3, V2.14 by sweep 9, each sweep lowering both errors and ratio). **V2.19
   enters a period-2 limit cycle** — two stops at one customer trading a fixed
   3,322.4 kg forever — bounded, with the error count pinned at 1385 and the
   ratio wobbling in the 5th decimal. So the test pins *stability*, not a fixed
   point.

### Does the surplus pass earn its place?

Standalone it looks bad: on V2.24 the forward pass alone decodes to **0 errors /
LR 0.03030**, and adding the surplus pass gives **7 errors / LR 0.02631**. The
errors are all *under*-delivery elsewhere (SHI16 minimum-drop, plus QS01/QS02
from the same cause): filling a tank through one shift starves another shift's
stop at that customer. Reserving headroom for other stops was tried and
**rejected** — 1 error of 13 recovered for an O(all shifts) scan in the inner
loop. The residual is temporal, not per-stop, so no per-shift decoder can see it.

But the standalone sweep is the wrong measurement. In the search (60 s, seed 1,
published `(errors, LR)`):

| inst | surplus=True | surplus=False |
| --- | --- | --- |
| V2.15 | **1** / 0.04794 | 1 / 0.05640 |
| V2.26 | 7 / 0.04621 | **6** / 0.04822 |
| V2.24 | **0** / 0.01846 | 0 / 0.02249 |
| V2.13 | **0** / 0.04655 | 0 / 0.05588 |

Equal or better errors on three of four, materially better ratio on all four:
the search repairs what the pass breaks and keeps the delivered quantity it
wins. `decode.SPEND_SURPLUS = True` stands, as a measured switch.

### Throughput

Wiring the decoder in cost 7,607 → 1,006 steps/s. Two fixes, both from a
profile rather than a guess:

1. `_tank_bounds` looped over the horizon per customer stop → vectorized with
   `np.cumsum`. Recovered to ~1,300.
2. The profile then showed **10.1 s of a 14.9 s run inside `_recompute_dirty`**,
   from 58,321 `set_quantity` calls. The decoder publishes each stop so later
   stops see it, but it only reads the *inventory projection*, which
   `_adjust_delivery` maintains by itself — the full rescore per stop was pure
   waste, and `_sum_shift_totals` made it O(stops × all shifts). Added staged
   primitives (`stage_operations`/`stage_quantity`/`finish_staged`) that publish
   without deriving, one `finish_staged` per shift. **~2,200–2,600 steps/s**, and
   rollback stays exact because `_touch` still snapshots the record.

Still ~3x under the pre-decoder baseline, which is the price of the decode
itself. The `decode=False` control puts the same runs at ~4,100–4,900 steps/s.

### Net effect on the open instances

V2.15 reached **1 internal error** with the decoder (60 s, seed 1) against **2**
before it, so the decoder is ahead of the hand-rolled quantity operators on the
instance closest to valid — and V2.15's official verdict before this was already
a single missed order.

---

## Interlude 3 — V2.15 closed: LLH0/LLH5 targeting and the flexibility boundary

**V2.15 is officially valid**, LR 0.055610,
`scratch/valid_V2.15_llh.xml` SHA-256
`d30d0dc5edb2b4a82e861b5e2abf77963f08f93be287fd215976ad9baf5800c8`. That makes
**10 of 15** Set B instances valid, and the first one closed by the rebuilt fast
search rather than by construction.

- [x] Add LLH0/LLH5 targeting (insert / replace using the customer with the
      earliest unsatisfied demand or order, TS2020 sec. 3.2). Operator count
      25 → 27; both clear the 90% firing gate.
- [x] Diagnose the last V2.15 error by reading the instance and reference XML.
- [x] Re-verify all 15 Set B candidates in one sweep (`scratch/verify_setb.py`).

### Why it was stuck: the decoder aimed at the one rejected value

The single error was `checkQS01 MissedOrder[0] of the customer[37]`. Order[0] is
`quantity=8000`, `window=[1440,1860]`, `orderQuantityFlexibility=80`, so the
satisfaction floor is `6400.0` — and the decoder delivered **exactly 6400.0** at
arrival 1808. The floor is exclusive: landing on it counts the order as missed.

This is a structural collision between TS2020's decoder rule and the checker.
"Assign the least possible amount at a customer" (sec. 3.1) makes the least
apparently-satisfying quantity the floor itself, so a faithful decoder converges
straight onto the rejected value. It is the QS01 ceiling defect from the other
direction, with the same error code — see the SKILL.md section on exclusive
"minimum to satisfy" bounds.

The repair is topological rather than numeric, since quantities are derived: the
targeted operators produced a route on which the order could be served properly.
The resulting solution is our own topology, 11 shifts / 53 operations against the
HUST reference's 21 / 80.

### The per-step tank count is the paper's measure, not a duplicate-counting bug

Checked against TS2020 sec. 3.1: the hard-constraint terms are "the number of
customer orders that were not met and **the time spent** by customers with an
inventory below their safety level." A count and a duration. The per-time-step
tank tally is correct and the large multiplicities are signal.

The reporting was wrong, though: summing a count and a duration into one "errors"
integer and comparing it across instances overweights long breaches. V2.18's 230
tank-steps span 9 distinct customers of 134; V2.22's 3,924 span 143 of 324.
Comparisons of search quality now use `(steps, distinct customers)`.

Also confirmed: the paper uses a **single manually weighted** objective over both
terms plus the ratio, with no lexicographic errors-then-ratio ranking. Our
lexicographic publication ranking is a local choice, justified by validity gating
rather than by the paper.

### Open

- V2.26 and V2.18 remain rejected. V2.18's checker breakdown is
  `DRI01 16, DRI03 4, DRI08 11, LAY02 3, DYN01 30` — driver and layover rules
  dominate, a different failure mode than the tank breaches previously assumed.
- The V2.15 seed-1 run logged 1 published error while both scorers report 0 on
  the saved file, so best-so-far selection and the logged best disagree about
  which state they describe. Worth localising before trusting run logs.

## Step 5 — Reintegrate and re-verify

- [ ] 5.1 New CLI path `native-solve --engine fast`, old path default.
- [ ] 5.2 Re-verify all 8 baseline instances through the new engine with the
      released checker. New artifacts, new hashes.
- [ ] 5.3 Attempt the 7 open instances.
- [ ] 5.4 Update `NATIVE_BENCHMARK_RESULTS.md` with measured results only.
      Withdraw nothing that is still true; claim nothing not checker-verified.
- [ ] 5.5 Promote durable findings into `skills/solve-roadef-irp/SKILL.md`
      (per project convention, general insights live there).
- [ ] 5.6 Switch the default engine only when >= 8 instances are valid through
      the new path.

**Gate:** >= 8 valid (no regression), with a genuine attempt logged on all 7
open instances.

**Contingency 5A — new engine regresses a currently-valid instance.** Keep the
old engine default and ship the new one as opt-in. Two engines is an
acceptable end state; a regression on a published artifact is not.

**Contingency 5B — new engine is fast but finds worse solutions.** Then
throughput was not the binding constraint and step 0's diagnosis was
incomplete. Re-diagnose with the new instrumentation, which will be far better
than what exists today. Record it here.

---

## Global contingency — if steps 1-3 do not move the open instances

Escalation order, cheapest first:

1. **Parallel independent searches** with different seeds and operator priors,
   sharing an elite pool. The analysis doc's recommended first parallel layer.
   Cheap, and we have cores.
2. **Randomised construction diversity** — many cheap seeds instead of two
   idle-cap variants, keeping the best.
3. **Selective exact optimisation, correctly placed:** timing MIP on one
   route, quantity LP on fixed visits, applied to *elite* candidates only,
   never during generation.
4. **Re-examine the constructor.** 7,008 vs 89 idle minutes against the
   reference says construction may be structurally wrong (too few, too long,
   too padded shifts). This is a bigger rewrite; only justified once the search
   substrate is fast enough to tell construction quality from search failure.

---

## Refuted / abandoned (append as we learn)

Carried over from `NATIVE_BENCHMARK_RESULTS.md` and project memory — do not
retry these:

- Layovers blocking retiming (layover-free hosts did *worse*: 0% vs 8.9%).
- Sparse candidate start times in the placement gate (enriching starts placed
  0 of 833; output byte-identical).
- Tighter construction idle caps on V2.15 (60/90/120 gave 422/421/384 vs
  uncapped 346; all far worse than the resume path's 71).
- "Constructor coverage" as the cause (it already serves 137/140 naturally
  breaching V2.12 tanks).
- Dispatch-before-breach clamping (provable no-op under a monotone inventory
  projection; implemented, measured zero change, reverted).
