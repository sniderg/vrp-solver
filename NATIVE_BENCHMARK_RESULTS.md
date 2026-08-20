# Native Solver Validation Status

The previous version of this document claimed that all 20 tracked native
solutions passed the official checker. That claim was false. The validation
path used at the time was not the released ROADEF V2 checker and produced false
positives.

Validity is now fail-closed: a result is publishable only when
`vrp-solver verify-official` runs the released checker and observes its exact
`THIS OUTPUT IS VALID` sentinel, a zero exit status, and no failure sentinel.

## Released-checker re-audit of the historical native XMLs

Re-audited 2026-08-05 using `Checker_V2.2_07032016.zip`, SHA-256
`fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a`.

| Instance | Provenance | Released checker | First reported hard rule |
| --- | --- | ---: | --- |
| V2.12 | historical native candidate | **INVALID** | DYN01 |
| V2.13 | historical native candidate | **INVALID** | LAY02 |
| V2.14 | historical native candidate | **INVALID** | DYN01 |
| V2.15 | historical native candidate | **INVALID** | SHI04 |
| V2.16.2 | historical native candidate | **INVALID** | SHI04 |
| V2.17 | historical native candidate | **INVALID** | DYN01 |
| V2.18 | historical native candidate | **INVALID** | DYN01 |
| V2.19 | historical native candidate | **INVALID** | DYN01 |
| V2.20.2 | historical native candidate | **INVALID** | DYN01 |
| V2.21.2 | historical native candidate | **INVALID** | DYN01 |
| V2.22 | historical native candidate | **INVALID** | DYN01 |
| V2.23 | historical native candidate | **INVALID** | DYN01 |
| V2.24 | historical native candidate | **INVALID** | LAY02 |
| V2.25 | historical native candidate | **INVALID** | LAY02 |
| V2.26 | historical native candidate | **INVALID** | LAY02 |
| X1 | historical native candidate | **INVALID** | DYN01 |
| X2 | historical native candidate | **INVALID** | SHI04 |
| X3 | historical native candidate | **INVALID** | DYN01 |
| X4 | historical native candidate | **INVALID** | DYN01 |
| X5 | historical native candidate | **INVALID** | DYN01 |
| **Total** | historical native candidates | **0/20 valid** | — |

The old costs, delivered volumes, logistic ratios, “wins”, and aggregate score
were calculated for invalid outputs and are withdrawn. Logistic ratio is only
meaningful among officially valid solutions.

## 2026-08-19 integrity audit and regression fix

- **Inputs authenticated:** the checker archive and all 20 Set B/Set X instance
  XMLs are byte-identical to fresh roadef.org downloads. The checker hash is
  now enforced in code and the instance hashes pinned
  (`src/vrp_solver/instance_manifest.py`, commit `2f2b437`).
- **Withdrawn:** the `f2b58f7` commit-message claim of a "complete multi-stop
  cold-start solver ... across all Set B and Set X instances"
  (`solve_instance.py`). No officially valid artifact from that pipeline
  survives re-verification; see the script's docstring for the measured
  defects. A same-day session also circulated an 11-row "all VALID in 5-11 s"
  table: 9 rows had no artifact at all, and the two artifacts that existed
  (V2.12, X1) are rejected by the released checker (391 and 1,159 runouts).
- **Regression fixed:** `f2b58f7` had routed every instance with a horizon
  >= 28 days through the untested horizon-master constructor, bypassing the
  proven cluster constructor (V2.14 seed: 0 errors -> 62,053). Fixed in
  `4338493` (horizon-master is now one portfolio candidate, never a
  replacement).
- **Reference restored:** the supplied V2.12 reference solution had been
  overwritten with an invalid file; restored from git and re-verified
  (`official_valid,True`, LR 0.018496, SHA-256 `61a7ef87b50f...`).
- **Confirmation sweep (in progress at pause):** `native-solve-batch`, seed 1,
  1800 s, HiGHS only — first 8 results all VALID: V2.13 0.070567,
  V2.14 0.084934, V2.16.2 0.042634, V2.19 0.096702, V2.20.2 0.031588,
  V2.21.2 0.033473, V2.24 0.028583, V2.25 0.038978 (exact or near-exact
  reproductions of the milestone table below). V2.15 (7 errors) and V2.26
  (8 errors, safety deficit 0) were still descending at pause;
  V2.12/V2.17/V2.18/V2.22/V2.23 sat at their documented plateaus. Artifacts
  and logs: `out/setB_sweep/`. **Gurobi is not required for any valid
  result**: the shipping path is Python/Cython + HiGHS; Gurobi remains an
  optional substitute for the quantity-repair MILP only
  (`ROADEF_SOLVER=gurobi`, single-process license).

## Current demonstrated milestones

Eleven of the fifteen Set B instances are officially valid (see the
best-known table below; open: V2.17, V2.18, V2.22, V2.23). Benchmark policy as of
2026-08-10: a result is competition-comparable only when produced by a single
30-minute run (construction + search, one seed, no resumed state), matching the
official protocol. Chained resume rounds remain a legitimate exploration tool
but are reported separately and never share a table with budget-compliant rows.

### Competition-budget regression, 2026-08-10

One `native-solve-batch` sweep over all 15 instances — seed 1, 1800 s each,
released-checker verification inside the run. Logs and artifacts under
`scratch/regress_full/`; every valid row is a genuine cold start.

| Instance | Released checker | LR (single 30-min run) | vs. milestone |
| --- | ---: | ---: | --- |
| V2.13 | **VALID** | 0.070567 | slightly better |
| V2.14 | **VALID** | 0.084934 | exact reproduction |
| V2.16.2 | **VALID** | 0.042634 | exact reproduction |
| V2.19 | **VALID** | 0.096702 | exact reproduction |
| V2.20.2 | **VALID** | 0.031588 | better |
| V2.21.2 | **VALID** | 0.033473 | slightly worse |
| V2.24 | **VALID** | 0.028583 | slightly worse |
| V2.25 | **VALID** | **0.038978** | **now closes inside 30 min** (milestone needed ~80 min) |
| V2.12, V2.15, V2.17, V2.18, V2.22, V2.23, V2.26 | INVALID | — | do not close from cold in 30 min |

All eight documented cold starts reproduce as valid at the competition budget,
so the recent engine work (quantity decoder, staged primitives, Cython kernels,
LLH0/LLH5, roulette selector) did not disturb the shipping path — expected,
since `surgical_search` remains the default engine until step 5. The V2.25 row
retires that instance's 80-minute exception:
`scratch/regress_full/V2.25_native.xml`, SHA-256
`f84829c3fd09df04465c388b067579ea7ba612b0e44df67d23b370525d657282`.
Three exact LR reproductions pin the pipeline as unchanged; the small wobbles
on V2.13/V2.20.2/V2.21.2/V2.24 are run-to-run variation of the restart-round
schedule.

### Best-known artifacts (any provenance)

**2026-08-20 fast-polish sweep** (`scratch/lr_polish_run.sh`, 15-min
fast-search resume from each best valid artifact, publication incumbent
ranked (errors, LR)): **all 11 valid instances improved**, re-verified with
the released checker. New best LRs, artifacts at
`artifacts/valid/<inst>_lrpolish.xml`:

| Instance | Previous best | Fast-polished | Change |
| --- | ---: | ---: | ---: |
| V2.12 | 0.027209 | **0.016696** | -38.6% |
| V2.13 | 0.042045 | **0.041129** | -2.2% |
| V2.14 | 0.075429 | **0.072574** | -3.8% |
| V2.15 | 0.039686 | **0.039516** | -0.4% |
| V2.16.2 | 0.025961 | **0.021969** | -15.4% |
| V2.19 | 0.080677 | **0.077745** | -3.6% |
| V2.20.2 | 0.031151 | **0.031124** | -0.1% |
| V2.21.2 | 0.032982 | **0.032366** | -1.9% |
| V2.24 | 0.020387 | **0.017312** | -15.1% |
| V2.25 | 0.027588 | **0.025184** | -8.7% |
| V2.26 | 0.030957 | **0.027171** | -12.2% |

**2026-08-20, later:** the new reload-insertion MIP
(`src/vrp_solver/joint_restructure.py`) improved two rows further, both
re-verified: **V2.12 0.016696 -> 0.016037** (one in-gap reload) and
**V2.16.2 0.021969 -> 0.021402** (two), artifacts
`artifacts/valid/{V2.12,V2.16.2}_reload_mip.xml`. On the other nine the MIP
proves quantities are already at their tank-ceiling optimum for the fixed
topology - their remaining LR headroom is topological.

V2.12 at **0.016037** is the first artifact to beat the supplied reference
solution (0.018496). The table below predates this sweep; where it
disagrees, the fast-polished numbers above are the current best.


Every row re-verified with `Checker_V2.2_07032016.zip` against the exact XML
named; `scratch/verify_setb.py` reproduces the sweep.

| Instance | Provenance | Released checker | LR (Baseline) | LR (Polished Multi-Drop) | LR Improvement | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V2.12 | native-repair | **VALID** | 0.027209 | — | Baseline | `scratch/v212_skill_orders_final_local.xml` |
| V2.13 | deep-polish | **VALID** | 0.070567 | **0.042045** | **-40.42%** | `out/deep_polished/V2.13_deep_polished.xml` |
| V2.14 | deep-polish | **VALID** | 0.084934 | **0.075429** | **-11.19%** | `out/deep_polished/V2.14_deep_polished.xml` |
| V2.15 | **native-cold-start (fast engine, single 30-min run)** | **VALID** | **0.039686** | — | Baseline | `artifacts/valid/V2.15_cold_fast.xml` |
| V2.16.2 | deep-polish | **VALID** | 0.042634 | **0.025961** | **-39.11%** | `out/deep_polished/V2.16.2_deep_polished.xml` |
| V2.19 | deep-polish | **VALID** | 0.096702 | **0.080677** | **-16.57%** | `out/deep_polished/V2.19_deep_polished.xml` |
| V2.20.2 | deep-polish | **VALID** | 0.031588 | **0.031151** | **-1.38%** | `out/deep_polished/V2.20.2_deep_polished.xml` |
| V2.21.2 | native-cold-start | **VALID** | **0.032982** | 0.033015 (worse; baseline kept) | Baseline | `scratch/opt_V2.21.2_native.xml` |
| V2.24 | deep-polish | **VALID** | 0.027025 | **0.020387** | **-24.56%** | `out/deep_polished/V2.24_deep_polished.xml` |
| V2.25 | deep-polish | **VALID** | 0.035982 | **0.027588** | **-23.33%** | `out/deep_polished/V2.25_deep_polished.xml` |
| V2.26 | deep-polish | **VALID** | 0.036609 | **0.030957** | **-15.44%** | `out/deep_polished/V2.26_deep_polished.xml` |
| V2.17 | fast-search (chained) | INVALID | — | — | — | `scratch/chain_V2.17_best.xml` (186 internal errors) |
| V2.18 | fast-search (LLH0/LLH5) | INVALID | — | — | — | `scratch/camp_V2.18_s11.xml` (116 internal errors) |
| V2.22 | fast-search (decoder) | INVALID | — | — | — | `scratch/dec_V2.22.xml` |
| V2.23 | fast-search + MIP | INVALID | — | — | see note below | `out/V2.23_INTEGRATED_RESULT.xml` |

All nine `deep_polished` rows re-verified 2026-08-20 with the released checker
(fail-closed wrapper); the LRs above are the checker's own numbers. The
intermediate `out/polished/*_polished.xml` generation is also valid at slightly
higher LRs. **The newer `out/universal_polished_*.xml` generation is rejected
by the checker on all nine instances — do not cite it.** V2.21.2's polish came
out marginally *worse* than its baseline, so the baseline artifact remains the
best-known.

## 2026-08-20 seed-portfolio experiment: V2.26 closed, twelfth valid instance

Eight parallel 30-minute cold runs per instance (seeds 2-9), then one
`--resume-from` round on the best checkpoint (scripts:
`scratch/portfolio_run.sh`; artifacts: `out/portfolio/`).

- **V2.26 CLOSED (chained-native): `official_valid,True`, LR 0.050572.**
  Seed 3 reached 1 error in its 30-minute budget; a 27-second resume round
  finished it. ~30.5 minutes total compute, instance XML + seeds only,
  fully hands-off. `out/portfolio/V2.26_s3_resume.xml`, SHA-256
  `597059138e30bd40f0aca053f750b33d37f7acabbc4d423634861108fc8fe673`.
  Not a single-run result; the best-known LR remains the deep-polished
  0.030957. Every one of the 8 seeds beat seed 1's 4-error plateau ceiling
  at round 6 pace or better (finals: 1,1,2,2,2,4,4,5).
- **V2.12: seed choice is worth 3x.** Seed 7 ended at 38 errors vs seed 1's
  119; a further 30-minute resume moved only 38 -> 37 (hard plateau,
  consistent with the trapped-capacity diagnosis — more budget does not
  help; the lever is constructor coverage / joint-slack repair).
- **Fast-engine CLI batch on the open four** (`native-solve-batch --engine
  fast`, first CLI reproduction of the 08-11 measurements): V2.17
  6,312 -> 491, V2.18 4,243 -> 729, V2.22 6,352 -> 953, V2.23 255 -> 112.
  All invalid, exploration only (`out/fast_open4/`). V2.23's 112 matches its
  documented plateau exactly.

Set B validity count: **11 of 15** (open: V2.17, V2.18, V2.22, V2.23; an
earlier revision of this line said 12 — a double-count of V2.26's
re-closure).

### 2026-08-20 artifact loss and recovery

The `748602b` scratch cleanup deleted the on-disk valid artifacts. Seven were
restored from git history and re-verified at their exact documented LRs.
V2.15's two valid artifacts (0.039373 and 0.055610) were untracked and are
lost; the recorded SHA-256s remain in this file's history. **Re-derivation
upgraded the claim:** three of four fresh `native-solve --engine fast` seeds
closed V2.15 in single 30-minute cold runs (LRs 0.047948 / 0.049595 /
**0.039686**), so V2.15 now closes at the competition budget — the lost
artifact had required chaining. Best artifact:
`artifacts/valid/V2.15_cold_fast.xml`, SHA-256 `949cb5b3fb53...`.
All valid XMLs now live in git-tracked `artifacts/valid/` with a SHA256SUMS
manifest; `scratch/` and `out/` must never be the only home of a
publishable artifact.

Same session, V2.23 fast-engine seed portfolio (seeds 2-5): finals
62 / 96 / 122 / 144 vs seed 1's 112-error plateau — **seed 2's 62 errors is
the best state V2.23 has ever reached**. The chained fast resume then ran
2.18M steps and moved 62 -> **61**: seed choice relocates the plateau, search
cannot descend from it. Checker breakdown of the 61-error state
(`out/v223_fast/V2.23_s2_resume.xml`): SHI06 x203 (trailer stock
consistency) dominates, plus DRI01 x7 and singleton LAY/SHI/TL codes — the
residual is a resource/stock-chain restructuring problem, reinforcing that
the lever is exact resource/quantity repair inside the search
(interval-clique MIP + quantity LP), not more budget.

**Decisive follow-up (2026-08-20):** a strict full-horizon quantity MILP
over the 61-error topology (fixed routes, `strict_inventory=True`) is
**Infeasible** — no quantity assignment makes this topology valid. The
boundary crossing therefore requires joint topology+resource restructuring,
not requantification; quantity-only repair on the plateau states is refuted
as a path to closing V2.23.

**Triple confirmation (2026-08-20):** the surgical engine (with the
structural operators that closed V2.26) ran 8 rotation rounds from the same
61-error state and moved it **zero** (61 -> 61). V2.23's boundary state is
now confirmed unreachable by (a) 2.18M fast-search steps, (b) an exact
quantity MILP over the fixed topology (infeasible), and (c) the full
surgical operator rotation. Only a joint topology+resource+quantity
restructuring move remains untried. **Extended 2026-08-20 (later):** the
same strict-LP infeasibility certificate now covers V2.17, V2.18, and
V2.22's best states as well - all four open instances provably require
structural change, not requantification. The in-gap reload MIP is also
infeasible on all four (3-27 candidates each). MILP monitor verdict: the
largest model we build (V2.23 min-delivered, 213k cols, 819 binaries)
root-solves in 25 s at 1 B&B node - HiGHS is not the bottleneck and a
stronger MIP solver would change nothing today (`out/highs_timings.csv`). Contrast V2.17: the same surgical
resume ground 491 -> 461 in 30 min — it is a depth/throughput grind, not a
boundary problem.

## 2026-08-20/21 joint restructuring MIP: every open instance moves

`src/vrp_solver/joint_restructure.py`, built in one day through five
escalations, each driven by an infeasibility certificate: quantities ->
+in-gap reloads -> +removals/end-insertions -> +customer stops -> +new-shift
columns in joint driver+trailer idle windows (multi-failure greedy-max-fill
coverage, pairwise overlap exclusions, penalty slack for provably
uncoverable customers). All error counts below are local; none of these
states is valid yet — they are the best exploration states each instance
has ever reached:

| Instance | Before (best ever) | After MIP(+search) | Mechanism |
| --- | ---: | ---: | --- |
| V2.23 | 61 | **15** | v4 Optimal (7 activations) + surgical |
| V2.18 | 638 | **166** | soft-all, 21 activations |
| V2.17 | 186 (5h chain) / 461 | **191..195** | soft-all, 18-21 activations, seconds |
| V2.22 | 953 | **288** | soft-all, 50 activations — Gurobi only |

**HiGHS/Gurobi capacity frontier (measured on this model family):**
V2.17/V2.18 (~75k vars, ~650 binaries): HiGHS 47/333 s, Gurobi 7.7/12.7 s.
V2.22 (117k vars, 2,778 binaries): HiGHS times out at 900 s (289 B&B nodes,
no incumbent); **Gurobi solves it in 24.5 s.** Set `ROADEF_SOLVER=gurobi`
for restructure MIPs above ~700 binaries; every extracted solution is
replayed through local validation and the released checker, so MPS
translation cannot fake a success. The `milp_monitor` slow-solve log
(`out/highs_timings.csv`) is the trigger for routing new model families.

## 2026-08-20 Multi-drop polish architecture (claims corrected 2026-08-20)

1. **Continuous multi-drop chaining & payload maximization**: maximizes
   delivered payload per shift within tank headspace and trailer capacity.
   Measured on the nine valid Set B artifacts above: LR reductions up to
   -40.42%, all re-verified officially valid. (An earlier draft of this
   section claimed "100% validity with zero regressions" — V2.21.2 regressed
   slightly and its baseline is kept instead.)
2. **Interval-clique MIP resource assignment** (`scipy.optimize.milp`,
   in `src/vrp_solver/fast/universal_polish.py`): assigns shifts to
   driver/trailer pairs. The "completely resolves all discrete resource
   conflicts" claim from the earlier draft is **withdrawn**: the surviving
   integrated artifact (`out/V2.23_INTEGRATED_RESULT.xml`) still fails the
   checker with DRI03 x3, SHI04 x38, LAY02 x1, plus DYN01 x15,256 and
   SHI06 x65,550; the claimed V2.17/V2.18/V2.22 integrated artifacts do not
   exist on disk.
3. **Convex customer inventory LP tuner** (`scipy.optimize.linprog`): tunes
   drop sizes along piecewise tank trajectories. Ships inside
   `universal_polish.py`, whose end-to-end outputs are currently rejected by
   the checker (see above) — the tuner is not yet validated in isolation.

V2.15's best artifact improved on 2026-08-10 from LR 0.055610 to **0.039373**
(a 2-minute fast-search polish of the valid solution, re-verified;
`scratch/valid_V2.15_lr2.xml`, SHA-256
`11591f5e3ad012ac9dab143c0065439b6864f330ff554f93959524c2e8ab88c2`). The
original closure artifact remains `scratch/valid_V2.15_llh.xml` at 0.055610.

### Exploration ceilings on the open instances (not competition-comparable)

Chained fast-search resume rounds (25 min each, each seeding from the prior
best) establish how far the current operator set can descend, irrespective of
budget:

- **V2.17: 1,206 -> 186 errors** over 12 rounds (~5 h), improving every round
  with no plateau at cutoff. The instance is closable by this operator set;
  what is missing is reaching that depth inside 1800 s.
- **V2.26: CLOSED 2026-08-11** (6 -> 0 errors, officially valid, LR 0.036609,
  `scratch/valid_V2.26_surgical.xml`, SHA-256
  `1172aecfe322ac19dfb4fd7924c9e10b31be4cf5797fee0d621270d50eb97775`). The
  descent: surgical rounds with the new `multiroute_pressure_block` /
  `pressure_band_resource_block` operators took 6 -> 5 -> (4 after a manual
  merge probe) -> 1; the last two errors fell to hand-constructed probes that
  are candidate production operators. (a) The DRI01 on a lone-reload shift was
  closed by *merging the reload into the predecessor shift* (same driver +
  trailer) and deleting the shift — delaying it was provably infeasible because
  the driver's 550-min gaps bound on both sides. (b) The final missed order
  (customer 10, floors of three co-hosted call-ins exceeded every reachable
  trailer's capacity) was closed by a *balanced reload+stop insertion*: append
  the delivery stop to a timing-feasible shift on a different trailer and add a
  same-size reload at the route head, so the trailer leaves the shift with its
  stock unchanged and nothing cascades downstream. Both moves are now
  production operators (`merge_lone_reload_shift`,
  `balanced_reload_stop_insert` in `surgical_search.py`, wired into the
  coverage rotation, the DRI01 structural schedule, and `native-solve`'s
  restart rotation, with unit tests). This artifact is
  a repaired native-lineage artifact, not a cold start.
- **V2.26 hands-off CLI closure (2026-08-11, same day):** with the new
  operators shipped, a fully autonomous chain closed the instance from the
  instance XML alone. `native-solve --seed 2 --time-limit 1800
  --restart-rounds 8` reached 1 error (construction seeded 13;
  `balanced_reload_stop_insert` accepted as a new best during the descent),
  then one `native-solve --resume-from` round (~4 min) closed it:
  `recombine_route_blocks` restructured the window, `create_shift` served the
  final order. Officially valid, LR 0.042263, `out/V2.26_coldstart_s2r.xml`,
  SHA-256 `71d5a200de02c65cc9c9ecc673e7efdcc9a80377b5b0d4ed082e88dcc3060d48`.
  A chained native result (~34 min total), not a single-budget cold start:
  seed 1 at the same budget plateaus at 2 errors, so the 30-minute single-run
  claim remains open. Best LR remains the surgical artifact (0.036609).
- **V2.18: 188 -> 116 errors** in one 15-minute run; driver/layover rules
  (DRI01/DRI08/LAY02) dominate its checker breakdown, a different failure mode
  from the tank-dominated instances.
- **V2.18 fast-engine cold runs (2026-08-11):** `native-solve --engine fast`
  (the step-5 CLI integration of the rebuilt search) took the cold seed
  4,243 -> **729** errors in one 30-minute budget (seed 1; seed 2 reached
  890), where the surgical engine made zero progress from the same seed —
  its 220-second rounds stall outright at 136 points. One fast
  `--resume-from` round continued 729 -> 638, so the chain still descends
  but the tail is a grind. Two structural blockers measured: the constructor
  leaves **69 of 134 customers unscheduled** (the seed itself is the
  ceiling), and both idle-cap settings now produce identical 4,243-error
  seeds on this instance. Artifacts: `out/V2.18_fast_s1.xml`,
  `out/V2.18_fast_r2.xml` (both invalid, exploration only).
- **V2.17 / V2.22 / V2.23 fast-engine budget runs (2026-08-11, seed 1, one
  30-minute `native-solve --engine fast` each, all invalid/exploration):**
  V2.17 seed 6,312 -> **508** (the previous 186-error mark needed ~5 hours of
  chained rounds; one budgeted run now lands within 3x of it). V2.22 seed
  6,352 -> **955**. V2.23 seed **255** -> **112** — the constructor now seeds
  V2.23 an order of magnitude better than the documented 2,642, and it is the
  clear next-closest instance. Its residual is spread across codes
  (85 QS02, 19 QS01, 11 DRI01, plus a few SHI/LAY), so the next lever there
  is chained fast resume rounds, then the surgical portfolio near zero.
  Artifacts: `out/V2.1{7,22,23}_fast_s1.xml`... (`out/V2.23_fast_s1.xml`).
  Continuation measured the plateau precisely: a second fast round ran
  **2,008,044 steps without moving 112**, and eight surgical rounds ground
  112 -> **104** (`out/V2.23_surgical_r3.xml`). The residual is
  82 QS02 tank-safety breach steps (safety deficit 435,670 qm) plus
  19 QS01 / 11 DRI01 — the trapped-capacity/resource-cadence class, the same
  signature as V2.15's plateau, not a single missing move. Next lever is
  seed coverage / joint-slack repair at scale, not more search budget.

### V2.15, and what closed it

V2.15 is the tenth instance and the first closed by the rebuilt fast search
rather than by construction. Its single remaining error was
`checkQS01 MissedOrder[0] of the customer[37]`, and reading the instance XML
made the cause exact. That order is `quantity=8000`,
`window=[1440,1860]`, `orderQuantityFlexibility=80`, so the least amount that
satisfies it is `0.80 * 8000 = 6400.0` — and the decoder was delivering
**exactly 6400.0**. The quantity decoder's "least possible amount" rule
(TS2020 sec. 3.1) lands precisely on the satisfaction boundary, and the released
checker resolves that tie against the solution: sitting *on* the minimum ratio
counts the order as missed. The HUST reference delivers the full 8000 to the
same customer.

The fix was not a quantity change. LLH0/LLH5 targeting (insert or replace using
the customer with the earliest unsatisfied demand or order, per TS2020 sec. 3.2)
gave the search a route on which the order could be served properly, and the
resulting topology is our own: 11 shifts / 53 operations, against the HUST
reference's 21 / 80.

`scratch/valid_V2.15_llh.xml`, SHA-256
`d30d0dc5edb2b4a82e861b5e2abf77963f08f93be287fd215976ad9baf5800c8`
(a stable copy of the run output `scratch/llh_V2.15.xml`).

One discrepancy is open and worth pinning down: the run log recorded 1 published
error for the V2.15 seed-1 row, while both the internal scorer and the checker
report zero on the saved file. The best-so-far selection and the logged best
therefore disagree about which solution they are describing.

The eight cold starts were produced on 2026-08-07 by `vrp-solver native-solve`
(instance XML + seed only; V2.25 additionally continued through
`surgical_search` restart rounds from its own native checkpoint) and verified
with the released checker on Windows. Seven of the eight finished inside the
official 30-minute budget at the time; the 2026-08-10 regression above closed
V2.25 inside the budget as well, so all eight are now demonstrated at
competition conditions. Exact XMLs (SHA-256):

- V2.13: `scratch/replicate_V2.13_native.xml`
  `6ec05835c64f41a002222e0792d132298edd4db53063ee632d0af4c02d68a9f2`
- V2.16.2: `scratch/cold_V2.16.2_batch.xml`
  `527becb29ddc9d21f835b01c19ede4844d86e09e1f13a37971f8cce91fba1a8b`
  (LR 0.042634, produced by `native-solve-batch`, superseding
  `scratch/opt_V2.16.2_native.xml` at LR 0.058143)
- V2.19: `scratch/opt_V2.19_native.xml`
  `a2063f882a53abff71f1c0a3c934f6dbd835fc63ddca5f97a084a03704ee4c1d`
- V2.20.2: `scratch/opt_V2.20.2_native.xml`
  `f96c58268188efdc46cb8f320de7370cd6f3e2bc258a949c75e241e3284b13eb`
- V2.21.2: `scratch/opt_V2.21.2_native.xml`
  `237cab0c61d310427dc2d50e1aa106704ee3fd293e8bc9fce3d489a40381db05`
- V2.24: `scratch/replicate_V2.24_native.xml`
  `5b74a6f36729937d5cece9d30bc9d04eab8f7f384bc96579ca43259906609e85`
- V2.25: `scratch/opt3_V2.25_native.xml`
  `a407cde5d2f0ddc1ba34a0471328ac7cd374067c48b046aee07b9b2f1c4bedc7`
- V2.14: `scratch/cold_V2.14_cadence.xml`
  `adec2c4f67100ffdb94bcaec244a5322ba00dcfa5c5a05d60cc24ae5f5e9c1bb`

V2.14 is the first instance to reach zero errors from construction alone, with
no topology search at all, using the mid-route idle cap
(`native-solve --idle-cap 180`, 73 shifts, 282 operations, 32 s).

Remaining Set B instances (V2.17, V2.18, V2.22, V2.23, V2.26, and a V2.12 cold
start) have not reached zero errors. The earlier "constructor
coverage" diagnosis was wrong: measurement showed the constructor already
serves 137 of 140 naturally breaching V2.12 tanks. The measured cause is
resource cadence — 46,624 idle minutes against 34,273 travel minutes, with 40
of 63 late first visits occurring while no shift was even under way.

Best cold-start error counts after a 25-minute search round with the idle-cap
seed portfolio (seed 1), for tracking the remaining gap.  The "resume" column
is after one `--resume-from` continuation round (~25 min each):

| Instance | Seed errors (uncapped) | Seed errors (cap 180) | After one search round | After resume round |
| --- | ---: | ---: | ---: | ---: |
| V2.12 | 2,135 | 1,402 | 1,331 | — |
| V2.15 | 622 | 732 | 189 | **36** (native resume with repair-time compaction; local only) |
| V2.17 | 11,520 | 6,082 | 5,889 | — |
| V2.18 | 13,017 | 2,115 | 1,769 | — |
| V2.22 | 14,402 | 6,245 | 6,176 | — |
| V2.23 | 3,482 | 2,642 | 2,598 | — |
| V2.26 | 614 | 544 | 222 | **74** (negative inventory eliminated) |

Resume rounds improved every tested open instance but all flattened after round
1.  The cause is measured: on V2.15, 13 of 25 search steps produced zero
candidates; `create_shift` builds 833 routes and all 833 die at the
resource-placement gate (1,052 attempts found a driver gap; zero found the
trailer free simultaneously — only 3 joint driver+trailer idle windows ≥300 min
exist across the full horizon, even though each resource alone is 44–76% idle).

Three theories were refuted with data: layovers blocking retiming, sparse
candidate start times, and tighter construction idle caps (60/90/120 all
finished worse than uncapped).  The idle cap that closed V2.14 does not
generalise; construction policy cannot compact routes an incumbent already
committed.

**Implemented mechanism and remaining blocker:**

1. **Repair-time compact-and-place.** The create-shift gate now falls back to a
   bounded joint timing MIP that compacts a connected driver/trailer block and
   places the new route atomically. On V2.15 this turned an empty gate into one
   candidate; normal quantity repair accepted it and lowered the native resume
   from 70 to **36** local errors (23 shifts). This is not a valid result: the
   released checker has not been run because local errors remain nonzero.
2. **Sequence redesign remains required.** A 600-second continuation from the
   36-error checkpoint returned to zero create-shift candidates. All 612 raw
   routes were trailer-compatible, but none fit a compacted eight-shift block in
   the incumbent resource order. The supplied V2.15 comparison XML has 31 shifts
   and just 89 in-route idle minutes, versus 22 shifts / 5,671 idle minutes in
   the pre-compaction native checkpoint. The next production operator must
   choose a larger resource-chain sequence (not merely retime a fixed order).
3. **Strict inventory is a proof gate, not a substitute for topology.** Applying
   `strict_inventory=True` to the newly reachable V2.15 route is infeasible:
   the route reduces safety errors but cannot cover every remaining tank. Keep
   strict repair for feasibility proof once a candidate topology has sufficient
   coverage; use soft repair while constructing that topology.

The V2.12 row is a native repair of a pre-existing candidate
(`scratch/v212_skill_orders_final_local.xml`, SHA-256
`32d5905ffd7495fbc37d4ad5b26d2d6dd4a589246a1f93b2f89f925c2a83b2f3`); a V2.12
cold start has not yet reached zero errors.

For comparison, the supplied V2.12 reference is also officially valid with
2,431,172.363 L delivered, shift cost 44,966.73, and LR 0.018496.

## Reproduction

### Reproduce a native cold start from scratch

V2.14 is the cheapest end-to-end check: the constructor alone closes it, so the
run takes about a minute. Solve, then verify.

```bash
vrp-solver native-solve \
  roadef_2016_data/set_B/Instances_B_V25-11042016/V2.14.xml \
  out/V2.14_native.xml \
  --seed 1 --time-limit 1200 --no-improvement-limit 10000 --restart-rounds 1

vrp-solver verify-official \
  roadef_2016_data/set_B/Instances_B_V25-11042016/V2.14.xml \
  out/V2.14_native.xml
```

Expected: `local_errors,0` and `search_steps,0` from the solve, then
`official_valid,True` with `official_logistic_ratio,0.084934`.

Substitute `.venv/Scripts/python.exe -m vrp_solver.cli` or `uv run vrp-solver`
for `vrp-solver` if it is not on `PATH`. The other instances in the table above
need longer time limits and do not close from construction alone; see the
runbook in [README.md](README.md#run-a-native-cold-start-solve-start-here) for
seed portfolios, flags, and what to do when a run stalls.

### Reproduce the whole corpus in one command

Every entry in the milestone table comes from the same `native-solve` pipeline,
so the corpus reproduces without any per-instance scripting:

```bash
vrp-solver native-solve-batch \
  roadef_2016_data/set_B/Instances_B_V25-11042016 \
  out/setB \
  --seed 1 --time-limit 1800 --concurrency 7 \
  --summary-csv out/setB/summary.csv
```

This solves each instance in its own process, then verifies each output with the
released checker and exits non-zero unless all are valid. Use
`--only V2.14 V2.16.2` for a two-minute smoke test; both are expected `valid`
at LR 0.084934 and 0.042634.

### Re-verify a published artifact

```bash
vrp-solver verify-official \
  roadef_2016_data/set_B/Instances_B_V25-11042016/V2.14.xml \
  scratch/cold_V2.14_cadence.xml
```

Expected publication marker:

```text
official_status,valid
official_valid,True
```

Checker archive `roadef_2016_data/Checker_V2.2_07032016.zip` must be present,
SHA-256 `fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a`.

Local simulation and native rule checks remain useful diagnostics, but they do
not confer official validity.
