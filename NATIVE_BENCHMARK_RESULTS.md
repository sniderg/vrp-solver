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

## Current demonstrated milestones

| Instance | Provenance | Released checker | Wall time | LR |
| --- | --- | ---: | ---: | ---: |
| V2.13 | **native-cold-start** | **VALID** | 345 s | — |
| V2.16.2 | **native-cold-start** | **VALID** | 62 s | 0.042634 |
| V2.19 | **native-cold-start** | **VALID** | 104 s | 0.096702 |
| V2.20.2 | **native-cold-start** | **VALID** | 150 s | 0.032622 |
| V2.21.2 | **native-cold-start** | **VALID** | 678 s | 0.032982 |
| V2.24 | **native-cold-start** | **VALID** | 789 s | — |
| V2.25 | **native-cold-start** | **VALID** | ~80 min total | 0.035982 |
| V2.14 | **native-cold-start** | **VALID** | 32 s | 0.084934 |
| V2.12 | native-repair | VALID | — | 0.027209 |

All eight cold starts were produced on 2026-08-07 by `vrp-solver native-solve`
(instance XML + seed only; V2.25 additionally continued through
`surgical_search` restart rounds from its own native checkpoint) and verified
with the released checker on Windows. Seven of the eight finished inside the
official 30-minute budget. Exact XMLs (SHA-256):

- V2.13: `scratch/replicate_V2.13_native.xml`
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
- V2.25: `scratch/opt3_V2.25_native.xml`
  `a407cde5d2f0ddc1ba34a0471328ac7cd374067c48b046aee07b9b2f1c4bedc7`
- V2.14: `scratch/cold_V2.14_cadence.xml`
  `adec2c4f67100ffdb94bcaec244a5322ba00dcfa5c5a05d60cc24ae5f5e9c1bb`

V2.14 is the first instance to reach zero errors from construction alone, with
no topology search at all, using the mid-route idle cap
(`native-solve --idle-cap 180`, 73 shifts, 282 operations, 32 s).

Remaining Set B instances (V2.15, V2.17, V2.18, V2.22, V2.23, V2.26, and a
V2.12 cold start) have not reached zero errors. The earlier "constructor
coverage" diagnosis was wrong: measurement showed the constructor already
serves 137 of 140 naturally breaching V2.12 tanks. The measured cause is
resource cadence — 46,624 idle minutes against 34,273 travel minutes, with 40
of 63 late first visits occurring while no shift was even under way.

Best cold-start error counts after a 25-minute search round with the idle-cap
seed portfolio (seed 1), for tracking the remaining gap:

| Instance | Seed errors (uncapped) | Seed errors (cap 180) | After one search round |
| --- | ---: | ---: | ---: |
| V2.12 | 2,135 | 1,402 | 1,331 |
| V2.15 | 622 | 732 | 189 |
| V2.17 | 11,520 | 6,082 | 5,889 |
| V2.18 | 13,017 | 2,115 | 1,769 |
| V2.22 | 14,402 | 6,245 | 6,176 |
| V2.23 | 3,482 | 2,642 | 2,598 |
| V2.26 | 614 | 544 | 222 |

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
