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

| Instance | Provenance | Released checker | Delivered volume | Shift cost | LR |
| --- | --- | ---: | ---: | ---: | ---: |
| V2.12 | **native-repair** | **VALID** | 1,809,906.418 L | 49,246.08 | 0.027209 |
| V2.13 | **native-cold-start** | **VALID** | 210,550.881 L | 13,346.60 | 0.063389 |

* **V2.13 Cold Start**: Generated purely from instance data in 166.8s using adaptive Markov sequence selection, Late Acceptance Hill Climbing, and multi-route block repair. Zero defects across all 10 days. XML hash: `4dfc0ad2b7fc37e0b2ec26350f8c89daf3279aae85ab39cf77814c1ac4c8fedf`.
* **V2.12 Native Repair**: Native repair of a pre-existing candidate (`scratch/v212_skill_orders_final_local.xml`, SHA-256 `32d5905ffd7495fbc37d4ad5b26d2d6dd4a589246a1f93b2f89f925c2a83b2f3`).

## Reproduction

```bash
uv run vrp-solver verify-official \
  roadef_2016_data/set_B/Instances_B_V25-11042016/V2.13.xml \
  scratch/markov_benchmark_results/V2.13.xml
```


```bash
uv run vrp-solver verify-official \
  roadef_2016_data/set_B/Instances_B_V25-11042016/V2.12.xml \
  scratch/v212_skill_orders_final_local.xml
```

Expected publication marker:

```text
official_status,valid
official_valid,True
```

Local simulation and native rule checks remain useful diagnostics, but they do
not confer official validity.
