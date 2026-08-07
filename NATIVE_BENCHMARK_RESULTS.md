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

| Instance | Provenance | Released checker | Wall time | Shift cost | LR |
| --- | --- | ---: | ---: | ---: | ---: |
| V2.13 | **native-cold-start** | **VALID** | 345 s | 15,895.40 | — |
| V2.24 | **native-cold-start** | **VALID** | 789 s | 14,384.82 | — |
| V2.25 | **native-cold-start** | **VALID** | ~80 min total | 36,994.19 | 0.035982 |
| V2.12 | native-repair | VALID | — | 49,246.08 | 0.027209 |

All three cold starts were produced on 2026-08-07 by `vrp-solver native-solve`
(instance XML + seed only; V2.25 additionally continued through
`surgical_search` restart rounds from its own native checkpoint) and verified
with the released checker on Windows. Exact XMLs:

- V2.13: `scratch/replicate_V2.13_native.xml`
- V2.24: `scratch/replicate_V2.24_native.xml`
- V2.25: `scratch/opt3_V2.25_native.xml`, SHA-256
  `a407cde5d2f0ddc1ba34a0471328ac7cd374067c48b046aee07b9b2f1c4bedc7`

The V2.12 row is a native repair of a pre-existing candidate
(`scratch/v212_skill_orders_final_local.xml`, SHA-256
`32d5905ffd7495fbc37d4ad5b26d2d6dd4a589246a1f93b2f89f925c2a83b2f3`); a V2.12
cold start has not yet reached zero errors.

For comparison, the supplied V2.12 reference is also officially valid with
2,431,172.363 L delivered, shift cost 44,966.73, and LR 0.018496.

## Reproduction

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
