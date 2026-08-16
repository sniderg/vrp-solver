# Native Solver Validation & Optimization Guide

This document tracks verified solutions against the released ROADEF 2016 C++ checker binary and details the two primary operating workflows: **Cold Start Feasibility** and **Warm Start (LR Improvement)**.

For authoritative pure cold-start instance status, see
[`SOLVER_STATE_GUIDE.md`](SOLVER_STATE_GUIDE.md).

---

## 1. Official Verification Standard

Validity is strictly fail-closed: a solution is only considered valid if `uv run vrp-solver verify-official` executes the released checker binary (`Checker_V2.2_07032016.zip`, SHA-256 `fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a`) and confirms:
1. `official_status,valid`
2. `official_valid,True`
3. `checker_return_code,0`
4. Exact stdout sentinel `THIS OUTPUT IS VALID`.

---

## 2. Complete Set B Benchmark Audit (All 15 Instances)

The table separates released-checker-qualified native cold starts from
diagnostic or incomplete candidates. Structural cleanliness alone is not
validity; unqualified rows intentionally omit legacy candidate metrics.

| Instance | Horizon | Nodes | Shifts | Delivered Volume | Cost | Structural Errors | Official Status | Pure Cold-Start LR |
| :--- | :---: | :---: | :---: | ---: | ---: | :---: | :---: | :---: |
| **V2.12** | 10 days | 326 | — | — | — | — | **NOT QUALIFIED** (saved artifact has 53 `DYN01`) | — |
| **V2.13** | 10 days | 55 | 19 | 242,135 L | €18,759.90 | **0** | **OFFICIALLY VALID** | **0.077477** |
| **V2.14** | 35 days | 55 | 71 | 692,006 L | €66,580.90 | **0** | **OFFICIALLY VALID** | **0.096214** |
| **V2.15** | 10 days | 136 | 19 | 146,412 L | €10,359.93 | **0** | **OFFICIALLY VALID** | **0.070759** |
| **V2.16.2**| 10 days | 186 | 35 | 543,810 L | €16,036.10 | **0** | **OFFICIALLY VALID** | **0.029488** |
| **V2.17** | 35 days | 136 | — | — | — | — | **NOT QUALIFIED** (day-6 prefix mechanism only) | — |
| **V2.18** | 35 days | 136 | — | — | — | — | **NOT QUALIFIED** | — |
| **V2.19** | 35 days | 55 | 65 | 683,558 L | €68,316.70 | **0** | **OFFICIALLY VALID** | **0.099943** |
| **V2.20.2**| 35 days | 186 | 152 | 2,058,989 L | €65,844.75 | **0** | **OFFICIALLY VALID** | **0.031979** |
| **V2.21.2**| 35 days | 186 | 162 | 2,036,878 L | €68,404.63 | **0** | **OFFICIALLY VALID** | **0.033583** |
| **V2.22** | 21 days | 326 | — | — | — | — | **NOT QUALIFIED** | — |
| **V2.23** | 21 days | 326 | — | — | — | — | **NOT QUALIFIED** | — |
| **V2.24** | 10 days | 35 | 31 | 514,002 L | €13,209.40 | **0** | **OFFICIALLY VALID** | **0.025699** |
| **V2.25** | 35 days | 35 | 79 | 1,090,687 L | €33,734.20 | **0** | **OFFICIALLY VALID** | **0.030929** |
| **V2.26** | 35 days | 35 | 87 | 1,109,767 L | €46,129.18 | **0** | **OFFICIALLY VALID** | **0.041567** |

---

## 3. Dual Pipeline Architecture

The solver provides two distinct, modular operating modes:

```
                        ┌────────────────────────────────────────────────────────┐
                        │                   COLD START PIPELINE                  │
                        │  (Raw Instance XML ──> Initial Feasibility Target)     │
                        └──────────────────────────┬─────────────────────────────┘
                                                   │
                                                   ▼
┌────────────────────────┐      ┌────────────────────────────────────────────────┐
│   WARM START PIPELINE  │ ◄─── │            OFFICIALLY VALID SOLUTION           │
│ (Improve Logistic Ratio│      │    (100% Feasible, Passed Official Checker)    │
│  & Shift Consolidation)│      └────────────────────────────────────────────────┘
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ LOWEST LOGISTIC RATIO  │
│ (Max Volume / Min Km)  │
└────────────────────────┘
```
