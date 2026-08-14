# Native Solver Validation & Optimization Guide

This document tracks verified solutions against the released ROADEF 2016 C++ checker binary and details the two primary operating workflows: **Cold Start Feasibility** and **Warm Start (LR Improvement)**.

---

## 1. Official Verification Standard

Validity is strictly fail-closed: a solution is only considered valid if `uv run vrp-solver verify-official` executes the released checker binary (`Checker_V2.2_07032016.zip`, SHA-256 `fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a`) and confirms:
1. `official_status,valid`
2. `official_valid,True`
3. `checker_return_code,0`
4. Exact stdout sentinel `THIS OUTPUT IS VALID`.

---

## 2. Verified Native Solutions (Set B)

All 7 solutions below have been independently re-verified with 0 errors by the official C++ checker binary:

| Instance | Horizon | Nodes | Provenance | Official Checker | Shift Cost | Delivered Volume | Official LR |
| :--- | :---: | :---: | :--- | :---: | ---: | ---: | ---: |
| **V2.13** | 10 days | 55 | `native-cold-start` | **VALID** | 13,346.60 | 210,550.88 L | **0.063389** |
| **V2.24** | 10 days | 35 | `native-cold-start` | **VALID** | 1,234.30 | 61,010.00 L | **0.020231** |
| **V2.25** | 35 days | 35 | `native-cold-start` | **VALID** | 4,499.70 | 125,078.00 L | **0.035975** |
| **V2.26** | 35 days | 35 | `native-cold-start` | **VALID** | 8,975.20 | 124,414.00 L | **0.072139** |
| **V2.15** | 10 days | 136 | `native-cold-start` | **VALID** | 16,367.40 | 233,794.75 L | **0.070007** |
| **V2.16.2**| 10 days | 186 | `native-cold-start` | **VALID** | 18,172.90 | 703,327.91 L | **0.025838** |
| **V2.12** | 21 days | 326 | `native-repair` | **VALID** | 49,246.08 | 1,809,906.42 L | **0.027209** |

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

### Pipeline A: Cold Start (Feasibility First)
* **Goal**: Generate a fully valid solution (`errors=0, deficit=0`) starting from scratch with only instance XML data.
* **Key Components**:
  1. `cluster_construct_solution` / `paper_construct_solution`: Initial topology generation.
  2. `generate_multi_reload_candidates`: Inserts mid-shift source refills to maximize shift payload.
  3. `robust_rolling_rescue` / `surgical_search`: Resolves time-window and stockout deficits.
  4. Feasibility Selector: Configured with `pressure_bonus=10.0`, `shift_penalty=200.0`, and volume delivery rewards.
* **Standard Cold Start Command**:
  ```bash
  uv run vrp-solver robust-rolling-rescue \
    roadef_2016_data/set_B/Instances_B_V25-11042016/V2.14.xml \
    scratch/v214_initial_cluster.xml \
    scratch/v214_feasible.xml \
    --mode deterministic \
    --horizon-days 35 \
    --commit-days 7 \
    --lookahead-days 3 \
    --cg-iterations 4 \
    --multi-reload-columns
  ```

---

### Pipeline B: Warm Start (Cost & Logistic Ratio Optimization)
* **Goal**: Given an already-feasible solution, minimize transportation cost and maximize delivered volume per shift to achieve the lowest possible Logistic Ratio ($\text{LR} = \frac{\text{Cost}}{\text{Volume}}$).
* **Key Components**:
  1. `recombine_route_blocks` & `route_recombination`: Consolidates customer stops across shifts.
  2. `delete_operation` & Shift Pruning: Eliminates redundant routes without inducing stockouts.
  3. Cost-Phase Selector: Uses `shift_penalty=10,000.0` and route density incentives.
  4. Late Acceptance Hill Climbing (`LAHC`, $L=50$) with exact continuous LP rebalancing.
* **Standard Warm Start / LR Improvement Command**:
  ```bash
  uv run vrp-solver surgical-search \
    roadef_2016_data/set_B/Instances_B_V25-11042016/V2.13.xml \
    scratch/markov_benchmark_results/V2.13.xml \
    scratch/v213_lower_lr.xml \
    --end-day 10 \
    --time-limit 180 \
    --use-lahc \
    --use-markov \
    --candidates-per-move 100
  ```

---

## 4. Verification Command

To verify any output XML with the official C++ checker:
```bash
uv run vrp-solver verify-official <instance_xml> <solution_xml>
```
