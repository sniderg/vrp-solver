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

## 2. Complete Set B Benchmark Audit (All 15 Instances)

Every instance in Set B has been generated and validated with zero structural violations (`DRI01=0, DRI02=0, DRI03=0, TL01=0, TL02=0, LAY01=0, LAY02=0, SHI01=0, SHI02=0, SHI03=0, SHI04=0, SHI05=0, SHI06=0, OPE01=0, OPE02=0, OPE03=0, OPE04=0, OPE05=0`):

| Instance | Horizon | Nodes | Shifts | Delivered Volume | Cost | Structural Errors | Official Status | Base Logistic Ratio (LR) |
| :--- | :---: | :---: | :---: | ---: | ---: | :---: | :---: | :---: |
| **V2.12** | 10 days | 326 | 85 | 2,431,172 L | €44,966.73 | **0** | **OFFICIALLY VALID** | **0.018496** |
| **V2.13** | 10 days | 55 | 18 | 210,551 L | €13,346.60 | **0** | **OFFICIALLY VALID** | **0.063389** |
| **V2.14** | 35 days | 55 | 51 | 600,484 L | €50,737.60 | **0** | 100% Structurally Clean | `0.084494` |
| **V2.15** | 10 days | 136 | 21 | 246,340 L | €12,692.00 | **0** | **OFFICIALLY VALID** | **0.051522** |
| **V2.16.2**| 10 days | 186 | 50 | 861,199 L | €18,878.67 | **0** | **OFFICIALLY VALID** | **0.021921** |
| **V2.17** | 35 days | 136 | 82 | 749,560 L | €44,246.07 | **0** | 100% Structurally Clean | `0.059029` |
| **V2.18** | 35 days | 136 | 69 | 652,338 L | €41,292.20 | **0** | 100% Structurally Clean | `0.063299` |
| **V2.19** | 35 days | 55 | 77 | 686,964 L | €77,036.90 | **0** | 100% Structurally Clean | `0.112141` |
| **V2.20.2**| 35 days | 186 | 159 | 2,082,728 L | €66,249.40 | **0** | 100% Structurally Clean | `0.031809` |
| **V2.21.2**| 35 days | 186 | 160 | 2,057,095 L | €69,420.53 | **0** | 100% Structurally Clean | `0.033747` |
| **V2.22** | 21 days | 326 | 133 | 3,997,700 L | €120,674.02 | **0** | 100% Structurally Clean | `0.030186` |
| **V2.23** | 21 days | 326 | 157 | 4,503,192 L | €112,163.39 | **0** | 100% Structurally Clean | `0.024908` |
| **V2.24** | 10 days | 35 | 33 | 617,438 L | €10,263.42 | **0** | **OFFICIALLY VALID** | **0.016623** |
| **V2.25** | 35 days | 35 | 70 | 1,230,180 L | €20,951.23 | **0** | **OFFICIALLY VALID** | **0.017031** |
| **V2.26** | 35 days | 35 | 160 | 1,155,875 L | €92,646.04 | **0** | **OFFICIALLY VALID** | **0.080152** |


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
