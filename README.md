# VRP Solver

A standalone, Cython-optimized combinatorial optimization toolkit for the Inventory Routing Problem (IRP), based on the ROADEF 2016 challenge. The project combines MILP-based route selection, column-generation-style rescue, rolling robust planning, ALNS/local search, fast inventory simulation, and optional ML-guided route priorities.

> **Validation correction (2026-08-05):** the historical claim that all 20
> tracked native Set B/Set X outputs were officially valid has been withdrawn.
> A fail-closed run of the released checker rejects all 20 historical XMLs.
> See [NATIVE_BENCHMARK_RESULTS.md](NATIVE_BENCHMARK_RESULTS.md) for the audited
> status. Only outputs with `official_valid,True` may be described as valid.

## Features

- **MILP route selection**: Selects compatible shifts under driver, trailer, timing, inventory, and order constraints using HiGHS or Gurobi.
- **Column-generation-style rescue**: Generates priced route candidates around pressure customers, then repeatedly selects and repairs routes to remove stockouts and capacity violations.
- **Rolling robust planning**: Supports deterministic, hedged, and robust rolling-horizon rescue using forecast scenarios, quantiles, and committed-window validation.
- **ALNS and local search**: Includes destroy/repair ALNS, route swaps, route pruning, source cleanup, quantity trimming, and benchmark-specific polishing scripts.
- **Recovered surgical search**: Implements the seven transactional neighborhoods verified by static analysis of the legacy Windows solver: create shift, insert/delete/replace operation, swap points, and inter-/intra-shift relocation.
- **ML-guided priorities**: Provides hooks for ML route/customer priors that influence candidate generation and MILP objective prizes.
- **Fast simulation and validation**: Uses Cython inventory simulation where available, plus local rule checks and optional bundled official-checker validation.
- **Benchmark tooling**: Includes Set A V1 comparison/polishing scripts and a tutorial notebook for comparing against Hexaly V1 scores.
- **Dual solver support**: Runs out of the box with the free/open-source **HiGHS** solver (`highspy`) and has native support for **Gurobi** (`gurobipy`) if available.

## Installation

### Prerequisites
You need a C compiler installed on your system (e.g. GCC/Clang on macOS/Linux, MSVC on Windows) to compile the Cython extension.

### Standard Install
Install the package in editable mode or from source:
```bash
pip install -e .
```

### With Gurobi Support
If you have a Gurobi license, you can install the optional Gurobi bindings:
```bash
pip install -e ".[gurobi]"
```
This extra also installs Numba, which is required by the native Gurobi solver.

## Run a native cold-start solve (start here)

A *native cold start* takes **only an instance XML plus an integer seed** and
produces a solution. It never reads an existing solution, a reference XML, or
`Solver.exe` output. This is the project's headline capability and the workflow
to use unless you were asked for something else. See
[Provenance rules](#provenance-rules) for why that boundary matters.

### Prerequisites

1. A Python environment with the package installed (`pip install -e .`). On
   Windows use the venv interpreter explicitly, e.g. `.venv/Scripts/python.exe`.
2. `roadef_2016_data/Checker_V2.2_07032016.zip` must exist for verification.
   Its SHA-256 must be
   `fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a`.
   Verification runs the `.exe` natively on Windows and under Mono elsewhere.

### Two commands

```bash
# 1) Solve. Instance XML in, solution XML out. Nothing else is read.
vrp-solver native-solve \
  roadef_2016_data/set_B/Instances_B_V25-11042016/V2.14.xml \
  out/V2.14_native.xml \
  --seed 1 \
  --time-limit 1200 \
  --no-improvement-limit 10000 \
  --restart-rounds 1

# 2) Verify with the released checker. This, not the local check, decides validity.
vrp-solver verify-official \
  roadef_2016_data/set_B/Instances_B_V25-11042016/V2.14.xml \
  out/V2.14_native.xml
```

If `vrp-solver` is not on `PATH`, substitute
`.venv/Scripts/python.exe -m vrp_solver.cli` (Windows) or
`uv run vrp-solver`. Add `-u` to the `python -m` form when you need unbuffered
progress output, otherwise a redirected log stays empty until the run ends.

The V2.14 command above is the known-good reference run: it finishes in about
60 s and yields `official_valid,True` with logistic ratio 0.084934.

### How to read the output

`native-solve` prints CSV-style `key,value` lines. The ones that matter:

| Line | Meaning |
| --- | --- |
| `seed_candidate,idle_cap,<cap>,errors,<n>,...` | One constructed seed per idle-cap setting; the best is kept |
| `native_round,<i>,errors,<n>,...` | Search round progress |
| `local_errors,<n>` | **0 is necessary but not sufficient** — it does not confer validity |
| `safety_deficit_qm,<n>` | Safety-breach quantity-minutes; use this to see progress when the error count plateaus |
| `wall_time_seconds` | Runtime |

`verify-official` is the only authority on validity:

| Line | Meaning |
| --- | --- |
| `official_valid,True` | The checker accepted this exact XML. Only now may you call it valid |
| `official_status,valid` | Same verdict in words |
| `official_logistic_ratio,<x>` | The objective. Comparable **only among valid solutions** |

A run that prints `local_errors,0` but fails `verify-official` is not a
solution. Never report success from the local check alone.

### Useful flags

| Flag | Default | When to change it |
| --- | --- | --- |
| `--seed` | 0 | Vary across parallel runs; seeds differ substantially in outcome |
| `--time-limit` | 300 | Official budget is 1800 s (30 min) |
| `--restart-rounds` | 2 | Use `1` with a high `--no-improvement-limit`; restart rounds discard progress |
| `--no-improvement-limit` | 24 | Set to `10000` for long single-round searches |
| `--candidates-per-move` | 120 | 120 beats 32 decisively; lower only to save time |
| `--idle-cap` | both tried | Caps mid-route idle waiting. Omit to build both an uncapped and a 180-minute seed and keep the better; pass `0` to force uncapped |

`--idle-cap` is the highest-leverage construction control. Capping mid-route
idle time stops a route from holding a driver and trailer while it waits for a
fuller delivery, which starves other tanks. It transforms some instances
(V2.14: 1,884 seed errors to 0) and hurts a few that already close uncapped, so
the default builds both and keeps the better seed.

### Solve a whole corpus in one command

`native-solve-batch` runs the identical `native-solve` pipeline once per
instance, each in its own process, then submits every output to the released
checker and prints a summary. No wrapper script and no per-instance babysitting.

```bash
vrp-solver native-solve-batch \
  roadef_2016_data/set_B/Instances_B_V25-11042016 \
  out/setB \
  --seed 1 --time-limit 1800 --concurrency 7 \
  --summary-csv out/setB/summary.csv
```

It writes `<instance>_native.xml` and `<instance>.log` per instance into the
output directory, skips `*_solution`/`*_rescued`/`*_best` files so reference XMLs
can never be mistaken for instances, and exits non-zero unless every instance is
officially valid. Restrict the set with `--only V2.14 V2.16.2`. Final lines:

```text
summary_valid,2,summary_total,2
summary,V2.14,valid,0.084934
summary,V2.16.2,valid,0.042634
```

HiGHS carries no licence restriction, so concurrency is limited only by cores;
on a 16-core machine 7–8 concurrent solves is comfortable. In-run `--workers`
parallelism helps little because candidate generation dominates — spend cores on
a portfolio of instances and seeds instead.

### If a run does not reach zero errors

1. Re-run with different `--seed` values before changing any code.
2. Watch `safety_deficit_qm` rather than the error count; it moves first.
3. Continue from the checkpoint rather than starting over. Rotating the first
   operator across rounds is what closed V2.25 (156 to 24 to 18 to 0 errors);
   `native-solve` does that internally across `--restart-rounds`, and
   `--resume-from` picks up an earlier native output:

   ```bash
   vrp-solver native-solve <instance.xml> out/next.xml \
     --resume-from out/previous_native.xml \
     --seed 5 --time-limit 1800 --restart-rounds 6 --no-improvement-limit 10000
   ```

   Only ever pass a checkpoint this solver produced. Passing a reference or
   oracle XML makes the result `native-repair` at best, not a cold start.
4. Before theorising about the constructor, measure. See
   [skills/solve-roadef-irp/SKILL.md](skills/solve-roadef-irp/SKILL.md) for the
   resource-time accounting procedure and previously refuted hypotheses.

Current per-instance status, artifacts, and hashes:
[NATIVE_BENCHMARK_RESULTS.md](NATIVE_BENCHMARK_RESULTS.md).

## Provenance rules

Label every output as exactly one of:

- **`native-cold-start`** — produced from an instance XML plus a seed only.
- **`native-repair`** — a native run that started from an existing solution XML.
- **`oracle`** — produced by, or derived from, `Solver.exe`.
- **`reference`** — a supplied third-party solution.

Reference and oracle XMLs may be read for diagnosis and aggregate comparison.
They must never be given to a solver as input, a seed, or a warm start on a run
you intend to describe as a cold start. `native-solve` enforces this
structurally by accepting no solution argument, so prefer it over hand-assembled
pipelines when provenance matters.

## Other solver entry points

Both wrappers prepare the project through `uv`, write a pending XML, and only
publish it after the released ROADEF V2 checker accepts it. They require bash.

```bash
# Native cold start with checker gating (solve.sh is an alias for this).
# Tunable through NATIVE_SEED, NATIVE_RESTART_ROUNDS, NATIVE_ITERATIONS,
# NATIVE_CANDIDATES_PER_MOVE, and NATIVE_WORKERS environment variables.
./solve_native.sh <instance.xml> <native-output.xml> [timeout-seconds]

# Original Windows Solver.exe under Wine, for oracle/comparison experiments.
# Produces `oracle` provenance, never a cold start.
# Extra arguments are seed, iterations (0 = constructor-only), and workers.
./solve_oracle.sh <instance.xml> <oracle-output.xml> [timeout-seconds] [seed] [iterations] [workers]
```

`solve_oracle.sh` requires the prepared Wine/Gurobi compatibility runtime in
`/private/tmp/solver-oracle`; set `WINEPREFIX` to use a different Wine prefix.

## Command Line Interface (CLI)

The package installs a command-line utility `vrp-solver`:

```bash
# Get help
vrp-solver --help

# Run targeted rescue on an instance and solution
vrp-solver targeted-rescue \
  --instance-xml /path/to/instance.xml \
  --solution-xml /path/to/solution.xml \
  --output-xml /path/to/output.xml
```

Common rescue and benchmark workflows:

```bash
# Run column-generation-style rescue from an existing solution
vrp-solver column-generation-rescue \
  --instance-xml /path/to/instance.xml \
  --solution-xml /path/to/seed.xml \
  --output-xml /path/to/rescued.xml \
  --iterations 5

# Run rolling robust rescue over a planning horizon
vrp-solver robust-rolling-rescue \
  --instance-xml /path/to/instance.xml \
  --solution-xml /path/to/seed.xml \
  --output-xml /path/to/rolling.xml \
  --mode hedged \
  --horizon-days 14

# Compare selected Set A V1 solutions against Hexaly V1 benchmarks
uv run python scripts/compare_a_v1.py

# Polish a Set A V1 benchmark solution without changing solver defaults
uv run python scripts/improve_a_v1.py --instance V_1.11

# Build a Set B V2 seed from scratch, separate from benchmark polishing
uv run python scripts/build_b_v2.py --instance V2.12 --no-official

# Improve a constructed seed with the recovered seven-neighborhood search
vrp-solver surgical-search \
  /path/to/V2.12-instance.xml \
  /path/to/constructed-seed.xml \
  /path/to/improved.xml \
  --end-day 10 \
  --iterations 500 \
  --workers 6
```

The surgical-search trace reports feasibility errors, hard violations,
inventory deficit, estimated cost, and logistic ratio (`lr`) as incumbents
move. Candidate mutations are evaluated transactionally and only committed
after the affected shifts have been retimed and the complete solution has
been rescored.

The reconstructed controller follows the legacy binary's verified structure:
seven surgical operators, adaptive per-operator rewards, recency-aware
selection, escalating perturbation, and incumbent preservation. Gurobi is
optional and is not called by these neighborhoods.

## Programmatic API

You can import and use the solver programmatically:

```python
from pathlib import Path
from vrp_solver.xml_io import load_instance, load_solution, save_solution
from vrp_solver.solver.column_loop import column_generation_rescue, ColumnLoopConfig

# Load data
instance = load_instance(Path("instance.xml"))
solution = load_solution(Path("solution.xml"))

# Configure
config = ColumnLoopConfig(
    start_day=0,
    end_day=14,
    iterations=3,
    quantity_objective="max-delivered"
)

# Run solver
rescued_sol, steps = column_generation_rescue(instance, solution, config=config)

# Save
save_solution(rescued_sol, Path("rescued_solution.xml"))
```

Rolling robust rescue:

```python
from vrp_solver.solver.rolling_cg import robust_rolling_rescue, RollingCGConfig

config = RollingCGConfig(
    mode="hedged",
    horizon_days=14,
    commit_days=7,
    lookahead_days=7,
    n_scenarios=20,
)

rolling_solution, diagnostics = robust_rolling_rescue(instance, solution, config=config)
```

ML priors can be passed into the column loop when trained route/customer priority signals are available:

```python
from vrp_solver.solver.ml_priors import MLRoutePriors

ml_priors = MLRoutePriors()
ml_priors.load("route_priors.json")

rescued_sol, steps = column_generation_rescue(
    instance,
    solution,
    config=config,
    ml_priors=ml_priors,
)
```

## Benchmark Notes

The scripts in `scripts/` are benchmark tooling, not default solver behavior:

- `compare_a_v1.py` compares Set A V1 XML artifacts against `roadef_2016_data/hexaly_a_benchmarks.csv`.
- `improve_a_v1.py` starts from an existing feasible XML seed and applies official-checker-gated polishing moves.
- `build_b_v2.py` builds Set B V2 seeds from scratch, then optionally runs column-generation rescue with local and official-checker gating.
- `notebooks/a_v1_benchmark_tutorial.ipynb` walks through comparison, polishing, and resume workflows.

Keep historical V1 tuning separate from modern B/X or robust-rolling solver work because V1 uses different objective weights.

## License
MIT
