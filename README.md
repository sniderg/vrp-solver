# VRP Solver

A standalone, Cython-optimized combinatorial optimization toolkit for the Inventory Routing Problem (IRP), based on the ROADEF 2016 challenge. The project combines MILP-based route selection, column-generation-style rescue, rolling robust planning, ALNS/local search, fast inventory simulation, and optional ML-guided route priorities.

## Features

- **MILP route selection**: Selects compatible shifts under driver, trailer, timing, inventory, and order constraints using HiGHS or Gurobi.
- **Column-generation-style rescue**: Generates priced route candidates around pressure customers, then repeatedly selects and repairs routes to remove stockouts and capacity violations.
- **Rolling robust planning**: Supports deterministic, hedged, and robust rolling-horizon rescue using forecast scenarios, quantiles, and committed-window validation.
- **ALNS and local search**: Includes destroy/repair ALNS, route swaps, route pruning, source cleanup, quantity trimming, and benchmark-specific polishing scripts.
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
```

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
