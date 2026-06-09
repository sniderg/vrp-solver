# VRP Solver

A standalone, Cython-optimized column generation and Adaptive Large Neighborhood Search (ALNS) solver for the Inventory Routing Problem (IRP), based on the ROADEF 2016 challenge.

## Features
- **Column Generation (CG)**: Solves the vehicle routing and timing selection problems using MILP (HiGHS or Gurobi).
- **ALNS Heuristics**: Neighborhood search with robust destroy and repair operators.
- **Targeted Rescue Heuristics**: Solves inventory stockout and capacity violations.
- **Cython Simulation**: High-speed, Cython-optimized inventory and movement checkers.
- **Dual Solver Support**: Runs out of the box with the free/open-source **HiGHS** solver (`highspy`) and has native support for **Gurobi** (`gurobipy`) if available.

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

## License
MIT
