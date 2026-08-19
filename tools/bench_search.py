"""Search-loop benchmark: the step 2 gate, and the step 4 ablation harness.

The step 2 gate ("on V2.15, zero steps with an empty neighbourhood over a 60 s
run, and >= 20% of steps accepted") is a property of the *neighbourhood*, so it
could only be measured once step 3's operator portfolio existed.  This is where
it gets measured.

    .venv/Scripts/python.exe tools/bench_search.py --instances V2.15 --limit 60
    .venv/Scripts/python.exe tools/bench_search.py --limit 20 --operator-table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrp_solver.fast.search import (  # noqa: E402
    RouletteSelector,
    UniformSelector,
    run_search,
)

SELECTORS = {"uniform": UniformSelector, "roulette": RouletteSelector}
from vrp_solver.fast.state import SearchState, instance_days  # noqa: E402
from vrp_solver.xml_io import load_instance, load_solution  # noqa: E402

from bench_moves import INSTANCE_DIR, SEEDS  # noqa: E402


def _instance_path(name: str) -> Path:
    return INSTANCE_DIR / f"{name}.xml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="*", default=sorted(SEEDS))
    parser.add_argument("--limit", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--operator-table", action="store_true")
    parser.add_argument(
        "--selector", choices=sorted(SELECTORS), default="uniform",
        help="which step 4 controller to drive the run with",
    )
    args = parser.parse_args(argv)

    print(
        f"{'instance':9s} {'steps':>8s} {'steps/s':>9s} {'accept%':>8s} "
        f"{'infeas%':>8s} {'empty':>6s} {'errors':>7s} {'LR':>10s}"
    )
    worst_accept = 1.0
    total_empty = 0
    for name in args.instances:
        seed_path = SEEDS.get(name)
        if seed_path is None or not seed_path.exists():
            print(f"{name:9s} (no seed solution)")
            continue
        instance = load_instance(_instance_path(name))
        solution = load_solution(seed_path)
        days = instance_days(instance)
        state = SearchState.from_solution(instance, solution, score_days=days)

        result = run_search(
            state, limit=args.limit, seed=args.seed,
            selector=SELECTORS[args.selector](),
        )
        t = result.telemetry
        worst_accept = min(worst_accept, t.accepted_fraction)
        total_empty += t.empty_neighbourhood
        print(
            f"{name:9s} {t.steps:8d} {result.steps_per_second:9.1f} "
            f"{t.accepted_fraction:8.1%} {t.infeasible_fraction:8.1%} "
            f"{t.empty_neighbourhood:6d} "
            f"{result.best_score.feasibility_errors:7d} "
            f"{result.best_score.logistic_ratio:10.6f}"
        )
        if args.operator_table:
            print(result.operator_table())
            print()

    print()
    print(
        f"step 2 gate: empty neighbourhoods total {total_empty} (target 0), "
        f"worst accept fraction {worst_accept:.1%} (target >= 20%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
