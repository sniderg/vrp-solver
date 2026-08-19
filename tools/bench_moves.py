"""Move-evaluation throughput benchmark for the fast search substrate.

Step 1.8 of ``REBUILD_PLAN.md``.  Reports evaluations/sec for the apply -> score
-> revert cycle that a local search actually performs, next to the legacy
full-rescore rate on the same instance, so the speedup is measured rather than
asserted.

    .venv/Scripts/python.exe tools/bench_moves.py
    .venv/Scripts/python.exe tools/bench_moves.py --instances V2.15 V2.22
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrp_solver.contest import score_prefix_with_feasibility_tail  # noqa: E402
from vrp_solver.fast.state import SearchState, instance_days  # noqa: E402
from vrp_solver.xml_io import load_instance, load_solution  # noqa: E402

INSTANCE_DIR = Path("roadef_2016_data/set_B/Instances_B_V25-11042016")

# Seeds to benchmark against: instance name -> an existing solution to load.
SEEDS = {
    "V2.13": Path("scratch/replicate_V2.13_native.xml"),
    "V2.14": Path("scratch/cold_V2.14_cadence.xml"),
    "V2.15": Path("scratch/V2.15_compact_full_master.xml"),
    "V2.16.2": Path("scratch/cold_V2.16.2_batch.xml"),
    "V2.19": Path("scratch/opt_V2.19_native.xml"),
    "V2.20.2": Path("scratch/opt_V2.20.2_native.xml"),
    "V2.21.2": Path("scratch/opt_V2.21.2_native.xml"),
    "V2.22": Path("scratch/best_V2.22_native.xml"),
    "V2.24": Path("scratch/replicate_V2.24_native.xml"),
    "V2.25": Path("scratch/opt3_V2.25_native.xml"),
}


def _reverse_block_move(state: SearchState, rng: random.Random):
    """The canonical cheap local-search move: reverse a block in one route.

    Chosen because it is representative (touches one shift, a handful of
    customers) and always available, so the benchmark measures the substrate
    instead of a candidate-generation failure.
    """
    candidates = [i for i, rec in enumerate(state.shifts) if len(rec.points) >= 2]
    if not candidates:
        return None
    position = rng.choice(candidates)
    rec = state.shifts[position]
    i = rng.randrange(len(rec.points) - 1)
    j = rng.randrange(i + 1, len(rec.points))
    points = list(rec.points)
    points[i : j + 1] = list(reversed(points[i : j + 1]))
    return position, points


def bench_fast(state: SearchState, *, seconds: float, seed: int) -> tuple[int, float]:
    rng = random.Random(seed)
    evaluations = 0
    deadline = time.perf_counter() + seconds
    while True:
        move = _reverse_block_move(state, rng)
        if move is None:
            break
        position, points = move
        rec = state.shifts[position]
        state.begin()
        state.set_operations(position, points, rec.arrivals, rec.quantities)
        state.score()
        state.rollback()
        evaluations += 1
        if evaluations % 64 == 0 and time.perf_counter() >= deadline:
            break
    elapsed = seconds if evaluations else 0.0
    return evaluations, elapsed


def bench_legacy(instance, solution, days: int, *, seconds: float) -> tuple[int, float]:
    evaluations = 0
    start = time.perf_counter()
    deadline = start + seconds
    while True:
        score_prefix_with_feasibility_tail(
            instance, solution, score_days=days, feasibility_days=days,
            ignore_tail_call_ins=True,
        )
        evaluations += 1
        if time.perf_counter() >= deadline:
            break
    return evaluations, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()

    names = args.instances or list(SEEDS)
    print(
        f"{'instance':10s} {'cust':>5s} {'shifts':>7s} "
        f"{'fast/s':>10s} {'legacy/s':>9s} {'speedup':>8s}"
    )
    rows = []
    for name in names:
        instance_path = INSTANCE_DIR / f"{name}.xml"
        solution_path = SEEDS.get(name)
        if not instance_path.exists() or solution_path is None or not solution_path.exists():
            print(f"{name:10s} (missing instance or seed solution; skipped)")
            continue

        instance = load_instance(instance_path)
        solution = load_solution(solution_path)
        days = instance_days(instance)
        state = SearchState.from_solution(instance, solution, score_days=days)

        fast_evals, fast_elapsed = bench_fast(
            state, seconds=args.seconds, seed=args.seed
        )
        fast_rate = fast_evals / fast_elapsed if fast_elapsed else 0.0

        if args.skip_legacy:
            legacy_rate = float("nan")
            speedup = float("nan")
        else:
            legacy_evals, legacy_elapsed = bench_legacy(
                instance, solution, days, seconds=min(args.seconds, 2.0)
            )
            legacy_rate = legacy_evals / legacy_elapsed if legacy_elapsed else 0.0
            speedup = fast_rate / legacy_rate if legacy_rate else float("inf")

        print(
            f"{name:10s} {len(instance.customers):5d} {len(state.shifts):7d} "
            f"{fast_rate:10.0f} {legacy_rate:9.0f} {speedup:7.0f}x"
        )
        rows.append((name, fast_rate))

    if rows:
        worst_name, worst_rate = min(rows, key=lambda row: row[1])
        print(f"\nslowest: {worst_name} at {worst_rate:.0f} evals/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
