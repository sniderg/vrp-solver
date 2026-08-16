#!/usr/bin/env python3
"""Run the unified native cold-start solver over every Set B instance."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from vrp_solver.diagnostics import violation_vector
from vrp_solver.official_verify import default_v2_archive, verify_v2_solution
from vrp_solver.rules import validate_solution
from vrp_solver.solver.unified_engine import solve_cold_start
from vrp_solver.xml_io import load_instance, save_solution


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_DIR = ROOT / "roadef_2016_data/set_B/Instances_B_V25-11042016"
OUTPUT_DIR = Path("/private/tmp/vrp-unified-set-b-audit")
PRIORITY = ("V2.13", "V2.14", "V2.24", "V2.25", "V2.26")


def audit_instance(instance_path_text: str, time_limit: float) -> dict[str, object]:
    instance_path = Path(instance_path_text)
    started = time.monotonic()
    instance = load_instance(instance_path)
    result = solve_cold_start(
        instance,
        num_seeds=10,
        time_limit=time_limit,
        frontier_size=3,
        search_iterations=64,
        search_workers=1,
        stop_when_feasible=True,
    )
    output_path = OUTPUT_DIR / instance_path.name
    save_solution(result.solution, output_path)
    violations = validate_solution(instance, result.solution)
    errors = [item for item in violations if item.severity == "error"]
    vector = violation_vector(instance, result.solution)
    official = None
    if not errors:
        official = verify_v2_solution(
            instance_path,
            output_path,
            checker_archive=default_v2_archive(ROOT),
        )
    return {
        "instance": instance_path.stem,
        "provenance": result.provenance,
        "time_limit_seconds": time_limit,
        "wall_seconds": round(time.monotonic() - started, 3),
        "solver_seconds": result.runtime_seconds,
        "shifts": result.shifts,
        "operations": sum(len(shift.operations) for shift in result.solution.shifts),
        "seeds_constructed": result.seeds_constructed,
        "unique_constructed": result.unique_constructed,
        "quantity_repairs": result.quantity_repairs,
        "search_steps": result.search_steps,
        "unscheduled_count": result.unscheduled_count,
        "local_error_count": len(errors),
        "local_error_codes": dict(sorted(Counter(item.code for item in errors).items())),
        "violation_vector": vector.flat(),
        "official_status": official.status if official else "not_run_local_errors",
        "official_valid": official.valid if official else False,
        "official_lr": official.logistic_ratio if official else None,
        "checker_sha256": official.checker_sha256 if official else None,
        "output": str(output_path),
    }


def main() -> None:
    time_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 1_800.0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    priority_rank = {name: rank for rank, name in enumerate(PRIORITY)}
    instance_paths = sorted(
        INSTANCE_DIR.glob("*.xml"),
        key=lambda path: (
            priority_rank.get(path.stem, len(PRIORITY)),
            path.stem,
        ),
    )
    paths = [str(path) for path in instance_paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(audit_instance, path, time_limit): path
            for path in paths
        }
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "instance": Path(futures[future]).stem,
                    "official_valid": False,
                    "official_status": "audit_failed",
                    "error": repr(exc),
                }
            results.append(row)
            results.sort(key=lambda item: str(item["instance"]))
            (OUTPUT_DIR / "results.json").write_text(
                json.dumps(results, indent=2) + "\n",
            )
            print(json.dumps(row), flush=True)
    print(f"results={OUTPUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
