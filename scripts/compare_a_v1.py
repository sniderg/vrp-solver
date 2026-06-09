#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrp_solver.contest import score_prefix_with_feasibility_tail
from vrp_solver.evaluate import run_checker
from vrp_solver.xml_io import load_instance, load_solution


DATA_DIR = PROJECT_ROOT / "roadef_2016_data"
RESULTS_DIR = DATA_DIR / "hust_smart_results"
INSTANCES_DIR = DATA_DIR / "set_A_v1_1" / "Instances V1.1"
HEXALY_CSV = DATA_DIR / "hexaly_a_benchmarks.csv"
V1_CHECKER = (
    DATA_DIR
    / "checker_v1_1"
    / "Checker V1 v1.1.0.0"
    / "Challenge_Roadef_EURO_Checker_V1"
    / "bin"
    / "Release"
    / "IRP_Roadef_Challenge_Checker.exe"
)

SELECTED_SOLUTIONS = {
    "V_1.1": "v1_1.1_cached_expand3_pruned_maxfill.xml",
    "V_1.2": "v1_1.2_improved.xml",
    "V_1.3": "v1_1.3_improved_squeezed.xml",
    "V_1.4": "v1_1.4_official_greedy.xml",
    "V_1.5": "v1_1.5_improved_squeezed.xml",
    "V_1.6": "v1_1.6_improved_squeezed.xml",
    "V_1.7": "v1_1.7_improved_squeezed.xml",
    "V_1.8": "v1_1.8_improved_squeezed.xml",
    "V_1.9": "v1_1.9_rescued_feasible.xml",
    "V_1.10": "v1_1.10_official_greedy.xml",
    "V_1.11": "v1_1.11_rescued.xml",
}


@dataclass(frozen=True)
class CandidateScore:
    instance: str
    solution_path: Path
    local_feasible: bool
    local_errors: int
    local_ratio: float
    official_valid: bool | None
    official_ratio: float | None
    official_first_rule: str

    @property
    def benchmark_ratio(self) -> float | None:
        return self.official_ratio if self.official_valid and self.official_ratio is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare feasible Set A V1 solution artifacts against Hexaly V1 scores."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_DIR / "a_v1_hexaly_comparison_generated.csv",
        help="Comparison CSV to write.",
    )
    parser.add_argument(
        "--scan-candidates",
        action="store_true",
        help="Scan all matching A V1 result XMLs and pick the best feasible solution per instance.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to scan when --scan-candidates is enabled.",
    )
    parser.add_argument(
        "--no-official",
        action="store_true",
        help="Skip the bundled official V1 checker and rank by local ratio only.",
    )
    args = parser.parse_args()

    use_official = not args.no_official and shutil.which("mono") is not None and V1_CHECKER.exists()
    hexaly = _load_hexaly(HEXALY_CSV)
    rows = []
    for instance_name in sorted(hexaly, key=_instance_sort_key):
        candidates = (
            _scan_candidates(args.candidate_dir, instance_name)
            if args.scan_candidates
            else [RESULTS_DIR / SELECTED_SOLUTIONS[instance_name]]
        )
        scored = []
        for path in candidates:
            if not path.exists():
                continue
            try:
                scored.append(score_candidate(instance_name, path, use_official=use_official))
            except Exception as exc:
                print(f"skipped,{instance_name},{path},{exc}", file=sys.stderr)
        best = _best_score(scored, use_official=use_official)
        rows.append(_comparison_row(instance_name, best, hexaly[instance_name], use_official))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.output_csv)
    _print_summary(rows, args.output_csv, use_official)
    return 0


def score_candidate(instance_name: str, solution_path: Path, *, use_official: bool) -> CandidateScore:
    instance_path = _instance_path(instance_name)
    instance = load_instance(instance_path)
    solution = load_solution(solution_path)
    days = (instance.horizon * instance.unit + 1439) // 1440
    score = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=days,
        feasibility_days=days,
        ignore_tail_call_ins=True,
    )
    local_ratio = score.scored_estimated_cost / max(1.0, score.scored_delivered_quantity)
    official_valid = None
    official_ratio = None
    official_first_rule = ""
    if use_official:
        try:
            official_valid, official_ratio, official_first_rule = run_checker(
                instance_path,
                solution_path,
                V1_CHECKER,
            )
        except RuntimeError as exc:
            official_first_rule = str(exc)
    return CandidateScore(
        instance=instance_name,
        solution_path=solution_path,
        local_feasible=score.feasible,
        local_errors=score.feasibility_errors,
        local_ratio=local_ratio,
        official_valid=official_valid,
        official_ratio=official_ratio,
        official_first_rule=official_first_rule,
    )


def _best_score(scores: list[CandidateScore], *, use_official: bool) -> CandidateScore | None:
    feasible = [
        score
        for score in scores
        if score.local_feasible and (not use_official or score.official_valid is True)
    ]
    if not feasible:
        return None
    if use_official:
        return min(feasible, key=lambda score: score.official_ratio or float("inf"))
    return min(feasible, key=lambda score: score.local_ratio)


def _comparison_row(
    instance_name: str,
    score: CandidateScore | None,
    hexaly_row: dict[str, str],
    use_official: bool,
) -> dict[str, object]:
    hexaly_ratio = float(hexaly_row["hexaly"])
    ratio = (
        score.official_ratio
        if use_official and score is not None and score.official_ratio is not None
        else score.local_ratio
        if score is not None
        else None
    )
    gap = ((ratio / hexaly_ratio) - 1.0) * 100.0 if ratio is not None else None
    return {
        "instance": instance_name,
        "solution": score.solution_path.name if score else "",
        "ratio_source": "official_v1" if use_official else "local",
        "our_ratio": f"{ratio:.6f}" if ratio is not None else "",
        "hexaly_v1_ratio": f"{hexaly_ratio:.6f}",
        "gap_vs_hexaly_pct": f"{gap:.1f}" if gap is not None else "",
        "status": _status(gap),
        "local_feasible": score.local_feasible if score else False,
        "local_errors": score.local_errors if score else "",
        "official_valid": score.official_valid if score else "",
        "official_first_rule": score.official_first_rule if score else "no feasible candidate",
    }


def _status(gap: float | None) -> str:
    if gap is None:
        return "missing"
    if gap <= -1e-9:
        return "better"
    if gap <= 1e-9:
        return "tie"
    return "worse"


def _load_hexaly(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["instance"]: row for row in csv.DictReader(handle)}


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "instance",
        "solution",
        "ratio_source",
        "our_ratio",
        "hexaly_v1_ratio",
        "gap_vs_hexaly_pct",
        "status",
        "local_feasible",
        "local_errors",
        "official_valid",
        "official_first_rule",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, object]], output_csv: Path, use_official: bool) -> None:
    print(f"wrote,{output_csv}")
    print(f"ratio_source,{'official_v1' if use_official else 'local'}")
    print("instance,our_ratio,hexaly_v1_ratio,gap_vs_hexaly_pct,status,solution")
    for row in rows:
        print(
            "{instance},{our_ratio},{hexaly_v1_ratio},{gap_vs_hexaly_pct},{status},{solution}".format(
                **row
            )
        )


def _scan_candidates(candidate_dir: Path, instance_name: str) -> list[Path]:
    suffix = instance_name.removeprefix("V_")
    result_suffix = instance_name.removeprefix("V_1.")
    patterns = (
        f"v1_1.{result_suffix}_*.xml",
        f"v1_1.{result_suffix}.xml",
        f"{suffix}_*.xml",
        f"{suffix}.xml",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(candidate_dir.glob(pattern))
    return sorted(paths)


def _instance_path(instance_name: str) -> Path:
    return INSTANCES_DIR / f"Instance_{instance_name}.xml"


def _instance_sort_key(instance_name: str) -> tuple[int, ...]:
    return tuple(int(part) for part in instance_name.removeprefix("V_").split("."))


if __name__ == "__main__":
    raise SystemExit(main())
