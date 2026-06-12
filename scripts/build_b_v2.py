#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from vrp_solver.contest import (
    ContestScore,
    score_prefix_with_feasibility_tail,
    truncate_instance,
    truncate_solution,
)
from vrp_solver.evaluate import run_checker
from vrp_solver.inventory import tank_violations
from vrp_solver.model import Solution
from vrp_solver.rules import validate_solution
from vrp_solver.solver.cluster_greedy import construct_cluster_solution
from vrp_solver.solver.column_loop import ColumnLoopConfig, column_generation_rescue
from vrp_solver.solver.greedy import construct_solution
from vrp_solver.xml_io import load_instance, save_solution


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "roadef_2016_data"
B_INSTANCES = DATA_ROOT / "set_B" / "Instances_B_V25-11042016"
V2_CHECKER = (
    DATA_ROOT
    / "checker_v2"
    / "Challenge_Roadef_EURO_Checker_V2"
    / "bin"
    / "Release"
    / "IRP_Roadef_Challenge_Checker.exe"
)


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    path: Path
    score: ContestScore
    official_valid: bool | None = None
    official_ratio: float | None = None
    official_first_rule: str = ""


@dataclass(frozen=True)
class DiagnosticRow:
    phase: str
    kind: str
    key: str
    count: int
    first_day: float | str = ""
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Set B V2 solutions from scratch, separate from benchmark improvers."
    )
    parser.add_argument("--instance", default="V2.12", help="B instance name, e.g. V2.12")
    parser.add_argument("--instance-xml", type=Path, help="Override instance XML path.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "scratch" / "b_v2_build")
    parser.add_argument(
        "--constructor",
        choices=("cluster", "greedy", "both"),
        default="cluster",
        help="Seed constructor to run before column rescue.",
    )
    parser.add_argument("--safety-buffer", type=float, default=0.20)
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--terminal-buffer-days", type=float, default=0.0)
    parser.add_argument("--max-shifts", type=int)
    parser.add_argument("--score-days", type=int)
    parser.add_argument("--feasibility-days", type=int)
    parser.add_argument("--rescue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rescue-iterations", type=int, default=5)
    parser.add_argument("--replace-from-day", type=int, default=0)
    parser.add_argument("--max-pressure-customers", type=int, default=24)
    parser.add_argument("--samples-per-customer", type=int, default=10)
    parser.add_argument("--max-chain-length", type=int, default=4)
    parser.add_argument("--nearest-chain-neighbors", type=int, default=10)
    parser.add_argument("--max-candidates-per-iteration", type=int, default=1800)
    parser.add_argument("--selector-time-limit", type=float, default=300.0)
    parser.add_argument("--selector-phase", choices=("auto", "feasibility", "cost"), default="feasibility")
    parser.add_argument("--multi-reload-columns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-top-n", type=int, default=12)
    parser.add_argument("--no-official", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    instance_path = args.instance_xml or B_INSTANCES / f"{args.instance}.xml"
    if not instance_path.exists():
        raise SystemExit(f"missing instance XML: {instance_path}")

    instance = load_instance(instance_path)
    score_days = args.score_days or _instance_days(instance)
    feasibility_days = args.feasibility_days or score_days
    output_dir = args.output_dir / args.instance
    output_dir.mkdir(parents=True, exist_ok=True)

    use_official = (
        not args.no_official
        and shutil.which("mono") is not None
        and V2_CHECKER.exists()
    )
    print(
        f"build,{args.instance},instance,{instance_path},score_days,{score_days},"
        f"feasibility_days,{feasibility_days},official,{use_official}"
    )

    results: list[PhaseResult] = []
    diagnostics: list[DiagnosticRow] = []
    constructors = ("cluster", "greedy") if args.constructor == "both" else (args.constructor,)
    for constructor in constructors:
        seed_path = output_dir / f"{args.instance}_{constructor}_seed.xml"
        if constructor == "cluster":
            seed, report = construct_cluster_solution(
                instance,
                safety_buffer=args.safety_buffer,
                neighborhood_size=args.neighborhood_size,
                max_shifts=args.max_shifts,
                score_cutoff_minute=score_days * 1440,
                terminal_buffer_days=args.terminal_buffer_days,
            )
        else:
            seed, report = construct_solution(
                instance,
                safety_buffer=args.safety_buffer,
                max_shifts=args.max_shifts,
            )
        save_solution(seed, seed_path)
        print(
            f"constructed,{constructor},path,{seed_path},shifts,{report.shifts},"
            f"operations,{report.operations},unscheduled,{len(report.unscheduled_customers)},"
            f"exhausted,{report.exhausted_resources}"
        )
        seed_result = score_phase(
            constructor,
            instance,
            instance_path,
            seed_path,
            seed,
            score_days,
            feasibility_days,
            use_official,
        )
        results.append(seed_result)
        print_phase(seed_result)
        if args.diagnostics:
            phase_diagnostics = phase_diagnostics_rows(
                constructor,
                instance,
                seed,
                score_days,
                feasibility_days,
                top_n=args.diagnostic_top_n,
            )
            diagnostics.extend(phase_diagnostics)
            print_diagnostics(phase_diagnostics)

        if args.rescue:
            rescued, steps = column_generation_rescue(
                instance,
                seed,
                config=ColumnLoopConfig(
                    start_day=0,
                    end_day=score_days,
                    replace_from_day=args.replace_from_day,
                    iterations=args.rescue_iterations,
                    max_pressure_customers=args.max_pressure_customers,
                    samples_per_customer=args.samples_per_customer,
                    max_chain_length=args.max_chain_length,
                    nearest_chain_neighbors=args.nearest_chain_neighbors,
                    max_candidates_per_iteration=args.max_candidates_per_iteration,
                    multi_reload_columns=args.multi_reload_columns,
                    selector_time_limit=args.selector_time_limit,
                    selector_phase=args.selector_phase,
                    commit_end_day=score_days,
                ),
            )
            rescued_path = output_dir / f"{args.instance}_{constructor}_column_rescue.xml"
            save_solution(rescued, rescued_path)
            for step in steps:
                print(
                    "rescue_step,{},{},generated,{},pool,{},selected,{},feasible,{},errors,{},hard,{}".format(
                        constructor,
                        step.iteration,
                        step.generated_candidates,
                        step.pool_size,
                        step.selected_extra_shifts,
                        step.feasible,
                        step.feasibility_errors,
                        step.hard_violations,
                    )
                )
                print(
                    "rescue_diag,{},{},pressure,{},generated_served,{},generated_pressure_cover,{},"
                    "selected_pressure_cover,{},selected_errors,{},selected_hard,{},"
                    "repaired_errors,{},repaired_hard,{},accepted,{},best_improved,{}".format(
                        constructor,
                        step.iteration,
                        "/".join(str(customer) for customer in step.pressure_customers[:12]),
                        step.generated_served_customers,
                        step.generated_pressure_coverage,
                        step.selected_pressure_coverage,
                        step.selected_feasibility_errors,
                        step.selected_hard_violations,
                        step.repaired_feasibility_errors,
                        step.repaired_hard_violations,
                        step.candidate_accepted,
                        step.best_improved,
                    )
                )
            rescued_result = score_phase(
                f"{constructor}_column_rescue",
                instance,
                instance_path,
                rescued_path,
                rescued,
                score_days,
                feasibility_days,
                use_official,
            )
            results.append(rescued_result)
            print_phase(rescued_result)
            if args.diagnostics:
                phase_diagnostics = phase_diagnostics_rows(
                    f"{constructor}_column_rescue",
                    instance,
                    rescued,
                    score_days,
                    feasibility_days,
                    top_n=args.diagnostic_top_n,
                )
                diagnostics.extend(phase_diagnostics)
                print_diagnostics(phase_diagnostics)

    summary_path = output_dir / "build_summary.csv"
    write_summary(results, summary_path)
    if args.diagnostics:
        diagnostics_path = output_dir / "diagnostics.csv"
        write_diagnostics(diagnostics, diagnostics_path)
        print(f"wrote,{diagnostics_path}")
    best = best_result(results)
    if best is not None:
        best_path = output_dir / f"{args.instance}_best.xml"
        shutil.copyfile(best.path, best_path)
        print(f"best,{best.phase},ratio,{ratio(best.score):.6f},path,{best_path}")
    print(f"wrote,{summary_path}")
    return 0


def score_phase(
    phase: str,
    instance,
    instance_path: Path,
    path: Path,
    solution,
    score_days: int,
    feasibility_days: int,
    use_official: bool,
) -> PhaseResult:
    score = score_prefix_with_feasibility_tail(
        instance,
        solution,
        score_days=score_days,
        feasibility_days=feasibility_days,
        ignore_tail_call_ins=True,
    )
    if not use_official or not score.feasible:
        return PhaseResult(phase=phase, path=path, score=score)
    valid, official_ratio, first_rule = run_checker(instance_path, path, V2_CHECKER)
    return PhaseResult(
        phase=phase,
        path=path,
        score=score,
        official_valid=valid,
        official_ratio=official_ratio,
        official_first_rule=first_rule,
    )


def print_phase(result: PhaseResult) -> None:
    print(
        "phase,{phase},path,{path},local_feasible,{feasible},errors,{errors},hard,{hard},"
        "ratio,{ratio:.6f},official_valid,{official_valid},official_ratio,{official_ratio},first_rule,{first_rule}".format(
            phase=result.phase,
            path=result.path,
            feasible=result.score.feasible,
            errors=result.score.feasibility_errors,
            hard=result.score.hard_violations,
            ratio=ratio(result.score),
            official_valid="" if result.official_valid is None else result.official_valid,
            official_ratio="" if result.official_ratio is None else f"{result.official_ratio:.6f}",
            first_rule=result.official_first_rule,
        )
    )


def phase_diagnostics_rows(
    phase: str,
    instance,
    solution: Solution,
    score_days: int,
    feasibility_days: int,
    *,
    top_n: int,
) -> list[DiagnosticRow]:
    score_cutoff = score_days * 1440
    feasibility_cutoff = feasibility_days * 1440
    scored_solution = truncate_solution(solution, score_cutoff)
    feasibility_instance = truncate_instance(
        instance,
        feasibility_cutoff,
        call_in_cutoff_minute=score_cutoff,
    )

    rule_violations = [
        violation
        for violation in validate_solution(feasibility_instance, scored_solution)
        if violation.severity == "error"
    ]
    tank_bounds = tank_violations(feasibility_instance, scored_solution)
    rows: list[DiagnosticRow] = []

    for code, count in Counter(violation.code for violation in rule_violations).most_common():
        rows.append(DiagnosticRow(phase, "rule_code", code, count))
    for code, count in Counter(violation.code for violation in tank_bounds).most_common():
        rows.append(DiagnosticRow(phase, "tank_code", code, count))

    by_point: dict[int, list] = defaultdict(list)
    for violation in tank_bounds:
        by_point[violation.point].append(violation)
    ranked_points = sorted(
        by_point.items(),
        key=lambda item: (
            min(violation.time_start for violation in item[1]),
            -len(item[1]),
            item[0],
        ),
    )
    for point, violations in ranked_points[:top_n]:
        code_counts = Counter(violation.code for violation in violations)
        first_minute = min(violation.time_start for violation in violations)
        worst_margin = min(violation.inventory - violation.limit for violation in violations)
        detail = ";".join(
            f"{code}:{count}" for code, count in sorted(code_counts.items())
        )
        rows.append(
            DiagnosticRow(
                phase=phase,
                kind="tank_point",
                key=str(point),
                count=len(violations),
                first_day=f"{first_minute / 1440:.2f}",
                detail=f"{detail};worst_margin={worst_margin:.3f}",
            )
        )

    point_counts = Counter(
        violation.point
        for violation in rule_violations
        if violation.point is not None
    )
    for point, count in point_counts.most_common(top_n):
        rows.append(
            DiagnosticRow(
                phase=phase,
                kind="rule_point",
                key=str(point),
                count=count,
            )
        )
    return rows


def print_diagnostics(rows: list[DiagnosticRow]) -> None:
    for row in rows:
        print(
            "diagnostic,{phase},{kind},{key},count,{count},first_day,{first_day},detail,{detail}".format(
                phase=row.phase,
                kind=row.kind,
                key=row.key,
                count=row.count,
                first_day=row.first_day,
                detail=row.detail,
            )
        )


def write_diagnostics(rows: list[DiagnosticRow], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["phase", "kind", "key", "count", "first_day", "detail"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "phase": row.phase,
                    "kind": row.kind,
                    "key": row.key,
                    "count": row.count,
                    "first_day": row.first_day,
                    "detail": row.detail,
                }
            )


def write_summary(results: list[PhaseResult], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "phase",
                "path",
                "local_feasible",
                "errors",
                "hard",
                "ratio",
                "official_valid",
                "official_ratio",
                "official_first_rule",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "phase": result.phase,
                    "path": result.path,
                    "local_feasible": result.score.feasible,
                    "errors": result.score.feasibility_errors,
                    "hard": result.score.hard_violations,
                    "ratio": f"{ratio(result.score):.6f}",
                    "official_valid": result.official_valid if result.official_valid is not None else "",
                    "official_ratio": (
                        f"{result.official_ratio:.6f}" if result.official_ratio is not None else ""
                    ),
                    "official_first_rule": result.official_first_rule,
                }
            )


def best_result(results: list[PhaseResult]) -> PhaseResult | None:
    feasible = [
        result
        for result in results
        if result.score.feasible and result.official_valid is not False
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda result: result.official_ratio or ratio(result.score))


def ratio(score: ContestScore) -> float:
    return score.scored_estimated_cost / max(score.scored_delivered_quantity, 1e-9)


def _instance_days(instance) -> int:
    return (instance.horizon * instance.unit + 1439) // 1440


if __name__ == "__main__":
    raise SystemExit(main())
