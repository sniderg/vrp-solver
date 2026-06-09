from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .analysis import customer_inventory_summary, summarize_solution
from .inventory import tank_violations
from .model import Instance, Solution
from .penalties import penalty_breakdown
from .rules import validate_solution
from .rolling import rolling_summary
from .smoothness import period_buckets, smoothness_summary


RATIO_RE = re.compile(r"Logistic Ratio\s*=\s*([0-9]+(?:[.,][0-9]+)?)")
VALID_RE = re.compile(r"THIS OUTPUT IS VALID")
FAIL_RE = re.compile(r"CHECKING FAILED")


@dataclass(frozen=True)
class Evaluation:
    feasible_local: bool
    local_errors: int
    local_warnings: int
    rule_counts: dict[str, int]
    tank_violation_counts: dict[str, int]
    delivered_quantity: float
    loaded_quantity: float
    estimated_distance_cost_plus_time_cost: float
    shifts: int
    operations: int
    min_vmi_safety_margin: float
    min_vmi_capacity_margin: float
    customers_below_safety: int
    customers_over_capacity: int
    official_valid: bool | None = None
    official_logistic_ratio: float | None = None
    official_first_rule_message: str = ""
    delivered_cv_daily: float = 0.0
    delivered_gini_daily: float = 0.0
    delivered_peak_share_daily: float = 0.0
    delivered_first_day_share: float = 0.0
    delivered_first_3_day_share: float = 0.0
    shift_starts_cv_daily: float = 0.0
    shift_starts_peak_daily: int = 0
    rolling_gap_to_required: float = 0.0
    rolling_gap_to_smooth_target: float = 0.0
    rolling_gap_to_planned_average: float = 0.0
    rolling_delivery_t_statistic: float | None = None
    hard_violations: int = 0
    total_soft_penalty: float = 0.0
    total_penalty: float = 0.0
    safety_kg_min: float = 0.0

    def flat(self) -> dict[str, object]:
        row: dict[str, object] = {
            "feasible_local": self.feasible_local,
            "local_errors": self.local_errors,
            "local_warnings": self.local_warnings,
            "delivered_quantity": self.delivered_quantity,
            "loaded_quantity": self.loaded_quantity,
            "estimated_distance_cost_plus_time_cost": self.estimated_distance_cost_plus_time_cost,
            "shifts": self.shifts,
            "operations": self.operations,
            "min_vmi_safety_margin": self.min_vmi_safety_margin,
            "min_vmi_capacity_margin": self.min_vmi_capacity_margin,
            "customers_below_safety": self.customers_below_safety,
            "customers_over_capacity": self.customers_over_capacity,
            "official_valid": self.official_valid,
            "official_logistic_ratio": self.official_logistic_ratio,
            "official_first_rule_message": self.official_first_rule_message,
            "delivered_cv_daily": self.delivered_cv_daily,
            "delivered_gini_daily": self.delivered_gini_daily,
            "delivered_peak_share_daily": self.delivered_peak_share_daily,
            "delivered_first_day_share": self.delivered_first_day_share,
            "delivered_first_3_day_share": self.delivered_first_3_day_share,
            "shift_starts_cv_daily": self.shift_starts_cv_daily,
            "shift_starts_peak_daily": self.shift_starts_peak_daily,
            "rolling_gap_to_required": self.rolling_gap_to_required,
            "rolling_gap_to_smooth_target": self.rolling_gap_to_smooth_target,
            "rolling_gap_to_planned_average": self.rolling_gap_to_planned_average,
            "rolling_delivery_t_statistic": self.rolling_delivery_t_statistic,
            "hard_violations": self.hard_violations,
            "total_soft_penalty": self.total_soft_penalty,
            "total_penalty": self.total_penalty,
            "safety_kg_min": self.safety_kg_min,
        }
        for code, count in sorted(self.rule_counts.items()):
            row[f"rule_{code}"] = count
        for code, count in sorted(self.tank_violation_counts.items()):
            row[f"tank_{code}"] = count
        return row


def evaluate_solution(
    instance: Instance,
    solution: Solution,
    *,
    instance_xml: Path | None = None,
    solution_xml: Path | None = None,
    checker_exe: Path | None = None,
    run_official_checker: bool = False,
) -> Evaluation:
    rule_violations = validate_solution(instance, solution)
    tank_bounds = tank_violations(instance, solution)
    shift_summaries = summarize_solution(instance, solution)
    inventory_summaries = customer_inventory_summary(instance, solution)
    daily_smoothness = smoothness_summary(period_buckets(instance, solution))
    rolling = rolling_summary(instance, solution)
    penalties = penalty_breakdown(instance, solution)
    vmi_inventory = [
        summary
        for summary in inventory_summaries
        if not instance.customer_by_point[summary.point].call_in
    ]

    rule_counts = Counter(violation.code for violation in rule_violations)
    tank_counts = Counter(violation.code for violation in tank_bounds)
    over_capacity_points = {
        violation.point for violation in tank_bounds if violation.code == "TANK_OVERFILL"
    }
    local_errors = sum(1 for violation in rule_violations if violation.severity == "error")
    local_warnings = sum(1 for violation in rule_violations if violation.severity == "warning")

    official_valid = None
    official_ratio = None
    official_first_rule = ""
    if run_official_checker:
        if instance_xml is None or solution_xml is None or checker_exe is None:
            raise ValueError("instance_xml, solution_xml, and checker_exe are required")
        official_valid, official_ratio, official_first_rule = run_checker(
            instance_xml,
            solution_xml,
            checker_exe,
        )

    return Evaluation(
        feasible_local=local_errors == 0,
        local_errors=local_errors,
        local_warnings=local_warnings,
        rule_counts=dict(rule_counts),
        tank_violation_counts=dict(tank_counts),
        delivered_quantity=sum(summary.delivered_quantity for summary in shift_summaries),
        loaded_quantity=sum(summary.loaded_quantity for summary in shift_summaries),
        estimated_distance_cost_plus_time_cost=sum(
            summary.estimated_cost for summary in shift_summaries
        ),
        shifts=len(solution.shifts),
        operations=sum(summary.operations for summary in shift_summaries),
        min_vmi_safety_margin=min(summary.min_margin_to_safety for summary in vmi_inventory),
        min_vmi_capacity_margin=_min_capacity_margin(instance, tank_bounds),
        customers_below_safety=sum(
            1 for summary in vmi_inventory if summary.min_margin_to_safety < -1e-6
        ),
        customers_over_capacity=len(over_capacity_points),
        official_valid=official_valid,
        official_logistic_ratio=official_ratio,
        official_first_rule_message=official_first_rule,
        delivered_cv_daily=daily_smoothness.delivered_cv,
        delivered_gini_daily=daily_smoothness.delivered_gini,
        delivered_peak_share_daily=daily_smoothness.delivered_peak_share,
        delivered_first_day_share=daily_smoothness.delivered_first_period_share,
        delivered_first_3_day_share=daily_smoothness.delivered_first_3_period_share,
        shift_starts_cv_daily=daily_smoothness.shift_starts_cv,
        shift_starts_peak_daily=daily_smoothness.shift_starts_peak,
        rolling_gap_to_required=rolling.monitored_gap_to_required,
        rolling_gap_to_smooth_target=rolling.monitored_gap_to_smooth_target,
        rolling_gap_to_planned_average=rolling.monitored_gap_to_planned_average,
        rolling_delivery_t_statistic=rolling.monitored_delivery_t_statistic,
        hard_violations=penalties.hard_violations,
        total_soft_penalty=penalties.total_soft_penalty,
        total_penalty=penalties.total_penalty,
        safety_kg_min=penalties.safety_kg_min,
    )


def _min_capacity_margin(instance: Instance, tank_bounds) -> float:
    if not tank_bounds:
        return 0.0
    margins = []
    for violation in tank_bounds:
        if violation.code == "TANK_OVERFILL":
            margins.append(violation.limit - violation.inventory)
    return min(margins) if margins else 0.0


def run_checker(
    instance_xml: Path,
    solution_xml: Path,
    checker_exe: Path,
) -> tuple[bool, float | None, str]:
    mono = shutil.which("mono")
    if mono is None:
        raise RuntimeError("mono is not installed or not on PATH")

    process = subprocess.run(
        [mono, str(checker_exe), str(instance_xml), str(solution_xml)],
        input="\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = process.stdout
    valid = bool(VALID_RE.search(output)) and not bool(FAIL_RE.search(output))
    ratio_match = RATIO_RE.search(output)
    ratio = float(ratio_match.group(1).replace(",", ".")) if ratio_match else None
    first_rule = ""
    for line in output.splitlines():
        if line.strip().startswith("["):
            first_rule = line.strip()
            break
    return valid, ratio, first_rule
