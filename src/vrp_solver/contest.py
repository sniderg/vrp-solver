from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from .analysis import summarize_solution
from .inventory import tank_violations
from .model import Customer, Instance, Shift, Solution
from .penalties import penalty_breakdown
from .rules import validate_solution


MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class ContestScore:
    score_days: int
    feasibility_days: int
    score_cutoff_minute: int
    feasibility_cutoff_minute: int
    submitted_shifts: int
    submitted_operations: int
    scored_shifts: int
    scored_operations: int
    scored_delivered_quantity: float
    scored_loaded_quantity: float
    scored_estimated_cost: float
    feasible: bool
    feasibility_errors: int
    feasibility_warnings: int
    hard_violations: int
    safety_kg_min: float
    tank_safety_breach_steps: int
    tank_negative_steps: int
    tank_overfill_steps: int
    vmi_customers_below_safety: int
    first_safety_breach_minute: int | None

    def flat(self) -> dict[str, object]:
        return self.__dict__.copy()


def score_prefix_with_feasibility_tail(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    feasibility_days: int | None = None,
    ignore_tail_call_ins: bool = False,
) -> ContestScore:
    """Score a route prefix while validating a longer no-free-delivery tail.

    Operations at or after ``score_days`` are dropped before both cost scoring and
    feasibility validation. The instance is then truncated to ``feasibility_days``
    so VMI inventory must remain feasible through the tail without extra work.
    """

    instance_days = _instance_days(instance)
    feasibility_days = instance_days if feasibility_days is None else feasibility_days
    if score_days <= 0:
        raise ValueError("score_days must be positive")
    if feasibility_days < score_days:
        raise ValueError("feasibility_days must be greater than or equal to score_days")
    if feasibility_days > instance_days:
        raise ValueError(
            f"feasibility_days={feasibility_days} exceeds instance horizon {instance_days}"
        )

    score_cutoff = score_days * MINUTES_PER_DAY
    feasibility_cutoff = feasibility_days * MINUTES_PER_DAY
    scored_solution = truncate_solution(solution, score_cutoff)
    feasibility_instance = truncate_instance(
        instance,
        feasibility_cutoff,
        call_in_cutoff_minute=score_cutoff if ignore_tail_call_ins else None,
    )

    shift_summaries = summarize_solution(instance, scored_solution)
    rule_violations = validate_solution(feasibility_instance, scored_solution)
    tank_bounds = tank_violations(feasibility_instance, scored_solution)
    penalties = penalty_breakdown(feasibility_instance, scored_solution)
    tank_counts = Counter(violation.code for violation in tank_bounds)
    safety_points = {
        violation.point
        for violation in tank_bounds
        if violation.code == "TANK_SAFETY_BREACH"
    }
    first_safety = min(
        (
            violation.time_start
            for violation in tank_bounds
            if violation.code == "TANK_SAFETY_BREACH"
        ),
        default=None,
    )
    errors = sum(1 for violation in rule_violations if violation.severity == "error")
    warnings = sum(1 for violation in rule_violations if violation.severity == "warning")

    return ContestScore(
        score_days=score_days,
        feasibility_days=feasibility_days,
        score_cutoff_minute=score_cutoff,
        feasibility_cutoff_minute=feasibility_cutoff,
        submitted_shifts=len(solution.shifts),
        submitted_operations=sum(len(shift.operations) for shift in solution.shifts),
        scored_shifts=len(scored_solution.shifts),
        scored_operations=sum(len(shift.operations) for shift in scored_solution.shifts),
        scored_delivered_quantity=sum(summary.delivered_quantity for summary in shift_summaries),
        scored_loaded_quantity=sum(summary.loaded_quantity for summary in shift_summaries),
        scored_estimated_cost=sum(summary.estimated_cost for summary in shift_summaries),
        feasible=errors == 0,
        feasibility_errors=errors,
        feasibility_warnings=warnings,
        hard_violations=penalties.hard_violations,
        safety_kg_min=penalties.safety_kg_min,
        tank_safety_breach_steps=tank_counts.get("TANK_SAFETY_BREACH", 0),
        tank_negative_steps=tank_counts.get("TANK_NEGATIVE", 0),
        tank_overfill_steps=tank_counts.get("TANK_OVERFILL", 0),
        vmi_customers_below_safety=len(safety_points),
        first_safety_breach_minute=first_safety,
    )


def truncate_solution(solution: Solution, cutoff_minute: int) -> Solution:
    shifts: list[Shift] = []
    for shift in solution.shifts:
        if shift.start >= cutoff_minute:
            continue
        operations = tuple(
            operation
            for operation in shift.operations
            if operation.arrival < cutoff_minute
        )
        if operations:
            shifts.append(replace(shift, operations=operations))
    return Solution(shifts=tuple(shifts))


def truncate_instance(
    instance: Instance,
    cutoff_minute: int,
    *,
    call_in_cutoff_minute: int | None = None,
) -> Instance:
    horizon = min(instance.horizon, (cutoff_minute + instance.unit - 1) // instance.unit)
    customers = tuple(
        _truncate_customer(customer, horizon, call_in_cutoff_minute)
        for customer in instance.customers
    )
    return replace(instance, horizon=horizon, customers=customers)


def _truncate_customer(
    customer: Customer,
    horizon: int,
    call_in_cutoff_minute: int | None,
) -> Customer:
    orders = customer.orders
    if customer.call_in and call_in_cutoff_minute is not None:
        orders = tuple(
            order
            for order in customer.orders
            if order.earliest_time < call_in_cutoff_minute
        )
    return replace(customer, forecast=tuple(customer.forecast[:horizon]), orders=orders)


def _instance_days(instance: Instance) -> int:
    horizon_minutes = instance.horizon * instance.unit
    return (horizon_minutes + MINUTES_PER_DAY - 1) // MINUTES_PER_DAY
