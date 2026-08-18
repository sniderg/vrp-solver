from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import os

import numpy as np

from .analysis import summarize_solution
from .inventory import delivery_by_customer_step, tank_violations
from .model import Customer, Instance, Shift, Solution
from .penalties import penalty_breakdown
from .rules import derive_solution, validate_solution

try:
    if os.environ.get("VRP_DISABLE_FAST", "").lower() in {"1", "true", "yes"}:
        raise ImportError("accelerated scoring disabled by VRP_DISABLE_FAST")
    from .inventory_fast import score_all_customers as _cy_score_all_customers
    _HAS_FAST_SCORING = True
except ImportError:
    _HAS_FAST_SCORING = False


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

    When the Cython extension is available, this uses a fast path that avoids
    creating TankEvent objects and eliminates redundant derive_solution calls.
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

    if _HAS_FAST_SCORING:
        return _score_fast(
            instance, feasibility_instance, solution, scored_solution,
            score_days, feasibility_days, score_cutoff, feasibility_cutoff,
        )

    # Legacy Python fallback path
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
        # This score is used to decide whether a generated solution can
        # replace an incumbent.  The official checker rejects *every* rule
        # error, including VMI safety-level breaches (QS02) and missed
        # call-in orders (QS01).  Do not let an internal "soft" classification
        # make an officially-invalid candidate look preferable.
        hard_violations=errors,
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


def truncate_solution_atomic(solution: Solution, cutoff_minute: int) -> Solution:
    """Truncate solution keeping full shift operations intact to avoid mid-shift splits."""
    shifts: list[Shift] = [
        shift for shift in solution.shifts
        if shift.start < cutoff_minute and shift.operations
    ]
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


def _build_customer_arrays(
    instance: Instance,
    solution: Solution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-build numpy arrays for all customers for the Cython kernel."""
    n = len(instance.customers)
    h = instance.horizon

    initial_quantities = np.empty(n, dtype=np.float64)
    forecasts = np.zeros((n, h), dtype=np.float64)
    capacities = np.empty(n, dtype=np.float64)
    safety_levels = np.empty(n, dtype=np.float64)
    is_call_in = np.zeros(n, dtype=np.int32)
    deliveries_matrix = np.zeros((n, h), dtype=np.float64)

    # Map customer point -> customer index in arrays
    point_to_idx: dict[int, int] = {}
    for i, customer in enumerate(instance.customers):
        point_to_idx[customer.index] = i
        initial_quantities[i] = customer.initial_tank_quantity
        capacities[i] = customer.capacity
        safety_levels[i] = customer.safety_level
        is_call_in[i] = 1 if customer.call_in else 0
        forecast = customer.forecast
        length = min(len(forecast), h)
        for step in range(length):
            forecasts[i, step] = forecast[step]

    # Fill deliveries from solution
    for shift in solution.shifts:
        for operation in shift.operations:
            if operation.quantity <= 0:
                continue
            idx = point_to_idx.get(operation.point)
            if idx is None:
                continue
            step = min(max(operation.arrival // instance.unit, 0), h - 1)
            deliveries_matrix[idx, step] += operation.quantity

    return initial_quantities, forecasts, capacities, safety_levels, is_call_in, deliveries_matrix


def _validate_non_tank_rules(
    instance: Instance,
    derived_shifts,
    solution: Solution,
) -> tuple[int, int, int]:
    """Run all validation EXCEPT tank bounds (which is handled by Cython).

    Returns (error_count, warning_count, hard_violation_count).
    Hard violations are SHI06, REF_DRIVER, REF_TRAILER from non-tank rules.
    """
    from .rules import (
        _validate_shift_references,
        _validate_shift_operations,
        _validate_resource_constraints,
        _validate_service_quality,
    )
    violations = []
    violations.extend(_validate_shift_references(instance, solution))
    violations.extend(_validate_shift_operations(instance, derived_shifts))
    violations.extend(_validate_resource_constraints(instance, derived_shifts))
    violations.extend(_validate_service_quality(instance, solution))
    errors = sum(1 for v in violations if v.severity == "error")
    warnings = sum(1 for v in violations if v.severity == "warning")
    hard = sum(
        1
        for v in violations
        if v.code in {"SHI06", "REF_DRIVER", "REF_TRAILER"}
    )
    return errors, warnings, hard


def _score_fast(
    instance: Instance,
    feasibility_instance: Instance,
    solution: Solution,
    scored_solution: Solution,
    score_days: int,
    feasibility_days: int,
    score_cutoff: int,
    feasibility_cutoff: int,
) -> ContestScore:
    """Cython-accelerated scoring path.

    Key optimizations vs the legacy path:
    1. derive_solution called twice max (once for cost on full instance,
       once for feasibility on truncated instance) vs 3× in legacy
    2. Tank stats computed by C-level score_all_customers (no TankEvent objects)
    3. Non-tank rule validation uses pre-computed derived shifts
    """
    # 1. Derive shifts for cost computation (using full instance for correct costs)
    derived = derive_solution(instance, scored_solution)

    # 2. Cost summary from derived shifts (inline summarize_solution logic)
    total_delivered = 0.0
    total_loaded = 0.0
    total_cost = 0.0
    for ds in derived:
        shift = ds.shift
        driver = instance.drivers[shift.driver]
        trailer = instance.trailers[shift.trailer]
        prev_point = instance.base_index
        distance = 0.0
        for op in shift.operations:
            distance += instance.distance_matrix[prev_point][op.point]
            prev_point = op.point
        distance += instance.distance_matrix[prev_point][instance.base_index]
        delivered = sum(op.quantity for op in shift.operations if op.quantity > 0)
        loaded = sum(-op.quantity for op in shift.operations if op.quantity < 0)
        distance_cost = distance * trailer.distance_cost
        working_time = ds.end - shift.start - ds.layovers * driver.layover_duration
        time_cost = working_time * driver.time_cost
        layover_cost = ds.layovers * driver.layover_cost
        total_delivered += delivered
        total_loaded += loaded
        total_cost += distance_cost + time_cost + layover_cost

    # 3. Derive shifts for feasibility instance (may differ from full instance)
    feasibility_derived = derive_solution(feasibility_instance, scored_solution)

    # 4. Non-tank rule validation using pre-computed feasibility derived shifts
    non_tank_errors, non_tank_warnings, non_tank_hard = _validate_non_tank_rules(
        feasibility_instance, feasibility_derived, scored_solution,
    )

    # 5. Cython tank stats (single pass, no TankEvent objects)
    (
        initial_quantities, forecasts, capacities, safety_levels,
        is_call_in_arr, deliveries_matrix,
    ) = _build_customer_arrays(feasibility_instance, scored_solution)

    (
        safety_breach_count, negative_count, overfill_count,
        _hard_tank_cy, safety_kg_min,
        breach_points_count, negative_points_count,
    ) = _cy_score_all_customers(
        initial_quantities, forecasts, capacities, safety_levels,
        is_call_in_arr, len(feasibility_instance.customers),
        feasibility_instance.horizon, feasibility_instance.unit,
        deliveries_matrix,
    )

    # Tank violations map to rule codes:
    # TANK_NEGATIVE -> DYN01 (error, hard), TANK_OVERFILL -> DYN01 (error, hard),
    # TANK_SAFETY_BREACH -> QS02 (error, NOT hard)
    # All three are severity="error" in validate_solution.
    tank_errors = negative_count + overfill_count + safety_breach_count
    total_errors = non_tank_errors + tank_errors
    total_warnings = non_tank_warnings

    # See the fallback path above: for native search acceptance, all official
    # rule errors are hard.  QS02 safety breaches are therefore included.
    total_hard = total_errors

    return ContestScore(
        score_days=score_days,
        feasibility_days=feasibility_days,
        score_cutoff_minute=score_cutoff,
        feasibility_cutoff_minute=feasibility_cutoff,
        submitted_shifts=len(solution.shifts),
        submitted_operations=sum(len(shift.operations) for shift in solution.shifts),
        scored_shifts=len(scored_solution.shifts),
        scored_operations=sum(len(shift.operations) for shift in scored_solution.shifts),
        scored_delivered_quantity=total_delivered,
        scored_loaded_quantity=total_loaded,
        scored_estimated_cost=total_cost,
        feasible=total_errors == 0,
        feasibility_errors=total_errors,
        feasibility_warnings=total_warnings,
        hard_violations=total_hard,
        safety_kg_min=safety_kg_min,
        tank_safety_breach_steps=safety_breach_count,
        tank_negative_steps=negative_count,
        tank_overfill_steps=overfill_count,
        vmi_customers_below_safety=breach_points_count,
        first_safety_breach_minute=None,  # Not tracked in fast path
    )
