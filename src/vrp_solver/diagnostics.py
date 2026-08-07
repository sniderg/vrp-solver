"""Deterministic search diagnostics independent of a solver objective."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .inventory import tank_aggregates
from .model import Instance, Solution
from .rules import validate_structural


EPSILON = 1e-6
REFERENCE_CODES = {"REF_DRIVER", "REF_TRAILER", "SHI03"}
PHYSICAL_CODES = {"SHI06", "SHI11", "SHI16", "DYN01"}
RESOURCE_TIMING_CODES = {
    "DRI01", "DRI03", "DRI08", "LAY02", "LAY03", "SHI02", "SHI04",
    "SHI05", "TL01", "TL03",
}


@dataclass(frozen=True)
class ViolationVector:
    """Lexicographic feasibility measure for diagnostic search.

    Amount-duration fields are quantity multiplied by the instance time unit.
    This prevents a long stockout from tying a one-bucket epsilon breach.
    """

    non_finite_values: int
    reference_errors: int
    physical_errors: int
    missed_orders: int
    missed_order_deficit: float
    negative_quantity_minutes: float
    overfill_quantity_minutes: float
    safety_deficit_quantity_minutes: float
    resource_timing_errors: int
    other_errors: int

    @property
    def locally_feasible(self) -> bool:
        return self.key() == (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    def key(self) -> tuple[int, int, int, int, float, float, float, float, int, int]:
        return (
            self.non_finite_values,
            self.reference_errors,
            self.physical_errors,
            self.missed_orders,
            self.missed_order_deficit,
            self.negative_quantity_minutes,
            self.overfill_quantity_minutes,
            self.safety_deficit_quantity_minutes,
            self.resource_timing_errors,
            self.other_errors,
        )

    def flat(self) -> dict[str, int | float | bool]:
        return {**asdict(self), "locally_feasible": self.locally_feasible}


@dataclass(frozen=True)
class AtomicRepairDecision:
    accepted: bool
    reason: str
    before: ViolationVector
    after: ViolationVector


def assess_atomic_repair(
    instance: Instance,
    incumbent: Solution,
    candidate: Solution,
) -> AtomicRepairDecision:
    """Accept a repair only when it improves without moving damage elsewhere.

    Diversification search can maintain a separate diagnostic archive. This
    gate is deliberately stricter: it is for claims that a block was repaired.
    """
    before = violation_vector(instance, incumbent)
    after = violation_vector(instance, candidate)
    guarded = (
        "non_finite_values",
        "reference_errors",
        "physical_errors",
        "negative_quantity_minutes",
        "overfill_quantity_minutes",
        "safety_deficit_quantity_minutes",
        "resource_timing_errors",
        "other_errors",
    )
    regressions = [
        field for field in guarded
        if getattr(after, field) > getattr(before, field) + EPSILON
    ]
    if regressions:
        return AtomicRepairDecision(
            False, "regressed:" + ",".join(regressions), before, after,
        )
    if after.key() >= before.key():
        return AtomicRepairDecision(False, "no_strict_improvement", before, after)
    return AtomicRepairDecision(True, "strict_improvement", before, after)


def solution_fingerprint(solution: Solution) -> str:
    """Hash every value that can change topology, timing, or state flow."""
    payload = [
        {
            "index": shift.index,
            "driver": shift.driver,
            "trailer": shift.trailer,
            "start": shift.start,
            "operations": [
                [operation.point, operation.arrival, operation.quantity]
                for operation in shift.operations
            ],
        }
        for shift in solution.shifts
    ]
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def violation_vector(instance: Instance, solution: Solution) -> ViolationVector:
    non_finite = _non_finite_count(solution)
    # Avoid sending malformed non-finite values through arithmetic-heavy replay.
    if non_finite:
        return ViolationVector(non_finite, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    # DYN01 is the only vector component sourced from tank replay; every other
    # counted code is emitted by the structural (per-shift) validators, and
    # QS01/QS02 are deliberately excluded.  The aggregate pass projects all
    # tanks vectorized instead of building per-step TankEvent objects.
    violations = [v for v in validate_structural(instance, solution) if v.severity == "error"]
    dyn01_events, negative, overfill, safety = tank_aggregates(instance, solution)
    reference = sum(v.code in REFERENCE_CODES for v in violations)
    physical = sum(v.code in PHYSICAL_CODES for v in violations) + dyn01_events
    resource_timing = sum(v.code in RESOURCE_TIMING_CODES for v in violations)
    known = REFERENCE_CODES | PHYSICAL_CODES | RESOURCE_TIMING_CODES | {"QS01", "QS02"}
    other = sum(v.code not in known for v in violations)

    missed_orders, missed_deficit = _missed_order_deficit(instance, solution)

    return ViolationVector(
        non_finite_values=0,
        reference_errors=reference,
        physical_errors=physical,
        missed_orders=missed_orders,
        missed_order_deficit=float(missed_deficit),
        negative_quantity_minutes=float(negative),
        overfill_quantity_minutes=float(overfill),
        safety_deficit_quantity_minutes=float(safety),
        resource_timing_errors=resource_timing,
        other_errors=other,
    )


def _non_finite_count(solution: Solution) -> int:
    return sum(
        not math.isfinite(operation.quantity)
        for shift in solution.shifts
        for operation in shift.operations
    )


def _missed_order_deficit(instance: Instance, solution: Solution) -> tuple[int, float]:
    delivered: dict[tuple[int, int], float] = {}
    for shift in solution.shifts:
        for operation in shift.operations:
            customer = instance.customer_by_point.get(operation.point)
            if customer is None or not customer.call_in or operation.quantity <= 0:
                continue
            for order_index, order in enumerate(customer.orders):
                if order.earliest_time <= operation.arrival <= order.latest_time:
                    key = (customer.index, order_index)
                    delivered[key] = delivered.get(key, 0.0) + operation.quantity

    latest_required = instance.horizon * instance.unit
    count = 0
    deficit = 0.0
    for customer in instance.customers:
        if not customer.call_in:
            continue
        for order_index, order in enumerate(customer.orders):
            if order.latest_time > latest_required:
                continue
            remaining = order.min_quantity_to_satisfy - delivered.get(
                (customer.index, order_index), 0.0,
            )
            if remaining > EPSILON:
                count += 1
                deficit += remaining
    return count, deficit
