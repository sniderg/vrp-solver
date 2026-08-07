"""Atomic, state-preserving multi-shift topology repairs."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations

from ..diagnostics import assess_atomic_repair, solution_fingerprint
from ..highs_repair import repair_quantities_with_highs
from ..highs_time_opt import try_optimize_shift_times
from ..model import Instance, Shift, Solution
from ..rules import derive_solution
from .targeted_rescue import normalize_source_loads


@dataclass(frozen=True)
class SplitFunnel:
    enumerated: int = 0
    prefix_feasible: int = 0
    resource_gaps: int = 0
    suffix_timing_feasible: int = 0
    unique_candidates: int = 0
    strict_improvements: int = 0
    rejected_physical: int = 0
    rejected_inventory: int = 0
    rejected_resource_timing: int = 0
    rejected_other: int = 0


@dataclass(frozen=True)
class ReorderFunnel:
    enumerated: int = 0
    timing_feasible: int = 0
    unique_candidates: int = 0
    strict_improvements: int = 0


@dataclass(frozen=True)
class EjectionFunnel:
    enumerated: int = 0
    donor_timing_feasible: int = 0
    recipient_timing_feasible: int = 0
    quantity_models: int = 0
    quantity_solved: int = 0
    strict_improvements: int = 0


@dataclass(frozen=True)
class TwoRecipientFunnel:
    donor_fragments: int = 0
    donor_timing_feasible: int = 0
    first_recipient_variants: int = 0
    second_recipient_variants: int = 0
    quantity_models: int = 0
    quantity_solved: int = 0
    strict_improvements: int = 0


@dataclass(frozen=True)
class DirectEjectionFunnel:
    donor_timing_feasible: int = 0
    direct_routes: int = 0
    resource_feasible: int = 0
    quantity_models: int = 0
    quantity_solved: int = 0
    strict_improvements: int = 0


def direct_ejection_candidates(
    instance: Instance,
    solution: Solution,
    shift_indices: tuple[int, ...],
    *,
    arrival_radius: int = 1_800,
    max_quantity_models: int = 128,
) -> tuple[list[Solution], DirectEjectionFunnel]:
    """Eject one customer to a new source-backed shift."""
    by_index = {shift.index: shift for shift in solution.shifts}
    derived = {item.shift.index: item for item in derive_solution(instance, solution)}
    horizon_days = max(1, (instance.horizon * instance.unit + 1_439) // 1_440)
    results: list[Solution] = []
    seen: set[str] = set()
    counts = {name: 0 for name in DirectEjectionFunnel.__dataclass_fields__}
    for donor_index in shift_indices:
        donor = by_index.get(donor_index)
        if donor is None:
            continue
        for position in reversed(range(len(donor.operations))):
            moved = donor.operations[position]
            if moved.point not in instance.customer_by_point:
                continue
            donor_ops = donor.operations[:position] + donor.operations[position + 1:]
            if not donor_ops:
                continue
            repaired_donor = _retime_with_segment_reorder(
                instance, solution, replace(donor, operations=donor_ops),
            )
            if repaired_donor is None:
                continue
            counts["donor_timing_feasible"] += 1
            customer = instance.customer_by_point[moved.point]
            for driver in instance.drivers:
                for trailer_id in driver.trailer_ids:
                    if trailer_id not in customer.allowed_trailers:
                        continue
                    for source in instance.sources:
                        if trailer_id not in source.allowed_trailers:
                            continue
                        lead = (
                            instance.time_matrix[instance.base_index][source.index]
                            + source.setup_time
                            + instance.time_matrix[source.index][customer.index]
                        )
                        arrivals = {moved.arrival}
                        for window in customer.time_windows:
                            if window.end < moved.arrival - arrival_radius or window.start > moved.arrival + arrival_radius:
                                continue
                            arrivals.add(max(window.start, moved.arrival - arrival_radius))
                            arrivals.add(min(window.end - customer.setup_time, moved.arrival + arrival_radius))
                        for arrival in sorted(arrivals):
                            start = arrival - lead
                            if start < 0:
                                continue
                            shift = Shift(
                                index=max(by_index, default=-1) + 1,
                                driver=driver.index,
                                trailer=trailer_id,
                                start=start,
                                operations=(
                                    # Quantity repair may change both values;
                                    # this initial pair is materially balanced.
                                    replace(moved, point=source.index, arrival=start + instance.time_matrix[instance.base_index][source.index], quantity=-moved.quantity),
                                    replace(moved, arrival=arrival),
                                ),
                            )
                            shift = try_optimize_shift_times(instance, shift)
                            if shift is None:
                                continue
                            counts["direct_routes"] += 1
                            end = derive_solution(instance, Solution((replace(shift, index=0),)))[0].end
                            if _overlaps_unchanged_resources(
                                instance, solution, derived, donor.index, shift, end,
                            ):
                                continue
                            counts["resource_feasible"] += 1
                            topology = _reindex([
                                repaired_donor if item.index == donor.index else item
                                for item in solution.shifts
                            ] + [shift])
                            topology = normalize_source_loads(instance, topology)
                            fingerprint = solution_fingerprint(topology)
                            if fingerprint in seen:
                                continue
                            seen.add(fingerprint)
                            if counts["quantity_models"] >= max_quantity_models:
                                return results, DirectEjectionFunnel(**counts)
                            counts["quantity_models"] += 1
                            repaired, report = repair_quantities_with_highs(
                                instance, topology,
                                score_days=horizon_days,
                                feasibility_days=horizon_days,
                                quantity_objective="max-delivered",
                                strict_inventory=True,
                            )
                            if report.status != "Optimal":
                                continue
                            counts["quantity_solved"] += 1
                            if assess_atomic_repair(instance, solution, repaired).accepted:
                                counts["strict_improvements"] += 1
                                results.append(repaired)
    return results, DirectEjectionFunnel(**counts)


def _overlaps_unchanged_resources(
    instance: Instance,
    solution: Solution,
    derived: dict[int, object],
    removed_index: int,
    candidate: Shift,
    candidate_end: int,
) -> bool:
    rest = instance.drivers[candidate.driver].min_inter_shift_duration
    for shift in solution.shifts:
        if shift.index == removed_index:
            continue
        other_end = derived[shift.index].end
        if shift.driver == candidate.driver and (
            candidate.start < other_end + rest
            and shift.start < candidate_end + rest
        ):
            return True
        if shift.trailer == candidate.trailer and (
            candidate.start < other_end and shift.start < candidate_end
        ):
            return True
    return False


def two_recipient_ejection_candidates(
    instance: Instance,
    solution: Solution,
    shift_indices: tuple[int, ...],
    *,
    arrival_radius: int = 1_500,
    max_recipient_variants: int = 32,
    max_quantity_models: int = 128,
) -> tuple[list[Solution], TwoRecipientFunnel]:
    """Move a two-customer fragment into two distinct recipient routes."""
    by_index = {shift.index: shift for shift in solution.shifts}
    horizon_days = max(1, (instance.horizon * instance.unit + 1_439) // 1_440)
    results: list[Solution] = []
    seen: set[str] = set()
    counts = {
        "donor_fragments": 0,
        "donor_timing_feasible": 0,
        "first_recipient_variants": 0,
        "second_recipient_variants": 0,
        "quantity_models": 0,
        "quantity_solved": 0,
        "strict_improvements": 0,
    }
    for donor_index in shift_indices:
        donor = by_index.get(donor_index)
        if donor is None:
            continue
        for start in range(len(donor.operations) - 2, -1, -1):
            moved = donor.operations[start:start + 2]
            if not all(op.point in instance.customer_by_point for op in moved):
                continue
            counts["donor_fragments"] += 1
            donor_ops = donor.operations[:start] + donor.operations[start + 2:]
            if not donor_ops:
                continue
            repaired_donor = _retime_with_segment_reorder(
                instance, solution, replace(donor, operations=donor_ops),
            )
            if repaired_donor is None:
                continue
            counts["donor_timing_feasible"] += 1
            first = _single_operation_recipient_variants(
                instance, solution, donor, moved[0], arrival_radius,
            )[:max_recipient_variants]
            second = _single_operation_recipient_variants(
                instance, solution, donor, moved[1], arrival_radius,
            )[:max_recipient_variants]
            counts["first_recipient_variants"] += len(first)
            counts["second_recipient_variants"] += len(second)
            for first_index, first_shift in first:
                for second_index, second_shift in second:
                    if first_index == second_index:
                        continue
                    topology = _reindex([
                        repaired_donor if shift.index == donor.index
                        else first_shift if shift.index == first_index
                        else second_shift if shift.index == second_index
                        else shift
                        for shift in solution.shifts
                    ])
                    topology = normalize_source_loads(instance, topology)
                    fingerprint = solution_fingerprint(topology)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    if counts["quantity_models"] >= max_quantity_models:
                        return results, TwoRecipientFunnel(**counts)
                    counts["quantity_models"] += 1
                    repaired, report = repair_quantities_with_highs(
                        instance,
                        topology,
                        score_days=horizon_days,
                        feasibility_days=horizon_days,
                        quantity_objective="max-delivered",
                        strict_inventory=True,
                    )
                    if report.status != "Optimal":
                        continue
                    counts["quantity_solved"] += 1
                    if assess_atomic_repair(instance, solution, repaired).accepted:
                        counts["strict_improvements"] += 1
                        results.append(repaired)
    return results, TwoRecipientFunnel(**counts)


def _single_operation_recipient_variants(
    instance: Instance,
    solution: Solution,
    donor: Shift,
    moved,
    arrival_radius: int,
) -> list[tuple[int, Shift]]:
    customer = instance.customer_by_point[moved.point]
    variants: list[tuple[int, Shift]] = []
    for recipient in solution.shifts:
        if recipient.index == donor.index:
            continue
        if abs(recipient.start - moved.arrival) > arrival_radius:
            continue
        if recipient.trailer not in customer.allowed_trailers:
            continue
        for insertion in range(len(recipient.operations) + 1):
            operations = list(recipient.operations)
            operations.insert(insertion, moved)
            timed = try_optimize_shift_times(
                instance,
                replace(recipient, operations=tuple(operations)),
                latest_end=_latest_end_for_unchanged_successors(
                    instance, solution, recipient,
                ),
            )
            if timed is not None:
                variants.append((recipient.index, timed))
    return variants


def _retime_with_segment_reorder(
    instance: Instance,
    solution: Solution,
    shift: Shift,
) -> Shift | None:
    latest_end = _latest_end_for_unchanged_successors(
        instance, solution, shift,
    )
    direct = try_optimize_shift_times(
        instance, shift, latest_end=latest_end,
    )
    if direct is not None:
        return direct
    for start, end in _customer_segments(instance, shift):
        segment = shift.operations[start:end]
        if len(segment) < 2 or len(segment) > 6:
            continue
        for reordered in permutations(segment):
            if reordered == segment:
                continue
            timed = try_optimize_shift_times(
                instance,
                replace(
                    shift,
                    operations=(
                        shift.operations[:start]
                        + reordered
                        + shift.operations[end:]
                    ),
                ),
                latest_end=latest_end,
            )
            if timed is not None:
                return timed
    return None


def cross_shift_ejection_candidates(
    instance: Instance,
    solution: Solution,
    shift_indices: tuple[int, ...],
    *,
    arrival_radius: int = 1_440,
    max_quantity_models: int = 128,
    max_fragment_operations: int = 3,
    allow_neutral_bridge: bool = False,
) -> tuple[list[Solution], EjectionFunnel]:
    """Move a customer operation between routes, then repair material flow."""
    by_index = {shift.index: shift for shift in solution.shifts}
    results: list[Solution] = []
    seen: set[str] = set()
    enumerated = donor_ok = recipient_ok = models = solved = improved = 0
    horizon_days = max(
        1, (instance.horizon * instance.unit + 1_439) // 1_440,
    )
    for donor_index in shift_indices:
        donor = by_index.get(donor_index)
        if donor is None:
            continue
        fragments = [
            (start, end)
            for length in range(1, min(max_fragment_operations, len(donor.operations) - 1) + 1)
            for start in range(len(donor.operations) - length, -1, -1)
            for end in (start + length,)
            if all(
                operation.point in instance.customer_by_point
                for operation in donor.operations[start:end]
            )
        ]
        # Work backward and prefer longer tail fragments: final operations are
        # the usual return-driving cause, and some routes remain impossible
        # after any one-customer removal.
        fragments.sort(key=lambda item: (item[1] != len(donor.operations), -(item[1] - item[0]), -item[0]))
        for operation_start, operation_end in fragments:
            moved = donor.operations[operation_start:operation_end]
            donor_operations = (
                donor.operations[:operation_start]
                + donor.operations[operation_end:]
            )
            if not donor_operations:
                continue
            repaired_donor = _retime_with_segment_reorder(
                instance,
                solution,
                replace(donor, operations=donor_operations),
            )
            if repaired_donor is None:
                continue
            donor_ok += 1
            for recipient in solution.shifts:
                if recipient.index == donor.index:
                    continue
                if abs(recipient.start - moved[0].arrival) > arrival_radius:
                    continue
                if any(
                    recipient.trailer
                    not in instance.customer_by_point[operation.point].allowed_trailers
                    for operation in moved
                ):
                    continue
                if recipient.trailer not in instance.drivers[recipient.driver].trailer_ids:
                    continue
                for insertion in range(len(recipient.operations) + 1):
                    enumerated += 1
                    operations = list(recipient.operations)
                    operations[insertion:insertion] = moved
                    repaired_recipient = try_optimize_shift_times(
                        instance,
                        replace(recipient, operations=tuple(operations)),
                        latest_end=_latest_end_for_unchanged_successors(
                            instance, solution, recipient,
                        ),
                    )
                    if repaired_recipient is None:
                        continue
                    recipient_ok += 1
                    topology = _reindex([
                        repaired_donor if shift.index == donor.index
                        else repaired_recipient if shift.index == recipient.index
                        else shift
                        for shift in solution.shifts
                    ])
                    topology = normalize_source_loads(instance, topology)
                    fingerprint = solution_fingerprint(topology)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    if models >= max_quantity_models:
                        return results, EjectionFunnel(
                            enumerated, donor_ok, recipient_ok, models, solved,
                            improved,
                        )
                    models += 1
                    repaired, report = repair_quantities_with_highs(
                        instance,
                        topology,
                        score_days=horizon_days,
                        feasibility_days=horizon_days,
                        quantity_objective="max-delivered",
                        strict_inventory=True,
                    )
                    if report.status != "Optimal":
                        continue
                    solved += 1
                    decision = assess_atomic_repair(instance, solution, repaired)
                    if decision.accepted:
                        improved += 1
                        results.append(repaired)
                    elif (
                        allow_neutral_bridge
                        and decision.reason == "no_strict_improvement"
                        and decision.after.key() == decision.before.key()
                    ):
                        results.append(repaired)
    return results, EjectionFunnel(
        enumerated, donor_ok, recipient_ok, models, solved, improved,
    )


def segment_reorder_candidates(
    instance: Instance,
    solution: Solution,
    shift_indices: tuple[int, ...],
    *,
    max_segment_operations: int = 7,
) -> tuple[list[Solution], ReorderFunnel]:
    """Permute customer order within reload-delimited route segments."""
    by_index = {shift.index: shift for shift in solution.shifts}
    candidates: list[Solution] = []
    seen: set[str] = set()
    enumerated = timing_feasible = unique = improved = 0
    for shift_index in shift_indices:
        shift = by_index.get(shift_index)
        if shift is None:
            continue
        for start, end in _customer_segments(instance, shift):
            segment = shift.operations[start:end]
            if len(segment) < 2 or len(segment) > max_segment_operations:
                continue
            for reordered in permutations(segment):
                if reordered == segment:
                    continue
                enumerated += 1
                operations = shift.operations[:start] + reordered + shift.operations[end:]
                latest_end = _latest_end_for_unchanged_successors(
                    instance, solution, shift,
                )
                timed = try_optimize_shift_times(
                    instance,
                    replace(shift, operations=operations),
                    latest_end=latest_end,
                )
                if timed is None:
                    continue
                timing_feasible += 1
                candidate = _reindex([
                    timed if item.index == shift.index else item
                    for item in solution.shifts
                ])
                fingerprint = solution_fingerprint(candidate)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                unique += 1
                if assess_atomic_repair(instance, solution, candidate).accepted:
                    improved += 1
                    candidates.append(candidate)
    return candidates, ReorderFunnel(enumerated, timing_feasible, unique, improved)


def state_preserving_split_candidates(
    instance: Instance,
    solution: Solution,
    shift_indices: tuple[int, ...],
    *,
    max_candidates: int = 256,
) -> tuple[list[Solution], SplitFunnel]:
    """Split routes without changing trailer ownership or material flow.

    The original prefix keeps its driver/trailer. The suffix keeps the same
    trailer (and therefore its exact carried stock), but may use another
    compatible driver. Both pieces are scheduled inside unchanged resource
    predecessor/successor boundaries before the full transaction is replayed.
    """
    by_index = {shift.index: shift for shift in solution.shifts}
    derived = {item.shift.index: item for item in derive_solution(instance, solution)}
    candidates: list[Solution] = []
    seen: set[str] = set()
    counts = {
        "enumerated": 0,
        "prefix_feasible": 0,
        "resource_gaps": 0,
        "suffix_timing_feasible": 0,
        "unique_candidates": 0,
        "strict_improvements": 0,
        "rejected_physical": 0,
        "rejected_inventory": 0,
        "rejected_resource_timing": 0,
        "rejected_other": 0,
    }

    for shift_index in shift_indices:
        original = by_index.get(shift_index)
        if original is None:
            continue
        for cut in range(1, len(original.operations)):
            counts["enumerated"] += 1
            prefix = try_optimize_shift_times(
                instance,
                replace(original, operations=original.operations[:cut]),
            )
            if prefix is None:
                continue
            counts["prefix_feasible"] += 1
            prefix_end = derive_solution(
                instance, Solution((replace(prefix, index=0),)),
            )[0].end
            suffix_operations = original.operations[cut:]
            for driver in instance.drivers:
                if original.trailer not in driver.trailer_ids:
                    continue
                earliest, latest_end = _resource_gap(
                    instance,
                    solution,
                    derived,
                    original,
                    prefix_end,
                    driver.index,
                )
                if earliest >= latest_end:
                    continue
                counts["resource_gaps"] += 1
                for start in _start_candidates(
                    instance, suffix_operations[0].point, driver.index,
                    earliest, latest_end,
                ):
                    suffix = try_optimize_shift_times(
                        instance,
                        Shift(
                            index=max(by_index, default=-1) + 1,
                            driver=driver.index,
                            trailer=original.trailer,
                            start=start,
                            operations=suffix_operations,
                        ),
                        latest_end=latest_end,
                    )
                    if suffix is None:
                        continue
                    counts["suffix_timing_feasible"] += 1
                    rebuilt = [
                        prefix if shift.index == original.index else shift
                        for shift in solution.shifts
                    ]
                    rebuilt.append(suffix)
                    candidate = _reindex(rebuilt)
                    fingerprint = solution_fingerprint(candidate)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    counts["unique_candidates"] += 1
                    decision = assess_atomic_repair(instance, solution, candidate)
                    if decision.accepted:
                        counts["strict_improvements"] += 1
                        candidates.append(candidate)
                        if len(candidates) >= max_candidates:
                            return candidates, SplitFunnel(**counts)
                    elif any(
                        name in decision.reason
                        for name in (
                            "non_finite_values", "reference_errors",
                            "physical_errors", "negative_quantity_minutes",
                            "overfill_quantity_minutes",
                        )
                    ):
                        counts["rejected_physical"] += 1
                    elif "safety_deficit_quantity_minutes" in decision.reason:
                        counts["rejected_inventory"] += 1
                    elif "resource_timing_errors" in decision.reason:
                        counts["rejected_resource_timing"] += 1
                    else:
                        counts["rejected_other"] += 1
    return candidates, SplitFunnel(**counts)


def _resource_gap(
    instance: Instance,
    solution: Solution,
    derived: dict[int, object],
    original: Shift,
    prefix_end: int,
    suffix_driver: int,
) -> tuple[int, int]:
    """Find the unchanged resource gap containing the original shift."""
    driver = instance.drivers[suffix_driver]
    earliest = prefix_end
    latest_end = instance.latest_time
    for shift in solution.shifts:
        if shift.index == original.index:
            continue
        item = derived[shift.index]
        if shift.trailer == original.trailer:
            if item.end <= original.start:
                earliest = max(earliest, item.end)
            elif shift.start > original.start:
                latest_end = min(latest_end, shift.start)
        if shift.driver == suffix_driver:
            if item.end <= original.start:
                earliest = max(earliest, item.end + driver.min_inter_shift_duration)
            elif shift.start > original.start:
                latest_end = min(
                    latest_end, shift.start - driver.min_inter_shift_duration,
                )
    return earliest, latest_end


def _latest_end_for_unchanged_successors(
    instance: Instance,
    solution: Solution,
    original: Shift,
) -> int:
    bound = instance.latest_time
    driver = instance.drivers[original.driver]
    for shift in solution.shifts:
        if shift.index == original.index or shift.start <= original.start:
            continue
        if shift.driver == original.driver:
            bound = min(bound, shift.start - driver.min_inter_shift_duration)
        if shift.trailer == original.trailer:
            bound = min(bound, shift.start)
    return bound


def _customer_segments(instance: Instance, shift: Shift) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start = 0
    for index, operation in enumerate(shift.operations):
        if operation.point in instance.source_by_point:
            if index - start >= 2:
                result.append((start, index))
            start = index + 1
    if len(shift.operations) - start >= 2:
        result.append((start, len(shift.operations)))
    return result


def _start_candidates(
    instance: Instance,
    first_point: int,
    driver_index: int,
    earliest: int,
    latest_end: int,
) -> tuple[int, ...]:
    travel = instance.time_matrix[instance.base_index][first_point]
    starts = {earliest}
    for window in instance.drivers[driver_index].time_windows:
        starts.add(max(earliest, window.start))
    customer = instance.customer_by_point.get(first_point)
    if customer is not None:
        for window in customer.time_windows:
            starts.add(max(earliest, window.start - travel))
        for order in customer.orders:
            starts.add(max(earliest, order.earliest_time - travel))
    return tuple(sorted(start for start in starts if start < latest_end))


def _reindex(shifts: list[Shift]) -> Solution:
    ordered = sorted(shifts, key=lambda shift: (shift.start, shift.index))
    return Solution(tuple(
        replace(shift, index=index) for index, shift in enumerate(ordered)
    ))
