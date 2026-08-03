"""Prototype the constructor-stage joint shift timing/resource assignment MIP."""
from __future__ import annotations

import argparse
from dataclasses import replace

import gurobipy as gp
from gurobipy import GRB

from vrp_solver.model import Solution
from vrp_solver.rules import derive_solution, validate_solution
from vrp_solver.solver.targeted_rescue import normalize_source_loads
from vrp_solver.xml_io import load_instance, load_solution, save_solution


def translation_bounds(instance, shift, derived):
    low = -shift.start
    high = instance.latest_time - derived.end
    for operation, derived_operation in zip(
        shift.operations, derived.operations,
    ):
        customer = instance.customer_by_point.get(operation.point)
        if customer is None:
            continue
        containing = [
            window
            for window in customer.time_windows
            if (
                window.start <= operation.arrival
                and derived_operation.departure <= window.end
            )
        ]
        if not containing:
            return 0, 0
        window = containing[0]
        low = max(low, window.start - operation.arrival)
        high = min(
            high, window.end - derived_operation.departure,
        )
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("solution")
    parser.add_argument("output")
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--max-dropped", type=int)
    parser.add_argument("--warm-start")
    parser.add_argument(
        "--max-delay", type=int,
        help="forbid a route from starting more than this many minutes late",
    )
    parser.add_argument(
        "--pin-point", type=int, action="append", default=[],
        help="keep routes serving this inventory point at or before their seed time",
    )
    args = parser.parse_args()

    instance = load_instance(args.instance)
    solution = load_solution(args.solution)
    derived = derive_solution(instance, solution)
    bounds = [
        translation_bounds(instance, shift, item)
        for shift, item in zip(solution.shifts, derived)
    ]
    durations = [
        item.end - shift.start
        for shift, item in zip(solution.shifts, derived)
    ]

    compatible: list[list[int]] = []
    trailer_driver: dict[int, int] = {}
    for driver in instance.drivers:
        for trailer in driver.trailer_ids:
            trailer_driver[trailer] = driver.index
    for shift in solution.shifts:
        allowed = []
        for trailer in instance.trailers:
            if all(
                (
                    operation.point not in instance.source_by_point
                    or trailer.index
                    in instance.source_by_point[
                        operation.point
                    ].allowed_trailers
                )
                and (
                    operation.point not in instance.customer_by_point
                    or trailer.index
                    in instance.customer_by_point[
                        operation.point
                    ].allowed_trailers
                )
                for operation in shift.operations
            ):
                allowed.append(trailer.index)
        compatible.append(allowed)

    model = gp.Model("joint_resource_schedule")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 8
    model.Params.MIPGap = 0
    starts = {}
    # A trailer can be used only with its owning driver, whose availability
    # may contain several independent duty windows.  The original prototype
    # assumed a single window and consequently rejected the Set-B instances
    # before building the resource model.  Keep the selected window with the
    # assignment so every route is explicitly placed in one legal duty span.
    assignments = {}
    deviations = {}
    dropped = {}
    for position, shift in enumerate(solution.shifts):
        low, high = bounds[position]
        pinned = any(
            operation.point in args.pin_point
            for operation in shift.operations
        )
        starts[position] = model.addVar(
            lb=shift.start + low,
            ub=min(
                shift.start + high,
                shift.start
                if pinned else shift.start + args.max_delay
                if args.max_delay is not None else shift.start + high,
            ),
            vtype=GRB.CONTINUOUS,
            name=f"start_{position}",
        )
        deviations[position] = model.addVar(
            lb=0, vtype=GRB.CONTINUOUS, name=f"deviation_{position}",
        )
        dropped[position] = model.addVar(
            vtype=GRB.BINARY, name=f"drop_{position}",
        )
        model.addConstr(
            deviations[position] >= starts[position] - shift.start,
        )
        model.addConstr(
            deviations[position] >= shift.start - starts[position],
        )
        variables = []
        for trailer in compatible[position]:
            driver = instance.drivers[trailer_driver[trailer]]
            for window_index, window in enumerate(driver.time_windows):
                variable = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"use_{position}_{trailer}_{window_index}",
                )
                assignments[position, trailer, window_index] = variable
                variables.append(variable)
                model.addGenConstrIndicator(
                    variable, True, starts[position] >= window.start,
                )
                model.addGenConstrIndicator(
                    variable,
                    True,
                    starts[position] + durations[position] <= window.end,
                )
        model.addConstr(
            gp.quicksum(variables) + dropped[position] == 1,
        )

    by_driver = {
        driver.index: tuple(driver.trailer_ids)
        for driver in instance.drivers
    }
    big_m = instance.latest_time * 2
    for left in range(len(solution.shifts)):
        for right in range(left + 1, len(solution.shifts)):
            for driver in instance.drivers:
                left_vars = [
                    variable
                    for (position, trailer, _), variable in assignments.items()
                    if position == left and trailer in by_driver[driver.index]
                ]
                right_vars = [
                    variable
                    for (position, trailer, _), variable in assignments.items()
                    if position == right and trailer in by_driver[driver.index]
                ]
                if not left_vars or not right_vars:
                    continue
                left_use = gp.quicksum(left_vars)
                right_use = gp.quicksum(right_vars)
                order = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"order_{left}_{right}_{driver.index}",
                )
                active_slack = big_m * (2 - left_use - right_use)
                model.addConstr(
                    starts[right]
                    >= starts[left]
                    + durations[left]
                    + driver.min_inter_shift_duration
                    - big_m * (1 - order)
                    - active_slack
                )
                model.addConstr(
                    starts[left]
                    >= starts[right]
                    + durations[right]
                    + driver.min_inter_shift_duration
                    - big_m * order
                    - active_slack
                )

    changed = gp.quicksum(
        variable
        for (position, trailer, _), variable in assignments.items()
        if trailer != solution.shifts[position].trailer
    )
    lost_delivery = gp.quicksum(
        dropped[position]
        * sum(
            max(0.0, operation.quantity)
            for operation in shift.operations
        )
        for position, shift in enumerate(solution.shifts)
    )
    if args.max_dropped is None:
        model.setObjectiveN(
            gp.quicksum(dropped.values()),
            index=0,
            priority=3,
            name="dropped_routes",
        )
        model.setObjectiveN(
            lost_delivery,
            index=1,
            priority=2,
            name="lost_delivery",
        )
        model.setObjectiveN(
            1000 * changed + gp.quicksum(deviations.values()),
            index=2,
            priority=1,
            name="assignment_disruption",
        )
    else:
        model.addConstr(
            gp.quicksum(dropped.values()) <= args.max_dropped,
            name="maximum_dropped_routes",
        )
        model.setObjective(
            1000 * lost_delivery
            + changed
            + 0.001 * gp.quicksum(deviations.values()),
            GRB.MINIMIZE,
        )

    if args.warm_start:
        warm = load_solution(args.warm_start)
        warm_position = 0
        for position, shift in enumerate(solution.shifts):
            key = tuple(
                operation.point for operation in shift.operations
            )
            warm_shift = None
            if warm_position < len(warm.shifts):
                next_warm = warm.shifts[warm_position]
                next_key = tuple(
                    operation.point
                    for operation in next_warm.operations
                )
                if next_key == key:
                    warm_shift = next_warm
                    warm_position += 1
            dropped[position].Start = 1.0 if warm_shift is None else 0.0
            starts[position].Start = (
                shift.start if warm_shift is None else warm_shift.start
            )
            for trailer in compatible[position]:
                driver = instance.drivers[trailer_driver[trailer]]
                for window_index, window in enumerate(driver.time_windows):
                    assignments[position, trailer, window_index].Start = float(
                        warm_shift is not None
                        and trailer == warm_shift.trailer
                        and window.start <= warm_shift.start
                        and warm_shift.start + durations[position] <= window.end
                    )
    model.optimize()
    if model.SolCount == 0:
        raise SystemExit(f"no resource schedule: {model.Status}")

    shifts = []
    for position, shift in enumerate(solution.shifts):
        if dropped[position].X > 0.5:
            continue
        trailer = max(
            compatible[position],
            key=lambda item: sum(
                assignments[position, item, window_index].X
                for window_index in range(
                    len(instance.drivers[trailer_driver[item]].time_windows)
                )
            ),
        )
        driver = trailer_driver[trailer]
        start = int(round(starts[position].X))
        delta = start - shift.start
        shifts.append(
            replace(
                shift,
                driver=driver,
                trailer=trailer,
                start=start,
                operations=tuple(
                    replace(
                        operation,
                        arrival=operation.arrival + delta,
                    )
                    for operation in shift.operations
                ),
            )
        )
    repaired = normalize_source_loads(
        instance, Solution(tuple(shifts)),
    )
    violations = validate_solution(instance, repaired)
    changed_count = sum(
        variable.X > 0.5
        and trailer != solution.shifts[position].trailer
        for (position, trailer, _), variable in assignments.items()
    )
    print(
        "resource_schedule,"
        f"status,{model.Status},"
        f"errors,{sum(v.severity == 'error' for v in violations)},"
        f"driver_conflicts,{sum(v.code == 'DRI01' for v in violations)},"
        f"trailer_conflicts,{sum(v.code == 'TL01' for v in violations)},"
        f"dropped,{sum(variable.X > 0.5 for variable in dropped.values())},"
        f"changed,{changed_count}"
    )
    save_solution(repaired, args.output)


if __name__ == "__main__":
    main()
