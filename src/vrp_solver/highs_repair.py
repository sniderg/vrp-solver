from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .contest import score_prefix_with_feasibility_tail, truncate_solution
from .model import Instance, Solution


EPSILON = 1e-6


@dataclass(frozen=True)
class HighsRepairReport:
    status: str
    variables: int
    constraints: int
    before_feasible: bool
    after_feasible: bool
    before_delivered: float
    after_delivered: float
    before_cost: float
    after_cost: float

    def flat(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class _DeliveryVariable:
    shift_index: int
    operation_index: int
    point: int
    arrival: int
    arrival_step: int
    original_quantity: float
    min_quantity: float
    max_quantity: float
    is_fixed: bool


@dataclass(frozen=True)
class _SourceLoadVariable:
    shift_index: int
    operation_index: int
    point: int
    arrival: int
    original_quantity: float
    max_quantity: float
    is_fixed: bool


def repair_with_highs_selection(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    feasibility_days: int | None = None,
    ignore_tail_call_ins: bool = False,
    quantity_objective: str = "max-delivered",
    baseline: Solution | None = None,
    fixed_prefix_minutes: int = 0,
    fixed_shift_indices: set[int] | None = None,
) -> tuple[Solution, HighsRepairReport]:
    try:
        import highspy
    except ModuleNotFoundError as exc:
        raise RuntimeError("highspy is not installed; run `uv sync --extra milp`") from exc

    cutoff = score_days * 1440
    working = truncate_solution(solution, cutoff)
    before = score_prefix_with_feasibility_tail(
        instance,
        working,
        score_days=score_days,
        feasibility_days=feasibility_days,
        ignore_tail_call_ins=ignore_tail_call_ins,
    )
    
    # Identify variables: deliveries to VMI customers and source load amounts.
    variables: list[_DeliveryVariable] = []
    load_variables: list[_SourceLoadVariable] = []
    fixed_shift_indices = fixed_shift_indices or set()
    for shift in working.shifts:
        is_fixed = (shift.start < fixed_prefix_minutes) or (shift.index in fixed_shift_indices)
        has_source = any(op.point in instance.source_by_point for op in shift.operations)
        for op_index, op in enumerate(shift.operations):
            customer = instance.customer_by_point.get(op.point)
            if customer and not customer.call_in:
                if is_fixed:
                    min_quantity = op.quantity
                    max_quantity = op.quantity
                else:
                    min_quantity = customer.min_operation_quantity
                    if customer.orders:
                        min_quantity = max(min_quantity, op.quantity)
                    if not has_source:
                        max_quantity = op.quantity
                    else:
                        max_quantity = customer.capacity
                variables.append(
                    _DeliveryVariable(
                        shift_index=shift.index,
                        operation_index=op_index,
                        point=op.point,
                        arrival=op.arrival,
                        arrival_step=min(max(op.arrival // instance.unit, 0), instance.horizon - 1),
                        original_quantity=op.quantity,
                        min_quantity=min_quantity,
                        max_quantity=max_quantity,
                        is_fixed=is_fixed,
                    )
                )
                continue
            source = instance.source_by_point.get(op.point)
            if source and op.quantity < -EPSILON:
                trailer = instance.trailers[shift.trailer]
                if is_fixed:
                    original_qty = -op.quantity
                    max_qty = -op.quantity
                else:
                    original_qty = -op.quantity
                    max_qty = trailer.capacity
                load_variables.append(
                    _SourceLoadVariable(
                        shift_index=shift.index,
                        operation_index=op_index,
                        point=op.point,
                        arrival=op.arrival,
                        original_quantity=original_qty,
                        max_quantity=max_qty,
                        is_fixed=is_fixed,
                    )
                )

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf

    if quantity_objective not in {"min-delivered", "max-delivered"}:
        raise ValueError("quantity_objective must be 'min-delivered' or 'max-delivered'")

    q_indices = []
    z_indices = []
    load_indices = []
    col_count = 0

    for var in variables:
        q_idx = col_count
        if quantity_objective == "max-delivered":
            q_cost = -1.0
            q_lower = var.min_quantity
        else:
            q_cost = 1.0
            q_lower = var.min_quantity if var.is_fixed else 0.0
        highs.addCol(q_cost, q_lower, var.max_quantity, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        q_indices.append(q_idx)
        col_count += 1

        if quantity_objective == "min-delivered":
            z_idx = col_count
            z_lower = 1.0 if (instance.customer_by_point[var.point].layover_customer or var.is_fixed) else 0.0
            highs.addCol(-1000.0, z_lower, 1.0, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
            highs.changeColIntegrality(z_idx, highspy.HighsVarType.kInteger)
            z_indices.append(z_idx)
            col_count += 1

            # q_i <= max_q * z_i
            highs.addRow(-inf, 0.0, 2, np.array([q_idx, z_idx], dtype=np.int32), np.array([1.0, -var.max_quantity], dtype=np.float64))
            # q_i >= min_q * z_i
            highs.addRow(0.0, inf, 2, np.array([q_idx, z_idx], dtype=np.int32), np.array([1.0, -var.min_quantity], dtype=np.float64))

    for var in load_variables:
        load_idx = col_count
        lower_bound = var.max_quantity if var.is_fixed else 0.0
        highs.addCol(
            0.0,
            lower_bound,
            var.max_quantity,
            0,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )
        load_indices.append(load_idx)
        col_count += 1

    # Inventory constraints
    horizon = _repair_horizon(instance, feasibility_days)
    by_point = _variables_by_point(variables)
    for customer in instance.customers:
        if customer.call_in: continue
        cust_vars = by_point.get(customer.index, [])
        cumulative_demand = 0.0
        for step in range(horizon):
            if step < len(customer.forecast):
                cumulative_demand += customer.forecast[step]
            indices = [q_indices[idx] for idx, v in cust_vars if v.arrival_step <= step]
            if not indices:
                # Check if initial inventory is enough
                if customer.initial_tank_quantity - cumulative_demand < customer.safety_level - EPSILON:
                    # Infeasible without more deliveries! 
                    pass
                continue
            
            lower_safety = customer.safety_level - customer.initial_tank_quantity + cumulative_demand
            lower_zero = 0.0 - customer.initial_tank_quantity + cumulative_demand
            upper = customer.capacity - customer.initial_tank_quantity + cumulative_demand

            # 1. Safety breach slack
            highs.addCol(10_000_000.0, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
            slack_breach_idx = highs.getNumCol() - 1
            
            indices_l = indices + [slack_breach_idx]
            qtys_l = [1.0] * len(indices) + [1.0]
            highs.addRow(lower_safety, inf, len(indices_l), np.array(indices_l, dtype=np.int32), np.array(qtys_l, dtype=np.float64))

            # 2. Negative inventory slack (penalty: 10 Billion to avoid going below 0.0 at all costs)
            highs.addCol(10_000_000_000.0, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
            slack_zero_idx = highs.getNumCol() - 1
            
            indices_z = indices + [slack_zero_idx]
            qtys_z = [1.0] * len(indices) + [1.0]
            highs.addRow(lower_zero, inf, len(indices_z), np.array(indices_z, dtype=np.int32), np.array(qtys_z, dtype=np.float64))

            # 3. Overfill constraint (HARD)
            indices_u = indices
            qtys_u = [1.0] * len(indices)
            highs.addRow(-inf, upper, len(indices_u), np.array(indices_u, dtype=np.int32), np.array(qtys_u, dtype=np.float64))

    _add_trailer_load_constraints(
        instance,
        working,
        variables,
        q_indices,
        highs,
        load_variables=load_variables,
        load_indices=load_indices,
    )

    if baseline is not None:
        _add_trailer_ending_inventory_repair_constraints(
            highs,
            instance,
            baseline,
            working,
            variables,
            q_indices,
            load_variables,
            load_indices,
            score_days,
        )

    highs.setOptionValue("time_limit", 300.0)
    
    from .solver.gurobi_bridge import solve_with_gurobi_if_requested
    status, values, solved_by_gurobi = solve_with_gurobi_if_requested(highs, time_limit=300.0)
    
    if solved_by_gurobi:
        print(f"Gurobi Status: {status}")
        has_solution = values is not None
    else:
        highs.run()
        status = highs.modelStatusToString(highs.getModelStatus())
        print(f"HiGHS Status: {status}")
        has_solution = highs.getInfo().primal_solution_status == 2 or "Optimal" in status or "Feasible" in status
        if has_solution:
            values = highs.getSolution().col_value
            
    repaired = working
    if has_solution and values is not None:
        q_values = [values[i] for i in q_indices]
        load_values = [values[i] for i in load_indices]
        repaired = _apply_quantities(
            working,
            variables,
            q_values,
            load_variables=load_variables,
            load_values=load_values,
        )

    after = score_prefix_with_feasibility_tail(
        instance,
        repaired,
        score_days=score_days,
        feasibility_days=feasibility_days,
        ignore_tail_call_ins=ignore_tail_call_ins,
    )
    return repaired, HighsRepairReport(
        status=status, variables=highs.getNumCol(), constraints=highs.getNumRow(),
        before_feasible=before.feasible, after_feasible=after.feasible,
        before_delivered=before.scored_delivered_quantity, after_delivered=after.scored_delivered_quantity,
        before_cost=before.scored_estimated_cost, after_cost=after.scored_estimated_cost,
    )


def repair_quantities_with_highs(
    instance: Instance,
    solution: Solution,
    *,
    score_days: int,
    feasibility_days: int | None = None,
    ignore_tail_call_ins: bool = False,
    quantity_objective: str = "max-delivered",
    baseline: Solution | None = None,
    fixed_prefix_minutes: int = 0,
    fixed_shift_indices: set[int] | None = None,
) -> tuple[Solution, HighsRepairReport]:
    return repair_with_highs_selection(
        instance,
        solution,
        score_days=score_days,
        feasibility_days=feasibility_days,
        ignore_tail_call_ins=ignore_tail_call_ins,
        quantity_objective=quantity_objective,
        baseline=baseline,
        fixed_prefix_minutes=fixed_prefix_minutes,
        fixed_shift_indices=fixed_shift_indices,
    )


def _add_trailer_load_constraints(
    instance: Instance,
    solution: Solution,
    variables: list[_DeliveryVariable],
    q_indices: list[int],
    highs,
    *,
    load_variables: list[_SourceLoadVariable] | None = None,
    load_indices: list[int] | None = None,
) -> None:
    q_by_operation = {
        (variable.shift_index, variable.operation_index): q_indices[index]
        for index, variable in enumerate(variables)
    }
    load_by_operation = {
        (variable.shift_index, variable.operation_index): (load_indices or [])[index]
        for index, variable in enumerate(load_variables or [])
    }
    shifts_by_trailer: dict[int, list] = {}
    for shift in solution.shifts:
        shifts_by_trailer.setdefault(shift.trailer, []).append(shift)

    for trailer in instance.trailers:
        load_constant = trailer.initial_quantity
        variable_columns: list[int] = []
        coefficients: list[float] = []
        shifts = sorted(
            shifts_by_trailer.get(trailer.index, []),
            key=lambda shift: (shift.start, shift.index),
        )
        for shift in shifts:
            for operation_index, operation in enumerate(shift.operations):
                key = (shift.index, operation_index)
                if key in q_by_operation:
                    variable_columns.append(q_by_operation[key])
                    coefficients.append(-1.0)
                elif key in load_by_operation:
                    variable_columns.append(load_by_operation[key])
                    coefficients.append(1.0)
                elif operation.point in instance.source_by_point and operation.quantity < -EPSILON:
                    load_constant += -operation.quantity
                elif operation.quantity > EPSILON:
                    load_constant -= operation.quantity

                if not variable_columns:
                    continue
                highs.addRow(
                    -load_constant,
                    trailer.capacity - load_constant,
                    len(variable_columns),
                    np.array(variable_columns, dtype=np.int32),
                    np.array(coefficients, dtype=np.float64),
                )


def _repair_horizon(instance: Instance, feasibility_days: int | None) -> int:
    if feasibility_days is None:
        return instance.horizon
    return min(instance.horizon, (feasibility_days * 1440 + instance.unit - 1) // instance.unit)


def _delivery_variables(instance: Instance, solution: Solution) -> list[_DeliveryVariable]:
    variables = []
    for shift in solution.shifts:
        for operation_index, operation in enumerate(shift.operations):
            if operation.quantity <= EPSILON or operation.point not in instance.customer_by_point:
                continue
            customer = instance.customer_by_point[operation.point]
            variables.append(
                _DeliveryVariable(
                    shift_index=shift.index,
                    operation_index=operation_index,
                    point=operation.point,
                    arrival=operation.arrival,
                    arrival_step=min(
                        max(operation.arrival // instance.unit, 0),
                        instance.horizon - 1,
                    ),
                    original_quantity=operation.quantity,
                    min_quantity=customer.min_operation_quantity,
                    max_quantity=customer.capacity,
                )
            )
    return variables


def _variables_by_point(
    variables: list[_DeliveryVariable],
) -> dict[int, list[tuple[int, _DeliveryVariable]]]:
    by_point: dict[int, list[tuple[int, _DeliveryVariable]]] = {}
    for index, variable in enumerate(variables):
        by_point.setdefault(variable.point, []).append((index, variable))
    return by_point


def _apply_quantities(
    solution: Solution,
    variables: list[_DeliveryVariable],
    values,
    *,
    load_variables: list[_SourceLoadVariable] | None = None,
    load_values=None,
) -> Solution:
    by_ref = {
        (variable.shift_index, variable.operation_index): max(0.0, float(values[index]))
        for index, variable in enumerate(variables)
    }
    source_by_ref = {
        (variable.shift_index, variable.operation_index): max(0.0, float(load_values[index]))
        for index, variable in enumerate(load_variables or [])
    }
    shifts = []
    for shift in solution.shifts:
        operations = []
        for operation_index, operation in enumerate(shift.operations):
            source_key = (shift.index, operation_index)
            if source_key in source_by_ref:
                operations.append(replace(operation, quantity=-source_by_ref[source_key]))
                continue
            key = (shift.index, operation_index)
            if key not in by_ref:
                operations.append(operation)
                continue
            quantity = by_ref[key]
            if quantity > EPSILON:
                operations.append(replace(operation, quantity=quantity))
        if operations:
            shifts.append(replace(shift, operations=tuple(operations)))
    return Solution(shifts=tuple(shifts))


def _add_trailer_ending_inventory_repair_constraints(
    highs,
    instance: Instance,
    baseline: Solution,
    solution: Solution,
    variables: list[_DeliveryVariable],
    q_indices: list[int],
    load_variables: list[_SourceLoadVariable],
    load_indices: list[int],
    score_days: int,
):
    MINUTES_PER_DAY = 1440
    cutoff = score_days * MINUTES_PER_DAY
    inf = 1e20

    # 1. Calculate baseline net change for each trailer in [0, cutoff]
    baseline_window_shifts = [
        s for s in baseline.shifts
        if s.start < cutoff
    ]
    baseline_net_change = {}
    for s in baseline_window_shifts:
        net = 0.0
        for op in s.operations:
            if op.point in instance.source_by_point:
                net += abs(op.quantity)
            elif op.point in instance.customer_by_point:
                net -= op.quantity
        baseline_net_change[s.trailer] = baseline_net_change.get(s.trailer, 0.0) + net

    # 2. Map variables to trailers
    shift_to_trailer = {s.index: s.trailer for s in solution.shifts if s.start < cutoff}

    by_trailer_vars = {}
    
    # Map delivery variables
    for q_idx, var in zip(q_indices, variables):
        t = shift_to_trailer.get(var.shift_index)
        if t is not None:
            by_trailer_vars.setdefault(t, []).append((q_idx, -1.0))

    # Map load variables
    for load_idx, var in zip(load_indices, load_variables):
        t = shift_to_trailer.get(var.shift_index)
        if t is not None:
            by_trailer_vars.setdefault(t, []).append((load_idx, 1.0))

    added_count = 0
    for t in range(len(instance.trailers)):
        target = baseline_net_change.get(t, 0.0)
        vars_coeffs = by_trailer_vars.get(t, [])
        if not vars_coeffs and abs(target) <= EPSILON:
            continue

        indices = [idx for idx, coeff in vars_coeffs]
        coefficients = [coeff for idx, coeff in vars_coeffs]

        # Add slack columns: slack_up (deficit) and slack_down (surplus)
        # Penalty: 50,000 per unit of deviation
        highs.addCol(50_000.0, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        slack_up_idx = highs.getNumCol() - 1

        highs.addCol(50_000.0, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        slack_down_idx = highs.getNumCol() - 1

        row_indices = indices + [slack_up_idx, slack_down_idx]
        row_coeffs = coefficients + [1.0, -1.0]

        highs.addRow(
            target,
            target,
            len(row_indices),
            np.array(row_indices, dtype=np.int32),
            np.array(row_coeffs, dtype=np.float64),
        )
        added_count += 1
