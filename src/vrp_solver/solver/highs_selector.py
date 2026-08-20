from __future__ import annotations

import numpy as np
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, List, Dict, Tuple, Set
import os
import time

from ..model import Instance, Solution, Shift, Operation
from ..inventory import project_customer_inventory, tank_events
from ..highs_time_opt import _feasible_operation_windows
from ..rules import derive_solution
from .gurobi_bridge import _shared_gurobi_env
from .ml_priors import MLRoutePriors

EPSILON = 1e-6


@dataclass(frozen=True)
class _QuantityVariable:
    shift_index: int
    operation_index: int
    point: int
    arrival: int
    arrival_step: int
    min_quantity: float
    max_quantity: float


@dataclass(frozen=True)
class SelectorConfig:
    solver: str | None = None
    time_limit: float = 300.0
    mip_gap: float | None = None
    threads: int | None = None
    mip_focus: int | None = None
    node_limit: int | None = None
    output: bool = False
    selector_phase: Literal["auto", "feasibility", "cost"] = "auto"
    mip_start_shift_indices: tuple[int, ...] = ()
    priority_shift_indices: tuple[int, ...] = ()
    priority_shift_bonus: float = 1_000_000.0
    # Diagnostic/repair aid: pin known-valid route columns to one.  This is
    # also useful for preserving an essential carried-load route while the
    # selector rebuilds the surrounding schedule.
    force_shift_indices: tuple[int, ...] = ()
    # Strict mode turns the route selector into a feasibility MIP: inventory
    # and order requirements are constraints, rather than penalised slacks.
    # This is deliberately opt-in because column generation normally benefits
    # from returning the least-infeasible incumbent while its pool is sparse.
    strict_feasibility: bool = False
    strict_inventory: bool = False
    enforce_resource_conflicts: bool = True
    enforce_driver_conflicts: bool = True
    enforce_trailer_conflicts: bool = True
    driver_day_capacity: int | None = None
    flexible_driver_conflicts: bool = False
    # Restrict hard safety constraints to a retailer-focused neighbourhood.
    # This supports an IRP destroy/reinsert probe without pretending that a
    # sparse global column pool can repair every unrelated retailer at once.
    strict_inventory_points: tuple[int, ...] = ()
    # Alternative columns for a single incumbent route can be required to
    # choose exactly one. This is needed by retailer destroy/reinsert: every
    # unchanged route remains present unless one of its insertion variants is
    # selected in its place.
    exactly_one_shift_groups: tuple[tuple[int, ...], ...] = ()
    # Require inventory to remain nonnegative while allowing a separate
    # safety-level improvement phase.  This is the useful bridge between a
    # zero-runout incumbent and strict call-in repair.
    strict_nonnegative_inventory: bool = False
    strict_orders: bool = False
    # If supplied, only these (customer, order) pairs are hard.  This lets a
    # local repair model prove a specific call-in window feasible before its
    # column pool covers every unrelated order in the full horizon.
    strict_order_keys: tuple[tuple[int, int], ...] = ()
    # Enforce trailer stock after every selected source/customer operation,
    # rather than only at the end of the repair window.
    strict_trailer_inventory: bool = False
    # Gurobi-only audit artifacts.  `.lp` is human-readable and `.mps` is a
    # portable standard interchange format; `.sol` records the incumbent.
    model_export_path: Path | None = None
    solution_export_path: Path | None = None
    iis_export_path: Path | None = None


def select_shifts_with_highs(
    instance: Instance,
    prefix: Solution,
    candidates: List[Shift],
    *,
    start_day: int,
    end_day: int,
    variable_quantities: bool = False,
    pressure_pricing: bool = True,
    baseline: Solution | None = None,
    selector_config: SelectorConfig = SelectorConfig(),
    ml_priors: MLRoutePriors | None = None,
) -> Solution:
    solver = (selector_config.solver or os.environ.get("ROADEF_SOLVER", "highs")).lower()
    solved_by_gurobi = solver == "gurobi"
    if solved_by_gurobi:
        highs = _GurobiSelectorModel(selector_config)
        integer_type = "B"
        inf = 1e20
    else:
        try:
            import highspy
        except ModuleNotFoundError:
            raise RuntimeError("highspy is not installed")

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", selector_config.output)
        if selector_config.threads is not None and selector_config.threads > 1:
            # HiGHS >= 1.15: parallel MIP search is opt-in via "parallel".
            # Leave "threads" at 0 (auto): the process-global scheduler is
            # sized by the first HiGHS run in the process, and a later run
            # with a *different* explicit thread count fails with kError.
            # Auto always adapts to the existing scheduler.
            highs.setOptionValue("parallel", "on")
        inf = highspy.kHighsInf
        integer_type = highspy.HighsVarType.kInteger

    # Variables: x_s is binary, 1 if candidate shift s is selected
    x_indices = []
    
    ml_prizes_by_day: dict[int, dict[int, float]] = {}
    ml_prizes_flat: dict[int, float] = {}
    if ml_priors is not None:
        ml_prizes_by_day = ml_priors.predict_prizes_by_day(
            instance, prefix, start_day, end_day,
        )
        # Flat fallback: max prize across all days (used for candidate gen steering)
        for day_prizes in ml_prizes_by_day.values():
            for cid, prize in day_prizes.items():
                ml_prizes_flat[cid] = max(ml_prizes_flat.get(cid, 0.0), prize)

    pressure_by_customer = (
        _inventory_pressure_by_customer(instance, prefix, start_day, end_day)
        if pressure_pricing and not ml_prizes_flat
        else {}
    )
    for i, s in enumerate(candidates):
        travel_cost = _estimate_shift_cost(instance, s)
        served_customers = {
            op.point
            for op in s.operations
            if op.quantity > 0 and op.point in instance.customer_by_point
        }
        order_stops = sum(
            1
            for op in s.operations
            if op.quantity > 0
            and op.point in instance.customer_by_point
            and instance.customer_by_point[op.point].orders
        )
        
        if ml_prizes_by_day:
            phase = selector_config.selector_phase
            if phase == "auto":
                phase = "feasibility"
            shift_penalty = 1_000.0 if phase == "feasibility" else 10_000.0
            # Phase 2: use the prize for the day this shift operates on
            shift_day = min(s.start // 1440, end_day - 1)
            shift_day = max(shift_day, start_day)
            day_prizes = ml_prizes_by_day.get(shift_day, ml_prizes_flat)
            obj_coeff = (
                (0.05 * travel_cost if phase == "feasibility" else travel_cost)
                + shift_penalty
                - sum(day_prizes.get(c, 0.0) for c in served_customers)
            )
            if i in selector_config.priority_shift_indices:
                obj_coeff -= selector_config.priority_shift_bonus
        else:
            pressure_bonus = _candidate_pressure_bonus(
                instance,
                s,
                pressure_by_customer,
            )
            # Reward coverage and route density more than raw volume. Early top-up chains
            # often carry smaller quantities but are exactly what prevents later cliffs.
            # We add a 10,000 flat shift penalty to aggressively force shift consolidation,
            # and scale down the customer coverage rewards.
            phase = selector_config.selector_phase
            if phase == "auto":
                phase = "feasibility" if pressure_by_customer else "cost"
            if phase == "feasibility":
                obj_coeff = (
                    0.05 * travel_cost
                    + 1_000.0
                    - (2_500.0 * len(served_customers))
                    - (1_250.0 * max(0, len(served_customers) - 1))
                    - (2_500.0 * order_stops)
                    - (3.0 * pressure_bonus)
                )
                if i in selector_config.priority_shift_indices:
                    obj_coeff -= selector_config.priority_shift_bonus
            else:
                obj_coeff = (
                    travel_cost
                    + 10_000.0
                    - (1_000.0 * len(served_customers))
                    - (500.0 * max(0, len(served_customers) - 1))
                    - (1_000.0 * order_stops)
                    - pressure_bonus
                )
        
        highs.addCol(obj_coeff, 0.0, 1.0, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        idx = highs.getNumCol() - 1
        highs.changeColIntegrality(idx, integer_type)
        if i in selector_config.force_shift_indices:
            highs.changeColBounds(idx, 1.0, 1.0)
        if solved_by_gurobi and i in selector_config.mip_start_shift_indices:
            highs.set_start(idx, 1.0)
        x_indices.append(idx)

    for group in selector_config.exactly_one_shift_groups:
        if not group:
            continue
        highs.addRow(
            1.0,
            1.0,
            len(group),
            np.array([x_indices[index] for index in group], dtype=np.int32),
            np.ones(len(group), dtype=np.float64),
        )

    q_variables: list[_QuantityVariable] = []
    q_indices: list[int] = []
    if variable_quantities:
        q_variables, q_indices = _add_quantity_variables(highs, instance, candidates, x_indices)

    # 1. Resource Overlap Constraints
    intervals = _candidate_intervals(instance, candidates)
    if selector_config.enforce_resource_conflicts:
        if selector_config.enforce_driver_conflicts:
            if selector_config.flexible_driver_conflicts:
                _add_flexible_driver_conflict_constraints(
                    highs, instance, candidates, x_indices, intervals,
                )
            elif selector_config.driver_day_capacity is None:
                _add_driver_overlap_constraints(
                    highs, instance, candidates, x_indices, intervals,
                )
            if selector_config.driver_day_capacity is not None:
                _add_driver_day_capacity_constraints(
                    highs,
                    candidates,
                    x_indices,
                    selector_config.driver_day_capacity,
                )
        if selector_config.enforce_trailer_conflicts:
            _add_trailer_overlap_constraints(
                highs, candidates, x_indices, intervals,
            )
        _add_prefix_conflict_constraints(highs, instance, prefix, candidates, x_indices, intervals)
    # The legacy ending-net equality is a coarse proxy for trailer continuity.
    # Once the exact operation-by-operation state flow is enabled it becomes
    # harmful: it forbids spending genuine residual stock on a new call-in.
    if baseline is not None and not selector_config.strict_trailer_inventory:
        _add_trailer_ending_inventory_constraints(
            highs,
            instance,
            baseline,
            candidates,
            x_indices,
            q_variables,
            q_indices,
            start_day,
            end_day,
        )
    if selector_config.strict_trailer_inventory:
        _add_trailer_inventory_path_constraints(
            highs,
            instance,
            prefix,
            candidates,
            x_indices,
            q_variables,
            q_indices,
        )
    if variable_quantities:
        _add_shift_quantity_capacity_constraints(highs, instance, candidates, x_indices, q_variables, q_indices)

    # 2. Inventory Constraints with Slacks
    if variable_quantities:
        _add_inventory_constraints_with_slacks(
            highs,
            instance,
            prefix,
            q_variables,
            q_indices,
            start_day,
            end_day,
            strict=(selector_config.strict_feasibility or selector_config.strict_inventory or selector_config.strict_nonnegative_inventory),
            strict_nonnegative=selector_config.strict_nonnegative_inventory,
            strict_points=set(selector_config.strict_inventory_points),
        )
    else:
        _add_fixed_inventory_constraints_with_slacks(
            highs,
            instance,
            prefix,
            candidates,
            x_indices,
            start_day,
            end_day,
            strict=(selector_config.strict_feasibility or selector_config.strict_inventory or selector_config.strict_nonnegative_inventory),
            strict_nonnegative=selector_config.strict_nonnegative_inventory,
            strict_points=set(selector_config.strict_inventory_points),
        )
    _add_order_coverage_constraints(
        highs,
        instance,
        prefix,
        candidates,
        x_indices,
        end_day=end_day,
        q_variables=q_variables if variable_quantities else None,
        q_indices=q_indices if variable_quantities else None,
        strict=(selector_config.strict_feasibility or selector_config.strict_orders),
        strict_keys=selector_config.strict_order_keys,
    )

    if solved_by_gurobi:
        if selector_config.model_export_path is not None:
            highs.write_model(selector_config.model_export_path)
        status, values = highs.optimize()
        if values is not None and selector_config.solution_export_path is not None:
            highs.write_solution(selector_config.solution_export_path)
        elif status == "Infeasible" and selector_config.iis_export_path is not None:
            highs.write_iis(selector_config.iis_export_path)
        highs.close()
        print(f"Gurobi Status: {status}")
        if status == "GurobiError":
            print("Falling back to HiGHS selector for this window.")
            return select_shifts_with_highs(
                instance,
                prefix,
                candidates,
                start_day=start_day,
                end_day=end_day,
                variable_quantities=variable_quantities,
                pressure_pricing=pressure_pricing,
                baseline=baseline,
                selector_config=replace(selector_config, solver="highs"),
            )
        has_solution = values is not None
    else:
        highs.setOptionValue("time_limit", selector_config.time_limit)
        solve_start = time.perf_counter()
        from ..milp_monitor import timed_run
        timed_run(highs, "column_selector")
        solve_seconds = time.perf_counter() - solve_start
        status = highs.modelStatusToString(highs.getModelStatus())
        print(
            f"HiGHS Status: {status} "
            f"({solve_seconds:.1f}s, {highs.getNumCol()} cols, "
            f"{highs.getInfo().mip_node_count} nodes)"
        )
        has_solution = highs.getInfo().primal_solution_status == 2 or "Optimal" in status or "Feasible" in status
        if has_solution:
            values = highs.getSolution().col_value
    
    selected_shifts = list(prefix.shifts)
    if has_solution and values is not None:
        for i, val in enumerate(values[:len(candidates)]):
            if val > 0.5:
                if variable_quantities:
                    selected_shifts.append(_apply_quantities_to_shift(candidates[i], q_variables, q_indices, values))
                else:
                    selected_shifts.append(candidates[i])
                
    # Re-index shifts
    for i, s in enumerate(selected_shifts):
        selected_shifts[i] = replace(s, index=i)
        
    return Solution(shifts=tuple(selected_shifts))


class _GurobiSelectorModel:
    def __init__(self, config: SelectorConfig):
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise RuntimeError("gurobipy is not installed but ROADEF_SOLVER=gurobi was requested.") from exc

        self.gp = gp
        self.model = gp.Model("roadef_selector", env=_shared_gurobi_env(gp))
        self.model.Params.OutputFlag = 1 if config.output else 0
        self.model.Params.TimeLimit = config.time_limit
        if config.mip_gap is not None:
            self.model.Params.MIPGap = config.mip_gap
        if config.threads is not None:
            self.model.Params.Threads = config.threads
        if config.mip_focus is not None:
            self.model.Params.MIPFocus = config.mip_focus
        if config.node_limit is not None:
            self.model.Params.NodeLimit = config.node_limit
        self.vars = []

    def addCol(self, obj, lower, upper, _nnz, _indices, _coefficients):
        var = self.model.addVar(
            lb=lower,
            ub=upper,
            obj=obj,
            vtype=self.gp.GRB.CONTINUOUS,
            name=f"v_{len(self.vars)}",
        )
        var.Start = lower if lower == upper else 0.0
        self.vars.append(var)

    def getNumCol(self):
        return len(self.vars)

    def changeColIntegrality(self, index, _integrality):
        self.vars[index].VType = self.gp.GRB.BINARY

    def changeColBounds(self, index, lower, upper):
        self.vars[index].LB = lower
        self.vars[index].UB = upper
        if lower == upper:
            self.vars[index].Start = lower

    def set_start(self, index, value):
        self.vars[index].Start = value

    def addRow(self, lower, upper, nnz, indices, coefficients):
        expr = self.gp.LinExpr()
        for index, coefficient in zip(indices[:nnz], coefficients[:nnz]):
            expr.add(self.vars[int(index)], float(coefficient))
        inf = 1e19
        if lower > -inf and upper < inf and abs(lower - upper) <= EPSILON:
            self.model.addConstr(expr == float(lower))
        else:
            if lower > -inf:
                self.model.addConstr(expr >= float(lower))
            if upper < inf:
                self.model.addConstr(expr <= float(upper))

    def optimize(self) -> tuple[str, list[float] | None]:
        self.model.ModelSense = self.gp.GRB.MINIMIZE
        try:
            self.model.optimize()
        except self.gp.GurobiError as exc:
            print(f"Gurobi Solver Warning: {exc}")
            return "GurobiError", None
        status_map = {
            self.gp.GRB.OPTIMAL: "Optimal",
            self.gp.GRB.INFEASIBLE: "Infeasible",
            self.gp.GRB.UNBOUNDED: "Unbounded",
            self.gp.GRB.TIME_LIMIT: "TimeLimit",
            self.gp.GRB.NODE_LIMIT: "NodeLimit",
            self.gp.GRB.INTERRUPTED: "Interrupted",
        }
        status = status_map.get(self.model.Status, f"Status{self.model.Status}")
        if self.model.SolCount <= 0:
            return status, None
        return status, [var.X for var in self.vars]

    def close(self) -> None:
        self.model.dispose()

    def write_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.update()
        self.model.write(str(path))

    def write_solution(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.write(str(path))

    def write_iis(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.computeIIS()
        self.model.write(str(path))


def _inventory_pressure_by_customer(
    instance: Instance,
    prefix: Solution,
    start_day: int,
    end_day: int,
) -> dict[int, dict[int, float]]:
    start_step = max(0, start_day * 1440 // instance.unit)
    end_step = min(instance.horizon - 1, end_day * 1440 // instance.unit - 1)
    pressure: dict[int, dict[int, float]] = {}
    for event in tank_events(instance, prefix):
        if event.point not in instance.customer_by_point:
            continue
        if not (start_step <= event.step <= end_step):
            continue
        deficit = max(0.0, event.safety_level - event.ending_inventory)
        if deficit <= EPSILON:
            continue
        pressure.setdefault(event.point, {})[event.step] = deficit
    return pressure


def _candidate_pressure_bonus(
    instance: Instance,
    shift: Shift,
    pressure_by_customer: dict[int, dict[int, float]],
) -> float:
    bonus = 0.0
    for operation in shift.operations:
        if operation.quantity <= EPSILON or operation.point not in pressure_by_customer:
            continue
        arrival_step = min(max(operation.arrival // instance.unit, 0), instance.horizon - 1)
        future_deficits = [
            deficit
            for step, deficit in pressure_by_customer[operation.point].items()
            if step >= arrival_step
        ]
        if not future_deficits:
            continue
        breach_steps = len(future_deficits)
        deficit_area = sum(future_deficits)
        useful_quantity = min(operation.quantity, max(future_deficits))
        bonus += min(
            18_000.0,
            900.0
            + 25.0 * breach_steps
            + 0.0015 * deficit_area
            + 0.35 * useful_quantity,
        )
    return bonus


def _add_quantity_variables(highs, instance: Instance, candidates: List[Shift], x_indices):
    q_variables: list[_QuantityVariable] = []
    q_indices: list[int] = []
    inf = 1e20

    for shift_index, shift in enumerate(candidates):
        for operation_index, operation in enumerate(shift.operations):
            customer = instance.customer_by_point.get(operation.point)
            if customer is None or operation.quantity <= EPSILON:
                continue
            min_quantity = customer.min_operation_quantity
            max_quantity = min(customer.capacity, max(operation.quantity, min_quantity))
            if max_quantity <= EPSILON:
                continue

            # Inventory slacks decide how much volume is useful. Keeping q neutral
            # avoids selecting routes merely because they can carry more kilograms.
            highs.addCol(
                0.0,
                0.0,
                max_quantity,
                0,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )
            q_idx = highs.getNumCol() - 1
            q_indices.append(q_idx)
            q_variables.append(
                _QuantityVariable(
                    shift_index=shift_index,
                    operation_index=operation_index,
                    point=operation.point,
                    arrival=operation.arrival,
                    arrival_step=min(max(operation.arrival // instance.unit, 0), instance.horizon - 1),
                    min_quantity=min_quantity,
                    max_quantity=max_quantity,
                )
            )

            # q <= max_quantity * x
            highs.addRow(
                -inf,
                0.0,
                2,
                np.array([q_idx, x_indices[shift_index]], dtype=np.int32),
                np.array([1.0, -max_quantity], dtype=np.float64),
            )
            # q >= min_quantity * x
            highs.addRow(
                0.0,
                inf,
                2,
                np.array([q_idx, x_indices[shift_index]], dtype=np.int32),
                np.array([1.0, -min_quantity], dtype=np.float64),
            )

    return q_variables, q_indices

def _estimate_shift_cost(instance: Instance, shift: Shift) -> float:
    cost = 0.0
    prev = instance.base_index
    for op in shift.operations:
        cost += instance.time_matrix[prev][op.point]
        prev = op.point
    cost += instance.time_matrix[prev][instance.base_index]
    return cost

def _add_driver_overlap_constraints(highs, instance, candidates, x_indices, intervals):
    by_driver = {}
    for i, s in enumerate(candidates):
        by_driver.setdefault(s.driver, []).append(i)
    
    for d, indices in by_driver.items():
        driver = instance.drivers[d]
        for left_pos, i in enumerate(indices):
            start_i, end_i = intervals[i]
            end_i += driver.min_inter_shift_duration
            for j in indices[left_pos + 1:]:
                start_j, end_j = intervals[j]
                end_j += driver.min_inter_shift_duration
                if _intervals_overlap(start_i, end_i, start_j, end_j):
                    highs.addRow(
                        0.0,
                        1.0,
                        2,
                        np.array([x_indices[i], x_indices[j]], dtype=np.int32),
                        np.ones(2, dtype=np.float64),
                    )


def _add_driver_day_capacity_constraints(
    highs,
    candidates,
    x_indices,
    capacity: int,
) -> None:
    by_driver_day: dict[tuple[int, int], list[int]] = {}
    for index, shift in enumerate(candidates):
        by_driver_day.setdefault(
            (shift.driver, shift.start // 1440), [],
        ).append(x_indices[index])
    for columns in by_driver_day.values():
        if len(columns) <= capacity:
            continue
        highs.addRow(
            -1e20,
            float(capacity),
            len(columns),
            np.array(columns, dtype=np.int32),
            np.ones(len(columns), dtype=np.float64),
        )


def _add_flexible_driver_conflict_constraints(
    highs,
    instance,
    candidates,
    x_indices,
    intervals,
) -> None:
    bounds = {
        index: _uniform_shift_start_bounds(
            instance, shift, intervals[index][1],
        )
        for index, shift in enumerate(candidates)
    }
    by_driver: dict[int, list[int]] = {}
    for index, shift in enumerate(candidates):
        by_driver.setdefault(shift.driver, []).append(index)
    for driver_index, indices in by_driver.items():
        gap = instance.drivers[driver_index].min_inter_shift_duration
        for left_position, left in enumerate(indices):
            left_earliest, left_latest, left_duration = bounds[left]
            for right in indices[left_position + 1:]:
                right_earliest, right_latest, right_duration = bounds[right]
                if (
                    left_earliest + left_duration + gap <= right_latest
                    or right_earliest + right_duration + gap <= left_latest
                ):
                    continue
                highs.addRow(
                    0.0,
                    1.0,
                    2,
                    np.array(
                        [x_indices[left], x_indices[right]],
                        dtype=np.int32,
                    ),
                    np.ones(2, dtype=np.float64),
                )


def _uniform_shift_start_bounds(
    instance,
    shift,
    end: int,
) -> tuple[int, int, int]:
    lower_delta = -shift.start
    upper_delta = instance.horizon * instance.unit - end
    driver = instance.drivers[shift.driver]
    containing_driver_windows = [
        window
        for window in driver.time_windows
        if window.start <= shift.start and end <= window.end
    ]
    if containing_driver_windows:
        lower_delta = max(
            lower_delta,
            min(
                window.start - shift.start
                for window in containing_driver_windows
            ),
        )
        upper_delta = min(
            upper_delta,
            max(window.end - end for window in containing_driver_windows),
        )
    for operation in shift.operations:
        setup = instance.setup_time_for_point(operation.point)
        containing = [
            window
            for window in _feasible_operation_windows(instance, operation)
            if window.start <= operation.arrival
            and operation.arrival + setup <= window.end
        ]
        if containing:
            lower_delta = max(
                lower_delta,
                min(
                    window.start - operation.arrival
                    for window in containing
                ),
            )
            upper_delta = min(
                upper_delta,
                max(
                    window.end - setup - operation.arrival
                    for window in containing
                ),
            )
        customer = instance.customer_by_point.get(operation.point)
        if (
            customer is not None
            and not customer.call_in
            and operation.quantity > EPSILON
        ):
            step = min(
                max(operation.arrival // instance.unit, 0),
                instance.horizon - 1,
            )
            lower_delta = max(
                lower_delta,
                step * instance.unit - operation.arrival,
            )
            upper_delta = min(
                upper_delta,
                (step + 1) * instance.unit - 1 - operation.arrival,
            )
    return (
        shift.start + lower_delta,
        shift.start + max(lower_delta, upper_delta),
        end - shift.start,
    )


def _add_trailer_overlap_constraints(highs, candidates, x_indices, intervals):
    by_trailer = {}
    for i, s in enumerate(candidates):
        by_trailer.setdefault(s.trailer, []).append(i)
        
    for t, indices in by_trailer.items():
        for left_pos, i in enumerate(indices):
            start_i, end_i = intervals[i]
            for j in indices[left_pos + 1:]:
                start_j, end_j = intervals[j]
                if _intervals_overlap(start_i, end_i, start_j, end_j):
                    highs.addRow(
                        0.0,
                        1.0,
                        2,
                        np.array([x_indices[i], x_indices[j]], dtype=np.int32),
                        np.ones(2, dtype=np.float64),
                    )


def _candidate_intervals(instance: Instance, candidates: List[Shift]) -> dict[int, tuple[int, int]]:
    solution = Solution(shifts=tuple(replace(s, index=i) for i, s in enumerate(candidates)))
    return {
        derived.shift.index: (derived.shift.start, derived.end)
        for derived in derive_solution(instance, solution)
    }


def _intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _add_prefix_conflict_constraints(highs, instance, prefix, candidates, x_indices, intervals):
    prefix_derived = derive_solution(instance, prefix)
    prefix_driver = {}
    prefix_trailer = {}
    for derived in prefix_derived:
        prefix_driver.setdefault(derived.shift.driver, []).append(
            (derived.shift.start, derived.end)
        )
        prefix_trailer.setdefault(derived.shift.trailer, []).append(
            (derived.shift.start, derived.end)
        )

    for i, candidate in enumerate(candidates):
        start, end = intervals[i]
        driver = instance.drivers[candidate.driver]
        conflicts = False

        for prefix_start, prefix_end in prefix_driver.get(candidate.driver, []):
            left_end = end + driver.min_inter_shift_duration
            right_end = prefix_end + driver.min_inter_shift_duration
            if _intervals_overlap(start, left_end, prefix_start, right_end):
                conflicts = True
                break

        if not conflicts:
            for prefix_start, prefix_end in prefix_trailer.get(candidate.trailer, []):
                if _intervals_overlap(start, end, prefix_start, prefix_end):
                    conflicts = True
                    break

        if conflicts:
            highs.changeColBounds(x_indices[i], 0.0, 0.0)


def _add_shift_quantity_capacity_constraints(
    highs,
    instance: Instance,
    candidates: List[Shift],
    x_indices,
    q_variables: list[_QuantityVariable],
    q_indices,
):
    by_shift: dict[int, list[int]] = {}
    for variable_index, variable in enumerate(q_variables):
        by_shift.setdefault(variable.shift_index, []).append(variable_index)

    for shift_index, variable_indices in by_shift.items():
        shift = candidates[shift_index]
        trailer = instance.trailers[shift.trailer]
        sum_loads = sum(
            abs(op.quantity)
            for op in shift.operations
            if op.point in instance.source_by_point
        )
        max_total_delivery = trailer.capacity + sum_loads
        indices = [q_indices[index] for index in variable_indices] + [x_indices[shift_index]]
        coefficients = [1.0] * len(variable_indices) + [-max_total_delivery]
        highs.addRow(
            -1e20,
            0.0,
            len(indices),
            np.array(indices, dtype=np.int32),
            np.array(coefficients, dtype=np.float64),
        )


def _add_inventory_constraints_with_slacks(
    highs,
    instance,
    prefix,
    q_variables: list[_QuantityVariable],
    q_indices,
    start_day,
    end_day,
    *,
    strict: bool = False,
    strict_nonnegative: bool = False,
    strict_points: set[int] | None = None,
):
    deliveries_by_customer: dict[int, list[tuple[int, _QuantityVariable]]] = {}
    candidate_delivery_steps = {}
    for variable_index, variable in enumerate(q_variables):
        deliveries_by_customer.setdefault(variable.point, []).append((variable_index, variable))
        candidate_delivery_steps.setdefault(variable.point, set()).add(variable.arrival_step)
                
    events = tank_events(instance, prefix)
    events_by_cust_step = {(e.point, e.step): e for e in events}
    checkpoint_steps = _inventory_checkpoint_steps(
        instance,
        events,
        start_day,
        end_day,
        candidate_delivery_steps,
    )
    
    inf = 1e20
    
    for customer in instance.customers:
        if customer.call_in: continue
        customer_strict = strict or customer.index in (strict_points or set())
        
        for step in checkpoint_steps.get(customer.index, ()):
            step_time = (step + 1) * instance.unit
            
            baseline_event = events_by_cust_step.get((customer.index, step))
            if baseline_event is None: continue
            
            end_step = min(instance.horizon - 1, end_day * 1440 // instance.unit)
            target_level = customer.safety_level
            slack_penalty = 10_000_000.0
            if step == end_step and not customer_strict:
                target_level = customer.safety_level + 0.35 * (customer.capacity - customer.safety_level)
                slack_penalty = 100_000.0
            
            rhs_lower = (
                -baseline_event.ending_inventory
                if strict_nonnegative
                else target_level - baseline_event.ending_inventory
            )
            rhs_upper = customer.capacity - baseline_event.ending_inventory
            
            relevant = [
                variable_index
                for variable_index, variable in deliveries_by_customer.get(customer.index, ())
                # Inventory buckets use ``arrival // unit``; an arrival
                # exactly at step_time belongs to the *next* bucket.
                if variable.arrival < step_time
            ]

            indices = [q_indices[variable_index] for variable_index in relevant]
            qtys = [1.0] * len(relevant)
            if customer_strict:
                highs.addRow(rhs_lower, inf, len(indices), np.array(indices, dtype=np.int32), np.array(qtys, dtype=np.float64))
            else:
                highs.addCol(slack_penalty, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
                slack_breach_idx = highs.getNumCol() - 1
                highs.addRow(rhs_lower, inf, len(indices) + 1, np.array([*indices, slack_breach_idx], dtype=np.int32), np.array([*qtys, 1.0], dtype=np.float64))

            # Overfill constraint (HARD) - Physical impossibility, must be avoided at all costs.
            indices_u = [q_indices[variable_index] for variable_index in relevant]
            qtys_u = [1.0] * len(relevant)
            highs.addRow(-inf, rhs_upper, len(indices_u), np.array(indices_u, dtype=np.int32), np.array(qtys_u, dtype=np.float64))


def _add_fixed_inventory_constraints_with_slacks(
    highs,
    instance,
    prefix,
    candidates,
    x_indices,
    start_day,
    end_day,
    *,
    strict: bool = False,
    strict_nonnegative: bool = False,
    strict_points: set[int] | None = None,
):
    shift_cust_deliveries = {}
    candidate_delivery_steps = {}
    for i, shift in enumerate(candidates):
        for operation in shift.operations:
            if operation.point in instance.customer_by_point:
                shift_cust_deliveries.setdefault((i, operation.point), []).append(
                    (operation.quantity, operation.arrival)
                )
                if operation.quantity > EPSILON:
                    step = min(max(operation.arrival // instance.unit, 0), instance.horizon - 1)
                    candidate_delivery_steps.setdefault(operation.point, set()).add(step)

    events = tank_events(instance, prefix)
    events_by_cust_step = {(event.point, event.step): event for event in events}
    checkpoint_steps = _inventory_checkpoint_steps(
        instance,
        events,
        start_day,
        end_day,
        candidate_delivery_steps,
    )
    inf = 1e20

    for customer in instance.customers:
        if customer.call_in:
            continue
        customer_strict = strict or customer.index in (strict_points or set())

        for step in checkpoint_steps.get(customer.index, ()):
            step_time = (step + 1) * instance.unit
            baseline_event = events_by_cust_step.get((customer.index, step))
            if baseline_event is None:
                continue

            end_step = min(instance.horizon - 1, end_day * 1440 // instance.unit)
            target_level = customer.safety_level
            slack_penalty = 10_000_000.0
            if step == end_step and not customer_strict:
                target_level = customer.safety_level + 0.35 * (customer.capacity - customer.safety_level)
                slack_penalty = 100_000.0

            rhs_lower = (
                -baseline_event.ending_inventory
                if strict_nonnegative
                else target_level - baseline_event.ending_inventory
            )
            rhs_upper = customer.capacity - baseline_event.ending_inventory

            relevant_shifts = {}
            for (shift_index, customer_id), deliveries in shift_cust_deliveries.items():
                if customer_id != customer.index:
                    continue
                total_qty = sum(qty for qty, arrival in deliveries if arrival < step_time)
                if total_qty > EPSILON:
                    relevant_shifts[shift_index] = total_qty

            qtys = [relevant_shifts[shift_index] for shift_index in relevant_shifts]
            indices = [x_indices[shift_index] for shift_index in relevant_shifts]
            if customer_strict:
                highs.addRow(rhs_lower, inf, len(indices), np.array(indices, dtype=np.int32), np.array(qtys, dtype=np.float64))
            else:
                highs.addCol(slack_penalty, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
                slack_breach_idx = highs.getNumCol() - 1
                highs.addRow(rhs_lower, inf, len(indices) + 1, np.array([*indices, slack_breach_idx], dtype=np.int32), np.array([*qtys, 1.0], dtype=np.float64))

            # Overfill constraint (HARD)
            indices_u = [x_indices[shift_index] for shift_index in relevant_shifts]
            qtys_u = [relevant_shifts[shift_index] for shift_index in relevant_shifts]
            highs.addRow(-inf, rhs_upper, len(indices_u), np.array(indices_u, dtype=np.int32), np.array(qtys_u, dtype=np.float64))


def _add_order_coverage_constraints(
    highs,
    instance,
    prefix,
    candidates,
    x_indices,
    *,
    end_day: int,
    q_variables: list[_QuantityVariable] | None = None,
    q_indices: list[int] | None = None,
    strict: bool = False,
    strict_keys: tuple[tuple[int, int], ...] = (),
):
    prefix_deliveries: dict[tuple[int, int], float] = {}
    for shift in prefix.shifts:
        for operation in shift.operations:
            if operation.quantity <= EPSILON:
                continue
            customer = instance.customer_by_point.get(operation.point)
            if customer is None:
                continue
            for order_index, order in enumerate(customer.orders):
                if order.earliest_time <= operation.arrival <= order.latest_time:
                    prefix_deliveries[(operation.point, order_index)] = (
                        prefix_deliveries.get((operation.point, order_index), 0.0)
                        + operation.quantity
                    )

    # With fixed quantities an order is covered by a binary shift selection.
    # With variable quantities it must be covered by the *actual* quantity
    # variable.  Using the original XML quantity in the latter case lets a
    # selected call-in route be reduced below its order minimum while the MIP
    # still believes the order is satisfied.
    quantity_by_operation: dict[tuple[int, int], int] = {}
    if q_variables is not None and q_indices is not None:
        quantity_by_operation = {
            (variable.shift_index, variable.operation_index): q_indices[variable_index]
            for variable_index, variable in enumerate(q_variables)
        }
    candidate_deliveries: dict[tuple[int, int], dict[int, float]] = {}
    for candidate_index, shift in enumerate(candidates):
        for operation_index, operation in enumerate(shift.operations):
            if operation.quantity <= EPSILON or operation.point not in instance.customer_by_point:
                continue
            customer = instance.customer_by_point[operation.point]
            for order_index, order in enumerate(customer.orders):
                if order.earliest_time <= operation.arrival <= order.latest_time:
                    key = (customer.index, order_index)
                    by_shift = candidate_deliveries.setdefault(key, {})
                    variable_index = quantity_by_operation.get((candidate_index, operation_index))
                    if variable_index is None:
                        index = x_indices[candidate_index]
                        coefficient = operation.quantity
                    else:
                        index = variable_index
                        coefficient = 1.0
                    by_shift[index] = by_shift.get(index, 0.0) + coefficient

    inf = 1e20
    end_minute = end_day * 1440
    for customer in instance.customers:
        for order_index, order in enumerate(customer.orders):
            if order.latest_time > end_minute:
                continue
            required = order.min_quantity_to_satisfy - prefix_deliveries.get((customer.index, order_index), 0.0)
            if required <= EPSILON:
                continue
            relevant = candidate_deliveries.get((customer.index, order_index), {})
            indices = list(relevant)
            quantities = [relevant[index] for index in indices]
            is_strict = strict and (not strict_keys or (customer.index, order_index) in strict_keys)
            if is_strict:
                highs.addRow(required, inf, len(indices), np.array(indices, dtype=np.int32), np.array(quantities, dtype=np.float64))
            else:
                highs.addCol(25_000_000.0, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
                slack_idx = highs.getNumCol() - 1
                highs.addRow(required, inf, len(indices) + 1, np.array([*indices, slack_idx], dtype=np.int32), np.array([*quantities, 1.0], dtype=np.float64))


def _apply_quantities_to_shift(
    shift: Shift,
    q_variables: list[_QuantityVariable],
    q_indices,
    values,
) -> Shift:
    by_operation = {
        variable.operation_index: max(0.0, float(values[q_indices[index]]))
        for index, variable in enumerate(q_variables)
        if variable.shift_index == shift.index
    }
    operations = []
    for operation_index, operation in enumerate(shift.operations):
        if operation_index in by_operation:
            quantity = by_operation[operation_index]
            if quantity > EPSILON:
                operations.append(replace(operation, quantity=quantity))
            continue
        operations.append(operation)
    return replace(shift, operations=tuple(operations))


def _inventory_checkpoint_steps(
    instance,
    events,
    start_day,
    end_day,
    candidate_delivery_steps=None,
):
    start_step = max(0, start_day * 1440 // instance.unit)
    end_step = min(instance.horizon - 1, end_day * 1440 // instance.unit)
    interval_steps = max(1, 240 // instance.unit)
    by_customer = {}

    base_steps = set(range(start_step, end_step + 1, interval_steps))
    base_steps.add(end_step)
    for day in range(start_day, end_day):
        day_end = min(instance.horizon - 1, ((day + 1) * 1440) // instance.unit)
        if start_step <= day_end <= end_step:
            base_steps.add(day_end)

    for customer in instance.customers:
        if customer.call_in:
            continue
        by_customer[customer.index] = set(base_steps)

    for event in events:
        if event.safety_breach and start_step <= event.step <= end_step:
            by_customer.setdefault(event.point, set()).add(event.step)
            if event.step > start_step:
                by_customer[event.point].add(event.step - 1)

    for customer_id, steps in (candidate_delivery_steps or {}).items():
        by_customer.setdefault(customer_id, set()).update(
            step for step in steps if start_step <= step <= end_step
        )

    return {
        customer_id: tuple(sorted(steps))
        for customer_id, steps in by_customer.items()
    }
def rebalance_drivers(instance: Instance, solution: Solution, threshold_hrs: float = 12.0) -> Solution:
    """Attempts to swap shifts from overworked drivers to idle, compatible drivers."""
    new_shifts = list(solution.shifts)
    drivers = instance.drivers
    
    # 1. Calculate daily hours per driver
    driver_days: dict[int, dict[int, float]] = {} # driver_idx -> day -> hours
    for s in new_shifts:
        last_op = s.operations[-1]
        setup = instance.setup_time_for_point(last_op.point)
        duration = (last_op.arrival + setup - s.start) / 60.0
        day = s.start // 1440
        d_map = driver_days.setdefault(s.driver, {})
        d_map[day] = d_map.get(day, 0.0) + duration

    # 2. Identify overworked drivers and candidate shifts for swapping
    for d_idx, days in driver_days.items():
        for day, hours in days.items():
            if hours <= threshold_hrs:
                continue
                
            # This driver is overworked on this day. Try to offload a shift.
            overworked_shifts = [s for s in new_shifts if s.driver == d_idx and (s.start // 1440) == day]
            # Sort by duration descending to offload the biggest problem
            overworked_shifts.sort(key=lambda x: (x.operations[-1].arrival - x.start), reverse=True)
            
            for s_to_swap in overworked_shifts:
                # 3. Find a Shadow Driver
                # A shadow driver must be:
                # - Compatible with the trailer
                # - Idle during the shift window (plus rest buffer)
                # - Not overworked themselves
                
                shift_duration = (s_to_swap.operations[-1].arrival + instance.setup_time_for_point(s_to_swap.operations[-1].point) - s_to_swap.start)
                
                best_shadow = None
                for shadow_idx, shadow in enumerate(drivers):
                    if shadow_idx == d_idx:
                        continue
                    
                    # Check trailer compatibility
                    if s_to_swap.trailer not in shadow.trailer_ids:
                        continue
                        
                    # Check if idle during this shift window
                    shadow_shifts = [s for s in new_shifts if s.driver == shadow_idx]
                    conflict = False
                    for existing in shadow_shifts:
                        # Simple overlap check with 11h rest buffer (660 mins)
                        REST = 660
                        s_end = s_to_swap.operations[-1].arrival + instance.setup_time_for_point(s_to_swap.operations[-1].point)
                        e_end = existing.operations[-1].arrival + instance.setup_time_for_point(existing.operations[-1].point)
                        
                        if not (s_end + REST <= existing.start or e_end + REST <= s_to_swap.start):
                            conflict = True
                            break
                    
                    if conflict:
                        continue
                        
                    # Check if shadow would become overworked
                    shadow_day_hours = driver_days.get(shadow_idx, {}).get(day, 0.0)
                    if shadow_day_hours + (shift_duration / 60.0) > threshold_hrs:
                        continue
                        
                    best_shadow = shadow_idx
                    break
                
                if best_shadow is not None:
                    # Perform the swap!
                    shift_idx_in_list = next(i for i, s in enumerate(new_shifts) if s.index == s_to_swap.index)
                    new_shifts[shift_idx_in_list] = replace(s_to_swap, driver=best_shadow)
                    
                    # Update local tracking
                    driver_days[d_idx][day] -= (shift_duration / 60.0)
                    shadow_day_map = driver_days.setdefault(best_shadow, {})
                    shadow_day_map[day] = shadow_day_map.get(day, 0.0) + (shift_duration / 60.0)
                    
                    # If we are below threshold, stop offloading for this day
                    if driver_days[d_idx][day] <= threshold_hrs:
                        break

    return Solution(shifts=tuple(new_shifts))


def _add_trailer_ending_inventory_constraints(
    highs,
    instance: Instance,
    baseline: Solution,
    candidates: List[Shift],
    x_indices,
    q_variables: list[_QuantityVariable],
    q_indices,
    start_day: int,
    end_day: int,
):
    MINUTES_PER_DAY = 1440
    start = start_day * MINUTES_PER_DAY
    end = end_day * MINUTES_PER_DAY
    inf = 1e20

    # 1. Calculate baseline net change in the window for each trailer
    baseline_window_shifts = [
        s for s in baseline.shifts
        if start <= s.start < end
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

    # 2. Add net change constraint for each trailer in candidates
    by_trailer_candidates = {}
    for i, s in enumerate(candidates):
        by_trailer_candidates.setdefault(s.trailer, []).append((i, s))

    q_by_shift_op = {}
    for q_idx, q_var in zip(q_indices, q_variables):
        q_by_shift_op[(q_var.shift_index, q_var.operation_index)] = q_idx

    for t in range(len(instance.trailers)):
        target = baseline_net_change.get(t, 0.0)
        cands = by_trailer_candidates.get(t, [])
        if not cands and abs(target) <= EPSILON:
            continue

        indices = []
        coefficients = []

        for idx, s in cands:
            # Source loads for this shift
            source_load = sum(
                abs(op.quantity)
                for op in s.operations
                if op.point in instance.source_by_point
            )
            if source_load > EPSILON:
                indices.append(x_indices[idx])
                coefficients.append(source_load)

            # Deliveries
            for op_idx, op in enumerate(s.operations):
                if op.point in instance.customer_by_point and op.quantity > EPSILON:
                    q_idx = q_by_shift_op.get((idx, op_idx))
                    if q_idx is not None:
                        # Variable quantity
                        indices.append(q_idx)
                        coefficients.append(-1.0)
                    else:
                        # Fixed quantity
                        indices.append(x_indices[idx])
                        coefficients.append(-op.quantity)

        # Add slack columns: slack_up (deficit, net_change < target, so slack_up > 0)
        # and slack_down (surplus, net_change > target, so slack_down > 0)
        # Penalty: 50,000 per unit of deviation
        highs.addCol(50_000.0, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        slack_up_idx = highs.getNumCol() - 1

        highs.addCol(50_000.0, 0.0, inf, 0, np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        slack_down_idx = highs.getNumCol() - 1

        # Sum coefficients for duplicate indices to avoid HiGHS duplicate index error
        combined = {}
        for idx_val, coeff_val in zip(indices, coefficients):
            combined[idx_val] = combined.get(idx_val, 0.0) + coeff_val

        row_indices = list(combined.keys()) + [slack_up_idx, slack_down_idx]
        row_coeffs = list(combined.values()) + [1.0, -1.0]

        highs.addRow(
            target,
            target,
            len(row_indices),
            np.array(row_indices, dtype=np.int32),
            np.array(row_coeffs, dtype=np.float64),
        )


def _add_trailer_inventory_path_constraints(
    highs,
    instance: Instance,
    prefix: Solution,
    candidates: List[Shift],
    x_indices,
    q_variables: list[_QuantityVariable],
    q_indices,
):
    """Keep each selected trailer's stock in [0, capacity] throughout time.

    A route column is locally valid, but two non-overlapping columns can still
    be incompatible if the first leaves a different load from the load assumed
    by the second.  These cumulative constraints make the hand-off explicit.
    """
    q_by_shift_op = {
        (variable.shift_index, variable.operation_index): q_index
        for variable, q_index in zip(q_variables, q_indices)
    }
    fixed_events: dict[int, dict[int, float]] = {}
    for shift in prefix.shifts:
        trailer_events = fixed_events.setdefault(shift.trailer, {})
        for operation in shift.operations:
            trailer_events[operation.arrival] = (
                trailer_events.get(operation.arrival, 0.0) - operation.quantity
            )

    candidate_events: dict[int, dict[int, list[tuple[int, float]]]] = {}
    for candidate_index, shift in enumerate(candidates):
        trailer_events = candidate_events.setdefault(shift.trailer, {})
        for operation_index, operation in enumerate(shift.operations):
            variable_index = q_by_shift_op.get((candidate_index, operation_index))
            if variable_index is not None:
                contribution = -1.0
                index = variable_index
            else:
                contribution = -operation.quantity
                index = x_indices[candidate_index]
            trailer_events.setdefault(operation.arrival, []).append((index, contribution))

    inf = 1e20
    for trailer in instance.trailers:
        fixed_by_time = fixed_events.get(trailer.index, {})
        candidate_by_time = candidate_events.get(trailer.index, {})
        if not candidate_by_time:
            continue
        stock = trailer.initial_quantity
        coefficients: dict[int, float] = {}
        for minute in sorted(set(fixed_by_time) | set(candidate_by_time)):
            stock += fixed_by_time.get(minute, 0.0)
            for index, contribution in candidate_by_time.get(minute, ()):
                coefficients[index] = coefficients.get(index, 0.0) + contribution
            if minute not in candidate_by_time:
                continue
            nonzero = [(index, value) for index, value in coefficients.items() if abs(value) > EPSILON]
            if not nonzero:
                continue
            indices, values = zip(*nonzero)
            highs.addRow(
                -stock,
                trailer.capacity - stock,
                len(indices),
                np.array(indices, dtype=np.int32),
                np.array(values, dtype=np.float64),
            )
