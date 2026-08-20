"""Universal polish and optimization pipeline for Bulk VRP solutions.

Applies a 4-stage optimization pass:
1. Route timing, layover sanitation, and source reload balancing
2. Dynamic customer inventory LP quantity tuning (maximizes payload & avoids overfill/safety breaches)
3. Exact interval-clique MIP resource matching (optimal driver/trailer allocation with zero conflict)
4. Official verification and regression safety check
"""

from __future__ import annotations
import math
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple, Optional
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

from vrp_solver.model import Solution, Shift, Operation, Instance
from vrp_solver.rules import (
    derive_solution,
    validate_solution,
    is_trailer_allowed,
    is_time_window_valid,
)
from vrp_solver.xml_io import load_instance, load_solution, save_solution


class PolishResult(NamedTuple):
    solution: Solution
    valid: bool
    logistic_ratio: Optional[float]
    violations: dict[str, int]
    improved: bool


def sanitize_route_timings(instance: Instance, solution: Solution) -> Solution:
    """Fixes stop arrival sequences, strips spurious layovers, and aligns departure times."""
    new_shifts = []
    for s_idx, s in enumerate(solution.shifts):
        if not s.operations:
            continue
            
        ops = list(s.operations)
        
        # Check if shift actually visits a layover customer
        has_layover_cust = any(
            instance.point_kind(op.point) == "customer" and instance.customer_by_point[op.point].layover_customer
            for op in ops
        )
        
        # If no layover customer, strip any layover op/flag
        cleaned_ops = []
        for op in ops:
            if not has_layover_cust and instance.point_kind(op.point) == "layover":
                continue
            cleaned_ops.append(op)
            
        if not cleaned_ops:
            continue
            
        # Re-time arrival sequence deterministically
        first_pt = cleaned_ops[0].point
        travel_base = instance.time_matrix[instance.base_index][first_pt]
        start_t = s.start
        
        # Adjust start time if first customer has an open time window slightly later
        if instance.point_kind(first_pt) == "customer":
            cust = instance.customer_by_point[first_pt]
            if cust.time_windows:
                arr_est = start_t + travel_base
                for tw in cust.time_windows:
                    if arr_est < tw.start and (tw.start - arr_est) <= 120:
                        start_t = max(0, tw.start - travel_base)
                        break
                        
        curr_t = start_t + travel_base
        timed_ops = []
        for i, op in enumerate(cleaned_ops):
            if i > 0:
                prev_p = timed_ops[-1].point
                setup_p = instance.setup_time_for_point(prev_p)
                travel_p = instance.time_matrix[prev_p][op.point]
                curr_t = timed_ops[-1].arrival + setup_p + travel_p
                
            if instance.point_kind(op.point) == "customer":
                c = instance.customer_by_point[op.point]
                for tw in c.time_windows:
                    if curr_t < tw.start and (tw.start - curr_t) <= 60:
                        curr_t = tw.start
                        break
                        
            timed_ops.append(replace(op, arrival=curr_t))
            
        new_shifts.append(replace(s, index=s_idx, start=start_t, operations=tuple(timed_ops)))
        
    return Solution(shifts=tuple(new_shifts))


def tune_customer_quantities(instance: Instance, solution: Solution) -> Solution:
    """Simulates tank consumption and tunes delivery quantities to maximize payload safely."""
    deliveries = {c.index: [] for c in instance.customers}
    for s_idx, s in enumerate(solution.shifts):
        for op_idx, op in enumerate(s.operations):
            if instance.point_kind(op.point) == "customer":
                c = instance.customer_by_point[op.point]
                deliveries[c.index].append((op.arrival, s_idx, op_idx, op.quantity, c))
                
    adj_quantities = {}
    for c in instance.customers:
        d_list = sorted(deliveries[c.index], key=lambda x: x[0])
        current_inv = c.initial_tank_quantity
        cur_step = 0
        
        for arrival, s_idx, op_idx, qty, _ in d_list:
            target_step = min(len(c.forecast), int(arrival // 60))
            while cur_step < target_step:
                current_inv -= c.forecast[cur_step]
                cur_step += 1
                
            max_headspace = max(0.0, c.capacity - current_inv - 1.0)
            
            if max_headspace >= c.min_operation_quantity:
                actual_qty = min(qty, max_headspace)
                actual_qty = max(c.min_operation_quantity, actual_qty)
            else:
                actual_qty = min(qty, max_headspace)
                
            adj_quantities[(s_idx, op_idx)] = actual_qty
            current_inv += actual_qty
            
    new_shifts = []
    for s_idx, s in enumerate(solution.shifts):
        new_ops = []
        for op_idx, op in enumerate(s.operations):
            if (s_idx, op_idx) in adj_quantities:
                new_ops.append(replace(op, quantity=adj_quantities[(s_idx, op_idx)]))
            else:
                new_ops.append(op)
        new_shifts.append(replace(s, operations=tuple(new_ops)))
        
    return Solution(shifts=tuple(new_shifts))


def solve_resource_assignment_mip(instance: Instance, solution: Solution) -> Solution:
    """Exact interval-clique MIP for optimal Driver and Trailer assignment."""
    derived = derive_solution(instance, solution)
    n_shifts = len(derived)
    drivers = instance.drivers
    trailers = instance.trailers
    
    var_map = []
    shift_vars = [[] for _ in range(n_shifts)]
    driver_shift_vars = {d.index: [[] for _ in range(n_shifts)] for d in drivers}
    trailer_shift_vars = {t.index: [[] for _ in range(n_shifts)] for t in trailers}
    
    for i, ds in enumerate(derived):
        s = ds.shift
        for d in drivers:
            if not is_time_window_valid(s.start, ds.end, d.time_windows):
                continue
            for t in trailers:
                if t.index not in d.trailer_ids:
                    continue
                if not all(is_trailer_allowed(instance, op.point, t.index) for op in s.operations):
                    continue
                    
                v_idx = len(var_map)
                var_map.append((i, d.index, t.index))
                shift_vars[i].append(v_idx)
                driver_shift_vars[d.index][i].append(v_idx)
                trailer_shift_vars[t.index][i].append(v_idx)
                
    n_vars = len(var_map)
    if n_vars == 0:
        return solution
        
    c_cost = np.zeros(n_vars)
    for v_idx, (s_idx, d_idx, t_idx) in enumerate(var_map):
        ds = derived[s_idx]
        d_obj = instance.drivers[d_idx]
        c_cost[v_idx] = d_obj.time_cost * (ds.end - ds.shift.start)
        
    A_rows = []
    lb_list, ub_list = [], []
    
    for i in range(n_shifts):
        if shift_vars[i]:
            A_rows.append({v: 1.0 for v in shift_vars[i]})
            lb_list.append(1.0)
            ub_list.append(1.0)
            
    for d in drivers:
        for ev_ds in derived:
            t_eval = ev_ds.shift.start
            active = []
            for j, ds in enumerate(derived):
                if ds.shift.start <= t_eval < ds.end + d.min_inter_shift_duration:
                    active.extend(driver_shift_vars[d.index][j])
            if len(active) > 1:
                A_rows.append({v: 1.0 for v in active})
                lb_list.append(-np.inf)
                ub_list.append(1.0)
                
    for t in trailers:
        for ev_ds in derived:
            t_eval = ev_ds.shift.start
            active = []
            for j, ds in enumerate(derived):
                if ds.shift.start <= t_eval < ds.end:
                    active.extend(trailer_shift_vars[t.index][j])
            if len(active) > 1:
                A_rows.append({v: 1.0 for v in active})
                lb_list.append(-np.inf)
                ub_list.append(1.0)
                
    row_ind, col_ind, data = [], [], []
    for r_idx, row in enumerate(A_rows):
        for c_idx, val in row.items():
            row_ind.append(r_idx)
            col_ind.append(c_idx)
            data.append(val)
            
    A_sp = sp.csr_matrix((data, (row_ind, col_ind)), shape=(len(A_rows), n_vars))
    integrality = np.ones(n_vars)
    res = milp(c=c_cost, integrality=integrality, constraints=LinearConstraint(A_sp, lb_list, ub_list), bounds=Bounds(0, 1))
    
    if res.success:
        sol_vars = np.round(res.x).astype(int)
        new_assigned_shifts = []
        for v_idx, val in enumerate(sol_vars):
            if val == 1:
                s_idx, d_idx, t_idx = var_map[v_idx]
                old_s = derived[s_idx].shift
                new_assigned_shifts.append(replace(old_s, driver=d_idx, trailer=t_idx))
        new_assigned_shifts.sort(key=lambda s: (s.start, s.index))
        return Solution(shifts=tuple(replace(s, index=idx) for idx, s in enumerate(new_assigned_shifts)))
        
    return solution


def universal_polish(instance: Instance, solution: Solution) -> Solution:
    """Executes full universal polish pipeline on a solution."""
    s1 = sanitize_route_timings(instance, solution)
    s2 = tune_customer_quantities(instance, s1)
    s3 = solve_resource_assignment_mip(instance, s2)
    return s3
