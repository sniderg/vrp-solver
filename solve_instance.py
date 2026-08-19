"""EXPERIMENTAL one-command cold-start pipeline (construction + Gurobi LP).

Status correction (2026-08-19): this script has produced NO officially valid
solution that survives re-verification.  Its only saved artifact (V2.12) is
rejected by the released checker (3 missed orders, 391 runouts), and on V2.14
every parameter-grid candidate yields an infeasible LP.  Known defects: the
order lower bound sits exactly on the exclusive satisfaction floor (missed
orders by construction), quantities are rounded *down* to 4 decimals below the
LP's own bounds (SHI16 rejections), the tank model diverges from the checker's
simulation, and the candidate loop swallows all exceptions.  Requires Gurobi.

Use `vrp-solver native-solve` + `vrp-solver verify-official` for results you
intend to report.

Usage:
    uv run python solve_instance.py <input_instance_xml> <output_solution_xml> [--no-verify]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from vrp_solver.fast.state import instance_days
from vrp_solver.model import Operation, Shift, Solution
from vrp_solver.official_verify import default_v2_archive
from vrp_solver.solver.cluster_greedy import construct_cluster_solution
from vrp_solver.xml_io import load_instance, save_solution


def maximize_quantities_for_solution(inst, sol, env: gp.Env) -> Solution | None:
    shifts = list(sol.shifts)
    model = gp.Model("Quantity_Maximizer", env=env)
    model.setParam("OutputFlag", 0)

    q_vars = {}
    obj_terms = []

    for s_idx, s in enumerate(shifts):
        for op_idx, op in enumerate(s.operations):
            if op.point == 1:
                t_cap = inst.trailers[s.trailer].capacity
                q_vars[(s_idx, op_idx)] = model.addVar(lb=-t_cap, ub=0.0, name=f"reload_{s_idx}_{op_idx}")
            else:
                c = inst.customer_by_point[op.point]
                if c.orders:
                    ord_match = None
                    for o in c.orders:
                        if o.earliest_time <= op.arrival <= o.latest_time:
                            ord_match = o
                            break
                    if ord_match is None:
                        ord_match = c.orders[0]
                    min_q = ord_match.quantity * (ord_match.quantity_flexibility / 100.0)
                    max_q = min(c.capacity, ord_match.quantity)
                    q_vars[(s_idx, op_idx)] = model.addVar(lb=min_q, ub=max_q, name=f"drop_{s_idx}_{op_idx}")
                else:
                    q_vars[(s_idx, op_idx)] = model.addVar(lb=c.min_operation_quantity, ub=c.capacity, name=f"drop_{s_idx}_{op_idx}")
                obj_terms.append(q_vars[(s_idx, op_idx)])

    model.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)

    for t in inst.trailers:
        curr_q = t.initial_quantity
        for s_idx, s in enumerate(shifts):
            if s.trailer == t.index:
                for op_idx, op in enumerate(s.operations):
                    if op.point == 1:
                        curr_q = curr_q - q_vars[(s_idx, op_idx)]
                        model.addConstr(curr_q <= t.capacity)
                    else:
                        model.addConstr(curr_q >= q_vars[(s_idx, op_idx)])
                        curr_q = curr_q - q_vars[(s_idx, op_idx)]

    EPS = 1e-4
    for c in inst.customers:
        if c.orders:
            continue
        num_steps = len(c.forecast)
        step_drops = [[] for _ in range(num_steps)]
        for s_idx, s in enumerate(shifts):
            for op_idx, op in enumerate(s.operations):
                if op.point == c.index:
                    st = min(op.arrival // 60, num_steps - 1)
                    step_drops[st].append(q_vars[(s_idx, op_idx)])

        curLevel = c.initial_tank_quantity
        for i in range(num_steps):
            drop_sum = gp.quicksum(step_drops[i]) if step_drops[i] else 0.0
            curLevel = curLevel - c.forecast[i] + drop_sum
            model.addConstr(curLevel >= EPS)
            model.addConstr(curLevel <= c.capacity - EPS)

    model.optimize()
    if model.status == GRB.OPTIMAL:
        opt_shifts = []
        for s_idx, s in enumerate(shifts):
            new_ops = []
            for op_idx, op in enumerate(s.operations):
                new_ops.append(Operation(point=op.point, arrival=op.arrival, quantity=round(q_vars[(s_idx, op_idx)].X, 4)))
            opt_shifts.append(Shift(index=s.index, driver=s.driver, trailer=s.trailer, start=s.start, operations=tuple(new_ops)))
        return Solution(shifts=tuple(opt_shifts))
    return None


def eliminate_redundant_shifts(inst, sol: Solution, exe_path: str, inst_path: Path, temp_dir: Path, env: gp.Env) -> Solution:
    current_sol = sol
    improved = True
    temp_sol_file = temp_dir / "temp_elim.xml"
    while improved:
        improved = False
        shifts = list(current_sol.shifts)
        for i in range(len(shifts)):
            cand_shifts = [s for idx, s in enumerate(shifts) if idx != i]
            cand_sol = Solution(shifts=tuple(cand_shifts))
            cand_opt = maximize_quantities_for_solution(inst, cand_sol, env)
            if cand_opt is not None:
                save_solution(cand_opt, str(temp_sol_file))
                p = subprocess.run([exe_path, str(inst_path), str(temp_sol_file)], capture_output=True, text=True)
                if "THIS OUTPUT IS VALID" in p.stdout:
                    current_sol = cand_opt
                    improved = True
                    break
    return current_sol


def run_official_checker(inst_path: Path, sol_path: Path, archive_path: Path) -> tuple[bool, float | None, str]:
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(td)
        exe = str(Path(td) / "Challenge_Roadef_EURO_Checker_V2/bin/Release/IRP_Roadef_Challenge_Checker.exe")
        p = subprocess.run([exe, str(inst_path), str(sol_path)], capture_output=True, text=True)
        is_valid = "THIS OUTPUT IS VALID" in p.stdout
        lr = None
        for line in p.stdout.splitlines():
            if "LOGISTIC" in line:
                lr = float(line.split(":")[-1].strip())
        return is_valid, lr, p.stdout


def solve(instance_path: Path, output_solution_path: Path, verify: bool = True) -> int:
    t0 = time.time()
    print(f"Loading instance from: {instance_path}")
    inst = load_instance(str(instance_path))
    days = instance_days(inst)
    print(f"Instance loaded: {len(inst.customers)} customers, {len(inst.drivers)} drivers, horizon: {days} days.")

    project_root = Path(__file__).resolve().parent
    archive = default_v2_archive(project_root)

    best_solution = None
    best_lr = None

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(td)
        exe = str(Path(td) / "Challenge_Roadef_EURO_Checker_V2/bin/Release/IRP_Roadef_Challenge_Checker.exe")

        with gp.Env() as env:
            candidates = [
                (4.0, 16, 12),
                (3.5, 14, 10),
                (4.5, 18, 12),
                (3.0, 12, 8),
            ]
            for doi, n_size, g_fill in candidates:
                try:
                    cand_sol, rep = construct_cluster_solution(
                        inst,
                        safety_buffer=0.015,
                        neighborhood_size=n_size,
                        score_cutoff_minute=days * 1440,
                        global_pressure_fill=g_fill,
                        fill_doi_days=doi,
                        first_stop_targeted=True,
                        tie_break_seed=0,
                    )
                    cand_opt = maximize_quantities_for_solution(inst, cand_sol, env)
                    if cand_opt is not None:
                        temp_p = Path(td) / "cand.xml"
                        save_solution(cand_opt, str(temp_p))
                        p = subprocess.run([exe, str(instance_path), str(temp_p)], capture_output=True, text=True)
                        if "THIS OUTPUT IS VALID" in p.stdout:
                            final_sol = eliminate_redundant_shifts(inst, cand_opt, exe, instance_path, Path(td), env)
                            save_solution(final_sol, str(output_solution_path))
                            p2 = subprocess.run([exe, str(instance_path), str(output_solution_path)], capture_output=True, text=True)
                            for l in p2.stdout.splitlines():
                                if "LOGISTIC" in l:
                                    best_lr = float(l.split(":")[-1].strip())
                            best_solution = final_sol
                            print(f"Feasible solution constructed! Logistic Ratio = {best_lr:.6f}, Shifts = {len(final_sol.shifts)}")
                            break
                except Exception as e:
                    pass

    if best_solution is None:
        print("Failed to construct a valid solution with default multi-stop parameter grid.", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    print(f"Solution successfully written to: {output_solution_path} in {elapsed:.2f}s")

    if verify:
        print("\nVerifying solution with official ROADEF 2016 C++ checker...")
        is_valid, lr, raw_report = run_official_checker(instance_path, output_solution_path, archive)
        if is_valid:
            print("************************* THIS OUTPUT IS VALID ***********************")
            print(f"Official Logistic Ratio: {lr:.6f}")
            return 0
        else:
            print("Official Checker Verification FAILED!", file=sys.stderr)
            print(raw_report, file=sys.stderr)
            return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Cold-start ROADEF 2016 VRP Solver CLI")
    parser.add_argument("instance_xml", type=Path, help="Path to input instance XML (e.g. V2.12.xml or X1.xml)")
    parser.add_argument("solution_xml", type=Path, help="Path to output solution XML to write")
    parser.add_argument("--no-verify", action="store_true", help="Skip running the official C++ checker after solve")
    args = parser.parse_args()

    return solve(args.instance_xml, args.solution_xml, verify=not args.no_verify)


if __name__ == "__main__":
    sys.exit(main())
