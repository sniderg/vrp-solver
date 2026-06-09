from __future__ import annotations
import os
import tempfile
import highspy

def solve_with_gurobi_if_requested(
    highs: highspy.Highs,
    time_limit: float = 300.0,
) -> tuple[str, list[float] | None, bool]:
    """
    Checks if Gurobi is requested via environment variable ROADEF_SOLVER=gurobi.
    If so, writes the Highs model to a temporary MPS file, solves it with gurobipy,
    and returns (status, col_values, True).
    Otherwise, returns ("Unsolved", None, False).
    """
    solver_env = os.environ.get("ROADEF_SOLVER", "highs").lower()
    if solver_env != "gurobi":
        return "Unsolved", None, False

    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError("gurobipy is not installed but ROADEF_SOLVER=gurobi was requested.") from exc

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mps_path = os.path.join(tmpdir, "model.mps")
            highs.writeModel(mps_path)

            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("TimeLimit", time_limit)
            env.start()

            model = gp.read(mps_path, env=env)
            model.optimize()

            status_map = {
                gp.GRB.OPTIMAL: "Optimal",
                gp.GRB.INFEASIBLE: "Infeasible",
                gp.GRB.UNBOUNDED: "Unbounded",
                gp.GRB.TIME_LIMIT: "TimeLimit",
            }
            
            status = status_map.get(model.Status, "Unknown")
            
            col_values = None
            if model.SolCount > 0:
                vars_list = model.getVars()
                def var_key(v):
                    name = v.VarName
                    if (name.startswith("C") or name.startswith("c")) and name[1:].isdigit():
                        return (0, int(name[1:]))
                    return (1, name)
                vars_sorted = sorted(vars_list, key=var_key)
                col_values = [v.X for v in vars_sorted]
                
            return status, col_values, True
    except gp.GurobiError as exc:
        print(f"⚠️ Gurobi Solver Warning: Gurobi failed with error: {exc}")
        print("💡 Automatically falling back to open-source HiGHS solver to complete the run!")
        return "Unsolved", None, False
