from __future__ import annotations
import os
import tempfile
import threading
import highspy


_GUROBI_ENV = None
_GUROBI_ENV_LOCK = threading.Lock()


def _shared_gurobi_env(gp):
    """Create one Gurobi environment per Python process.

    WLS single-use licenses allow several models in one environment, but reject
    repeatedly starting new environments.  Column generation solves many
    short MIPs, so environment reuse is required for that license class.
    """
    global _GUROBI_ENV
    with _GUROBI_ENV_LOCK:
        if _GUROBI_ENV is None:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            _GUROBI_ENV = env
        return _GUROBI_ENV

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
            env = _shared_gurobi_env(gp)
            model = gp.read(mps_path, env=env)
            try:
                model.Params.TimeLimit = time_limit
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

                    col_values = [v.X for v in sorted(vars_list, key=var_key)]
                return status, col_values, True
            finally:
                model.dispose()
    except gp.GurobiError as exc:
        print(f"⚠️ Gurobi Solver Warning: Gurobi failed with error: {exc}")
        print("💡 Automatically falling back to open-source HiGHS solver to complete the run!")
        return "Unsolved", None, False
