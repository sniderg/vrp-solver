"""Emit a pure cold-start construction with the idle cadence cap, to XML.

Usage: emit_cadence_seed.py <instance.xml> <out.xml> [cap] [seed]
Native cold start: instance XML + seed only, no external candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrp_solver.cli import load_instance, save_solution
from vrp_solver.inventory import tank_aggregates
from vrp_solver.rules import validate_solution
from vrp_solver.solver import cluster_greedy as cg

instance_path, out_path = sys.argv[1], sys.argv[2]
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 180
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

instance = load_instance(instance_path)
policy = cg.derive_cluster_construction_policy(instance)
end_day = max(1, (instance.horizon * instance.unit + 1439) // 1440)
solution, report = cg.construct_cluster_solution(
    instance,
    safety_buffer=0.20,
    neighborhood_size=policy.neighborhood_size,
    score_cutoff_minute=end_day * 1440,
    global_pressure_fill=policy.global_pressure_fill,
    tie_break_seed=seed,
    max_idle_wait_minutes=cap,
)
errors = [v for v in validate_solution(instance, solution) if v.severity == "error"]
_, _, _, safety_qm = tank_aggregates(instance, solution)
save_solution(solution, out_path)
print(f"instance,{instance.name},cap,{cap},seed,{seed}")
print(f"wrote,{out_path}")
print(f"local_errors,{len(errors)},safety_qm,{safety_qm:.0f}")
print(f"shifts,{report.shifts},operations,{report.operations}")
for violation in errors[:10]:
    print(f"error,{violation.code},{violation.message}")
