"""Break down the reported errors by rule code and compare with pressure targets."""
import sys
from collections import Counter
from pathlib import Path

from vrp_solver.xml_io import load_instance, load_solution
from vrp_solver.rules import validate_solution
from vrp_solver.solver.pressure import pressure_points

instance = load_instance(Path(sys.argv[1]))
solution = load_solution(Path(sys.argv[2]))
errors = validate_solution(instance, solution)
print(f"total_errors,{len(errors)}")
codes = Counter()
for e in errors:
    code = getattr(e, "code", None) or getattr(e, "rule", None) or str(e)[:40]
    codes[str(code)] += 1
for code, n in codes.most_common():
    print(f"error_code,{code},{n}")
p = pressure_points(instance, solution, end_day=instance.horizon)
print(f"pressure_points,{len(p)}")
sample = errors[:5]
for e in sample:
    print("sample_error", repr(e)[:200])
