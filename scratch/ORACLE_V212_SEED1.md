# Original Solver.exe constructor oracle: V2.12 seed 1

This directory contains `oracle_v212_seed1.xml`, produced from the original
S24 `Solver.exe` without supplying any solution seed or published result.

The binary was run under Wine with its original `gurobi65.dll` imports routed
through a name-only forwarding DLL to the corresponding Gurobi 13.0.1 C API.
The original EXE itself was not modified.

Constructor-only invocation:

```text
Solver.exe
  -p V2.12.xml
  -o oracle_v212_seed1.xml
  -s 1
  -t 120
  -iter 0
  -j 1
```

Original solver report:

```text
[init] time: 2.779, obj: 0.0250961651=61554.6/2452749.836146
```

Independent native `contest-score` verification:

```text
submitted_shifts,85
submitted_operations,625
scored_delivered_quantity,2452749.8361462657
scored_estimated_cost,61554.61472331607
feasible,True
feasibility_errors,0
feasibility_warnings,6
hard_violations,0
safety_kg_min,0.0
tank_safety_breach_steps,0
```

The six warnings are QS01 nominal-quantity warnings. Every affected call-in
order is above its permitted flexible minimum, so they are not feasibility
errors.
