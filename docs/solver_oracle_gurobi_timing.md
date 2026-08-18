# Solver.exe / Gurobi 13 timing observation

## Scope and provenance

This is a behavioral oracle observation, not a native solver result. The
original `Solver.exe` was run under Wine against the raw Set B instance with
the legacy `gurobi65.dll` name resolved to the Gurobi 13 Windows runtime.
The existing WLS licence was used. No native solution was supplied as a seed.

## Recreated V2.18 runs

The isolated Wine prefix was recreated with the official Gurobi 13.0.1
Windows runtime. A constructor-only run and a one-iteration run were made
with seed 1, one worker, and a 15-second limit:

```text
Solver.exe -p V2.18.xml -o v218.xml -s 1 -t 15 -iter 0 -j 1
Solver.exe -p V2.18.xml -o v218.xml -s 1 -t 15 -iter 1 -j 1
```

Observed output:

| Run | Constructor time | Total process time |
| --- | ---: | ---: |
| `-iter 0` | 6.308 s | 6.419 s |
| `-iter 1` | 6.418 s | 6.529 s |

Both runs selected the same internal configuration and constructor objective
as the archived V2.18 oracle run. The resulting schedules contain operations
across the 35-day horizon; this is not an early-days-first rolling commit.

## Gurobi logging result

With a `gurobi.env` containing `OutputFlag 1` and `LogFile`, the Gurobi log
contained only:

```text
Gurobi 13.0.1 (win64) logging started
Set parameter LogFile ...
Read parameters from file gurobi.env
WLS license ... registered
```

There were no `Optimize`, presolve, model-size, incumbent, or node-log lines
in either run. Therefore the demonstrated conclusion is:

> The V2.18 constructor completes the full-horizon plan without an observable
> Gurobi optimization call. Gurobi is initialized and licensed, but the
> constructor/local-search smoke tests do not show a MIP solve.

This does **not** prove that no rows or columns were added through the C API;
Gurobi's normal log does not report every `GRBaddvars` or `GRBaddconstrs`
call. Definitively distinguishing silent full-horizon model assembly from no
model assembly requires an instrumented compatibility DLL that timestamps
those API calls and `GRBoptimize`.

## Implication for the day-7–10 failure

The observed failure is not explained by Gurobi choosing an early-day versus
late-day optimization window in these runs. The legacy constructor emits a
complete 35-day schedule before local search. The native rolling solver should
therefore be analyzed as a topology/commitment problem: a protected prefix
and lookahead tail are needed because terminal quantity repair cannot create
missing route or resource topology.

## Reproduction artifacts

- Gurobi internal log: `/private/tmp/v218_iter1_gurobi.log`
- Licensed Wine prefix: `/private/var/folders/.../solver-oracle.5rAqFm`

The prefix is temporary and intentionally contains the local licence outside
the repository; licence files must not be committed.
