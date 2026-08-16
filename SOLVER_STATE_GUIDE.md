# Open-Source ROADEF IRP Solver: State and User Guide

## Mission and validity contract

This project is building a deterministic, open-source replacement for the
proprietary ROADEF/EURO 2016 inventory-routing solver.

Production solver runs must satisfy three constraints:

1. No instance names, customer IDs, copied routes, or manual schedules in
   production policy.
2. A `native-cold-start` run receives only the instance and seed/configuration.
3. Local replay is necessary but insufficient. Published validity requires the
   released V2 checker to accept the exact output XML.

The released checker archive used by the project has SHA-256:

```text
fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a
```

## Current unified-engine status

The initial 2026-08-16 audit ran the connected engine on all 15 Set B instances with
a 1,800-second feasibility budget per instance. Runs stopped when local
feasibility was first reached; the exact XML was then submitted to the released
checker. That baseline established 6/15 officially valid native cold starts.
Subsequent general, raw-entry-point reruns closed four more instances, bringing
the current reproduced total to **10/15**, with zero local/official checker
disagreements.

| Instance | Days | Customers | Drivers / Trailers | Result | Time | Errors or official LR |
|---|---:|---:|---:|---|---:|---:|
| `V2.13` | 10 | 53 | 5 / 5 | Officially valid | 1 s | LR `0.077477` |
| `V2.14` | 35 | 53 | 5 / 5 | Officially valid | 95 s | LR `0.096214` |
| `V2.16.2` | 10 | 184 | 7 / 4 | Officially valid | 1,762 s | LR `0.029289` |
| `V2.24` | 10 | 32 | 5 / 6 | Officially valid | 3 s | LR `0.025699` |
| `V2.25` | 35 | 32 | 5 / 6 | Officially valid | 16 s | LR `0.033212` |
| `V2.26` | 35 | 32 | 5 / 6 | Officially valid | 16 s | LR `0.044782` |
| `V2.15` | 10 | 134 | 4 / 3 | Officially valid | 43 s | LR `0.068480` |
| `V2.20.2` | 35 | 184 | 7 / 4 | Officially valid | 40 s | LR `0.031979` |
| `V2.21.2` | 35 | 184 | 7 / 4 | Officially valid | 223 s | LR `0.034288` |
| `V2.19` | 35 | 53 | 5 / 5 | Officially valid | 17 s | LR `0.099943` |
| `V2.12` | 10 | 324 | 13 / 15 | Constructor failure | 1,848 s | 297 errors |
| `V2.17` | 35 | 134 | 4 / 3 | Constructor failure | 1,828 s | 7,348 errors |
| `V2.18` | 35 | 134 | 4 / 3 | Constructor failure | 1,806 s | 8,020 errors |
| `V2.22` | 21 | 324 | 13 / 15 | Constructor failure | 1,811 s | 6,706 errors |
| `V2.23` | 21 | 324 | 13 / 15 | Constructor failure | 1,811 s | 2,547 errors |

See [UNIFIED_SET_B_30MIN_AUDIT.md](UNIFIED_SET_B_30MIN_AUDIT.md) for the
methodology, violation vectors, interpretation, and next experiments.

Saved reference or repaired artifacts are valuable regression fixtures, but
they do not change the unified cold-start count unless the general entry point
reproduces them from raw instance XML.

## Connected solver architecture

The entry point is `src/vrp_solver/solver/unified_engine.py`.

```text
raw instance
  -> seeded cluster constructions
  -> fingerprinted diagnostic frontier (ViolationVector)
  -> hard full-horizon quantity repair
  -> bounded surgical topology search
       -> pressure blocks and internal/tail insertions
       -> route recombination and ejection
       -> multi-reload candidates
       -> connected resource-block retiming
       -> hard quantity repair and global replay
  -> local feasibility
  -> exact XML serialization
  -> released-checker verification
```

The engine defaults to feasibility-first behavior:

- stop seed generation on the first locally feasible construction;
- skip quantity and topology polishing when already feasible;
- stop repaired-frontier evaluation on feasibility;
- stop surgical search on the first feasible incumbent.

Set `stop_when_feasible=False` only for explicit LR-improvement experiments.
The `SolverResult.valid` field means **locally feasible**, while
`validation_status` makes that scope explicit. Official validity is established
only after saving and checking the exact XML.

### Dual solver harness contract

The production harness has two explicit phases with different acceptance
contracts:

1. **Native cold start:** raw instance XML only, optimize lexicographically for
   hard feasibility, serialize and run the released checker, and stop at the
   first officially valid solution.
2. **Verified warm start:** begin only from an officially valid incumbent and
   improve logistic ratio. Every accepted checkpoint must remain locally and
   officially valid; an invalid candidate can never replace the incumbent.

Keep phase telemetry and provenance separate. Cold-start qualification must
not depend on an existing solution artifact, and LR polishing must not consume
the feasibility budget or delay the first-valid stopping condition.

This is a permanent **dual-harness** requirement, not merely the current
benchmark protocol:

- Harness A owns raw-instance construction and terminates on the first released-
  checker-valid XML. Its score is time-to-validity and qualification coverage.
- Harness B accepts only that checker-verified XML as its warm start, preserves
  it as a rollback incumbent, and spends a separately declared budget improving
  LR. Its score is best valid LR versus elapsed improvement time.
- Candidate generation and exact-repair components may be shared, but run
  control, budgets, incumbent acceptance, artifacts, and reports must remain
  distinct. An `Unknown` backend result is not validity and cannot cross the
  handoff gate; the last officially valid incumbent always survives.

## Running the solver

### Python API

```python
from vrp_solver.solver.unified_engine import solve_cold_start
from vrp_solver.xml_io import load_instance, save_solution

instance_path = (
    "roadef_2016_data/set_B/Instances_B_V25-11042016/V2.13.xml"
)
instance = load_instance(instance_path)
result = solve_cold_start(
    instance,
    num_seeds=10,
    time_limit=1_800,
    stop_when_feasible=True,
)
save_solution(result.solution, "my_solution.xml")
print(result.validation_status, result.errors, result.runtime_seconds)
```

### Official verification

```bash
.venv/bin/vrp-solver verify-official \
  roadef_2016_data/set_B/Instances_B_V25-11042016/V2.13.xml \
  my_solution.xml
```

### Full Set B audit

The audit prioritizes the five high-confidence instances, uses three isolated
workers, stops on feasibility, checkpoints after every instance, and invokes
the official checker only for locally feasible outputs.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python3 -u scratch/audit_unified_set_b.py 1800 3
```

Generated audit XML and JSON are written under
`/private/tmp/vrp-unified-set-b-audit/` and do not overwrite repository
benchmark artifacts.

## Development priorities

1. Close `V2.20.2`, `V2.21.2`, and `V2.15`. They have only 3–12 safety errors
   and no physical, call-in, or resource violations.
2. Test Gurobi **serially** on the near-feasible group. The available token
   license cannot serve parallel tasks. Quantity repair and shift selection
   already support `ROADEF_SOLVER=gurobi`; connected timing remains HiGHS-only.
3. Treat `V2.19` as a moderate topology-density problem rather than a
   quantity-only repair.
4. Redesign construction for the collapse group using compatibility-aware
   resource reservation, denser multi-stop chains, proactive reloads, and
   periodic exact repair during construction.
5. Add a hard process-level audit timeout. The internal deadline can be
   exceeded by an in-flight exact model.

## OR implementation plan (August 2026)

The next development sequence is feasibility-first and stops as soon as the
released checker accepts the native cold-start XML:

1. Preserve the six known-valid instances as regression sentinels.
2. Preserve essential topology under bounded candidate pools. In particular,
   keep explicit reload columns rather than allowing cheaper direct routes to
   crowd them out.
3. Rank service requirements by compatible driver-trailer-source chain
   scarcity, deadline/first breach, and regret between feasible alternatives.
   Insert and reserve complete resource chains, not isolated customer visits.
4. For the near-feasible group (`V2.15`, `V2.20.2`, `V2.21.2`), use pressure
   donor exchange, internal/tail insertion, joint retiming, and hard quantity
   repair. Treat `V2.19` as a moderate multi-route topology deficiency.
5. For constructor-collapse instances (`V2.12`, `V2.17`, `V2.18`, `V2.22`,
   `V2.23`), add scarcity-aware regret construction, short rolling horizons,
   proactive reloads, and bounded ejection chains. More terminal LP time is
   not a substitute for missing topology.
6. Cache forward earliest and backward latest times on accepted resource
   chains so obviously impossible insertions are rejected before exact timing
   and quantity optimization.
7. Once the route pool is sufficiently diverse, periodically solve a
   restricted chain-column master. Use HiGHS normally and Gurobi only in a
   single serialized task after controlled backend comparison.

These items belong first to Harness A until all target instances reach official
validity. Once an instance crosses that gate, Harness B may reuse the route
pool, cached timing slack, ALNS operators, and restricted master to reduce LR,
but it must rank only officially valid incumbents and must never delay or alter
Harness A's stop-on-valid behavior.

The current audit regimes are based on full violation magnitude, not raw error
count: the near-feasible cases contain only `QS02`; collapse cases also contain
missed orders and/or physical negative-stock duration.

### Initial scarcity ablation

Static chain scarcity counts compatible driver-trailer-source triples. A first
ablation compared two lexicographic constructor orders under identical native
settings:

- **scarcity before breach:** all five fast sentinels (`V2.13`, `V2.14`,
  `V2.24`, `V2.25`, `V2.26`) remained officially valid; `V2.14` reached
  validity in about 7 seconds in that run;
- **breach before scarcity:** `V2.14` was locally invalid with 192 errors after
  the 180-second cap, so this ordering was rejected;
- a 180-second `V2.15` scarcity-first probe remained invalid with 273 `QS02`.
  This is a short mechanism probe, not directly comparable with the earlier
  1,800-second result of 12 `QS02`.

The retained ordering is scarcity-first. The next refinement should compute
regret over dynamic feasible insertion chains rather than further tuning a
static priority tuple.

## Universal portfolio qualification (August 16, 2026)

The production `solve_cold_start` entry point now evaluates deterministic
structural strategies before stochastic restarts:

1. urgency-band ordering with native feature-derived neighborhood breadth;
2. urgency-band ordering with a narrow three-neighbor topology;
3. urgency-band ordering with a dense four-neighbor topology and proactive
   reload threshold;
4. compatibility-scarcity ordering;
5. scarcity variants with seeded tie breaks, if budget remains.

These are general policies: none branches on an instance name, customer ID,
day, or imported route. Every run below started from raw instance XML, stopped
at the first locally feasible construction, serialized the returned solution,
and passed the released checker with SHA-256
`fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a`.

The current V2.17 protected-prefix investigation, including demonstrated
day-6 repair, failed approaches, and the exact next step, is preserved in
[`docs/v2_17_rolling_handoff.md`](docs/v2_17_rolling_handoff.md). It is a
mechanism checkpoint only; V2.17 remains outside the 10/15 qualified set.

| Instance | Universal strategy | Attempts | Runtime | Official status | LR |
|---|---|---:|---:|---|---:|
| `V2.20.2` | `urgency-band` | 1 | 39.89 s | valid | 0.031979 |
| `V2.19` | `urgency-band-narrow` | 2 | 16.72 s | valid | 0.099943 |
| `V2.15` | `urgency-band-dense-reload` + pressure reload substitution | 10 | 42.65 s | valid | 0.068480 |
| `V2.21.2` | `urgency-band-dense-reload` | 3 | 222.53 s | valid | 0.034288 |

This raises the presently reproduced universal cold-start set from **6/15 to
10/15**. The existing six were also regression-checked through the same updated
entry point: `V2.13`, `V2.14`, `V2.16.2`, `V2.24`, `V2.25`, and `V2.26` all
remained officially valid. LR differences are acceptable in phase one because
the cold-start contract stops at validity; phase two owns LR improvement from
the verified incumbent.

The two August 16 additions used only the raw instance XML and the production
`solve_cold_start` entry point with one search worker. `V2.15` required one
general pressure-band move: an early missing VMI visit replaced a redundant
reload slot, the connected five-shift resource block was retimed, and strict
full-horizon HiGHS quantity repair closed the inventory deficit. `V2.21.2`
was already feasible in the third construction basin and therefore stopped
without search or quantity repair. Both exact serialized XML files passed the
released V2 checker (return code 0, success sentinel present, failure sentinel
absent) with checker SHA-256
`fc5c4aec01b78fd10d6fd733ea6659baf676b34b6d3a0e93fab8751bbb5b494a`.

### Algorithm references

- Røpke and Pisinger, *An Adaptive Large Neighborhood Search Heuristic for the
  Pickup and Delivery Problem with Time Windows*: adaptive destroy/repair and
  regret insertion. <https://pubsonline.informs.org/doi/10.1287/trsc.1050.0135>
- Nagata and Kobayashi, *Large neighbourhood search with adaptive guided
  ejection search for the pickup and delivery problem with time windows*:
  guided removal/reinsertion and ejection-style feasibility recovery.
  <https://link.springer.com/article/10.1007/s13676-017-0115-6>
- Hà et al., *A new constraint programming model and a linear
  programming-based adaptive large neighborhood search for the vehicle
  routing problem with synchronization constraints*: LP feasibility checks
  and screening before expensive search. <https://arxiv.org/abs/1910.13513>
- Coelho, Cordeau, and Laporte, *Consistency in Multi-Vehicle Inventory-Routing*:
  short rolling-horizon IRP construction with exact quantity recourse.
  <https://www.cirrelt.ca/documentstravail/cirrelt-2012-37.pdf>
- Desaulniers, Rakke, and Coelho, *A Branch-Price-and-Cut Algorithm for the
  Inventory-Routing Problem*: route-column master and pricing structure.
  <https://ideas.repec.org/a/inm/ortrsc/v50y2016i3p1060-1076.html>
