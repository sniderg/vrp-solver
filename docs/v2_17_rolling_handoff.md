# V2.17 rolling-construction handoff

Status as of 2026-08-16: **not full-horizon valid and not officially
qualified**. The durable result is a general mechanism demonstration: native
construction plus the production pressure search can build a locally feasible
prefix through day 6. Day 7 and the remaining 35-day horizon are unresolved.

## Authority and provenance

- Input: raw `V2.17.xml` only (`native-cold-start`).
- Exact checker validity has **not** been reached for V2.17.
- No reference/oracle topology or customer-specific production branch was
  used. Customer IDs below describe diagnostic probes only.
- The authoritative Set B qualification count remains 10/15 in
  `SOLVER_STATE_GUIDE.md`.
- Gurobi was not used. HiGHS is the demonstrated path. Gurobi must remain
  serialized because the available licence cannot serve parallel tasks.

## What was established

### Full-horizon baseline is constructor collapse

An authoritative raw `solve_cold_start` attempt with a 1,800-second nominal
budget constructed for roughly nine minutes, ran three infeasible strict
full-horizon quantity repairs, and entered surgical search. An in-flight exact
operation overran the outer deadline; the process was killed after roughly 36
minutes without returning a valid incumbent. A dense diagnostic construction
still left 46 customers unscheduled and large safety/negative-stock areas.

This is missing route topology, not a quantity-only problem. Do not spend a
full run on terminal LP repair or assume a stronger backend will create the
missing routes.

### Short prefixes are feasible

With the general dense urgency policy (`neighborhood_size=8`, proactive reload
ratio `0.48`, urgency-band ordering, global pressure fill `16`, no terminal
preload), static cutoffs behaved as follows:

- day 3: zero local errors, 8 shifts;
- day 5: zero local errors, 12 shifts;
- day 7: 23 errors;
- day 10: 133 errors;
- day 14: 249 errors.

This proves the constructor can solve short windows and motivates protected
prefix continuation.

### Single-path rolling failed for a structural reason

Freezing a whole five-day plan consumes or commits the resource windows needed
immediately afterward. Daily or overlapping single-path continuations reached
day 4 or 5 and then failed. A small beam must preserve distinct lookahead
plans; deduplicating only by the currently empty/identical committed prefix
incorrectly discards future topology diversity.

A beam with distinct tail fingerprints retained exact-feasible committed
prefixes through day 5. Its best day-6 lookahead reduced the original collapse
to a pure safety-only vector:

```text
(nonfinite=0, reference=0, physical=0, missed_orders=0,
 negative_area=0, overfill_area=0, safety_area=3535.771843,
 resource=0, other=0)
```

The four replay errors were three final-bucket safety breaches at diagnostic
customer 17 and one at customer 45. Strict quantity repair alone was
infeasible because the necessary visit/resource topology was absent.

### General paired ejection closes day 6

The feasible diagnostic recipe was:

1. replace a flexible late singleton on the scarce compatible chain with the
   scarce pressured customer;
2. relocate the displaced flexible VMI stop into a much earlier compatible
   route;
3. jointly retime the connected resource block;
4. run strict HiGHS quantity repair and replay the committed horizon.

This pattern already existed in
`generate_pressure_substitution_ejections`, but three orchestration defects
hid it:

- the relocation radius excluded a feasible recipient about 4.7 days earlier;
- ordinary insertions/substitutions consumed the candidate-generation budget
  before paired ejections were evaluated;
- full-horizon violation vectors rejected valid rolling-prefix improvements
  because their speculative tail was necessarily incomplete.

The current code generalizes the fix:

- rolling-prefix resource state can be reconstructed from `initial_solution`;
- pressure ejections use an adaptive historical radius;
- ejections are generated first and yielded to exact recourse immediately;
- insertion/ejection enumeration observes the outer deadline between exact
  calls;
- search acceptance uses a violation vector truncated to the same committed
  horizon as its score.

On the saved diagnostic day-5 tail, production `surgical_search` with
`first_operator="pressure_band_resource_block"`, `end_day=6`, one worker, and
HiGHS found a zero-error day-6 prefix in one iteration in about 20 seconds.
This is a **locally feasible prefix mechanism test**, not a full cold-start or
official-validity claim.

## Approaches not to repeat

- Do not use terminal quantity repair when visits are absent.
- Do not treat `Unknown`, time-limit status, or a local zero-error prefix as
  official validity.
- Do not freeze the whole lookahead tail; commit only a protected prefix.
- Do not deduplicate beam states only by the committed prefix when their
  lookahead plans differ.
- Do not rank/evaluate rolling candidates with a full-horizon violation vector.
- Do not exhaust the candidate pool on single-route insertions before bounded
  paired ejections.
- Do not restrict displaced VMI demand to an arbitrary three-day history;
  inventory flexibility can make much earlier recipients feasible.
- Do not trust the internal deadline as a strict SLA. An in-flight timing/MIP
  call can overrun it; use a process-level 1,800-second timeout and checkpoint
  after every accepted prefix.
- Do not run Gurobi tasks concurrently.
- Do not use the old `scratch/native_solutions/V2.12.xml` as validity evidence;
  the released checker reports 53 `DYN01` errors.

## Next implementation step

Make rolling-prefix beam construction a production strategy rather than a
temporary orchestration script:

1. retain a small beam keyed by both committed prefix and lookahead topology;
2. generate several general short-horizon policies per beam state;
3. require exact feasibility through the next commit boundary;
4. when the boundary is safety-only/near-feasible, run ejection-first surgical
   repair before discarding the state;
5. checkpoint every accepted prefix;
6. advance V2.17 from the valid day-6 mechanism to day 7, then through day 35;
7. serialize the raw production result and require the released checker;
8. apply the same general production path to V2.18.

Only full-horizon raw `solve_cold_start` outputs accepted by the released
checker count toward the requested two additional instances.
