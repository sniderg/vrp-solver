# Native Set B matheuristic reference

## Success contract

A native cold-start run receives only an instance XML plus seed/configuration and emits a solution accepted by the released ROADEF V2 checker. Oracle/reference outputs may inform aggregate analysis but never seed routes, timings, or quantities. Require status `valid`, return code zero, success sentinel present, failure sentinel absent, and checker hash recorded.

## Representation and feasibility

Represent stable shift IDs, ordered operations, explicit reloads, route chains, chronological driver/trailer chains, predecessor stock, successor boundaries, and full-state fingerprints. Invalidate derived state after any topology, timing, quantity, driver, or trailer change.

Replay references, finite values, travel/setup precedence, return travel, windows, compatibility, quantity bounds, trailer continuity, layovers, driving limits, rest, resource non-overlap, call-in coverage, VMI safety/capacity, and exact boundaries. Do not invent source inventory.

## Cold-start construction

1. Derive call-in intervals and VMI critical intervals independently.
2. Build compatibility lists from trailers, windows, travel, driving feasibility, and insertion slack.
3. Process hard call-ins and earliest stockout intervals with regret-k.
4. Enumerate insertion position, resources, start windows, and optional reloads; prefer existing chains.
5. Use bounded ejection chains when insertion fails.
6. Periodically run block timing and full-horizon quantity repair.

End only with a locally feasible solution or explicit exhaustion.

## Chain-first neighbourhood search

Use `select block -> destroy -> rebuild -> exact repair -> replay -> accept`.

Select 2–8 related shifts around pressure, missed orders, sparse routes, conflicts, costly fragments, or diversified time bands. Include predecessor/successor resource boundaries. Destroy contiguous subsequences, related customers, deadline bands, reload segments, sparse shifts, or conflicts. Rebuild with regret insertion, split/merge, cross-exchange, fragment relocation, ejection chains, reload movement, and joint resource assignment.

Treat tails and internal waiting gaps as separate density surfaces. Extend tails with `source reload -> related chain`; insert the same chain internally and jointly retime the resource block. Do not require tail slack for internal gaps. Seed quantities only to activate visits; temporary overfill must disappear under full-horizon repair before acceptance.

## Pressure-band resource-block operator

1. Rank pressure by safety-deficit area, first breach, and repairability—not customer size alone.
2. Select target routes plus driver/trailer predecessors and successors inside an adaptive radius.
3. Destroy fragments around reloads and tight-window waits.
4. Rebuild using direct shifts, cross-reload relocation, proactive reassignment, predecessor evacuation, equal-length fragment exchange, three-route ejection cycles, and one-way optional VMI ejection.
5. Permit optional VMI deactivation but force call-in and layover-enabling visits active. Retain a positive quantity for zero-minimum mandatory visits.
6. Jointly retime after activation; deleted visits with stale arrivals can create false layovers.
7. Run full-horizon quantity repair and global replay before acceptance.

Activation can remove every visit to a pressured customer. Price missing topology into a compatible optional slot using breach/window features. If a late visit structurally enables a layover, retain it at a positive quantity and add an earlier duplicate. Expand the radius to include the preceding legal service window; periodic windows may be more than 1,440 minutes apart.

Customer-ID probes are only for discovering mechanics or regression fixtures. Promote reusable patterns and return to native cold-start execution.

## Exact repair and acceptance

Block timing must choose starts, arrivals, windows, legal layovers, and enforce travel, rest, resource chains, and fixed boundaries. Per-shift retiming is insufficient for shared resources.

Quantity repair must choose deliveries, source loads, and optional activation while enforcing call-ins, cumulative VMI bounds, trailer paths, and protected prefixes. Hard mode has no inventory/order/trailer slack; elastic output is diagnostic only.

Keep feasible candidates ordered by LR and diagnostics ordered by malformed/reference errors, physical violations, missed-order deficit, negative/overfill amount-duration, safety-deficit amount-duration, resource errors, then cost. Log candidate funnels, solver statuses/times, violation vectors, topology metrics, acceptance reason, and official verdict/hash.

## Set B generalisation

Profile order/resource counts, horizon, call-in share, VMI pressure, compatibility sparsity, window/rest/driving tightness, multi-stop feasibility, reload/layover prevalence, geography, and scale. Set policy continuously from those features; never hardcode names, IDs, or days.

Qualify feature-stratified Set B regimes with repeated fixed seeds. Report official-valid rate, time to first validity, valid LR, checker disagreements, and failure modes. Do not infer generality from V2.12 alone.

## Implementation order

1. Structured violations, fingerprints, and edge tests.
2. Multi-shift timing with boundary contracts.
3. Full-horizon quantity/activation repair.
4. Atomic replacement and rollback.
5. Ejection, split/merge, and pressure-band rebuilds.
6. Cold-start qualification across Set B.

A mechanism is done only with an automated regression test. The solver goal is done only when exact cold-start XML passes the released checker.
