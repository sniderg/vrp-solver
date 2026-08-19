"""Low-level heuristic portfolio over :class:`~vrp_solver.fast.state.SearchState`.

Step 3 of ``REBUILD_PLAN.md``.  Derived from the one-line descriptions of
Kheiri's LLH0-LLH18 (TS2020 SS3.2); no source exists to port, so these are our
own implementations.

The contract every operator obeys
---------------------------------
``operator(state, rng, lists) -> bool``.  It mutates ``state`` in place through
the primitives and returns whether it changed anything.  It is always called
inside a ``begin()`` transaction, so the caller can ``rollback()`` and the
operator itself never needs to worry about undo.

Two rules distinguish this portfolio from the legacy generator, and they are
the whole point of the rebuild:

1. **An operator never rejects its own output for infeasibility.** It picks a
   structurally sensible edit and applies it.  Whether the result breaks a rule
   is the objective's business.  The legacy generator filtered candidates
   against the rules and consequently returned zero candidates from a tight
   incumbent -- 96% of wall time spent generating nothing.
2. **An operator returns ``False`` only when the edit is not *expressible*,**
   e.g. swapping two stops when the route has one stop.  "No legal target
   exists" is not a reason to return ``False``; place the edit and let it be
   priced.

Cost discipline: each operator makes O(1) random choices and at most one
route-length pass (plus retiming, which is another such pass).  Nothing here
scans the whole instance; neighbour choice goes through
:class:`~vrp_solver.fast.retime.CandidateLists`.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import numpy as np

from .retime import CandidateLists, earliest_arrivals
from .state import EPSILON, SearchState, ShiftRec

Operator = Callable[[SearchState, random.Random, CandidateLists], bool]

_KIND_SOURCE = 1
_KIND_CUSTOMER = 2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _nonempty(state: SearchState, minimum: int = 1) -> list[int]:
    return [i for i, rec in enumerate(state.shifts) if len(rec.points) >= minimum]


def _pick_shift(state: SearchState, rng: random.Random, minimum: int = 1) -> int:
    positions = _nonempty(state, minimum)
    return rng.choice(positions) if positions else -1


def _clip_to_cutoff(
    fi, points: Sequence[int], arrivals: Sequence[int], quantities: Sequence[float]
):
    """Drop the tail of a route that would arrive at or after the score cutoff.

    ``SearchState``'s invariant is that every arrival lies strictly before the
    cutoff, because ``contest.truncate_solution`` deletes later operations
    before scoring.  A move that lengthens a route can push its last stops past
    that line; keeping them would mean the state scores operations the official
    pipeline never sees, and the equivalence guarantee would quietly break.

    Clipping rather than refusing is the right response: it is precisely what
    the scorer does to the same route, so the state stays faithful, and the move
    still fires. Arrivals are monotone under ``earliest_arrivals``, so a prefix
    scan suffices.
    """
    cutoff = fi.cutoff
    keep = len(arrivals)
    for i, arrival in enumerate(arrivals):
        if arrival >= cutoff:
            keep = i
            break
    if keep == len(arrivals):
        return points, arrivals, quantities
    return points[:keep], arrivals[:keep], quantities[:keep]


def _apply_route(
    state: SearchState,
    position: int,
    points: Sequence[int],
    quantities: Sequence[float],
    *,
    retime: bool = True,
) -> bool:
    """Install a new sequence, retiming it to the earliest legal arrivals.

    Retiming is the default because a resequencing move that keeps the old
    arrival times almost always breaks SHI02 (arrival earlier than physically
    possible), which is an artifact of the move rather than a property of the
    route.  Charging for it would make every resequencing move look bad and the
    search would never resequence.
    """
    rec = state.shifts[position]
    if retime:
        arrivals = earliest_arrivals(state.fi, rec.start, points, rec.driver)
    else:
        arrivals = list(rec.arrivals)
    points, arrivals, quantities = _clip_to_cutoff(
        state.fi, list(points), list(arrivals), list(quantities)
    )
    state.set_operations(position, list(points), list(arrivals), list(quantities))
    return True


def _apply_timing(
    state: SearchState, position: int, start: int, arrivals: Sequence[int]
) -> bool:
    """Retime in place, clipping any stop pushed past the cutoff."""
    rec = state.shifts[position]
    points, arrivals, quantities = _clip_to_cutoff(
        state.fi, list(rec.points), list(arrivals), list(rec.quantities)
    )
    if len(points) != len(rec.points):
        state.set_operations(position, points, arrivals, quantities)
        if start != rec.start:
            state.set_shift_timing(position, start, arrivals)
        return True
    state.set_shift_timing(position, start, list(arrivals))
    return True


def _customer_points(state: SearchState, rng: random.Random) -> int:
    fi = state.fi
    if not fi.customer_points:
        return -1
    return rng.choice(fi.customer_points)


def _earliest_unsatisfied(state: SearchState) -> tuple[int, int]:
    """``(point, deadline)`` of the earliest unsatisfied demand or order.

    TS2020 sec. 3.2 makes two of the nineteen heuristics feasibility-aware:

    > if the best recorded solution in hand is not feasible, then this heuristic
    > lists all the demands and orders and inserts the customer site with the
    > earliest unsatisfied demand or order into a randomly selected route.

    "Unsatisfied" is read against the rule that actually charges an error, so
    the two customer kinds are queried differently:

    * a **call-in** order is unsatisfied when its window total is outside
      ``[min_quantity, quantity]`` -- under the flexible minimum or over the
      nominal ceiling both count as a missed order.  Its deadline is the window's
      ``latest``.
    * a **VMI** customer is unsatisfied when its tank projection breaches the
      safety level.  Its deadline is the first breaching step, so the earliest
      breach across the instance competes on the same axis as an order deadline.

    Returns ``(-1, -1)`` when nothing is unsatisfied, which is the feasible case
    and tells the caller to fall back to its unbiased draw.
    """
    fi = state.fi
    best_point = -1
    best_deadline = -1

    for row, point in enumerate(fi.customer_points):
        if fi.cust_is_call_in[row]:
            ops = state._cust_ops[row]
            for earliest, latest, quantity, min_quantity in fi.cust_qs01_orders[row]:
                total = 0.0
                for arrival, op_quantity in ops:
                    if earliest <= arrival <= latest and op_quantity > 0.0:
                        total += op_quantity
                if total + EPSILON < min_quantity or total > quantity + EPSILON:
                    if best_deadline < 0 or latest < best_deadline:
                        best_point = point
                        best_deadline = latest
                    break
            continue

        # VMI: the first step where the projection drops below safety.
        if not state._cust_tank[row][0]:
            continue
        level = float(fi.cust_initial[row]) + np.cumsum(
            state._deliveries[row] - fi.cust_forecast[row]
        )
        breaches = np.flatnonzero(level < float(fi.cust_safety[row]) - EPSILON)
        if breaches.size == 0:
            continue
        deadline = int(breaches[0]) * fi.unit
        if best_deadline < 0 or deadline < best_deadline:
            best_point = point
            best_deadline = deadline

    return best_point, best_deadline


def _targeted_customer(state: SearchState, rng: random.Random) -> int:
    """The LLH0/LLH5 target when the state is infeasible, else an unbiased draw.

    Kheiri gates the targeting on the *best recorded* solution being infeasible.
    We gate on the live state instead, which is the same signal one step later
    and avoids threading the incumbent through every operator signature.

    The gate reads the state's error counters directly rather than calling
    ``score()``: the counters are already maintained incrementally, whereas
    ``score()`` builds a result object with two O(shifts) sums, and this runs on
    every invocation of two operators.
    """
    unsatisfied = state._callin_errors + state._tank_errors
    if unsatisfied:
        point, _deadline = _earliest_unsatisfied(state)
        if point >= 0:
            return point
    return _customer_points(state, rng)


def _compatible_trailers(state: SearchState, driver: int) -> tuple[int, ...]:
    return tuple(sorted(state.fi.driver_trailers[driver]))


def _placeholder_quantity(state: SearchState, point: int) -> float:
    """The quantity a newly inserted stop carries before decoding.

    Always zero. Quantities are derived, not chosen (see ``fast.decode``), so an
    operator that guessed a starting amount would only be supplying a value the
    decoder immediately overwrites. This used to guess half the tank's headroom,
    which mattered when quantity operators refined it and is now just noise the
    scorer would charge if a decode were ever skipped.
    """
    return 0.0


# --------------------------------------------------------------------------
# 3.1 route-internal
# --------------------------------------------------------------------------


def relocate_within(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Move one stop to another position in the same route."""
    position = _pick_shift(state, rng, 2)
    if position < 0:
        return False
    rec = state.shifts[position]
    n = len(rec.points)
    # Draw two *distinct* positions rather than drawing independently and
    # declining on a collision.  Independent draws decline 1/n of the time,
    # which on a short route is most of the time -- V2.25's routes average
    # ~2 stops and this operator fired on only 61% of invocations before the
    # change.  A declined invocation is a wasted search step.
    i, j = rng.sample(range(n), 2)
    points = list(rec.points)
    quantities = list(rec.quantities)
    point = points.pop(i)
    quantity = quantities.pop(i)
    points.insert(j, point)
    quantities.insert(j, quantity)
    return _apply_route(state, position, points, quantities)


def relocate_block_within(
    state: SearchState, rng: random.Random, lists: CandidateLists
) -> bool:
    """Move a contiguous run of stops to another position in the same route."""
    position = _pick_shift(state, rng, 3)
    if position < 0:
        return False
    rec = state.shifts[position]
    n = len(rec.points)
    # Cap the block at n-1 so at least one stop stays put; a block of the whole
    # route can only be reinserted where it was.
    length = rng.randrange(2, min(4, n - 1) + 1)
    i = rng.randrange(n - length + 1)
    points = list(rec.points)
    quantities = list(rec.quantities)
    block_p = points[i : i + length]
    block_q = quantities[i : i + length]
    del points[i : i + length]
    del quantities[i : i + length]
    # Choose from the positions that actually move the block, instead of
    # drawing freely and declining when the draw lands back at ``i``.
    targets = [j for j in range(len(points) + 1) if j != i]
    if not targets:
        return False
    j = targets[rng.randrange(len(targets))]
    points[j:j] = block_p
    quantities[j:j] = block_q
    return _apply_route(state, position, points, quantities)


def swap_within(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Exchange two stops of the same route."""
    position = _pick_shift(state, rng, 2)
    if position < 0:
        return False
    rec = state.shifts[position]
    i, j = rng.sample(range(len(rec.points)), 2)
    points = list(rec.points)
    quantities = list(rec.quantities)
    points[i], points[j] = points[j], points[i]
    quantities[i], quantities[j] = quantities[j], quantities[i]
    return _apply_route(state, position, points, quantities)


def swap_block_within(
    state: SearchState, rng: random.Random, lists: CandidateLists
) -> bool:
    """Exchange two disjoint equal-length runs of the same route."""
    position = _pick_shift(state, rng, 4)
    if position < 0:
        return False
    rec = state.shifts[position]
    n = len(rec.points)
    length = rng.randrange(1, max(1, n // 2) + 1)
    i = rng.randrange(n - 2 * length + 1)
    j = rng.randrange(i + length, n - length + 1)
    if i + length > j:
        return False
    points = list(rec.points)
    quantities = list(rec.quantities)
    points[i : i + length], points[j : j + length] = (
        points[j : j + length], points[i : i + length]
    )
    quantities[i : i + length], quantities[j : j + length] = (
        quantities[j : j + length], quantities[i : i + length]
    )
    return _apply_route(state, position, points, quantities)


def two_opt(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Reverse a run of stops (the classic 2-opt on a single route)."""
    position = _pick_shift(state, rng, 2)
    if position < 0:
        return False
    rec = state.shifts[position]
    n = len(rec.points)
    i = rng.randrange(n - 1)
    j = rng.randrange(i + 1, n)
    points = list(rec.points)
    quantities = list(rec.quantities)
    points[i : j + 1] = points[i : j + 1][::-1]
    quantities[i : j + 1] = quantities[i : j + 1][::-1]
    return _apply_route(state, position, points, quantities)


def move_layover(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Insert or remove the idle gap that ``_recompute_shift`` reads as a layover.

    A layover is not stored; it is inferred wherever the gap between one
    departure and the next arrival is at least the driver's layover duration
    plus the travel time.  So this operator manipulates the gap directly:
    either open one at a random stop, or close every gap by retiming.
    """
    position = _pick_shift(state, rng, 2)
    if position < 0:
        return False
    rec = state.shifts[position]
    if rng.random() < 0.5:
        # Close all gaps: retime to earliest, which removes inferred layovers.
        return _apply_route(state, position, rec.points, rec.quantities)
    # Open a gap before a random stop by delaying it and everything after it.
    fi = state.fi
    duration = fi.driver_layover_duration[rec.driver]
    i = rng.randrange(1, len(rec.points))
    arrivals = list(rec.arrivals)
    for k in range(i, len(arrivals)):
        arrivals[k] += duration
    return _apply_timing(state, position, rec.start, arrivals)


# --------------------------------------------------------------------------
# 3.2 inter-route
# --------------------------------------------------------------------------


def _two_shifts(state: SearchState, rng: random.Random, minimum: int = 1):
    positions = _nonempty(state, minimum)
    if len(positions) < 2:
        return None
    a, b = rng.sample(positions, 2)
    return a, b


def relocate_between(
    state: SearchState, rng: random.Random, lists: CandidateLists
) -> bool:
    """Move one stop from one shift into another."""
    pair = _two_shifts(state, rng)
    if pair is None:
        return False
    source, target = pair
    src = state.shifts[source]
    dst = state.shifts[target]
    i = rng.randrange(len(src.points))
    point = src.points[i]
    quantity = src.quantities[i]

    src_points = list(src.points)
    src_quantities = list(src.quantities)
    del src_points[i]
    del src_quantities[i]

    j = rng.randrange(len(dst.points) + 1)
    dst_points = list(dst.points)
    dst_quantities = list(dst.quantities)
    dst_points.insert(j, point)
    dst_quantities.insert(j, quantity)

    _apply_route(state, source, src_points, src_quantities)
    _apply_route(state, target, dst_points, dst_quantities)
    return True


def swap_between(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Exchange one stop of one shift with one stop of another."""
    pair = _two_shifts(state, rng)
    if pair is None:
        return False
    a, b = pair
    ra = state.shifts[a]
    rb = state.shifts[b]
    i = rng.randrange(len(ra.points))
    j = rng.randrange(len(rb.points))

    pa = list(ra.points)
    qa = list(ra.quantities)
    pb = list(rb.points)
    qb = list(rb.quantities)
    pa[i], pb[j] = pb[j], pa[i]
    qa[i], qb[j] = qb[j], qa[i]

    _apply_route(state, a, pa, qa)
    _apply_route(state, b, pb, qb)
    return True


def two_opt_star(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Exchange the tails of two routes."""
    pair = _two_shifts(state, rng)
    if pair is None:
        return False
    a, b = pair
    ra = state.shifts[a]
    rb = state.shifts[b]
    na = len(ra.points)
    nb = len(rb.points)
    # Exactly one of the (na+1)*(nb+1) cut pairs is a no-op -- both cuts at the
    # end, which exchanges two empty tails. Draw over the space *without* it
    # instead of drawing freely and declining: that decline was the whole reason
    # this operator sat at 6.7% empty invocations and so was the only remaining
    # source of empty steps besides ``create_shift``. Row-major indexing puts
    # the no-op last, so a draw below the total simply cannot hit it.
    t = rng.randrange((na + 1) * (nb + 1) - 1)
    i, j = divmod(t, nb + 1)

    pa = list(ra.points[:i]) + list(rb.points[j:])
    qa = list(ra.quantities[:i]) + list(rb.quantities[j:])
    pb = list(rb.points[:j]) + list(ra.points[i:])
    qb = list(rb.quantities[:j]) + list(ra.quantities[i:])

    _apply_route(state, a, pa, qa)
    _apply_route(state, b, pb, qb)
    return True


def cross_exchange(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Exchange a contiguous run between two routes."""
    pair = _two_shifts(state, rng, 2)
    if pair is None:
        return False
    a, b = pair
    ra = state.shifts[a]
    rb = state.shifts[b]
    la = rng.randrange(1, min(3, len(ra.points)) + 1)
    lb = rng.randrange(1, min(3, len(rb.points)) + 1)
    i = rng.randrange(len(ra.points) - la + 1)
    j = rng.randrange(len(rb.points) - lb + 1)

    pa = list(ra.points)
    qa = list(ra.quantities)
    pb = list(rb.points)
    qb = list(rb.quantities)
    block_pa, block_qa = pa[i : i + la], qa[i : i + la]
    block_pb, block_qb = pb[j : j + lb], qb[j : j + lb]
    pa[i : i + la], qa[i : i + la] = block_pb, block_qb
    pb[j : j + lb], qb[j : j + lb] = block_pa, block_qa

    _apply_route(state, a, pa, qa)
    _apply_route(state, b, pb, qb)
    return True


def merge_shifts(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Append one shift's route onto another and empty the donor.

    The donor is emptied instead of deleted: an empty shift is invisible to the
    scorer (``truncate_solution`` drops it) but stays available as a host for a
    later insertion, and keeping the shift list stable means an operator that
    remembers a position is not invalidated mid-transaction.
    """
    pair = _two_shifts(state, rng)
    if pair is None:
        return False
    keep, donor = pair
    rk = state.shifts[keep]
    rd = state.shifts[donor]
    points = list(rk.points) + list(rd.points)
    quantities = list(rk.quantities) + list(rd.quantities)
    _apply_route(state, donor, [], [])
    _apply_route(state, keep, points, quantities)
    return True


def split_shift(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Cut a route in two, moving the tail into a new shift.

    The new shift reuses the same driver and trailer and starts at the tail's
    first arrival.  That very likely breaks the driver's inter-shift separation
    (DRI01) and the trailer's load carry-over (SHI06); both are priced, and a
    later resource operator can fix them.  Refusing to split until a legal
    resource slot exists is exactly the gate this rebuild removes.
    """
    position = _pick_shift(state, rng, 2)
    if position < 0:
        return False
    rec = state.shifts[position]
    cut = rng.randrange(1, len(rec.points))
    tail_points = list(rec.points[cut:])
    tail_quantities = list(rec.quantities[cut:])
    head_points = list(rec.points[:cut])
    head_quantities = list(rec.quantities[:cut])

    start = rec.arrivals[cut]
    if start >= state.fi.cutoff:
        return False
    tail_arrivals = list(rec.arrivals[cut:])
    tail_points, tail_arrivals, tail_quantities = _clip_to_cutoff(
        state.fi, tail_points, tail_arrivals, tail_quantities
    )
    if not tail_points:
        return False
    new = ShiftRec(
        index=len(state.shifts),
        driver=rec.driver,
        trailer=rec.trailer,
        start=start,
        points=tail_points,
        arrivals=tail_arrivals,
        quantities=tail_quantities,
    )
    _apply_route(state, position, head_points, head_quantities)
    state.insert_shift(new)
    return True


# --------------------------------------------------------------------------
# 3.3 structural
# --------------------------------------------------------------------------


def insert_customer(
    state: SearchState, rng: random.Random, lists: CandidateLists
) -> bool:
    """Add a customer stop to an existing shift.

    The insertion point is drawn from the candidate list of a stop already in
    the route when one exists, so insertions are geographically sensible without
    scanning the instance.
    """
    if not state.shifts:
        return False
    position = rng.randrange(len(state.shifts))
    rec = state.shifts[position]
    fi = state.fi

    point = -1
    if rec.points:
        anchor = rec.points[rng.randrange(len(rec.points))]
        near = [p for p in lists.near(anchor) if fi.point_kind[p] == _KIND_CUSTOMER]
        if near:
            point = near[rng.randrange(len(near))]
    if point < 0:
        point = _customer_points(state, rng)
    if point < 0:
        return False

    j = rng.randrange(len(rec.points) + 1)
    points = list(rec.points)
    quantities = list(rec.quantities)
    points.insert(j, point)
    quantities.insert(j, _placeholder_quantity(state, point))
    return _apply_route(state, position, points, quantities)


def insert_unsatisfied(
    state: SearchState, rng: random.Random, lists: CandidateLists
) -> bool:
    """LLH0: insert the earliest unsatisfied customer into a random route.

    The feasibility-aware twin of :func:`insert_customer`. Both insert a customer
    stop; the difference is entirely in *which* customer, and that difference is
    the point. Geographic insertion improves the ratio and is blind to which
    order is going unserved, so on an instance whose only remaining errors are
    unserved call-in orders it can search indefinitely without ever addressing
    them -- which is exactly where V2.15 and V2.26 sat.

    When the state is feasible this degrades to an unbiased draw, so the operator
    stays useful (and keeps clearing the firing-rate gate) rather than declining.
    """
    if not state.shifts:
        return False
    point = _targeted_customer(state, rng)
    if point < 0:
        return False
    position = rng.randrange(len(state.shifts))
    rec = state.shifts[position]
    j = rng.randrange(len(rec.points) + 1)
    points = list(rec.points)
    quantities = list(rec.quantities)
    points.insert(j, point)
    quantities.insert(j, _placeholder_quantity(state, point))
    return _apply_route(state, position, points, quantities)


def replace_with_unsatisfied(
    state: SearchState, rng: random.Random, lists: CandidateLists
) -> bool:
    """LLH5: replace a random customer stop with the earliest unsatisfied one.

    The paper pairs insertion with replacement so the move is available when
    every route is already at its time limit and there is no room to add a stop.
    """
    fi = state.fi
    point = _targeted_customer(state, rng)
    if point < 0:
        return False
    candidates = [
        (position, index)
        for position, rec in enumerate(state.shifts)
        for index, existing in enumerate(rec.points)
        if fi.point_kind[existing] == _KIND_CUSTOMER and existing != point
    ]
    if not candidates:
        return False
    position, index = candidates[rng.randrange(len(candidates))]
    rec = state.shifts[position]
    points = list(rec.points)
    quantities = list(rec.quantities)
    points[index] = point
    quantities[index] = _placeholder_quantity(state, point)
    return _apply_route(state, position, points, quantities)


def insert_source(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Add a reload stop, so the trailer can carry more than one load.

    The load quantity is negative by convention.  It is set to the trailer's
    remaining headroom given what the route already dispenses after this point,
    which is the useful amount rather than an arbitrary one.
    """
    fi = state.fi
    sources = lists.sources
    if not sources or not state.shifts:
        return False
    position = rng.randrange(len(state.shifts))
    rec = state.shifts[position]
    source = sources[rng.randrange(len(sources))]
    j = rng.randrange(len(rec.points) + 1)

    remaining = 0.0
    for k in range(j, len(rec.points)):
        q = rec.quantities[k]
        if q > 0.0:
            remaining += q
    capacity = fi.trailer_capacity[rec.trailer] if 0 <= rec.trailer < len(fi.trailers) else 0.0
    load = -min(capacity, remaining) if remaining > EPSILON else -capacity

    points = list(rec.points)
    quantities = list(rec.quantities)
    points.insert(j, source)
    quantities.insert(j, load)
    return _apply_route(state, position, points, quantities)


def delete_operation(
    state: SearchState, rng: random.Random, lists: CandidateLists
) -> bool:
    """Drop one stop."""
    position = _pick_shift(state, rng, 1)
    if position < 0:
        return False
    rec = state.shifts[position]
    i = rng.randrange(len(rec.points))
    points = list(rec.points)
    quantities = list(rec.quantities)
    del points[i]
    del quantities[i]
    return _apply_route(state, position, points, quantities)


def replace_point(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Swap one stop for a nearby compatible point of the same kind."""
    position = _pick_shift(state, rng, 1)
    if position < 0:
        return False
    rec = state.shifts[position]
    fi = state.fi
    # Only stops that *have* a same-kind alternative are eligible.  Several Set
    # B instances (V2.13, V2.14, V2.19) define exactly one source, so a source
    # stop can never be replaced there -- a quarter of all stops on V2.13.
    # Drawing the stop uniformly and then declining wasted those invocations;
    # restricting the draw to eligible stops converts them into work while
    # leaving the operator's meaning unchanged.
    eligible = [
        i for i, p in enumerate(rec.points)
        if fi.point_kind[p] != _KIND_SOURCE or len(lists.sources) > 1
    ]
    if not eligible:
        return False
    i = eligible[rng.randrange(len(eligible))]
    old = rec.points[i]
    kind = fi.point_kind[old]
    near = [p for p in lists.near(old) if fi.point_kind[p] == kind]
    if not near:
        # The nearest-K list happened to contain no point of the same kind.
        # Widen to the whole instance rather than decline; this is the rare
        # path, so the O(n) scan costs nothing on average.
        near = [
            p for p in range(fi.n_points)
            if p != old and fi.point_kind[p] == kind
        ]
        if not near:
            return False
    new = near[rng.randrange(len(near))]
    points = list(rec.points)
    quantities = list(rec.quantities)
    points[i] = new
    if kind == _KIND_CUSTOMER:
        quantities[i] = _placeholder_quantity(state, new)
    return _apply_route(state, position, points, quantities)


def create_shift(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Add a shift, unconditionally.

    Contingency 3A, adopted up front: this operator does **not** look for a
    legal driver/trailer slot.  It picks a compatible driver/trailer pair (the
    only compatibility it respects, because TL03 is cheap to honour), picks a
    start inside a driver window, and places one customer.  Any resource-timing
    violation is priced and left for a later move to repair.  The legacy version
    of this operator spent 134.55 s on V2.15 and produced zero candidates
    precisely because it insisted on legality up front.
    """
    fi = state.fi
    if not fi.drivers:
        return False
    driver = rng.randrange(len(fi.drivers))
    trailers = _compatible_trailers(state, driver)
    if not trailers:
        return False
    trailer = trailers[rng.randrange(len(trailers))]

    point = _customer_points(state, rng)
    if point < 0:
        return False

    # A shift whose only stop arrives at or after the score cutoff is invisible
    # to the scorer, so drawing the start freely and then discovering that meant
    # declining 4.1% of invocations. Draw only from starts that leave room for
    # the drive out, which is the binding constraint: the arrival can still be
    # pushed later by a time window, and the resulting SHI04 is priced.
    latest = fi.cutoff - 1 - fi.time_matrix[fi.base][point]
    windows = [w for w in fi.driver_windows[driver] if w[0] <= latest]
    if not windows:
        return False
    w_start, w_end = windows[rng.randrange(len(windows))]
    limit = min(w_end, latest)
    start = w_start if limit <= w_start else rng.randrange(w_start, limit + 1)

    rec = ShiftRec(
        index=len(state.shifts),
        driver=driver,
        trailer=trailer,
        start=start,
        points=[point],
        arrivals=[start],
        quantities=[_placeholder_quantity(state, point)],
    )
    rec.arrivals = earliest_arrivals(fi, start, rec.points, driver)
    points, arrivals, quantities = _clip_to_cutoff(
        fi, rec.points, rec.arrivals, rec.quantities
    )
    if not points:
        # The one stop would land past the cutoff, so the shift would be
        # invisible to the scorer; nothing was mutated.
        return False
    rec.points, rec.arrivals, rec.quantities = points, arrivals, quantities
    state.insert_shift(rec)
    return True


def delete_shift(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Empty a shift's route (see ``merge_shifts`` on why empty, not removed)."""
    position = _pick_shift(state, rng, 1)
    if position < 0:
        return False
    return _apply_route(state, position, [], [])


# --------------------------------------------------------------------------
# 3.4 quantity / timing
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 3.5 quantities -- deliberately absent
# --------------------------------------------------------------------------
#
# There are no quantity operators. Quantities are not decision variables: they
# are derived from the route by ``fast.decode`` after every step, following
# Kheiri's design (TS2020 sec. 3.1), where a solution is routes only and none of
# the nineteen low-level heuristics touches a quantity.
#
# The four operators that used to live here (``increase_quantity``,
# ``decrease_quantity``, ``fill_quantity``, ``minimal_quantity``) plus their
# ``_quantity_ceiling`` bound were removed with the decoder. They caused the two
# worst defects of this rebuild -- an unbounded multiply that grew one drop to
# 107,593,087 kg for an LR of 0.000037, and the QS01 over-delivery that let a
# solution read as zero errors internally while the checker rejected it -- and a
# derived quantity makes both unreachable rather than merely penalised.


def retime_shift(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Slide the whole shift earlier or later."""
    if not state.shifts:
        return False
    position = rng.randrange(len(state.shifts))
    rec = state.shifts[position]
    delta = rng.choice((-480, -240, -120, -60, 60, 120, 240, 480))
    start = rec.start + delta
    if start < 0:
        return False
    arrivals = [a + delta for a in rec.arrivals]
    if arrivals and min(arrivals) < 0:
        return False
    # Stops pushed past the cutoff are clipped rather than refused: declining
    # here would make the operator unable to fire on any late shift, which is
    # exactly where retiming is most valuable.
    return _apply_timing(state, position, start, arrivals)


def compact_shift(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Pull every arrival to its earliest legal minute, shortening the shift.

    Working time is charged per minute, so removing slack is a direct cost
    reduction; it also frees resource time for other shifts.
    """
    position = _pick_shift(state, rng, 1)
    if position < 0:
        return False
    rec = state.shifts[position]
    return _apply_route(state, position, rec.points, rec.quantities)


def shift_one_arrival(
    state: SearchState, rng: random.Random, lists: CandidateLists
) -> bool:
    """Delay or advance a single stop, leaving the rest of the route alone."""
    position = _pick_shift(state, rng, 1)
    if position < 0:
        return False
    rec = state.shifts[position]
    i = rng.randrange(len(rec.points))
    delta = rng.choice((-120, -60, -30, 30, 60, 120))
    arrivals = list(rec.arrivals)
    if arrivals[i] + delta < 0:
        return False
    arrivals[i] += delta
    return _apply_timing(state, position, rec.start, arrivals)


# --------------------------------------------------------------------------
# 3.5 resource
# --------------------------------------------------------------------------


def change_trailer(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Reassign the trailer, preferring one the driver may pull.

    Contingency 3A applies here, and it is not hypothetical: across Set B
    almost every driver is qualified on exactly *one* trailer (V2.13, V2.24 and
    V2.20.2 have a single trailer per driver; V2.22 has one for 12 of its 13).
    A TL03-respecting version of this operator therefore cannot fire at all,
    and the trailer assignment becomes frozen for the whole search -- which also
    freezes SHI06, since load carry-over follows the trailer.

    So when no compatible alternative exists, pick any other trailer and let
    TL03 be charged. The repair is one `change_driver` away, and the pair of
    moves together is how the search reassigns a route's resources.
    """
    fi = state.fi
    if not state.shifts or len(fi.trailers) < 2:
        return False
    position = rng.randrange(len(state.shifts))
    rec = state.shifts[position]
    compatible = [
        t for t in _compatible_trailers(state, rec.driver) if t != rec.trailer
    ]
    if compatible:
        choices = compatible
    else:
        choices = [t for t in range(len(fi.trailers)) if t != rec.trailer]
    if not choices:
        return False
    state.set_shift_resources(position, trailer=choices[rng.randrange(len(choices))])
    return True


def change_driver(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    """Reassign the driver, keeping TL03 satisfied where possible.

    Drivers that can pull the current trailer are preferred; when none can, any
    driver is used and TL03 is charged. Restricting the choice to compatible
    drivers only would make a shift with a rare trailer unreassignable, which
    is the kind of dead end the rebuild is meant to avoid.
    """
    fi = state.fi
    if not state.shifts or len(fi.drivers) < 2:
        return False
    position = rng.randrange(len(state.shifts))
    rec = state.shifts[position]
    compatible = [
        d for d in range(len(fi.drivers))
        if d != rec.driver and rec.trailer in fi.driver_trailers[d]
    ]
    if compatible:
        new = compatible[rng.randrange(len(compatible))]
    else:
        others = [d for d in range(len(fi.drivers)) if d != rec.driver]
        if not others:
            return False
        new = others[rng.randrange(len(others))]
    state.set_shift_resources(position, driver=new)
    return True


def _pick_differing_pair(state: SearchState, rng: random.Random, attribute: str):
    """Two non-empty shifts whose ``attribute`` differs, or ``None``.

    Sampling two shifts at random and declining when they share the resource
    wastes a large share of invocations: with few resources, many shifts do
    share one (V2.25 declined ~29% of these). Filtering the second draw to
    shifts that differ from the first turns those declines into work.
    """
    positions = _nonempty(state)
    if len(positions) < 2:
        return None
    a = positions[rng.randrange(len(positions))]
    value = getattr(state.shifts[a], attribute)
    others = [
        p for p in positions
        if p != a and getattr(state.shifts[p], attribute) != value
    ]
    if not others:
        return None
    return a, others[rng.randrange(len(others))]


def swap_trailers(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    pair = _pick_differing_pair(state, rng, "trailer")
    if pair is None:
        return False
    a, b = pair
    ta = state.shifts[a].trailer
    tb = state.shifts[b].trailer
    state.set_shift_resources(a, trailer=tb)
    state.set_shift_resources(b, trailer=ta)
    return True


def swap_drivers(state: SearchState, rng: random.Random, lists: CandidateLists) -> bool:
    pair = _pick_differing_pair(state, rng, "driver")
    if pair is None:
        return False
    a, b = pair
    da = state.shifts[a].driver
    db = state.shifts[b].driver
    state.set_shift_resources(a, driver=db)
    state.set_shift_resources(b, driver=da)
    return True


# --------------------------------------------------------------------------
# the portfolio
# --------------------------------------------------------------------------

#: Every operator, in plan order (3.1 route-internal, 3.2 inter-route,
#: 3.3 structural, 3.4 quantity/timing, 3.5 resource).  Index into this tuple
#: is the "low-level heuristic id" the selection mechanism learns over, so the
#: order is stable and appended to rather than reshuffled.
OPERATORS: tuple[tuple[str, Operator], ...] = (
    # 3.1
    ("relocate_within", relocate_within),
    ("relocate_block_within", relocate_block_within),
    ("swap_within", swap_within),
    ("swap_block_within", swap_block_within),
    ("two_opt", two_opt),
    ("move_layover", move_layover),
    # 3.2
    ("relocate_between", relocate_between),
    ("swap_between", swap_between),
    ("two_opt_star", two_opt_star),
    ("cross_exchange", cross_exchange),
    ("merge_shifts", merge_shifts),
    ("split_shift", split_shift),
    # 3.3
    ("insert_customer", insert_customer),
    ("insert_unsatisfied", insert_unsatisfied),
    ("replace_with_unsatisfied", replace_with_unsatisfied),
    ("insert_source", insert_source),
    ("delete_operation", delete_operation),
    ("replace_point", replace_point),
    ("create_shift", create_shift),
    ("delete_shift", delete_shift),
    # 3.4  (no quantity operators: see "3.5 quantities" above)
    ("retime_shift", retime_shift),
    ("compact_shift", compact_shift),
    ("shift_one_arrival", shift_one_arrival),
    # 3.5
    ("change_trailer", change_trailer),
    ("change_driver", change_driver),
    ("swap_trailers", swap_trailers),
    ("swap_drivers", swap_drivers),
)

OPERATOR_NAMES: tuple[str, ...] = tuple(name for name, _ in OPERATORS)
