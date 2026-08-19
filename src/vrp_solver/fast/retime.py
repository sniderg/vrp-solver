"""Earliest-feasible retiming of a route, and per-point candidate lists.

Step 3 support code for ``REBUILD_PLAN.md``.

Every structural operator changes a route's *sequence*, which invalidates its
arrival times.  Recomputing arrivals is therefore the one thing every operator
needs and the one place a subtle error would poison the whole portfolio, so it
lives here with its own tests.

The design decision that matters: :func:`earliest_arrivals` is *best effort*
and never refuses.  It places each stop as early as the travel time and the
point's time windows allow, and when no window can hold the stop it places the
arrival at the physically required minute anyway.  The resulting state is then
priced by the objective (SHI04 fires) instead of being rejected.  Refusing here
is exactly the gate that made the legacy search generate nothing: a route with
one badly-timed stop is often two moves away from a good one, and it has to be
reachable.
"""

from __future__ import annotations

from typing import Sequence

from .state import FastInstance

_KIND_SOURCE = 1
_KIND_CUSTOMER = 2


def earliest_arrivals(
    fi: FastInstance,
    start: int,
    points: Sequence[int],
    driver_id: int,
) -> list[int]:
    """Place every stop of ``points`` as early as travel and windows allow.

    Returns one arrival minute per point.

    A layover is inserted where the accumulated driving would otherwise exceed
    the driver's maximum (DRI03), including on the return leg to base.
    Layovers are not stored anywhere: ``_recompute_shift`` infers one wherever
    the gap between a departure and the next arrival is at least the layover
    duration plus the travel time, so "inserting a layover" means widening that
    gap to exactly that amount.

    Doing this here rather than leaving it to an operator is deliberate.
    Collapsing a route to its earliest arrivals removes the gaps that carried
    its existing layovers, so a naive retiming turns a legal route into a
    DRI03-violating one -- 22 fresh DRI03 errors on V2.13 -- purely as an
    artifact of the move. That is the same class of self-inflicted noise as the
    SHI02 errors retiming exists to avoid.

    One layover per shift is the limit (LAY03), so once one has been placed the
    driving cap can no longer be respected by adding another; later legs simply
    accumulate and DRI03 is charged. That is a real property of the route,
    not an artifact, and the objective should see it.

    A layover is also only placed when the route visits a customer designated
    for layovers, because LAY02 forbids one otherwise. When no such customer is
    on the route, an over-cap route is over-cap: DRI03 is charged and the search
    is left to fix it by inserting a layover customer or by shortening the
    route, which are the two real remedies.
    """
    time_matrix = fi.time_matrix
    setup_of = fi.setup_time
    kind_of = fi.point_kind
    layover_duration = fi.driver_layover_duration[driver_id]
    max_driving = fi.driver_max_driving[driver_id]
    layover_allowed = any(
        0 <= p < fi.n_points and fi.is_layover_customer[p] for p in points
    )

    arrivals: list[int] = []
    last_point = fi.base
    last_departure = start
    n_points = fi.n_points
    cumulated_driving = 0
    layovers = 0

    for index, point in enumerate(points):
        if point < 0 or point >= n_points:
            # An out-of-range point is already an error the objective prices;
            # keep the list aligned with ``points`` and carry on.
            arrivals.append(last_departure)
            continue

        travel = time_matrix[last_point][point]

        # Would this leg, plus the leg home afterwards, blow the driving cap?
        # Checking the return leg here as well as the outbound one matters:
        # ``_recompute_shift`` charges DRI03 on the way back to base too, and a
        # layover can only be placed between stops.
        remaining_after = 0
        if index == len(points) - 1:
            remaining_after = time_matrix[point][fi.base]
        needs_layover = (
            layover_allowed
            and layovers == 0
            and cumulated_driving + travel + remaining_after > max_driving
        )

        required = last_departure + travel
        if needs_layover:
            required += layover_duration

        arrival = required
        if kind_of[point] == _KIND_CUSTOMER:
            setup = setup_of[point]
            windows = fi.customer_windows[point]
            # For a call-in customer the arrival must also fall inside one of
            # its open orders (QS03), so intersect the two sets of intervals.
            # Skipping this made retiming invent QS03 errors on a route that
            # had none -- again, noise created by the move.
            order_spans = ()
            row = fi.customer_row[point]
            if row >= 0 and fi.cust_is_call_in[row]:
                order_spans = tuple(
                    (earliest, latest) for earliest, latest, _q, _mq in fi.cust_orders[row]
                )
            if windows:
                best: int | None = None
                for w_start, w_end in windows:
                    candidate = w_start if w_start > required else required
                    if candidate + setup > w_end:
                        continue
                    if order_spans:
                        # QS03 tests the arrival itself, not the departure.
                        for o_start, o_end in order_spans:
                            inner = candidate if candidate > o_start else o_start
                            if inner > o_end or inner + setup > w_end:
                                continue
                            if best is None or inner < best:
                                best = inner
                    elif best is None or candidate < best:
                        best = candidate
                if best is not None:
                    arrival = best
                # else: no window can hold this stop.  Leave the arrival at the
                # physically required minute and let SHI04 be charged for it.

        # Recheck against the gap actually taken: a window may have pushed the
        # arrival late enough to imply a layover we did not plan, or (when the
        # arrival stayed at ``required``) confirm the one we did.
        if (arrival - last_departure) >= (layover_duration + travel):
            layovers += 1
            # ``_recompute_shift``: driving up to the cap happens before the
            # layover, the remainder after it.
            cumulated_driving = travel - min(
                max(0, max_driving - cumulated_driving), travel
            )
        else:
            cumulated_driving += travel

        arrivals.append(arrival)
        last_point = point
        last_departure = arrival + setup_of[point]

    return arrivals


def shift_end(
    fi: FastInstance,
    points: Sequence[int],
    arrivals: Sequence[int],
    driver_id: int,
) -> int:
    """When the shift gets back to base, ignoring any end-of-shift layover.

    Used by operators that need to know whether a route still fits inside a
    driver window before committing to it.  This is an estimate for *targeting*
    only; ``_recompute_shift`` owns the authoritative end.
    """
    if not points:
        return 0
    last = points[-1]
    if last < 0 or last >= fi.n_points:
        return arrivals[-1] if arrivals else 0
    return arrivals[-1] + fi.setup_time[last] + fi.time_matrix[last][fi.base]


def latest_start_for(fi: FastInstance, driver_id: int) -> int:
    """The last minute of the driver's final time window."""
    windows = fi.driver_windows[driver_id]
    return max((w_end for _w_start, w_end in windows), default=0)


class CandidateLists:
    """Per-point nearest-K neighbour lists (step 3.6).

    HUST's matheuristic restricted every insertion to roughly the nearest 10%
    of compatible points; that is what turns an O(n) insertion scan into an
    O(K) one and is the difference between a portfolio that fires thousands of
    times per second and one that does not.

    ``fraction`` is the share of all points kept per point, swept in the
    step 3.6 experiment.  Neighbour lists are built once per instance and
    shared by every operator.
    """

    __slots__ = ("fi", "fraction", "k", "neighbours", "sources")

    def __init__(self, fi: FastInstance, *, fraction: float = 0.10) -> None:
        self.fi = fi
        self.fraction = fraction
        n = fi.n_points
        # Source points, precomputed here because several operators need them
        # per invocation and scanning ``point_kind`` each time is pure waste.
        self.sources: tuple[int, ...] = tuple(
            p for p in range(n) if fi.point_kind[p] == _KIND_SOURCE
        )
        # At least 5 neighbours, else small instances get a list too short to
        # offer any choice and every operator degenerates to one candidate.
        self.k = max(5, min(n - 1, int(round(n * fraction))))
        self.neighbours = self._build()

    def _build(self) -> list[tuple[int, ...]]:
        fi = self.fi
        n = fi.n_points
        time_matrix = fi.time_matrix
        k = self.k
        out: list[tuple[int, ...]] = []
        for i in range(n):
            row = time_matrix[i]
            order = sorted(
                (j for j in range(n) if j != i), key=lambda j: row[j]
            )
            out.append(tuple(order[:k]))
        return out

    def near(self, point: int) -> tuple[int, ...]:
        if 0 <= point < len(self.neighbours):
            return self.neighbours[point]
        return ()
