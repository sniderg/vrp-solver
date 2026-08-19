"""Quantity decoding: derive quantities from a route, never search over them.

This follows Kheiri's winning design (TS2020, section 3.1) rather than our own
earlier scheme.  There, a solution is encoded as *routes only* -- "a set of
routes (shifts), each defines a driver, a trailer and a set of sites to visit"
-- and quantities are not decision variables at all.  Evaluation invokes two
consecutive decoders: first timing (schedule every operation at its earliest
feasible time, which we already have in ``retime``), then quantity:

    if the current site is a customer, then we assign the least possible amount
    of product to deliver; otherwise if the site is a source, then all the
    remaining quantities will be given to the previous customer sites, starting
    from the nearest ones.

None of Kheiri's nineteen low-level heuristics (LLH0-LLH18) touches a quantity.
That is the point.  Our three quantity operators were the direct cause of the
two worst defects this rebuild hit -- an unbounded multiply that grew one drop
to 107,593,087 kg, and the QS01 over-delivery that let a solution read as zero
errors internally while the checker rejected it -- and both are *structurally
unreachable* once there is no quantity variable to game.  A ceiling patched onto
a free variable is strictly weaker than not exposing the variable.

The paper's sentence is a sketch, and three gaps have to be filled.  Each is
resolved toward its own binding rule rather than toward a guess:

* **How much a source loads.** The sentence describes what a source's load is
  *for*, not how large it is.  Loading is free once the trip is committed (cost
  is distance and duration, both already fixed by the route), and the trailer's
  capacity is the only bound, so a source fills the trailer.  Deciding it from
  downstream demand instead would make a leading source load nothing, which is
  what our first attempt did: on V2.24 every route beginning with a source
  loaded 0 and the decode produced 358 errors.
* **What "least possible" means for a VMI customer.** Its rules bound it from
  both sides: SHI16 wants at least ``min_operation_quantity``, and DYN01 wants
  the tank never below its safety level.  We take the larger, clipped to the
  tank's headroom at arrival so the projection cannot charge an overfill.
* **What "least possible" means for a call-in customer.** C24's flexibility
  minimum is the least amount that counts the order satisfied and C23 caps the
  order's total at the ordered quantity, so the least possible amount is the
  flexible minimum less whatever is already booked against that same order.

The surplus pass is what keeps the ratio honest: leftover load at the end of a
route is delivered quantity thrown away, and the logistic ratio is a cost *per
unit delivered*, so pushing it into customers already visited (nearest first, as
the paper says) improves the objective without touching the route.
"""

from __future__ import annotations

import numpy as np

from .state import EPSILON, SearchState, _KIND_CUSTOMER, _KIND_SOURCE

try:  # The tail min/max projection in C: one call per customer stop per step.
    from ..inventory_fast import tank_bounds as _tank_bounds_fast
except ImportError:  # pragma: no cover - pure-Python fallback
    _tank_bounds_fast = None

__all__ = ["decode_quantities", "decode_shift_quantities"]


# --------------------------------------------------------------------------
# per-customer bounds
# --------------------------------------------------------------------------


def _order_window(fi, row: int, arrival: int):
    """The call-in order covering ``arrival`` as ``(earliest, latest, min, cap)``.

    ``None`` when the arrival falls in no order window: C21 forbids delivering
    to a call-in customer with no order, so the least possible amount is zero.
    """
    for earliest, latest, quantity, min_quantity in fi.cust_qs01_orders[row]:
        if earliest <= arrival <= latest:
            return earliest, latest, min_quantity, quantity
    return None


def _booked_against_order(
    state: SearchState, row: int, window: tuple[int, int]
) -> float:
    """Total already booked against one order window, from the live state."""
    earliest, latest = window
    total = 0.0
    for op_arrival, op_quantity in state._cust_ops[row]:
        if earliest <= op_arrival <= latest and op_quantity > 0.0:
            total += op_quantity
    return total


def _step_of(fi, arrival: int) -> int:
    step = arrival // fi.unit
    if step < 0:
        return 0
    if step >= fi.horizon:
        return fi.horizon - 1
    return step


def _tank_bounds(state: SearchState, row: int, arrival: int) -> tuple[float, float]:
    """``(safety_need, headroom)`` for a VMI customer at ``arrival``.

    Both come from the same discrete projection the scorer uses, so a decoded
    quantity cannot create a DYN01 breach or overfill the scorer would charge.
    ``safety_need`` is the deepest future shortfall against the safety level,
    which is the smallest single drop that clears every breach this stop can
    reach; ``headroom`` is the spare volume at the moment of arrival.
    """
    fi = state.fi
    step = _step_of(fi, arrival)
    capacity = float(fi.cust_capacity[row])
    safety = float(fi.cust_safety[row])

    if _tank_bounds_fast is not None:
        return _tank_bounds_fast(
            float(fi.cust_initial[row]),
            fi.cust_forecast[row],
            state._deliveries[row],
            step,
            capacity,
            safety,
        )

    # Vectorized deliberately: a Python loop over the horizon here, called once
    # per customer stop, cost 7x throughput (7,607 -> 1,006 steps/s) and the
    # decoder runs on every step of the search.  The compiled version above is
    # the same projection without the per-call cumsum allocation.
    level = float(fi.cust_initial[row]) + np.cumsum(
        state._deliveries[row] - fi.cust_forecast[row]
    )
    tail = level[step:]
    if tail.size == 0:
        return 0.0, 0.0
    # Adding delta at this stop raises the level at *every* later step too, so
    # the overfill bound is the tightest headroom from here to the end of the
    # horizon, not the headroom at arrival. Using the arrival-only headroom
    # filled tanks to the brim and then charged 57 overfill steps on V2.24 once
    # a later delivery landed on top. The safety need is the mirror image: the
    # deepest shortfall over the same tail.
    headroom = capacity - float(tail.max())
    worst_short = safety - float(tail.min())
    return (
        (worst_short if worst_short > 0.0 else 0.0),
        (headroom if headroom > 0.0 else 0.0),
    )


def _least_and_most(
    state: SearchState, point: int, arrival: int
) -> tuple[float, float]:
    """``(least, most)`` deliverable at one customer stop, ignoring the trailer.

    ``least`` is the paper's "least possible amount"; ``most`` is the ceiling the
    surplus pass may fill up to.  Both exclude the stop's own current quantity,
    which the caller has already zeroed out of the live state.
    """
    fi = state.fi
    row = fi.customer_row[point]
    if row < 0:
        return 0.0, 0.0

    if fi.cust_is_call_in[row]:
        window = _order_window(fi, row, arrival)
        if window is None:
            return 0.0, 0.0  # C21
        earliest, latest, minimum, ceiling = window
        booked = _booked_against_order(state, row, (earliest, latest))
        least = minimum - booked
        most = ceiling - booked  # C23
        if least < 0.0:
            least = 0.0
        if most < 0.0:
            most = 0.0
        if least > most:
            least = most
        return least, most

    need, headroom = _tank_bounds(state, row, arrival)
    least = float(fi.min_operation_quantity[point])
    if need > least:
        least = need
    if least > headroom:
        least = headroom
    if least < 0.0:
        least = 0.0
    return least, headroom


# --------------------------------------------------------------------------
# the decoder
# --------------------------------------------------------------------------


def _starting_load(state: SearchState, rec) -> float:
    """The trailer's load when this shift begins (C18).

    The end quantity of the trailer's previous shift, which is the initial
    quantity less everything earlier in the chain moved.
    """
    fi = state.fi
    trailer_id = rec.trailer
    if not (0 <= trailer_id < len(fi.trailers)):
        return 0.0
    load = float(fi.trailer_initial[trailer_id])
    key = (rec.start, rec.index)
    for other in state._by_trailer[trailer_id]:
        if other is not rec and (other.start, other.index) < key:
            load -= other.net_quantity
    capacity = float(fi.trailer_capacity[trailer_id])
    if load < 0.0:
        return 0.0
    return capacity if load > capacity else load


#: Whether the surplus pass runs.  The paper prescribes it ("all the remaining
#: quantities will be given to the previous customer sites"), and it does buy
#: logistic ratio, but it also creates errors it cannot see: pushing surplus
#: into a customer's tank through one shift can leave another shift's stop at
#: that same customer unable to reach its own ``min_operation_quantity``, which
#: is an SHI16 error, and the same coupling produces QS01 shortfalls and QS02
#: safety breaches. On V2.24 standalone the pass trades 0 errors / LR 0.03030
#: for 7 errors / LR 0.02631, and publication ranks errors before ratio.
#: A per-shift decoder cannot resolve a cross-shift bound, so this stays a
#: measured switch until the decode is made customer-aware.
#:
#: Reserving headroom for other shifts' unserved stops at the same customer was
#: tried and rejected: it recovered 1 error of 13 on V2.24 and cost an
#: O(all shifts) scan per surplus candidate, in the inner loop. The residual is
#: temporal rather than per-stop -- surplus delivered early raises the tank level
#: at every later step, so the breach it causes can be a QS02 safety breach in a
#: window no single shift's decode can see.
SPEND_SURPLUS = True


def decode_shift_quantities(state: SearchState, position: int) -> bool:
    """Assign every quantity in one shift by the least-possible rule.

    Returns whether anything changed.  Must be called inside a transaction: it
    goes through the state's own primitives, so it journals and reverts like any
    other move.
    """
    fi = state.fi
    rec = state.shifts[position]
    if not rec.points:
        return False

    points = list(rec.points)
    arrivals = list(rec.arrivals)
    before = list(rec.quantities)
    n = len(points)

    capacity = float(fi.trailer_capacity[rec.trailer]) if 0 <= rec.trailer < len(
        fi.trailers
    ) else 0.0

    # The bounds are read from the live state, so this shift's own quantities
    # must not be double-counted: zero them first, then write decoded values
    # back one stop at a time so each later stop sees the earlier ones.
    #
    # Staged writes publish the inventory projection without rescoring -- the
    # bounds need the projection and nothing else, and one `finish_staged` at
    # the end costs what a single primitive edit costs. See "staged edits" in
    # ``state.py``.
    rows_before = state._customer_rows(rec)
    quantities = [0.0] * n
    state.stage_operations(position, points, arrivals, quantities)

    on_hand = _starting_load(state, rec)

    # -- forward pass: sources fill, customers take the least they need.
    for index, point in enumerate(points):
        kind = fi.point_kind[point]
        if kind == _KIND_SOURCE:
            room = capacity - on_hand
            if room <= EPSILON:
                continue
            quantities[index] = -room
            on_hand += room
            state.stage_quantity(position, index, -room)
        elif kind == _KIND_CUSTOMER:
            least, _most = _least_and_most(state, point, arrivals[index])
            if least > on_hand:
                least = on_hand
            if least <= 0.0:
                continue
            quantities[index] = least
            on_hand -= least
            state.stage_quantity(position, index, least)

    # -- surplus pass: leftover load is delivered quantity thrown away.
    if SPEND_SURPLUS and on_hand > EPSILON:
        _spend_surplus(state, position, points, arrivals, quantities)

    state.finish_staged(position, rows_before)
    return quantities != before


def _tail_slack(quantities: list[float], start: float, index: int) -> float:
    """How much more can be dropped at ``index`` without draining the trailer.

    Adding delta at ``index`` lowers the running load at every later stop, so
    the bound is the minimum load from ``index`` onward.  Enforcing it here is
    what makes SHI06 (negative trailer quantity) unreachable by decoding, rather
    than something the objective has to price.
    """
    load = start
    worst = None
    for position, quantity in enumerate(quantities):
        load -= quantity
        if position >= index and (worst is None or load < worst):
            worst = load
    return 0.0 if worst is None or worst < 0.0 else worst


def _spend_surplus(
    state: SearchState,
    position: int,
    points: list[int],
    arrivals: list[int],
    quantities: list[float],
) -> None:
    """Push leftover load into customers on the route, nearest first.

    "Nearest" is measured from the last source visited, which is the paper's
    reference point ("starting from the nearest ones").  Every candidate is
    bounded by its own ceiling (tank headroom, or the order's C23 cap) and by
    the trailer's remaining load along the rest of the route.

    The ceiling is recomputed here rather than reused from the forward pass, and
    that is load-bearing.  Reusing the forward pass's value was wrong in two
    ways at once: it was measured before the surplus pushes existed, and two
    stops at the same customer each saw a ceiling that ignored the other's push.
    On V2.24 the stale version cost the decode 0 -> 7 errors on the first pass
    and 47 QS02 tank overfills on the second, and it made the decode
    non-idempotent -- which matters because the search re-decodes a touched
    shift thousands of times, so the second pass is the regime it lives in.
    Each push is published to the live state immediately for the same reason.
    """
    fi = state.fi
    start = _starting_load(state, state.shifts[position])

    last_source = fi.base
    for index in range(len(points) - 1, -1, -1):
        if fi.point_kind[points[index]] == _KIND_SOURCE:
            last_source = points[index]
            break

    row_of = fi.customer_row
    candidates = sorted(
        (
            index
            for index in range(len(points))
            if fi.point_kind[points[index]] == _KIND_CUSTOMER
            and row_of[points[index]] >= 0
        ),
        key=lambda index: fi.time_matrix[last_source][points[index]],
    )

    for index in candidates:
        # Live ceiling, which here is *incremental* room rather than a total:
        # unlike the forward pass, this stop's current quantity is already in
        # the projection, so a VMI headroom is what fits on top of it and a
        # call-in `most` already nets off what is booked. Subtracting
        # `quantities[index]` again would double-count it.
        _least, room = _least_and_most(state, points[index], arrivals[index])
        if room <= EPSILON:
            continue
        slack = _tail_slack(quantities, start, index)
        if slack <= EPSILON:
            continue
        extra = room if room < slack else slack
        if extra <= EPSILON:
            continue
        quantities[index] += extra
        state.stage_quantity(position, index, quantities[index])


def decode_quantities(state: SearchState) -> bool:
    """Decode every shift.  Returns whether anything changed.

    Shifts are decoded in trailer-chain order, because a shift's starting load
    is the previous shift's ending load (C18) and decoding the predecessor
    changes it.
    """
    order = sorted(
        range(len(state.shifts)),
        key=lambda position: (
            state.shifts[position].trailer,
            state.shifts[position].start,
            state.shifts[position].index,
        ),
    )
    changed = False
    for position in order:
        if decode_shift_quantities(state, position):
            changed = True
    return changed
