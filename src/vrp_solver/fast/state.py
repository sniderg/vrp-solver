"""Mutable search state with incremental rescoring.

Step 1 of ``REBUILD_PLAN.md``.

Design contract
---------------
``SearchState`` holds a *truncated* solution: every operation arrives strictly
before the score cutoff and every shift starts strictly before it.  Under that
invariant the state reproduces, exactly, the scored fields that
:func:`vrp_solver.contest.score_prefix_with_feasibility_tail` reports with
``score_days == feasibility_days`` and ``ignore_tail_call_ins=True`` -- the
settings the native search actually uses.

The score decomposes into five independently-dirtyable groups:

===================  ==============================================================
group                what depends on it
===================  ==============================================================
per shift            cost, distance, delivered, loaded, and the shift-local rules
                     (REF_*, LAY02, LAY03, SHI02..SHI05, SHI11, SHI16, QS03,
                     DRI03, DRI08, TL03)
per trailer chain    SHI06 (start-of-shift trailer load carries between shifts)
                     and TL01
per driver chain     DRI01
per VMI customer     DYN01 (negative, overfill), QS02 (safety breach), and the
                     ``safety_kg_min`` penalty
per call-in customer QS01
===================  ==============================================================

Mutation is copy-on-write per shift: a move snapshots only the shift records it
touches, so ``revert`` is exact by construction.  This is contingency 1C in the
plan, chosen up front because a subtly wrong in-place revert is worse than a
2x slower correct one, and the shift records are small.

``derive_solution`` reads only ``time_matrix``, ``setup_time_for_point``,
``drivers`` and ``base_index`` -- none of which truncation touches -- so the
derived shifts are identical on the full and the truncated instance.  The
legacy fast scorer derives twice for that reason; here one derivation serves
both cost and feasibility.  ``tests/test_fast_state.py`` pins that claim.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

from ..model import Instance, Operation, Shift, Solution

try:  # Contingency 1A: the tank aggregate loop in C.
    from ..inventory_fast import score_customer_row as _score_customer_row
except ImportError:  # pragma: no cover - pure-Python fallback
    _score_customer_row = None

try:  # The per-mutation shift re-sum in C (13.8 s of a 60 s V2.17 profile).
    from ..inventory_fast import sum_shift_totals as _sum_shift_totals_fast
except ImportError:  # pragma: no cover - pure-Python fallback
    _sum_shift_totals_fast = None

EPSILON = 1e-6
MINUTES_PER_DAY = 1440

_KIND_BASE = 0
_KIND_SOURCE = 1
_KIND_CUSTOMER = 2
_KIND_UNKNOWN = 3


def instance_days(instance: Instance) -> int:
    horizon_minutes = instance.horizon * instance.unit
    return (horizon_minutes + MINUTES_PER_DAY - 1) // MINUTES_PER_DAY


# --------------------------------------------------------------------------
# Static preprocessing
# --------------------------------------------------------------------------


class FastInstance:
    """Flat, lookup-free view of an instance truncated to ``score_days``.

    Every per-point property the inner loop needs becomes an array indexed by
    point id, so the hot path performs no dict lookups and no dataclass
    attribute chains.
    """

    __slots__ = (
        "instance",
        "score_days",
        "cutoff",
        "horizon",
        "unit",
        "base",
        "n_points",
        "time_matrix",
        "distance_matrix",
        "setup_time",
        "point_kind",
        "customer_row",
        "trailer_allowed",
        "customer_windows",
        "driver_windows",
        "is_layover_customer",
        "is_call_in",
        "min_operation_quantity",
        "customer_capacity",
        "n_customers",
        "customer_points",
        "cust_initial",
        "cust_capacity",
        "cust_safety",
        "cust_forecast",
        "cust_is_call_in",
        "cust_orders",
        "cust_qs01_orders",
        "drivers",
        "trailers",
        "driver_trailers",
        "trailer_capacity",
        "trailer_initial",
        "trailer_distance_cost",
        "driver_time_cost",
        "driver_layover_cost",
        "driver_layover_duration",
        "driver_max_driving",
        "driver_min_inter_shift",
    )

    def __init__(self, instance: Instance, *, score_days: int) -> None:
        if score_days <= 0:
            raise ValueError("score_days must be positive")
        if score_days > instance_days(instance):
            raise ValueError(
                f"score_days={score_days} exceeds instance horizon "
                f"{instance_days(instance)}"
            )

        self.instance = instance
        self.score_days = score_days
        self.cutoff = score_days * MINUTES_PER_DAY
        self.unit = instance.unit
        # Matches contest.truncate_instance.
        self.horizon = min(instance.horizon, (self.cutoff + instance.unit - 1) // instance.unit)
        self.base = instance.base_index

        n = len(instance.time_matrix)
        self.n_points = n
        self.time_matrix = instance.time_matrix
        self.distance_matrix = instance.distance_matrix

        self.setup_time = [0] * n
        self.point_kind = [_KIND_UNKNOWN] * n
        self.customer_row = [-1] * n
        self.trailer_allowed: list[frozenset[int] | None] = [None] * n
        self.customer_windows: list[tuple[tuple[int, int], ...]] = [()] * n
        self.is_layover_customer = [False] * n
        self.is_call_in = [False] * n
        self.min_operation_quantity = [0.0] * n
        self.customer_capacity = [0.0] * n

        if 0 <= self.base < n:
            self.point_kind[self.base] = _KIND_BASE

        for source in instance.sources:
            p = source.index
            self.point_kind[p] = _KIND_SOURCE
            self.setup_time[p] = source.setup_time
            self.trailer_allowed[p] = frozenset(source.allowed_trailers)

        # Truncated customers, in instance order (the row order the legacy
        # scorer's delivery matrix uses).
        self.n_customers = len(instance.customers)
        h = self.horizon
        self.customer_points = [c.index for c in instance.customers]
        self.cust_initial = np.empty(self.n_customers, dtype=np.float64)
        self.cust_capacity = np.empty(self.n_customers, dtype=np.float64)
        self.cust_safety = np.empty(self.n_customers, dtype=np.float64)
        self.cust_forecast = np.zeros((self.n_customers, h), dtype=np.float64)
        self.cust_is_call_in = np.zeros(self.n_customers, dtype=np.int8)
        # Orders surviving truncation, as (earliest, latest, quantity, min_qty).
        self.cust_orders: list[tuple[tuple[int, int, float, float], ...]] = []
        # Subset of the above that QS01 actually judges.
        self.cust_qs01_orders: list[tuple[tuple[int, int, float, float], ...]] = []

        latest_required = h * instance.unit
        for row, customer in enumerate(instance.customers):
            p = customer.index
            self.point_kind[p] = _KIND_CUSTOMER
            self.setup_time[p] = customer.setup_time
            self.trailer_allowed[p] = frozenset(customer.allowed_trailers)
            self.customer_windows[p] = tuple(
                (w.start, w.end) for w in customer.time_windows
            )
            self.is_layover_customer[p] = customer.layover_customer
            self.is_call_in[p] = customer.call_in
            self.min_operation_quantity[p] = customer.min_operation_quantity
            self.customer_capacity[p] = customer.capacity
            self.customer_row[p] = row

            self.cust_initial[row] = customer.initial_tank_quantity
            self.cust_capacity[row] = customer.capacity
            self.cust_safety[row] = customer.safety_level
            self.cust_is_call_in[row] = 1 if customer.call_in else 0
            forecast = customer.forecast or ()
            length = min(len(forecast), h)
            if length:
                self.cust_forecast[row, :length] = forecast[:length]

            # contest._truncate_customer drops call-in orders that open at or
            # after the cutoff; VMI orders are untouched.
            orders = customer.orders
            if customer.call_in:
                orders = tuple(o for o in orders if o.earliest_time < self.cutoff)
            packed = tuple(
                (o.earliest_time, o.latest_time, o.quantity, o.min_quantity_to_satisfy)
                for o in orders
            )
            self.cust_orders.append(packed)
            self.cust_qs01_orders.append(
                tuple(o for o in packed if o[1] <= latest_required)
                if customer.call_in
                else ()
            )

        self.drivers = instance.drivers
        self.trailers = instance.trailers
        self.driver_trailers = [frozenset(d.trailer_ids) for d in instance.drivers]
        self.trailer_capacity = [t.capacity for t in instance.trailers]
        self.trailer_initial = [t.initial_quantity for t in instance.trailers]
        self.trailer_distance_cost = [t.distance_cost for t in instance.trailers]
        self.driver_time_cost = [d.time_cost for d in instance.drivers]
        self.driver_layover_cost = [d.layover_cost for d in instance.drivers]
        self.driver_layover_duration = [d.layover_duration for d in instance.drivers]
        self.driver_max_driving = [d.max_driving_duration for d in instance.drivers]
        self.driver_min_inter_shift = [d.min_inter_shift_duration for d in instance.drivers]
        self.driver_windows = [
            tuple((w.start, w.end) for w in d.time_windows) for d in instance.drivers
        ]


# --------------------------------------------------------------------------
# Per-shift record
# --------------------------------------------------------------------------


class ShiftRec:
    """One shift plus every cached quantity derivable from it alone."""

    __slots__ = (
        "index",
        "driver",
        "trailer",
        "start",
        "points",
        "arrivals",
        "quantities",
        # cached, filled by SearchState._recompute_shift
        "end",
        "layovers",
        "cost",
        "distance",
        "delivered",
        "loaded",
        "local_errors",
        "local_warnings",
        "net_quantity",
        "cum_sorted",
    )

    def __init__(
        self,
        index: int,
        driver: int,
        trailer: int,
        start: int,
        points: list[int],
        arrivals: list[int],
        quantities: list[float],
    ) -> None:
        self.index = index
        self.driver = driver
        self.trailer = trailer
        self.start = start
        self.points = points
        self.arrivals = arrivals
        self.quantities = quantities
        self.end = start
        self.layovers = 0
        self.cost = 0.0
        self.distance = 0.0
        self.delivered = 0.0
        self.loaded = 0.0
        self.local_errors = 0
        self.local_warnings = 0
        self.net_quantity = 0.0
        self.cum_sorted: list[float] = []

    def restore_from(self, other: "ShiftRec") -> None:
        """Copy every field back from a snapshot, keeping this identity.

        Undo works on the live object rather than on a list position, so a
        transaction that also inserts or removes shifts cannot make a recorded
        position stale.
        """
        self.index = other.index
        self.driver = other.driver
        self.trailer = other.trailer
        self.start = other.start
        self.points = list(other.points)
        self.arrivals = list(other.arrivals)
        self.quantities = list(other.quantities)
        self.end = other.end
        self.layovers = other.layovers
        self.cost = other.cost
        self.distance = other.distance
        self.delivered = other.delivered
        self.loaded = other.loaded
        self.local_errors = other.local_errors
        self.local_warnings = other.local_warnings
        self.net_quantity = other.net_quantity
        self.cum_sorted = list(other.cum_sorted)

    def clone(self) -> "ShiftRec":
        rec = ShiftRec(
            self.index,
            self.driver,
            self.trailer,
            self.start,
            list(self.points),
            list(self.arrivals),
            list(self.quantities),
        )
        rec.end = self.end
        rec.layovers = self.layovers
        rec.cost = self.cost
        rec.distance = self.distance
        rec.delivered = self.delivered
        rec.loaded = self.loaded
        rec.local_errors = self.local_errors
        rec.local_warnings = self.local_warnings
        rec.net_quantity = self.net_quantity
        rec.cum_sorted = list(self.cum_sorted)
        return rec


@dataclass(frozen=True)
class StateScore:
    """The scored fields, matching :class:`vrp_solver.contest.ContestScore`."""

    scored_shifts: int
    scored_operations: int
    scored_delivered_quantity: float
    scored_loaded_quantity: float
    scored_estimated_cost: float
    feasibility_errors: int
    feasibility_warnings: int
    safety_kg_min: float
    tank_safety_breach_steps: int
    tank_negative_steps: int
    tank_overfill_steps: int
    vmi_customers_below_safety: int

    # Violation groups, broken out so a penalized objective can weight each
    # kind separately.  ``shift_errors + trailer_errors + driver_errors +
    # tank_negative_steps + tank_overfill_steps + tank_safety_breach_steps +
    # callin_errors == feasibility_errors`` by construction.
    shift_errors: int = 0
    trailer_errors: int = 0
    driver_errors: int = 0
    callin_errors: int = 0

    @property
    def feasible(self) -> bool:
        return self.feasibility_errors == 0

    @property
    def hard_violations(self) -> int:
        # Matches contest._score_fast: for search acceptance every official
        # rule error is hard, QS02 safety breaches included.
        return self.feasibility_errors

    @property
    def logistic_ratio(self) -> float:
        return self.scored_estimated_cost / max(1.0, self.scored_delivered_quantity)


# --------------------------------------------------------------------------
# Mutable state
# --------------------------------------------------------------------------


class SearchState:
    """Mutable, incrementally scored solution."""

    __slots__ = (
        "fi",
        "shifts",
        "_deliveries",
        "_cust_ops",
        "_shift_errors",
        "_shift_warnings",
        "_cost",
        "_distance",
        "_delivered",
        "_loaded",
        "_trailer_errors",
        "_driver_errors",
        "_tank_errors",
        "_tank_breach_steps",
        "_tank_negative_steps",
        "_tank_overfill_steps",
        "_tank_breach_points",
        "_safety_kg_min",
        "_callin_errors",
        "_callin_warnings",
        "_cust_tank",
        "_cust_callin",
        "_trailer_chain",
        "_driver_chain",
        "_by_trailer",
        "_by_driver",
        "_journal",
    )

    def __init__(self, fi: FastInstance, shifts: Iterable[ShiftRec]) -> None:
        self.fi = fi
        self.shifts: list[ShiftRec] = list(shifts)
        h = fi.horizon
        self._deliveries = np.zeros((fi.n_customers, h), dtype=np.float64)
        # arrival -> quantity multiset per customer row, for QS01.
        self._cust_ops: list[list[tuple[int, float]]] = [
            [] for _ in range(fi.n_customers)
        ]
        # per-customer cached contributions, so a dirty customer updates by delta
        self._cust_tank: list[tuple[int, int, int, int, float]] = [
            (0, 0, 0, 0, 0.0) for _ in range(fi.n_customers)
        ]
        self._cust_callin: list[tuple[int, int]] = [(0, 0) for _ in range(fi.n_customers)]
        # Per-resource chain error counts, so a move recomputes only the chains
        # whose membership or timing it actually changed.
        self._trailer_chain = [0] * len(fi.trailers)
        self._driver_chain = [0] * len(fi.drivers)
        # Non-empty shifts grouped by resource, rebuilt by _sum_shift_totals so
        # the chain walks never re-scan the whole shift list.
        self._by_trailer = [[] for _ in fi.trailers]
        self._by_driver = [[] for _ in fi.drivers]
        self._journal: list[dict] | None = None

        for rec in self.shifts:
            self._register_ops(rec)
        self._rebuild_all()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_solution(
        cls,
        instance: Instance | FastInstance,
        solution: Solution,
        *,
        score_days: int | None = None,
    ) -> "SearchState":
        """Build state from a solution, truncating exactly as ``contest`` does."""
        if isinstance(instance, FastInstance):
            fi = instance
        else:
            if score_days is None:
                score_days = instance_days(instance)
            fi = FastInstance(instance, score_days=score_days)

        cutoff = fi.cutoff
        recs: list[ShiftRec] = []
        for shift in solution.shifts:
            if shift.start >= cutoff:
                continue
            points: list[int] = []
            arrivals: list[int] = []
            quantities: list[float] = []
            for op in shift.operations:
                if op.arrival >= cutoff:
                    continue
                points.append(op.point)
                arrivals.append(op.arrival)
                quantities.append(op.quantity)
            if not points:
                continue
            recs.append(
                ShiftRec(shift.index, shift.driver, shift.trailer, shift.start,
                         points, arrivals, quantities)
            )
        return cls(fi, recs)

    def to_solution(self, *, drop_empty: bool = False) -> Solution:
        """The current state as a ``Solution``.

        Operators empty shifts rather than removing them, so index-stable
        positions survive a rollback.  Scoring is unaffected -- the official
        truncation drops operation-free shifts before it scores -- but the
        released checker *crashes* on an ``<operations />`` element
        (return code 0xE0434352, a .NET unhandled exception), so anything
        written to disk for the checker must pass ``drop_empty=True``.
        """
        shifts = tuple(
            Shift(
                index=rec.index,
                driver=rec.driver,
                trailer=rec.trailer,
                start=rec.start,
                operations=tuple(
                    Operation(point=p, arrival=a, quantity=q)
                    for p, a, q in zip(rec.points, rec.arrivals, rec.quantities)
                ),
            )
            for rec in self.shifts
            if rec.points or not drop_empty
        )
        return Solution(shifts=shifts)

    def copy(self) -> "SearchState":
        clone = SearchState.__new__(SearchState)
        clone.fi = self.fi
        clone.shifts = [rec.clone() for rec in self.shifts]
        clone._deliveries = self._deliveries.copy()
        clone._cust_ops = [list(ops) for ops in self._cust_ops]
        clone._cust_tank = list(self._cust_tank)
        clone._cust_callin = list(self._cust_callin)
        clone._journal = None
        for name in (
            "_shift_errors", "_shift_warnings", "_cost", "_distance", "_delivered",
            "_loaded", "_trailer_errors", "_driver_errors", "_tank_errors",
            "_tank_breach_steps", "_tank_negative_steps", "_tank_overfill_steps",
            "_tank_breach_points", "_safety_kg_min", "_callin_errors",
            "_callin_warnings",
        ):
            setattr(clone, name, getattr(self, name))
        clone._trailer_chain = list(self._trailer_chain)
        clone._driver_chain = list(self._driver_chain)
        clone._by_trailer = [[] for _ in self.fi.trailers]
        clone._by_driver = [[] for _ in self.fi.drivers]
        clone._sum_shift_totals()
        return clone

    # -- delivery bookkeeping ---------------------------------------------

    def _register_ops(self, rec: ShiftRec) -> None:
        fi = self.fi
        unit = fi.unit
        h = fi.horizon
        row_of = fi.customer_row
        deliveries = self._deliveries
        for point, arrival, quantity in zip(rec.points, rec.arrivals, rec.quantities):
            row = row_of[point]
            if row < 0:
                continue
            self._cust_ops[row].append((arrival, quantity))
            if quantity > 0.0:
                step = arrival // unit
                if step < 0:
                    step = 0
                elif step >= h:
                    step = h - 1
                deliveries[row, step] += quantity

    def _unregister_ops(self, rec: ShiftRec) -> None:
        fi = self.fi
        unit = fi.unit
        h = fi.horizon
        row_of = fi.customer_row
        deliveries = self._deliveries
        for point, arrival, quantity in zip(rec.points, rec.arrivals, rec.quantities):
            row = row_of[point]
            if row < 0:
                continue
            self._cust_ops[row].remove((arrival, quantity))
            if quantity > 0.0:
                step = arrival // unit
                if step < 0:
                    step = 0
                elif step >= h:
                    step = h - 1
                deliveries[row, step] -= quantity

    def _customer_rows(self, rec: ShiftRec) -> set[int]:
        row_of = self.fi.customer_row
        rows = set()
        for point in rec.points:
            row = row_of[point]
            if row >= 0:
                rows.add(row)
        return rows

    # -- per-shift local recompute ----------------------------------------

    def _recompute_shift(self, rec: ShiftRec) -> None:
        """Fill every cached field derivable from this shift alone.

        Mirrors ``rules.derive_solution`` plus the shift-attributed subset of
        ``rules._validate_shift_operations`` and
        ``rules._validate_resource_constraints``.  SHI06 and the inter-shift
        chain rules are excluded; they need the trailer/driver chains.
        """
        fi = self.fi
        base = fi.base
        time_matrix = fi.time_matrix
        distance_matrix = fi.distance_matrix
        setup_of = fi.setup_time
        kind_of = fi.point_kind
        n_points = fi.n_points

        errors = 0
        warnings = 0

        if not rec.points:
            # ``contest.truncate_solution`` drops operation-free shifts before
            # scoring, so such a shift is invisible to the official pipeline:
            # it contributes no cost and takes part in no rule, including the
            # driver and trailer chains.  Mirror that exactly instead of
            # forbidding the state, because an operator that empties a route
            # legitimately passes through this shape.
            rec.end = rec.start
            rec.layovers = 0
            rec.cost = 0.0
            rec.distance = 0.0
            rec.delivered = 0.0
            rec.loaded = 0.0
            rec.net_quantity = 0.0
            rec.cum_sorted = []
            rec.local_errors = 0
            rec.local_warnings = 0
            return

        driver_id = rec.driver
        trailer_id = rec.trailer
        if not (0 <= driver_id < len(fi.drivers)):
            errors += 1  # REF_DRIVER
            driver_ok = False
        else:
            driver_ok = True
        if not (0 <= trailer_id < len(fi.trailers)):
            errors += 1  # REF_TRAILER
            trailer_ok = False
        else:
            trailer_ok = True

        if not driver_ok or not trailer_ok:
            # A dangling reference makes the derived quantities meaningless;
            # record the reference errors and stop.  Search moves never create
            # this state, but scoring must not raise if a caller does.
            rec.end = rec.start
            rec.layovers = 0
            rec.cost = 0.0
            rec.distance = 0.0
            rec.delivered = 0.0
            rec.loaded = 0.0
            rec.net_quantity = 0.0
            rec.cum_sorted = []
            rec.local_errors = errors
            rec.local_warnings = warnings
            return

        layover_duration = fi.driver_layover_duration[driver_id]
        max_driving = fi.driver_max_driving[driver_id]
        trailer_cap = fi.trailer_capacity[trailer_id]

        points = rec.points
        arrivals = rec.arrivals
        quantities = rec.quantities
        n = len(points)

        last_point = base
        last_departure = rec.start
        cumulated_driving = 0
        layovers = 0
        distance = 0.0
        delivered = 0.0
        loaded = 0.0
        cum = 0.0
        cum_list: list[float] = []
        prev_driving = 0
        has_layover_customer = False

        for i in range(n):
            point = points[i]
            arrival = arrivals[i]
            quantity = quantities[i]

            if point < 0 or point >= n_points:
                # ``rules._validate_shift_operations`` records SHI03 and skips
                # the stop.  ``derive_solution`` has no such guard and would
                # raise IndexError, so the legacy scorer cannot score this
                # state at all -- there is no equivalence obligation here.
                # Count the error and skip; search moves never produce it.
                errors += 1
                continue

            travel = time_matrix[last_point][point]
            setup = setup_of[point]
            departure = arrival + setup
            layover_before = (arrival - last_departure) >= (layover_duration + travel)

            if layover_before:
                layovers += 1
                driving_before_layover = min(
                    max(0, max_driving - cumulated_driving), travel
                )
                cumulated_driving = travel - driving_before_layover
            else:
                driving_before_layover = 0
                cumulated_driving += travel

            distance += distance_matrix[last_point][point]
            cum += quantity
            cum_list.append(cum)
            if quantity > 0.0:
                delivered += quantity
            elif quantity < 0.0:
                loaded -= quantity

            # --- SHI02: arrival not earlier than physically required.
            # ``rules`` recomputes the leg from the previous departure/point,
            # which are exactly ``last_departure``/``last_point`` here.
            required = (
                last_departure
                + travel
                + (layover_duration if layover_before else 0)
            )
            if arrival + EPSILON < required:
                errors += 1

            kind = kind_of[point]
            if kind == _KIND_CUSTOMER:
                if fi.is_layover_customer[point]:
                    has_layover_customer = True
                windows = fi.customer_windows[point]
                inside = False
                for w_start, w_end in windows:
                    if w_start <= arrival and departure <= w_end:
                        inside = True
                        break
                if not inside:
                    errors += 1  # SHI04
                allowed = fi.trailer_allowed[point]
                if allowed is not None and trailer_id not in allowed:
                    errors += 1  # SHI05
                if quantity < -EPSILON:
                    errors += 1  # SHI11
                if not fi.is_call_in[point]:
                    if quantity - fi.customer_capacity[point] > EPSILON:
                        errors += 1  # SHI16 over capacity
                    if quantity + EPSILON < fi.min_operation_quantity[point]:
                        errors += 1  # SHI16 under minimum
                else:
                    orders = fi.cust_orders[fi.customer_row[point]]
                    if orders:
                        matched = False
                        for earliest, latest, _q, _mq in orders:
                            if earliest <= arrival <= latest:
                                matched = True
                                break
                        if not matched:
                            errors += 1  # QS03
            elif kind == _KIND_SOURCE:
                allowed = fi.trailer_allowed[point]
                if allowed is not None and trailer_id not in allowed:
                    errors += 1  # SHI05
                if quantity > EPSILON:
                    errors += 1  # SHI11

            # --- DRI03: driving between layovers
            if layover_before:
                driving_check = prev_driving + driving_before_layover
            else:
                driving_check = cumulated_driving
            if driving_check - max_driving > EPSILON:
                errors += 1

            last_point = point
            last_departure = departure
            prev_driving = cumulated_driving

        if cum_list:
            return_time = time_matrix[last_point][base]
            distance += distance_matrix[last_point][base]
            if cumulated_driving + return_time > max_driving and layovers == 0:
                layovers += 1
                end = last_departure + return_time + layover_duration
            else:
                end = last_departure + return_time
            # DRI03 including the return leg
            if cumulated_driving + return_time - max_driving > EPSILON:
                errors += 1
        else:
            end = rec.start

        # LAY02 / LAY03
        if layovers > 0 and not has_layover_customer:
            errors += 1
        if layovers > 1:
            errors += 1

        # TL03: driver/trailer compatibility
        if trailer_id not in fi.driver_trailers[driver_id]:
            errors += 1
        # DRI08: shift interval inside a driver window
        inside = False
        for w_start, w_end in fi.driver_windows[driver_id]:
            if w_start <= rec.start and end <= w_end:
                inside = True
                break
        if not inside:
            errors += 1

        working_time = end - rec.start - layovers * layover_duration
        rec.end = end
        rec.layovers = layovers
        rec.distance = distance
        rec.delivered = delivered
        rec.loaded = loaded
        rec.net_quantity = cum
        rec.cum_sorted = sorted(cum_list)
        rec.cost = (
            distance * fi.trailer_distance_cost[trailer_id]
            + working_time * fi.driver_time_cost[driver_id]
            + layovers * fi.driver_layover_cost[driver_id]
        )
        rec.local_errors = errors
        rec.local_warnings = warnings

    # -- chain recompute ---------------------------------------------------

    def _recompute_trailer_chain(self, trailer_id: int) -> None:
        """SHI06 (trailer load carry) and TL01 for one trailer.

        Each trailer's load chain is independent of every other trailer's, so a
        move that touches one trailer never changes another's contribution.
        ``derive_solution`` carries the load in ``(start, index)`` order, and
        TL01 compares consecutive shifts in the same order.
        """
        fi = self.fi
        if not (0 <= trailer_id < len(fi.trailers)):
            return
        ordered = sorted(
            self._by_trailer[trailer_id],
            key=lambda rec: (rec.start, rec.index),
        )
        load = fi.trailer_initial[trailer_id]
        cap = fi.trailer_capacity[trailer_id]
        prev_end: int | None = None
        errors = 0
        for rec in ordered:
            cum_sorted = rec.cum_sorted
            if cum_sorted:
                # trailer_quantity[i] == load - cum[i], so
                #   negative  <=> cum[i] > load + EPS
                #   over cap  <=> cum[i] < load - cap - EPS
                errors += len(cum_sorted) - bisect_right(cum_sorted, load + EPSILON)
                errors += bisect_left(cum_sorted, load - cap - EPSILON)
            load -= rec.net_quantity
            if prev_end is not None and rec.start < prev_end:
                errors += 1  # TL01
            prev_end = rec.end
        self._trailer_chain[trailer_id] = errors

    def _recompute_driver_chain(self, driver_id: int) -> None:
        """DRI01 (minimum inter-shift separation) for one driver."""
        fi = self.fi
        if not (0 <= driver_id < len(fi.drivers)):
            return
        ordered = sorted(
            self._by_driver[driver_id],
            key=lambda rec: (rec.start, rec.index),
        )
        separation = fi.driver_min_inter_shift[driver_id]
        prev_end: int | None = None
        errors = 0
        for rec in ordered:
            if prev_end is not None and rec.start < prev_end + separation:
                errors += 1
            prev_end = rec.end
        self._driver_chain[driver_id] = errors

    def _recompute_chains(
        self,
        trailers: Iterable[int] | None = None,
        drivers: Iterable[int] | None = None,
    ) -> None:
        """Recompute the given chains, or all of them when passed ``None``."""
        fi = self.fi
        for trailer_id in (range(len(fi.trailers)) if trailers is None else trailers):
            self._recompute_trailer_chain(trailer_id)
        for driver_id in (range(len(fi.drivers)) if drivers is None else drivers):
            self._recompute_driver_chain(driver_id)
        self._trailer_errors = sum(self._trailer_chain)
        self._driver_errors = sum(self._driver_chain)

    # -- per-customer recompute -------------------------------------------

    def _recompute_customer_tank(self, row: int) -> None:
        """DYN01 / QS02 / safety_kg_min for one VMI customer."""
        fi = self.fi
        if fi.cust_is_call_in[row]:
            new = (0, 0, 0, 0, 0.0)
        elif _score_customer_row is not None:
            new = _score_customer_row(
                float(fi.cust_initial[row]),
                fi.cust_forecast[row],
                self._deliveries[row],
                float(fi.cust_capacity[row]),
                float(fi.cust_safety[row]),
                fi.horizon,
                fi.unit,
            )
        else:
            ending = np.cumsum(self._deliveries[row] - fi.cust_forecast[row])
            ending += fi.cust_initial[row]
            safety = fi.cust_safety[row]
            capacity = fi.cust_capacity[row]
            negative = int(np.count_nonzero(ending < -EPSILON))
            overfill = int(np.count_nonzero(ending > capacity + EPSILON))
            breach = int(np.count_nonzero(ending < safety - EPSILON))
            deficit = safety - ending - EPSILON
            kg_min = float(deficit[deficit > 0.0].sum()) * fi.unit
            new = (breach, negative, overfill, 1 if breach else 0, kg_min)

        old = self._cust_tank[row]
        if new == old:
            return
        self._cust_tank[row] = new
        # Integer counters take a delta safely.
        self._tank_breach_steps += new[0] - old[0]
        self._tank_negative_steps += new[1] - old[1]
        self._tank_overfill_steps += new[2] - old[2]
        self._tank_breach_points += new[3] - old[3]
        # tank rule errors: negative + overfill (DYN01) + breach (QS02)
        self._tank_errors += (new[0] + new[1] + new[2]) - (old[0] + old[1] + old[2])
        # ``safety_kg_min`` is a float and must not be delta-accumulated (see
        # _sum_safety_kg_min); the caller re-sums it once per rescore.

    def _recompute_customer_callin(self, row: int) -> None:
        """QS01 for one call-in customer."""
        orders = self.fi.cust_qs01_orders[row]
        if not orders:
            new = (0, 0)
        else:
            ops = self._cust_ops[row]
            errors = 0
            warnings = 0
            for earliest, latest, quantity, min_quantity in orders:
                total = 0.0
                for arrival, op_quantity in ops:
                    if earliest <= arrival <= latest:
                        total += op_quantity
                if total + EPSILON < min_quantity:
                    errors += 1
                elif total > quantity + EPSILON:
                    # Nominal is a ceiling: the checker calls an over-delivered
                    # order a missed order.  Without this the search buys ratio
                    # by over-filling call-in orders for free -- which is what
                    # made V2.15 read as zero errors internally and fail the
                    # released checker with four QS01 MissedOrder lines.
                    errors += 1
                elif total + EPSILON < quantity:
                    warnings += 1
            new = (errors, warnings)

        old = self._cust_callin[row]
        if new == old:
            return
        self._cust_callin[row] = new
        self._callin_errors += new[0] - old[0]
        self._callin_warnings += new[1] - old[1]

    # -- full rebuild ------------------------------------------------------

    def _rebuild_all(self) -> None:
        for rec in self.shifts:
            self._recompute_shift(rec)
        self._sum_shift_totals()
        self._recompute_chains()

        self._tank_errors = 0
        self._tank_breach_steps = 0
        self._tank_negative_steps = 0
        self._tank_overfill_steps = 0
        self._tank_breach_points = 0
        self._safety_kg_min = 0.0
        self._callin_errors = 0
        self._callin_warnings = 0
        for row in range(self.fi.n_customers):
            self._cust_tank[row] = (0, 0, 0, 0, 0.0)
            self._cust_callin[row] = (0, 0)
            self._recompute_customer_tank(row)
            self._recompute_customer_callin(row)
        self._sum_safety_kg_min()

    # -- score -------------------------------------------------------------

    def score(self) -> StateScore:
        errors = (
            self._shift_errors
            + self._trailer_errors
            + self._driver_errors
            + self._tank_errors
            + self._callin_errors
        )
        return StateScore(
            scored_shifts=sum(1 for rec in self.shifts if rec.points),
            scored_operations=sum(len(rec.points) for rec in self.shifts),
            scored_delivered_quantity=self._delivered,
            scored_loaded_quantity=self._loaded,
            scored_estimated_cost=self._cost,
            feasibility_errors=errors,
            feasibility_warnings=self._shift_warnings + self._callin_warnings,
            safety_kg_min=self._safety_kg_min,
            tank_safety_breach_steps=self._tank_breach_steps,
            tank_negative_steps=self._tank_negative_steps,
            tank_overfill_steps=self._tank_overfill_steps,
            vmi_customers_below_safety=self._tank_breach_points,
            shift_errors=self._shift_errors,
            trailer_errors=self._trailer_errors,
            driver_errors=self._driver_errors,
            callin_errors=self._callin_errors,
        )

    @property
    def errors(self) -> int:
        return (
            self._shift_errors
            + self._trailer_errors
            + self._driver_errors
            + self._tank_errors
            + self._callin_errors
        )

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def delivered(self) -> float:
        return self._delivered

    @property
    def safety_kg_min(self) -> float:
        return self._safety_kg_min

    @property
    def logistic_ratio(self) -> float:
        return self._cost / max(1.0, self._delivered)

    # -- mutation ----------------------------------------------------------

    def begin(self) -> None:
        """Open a transaction.  Exactly one may be open at a time."""
        if self._journal is not None:
            raise RuntimeError("a transaction is already open")
        self._journal = []

    def commit(self) -> None:
        if self._journal is None:
            raise RuntimeError("no transaction is open")
        self._journal = None

    def rollback(self) -> None:
        """Undo every mutation since :meth:`begin`, exactly.

        Entries are keyed by shift *identity*, so inserts and removals in the
        same transaction cannot invalidate a recorded position.
        """
        journal = self._journal
        if journal is None:
            raise RuntimeError("no transaction is open")
        self._journal = None
        rows: set[int] = set()
        for entry in reversed(journal):
            kind = entry["kind"]
            if kind == "shift":
                live: ShiftRec = entry["live"]
                rows |= self._customer_rows(live)
                self._unregister_ops(live)
                # The snapshot carries its own cached derivation, so restoring
                # is exact without re-deriving.
                live.restore_from(entry["saved"])
                self._register_ops(live)
                rows |= self._customer_rows(live)
            elif kind == "insert":
                rec = entry["rec"]
                rows |= self._customer_rows(rec)
                self._unregister_ops(rec)
                self.shifts.remove(rec)
            elif kind == "remove":
                rec = entry["rec"]
                self.shifts.insert(min(entry["position"], len(self.shifts)), rec)
                self._register_ops(rec)
                rows |= self._customer_rows(rec)
        self._recompute_dirty(rows)

    def touched_positions(self) -> list[int]:
        """Positions of shifts modified in the open transaction.

        Lets a caller re-derive only what a move actually changed -- the
        quantity decoder runs after every step, so decoding all shifts instead
        of the one or two touched would dominate the step cost.
        """
        journal = self._journal
        if not journal:
            return []
        live = [
            entry["live"] for entry in journal if entry["kind"] == "shift"
        ] + [entry["rec"] for entry in journal if entry["kind"] == "insert"]
        index_of = {id(rec): i for i, rec in enumerate(self.shifts)}
        out = []
        for rec in live:
            position = index_of.get(id(rec))
            if position is not None and position not in out:
                out.append(position)
        return out

    def _touch(self, position: int) -> ShiftRec:
        """Snapshot the shift at ``position`` and return the live record."""
        rec = self.shifts[position]
        journal = self._journal
        if journal is not None:
            for entry in journal:
                if entry["kind"] == "shift" and entry["live"] is rec:
                    break
            else:
                journal.append({"kind": "shift", "live": rec, "saved": rec.clone()})
        return rec

    # -- primitive edits ---------------------------------------------------

    def set_operations(
        self,
        position: int,
        points: Sequence[int],
        arrivals: Sequence[int],
        quantities: Sequence[float],
    ) -> None:
        """Replace one shift's whole route.  The workhorse primitive."""
        rec = self._touch(position)
        rows = self._customer_rows(rec)
        self._unregister_ops(rec)
        rec.points = list(points)
        rec.arrivals = list(arrivals)
        rec.quantities = list(quantities)
        self._register_ops(rec)
        rows |= self._customer_rows(rec)
        self._recompute_dirty(
            rows, dirty_shift=rec,
            trailers=(rec.trailer,), drivers=(rec.driver,),
        )

    def set_quantity(self, position: int, op_index: int, quantity: float) -> None:
        rec = self._touch(position)
        point = rec.points[op_index]
        arrival = rec.arrivals[op_index]
        row = self.fi.customer_row[point]
        self._adjust_delivery(point, arrival, rec.quantities[op_index], -1)
        rec.quantities[op_index] = quantity
        self._adjust_delivery(point, arrival, quantity, +1)
        self._recompute_dirty(
            {row} if row >= 0 else set(), dirty_shift=rec,
            trailers=(rec.trailer,), drivers=(rec.driver,),
        )

    # -- staged edits ------------------------------------------------------
    #
    # The quantity decoder writes one stop at a time so each later stop sees
    # the earlier ones in the live inventory projection, but it does not read
    # any *score*: it reads `_deliveries` and `_cust_ops`, both of which
    # `_adjust_delivery` maintains on its own. Routing those writes through
    # `set_quantity` therefore paid for a full rescore per stop, and because
    # `_sum_shift_totals` sums over every shift that made the inner loop
    # O(stops * shifts). Measured on V2.15: 58,321 `set_quantity` calls, 10.1 s
    # of a 14.9 s run inside `_recompute_dirty`.
    #
    # These primitives publish the projection and skip the derivation; the
    # caller calls `finish_staged` once when the shift is fully decoded.
    # Rollback stays exact because `_touch` still snapshots the record, and the
    # snapshot carries its own cached derivation.

    def stage_operations(
        self,
        position: int,
        points: Sequence[int],
        arrivals: Sequence[int],
        quantities: Sequence[float],
    ) -> None:
        """Replace a route without rescoring.  Pair with :meth:`finish_staged`."""
        rec = self._touch(position)
        self._unregister_ops(rec)
        rec.points = list(points)
        rec.arrivals = list(arrivals)
        rec.quantities = list(quantities)
        self._register_ops(rec)

    def stage_quantity(self, position: int, op_index: int, quantity: float) -> None:
        """Set one quantity without rescoring.  Pair with :meth:`finish_staged`."""
        rec = self.shifts[position]
        point = rec.points[op_index]
        arrival = rec.arrivals[op_index]
        self._adjust_delivery(point, arrival, rec.quantities[op_index], -1)
        rec.quantities[op_index] = quantity
        self._adjust_delivery(point, arrival, quantity, +1)

    def finish_staged(self, position: int, rows: Iterable[int] = ()) -> None:
        """Rescore after a run of staged edits.

        ``rows`` are customer rows the staged edits touched but the shift no
        longer visits (a removed stop still needs its tank reprojected).
        """
        rec = self.shifts[position]
        dirty = self._customer_rows(rec)
        dirty.update(rows)
        self._recompute_dirty(
            dirty, dirty_shift=rec,
            trailers=(rec.trailer,), drivers=(rec.driver,),
        )

    def set_shift_resources(
        self, position: int, *, driver: int | None = None, trailer: int | None = None
    ) -> None:
        rec = self._touch(position)
        # Reassignment changes the membership of two chains, so the vacated
        # resource is dirty as well as the newly-assigned one.
        trailers = {rec.trailer}
        drivers = {rec.driver}
        if driver is not None:
            rec.driver = driver
            drivers.add(driver)
        if trailer is not None:
            rec.trailer = trailer
            trailers.add(trailer)
        self._recompute_dirty(
            set(), dirty_shift=rec, trailers=trailers, drivers=drivers
        )

    def set_shift_timing(
        self, position: int, start: int, arrivals: Sequence[int]
    ) -> None:
        rec = self._touch(position)
        rows = self._customer_rows(rec)
        self._unregister_ops(rec)
        rec.start = start
        rec.arrivals = list(arrivals)
        self._register_ops(rec)
        self._recompute_dirty(
            rows, dirty_shift=rec,
            trailers=(rec.trailer,), drivers=(rec.driver,),
        )

    def insert_shift(self, rec: ShiftRec, position: int | None = None) -> int:
        if position is None:
            position = len(self.shifts)
        self.shifts.insert(position, rec)
        if self._journal is not None:
            self._journal.append({"kind": "insert", "rec": rec})
        self._register_ops(rec)
        # A caller-supplied record may arrive with stale cached fields, so
        # derive it before the totals are summed.
        self._recompute_dirty(
            self._customer_rows(rec), dirty_shift=rec,
            trailers=(rec.trailer,), drivers=(rec.driver,),
        )
        return position

    def remove_shift(self, position: int) -> ShiftRec:
        rec = self.shifts[position]
        if self._journal is not None:
            self._journal.append({"kind": "remove", "position": position, "rec": rec})
        rows = self._customer_rows(rec)
        self._unregister_ops(rec)
        self.shifts.pop(position)
        self._recompute_dirty(
            rows, trailers=(rec.trailer,), drivers=(rec.driver,),
        )
        return rec

    def _adjust_delivery(
        self, point: int, arrival: int, quantity: float, sign: int
    ) -> None:
        fi = self.fi
        row = fi.customer_row[point]
        if row < 0:
            return
        if sign < 0:
            self._cust_ops[row].remove((arrival, quantity))
        else:
            self._cust_ops[row].append((arrival, quantity))
        if quantity > 0.0:
            step = arrival // fi.unit
            if step < 0:
                step = 0
            elif step >= fi.horizon:
                step = fi.horizon - 1
            self._deliveries[row, step] += sign * quantity

    def _sum_safety_kg_min(self) -> None:
        """Re-sum the safety penalty from the per-customer cache.

        Float delta accounting drifts in the last bits, so an apply/revert pair
        would not restore the original value and the exactness test would
        become an approximate one.  This is one pass over a list of tuples with
        no inventory projection, so it is cheap next to the numpy reprojection
        of even a single dirty customer.
        """
        total = 0.0
        for entry in self._cust_tank:
            total += entry[4]
        self._safety_kg_min = total

    def _sum_shift_totals(self) -> None:
        """Re-add the per-shift totals and rebuild the resource membership index.

        Deliberately a full sum instead of a running delta.  Delta accounting
        on floats drifts in the last bits, which makes ``rollback`` inexact and
        turns an exactness test into an approximate one.  The membership index
        is rebuilt in the same pass for the same reason: derived from the shift
        list every time, it cannot fall out of step with it, and the chain
        walks then never re-scan the full shift list.  No derivation and no
        inventory work happens here.

        The compiled version is the same loop verbatim (same full re-sum, same
        membership rebuild); it exists because this runs once per mutation and
        was the largest single line in the V2.17 profile.
        """
        if _sum_shift_totals_fast is not None:
            (
                self._shift_errors,
                self._shift_warnings,
                self._cost,
                self._distance,
                self._delivered,
                self._loaded,
            ) = _sum_shift_totals_fast(
                self.shifts, self._by_trailer, self._by_driver
            )
            return
        for bucket in self._by_trailer:
            bucket.clear()
        for bucket in self._by_driver:
            bucket.clear()
        n_trailers = len(self._by_trailer)
        n_drivers = len(self._by_driver)
        for rec in self.shifts:
            if not rec.points:
                continue
            if 0 <= rec.trailer < n_trailers:
                self._by_trailer[rec.trailer].append(rec)
            if 0 <= rec.driver < n_drivers:
                self._by_driver[rec.driver].append(rec)

        shift_errors = 0
        shift_warnings = 0
        cost = 0.0
        distance = 0.0
        delivered = 0.0
        loaded = 0.0
        for rec in self.shifts:
            shift_errors += rec.local_errors
            shift_warnings += rec.local_warnings
            cost += rec.cost
            distance += rec.distance
            delivered += rec.delivered
            loaded += rec.loaded
        self._shift_errors = shift_errors
        self._shift_warnings = shift_warnings
        self._cost = cost
        self._distance = distance
        self._delivered = delivered
        self._loaded = loaded

    def _recompute_dirty(
        self,
        rows: Iterable[int],
        *,
        dirty_shift: ShiftRec | None = None,
        trailers: Iterable[int] | None = None,
        drivers: Iterable[int] | None = None,
    ) -> None:
        """Rescore after a mutation, touching only what the mutation changed.

        Re-derives just ``dirty_shift``, re-walks just the named resource
        chains, and reprojects just the named customer rows.  ``trailers`` and
        ``drivers`` default to *every* chain, which is always correct and is
        what a multi-shift edit (rollback, removal) uses.
        """
        if dirty_shift is not None:
            self._recompute_shift(dirty_shift)
        self._sum_shift_totals()
        self._recompute_chains(trailers, drivers)
        for row in rows:
            self._recompute_customer_tank(row)
            self._recompute_customer_callin(row)
        self._sum_safety_kg_min()

    def refresh(
        self,
        rows: Iterable[int],
        *,
        trailers: Iterable[int] | None = None,
        drivers: Iterable[int] | None = None,
    ) -> None:
        """Public hook for operators that manage their own dirty set."""
        self._recompute_dirty(rows, trailers=trailers, drivers=drivers)


def state_from_files(
    instance: Instance, solution: Solution, *, score_days: int | None = None
) -> SearchState:
    return SearchState.from_solution(instance, solution, score_days=score_days)
