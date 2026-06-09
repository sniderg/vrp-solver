from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

from .model import Instance, Operation, Shift


@dataclass(frozen=True)
class RouteStats:
    sequence: tuple[int, ...]
    distance: float
    travel_time: int
    duration: int
    arrival_offsets: tuple[int, ...]
    end_offset: int


class RouteCache:
    """Memoize small route computations for one instance.

    Candidate generation repeatedly evaluates the same customer/source sequences
    across many starts, drivers, and trailers. This cache keeps those sequence
    computations instance-local and hashable without trying to cache full
    continuous inventory states.
    """

    def __init__(self, instance: Instance):
        self.instance = instance
        self.base = instance.base_index

    def stats(
        self,
        sequence: tuple[int, ...],
        *,
        start_point: int | None = None,
        end_point: int | None = None,
        include_setup: bool = True,
    ) -> RouteStats:
        return self._stats(
            tuple(sequence),
            self.base if start_point is None else start_point,
            self.base if end_point is None else end_point,
            include_setup,
        )

    @lru_cache(maxsize=500_000)
    def _stats(
        self,
        sequence: tuple[int, ...],
        start_point: int,
        end_point: int,
        include_setup: bool,
    ) -> RouteStats:
        previous = start_point
        distance = 0.0
        travel_time = 0
        elapsed = 0
        arrivals: list[int] = []

        for point in sequence:
            leg_time = self.instance.time_matrix[previous][point]
            distance += self.instance.distance_matrix[previous][point]
            travel_time += leg_time
            elapsed += leg_time
            arrivals.append(elapsed)
            if include_setup:
                elapsed += self.instance.setup_time_for_point(point)
            previous = point

        return_time = self.instance.time_matrix[previous][end_point]
        distance += self.instance.distance_matrix[previous][end_point]
        travel_time += return_time
        elapsed += return_time
        return RouteStats(
            sequence=sequence,
            distance=distance,
            travel_time=travel_time,
            duration=elapsed,
            arrival_offsets=tuple(arrivals),
            end_offset=elapsed,
        )

    def shift_distance(self, shift: Shift) -> float:
        return self.stats(tuple(operation.point for operation in shift.operations)).distance

    def best_order(
        self,
        customers: tuple[int, ...],
        *,
        start_point: int | None = None,
        end_point: int | None = None,
        include_setup: bool = True,
        max_bruteforce: int = 8,
    ) -> RouteStats:
        return self._best_order(
            tuple(sorted(customers)),
            self.base if start_point is None else start_point,
            self.base if end_point is None else end_point,
            include_setup,
            max_bruteforce,
        )

    @lru_cache(maxsize=250_000)
    def _best_order(
        self,
        customers: tuple[int, ...],
        start_point: int,
        end_point: int,
        include_setup: bool,
        max_bruteforce: int,
    ) -> RouteStats:
        if len(customers) <= 1:
            return self._stats(customers, start_point, end_point, include_setup)
        if len(customers) <= max_bruteforce:
            return min(
                (
                    self._stats(tuple(order), start_point, end_point, include_setup)
                    for order in permutations(customers)
                ),
                key=lambda stats: (stats.distance, stats.travel_time, stats.sequence),
            )
        return self._held_karp(customers, start_point, end_point, include_setup)

    def _held_karp(
        self,
        customers: tuple[int, ...],
        start_point: int,
        end_point: int,
        include_setup: bool,
    ) -> RouteStats:
        n = len(customers)
        setup = {
            point: self.instance.setup_time_for_point(point) if include_setup else 0
            for point in customers
        }
        states: dict[tuple[int, int], tuple[float, int, tuple[int, ...]]] = {}
        for index, point in enumerate(customers):
            mask = 1 << index
            states[(mask, index)] = (
                self.instance.distance_matrix[start_point][point],
                self.instance.time_matrix[start_point][point] + setup[point],
                (point,),
            )

        for size in range(2, n + 1):
            next_states: dict[tuple[int, int], tuple[float, int, tuple[int, ...]]] = {}
            for mask, last in [key for key in states if key[0].bit_count() == size - 1]:
                distance, elapsed, sequence = states[(mask, last)]
                last_point = customers[last]
                for nxt, point in enumerate(customers):
                    bit = 1 << nxt
                    if mask & bit:
                        continue
                    next_mask = mask | bit
                    candidate = (
                        distance + self.instance.distance_matrix[last_point][point],
                        elapsed + self.instance.time_matrix[last_point][point] + setup[point],
                        (*sequence, point),
                    )
                    key = (next_mask, nxt)
                    if key not in next_states or candidate < next_states[key]:
                        next_states[key] = candidate
            states.update(next_states)

        full_mask = (1 << n) - 1
        best: tuple[float, int, tuple[int, ...]] | None = None
        for last, point in enumerate(customers):
            distance, elapsed, sequence = states[(full_mask, last)]
            candidate = (
                distance + self.instance.distance_matrix[point][end_point],
                elapsed + self.instance.time_matrix[point][end_point],
                sequence,
            )
            if best is None or candidate < best:
                best = candidate

        assert best is not None
        sequence = best[2]
        return self._stats(sequence, start_point, end_point, include_setup)


def shift_sequence(shift: Shift) -> tuple[int, ...]:
    return tuple(operation.point for operation in shift.operations)


def operation_sequence(operations: tuple[Operation, ...]) -> tuple[int, ...]:
    return tuple(operation.point for operation in operations)
