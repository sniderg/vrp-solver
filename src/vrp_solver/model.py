from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeWindow:
    start: int
    end: int


@dataclass(frozen=True)
class Driver:
    index: int
    min_inter_shift_duration: int
    max_driving_duration: int
    trailer_ids: tuple[int, ...]
    time_windows: tuple[TimeWindow, ...]
    layover_duration: int
    time_cost: float
    layover_cost: float


@dataclass(frozen=True)
class Trailer:
    index: int
    capacity: float
    initial_quantity: float
    distance_cost: float


@dataclass(frozen=True)
class Source:
    index: int
    allowed_trailers: tuple[int, ...]
    setup_time: int


@dataclass(frozen=True)
class Order:
    quantity: float
    earliest_time: int
    latest_time: int
    quantity_flexibility: int

    @property
    def min_quantity_to_satisfy(self) -> float:
        return self.quantity * self.quantity_flexibility / 100.0


@dataclass(frozen=True)
class Customer:
    index: int
    layover_customer: bool
    call_in: bool
    orders: tuple[Order, ...]
    setup_time: int
    time_windows: tuple[TimeWindow, ...]
    allowed_trailers: tuple[int, ...]
    forecast: tuple[float, ...]
    capacity: float
    initial_tank_quantity: float
    min_operation_quantity: float
    safety_level: float


@dataclass(frozen=True)
class Instance:
    name: str | None
    unit: int
    horizon: int
    time_matrix: tuple[tuple[int, ...], ...]
    distance_matrix: tuple[tuple[float, ...], ...]
    base_index: int
    drivers: tuple[Driver, ...]
    trailers: tuple[Trailer, ...]
    sources: tuple[Source, ...]
    customers: tuple[Customer, ...]

    @property
    def latest_time(self) -> int:
        return (self.horizon + 1) * self.unit

    @property
    def customer_by_point(self) -> dict[int, Customer]:
        return {customer.index: customer for customer in self.customers}

    @property
    def source_by_point(self) -> dict[int, Source]:
        return {source.index: source for source in self.sources}

    def point_kind(self, point: int) -> str:
        if point == self.base_index:
            return "base"
        if point in self.source_by_point:
            return "source"
        if point in self.customer_by_point:
            return "customer"
        return "unknown"

    def setup_time_for_point(self, point: int) -> int:
        if point in self.source_by_point:
            return self.source_by_point[point].setup_time
        if point in self.customer_by_point:
            return self.customer_by_point[point].setup_time
        return 0


@dataclass(frozen=True)
class Operation:
    point: int
    arrival: int
    quantity: float


@dataclass(frozen=True)
class Shift:
    index: int
    driver: int
    trailer: int
    start: int
    operations: tuple[Operation, ...]


@dataclass(frozen=True)
class Solution:
    shifts: tuple[Shift, ...]
