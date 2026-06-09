from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from .model import Instance


@dataclass(frozen=True)
class MatrixSummary:
    points: int
    directed_edges: int
    zero_distance_edges: int
    asymmetric_distance_edges: int
    asymmetric_time_edges: int
    max_distance_asymmetry: float
    max_time_asymmetry: int
    min_positive_distance: float
    median_distance: float
    max_distance: float
    min_positive_time: int
    median_time: float
    max_time: int
    median_speed_distance_per_hour: float


@dataclass(frozen=True)
class MovementEdge:
    origin: int
    destination: int
    origin_kind: str
    destination_kind: str
    distance: float
    time: int
    reverse_distance: float
    reverse_time: int
    distance_delta: float
    time_delta: int
    speed_distance_per_hour: float | None


def summarize_matrices(instance: Instance) -> MatrixSummary:
    n = len(instance.distance_matrix)
    distances = []
    times = []
    speeds = []
    zero_distance_edges = 0
    asymmetric_distance_edges = 0
    asymmetric_time_edges = 0
    max_distance_asymmetry = 0.0
    max_time_asymmetry = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            distance = instance.distance_matrix[i][j]
            time = instance.time_matrix[i][j]
            reverse_distance = instance.distance_matrix[j][i]
            reverse_time = instance.time_matrix[j][i]
            if abs(distance) < 1e-9:
                zero_distance_edges += 1
            else:
                distances.append(distance)
            if time > 0:
                times.append(time)
                if distance > 0:
                    speeds.append(distance / (time / 60.0))
            if abs(distance - reverse_distance) > 1e-9:
                asymmetric_distance_edges += 1
                max_distance_asymmetry = max(
                    max_distance_asymmetry,
                    abs(distance - reverse_distance),
                )
            if time != reverse_time:
                asymmetric_time_edges += 1
                max_time_asymmetry = max(max_time_asymmetry, abs(time - reverse_time))

    return MatrixSummary(
        points=n,
        directed_edges=n * (n - 1),
        zero_distance_edges=zero_distance_edges,
        asymmetric_distance_edges=asymmetric_distance_edges,
        asymmetric_time_edges=asymmetric_time_edges,
        max_distance_asymmetry=max_distance_asymmetry,
        max_time_asymmetry=max_time_asymmetry,
        min_positive_distance=min(distances),
        median_distance=median(distances),
        max_distance=max(distances),
        min_positive_time=min(times),
        median_time=median(times),
        max_time=max(times),
        median_speed_distance_per_hour=median(speeds),
    )


def movement_edges(instance: Instance) -> list[MovementEdge]:
    n = len(instance.distance_matrix)
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            distance = instance.distance_matrix[i][j]
            time = instance.time_matrix[i][j]
            speed = None if time <= 0 else distance / (time / 60.0)
            edges.append(
                MovementEdge(
                    origin=i,
                    destination=j,
                    origin_kind=instance.point_kind(i),
                    destination_kind=instance.point_kind(j),
                    distance=distance,
                    time=time,
                    reverse_distance=instance.distance_matrix[j][i],
                    reverse_time=instance.time_matrix[j][i],
                    distance_delta=distance - instance.distance_matrix[j][i],
                    time_delta=time - instance.time_matrix[j][i],
                    speed_distance_per_hour=speed,
                )
            )
    return edges


def nearest_neighbors(
    instance: Instance,
    *,
    k: int,
    metric: str,
) -> list[dict[str, object]]:
    if metric not in {"distance", "time"}:
        raise ValueError("metric must be `distance` or `time`")
    rows: list[dict[str, object]] = []
    matrix = instance.distance_matrix if metric == "distance" else instance.time_matrix

    for i, values in enumerate(matrix):
        candidates = [
            (j, values[j], instance.point_kind(j))
            for j in range(len(values))
            if j != i
        ]
        candidates.sort(key=lambda item: (item[1], item[0]))
        for rank, (j, value, kind) in enumerate(candidates[:k], start=1):
            rows.append(
                {
                    "origin": i,
                    "origin_kind": instance.point_kind(i),
                    "rank": rank,
                    "destination": j,
                    "destination_kind": kind,
                    metric: value,
                    "reverse_" + metric: matrix[j][i],
                }
            )
    return rows


def collocation_groups(instance: Instance) -> list[tuple[int, ...]]:
    n = len(instance.distance_matrix)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(n):
        for j in range(i + 1, n):
            if (
                abs(instance.distance_matrix[i][j]) < 1e-9
                and abs(instance.distance_matrix[j][i]) < 1e-9
                and instance.time_matrix[i][j] == 0
                and instance.time_matrix[j][i] == 0
            ):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for point in range(n):
        groups.setdefault(find(point), []).append(point)

    return tuple(tuple(group) for group in groups.values() if len(group) > 1)


def distance_time_outliers(instance: Instance, limit: int = 25) -> list[MovementEdge]:
    edges = [
        edge
        for edge in movement_edges(instance)
        if edge.speed_distance_per_hour is not None and edge.distance > 0
    ]
    avg_speed = mean(edge.speed_distance_per_hour for edge in edges if edge.speed_distance_per_hour)
    edges.sort(
        key=lambda edge: abs((edge.speed_distance_per_hour or avg_speed) - avg_speed),
        reverse=True,
    )
    return edges[:limit]


def asymmetry_outliers(
    instance: Instance,
    *,
    metric: str,
    limit: int = 25,
) -> list[MovementEdge]:
    if metric not in {"distance", "time"}:
        raise ValueError("metric must be `distance` or `time`")
    edges = movement_edges(instance)
    if metric == "distance":
        edges.sort(key=lambda edge: abs(edge.distance_delta), reverse=True)
    else:
        edges.sort(key=lambda edge: abs(edge.time_delta), reverse=True)
    return edges[:limit]
