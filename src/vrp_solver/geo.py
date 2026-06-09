from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from .model import Instance


@dataclass(frozen=True)
class GeoPoint:
    point: int
    kind: str
    x: float
    y: float
    cluster: int


def mds_coordinates(
    instance: Instance,
    *,
    clusters: int = 12,
    random_state: int = 42,
    dissimilarity: str = "distance",
) -> list[GeoPoint]:
    matrix = np.array(
        instance.distance_matrix if dissimilarity == "distance" else instance.time_matrix,
        dtype=float,
    )
    symmetric = (matrix + matrix.T) / 2.0

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        normalized_stress="auto",
        random_state=random_state,
        n_init=4,
        max_iter=600,
    )
    coords = mds.fit_transform(symmetric)

    kmeans = KMeans(n_clusters=clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(coords)

    return [
        GeoPoint(
            point=point,
            kind=instance.point_kind(point),
            x=float(coords[point, 0]),
            y=float(coords[point, 1]),
            cluster=int(labels[point]),
        )
        for point in range(len(coords))
    ]


def write_geo_csv(points: list[GeoPoint], output_csv: str | Path) -> None:
    import csv

    with Path(output_csv).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0].__dict__))
        writer.writeheader()
        writer.writerows(point.__dict__ for point in points)


def plot_geo(points: list[GeoPoint], output_png: str | Path) -> None:
    colors = {"base": "#111111", "source": "#d1495b", "customer": "#3a5a40"}
    markers = {"base": "*", "source": "s", "customer": "o"}
    fig, ax = plt.subplots(figsize=(9, 7))

    for kind in ["customer", "source", "base"]:
        xs = [point.x for point in points if point.kind == kind]
        ys = [point.y for point in points if point.kind == kind]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            c=colors[kind],
            marker=markers[kind],
            s=70 if kind != "customer" else 22,
            alpha=0.9 if kind != "customer" else 0.65,
            label=kind,
        )

    ax.set_title("MDS Reconstruction from Directed Matrix Symmetrization")
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)


def cluster_members(points: list[GeoPoint]) -> dict[int, list[int]]:
    clusters: dict[int, list[int]] = {}
    for point in points:
        clusters.setdefault(point.cluster, []).append(point.point)
    return clusters
