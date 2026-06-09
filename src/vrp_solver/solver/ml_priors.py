from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from ..model import Instance, Customer, Solution
from ..inventory import days_of_inventory, tank_events

FEATURE_NAMES = [
    # Original features
    "inv_ratio",
    "safety_ratio",
    "doi",
    "avg_demand",
    "distance_depot",
    "distance_source",
    # New features — Phase 1
    "forecast_slope",
    "demand_variance",
    "tw_tightness",
    "n_allowed_trailers",
    "cluster_accessibility",
    "capacity_headroom",
    "has_orders",
    # Interaction features — Phase 1
    "inv_x_doi",
    "dist_x_demand",
]

# Default weights initialised from domain knowledge.
# Negative weight = "higher value makes the customer *less* urgent"
# Positive weight = "higher value makes the customer *more* urgent"
_DEFAULT_WEIGHTS: dict[str, float] = {
    "inv_ratio": -200.0,
    "safety_ratio": 300.0,
    "doi": -500.0,
    "avg_demand": 0.5,
    "distance_depot": -0.1,
    "distance_source": -0.5,
    # New features
    "forecast_slope": 200.0,        # Rising demand → more urgent
    "demand_variance": 50.0,        # Spiky demand → harder to schedule
    "tw_tightness": 300.0,          # Tight windows → need early commitment
    "n_allowed_trailers": -50.0,    # Fewer trailer options → needs priority
    "cluster_accessibility": -0.05, # Isolated → needs dedicated route
    "capacity_headroom": -200.0,    # Thin margin → breaches fast
    "has_orders": 500.0,            # Has orders → MIP must cover anyway
    # Interactions
    "inv_x_doi": -100.0,            # Low inv AND low DOI → very urgent
    "dist_x_demand": -0.001,        # Remote AND high demand → critical
}


@dataclass
class MLRoutePriors:
    weights: dict[str, float] = field(default_factory=lambda: _DEFAULT_WEIGHTS.copy())
    bias: float = 1000.0
    # Optional z-score normalization stats (Phase 5)
    feature_means: dict[str, float] | None = None
    feature_stds: dict[str, float] | None = None

    def save(self, path: str | Path) -> None:
        """Save weights to a JSON file."""
        data: dict = {
            "weights": self.weights,
            "bias": self.bias,
        }
        if self.feature_means is not None:
            data["feature_means"] = self.feature_means
        if self.feature_stds is not None:
            data["feature_stds"] = self.feature_stds
        with Path(path).open("w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load weights from a JSON file.

        Backwards-compatible: old weight files with 6 features are merged
        into the full feature set, keeping defaults for missing features.
        """
        with Path(path).open() as f:
            data = json.load(f)
        loaded_weights = data["weights"]
        # Start from defaults, overlay loaded values
        merged = _DEFAULT_WEIGHTS.copy()
        merged.update(loaded_weights)
        self.weights = merged
        self.bias = data["bias"]
        self.feature_means = data.get("feature_means")
        self.feature_stds = data.get("feature_stds")

    def _normalize(self, features: dict[str, float]) -> dict[str, float]:
        """Apply z-score normalization if stats are available."""
        if self.feature_means is None or self.feature_stds is None:
            return features
        normed = {}
        for name in FEATURE_NAMES:
            mean = self.feature_means.get(name, 0.0)
            std = self.feature_stds.get(name, 1.0)
            if std < 1e-12:
                std = 1.0
            normed[name] = (features.get(name, 0.0) - mean) / std
        return normed

    def compute_features(
        self,
        instance: Instance,
        customer: Customer,
        current_inventory: float,
        start_step: int,
    ) -> dict[str, float]:
        """Compute the feature vector for a customer at the start of a planning horizon."""
        # --- Original features ---
        inv_ratio = current_inventory / max(1.0, customer.capacity)
        safety_ratio = customer.safety_level / max(1.0, customer.capacity)

        # Average demand
        avg_demand = (
            sum(customer.forecast) / len(customer.forecast)
            if customer.forecast
            else 0.0
        )

        # Distance to depot and source
        distance_depot = float(instance.time_matrix[instance.base_index][customer.index])
        distance_source = float(
            min(
                instance.time_matrix[source.index][customer.index]
                for source in instance.sources
            )
            if instance.sources
            else distance_depot
        )

        # DOI calculation
        doi = days_of_inventory(
            instance,
            customer,
            current_inventory,
            start_step=start_step,
            lead_time_minutes=distance_source,
        )

        # --- New features (Phase 1) ---

        # Forecast slope: linear regression slope over next 7 days of forecast
        # Positive slope = demand is increasing
        forecast_window = min(
            7 * 1440 // max(instance.unit, 1),
            len(customer.forecast) - start_step,
        )
        if forecast_window > 1:
            window_forecast = [
                customer.forecast[start_step + i]
                for i in range(forecast_window)
                if start_step + i < len(customer.forecast)
            ]
            n = len(window_forecast)
            if n > 1:
                x_mean = (n - 1) / 2.0
                y_mean = sum(window_forecast) / n
                num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(window_forecast))
                denom = sum((i - x_mean) ** 2 for i in range(n))
                forecast_slope = num / denom if denom > 1e-12 else 0.0
            else:
                forecast_slope = 0.0
        else:
            forecast_slope = 0.0

        # Demand variance (coefficient of variation)
        if forecast_window > 1 and avg_demand > 1e-12:
            window_forecast = [
                customer.forecast[start_step + i]
                for i in range(forecast_window)
                if start_step + i < len(customer.forecast)
            ]
            variance = sum((f - avg_demand) ** 2 for f in window_forecast) / len(window_forecast)
            demand_variance = math.sqrt(variance) / avg_demand  # CV
        else:
            demand_variance = 0.0

        # Time-window tightness: fraction of total horizon that is "open"
        if customer.time_windows:
            total_open = sum(tw.end - tw.start for tw in customer.time_windows)
            horizon_minutes = instance.horizon * instance.unit
            tw_tightness = 1.0 - min(1.0, total_open / max(1.0, horizon_minutes))
        else:
            tw_tightness = 0.0  # No time windows = always open

        # Number of allowed trailers (normalized by total trailers)
        n_allowed_trailers = len(customer.allowed_trailers) / max(1.0, len(instance.trailers))

        # Cluster accessibility: average distance to 5 nearest non-call-in customers
        other_distances = sorted(
            instance.time_matrix[customer.index][other.index]
            for other in instance.customers
            if other.index != customer.index and not other.call_in
        )
        if other_distances:
            nearest = other_distances[:min(5, len(other_distances))]
            cluster_accessibility = sum(nearest) / len(nearest)
        else:
            cluster_accessibility = 0.0

        # Capacity headroom: usable capacity above safety level
        capacity_headroom = (customer.capacity - customer.safety_level) / max(1.0, customer.capacity)

        # Has orders (binary)
        has_orders = 1.0 if customer.orders else 0.0

        # --- Interaction features ---
        inv_x_doi = inv_ratio * doi
        dist_x_demand = distance_source * avg_demand

        return {
            "inv_ratio": inv_ratio,
            "safety_ratio": safety_ratio,
            "doi": doi,
            "avg_demand": avg_demand,
            "distance_depot": distance_depot,
            "distance_source": distance_source,
            "forecast_slope": forecast_slope,
            "demand_variance": demand_variance,
            "tw_tightness": tw_tightness,
            "n_allowed_trailers": n_allowed_trailers,
            "cluster_accessibility": cluster_accessibility,
            "capacity_headroom": capacity_headroom,
            "has_orders": has_orders,
            "inv_x_doi": inv_x_doi,
            "dist_x_demand": dist_x_demand,
        }

    def predict_prizes(
        self,
        instance: Instance,
        current_inventories: dict[int, float],
        start_day: int,
    ) -> dict[int, float]:
        """Predict the priority prizes for each customer at the start of a given day."""
        start_step = min((start_day * 1440) // instance.unit, instance.horizon - 1)
        prizes = {}
        for customer in instance.customers:
            if customer.call_in:
                continue
            curr_inv = current_inventories.get(customer.index, customer.initial_tank_quantity)
            features = self.compute_features(instance, customer, curr_inv, start_step)
            normed = self._normalize(features)

            # Linear model prediction: score = sum(w * x) + b
            score = sum(self.weights.get(name, 0.0) * normed[name] for name in FEATURE_NAMES) + self.bias
            # Prizes must be non-negative in prize-collecting formulations
            prizes[customer.index] = max(0.0, score)
        return prizes

    def predict_prizes_by_day(
        self,
        instance: Instance,
        solution: Solution,
        start_day: int,
        end_day: int,
    ) -> dict[int, dict[int, float]]:
        """Predict per-day prizes for each customer across the planning window.

        Returns: {day: {customer_index: prize}}

        Phase 2: Instead of computing features once at window start, we recompute
        per-day so that the MIP objective reflects evolving urgency.
        """
        # Project inventory at each day boundary
        day_inventories: dict[int, dict[int, float]] = {}
        day_inventories[start_day] = get_start_inventories(instance, solution, start_day)

        # For subsequent days, project forward using forecasts
        for day in range(start_day + 1, end_day):
            prev_inv = day_inventories[day - 1]
            next_inv = {}
            steps_per_day = 1440 // instance.unit
            for customer in instance.customers:
                if customer.call_in:
                    continue
                inv = prev_inv.get(customer.index, customer.initial_tank_quantity)
                day_start_step = (day - 1) * steps_per_day
                for s in range(steps_per_day):
                    step = day_start_step + s
                    if step < len(customer.forecast):
                        inv -= customer.forecast[step]
                next_inv[customer.index] = inv
            day_inventories[day] = next_inv

        # Now compute prizes per day
        prizes_by_day: dict[int, dict[int, float]] = {}
        for day in range(start_day, end_day):
            inventories = day_inventories[day]
            prizes_by_day[day] = self.predict_prizes(instance, inventories, day)

        return prizes_by_day

    def update_weights(
        self,
        features_by_customer: dict[int, dict[str, float]],
        predicted_y: dict[int, float],
        target_y: dict[int, float],
        lr: float,
        breach_weights: dict[int, float] | None = None,
    ) -> None:
        """Perform a structured subgradient descent step.

        Target y and predicted y are binary (1 if served, 0 otherwise).
        Loss = theta * y_predicted - theta * y_target
        Gradient w.r.t theta = y_predicted - y_target

        If breach_weights is provided (Phase 4), weight each customer's
        gradient contribution by its breach severity.
        """
        grad_W = {name: 0.0 for name in FEATURE_NAMES}
        grad_b = 0.0

        for c, feat in features_by_customer.items():
            diff = predicted_y.get(c, 0.0) - target_y.get(c, 0.0)
            if abs(diff) > 1e-5:
                w = breach_weights.get(c, 1.0) if breach_weights else 1.0
                normed = self._normalize(feat)
                for name in FEATURE_NAMES:
                    grad_W[name] += w * diff * normed.get(name, 0.0)
                grad_b += w * diff

        # Apply gradient descent update
        for name in FEATURE_NAMES:
            self.weights[name] = self.weights.get(name, 0.0) - lr * grad_W[name]
        self.bias -= lr * grad_b

    def compute_normalization_stats(
        self,
        all_features: list[dict[str, float]],
    ) -> None:
        """Compute z-score normalization stats from a collection of feature dicts.

        Phase 5: Call this once before training to set feature_means and feature_stds.
        """
        if not all_features:
            return
        n = len(all_features)
        means = {name: 0.0 for name in FEATURE_NAMES}
        for feat in all_features:
            for name in FEATURE_NAMES:
                means[name] += feat.get(name, 0.0)
        for name in FEATURE_NAMES:
            means[name] /= n

        stds = {name: 0.0 for name in FEATURE_NAMES}
        for feat in all_features:
            for name in FEATURE_NAMES:
                stds[name] += (feat.get(name, 0.0) - means[name]) ** 2
        for name in FEATURE_NAMES:
            stds[name] = math.sqrt(stds[name] / n)

        self.feature_means = means
        self.feature_stds = stds


def get_start_inventories(
    instance: Instance,
    solution: Solution,
    start_day: int,
) -> dict[int, float]:
    """Calculate the inventory of each customer at the start of start_day."""
    if start_day == 0:
        return {c.index: c.initial_tank_quantity for c in instance.customers}

    # Otherwise, project solution to get tank events
    events = tank_events(instance, solution)
    step_cutoff = (start_day * 1440) // instance.unit - 1

    inventories = {}
    for event in events:
        if event.step == step_cutoff:
            inventories[event.point] = event.ending_inventory

    # Fallback to initial quantity if not found
    for c in instance.customers:
        if c.index not in inventories:
            inventories[c.index] = c.initial_tank_quantity

    return inventories
