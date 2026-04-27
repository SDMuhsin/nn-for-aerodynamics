"""2024 ML4CFD competition scoring.

Implements `compute_global_score` exactly as defined in the official starting
kit notebooks `5_Scoring.ipynb` and `3_Reproduce_baseline_result.ipynb`.

Outputs four numbers (all out of 100):
    ML, Physics, OOD, Global

Global = 100 * (0.4 * ML + 0.3 * Physics + 0.3 * OOD)

ML and OOD are 75 % accuracy + 25 % speed-up (log-scaled, capped at 10 000×).
Physics is pure accuracy (no speed-up component).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

# These are baked into the competition. DO NOT change them — reproduction
# validity hinges on the scoring matching exactly.
THRESHOLDS: Dict[str, Tuple[float, float, str]] = {
    "x-velocity": (0.01, 0.02, "min"),
    "y-velocity": (0.01, 0.02, "min"),
    "pressure": (0.002, 0.01, "min"),
    "pressure_surfacic": (0.008, 0.02, "min"),
    "turbulent_viscosity": (0.05, 0.1, "min"),
    "mean_relative_drag": (0.4, 5.0, "min"),
    "mean_relative_lift": (0.1, 0.3, "min"),
    "spearman_correlation_drag": (0.8, 0.9, "max"),
    "spearman_correlation_lift": (0.96, 0.99, "max"),
}

CONFIGURATION = {
    "coefficients": {"ML": 0.4, "OOD": 0.3, "Physics": 0.3},
    "ratioRelevance": {"Speed-up": 0.25, "Accuracy": 0.75},
    "valueByColor": {"g": 2, "o": 1, "r": 0},
    "maxSpeedRatioAllowed": 10000,
    "reference_mean_simulation_time": 1500,
}

PHY_VARIABLES_FOR_SCORE = (
    "mean_relative_drag",
    "mean_relative_lift",
    "spearman_correlation_drag",
    "spearman_correlation_lift",
)


def _color(value: float, thr: Tuple[float, float, str]) -> str:
    lo, hi, mode = thr
    if mode == "min":
        if value < lo:
            return "g"
        if value < hi:
            return "o"
        return "r"
    if mode == "max":
        if value < lo:
            return "r"
        if value < hi:
            return "o"
        return "g"
    raise ValueError(f"Unknown threshold mode: {mode}")


def _accuracy_score(metrics: Dict[str, float]) -> Tuple[float, Dict[str, str]]:
    """Return accuracy fraction in [0, 1] and per-variable colors."""
    colors = {var: _color(val, THRESHOLDS[var]) for var, val in metrics.items()}
    by_color = CONFIGURATION["valueByColor"]
    points = sum(by_color[c] for c in colors.values())
    max_points = len(colors) * max(by_color.values())
    return points / max_points if max_points else 0.0, colors


def _speedup_score(speedup: float) -> float:
    max_ratio = CONFIGURATION["maxSpeedRatioAllowed"]
    if speedup <= 0:
        return 0.0
    return max(0.0, min(math.log10(speedup) / math.log10(max_ratio), 1.0))


def extract_metrics(fc_metrics_test: dict, fc_metrics_test_ood: dict) -> dict:
    """Pull the variables consumed by the score from raw LIPS evaluation output.

    Mirrors `5_Scoring.ipynb` and notebook 3 cell 49.
    """
    test = fc_metrics_test["test"]
    test_ood = fc_metrics_test_ood["test_ood"]

    ml = dict(test["ML"]["MSE_normalized"])
    ml["pressure_surfacic"] = test["ML"]["MSE_normalized_surfacic"]["pressure"]

    phy = {k: test["Physics"][k] for k in PHY_VARIABLES_FOR_SCORE}

    ood_ml = dict(test_ood["ML"]["MSE_normalized"])
    ood_ml["pressure_surfacic"] = test_ood["ML"]["MSE_normalized_surfacic"]["pressure"]
    ood_phy = {k: test_ood["Physics"][k] for k in PHY_VARIABLES_FOR_SCORE}
    ood = {**ood_ml, **ood_phy}

    return {"ML": ml, "Physics": phy, "OOD": ood}


def compute_global_score(metrics: dict, speedup: dict) -> dict:
    """Compute the four ML4CFD subscores from extracted metrics.

    Args:
        metrics: dict with keys "ML", "Physics", "OOD" mapping to dicts of
            {variable: error}. Variable names must match THRESHOLDS keys.
        speedup: dict with keys "ML" and "OOD", values = (1500 / mean_sim_time).

    Returns:
        {"ML": <0..100>, "Physics": <0..100>, "OOD": <0..100>,
         "Global": <0..100>, "colors": {...}, "components": {...}}
    """
    coeffs = CONFIGURATION["coefficients"]
    rr = CONFIGURATION["ratioRelevance"]

    ml_acc, ml_colors = _accuracy_score(metrics["ML"])
    phy_acc, phy_colors = _accuracy_score(metrics["Physics"])
    ood_acc, ood_colors = _accuracy_score(metrics["OOD"])

    ml_speed = _speedup_score(speedup.get("ML", 0.0))
    ood_speed = _speedup_score(speedup.get("OOD", 0.0))

    ml_subscore = ml_acc * rr["Accuracy"] + ml_speed * rr["Speed-up"]
    phy_subscore = phy_acc  # pure accuracy
    ood_subscore = ood_acc * rr["Accuracy"] + ood_speed * rr["Speed-up"]

    global_score = 100.0 * (
        coeffs["ML"] * ml_subscore
        + coeffs["Physics"] * phy_subscore
        + coeffs["OOD"] * ood_subscore
    )

    return {
        "ML": 100.0 * ml_subscore,
        "Physics": 100.0 * phy_subscore,
        "OOD": 100.0 * ood_subscore,
        "Global": global_score,
        "colors": {"ML": ml_colors, "Physics": phy_colors, "OOD": ood_colors},
        "components": {
            "ml_acc": ml_acc,
            "phy_acc": phy_acc,
            "ood_acc": ood_acc,
            "ml_speed": ml_speed,
            "ood_speed": ood_speed,
        },
    }


def score_from_lips_output(
    fc_metrics_test: dict,
    fc_metrics_test_ood: dict,
    test_mean_simulation_time: float,
    test_ood_mean_simulation_time: float,
) -> dict:
    """Convenience wrapper: extract metrics + compute speedup + score."""
    metrics = extract_metrics(fc_metrics_test, fc_metrics_test_ood)
    ref = CONFIGURATION["reference_mean_simulation_time"]
    speedup = {
        "ML": ref / test_mean_simulation_time if test_mean_simulation_time > 0 else 0.0,
        "OOD": ref / test_ood_mean_simulation_time if test_ood_mean_simulation_time > 0 else 0.0,
    }
    return compute_global_score(metrics, speedup)
