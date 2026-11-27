"""Data drift detection module using Population Stability Index (PSI).

Computes PSI for numerical features between baseline reference data and production
incoming data, generates drift audit reports, and flags significant drift events.
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import yaml

from src.wheat_yield.logger import get_logger

logger = get_logger("wheat_yield.drift")

NUMERICAL_FEATURES: List[str] = [
    "rainfall_mm",
    "fertilizer_kg",
    "sowing_delay_days",
    "irrigation_count",
    "soil_ph",
    "farm_acres",
]


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate the Population Stability Index (PSI) between two distributions.

    Args:
        expected: Baseline / reference feature array (e.g., training data).
        actual: Target / monitored feature array (e.g., later incoming data).
        num_bins: Number of quantile bins to partition the feature distribution.
        epsilon: Small smoothing constant to prevent zero division or log(0).

    Returns:
        PSI metric as float.
    """
    # Remove NaNs
    exp_clean = expected[~np.isnan(expected)]
    act_clean = actual[~np.isnan(actual)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    # Determine bin edges from reference dataset quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp_clean, quantiles)
    # Ensure unique monotonic bin edges
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    # Expand outer boundaries to capture all potential range values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate counts and proportions in each bin
    exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
    act_counts, _ = np.histogram(act_clean, bins=bin_edges)

    exp_pct = exp_counts / len(exp_clean)
    act_pct = act_counts / len(act_clean)

    # Apply epsilon smoothing
    exp_pct = np.where(exp_pct == 0, epsilon, exp_pct)
    act_pct = np.where(act_pct == 0, epsilon, act_pct)

    # Compute PSI summation
    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi_value, 4))


def run_drift_analysis(
    baseline_path: str = "data/processed/cleaned_train.csv",
    incoming_path: str = "data/raw/later.csv",
    output_report_path: str = "reports/drift_report.json",
    psi_warning_threshold: float = 0.1,
    psi_drift_threshold: float = 0.2,
) -> Dict[str, Any]:
    """Execute complete drift detection across all numerical features."""
    logger.info(
        f"Starting drift analysis: baseline={baseline_path}, incoming={incoming_path}",
        extra={"event_type": "drift_analysis_started"}
    )

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline file missing: {baseline_path}")
    if not os.path.exists(incoming_path):
        raise FileNotFoundError(f"Incoming monitoring file missing: {incoming_path}")

    df_base = pd.read_csv(baseline_path)
    df_incoming = pd.read_csv(incoming_path)

    psi_scores: Dict[str, float] = {}

    for feature in NUMERICAL_FEATURES:
        if feature in df_base.columns and feature in df_incoming.columns:
            psi = calculate_psi(
                df_base[feature].to_numpy(dtype=float),
                df_incoming[feature].to_numpy(dtype=float),
            )
            psi_scores[feature] = psi

            if psi >= psi_drift_threshold:
                logger.warning(
                    f"Significant data drift detected on feature '{feature}' (PSI={psi} >= {psi_drift_threshold})",
                    extra={
                        "event_type": "drift_alert",
                        "details": {"feature": feature, "psi": psi, "severity": "HIGH"}
                    }
                )
            elif psi >= psi_warning_threshold:
                logger.warning(
                    f"Moderate data drift observed on feature '{feature}' (PSI={psi})",
                    extra={
                        "event_type": "drift_warning",
                        "details": {"feature": feature, "psi": psi, "severity": "MODERATE"}
                    }
                )

    most_drifted = max(psi_scores, key=psi_scores.get)
    max_psi = psi_scores[most_drifted]

    report = {
        "status": "completed",
        "features_analyzed": len(psi_scores),
        "psi_by_feature": psi_scores,
        "most_drifted_feature": most_drifted,
        "highest_psi": max_psi,
        "retraining_recommended": max_psi >= psi_warning_threshold,
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"Drift analysis completed. Most drifted feature: '{most_drifted}' (PSI={max_psi})",
        extra={
            "event_type": "drift_analysis_completed",
            "details": report
        }
    )

    return report


if __name__ == "__main__":
    run_drift_analysis()
