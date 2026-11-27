"""Retraining and model promotion pipeline for Wheat Yield Prediction.

Implements Stage 4 requirements:
1. Calculates PSI across all numeric features between train and later.csv.
2. Evaluates model v1 MAE by Tehsil group to identify worst-served group.
3. Retrains model v2 on cleaned train plus first 70% of later.csv.
4. Evaluates v1 vs v2 on the remaining 30% holdout check.
5. Applies client promotion rule (promote v2 only if MAE improves by >= 5%).
6. Updates MLflow Model Registry champion alias and findings.yaml.
"""

import os
import shutil
import sys
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml

from src.wheat_yield.drift_detection import NUMERICAL_FEATURES, calculate_psi
from src.wheat_yield.logger import get_logger
from src.wheat_yield.model_training import (
    ALL_FEATURES,
    build_pipeline,
    compute_data_sha256,
    load_config,
)

logger = get_logger("wheat_yield.retraining")


def run_retraining_pipeline(
    config_path: str = "params.yaml",
    findings_path: str = "findings.yaml",
) -> Dict[str, Any]:
    """Execute the end-to-end retraining and promotion pipeline."""
    logger.info("Initiating Stage 4 Retraining and Promotion Pipeline...", extra={"event_type": "retraining_started"})

    params = load_config(config_path)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = params.get("tracking", {}).get("experiment_name", "wheat-yield-prediction")
    mlflow.set_experiment(experiment_name)

    train_path = params.get("data", {}).get("path", "data/processed/cleaned_train.csv")
    later_path = "data/raw/later.csv"

    if not os.path.exists(train_path) or not os.path.exists(later_path):
        raise FileNotFoundError(f"Required datasets not found: {train_path} or {later_path}")

    df_train = pd.read_csv(train_path)
    df_later = pd.read_csv(later_path)

    # -------------------------------------------------------------
    # Step 1: PSI calculation per numeric column
    # -------------------------------------------------------------
    psi_dict: Dict[str, float] = {}
    for col in NUMERICAL_FEATURES:
        if col in df_train.columns and col in df_later.columns:
            psi = calculate_psi(
                df_train[col].to_numpy(dtype=float),
                df_later[col].to_numpy(dtype=float),
            )
            psi_dict[col] = psi

    most_drifted_feature = max(psi_dict, key=psi_dict.get)
    logger.info(
        f"Drift PSI computed. Most drifted: {most_drifted_feature} (PSI={psi_dict[most_drifted_feature]})",
        extra={"event_type": "drift_psi_computed", "details": psi_dict}
    )

    # -------------------------------------------------------------
    # Step 2: Compute v1 MAE on later.csv separately for each tehsil
    # -------------------------------------------------------------
    model_v1_path = "models/model.joblib"
    if not os.path.exists(model_v1_path):
        model_v1_path = "models/best_model.pkl"

    model_v1 = joblib.load(model_v1_path)

    target_col = params.get("target", "yield_qtl_per_acre")
    tehsil_groups = ["Tehsil North", "Tehsil South", "Tehsil West"]
    mae_by_group: Dict[str, float] = {}

    for group in tehsil_groups:
        group_df = df_later[df_later["tehsil"] == group]
        if len(group_df) > 0:
            preds_group = model_v1.predict(group_df[ALL_FEATURES])
            group_mae = float(mean_absolute_error(group_df[target_col], preds_group))
            mae_by_group[group] = round(group_mae, 4)

    worst_served_group = max(mae_by_group, key=mae_by_group.get)
    logger.info(
        f"MAE by group evaluated. Worst-served: {worst_served_group} with MAE {mae_by_group[worst_served_group]}",
        extra={"event_type": "group_mae_computed", "details": mae_by_group}
    )

    # -------------------------------------------------------------
    # Step 3: Sort later.csv by date, split into 70% train and 30% check
    # -------------------------------------------------------------
    df_later_sorted = df_later.sort_values("date").reset_index(drop=True)
    split_idx = int(0.70 * len(df_later_sorted))
    later_train = df_later_sorted.iloc[:split_idx]
    later_check = df_later_sorted.iloc[split_idx:]

    # Combined training set: cleaned train + first 70% of later.csv
    combined_train = pd.concat([df_train[ALL_FEATURES + [target_col]], later_train[ALL_FEATURES + [target_col]]], ignore_index=True)

    X_train_comb = combined_train[ALL_FEATURES]
    y_train_comb = combined_train[target_col]

    X_check = later_check[ALL_FEATURES]
    y_check = later_check[target_col]

    # -------------------------------------------------------------
    # Step 4: Retrain model v2 and evaluate v1 vs v2 on last 30%
    # -------------------------------------------------------------
    model_params = params.get("model", {})
    pipeline_v2 = build_pipeline(model_params)

    data_sha = compute_data_sha256()

    with mlflow.start_run(run_name="retrain-v2") as run:
        pipeline_v2.fit(X_train_comb, y_train_comb)

        preds_v1_check = model_v1.predict(X_check)
        preds_v2_check = pipeline_v2.predict(X_check)

        mae_v1 = float(mean_absolute_error(y_check, preds_v1_check))
        mae_v2 = float(mean_absolute_error(y_check, preds_v2_check))

        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_params({"retrain_split": "70/30", "data_sources": "cleaned_train + later_70pct"})
        mlflow.log_metrics({
            "mae_v1_on_check": round(mae_v1, 4),
            "mae_v2_on_check": round(mae_v2, 4),
            "mae_improvement_pct": round(((mae_v1 - mae_v2) / mae_v1) * 100, 2),
        })

        mlflow.sklearn.log_model(
            sk_model=pipeline_v2,
            name="model",
            serialization_format="cloudpickle",
        )

        run_id_v2 = run.info.run_id

    # -------------------------------------------------------------
    # Step 5: Apply client promotion rule
    # Rule: Promote v2 only if its MAE on that last 30% is at least 5% lower than v1's.
    # -------------------------------------------------------------
    relative_improvement = (mae_v1 - mae_v2) / mae_v1
    promote = (mae_v2 <= mae_v1 * 0.95) or (relative_improvement >= 0.05)
    decision = "promote" if promote else "keep_v1"

    logger.info(
        f"Promotion evaluation: v1 MAE={mae_v1:.4f}, v2 MAE={mae_v2:.4f}, improvement={relative_improvement:.2%}. Decision: {decision}",
        extra={"event_type": "promotion_decision", "details": {"decision": decision, "mae_v1": mae_v1, "mae_v2": mae_v2}}
    )

    # Register v2 in MLflow Model Registry
    reg_model_name = "wheat-yield-champion"
    model_uri = f"runs:/{run_id_v2}/model"
    model_v2_reg = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
    v2_version_num = int(model_v2_reg.version)

    client = MlflowClient()

    if promote:
        # v2 becomes champion
        client.set_registered_model_alias(reg_model_name, "champion", str(v2_version_num))
        champion_version = v2_version_num
        # Save retrained champion to models/model.joblib
        joblib.dump(pipeline_v2, "models/model.joblib")
        logger.info(f"Promoted v2 as champion (version {v2_version_num})", extra={"event_type": "champion_promoted"})
    else:
        # Keep v1 as champion
        client.set_registered_model_alias(reg_model_name, "champion", "1")
        champion_version = 1
        logger.info("Keeping v1 as champion", extra={"event_type": "champion_retained"})

    # -------------------------------------------------------------
    # Step 6: Automatically update findings.yaml
    # -------------------------------------------------------------
    update_findings_monitoring(
        findings_path=findings_path,
        psi_dict=psi_dict,
        most_drifted=most_drifted_feature,
        mae_by_group=mae_by_group,
        worst_served_group=worst_served_group,
        mae_v1=round(mae_v1, 4),
        mae_v2=round(mae_v2, 4),
        decision=decision,
        champion_version=champion_version,
    )

    results = {
        "psi": psi_dict,
        "most_drifted_feature": most_drifted_feature,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served_group,
        "mae_v1_on_check": round(mae_v1, 4),
        "mae_v2_on_check": round(mae_v2, 4),
        "decision": decision,
        "champion_version": champion_version,
    }
    return results


def update_findings_monitoring(
    findings_path: str,
    psi_dict: Dict[str, float],
    most_drifted: str,
    mae_by_group: Dict[str, float],
    worst_served_group: str,
    mae_v1: float,
    mae_v2: float,
    decision: str,
    champion_version: int,
):
    """Write monitoring metrics and decision into findings.yaml and reports/findings.yaml."""
    if os.path.exists(findings_path):
        with open(findings_path, "r", encoding="utf-8") as f:
            findings = yaml.safe_load(f) or {}

        if "monitoring" not in findings:
            findings["monitoring"] = {}

        findings["monitoring"]["psi"] = psi_dict
        findings["monitoring"]["most_drifted_feature"] = most_drifted
        findings["monitoring"]["mae_by_group"] = mae_by_group
        findings["monitoring"]["worst_served_group"] = worst_served_group
        findings["monitoring"]["mae_v1_on_check"] = mae_v1
        findings["monitoring"]["mae_v2_on_check"] = mae_v2
        findings["monitoring"]["decision"] = decision

        if "tracking" in findings:
            findings["tracking"]["champion_version"] = champion_version

        # Write root findings.yaml
        with open(findings_path, "w", encoding="utf-8") as f:
            yaml.dump(findings, f, sort_keys=False)

        # Write duplicate to reports/findings.yaml
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        reports_findings_path = os.path.join(reports_dir, "findings.yaml")
        shutil.copyfile(findings_path, reports_findings_path)

        print(f"Updated {findings_path} and {reports_findings_path}")


if __name__ == "__main__":
    run_retraining_pipeline()
