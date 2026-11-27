"""Model training module for Wheat Yield Prediction.

Trains multiple model architectures, logs parameters, metrics, artifacts,
and models to MLflow SQLite database, registers the champion model, and
saves the model artifact for production serving.
"""

import hashlib
import os
from typing import Any, Dict, List, Tuple
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import yaml

# Core feature definitions
CATEGORICAL_FEATURES: List[str] = ["seed_variety", "tehsil"]
NUMERICAL_FEATURES: List[str] = [
    "rainfall_mm",
    "fertilizer_kg",
    "sowing_delay_days",
    "irrigation_count",
    "soil_ph",
    "farm_acres",
]
ALL_FEATURES: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def compute_data_sha256(file_path: str = "data/raw/train.csv") -> str:
    """Compute SHA-256 hash of the training data file."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    # Fallback placeholder if raw csv not present
    return "0000000000000000000000000000000000000000000000000000000000000000"


def load_config(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration parameters from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline(model_params: Dict[str, Any]) -> Pipeline:
    """Construct Scikit-Learn preprocessing and regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**model_params)),
        ]
    )
    return pipeline


def train_models(config_path: str = "params.yaml") -> Tuple[Pipeline, str, float]:
    """Execute model training, MLflow tracking, and champion registration."""
    params = load_config(config_path)

    # MLflow tracking setup to SQLite backend
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = params.get("tracking", {}).get(
        "experiment_name", "wheat-yield-prediction"
    )
    mlflow.set_experiment(experiment_name)

    data_path = params.get("data", {}).get("path", "data/processed/cleaned_train.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned training data not found at {data_path}")

    df = pd.read_csv(data_path)
    target = params.get("target", "yield_qtl_per_acre")

    # Ensure all required features are present
    X = df[ALL_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
    )

    data_sha = compute_data_sha256()

    base_model_params = params.get("model", {})
    configs = [
        {"name": "baseline", **base_model_params},
        {"name": "shallow", **{**base_model_params, "max_depth": 6}},
        {"name": "deeper", **{**base_model_params, "max_depth": 18, "n_estimators": 300}},
    ]

    best_mae = float("inf")
    best_model = None
    best_run_id = None

    for config in configs:
        run_name = config["name"]
        model_params = {k: v for k, v in config.items() if k != "name"}

        pipeline = build_pipeline(model_params)

        with mlflow.start_run(run_name=run_name) as run:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            mae = float(mean_absolute_error(y_test, predictions))
            rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
            r2 = float(r2_score(y_test, predictions))

            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_params(model_params)
            mlflow.log_metrics({
                "MAE": mae,
                "mae": mae,
                "RMSE": rmse,
                "R2": r2,
            })

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            print(f"Model: {run_name} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

            if mae < best_mae:
                best_mae = mae
                best_model = pipeline
                best_run_id = run.info.run_id

    print(f"\nChampion Selected: Run {best_run_id} with MAE {best_mae:.4f}")

    # Register champion model in MLflow Model Registry
    reg_model_name = "wheat-yield-champion"
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=reg_model_name)

    client = MlflowClient()
    client.set_registered_model_alias(reg_model_name, "champion", str(model_version.version))

    # Save to models/model.joblib for production API serving
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.joblib")
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"Model saved to models/model.joblib and models/best_model.pkl")

    # Update findings.yaml with tracking metadata
    update_findings_tracking(best_run_id, reg_model_name, int(model_version.version))

    return best_model, best_run_id, best_mae


def update_findings_tracking(run_id: str, reg_model_name: str, version: int):
    """Update findings.yaml tracking section."""
    findings_path = "findings.yaml"
    if os.path.exists(findings_path):
        with open(findings_path, "r", encoding="utf-8") as f:
            findings = yaml.safe_load(f) or {}

        if "tracking" not in findings:
            findings["tracking"] = {}

        findings["tracking"]["best_run_id"] = run_id
        findings["tracking"]["registered_model_name"] = reg_model_name
        findings["tracking"]["champion_version"] = version

        with open(findings_path, "w", encoding="utf-8") as f:
            yaml.dump(findings, f, sort_keys=False)
        print("Updated findings.yaml with champion tracking info")


if __name__ == "__main__":
    train_models()
