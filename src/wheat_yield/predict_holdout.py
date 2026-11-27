"""Holdout dataset inference module for Wheat Yield Prediction.

Loads the registered @champion model and scores unseen observations in
data/raw/holdout_features.csv, outputting predictions/holdout_predictions.csv
with exact columns (record_id, prediction).
"""

import os
import sys
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from src.wheat_yield.logger import get_logger
from src.wheat_yield.model_training import ALL_FEATURES

logger = get_logger("wheat_yield.holdout")


def generate_holdout_predictions(
    holdout_path: str = "data/raw/holdout_features.csv",
    output_path: str = "predictions/holdout_predictions.csv",
    model_path: str = "models/model.joblib",
) -> pd.DataFrame:
    """Score the unseen holdout dataset and save predictions to CSV."""
    logger.info(
        f"Generating holdout predictions from {holdout_path}...",
        extra={"event_type": "holdout_prediction_started"}
    )

    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"Holdout dataset not found at {holdout_path}")

    df_holdout = pd.read_csv(holdout_path)
    if "record_id" not in df_holdout.columns:
        raise ValueError("Holdout dataset must contain 'record_id' column")

    # Load champion model
    model = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # Fallback to MLflow champion alias
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            client = MlflowClient()
            champ = client.get_model_version_by_alias("wheat-yield-champion", "champion")
            model = mlflow.sklearn.load_model(champ.source)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise

    # Extract required feature columns
    X_holdout = df_holdout[ALL_FEATURES]
    predictions = model.predict(X_holdout)

    # Format exactly as: record_id,prediction
    results_df = pd.DataFrame({
        "record_id": df_holdout["record_id"],
        "prediction": [round(float(p), 2) for p in predictions]
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info(
        f"Successfully generated {len(results_df)} holdout predictions saved to {output_path}",
        extra={
            "event_type": "holdout_prediction_completed",
            "details": {
                "num_rows": len(results_df),
                "min_pred": float(results_df["prediction"].min()),
                "max_pred": float(results_df["prediction"].max()),
                "mean_pred": float(results_df["prediction"].mean()),
            }
        }
    )

    return results_df


if __name__ == "__main__":
    generate_holdout_predictions()
