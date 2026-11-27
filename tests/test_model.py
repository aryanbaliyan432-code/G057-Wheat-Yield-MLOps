"""Unit tests for model persistence, loading, and inference."""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from app.main import load_model


def test_model_artifact_exists():
    """Verify that champion model artifact exists on disk."""
    assert os.path.exists("models/model.joblib") or os.path.exists("models/best_model.pkl")


def test_load_model_function():
    """Verify that load_model successfully loads model pipeline."""
    model = load_model("models/model.joblib")
    assert model is not None
    assert hasattr(model, "predict")


def test_model_prediction_output():
    """Verify loaded model generates sensible non-negative predictions."""
    model = load_model("models/model.joblib")
    sample_input = pd.DataFrame([{
        "rainfall_mm": 600.0,
        "fertilizer_kg": 100.0,
        "sowing_delay_days": 10.0,
        "irrigation_count": 4.0,
        "soil_ph": 6.8,
        "farm_acres": 5.0,
        "seed_variety": "HD-2967",
        "tehsil": "Tehsil South",
    }])

    pred = model.predict(sample_input)
    assert len(pred) == 1
    assert isinstance(float(pred[0]), float)
    assert 5.0 <= float(pred[0]) <= 50.0
