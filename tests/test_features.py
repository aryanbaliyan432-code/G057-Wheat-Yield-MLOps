"""Unit tests for feature engineering, preprocessors, and transformers."""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.wheat_yield.model_training import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ALL_FEATURES,
    build_pipeline,
)


@pytest.fixture
def sample_feature_df():
    """Create sample feature dataframe for transformation tests."""
    return pd.DataFrame({
        "rainfall_mm": [500.0, 650.0],
        "fertilizer_kg": [100.0, 120.0],
        "sowing_delay_days": [10.0, 15.0],
        "irrigation_count": [4.0, 5.0],
        "soil_ph": [6.8, 7.2],
        "farm_acres": [5.0, 8.0],
        "seed_variety": ["HD-2967", "DBW-187"],
        "tehsil": ["Tehsil North", "Tehsil South"],
    })


def test_feature_list_definitions():
    """Ensure categorical and numerical feature definitions are distinct and non-empty."""
    assert len(CATEGORICAL_FEATURES) == 2
    assert len(NUMERICAL_FEATURES) == 6
    assert set(CATEGORICAL_FEATURES).isdisjoint(set(NUMERICAL_FEATURES))
    assert len(ALL_FEATURES) == 8


def test_build_pipeline_structure():
    """Verify that build_pipeline creates a Pipeline with ColumnTransformer and regressor."""
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    pipeline = build_pipeline(params)
    assert hasattr(pipeline, "named_steps")
    assert "preprocessor" in pipeline.named_steps
    assert "regressor" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)


def test_pipeline_transform_unseen_categories(sample_feature_df):
    """Verify preprocessor handles unknown categorical levels without error."""
    params = {"n_estimators": 5, "max_depth": 2, "random_state": 42}
    pipeline = build_pipeline(params)
    y = pd.Series([20.0, 22.0])

    pipeline.fit(sample_feature_df, y)

    # Input with unseen category
    unseen_df = pd.DataFrame({
        "rainfall_mm": [600.0],
        "fertilizer_kg": [110.0],
        "sowing_delay_days": [12.0],
        "irrigation_count": [3.0],
        "soil_ph": [7.0],
        "farm_acres": [4.5],
        "seed_variety": ["UNKNOWN_VARIETY"],
        "tehsil": ["UNKNOWN_TEHSIL"],
    })

    # Should predict without error thanks to handle_unknown="ignore"
    preds = pipeline.predict(unseen_df)
    assert len(preds) == 1
    assert isinstance(preds[0], float)
