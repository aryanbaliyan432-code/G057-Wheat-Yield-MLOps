"""Unit tests for data preprocessing and cleaning logic."""

import numpy as np
import pandas as pd
import pytest

from src.wheat_yield.preprocessing import clean_data


@pytest.fixture
def sample_raw_dataframe():
    """Build small self-contained in-memory raw dataframe with intentional anomalies."""
    data = {
        "record_id": ["R0001", "R0002", "R0002", "R0003", "R0004", "R0005"],
        "date": [
            "2026-01-10",
            "2026-02-15",
            "2026-02-15",  # Duplicate row
            "2026-03-20",
            "2026-04-10",
            "2026-05-05",
        ],
        "rainfall_mm": [50.0, 60.0, 60.0, 600.0, 700.0, 80.0],
        "fertilizer_kg": [100.0, 110.0, 110.0, 90.0, 95.0, 105.0],
        "sowing_delay_days": [5, 999, 999, 10, 15, 20],  # Sentinel value 999
        "irrigation_count": [3, 4, 4, 5, 2, 4],  # Stuck sensor (4 on 2026-02-15 in Tehsil North)
        "soil_ph": [6.5, 7.0, 7.0, 6.8, 7.2, 6.9],
        "farm_acres": [5.0, 4.0, 4.0, 8.0, 6.0, 3.5],
        "seed_variety": ["HD-2967", "PBW-343", "PBW-343", "DBW-187", "HD-2967", "PBW-343"],
        "tehsil": [
            "Tehsil North",
            "Tehsil North",
            "Tehsil North",
            "Tehsil South",
            "Tehsil West",
            "Tehsil North",
        ],
        "yield_from_mandi_receipt": [21.5, 22.0, 22.0, 24.5, 23.0, 20.0],
        "yield_qtl_per_acre": [21.5, 220.0, 220.0, 24.5, 23.0, 20.0],  # Typo: 220.0 instead of 22.0
    }
    return pd.DataFrame(data)

# Test 1: the duplicate row (R0002 appears twice) is dropped, so 6 rows become 5


def test_clean_data_removes_duplicates(sample_raw_dataframe):
    """Verify that duplicate rows are properly identified and dropped."""
    cleaned = clean_data(sample_raw_dataframe)
    assert len(cleaned) == 5  # One duplicate removed

# Test 2: only Tehsil North rainfall is multiplied by 10 (cm -> mm);
# the other tehsils must stay unchanged
def test_clean_data_corrects_rainfall_unit_mixup(sample_raw_dataframe):
    """Verify rainfall in Tehsil North is multiplied by 10 (unit mixup corrected)."""
    cleaned = clean_data(sample_raw_dataframe)
    # Row 0 was 50.0 in Tehsil North -> should become 500.0
    assert cleaned.iloc[0]["rainfall_mm"] == 500.0
    # Row 3 (Tehsil South) was 600.0 -> should remain 600.0
    assert cleaned[cleaned["tehsil"] == "Tehsil South"]["rainfall_mm"].values[0] == 600.0

# Test 3: the 999 placeholder is replaced and then filled, so no 999 and no
# missing values remain in sowing_delay_days 

def test_clean_data_handles_sentinel_values(sample_raw_dataframe):
    """Verify sentinel 999 in sowing_delay_days is replaced and imputed with median."""
    cleaned = clean_data(sample_raw_dataframe)
    assert 999 not in cleaned["sowing_delay_days"].values
    assert not cleaned["sowing_delay_days"].isnull().any()

# Test 4: the typo target 220.0 is divided by 10, so no target is above 100


def test_clean_data_corrects_target_typos(sample_raw_dataframe):
    """Verify targets > 100 are corrected by dividing by 10."""
    cleaned = clean_data(sample_raw_dataframe)
    assert (cleaned["yield_qtl_per_acre"] > 100).sum() == 0
    assert cleaned.loc[cleaned["tehsil"] == "Tehsil North", "yield_qtl_per_acre"].max() < 50.0


# Test 5: the leaking column and the identifier columns are removed, because
# the model must not use them as features

def test_clean_data_drops_leakage_and_id_columns(sample_raw_dataframe):
    """Verify that leaking column, date, and record_id are removed from output."""
    cleaned = clean_data(sample_raw_dataframe)
    assert "record_id" not in cleaned.columns
    assert "yield_from_mandi_receipt" not in cleaned.columns
    assert "date" not in cleaned.columns
