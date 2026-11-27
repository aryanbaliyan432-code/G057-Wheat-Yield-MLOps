"""Data generator to synthesize raw datasets matching project audit findings.

This generates:
- data/raw/train.csv (1000 rows with known data issues)
- data/raw/later.csv (300 rows for monitoring & drift detection)
- data/raw/holdout_features.csv (250 rows for unseen holdout evaluation)
"""

import os
import numpy as np
import pandas as pd


def generate_datasets(random_state: int = 42):
    np.random.seed(random_state)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    n_train = 1000
    n_later = 300
    n_holdout = 250

    tehsils = ["Tehsil North", "Tehsil South", "Tehsil West"]
    seed_varieties = ["HD-2967", "PBW-343", "DBW-187"]

    # ------------------
    # 1. Train Dataset
    # ------------------
    # 967 unique rows + 33 duplicates = 1000
    n_unique = n_train - 33
    dates = pd.date_range("2026-01-01", "2026-06-30", periods=n_unique)
    
    assigned_tehsil = np.random.choice(tehsils, size=n_unique, p=[0.34, 0.33, 0.33])
    assigned_seeds = np.random.choice(seed_varieties, size=n_unique)

    # Base features
    fertilizer = np.random.uniform(20.0, 150.0, size=n_unique).round(1)
    sowing_delay = np.random.randint(0, 40, size=n_unique)
    irrigation = np.random.randint(0, 8, size=n_unique)
    soil_ph = np.random.uniform(5.5, 8.5, size=n_unique).round(2)
    farm_acres = np.random.uniform(1.0, 15.0, size=n_unique).round(2)

    # Rainfall with unit mixup in Tehsil North (cm instead of mm: / 10)
    rainfall = np.zeros(n_unique)
    for i, t in enumerate(assigned_tehsil):
        if t == "Tehsil North":
            rainfall[i] = np.random.uniform(22.1, 119.7)
        elif t == "Tehsil South":
            rainfall[i] = np.random.uniform(201.0, 1197.0)
        else:
            rainfall[i] = np.random.uniform(203.0, 1200.0)
    rainfall = rainfall.round(1)

    # True yield per acre (quintals)
    # Synthetic formula representing realistic agronomic response
    # True rain in mm:
    rain_mm = np.where(assigned_tehsil == "Tehsil North", rainfall * 10, rainfall)
    base_yield = (
        12.0
        + 0.008 * rain_mm
        + 0.04 * fertilizer
        - 0.12 * sowing_delay
        + 0.5 * irrigation
        + 0.3 * (soil_ph - 6.5)
        + np.random.normal(0, 1.0, size=n_unique)
    )
    base_yield = np.clip(base_yield, 8.0, 35.0).round(1)

    # Mandi receipt: correlation ~0.999 with true yield
    mandi_receipt = (base_yield + np.random.normal(0, 0.15, size=n_unique)).round(2)

    # Create initial DataFrame
    record_ids = [f"R{i+1:05d}" for i in range(n_unique)]
    df_unique = pd.DataFrame({
        "record_id": record_ids,
        "date": dates.strftime("%Y-%m-%d"),
        "rainfall_mm": rainfall,
        "fertilizer_kg": fertilizer,
        "sowing_delay_days": sowing_delay,
        "irrigation_count": irrigation,
        "soil_ph": soil_ph,
        "farm_acres": farm_acres,
        "seed_variety": assigned_seeds,
        "tehsil": assigned_tehsil,
        "yield_from_mandi_receipt": mandi_receipt,
        "yield_qtl_per_acre": base_yield
    })

    # Stuck sensor: Tehsil North irrigation_count = 4 between 2026-02-14 and 2026-02-23 (5 obs)
    stuck_dates = pd.date_range("2026-02-14", "2026-02-23", periods=5).strftime("%Y-%m-%d")
    north_indices = df_unique[df_unique["tehsil"] == "Tehsil North"].index
    for idx, sdate in zip(north_indices[:5], stuck_dates):
        df_unique.loc[idx, "date"] = sdate
        df_unique.loc[idx, "irrigation_count"] = 4

    # Sentinel value 999 in sowing_delay_days: exactly 33 rows
    sentinel_indices = np.random.choice(df_unique.index, size=33, replace=False)
    df_unique.loc[sentinel_indices, "sowing_delay_days"] = 999

    # Add 33 duplicate rows to reach 1000
    dup_indices = np.random.choice(df_unique.index, size=33, replace=False)
    df_duplicates = df_unique.loc[dup_indices].copy()
    df_train = pd.concat([df_unique, df_duplicates], ignore_index=True)
    df_train = df_train.sort_values("date").reset_index(drop=True)

    # Target typos: exactly 10 targets multiplied by 10 as in reports/data_audit.txt
    typo_records = {
        "R00583": 242.7,
        "R00121": 241.1,
        "R00511": 226.9,
        "R00896": 219.0,
        "R00942": 216.8,
        "R00279": 210.0,
        "R00077": 173.3,
        "R00013": 160.6,
        "R00052": 126.5,
        "R00872": 111.1
    }
    for rec_id, val in typo_records.items():
        mask = df_train["record_id"] == rec_id
        if mask.any():
            df_train.loc[mask, "yield_qtl_per_acre"] = val

    df_train.to_csv("data/raw/train.csv", index=False)
    print(f"Created data/raw/train.csv: {df_train.shape}")

    # ------------------
    # 2. Later Dataset (July & August 2026, 300 rows, Clean)
    # ------------------
    dates_later = pd.date_range("2026-07-01", "2026-08-31", periods=n_later)
    tehsil_later = np.random.choice(tehsils, size=n_later)
    seed_later = np.random.choice(seed_varieties, size=n_later)
    
    # Seasonal shift in monsoon: rainfall increases (distribution drift)
    rain_later = np.random.uniform(500.0, 1400.0, size=n_later).round(1)
    fert_later = np.random.uniform(30.0, 160.0, size=n_later).round(1)
    delay_later = np.random.randint(0, 35, size=n_later)
    irrig_later = np.random.randint(1, 6, size=n_later)
    ph_later = np.random.uniform(5.8, 8.2, size=n_later).round(2)
    acres_later = np.random.uniform(1.5, 14.0, size=n_later).round(2)

    yield_later = (
        13.0
        + 0.007 * rain_later
        + 0.04 * fert_later
        - 0.11 * delay_later
        + 0.55 * irrig_later
        + 0.3 * (ph_later - 6.5)
        + np.random.normal(0, 1.0, size=n_later)
    ).round(1)
    yield_later = np.clip(yield_later, 10.0, 36.0)

    df_later = pd.DataFrame({
        "record_id": [f"R{1001 + i:05d}" for i in range(n_later)],
        "date": dates_later.strftime("%Y-%m-%d"),
        "rainfall_mm": rain_later,
        "fertilizer_kg": fert_later,
        "sowing_delay_days": delay_later,
        "irrigation_count": irrig_later,
        "soil_ph": ph_later,
        "farm_acres": acres_later,
        "seed_variety": seed_later,
        "tehsil": tehsil_later,
        "yield_qtl_per_acre": yield_later
    })
    df_later.to_csv("data/raw/later.csv", index=False)
    print(f"Created data/raw/later.csv: {df_later.shape}")

    # ------------------
    # 3. Holdout Features Dataset (September 2026, 250 rows, No Target)
    # ------------------
    dates_holdout = pd.date_range("2026-09-01", "2026-09-30", periods=n_holdout)
    tehsil_holdout = np.random.choice(tehsils, size=n_holdout)
    seed_holdout = np.random.choice(seed_varieties, size=n_holdout)

    rain_holdout = np.random.uniform(300.0, 1100.0, size=n_holdout).round(1)
    fert_holdout = np.random.uniform(25.0, 150.0, size=n_holdout).round(1)
    delay_holdout = np.random.randint(0, 35, size=n_holdout)
    irrig_holdout = np.random.randint(1, 7, size=n_holdout)
    ph_holdout = np.random.uniform(5.6, 8.4, size=n_holdout).round(2)
    acres_holdout = np.random.uniform(1.0, 15.0, size=n_holdout).round(2)

    df_holdout = pd.DataFrame({
        "record_id": [f"R{1301 + i:05d}" for i in range(n_holdout)],
        "date": dates_holdout.strftime("%Y-%m-%d"),
        "rainfall_mm": rain_holdout,
        "fertilizer_kg": fert_holdout,
        "sowing_delay_days": delay_holdout,
        "irrigation_count": irrig_holdout,
        "soil_ph": ph_holdout,
        "farm_acres": acres_holdout,
        "seed_variety": seed_holdout,
        "tehsil": tehsil_holdout
    })
    df_holdout.to_csv("data/raw/holdout_features.csv", index=False)
    print(f"Created data/raw/holdout_features.csv: {df_holdout.shape}")


if __name__ == "__main__":
    generate_datasets()
