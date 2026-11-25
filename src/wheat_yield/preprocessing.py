
import pandas as pd
import numpy as np
import os


def load_data():
    file_path = "data/raw/train.csv"
    df = pd.read_csv(file_path)
    print("Original shape:", df.shape)
    return df


def clean_data(df):
    df = df.copy()

    # 1. Remove duplicate rows
    df = df.drop_duplicates()

    # 2. Convert date
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # 3. Correct rainfall unit mixup in Tehsil North
    df.loc[df["tehsil"] == "Tehsil North",
           "rainfall_mm"] *= 10

    # 4. Replace sentinel value
    df.loc[df["sowing_delay_days"] == 999,
           "sowing_delay_days"] = np.nan

    # 5. Correct stuck irrigation sensor
    sensor_mask = (
        (df["tehsil"] == "Tehsil North") &
        (df["date"] >= "2026-02-14") &
        (df["date"] <= "2026-02-23") &
        (df["irrigation_count"] == 4)
    )
    df.loc[sensor_mask, "irrigation_count"] = np.nan

    # 6. Correct target values typed 10 times too large
    if "yield_qtl_per_acre" in df.columns:
        mask = df["yield_qtl_per_acre"] > 100
        df.loc[mask, "yield_qtl_per_acre"] /= 10

    # 7. Impute missing numerical values
    for col in ["sowing_delay_days", "irrigation_count"]:
        df[col] = df.groupby("tehsil")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # 8. Remove leakage and identifier columns
    df = df.drop(
        columns=["record_id", "yield_from_mandi_receipt", "date"],
        errors="ignore"
    )

    print("Cleaning completed!")
    print("Final shape:", df.shape)

    return df


def save_data(df):
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/cleaned_train.csv"
    df.to_csv(output_path, index=False)
    print("Saved:", output_path)


if __name__ == "__main__":
    data = load_data()
    cleaned_data = clean_data(data)
    save_data(cleaned_data)
    print("Preprocessing pipeline completed!")
