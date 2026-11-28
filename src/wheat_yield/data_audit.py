
import pandas as pd

# Load original dataset
df = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

# Print a banner so the report is easy to find in the terminal output
print("=" * 55)
print("WHEAT YIELD DATA QUALITY INVESTIGATION")
print("=" * 55)

# 1. LEAKING COLUMN
print("\n1. LEAKING COLUMN")

missing_from_holdout = set(df.columns) - set(holdout.columns)
print("Columns missing from holdout:", missing_from_holdout)

corr = df.loc[
    df["yield_qtl_per_acre"] < 100,
    ["yield_from_mandi_receipt", "yield_qtl_per_acre"]
].corr().iloc[0, 1]

print("Mandi receipt correlation (excluding extreme target values):", corr)

# 2. UNIT MIXUP
print("\n2. UNIT MIXUP")

# Compare rainfall per tehsil: North is about 10x smaller than the others,
# so it was recorded in cm instead of mm

print(df.groupby("tehsil")["rainfall_mm"].agg(
    ["min", "mean", "median", "max"]
))

# 3. SENTINEL VALUE
print("\n3. SENTINEL VALUE")

print("999 in sowing_delay_days:",
      (df["sowing_delay_days"] == 999).sum())

# 4. DUPLICATE ROWS
print("\n4. DUPLICATE ROWS")

print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicate record IDs:", df["record_id"].duplicated().sum())

# 5. STUCK SENSOR
print("\n5. STUCK SENSOR")

# For each tehsil, collect the irrigation values seen on each date, then look for
# a run of 4 or more consecutive dates with the same value (a stuck sensor)

sensor_data = df.copy()
sensor_data["date"] = pd.to_datetime(sensor_data["date"])

for group, data in sensor_data.groupby("tehsil"):
    daily = data.groupby("date")["irrigation_count"].agg(
        lambda x: tuple(sorted(x.unique()))
    )

    values = list(daily.items())
    start = 0

    for i in range(1, len(values) + 1):
        if i == len(values) or values[i][1] != values[start][1]:
            if i - start >= 4:
                print(
                    group,
                    "Value:", values[start][1],
                    "Start:", values[start][0].date(),
                    "End:", values[i - 1][0].date(),
                    "Repeated observations:", i - start
                )
            start = i

# 6. TARGET TYPOS
print("\n6. TARGET TYPOS")

print("Targets above 100:",
      (df["yield_qtl_per_acre"] > 100).sum())

print(df.loc[
    df["yield_qtl_per_acre"] > 100,
    ["record_id", "yield_qtl_per_acre"]
].to_string(index=False))

# 7. SPELLING VARIANTS
print("\n7. SPELLING VARIANTS")

for col in ["seed_variety", "tehsil"]:
    original = set(df[col].dropna().unique())
    normalized = set(
        df[col].dropna().astype(str).str.strip().str.lower().unique()
    )

    print("\nColumn:", col)
    print("Original categories:", sorted(original))
    print("Normalized categories:", sorted(normalized))
    print("Spelling variants present:", len(original) != len(normalized))

print("\n" + "=" * 55)
print("DATA INVESTIGATION COMPLETED")
print("=" * 55)
