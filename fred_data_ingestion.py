import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. Load API Key
# ---------------------------------------------------------

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("FRED_API_KEY")

if not API_KEY:
    raise ValueError("FRED_API_KEY not found.")

# ---------------------------------------------------------
# 2. Project folders
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw" / "fred"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------

START_DATE = "2009-01-02"
END_DATE = "2026-06-30"

SERIES = {
    "DGS10": "us_treasury_10y",
    "DFF": "fed_funds_rate"
}

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

dataframes = []

# ---------------------------------------------------------
# 4. Download each series
# ---------------------------------------------------------

for series_id, column_name in SERIES.items():

    print(f"\nDownloading {series_id}...")

    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": END_DATE
    }

    response = requests.get(BASE_URL, params=params, timeout=30)

    print("Status:", response.status_code)

    response.raise_for_status()

    data = response.json()

    # Save raw JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_file = RAW_DIR / f"{series_id}_{timestamp}.json"

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    df = pd.DataFrame(data["observations"])

    df = df[["date", "value"]]

    df.rename(
        columns={
            "value": column_name
        },
        inplace=True
    )

    df[column_name] = (
        df[column_name]
        .replace(".", pd.NA)
        .astype(float)
    )

    dataframes.append(df)

# ---------------------------------------------------------
# 5. Merge the two datasets
# ---------------------------------------------------------

fred_df = dataframes[0]

for df in dataframes[1:]:
    fred_df = fred_df.merge(
        df,
        on="date",
        how="outer"
    )

fred_df["date"] = pd.to_datetime(fred_df["date"])

fred_df = fred_df.sort_values("date")

fred_df = fred_df.ffill()

# ---------------------------------------------------------
# 6. Save CSV
# ---------------------------------------------------------

output_file = PROCESSED_DIR / "fred_rates.csv"

fred_df.to_csv(
    output_file,
    index=False
)

print("\nDataset Shape:")
print(fred_df.shape)

print("\nFirst Five Rows:")
print(fred_df.head())

print("\nSaved to:")
print(output_file)