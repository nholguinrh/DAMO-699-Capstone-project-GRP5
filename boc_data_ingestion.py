import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------
# 1. Project folders
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw" / "bank_of_canada"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. API configuration
# ---------------------------------------------------------

START_DATE = "2009-01-02"
END_DATE = "2026-06-30"

SERIES_MAP = {
    "CBC20210": "overnight_rate",
    "BD.CDN.2YR.DQ.YLD": "yield_2y",
    "BD.CDN.3YR.DQ.YLD": "yield_3y",
    "BD.CDN.5YR.DQ.YLD": "yield_5y",
    "BD.CDN.7YR.DQ.YLD": "yield_7y",
    "BD.CDN.10YR.DQ.YLD": "yield_10y",
    "BD.CDN.LONG.DQ.YLD": "yield_long",
}

series_names = ",".join(SERIES_MAP.keys())

url = (
    "https://www.bankofcanada.ca/valet/observations/"
    f"{series_names}/json"
)

params = {
    "start_date": START_DATE,
    "end_date": END_DATE,
}


# ---------------------------------------------------------
# 3. Download data
# ---------------------------------------------------------

print("Connecting to the Bank of Canada Valet API...")

response = requests.get(
    url,
    params=params,
    timeout=60,
)

print("Status Code:", response.status_code)

response.raise_for_status()

data = response.json()

if "observations" not in data:
    raise ValueError(
        "The API response does not contain an observations section."
    )

print("Number of API observations:", len(data["observations"]))


# ---------------------------------------------------------
# 4. Save raw JSON
# ---------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

raw_file = RAW_DIR / f"boc_yields_{timestamp}.json"

with raw_file.open(
    mode="w",
    encoding="utf-8",
) as file:
    json.dump(data, file, indent=4)

print("Raw JSON saved to:")
print(raw_file)


# ---------------------------------------------------------
# 5. Convert observations to rows
# ---------------------------------------------------------

rows = []

for observation in data["observations"]:

    row = {
        "date": observation.get("d")
    }

    for series_code, column_name in SERIES_MAP.items():

        series_value = observation.get(series_code, {})

        row[column_name] = series_value.get("v")

    rows.append(row)


# ---------------------------------------------------------
# 6. Create and clean DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(rows)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

for column in SERIES_MAP.values():

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df = df.dropna(subset=["date"])

df = df.drop_duplicates(
    subset=["date"]
)

df = df.sort_values("date")

df = df.reset_index(drop=True)

# Forward fill missing observations
df = df.ffill()

# ---------------------------------------------------------
# 7. Calculate the 10-year minus 2-year spread
# ---------------------------------------------------------

df["yield_spread_10y_2y"] = (
    df["yield_10y"] -
    df["yield_2y"]
)


# ---------------------------------------------------------
# 8. Validate the data
# ---------------------------------------------------------

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print("Start:", df["date"].min())
print("End:", df["date"].max())

print("\nMissing values:")
print(df.isna().sum())


# ---------------------------------------------------------
# 9. Save processed CSV
# ---------------------------------------------------------

processed_file = (
    PROCESSED_DIR /
    "bank_of_canada_yields.csv"
)

df.to_csv(
    processed_file,
    index=False,
)

print("\nProcessed CSV saved to:")
print(processed_file)

print("\nBank of Canada yield pipeline completed successfully.")