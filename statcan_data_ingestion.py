import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------
# 1. Project folders
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw" / "statcan"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Statistics Canada API configuration
# ---------------------------------------------------------

BASE_URL = (
    "https://www150.statcan.gc.ca/"
    "t1/wds/rest/getDataFromVectorByReferencePeriodRange"
)

VECTOR_ID = "41690973"
START_DATE = "2009-01-01"
END_DATE = "2026-06-30"


# ---------------------------------------------------------
# 3. Download CPI data
# ---------------------------------------------------------

params = {
    "vectorIds": VECTOR_ID,
    "startRefPeriod": START_DATE,
    "endReferencePeriod": END_DATE,
}

print("Downloading Statistics Canada CPI data...")

response = requests.get(
    BASE_URL,
    params=params,
    timeout=30,
)

print("Status Code:", response.status_code)

response.raise_for_status()

data = response.json()


# ---------------------------------------------------------
# 4. Validate the response
# ---------------------------------------------------------

if not data:
    raise ValueError("Statistics Canada returned an empty response.")

result = data[0]

if result.get("status") != "SUCCESS":
    raise ValueError(
        f"Statistics Canada request failed: {result}"
    )

observations = result["object"]["vectorDataPoint"]

print("Number of observations:", len(observations))


# ---------------------------------------------------------
# 5. Save raw JSON
# ---------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

raw_file = RAW_DIR / f"cpi_{VECTOR_ID}_{timestamp}.json"

with open(raw_file, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)

print("Raw JSON saved to:")
print(raw_file)


# ---------------------------------------------------------
# 6. Convert the CPI records into a DataFrame
# ---------------------------------------------------------

cpi_df = pd.DataFrame(observations)

cpi_df = cpi_df[["refPer", "value"]].copy()

cpi_df.rename(
    columns={
        "refPer": "date",
        "value": "cpi_all_items",
    },
    inplace=True,
)


# ---------------------------------------------------------
# 7. Clean the data
# ---------------------------------------------------------

cpi_df["date"] = pd.to_datetime(
    cpi_df["date"],
    errors="coerce",
)

cpi_df["cpi_all_items"] = pd.to_numeric(
    cpi_df["cpi_all_items"],
    errors="coerce",
)

cpi_df = (
    cpi_df
    .drop_duplicates(subset="date")
    .sort_values("date")
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# 8. Save processed CSV
# ---------------------------------------------------------

output_file = PROCESSED_DIR / "statcan_cpi.csv"

cpi_df.to_csv(
    output_file,
    index=False,
)

print("\nDataset Shape:")
print(cpi_df.shape)

print("\nFirst Five Rows:")
print(cpi_df.head())

print("\nLast Five Rows:")
print(cpi_df.tail())

print("\nMissing Values:")
print(cpi_df.isna().sum())

print("\nProcessed CSV saved to:")
print(output_file)