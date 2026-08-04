import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from http_utils import get_with_retry
from project_paths import PROCESSED_DIR, RAW_DIR


# ---------------------------------------------------------
# 1. Folder configuration
# ---------------------------------------------------------

BOC_RAW_DIR = RAW_DIR / "bank_of_canada"

BOC_RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. API configuration
# ---------------------------------------------------------

START_DATE = "2009-01-02"
END_DATE = "2026-06-30"

BASE_URL = (
    "https://www.bankofcanada.ca/"
    "valet/observations/{series_names}/json"
)

SERIES_IDS = {
    "CBC20210": "overnight_rate",
    "BD.CDN.2YR.DQ.YLD": "yield_2y",
    "BD.CDN.3YR.DQ.YLD": "yield_3y",
    "BD.CDN.5YR.DQ.YLD": "yield_5y",
    "BD.CDN.7YR.DQ.YLD": "yield_7y",
    "BD.CDN.10YR.DQ.YLD": "yield_10y",
    "BD.CDN.LONG.DQ.YLD": "yield_long",
    "IEXE0101": "usdcad_legacy",
    "FXUSDCAD": "usdcad_current",
}

METHODOLOGY_CHANGE_DATE = pd.Timestamp("2017-05-01")

PROCESSED_FILE = (
    PROCESSED_DIR
    / "bank_of_canada_data.csv"
)


# ---------------------------------------------------------
# 3. Download raw data
# ---------------------------------------------------------

def download_raw_data() -> Path:
    """
    Download Bank of Canada data sequentially using the Valet API.

    Returns
    -------
    Path
        Path to the cached raw JSON file.
    """

    series_names = ",".join(SERIES_IDS.keys())

    url = BASE_URL.format(
        series_names=series_names
    )

    params = {
        "start_date": START_DATE,
        "end_date": END_DATE,
    }

    print("Connecting to the Bank of Canada Valet API...")

    response = get_with_retry(
        url,
        params=params,
        timeout=60,
        max_retries=5,
    )

    print("Status Code:", response.status_code)

    data = response.json()

    if "observations" not in data:
        raise ValueError(
            "The API response does not contain "
            "an 'observations' section."
        )

    observation_count = len(data["observations"])

    if observation_count == 0:
        raise ValueError(
            "The Bank of Canada API returned no observations."
        )

    print(
        "Number of API observations:",
        observation_count,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    raw_file = (
        BOC_RAW_DIR
        / f"bank_of_canada_{timestamp}.json"
    )

    with raw_file.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )

    print("\nRaw JSON saved to:")
    print(raw_file)

    return raw_file


# ---------------------------------------------------------
# 4. Find cached raw data
# ---------------------------------------------------------

def find_latest_cached_json() -> Path:
    """
    Find the most recently modified Bank of Canada raw JSON file.
    """

    cached_files = sorted(
        BOC_RAW_DIR.glob("*.json"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )

    if not cached_files:
        raise FileNotFoundError(
            "No cached Bank of Canada JSON files were found in "
            f"{BOC_RAW_DIR}"
        )

    latest_file = cached_files[0]

    print("Using cached Bank of Canada JSON:")
    print(latest_file)

    return latest_file


# ---------------------------------------------------------
# 5. Load raw JSON
# ---------------------------------------------------------

def load_raw_json(raw_file: Path) -> dict:
    """
    Load and validate a cached Bank of Canada JSON response.
    """

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw JSON file does not exist: {raw_file}"
        )

    with raw_file.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if "observations" not in data:
        raise ValueError(
            "Cached JSON does not contain "
            "an 'observations' section."
        )

    if not data["observations"]:
        raise ValueError(
            "Cached JSON contains no observations."
        )

    return data


# ---------------------------------------------------------
# 6. Convert observations into a DataFrame
# ---------------------------------------------------------

def create_dataframe(data: dict) -> pd.DataFrame:
    """
    Convert the Bank of Canada API observations into a clean DataFrame.
    """

    rows = []

    for observation in data["observations"]:

        row = {
            "date": observation.get("d")
        }

        for series_code, column_name in SERIES_IDS.items():

            series_object = observation.get(
                series_code,
                {},
            )

            row[column_name] = series_object.get(
                "v"
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "No rows were created from the API observations."
        )

    return df


# ---------------------------------------------------------
# 7. Clean and transform data
# ---------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean dates and numeric fields, combine USD/CAD methodologies,
    and calculate the 10-year minus 2-year yield spread.

    Unrestricted forward filling is intentionally avoided.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    for column_name in SERIES_IDS.values():

        if column_name not in df.columns:
            df[column_name] = pd.NA

        df[column_name] = pd.to_numeric(
            df[column_name],
            errors="coerce",
        )

    df = (
        df
        .dropna(subset=["date"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Combine legacy and current USD/CAD methodologies
    #
    # IEXE0101 (legacy) and FXUSDCAD (current) are two different
    # Bank of Canada exchange-rate series, split by the date BoC
    # changed its methodology: 2017-05-01 (see METHODOLOGY_CHANGE_DATE
    # above). Both original series are preserved as usdcad_legacy /
    # usdcad_current in the output for traceability -- usdcad is the
    # stitched column downstream code should actually use.
    # -----------------------------------------------------

    legacy_mask = (
        df["date"] < METHODOLOGY_CHANGE_DATE
    )

    current_mask = (
        df["date"] >= METHODOLOGY_CHANGE_DATE
    )

    df["usdcad"] = pd.NA

    df.loc[
        legacy_mask,
        "usdcad",
    ] = df.loc[
        legacy_mask,
        "usdcad_legacy",
    ]

    df.loc[
        current_mask,
        "usdcad",
    ] = df.loc[
        current_mask,
        "usdcad_current",
    ]

    df["usdcad"] = pd.to_numeric(
        df["usdcad"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Calculate the 10-year minus 2-year yield spread
    # -----------------------------------------------------

    df["yield_spread_10y_2y"] = (
        df["yield_10y"]
        - df["yield_2y"]
    )

    # No unrestricted ffill() is used here.
    # Missing observations are preserved for later calendar
    # alignment and modeling decisions.

    return df


# ---------------------------------------------------------
# 8. Validate processed data
# ---------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
) -> None:
    """
    Validate the structure and content of the processed dataset.
    """

    required_columns = {
        "date",
        "overnight_rate",
        "yield_2y",
        "yield_3y",
        "yield_5y",
        "yield_7y",
        "yield_10y",
        "yield_long",
        "yield_spread_10y_2y",
        "usdcad",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "The processed Bank of Canada dataset is empty."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates remain in the processed dataset."
        )

    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "Dates are not sorted in ascending order."
        )


# ---------------------------------------------------------
# 9. Display data summary
# ---------------------------------------------------------

def display_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print a summary of the processed Bank of Canada dataset.
    """

    print("\nFirst five rows:")
    print(df.head())

    print("\nLast five rows:")
    print(df.tail())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nDate range:")
    print("Start:", df["date"].min())
    print("End:", df["date"].max())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nUSD/CAD methodology transition check:")

    transition_rows = df.loc[
        df["date"].between(
            "2017-04-24",
            "2017-05-05",
        ),
        [
            "date",
            "usdcad",
        ],
    ]

    print(transition_rows)


# ---------------------------------------------------------
# 10. Save processed CSV
# ---------------------------------------------------------

def save_processed_data(
    df: pd.DataFrame,
) -> Path:
    """
    Save the complete processed Bank of Canada dataset.
    """

    df.to_csv(
        PROCESSED_FILE,
        index=False,
    )

    print("\nProcessed CSV saved to:")
    print(PROCESSED_FILE)

    return PROCESSED_FILE


# ---------------------------------------------------------
# 11. Main reusable pipeline function
# ---------------------------------------------------------

def run(
    use_cache: bool = False,
    cache_file: Path | None = None,
) -> pd.DataFrame:
    """
    Run the complete Bank of Canada ingestion pipeline.

    Parameters
    ----------
    use_cache:
        When True, rebuild the processed CSV from cached JSON
        instead of making a new API request.

    cache_file:
        Optional path to a specific cached JSON file.

    Returns
    -------
    pandas.DataFrame
        Complete processed Bank of Canada dataset.
    """

    pipeline_start = time.perf_counter()

    print("\n" + "=" * 60)
    print("BANK OF CANADA DATA INGESTION")
    print("=" * 60)

    if use_cache:

        if cache_file is None:
            raw_file = find_latest_cached_json()
        else:
            raw_file = cache_file.resolve()

            print("Using specified cached JSON:")
            print(raw_file)

    else:
        raw_file = download_raw_data()

    data = load_raw_json(raw_file)

    df = create_dataframe(data)

    df = clean_dataframe(df)

    validate_dataframe(df)

    display_summary(df)

    save_processed_data(df)

    elapsed_time = (
        time.perf_counter()
        - pipeline_start
    )

    print(
        "\nBank of Canada execution time: "
        f"{elapsed_time:.2f} seconds"
    )

    print(
        "\nBank of Canada pipeline "
        "completed successfully."
    )

    return df


# ---------------------------------------------------------
# 12. Command-line arguments
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Download and process Bank of Canada "
            "macro-financial data."
        )
    )

    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Rebuild the processed CSV from cached raw JSON "
            "without making a new API request."
        ),
    )

    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a specific cached raw JSON file."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------
# 13. Script entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    arguments = parse_arguments()

    if (
        arguments.cache_file is not None
        and not arguments.from_cache
    ):
        raise ValueError(
            "--cache-file must be used together "
            "with --from-cache."
        )

    run(
        use_cache=arguments.from_cache,
        cache_file=arguments.cache_file,
    )