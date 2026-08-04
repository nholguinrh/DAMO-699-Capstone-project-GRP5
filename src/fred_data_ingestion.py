import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from http_utils import get_with_retry
from project_paths import PROJECT_ROOT, PROCESSED_DIR, RAW_DIR


# ---------------------------------------------------------
# 1. Folder configuration
# ---------------------------------------------------------

FRED_RAW_DIR = RAW_DIR / "fred"

FRED_RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_FILE = PROCESSED_DIR / "fred_rates.csv"


# ---------------------------------------------------------
# 2. FRED configuration
# ---------------------------------------------------------

START_DATE = "2009-01-02"
END_DATE = "2026-06-30"

BASE_URL = (
    "https://api.stlouisfed.org/"
    "fred/series/observations"
)

SERIES = {
    "DGS10": "us_treasury_10y",
    "DFF": "fed_funds_rate",
}


# ---------------------------------------------------------
# 3. Load the API key
# ---------------------------------------------------------

def load_api_key() -> str:
    """
    Load the FRED API key from the repository-root .env file.

    The key is loaded only when an API request is required.
    Cache rebuilding does not require an API key.
    """
    env_path = PROJECT_ROOT / ".env"

    load_dotenv(
        dotenv_path=env_path,
        override=False,
    )

    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise ValueError(
            "FRED_API_KEY was not found. "
            f"Expected .env location: {env_path}"
        )

    return api_key


# ---------------------------------------------------------
# 4. Download one FRED series
# ---------------------------------------------------------

def download_series(
    series_id: str,
    api_key: str,
) -> Path:
    """
    Download one FRED series using synchronous requests and
    manual exponential back-off supplied by get_with_retry().

    Returns
    -------
    Path
        Location of the saved raw JSON response.
    """

    if series_id not in SERIES:
        raise ValueError(
            f"Unsupported FRED series: {series_id}"
        )

    print(f"\nDownloading FRED series {series_id}...")

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": END_DATE,
    }

    response = get_with_retry(
        BASE_URL,
        params=params,
        timeout=60,
        max_retries=5,
    )

    print(
        f"{series_id} status code:",
        response.status_code,
    )

    data = response.json()

    observations = data.get("observations")

    if observations is None:
        raise ValueError(
            f"FRED response for {series_id} does not contain "
            "an 'observations' section."
        )

    if not observations:
        raise ValueError(
            f"FRED returned no observations for {series_id}."
        )

    print(
        f"{series_id} observation count:",
        len(observations),
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    raw_file = (
        FRED_RAW_DIR
        / f"{series_id}_{timestamp}.json"
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

    print(f"{series_id} raw JSON saved to:")
    print(raw_file)

    return raw_file


# ---------------------------------------------------------
# 5. Download all FRED series sequentially
# ---------------------------------------------------------

def download_all_series() -> dict[str, Path]:
    """
    Download each configured FRED series sequentially.

    DGS10 is downloaded first, followed by DFF.
    No concurrent or asynchronous requests are used.
    """

    api_key = load_api_key()

    raw_files = {}

    for series_id in SERIES:
        raw_files[series_id] = download_series(
            series_id=series_id,
            api_key=api_key,
        )

    return raw_files


# ---------------------------------------------------------
# 6. Find cached JSON files
# ---------------------------------------------------------

def find_latest_cached_json(
    series_id: str,
) -> Path:
    """
    Find the latest cached JSON file for a FRED series.
    """

    cached_files = sorted(
        FRED_RAW_DIR.glob(f"{series_id}_*.json"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )

    if not cached_files:
        raise FileNotFoundError(
            f"No cached JSON was found for {series_id} "
            f"in {FRED_RAW_DIR}"
        )

    latest_file = cached_files[0]

    print(f"Using cached {series_id} JSON:")
    print(latest_file)

    return latest_file


def find_all_latest_cached_files() -> dict[str, Path]:
    """
    Find the latest cached raw file for every configured series.
    """

    return {
        series_id: find_latest_cached_json(series_id)
        for series_id in SERIES
    }


# ---------------------------------------------------------
# 7. Load and validate raw JSON
# ---------------------------------------------------------

def load_raw_json(
    raw_file: Path,
    series_id: str,
) -> dict:
    """
    Load and validate one cached FRED JSON response.
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

    observations = data.get("observations")

    if observations is None:
        raise ValueError(
            f"Cached {series_id} JSON does not contain "
            "an 'observations' section."
        )

    if not observations:
        raise ValueError(
            f"Cached {series_id} JSON contains no observations."
        )

    return data


# ---------------------------------------------------------
# 8. Convert one series to a DataFrame
# ---------------------------------------------------------

def create_series_dataframe(
    data: dict,
    series_id: str,
) -> pd.DataFrame:
    """
    Convert one FRED JSON response into a two-column DataFrame.
    """

    column_name = SERIES[series_id]

    df = pd.DataFrame(
        data["observations"]
    )

    required_raw_columns = {
        "date",
        "value",
    }

    missing_raw_columns = (
        required_raw_columns
        - set(df.columns)
    )

    if missing_raw_columns:
        raise ValueError(
            f"{series_id} raw data is missing columns: "
            f"{sorted(missing_raw_columns)}"
        )

    df = df[
        [
            "date",
            "value",
        ]
    ].copy()

    df = df.rename(
        columns={
            "value": column_name,
        }
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df[column_name] = pd.to_numeric(
        df[column_name].replace(".", pd.NA),
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=["date"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------
# 9. Merge FRED series using a trading-day base
# ---------------------------------------------------------

def merge_series_dataframes(
    series_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge DGS10 and DFF without creating artificial weekend rows.

    DGS10 is used as the base calendar because Treasury yields
    are published on business/market observation dates.

    DFF is aligned only to those DGS10 dates. No unrestricted
    forward filling is performed.
    """

    required_series = set(SERIES)

    missing_series = (
        required_series
        - set(series_frames)
    )

    if missing_series:
        raise ValueError(
            "Missing FRED DataFrames for: "
            f"{sorted(missing_series)}"
        )

    dgs10_df = series_frames["DGS10"]
    dff_df = series_frames["DFF"]

    fred_df = dgs10_df.merge(
        dff_df,
        on="date",
        how="left",
        validate="one_to_one",
    )

    fred_df = (
        fred_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # No unrestricted ffill() is applied.
    # Missing observations remain visible for later calendar
    # alignment and modeling decisions.

    return fred_df


# ---------------------------------------------------------
# 10. Validate the processed dataset
# ---------------------------------------------------------

def validate_dataframe(
    fred_df: pd.DataFrame,
) -> None:
    """
    Validate the final processed FRED dataset.
    """

    required_columns = {
        "date",
        "us_treasury_10y",
        "fed_funds_rate",
    }

    missing_columns = (
        required_columns
        - set(fred_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Processed FRED dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if fred_df.empty:
        raise ValueError(
            "The processed FRED dataset is empty."
        )

    if fred_df["date"].isna().any():
        raise ValueError(
            "The processed FRED dataset contains invalid dates."
        )

    if fred_df["date"].duplicated().any():
        raise ValueError(
            "The processed FRED dataset contains duplicate dates."
        )

    if not fred_df["date"].is_monotonic_increasing:
        raise ValueError(
            "FRED dates are not sorted in ascending order."
        )

    weekend_rows = (
        fred_df["date"].dt.dayofweek >= 5
    ).sum()

    if weekend_rows:
        print(
            "\nWarning:",
            weekend_rows,
            "weekend rows remain in the FRED dataset.",
        )
    else:
        print(
            "\nTrading-calendar check: "
            "No weekend rows were created."
        )


# ---------------------------------------------------------
# 11. Display a data summary
# ---------------------------------------------------------

def display_summary(
    fred_df: pd.DataFrame,
) -> None:
    """
    Print a summary of the processed FRED dataset.
    """

    print("\nFirst five rows:")
    print(fred_df.head())

    print("\nLast five rows:")
    print(fred_df.tail())

    print("\nDataset shape:")
    print(fred_df.shape)

    print("\nColumns:")
    print(list(fred_df.columns))

    print("\nDate range:")
    print("Start:", fred_df["date"].min())
    print("End:", fred_df["date"].max())

    print("\nMissing values:")
    print(fred_df.isna().sum())


# ---------------------------------------------------------
# 12. Save processed output
# ---------------------------------------------------------

def save_processed_data(
    fred_df: pd.DataFrame,
) -> Path:
    """
    Save the processed FRED dataset.
    """

    fred_df.to_csv(
        PROCESSED_FILE,
        index=False,
    )

    print("\nProcessed FRED CSV saved to:")
    print(PROCESSED_FILE)

    return PROCESSED_FILE


# ---------------------------------------------------------
# 13. Reusable pipeline function
# ---------------------------------------------------------

def run(
    use_cache: bool = False,
    cache_files: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """
    Run the complete FRED ingestion pipeline.

    Parameters
    ----------
    use_cache:
        When True, rebuild the processed output from cached
        raw JSON without making API requests.

    cache_files:
        Optional mapping containing specific cached files:

        {
            "DGS10": Path(...),
            "DFF": Path(...)
        }

    Returns
    -------
    pandas.DataFrame
        Processed FRED dataset.
    """

    pipeline_start = time.perf_counter()

    print("\n" + "=" * 60)
    print("FRED DATA INGESTION")
    print("=" * 60)

    if use_cache:

        if cache_files is None:
            raw_files = find_all_latest_cached_files()
        else:
            missing_cache_keys = (
                set(SERIES)
                - set(cache_files)
            )

            if missing_cache_keys:
                raise ValueError(
                    "Specific cache mapping is missing: "
                    f"{sorted(missing_cache_keys)}"
                )

            raw_files = {
                series_id: Path(
                    cache_files[series_id]
                ).resolve()
                for series_id in SERIES
            }

            for series_id, raw_file in raw_files.items():
                print(
                    f"Using specified {series_id} cache:"
                )
                print(raw_file)

    else:
        raw_files = download_all_series()

    series_frames = {}

    for series_id in SERIES:

        data = load_raw_json(
            raw_file=raw_files[series_id],
            series_id=series_id,
        )

        series_frames[series_id] = (
            create_series_dataframe(
                data=data,
                series_id=series_id,
            )
        )

    fred_df = merge_series_dataframes(
        series_frames
    )

    validate_dataframe(fred_df)

    display_summary(fred_df)

    save_processed_data(fred_df)

    elapsed_time = (
        time.perf_counter()
        - pipeline_start
    )

    print(
        "\nFRED execution time: "
        f"{elapsed_time:.2f} seconds"
    )

    print(
        "\nFRED pipeline completed successfully."
    )

    return fred_df


# ---------------------------------------------------------
# 14. Command-line arguments
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Download and process FRED interest-rate data."
        )
    )

    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Rebuild fred_rates.csv from the latest cached "
            "DGS10 and DFF JSON files."
        ),
    )

    parser.add_argument(
        "--dgs10-cache-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a specific cached DGS10 JSON file."
        ),
    )

    parser.add_argument(
        "--dff-cache-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a specific cached DFF JSON file."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------
# 15. Script entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    arguments = parse_arguments()

    specific_cache_requested = (
        arguments.dgs10_cache_file is not None
        or arguments.dff_cache_file is not None
    )

    if (
        specific_cache_requested
        and not arguments.from_cache
    ):
        raise ValueError(
            "Cache-file arguments must be used together "
            "with --from-cache."
        )

    if specific_cache_requested:

        if arguments.dgs10_cache_file is None:
            raise ValueError(
                "--dgs10-cache-file is required when specifying "
                "individual cache files."
            )

        if arguments.dff_cache_file is None:
            raise ValueError(
                "--dff-cache-file is required when specifying "
                "individual cache files."
            )

        selected_cache_files = {
            "DGS10": arguments.dgs10_cache_file,
            "DFF": arguments.dff_cache_file,
        }

    else:
        selected_cache_files = None

    run(
        use_cache=arguments.from_cache,
        cache_files=selected_cache_files,
    )