"""
Gold-layer Feature Engineering Pipeline (Issue #30)
DAMO-699 Capstone Project, Group 5

Consolidates a single canonical Gold-layer feature set in `data/processed/gold_features.csv`
informed by both EDA paths and baseline model feature sets.

Scope:
- Merges Bank of Canada yield curve, FRED interest rates, USDCAD exchange rate, and StatCan CPI.
- Calculates publication release-date aligned CPI YoY percentage changes without lookahead bias.
- Computes yield curve spreads (10y-2y, 10y-5y, 5y-2y).
- Computes stationary first-differenced series (`d_*`) for downstream models (VAR, VECM, LSTM).
- Cleans and exports canonical dataset to `data/processed/gold_features.csv`.
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Handle imports whether executed directly or via module path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from project_paths import PROCESSED_DIR

logger = logging.getLogger(__name__)


def build_gold_features(
    output_path: Path | None = None,
    boc_path: Path | None = None,
    fred_path: Path | None = None,
    cpi_path: Path | None = None,
    save: bool = True,
    feature_type: str = "all",
) -> pd.DataFrame:
    """
    Construct the canonical Gold-layer feature dataset.

    Parameters
    ----------
    output_path : Path, optional
        Path to save the resulting CSV. Defaults to `data/processed/gold_features.csv`.
    boc_path : Path, optional
        Path to Bank of Canada CSV input. Defaults to `data/processed/bank_of_canada_data.csv`.
    fred_path : Path, optional
        Path to FRED CSV input. Defaults to `data/processed/fred_rates.csv`.
    cpi_path : Path, optional
        Path to StatCan CPI CSV input. Defaults to `data/processed/statcan_cpi.csv`.
    save : bool, default True
        Whether to write the resulting DataFrame to disk.
    feature_type : {"all", "levels", "stationary"}, default "all"
        Subset of features to return:
        - "all": Level features and first-differenced features (`d_*`).
        - "levels": Raw interest rates, yield spreads, USDCAD, and CPI YoY.
        - "stationary": Stationary first-differenced series (`d_*`).

    Returns
    -------
    pd.DataFrame
        Cleaned, merged, and feature-engineered Gold-layer dataset indexed by date.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "gold_features.csv"

    if feature_type not in {"all", "levels", "stationary"}:
        raise ValueError(
            f"Invalid feature_type '{feature_type}'. Expected one of 'all', 'levels', 'stationary'."
        )

    # 1. Load silver/processed input datasets
    boc_file = boc_path if boc_path is not None else PROCESSED_DIR / "bank_of_canada_data.csv"
    fred_file = fred_path if fred_path is not None else PROCESSED_DIR / "fred_rates.csv"
    cpi_file = cpi_path if cpi_path is not None else PROCESSED_DIR / "statcan_cpi.csv"

    for file_path in [boc_file, fred_file, cpi_file]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required input file missing: {file_path}. "
                "Ensure data collection and cleaning pipelines have run."
            )

    boc = pd.read_csv(boc_file, parse_dates=["date"])
    fred = pd.read_csv(fred_file, parse_dates=["date"])
    cpi = pd.read_csv(cpi_file, parse_dates=["reference_month", "release_date"])

    # 2. Schema Validation: verify mandatory columns exist
    mandatory_boc = ["date", "yield_2y", "yield_5y", "yield_10y", "overnight_rate"]
    missing_boc = [col for col in mandatory_boc if col not in boc.columns]
    if missing_boc:
        raise KeyError(f"Missing mandatory column(s) in Bank of Canada dataset: {missing_boc}")

    mandatory_fred = ["date", "us_treasury_10y", "fed_funds_rate"]
    missing_fred = [col for col in mandatory_fred if col not in fred.columns]
    if missing_fred:
        raise KeyError(f"Missing mandatory column(s) in FRED dataset: {missing_fred}")

    # 3. Merge daily series using outer join to preserve US/Canadian market holidays
    boc_cols = [
        "date", "overnight_rate", "yield_2y", "yield_3y", "yield_5y",
        "yield_7y", "yield_10y", "yield_long", "usdcad"
    ]
    avail_boc_cols = [c for c in boc_cols if c in boc.columns]

    df = (
        boc[avail_boc_cols]
        .merge(fred[["date", "us_treasury_10y", "fed_funds_rate"]], on="date", how="outer")
        .sort_values("date")
        .set_index("date")
    )

    # Forward-fill single-day market holiday gaps (up to 2 days) between US and CA calendars
    df = df.ffill(limit=2)

    # 4. Process StatCan CPI & calculate YoY percentage change
    cpi_sorted = cpi.sort_values("reference_month").copy()
    cpi_sorted["reference_month"] = pd.to_datetime(cpi_sorted["reference_month"]).dt.to_period("M").dt.to_timestamp()

    # Reindex onto full contiguous monthly grid to ensure 12-period lag is exactly 12 calendar months
    full_month_range = pd.date_range(
        cpi_sorted["reference_month"].min(), cpi_sorted["reference_month"].max(), freq="MS"
    )
    cpi_monthly = (
        cpi_sorted.set_index("reference_month")
        .reindex(full_month_range)
    )
    cpi_monthly["cpi_yoy"] = cpi_monthly["cpi_all_items"].pct_change(12) * 100

    # Restore reference_month and drop unneeded empty grid rows
    cpi_sorted = cpi_monthly.dropna(subset=["cpi_all_items"]).reset_index().rename(columns={"index": "reference_month"})

    # Drop rows without mapped release dates
    cpi_sorted = cpi_sorted.dropna(subset=["release_date"])

    # Deduplicate release dates (keep latest update if duplicate)
    cpi_by_release = (
        cpi_sorted.sort_values("release_date")
        .drop_duplicates(subset="release_date", keep="last")
        .set_index("release_date")
    )

    # Forward fill CPI series onto daily calendar grid up to max date
    daily_grid = pd.date_range(df.index.min(), df.index.max(), freq="D")
    cpi_yoy_daily = cpi_by_release["cpi_yoy"].reindex(daily_grid).ffill()
    cpi_all_daily = cpi_by_release["cpi_all_items"].reindex(daily_grid).ffill()

    # Reindex onto daily trading index
    df["cpi_yoy"] = cpi_yoy_daily.reindex(df.index)
    df["cpi_all_items"] = cpi_all_daily.reindex(df.index)

    # 5. Construct Yield Curve Spreads
    df["yield_spread_10y_2y"] = df["yield_10y"] - df["yield_2y"]
    df["yield_spread_10y_5y"] = df["yield_10y"] - df["yield_5y"]
    df["yield_spread_5y_2y"] = df["yield_5y"] - df["yield_2y"]

    # 6. Compute Stationary First Differences (`d_*`)
    diff_cols = [
        "yield_spread_10y_2y", "yield_spread_10y_5y", "yield_spread_5y_2y",
        "overnight_rate", "yield_2y", "yield_5y", "yield_10y",
        "us_treasury_10y", "fed_funds_rate", "usdcad", "cpi_yoy"
    ]
    for col in diff_cols:
        if col in df.columns:
            df[f"d_{col}"] = df[col].diff()

    # 7. Clean initial NAs resulting from YoY calculation & first differences
    df = df.dropna()

    # 8. Filter by requested feature_type
    diff_feature_names = [f"d_{c}" for c in diff_cols if f"d_{c}" in df.columns]
    level_feature_names = [c for c in df.columns if c not in diff_feature_names]

    if feature_type == "levels":
        df = df[level_feature_names]
    elif feature_type == "stationary":
        df = df[diff_feature_names]

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info(
            "Saved Gold feature dataset (%s) with %d rows and %d columns to %s",
            feature_type, len(df), len(df.columns), output_path
        )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df_gold = build_gold_features()
    print(f"Successfully generated Gold feature pipeline dataset:")
    print(f"Shape: {df_gold.shape}")
    print(f"Date range: {df_gold.index.min().strftime('%Y-%m-%d')} to {df_gold.index.max().strftime('%Y-%m-%d')}")
    print("Columns:", list(df_gold.columns))

