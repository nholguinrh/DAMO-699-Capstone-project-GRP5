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
    save: bool = True
) -> pd.DataFrame:
    """
    Construct the canonical Gold-layer feature dataset.

    Parameters
    ----------
    output_path : Path, optional
        Path to save the resulting CSV. Defaults to `data/processed/gold_features.csv`.
    save : bool, default True
        Whether to write the resulting DataFrame to disk.

    Returns
    -------
    pd.DataFrame
        Cleaned, merged, and feature-engineered Gold-layer dataset indexed by date.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "gold_features.csv"

    # 1. Load silver/processed input datasets
    boc_file = PROCESSED_DIR / "bank_of_canada_data.csv"
    fred_file = PROCESSED_DIR / "fred_rates.csv"
    cpi_file = PROCESSED_DIR / "statcan_cpi.csv"

    for file_path in [boc_file, fred_file, cpi_file]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required input file missing: {file_path}. "
                "Ensure data collection and cleaning pipelines have run."
            )

    boc = pd.read_csv(boc_file, parse_dates=["date"])
    fred = pd.read_csv(fred_file, parse_dates=["date"])
    cpi = pd.read_csv(cpi_file, parse_dates=["reference_month", "release_date"])

    # 2. Merge daily business-day series (BoC + FRED)
    boc_cols = [
        "date", "overnight_rate", "yield_2y", "yield_3y", "yield_5y",
        "yield_7y", "yield_10y", "yield_long", "usdcad"
    ]
    # Retain columns existing in boc
    avail_boc_cols = [c for c in boc_cols if c in boc.columns]
    
    df = (
        boc[avail_boc_cols]
        .merge(fred[["date", "us_treasury_10y", "fed_funds_rate"]], on="date", how="inner")
        .sort_values("date")
        .set_index("date")
    )

    # 3. Process StatCan CPI & calculate YoY percentage change
    cpi_sorted = cpi.sort_values("reference_month").copy()
    cpi_sorted["cpi_yoy"] = cpi_sorted["cpi_all_items"].pct_change(12) * 100
    
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

    # 4. Construct Yield Curve Spreads
    df["yield_spread_10y_2y"] = df["yield_10y"] - df["yield_2y"]
    df["yield_spread_10y_5y"] = df["yield_10y"] - df["yield_5y"]
    df["yield_spread_5y_2y"] = df["yield_5y"] - df["yield_2y"]

    # 5. Compute Stationary First Differences (`d_*`)
    diff_cols = [
        "yield_spread_10y_2y", "yield_spread_10y_5y", "yield_spread_5y_2y",
        "overnight_rate", "yield_2y", "yield_5y", "yield_10y",
        "us_treasury_10y", "fed_funds_rate", "usdcad", "cpi_yoy"
    ]
    for col in diff_cols:
        if col in df.columns:
            df[f"d_{col}"] = df[col].diff()

    # 6. Clean initial NAs resulting from YoY calculation & first differences
    df = df.dropna()

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info("Saved Gold feature dataset with %d rows and %d columns to %s", len(df), len(df.columns), output_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df_gold = build_gold_features()
    print(f"Successfully generated Gold feature pipeline dataset:")
    print(f"Shape: {df_gold.shape}")
    print(f"Date range: {df_gold.index.min().strftime('%Y-%m-%d')} to {df_gold.index.max().strftime('%Y-%m-%d')}")
    print("Columns:", list(df_gold.columns))
