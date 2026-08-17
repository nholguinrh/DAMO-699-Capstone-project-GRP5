"""
Unit tests for Gold-layer Feature Engineering Pipeline (Issue #30)
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure src/ is on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.gold_feature_pipeline import build_gold_features
from src.project_paths import PROCESSED_DIR


def test_build_gold_features_execution(tmp_path: Path):
    """Test that build_gold_features runs cleanly and generates expected CSV."""
    test_out = tmp_path / "test_gold_features.csv"
    df = build_gold_features(output_path=test_out, save=True)

    # 1. Output file exists
    assert test_out.exists()

    # 2. DataFrame shape & types
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 3000  # Expected multi-year trading days
    assert len(df.columns) == 26

    # 3. Required key target & feature columns present
    required_cols = [
        "yield_spread_10y_2y", "yield_spread_10y_5y", "yield_spread_5y_2y",
        "overnight_rate", "yield_2y", "yield_5y", "yield_10y", "yield_long",
        "us_treasury_10y", "fed_funds_rate", "usdcad", "cpi_yoy", "cpi_all_items",
        "d_yield_spread_10y_2y", "d_overnight_rate", "d_usdcad", "d_cpi_yoy"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col}"

    # 4. Zero null values
    assert df.isnull().sum().sum() == 0, "Gold feature dataset contains unexpected null values"

    # 5. Index is DatetimeIndex and sorted
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing, "Date index is not sorted monotonically"


def test_processed_gold_file_exists():
    """Test that canonical data/processed/gold_features.csv exists and is readable."""
    gold_file = PROCESSED_DIR / "gold_features.csv"
    assert gold_file.exists(), "data/processed/gold_features.csv does not exist"
    
    df = pd.read_csv(gold_file, parse_dates=["date"], index_col="date")
    assert not df.empty
    assert "yield_spread_10y_2y" in df.columns
    assert "d_yield_spread_10y_2y" in df.columns
