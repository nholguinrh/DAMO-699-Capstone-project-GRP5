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

    # 3. Required key target & feature columns present
    required_cols = [
        "yield_spread_10y_2y", "yield_spread_10y_5y", "yield_spread_5y_2y",
        "overnight_rate", "yield_2y", "yield_5y", "yield_10y",
        "us_treasury_10y", "fed_funds_rate", "cpi_yoy", "cpi_all_items",
        "d_yield_spread_10y_2y", "d_overnight_rate", "d_cpi_yoy"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col}"

    # 4. Math formula correctness assertions
    expected_spread_10_2 = df["yield_10y"] - df["yield_2y"]
    pd.testing.assert_series_equal(
        df["yield_spread_10y_2y"], expected_spread_10_2, check_names=False
    )

    expected_spread_10_5 = df["yield_10y"] - df["yield_5y"]
    pd.testing.assert_series_equal(
        df["yield_spread_10y_5y"], expected_spread_10_5, check_names=False
    )

    # 5. Zero null values
    assert df.isnull().sum().sum() == 0, "Gold feature dataset contains unexpected null values"

    # 6. Index is DatetimeIndex and sorted
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing, "Date index is not sorted monotonically"


def test_feature_type_filtering(tmp_path: Path):
    """Test that feature_type parameter filters levels vs stationary columns correctly."""
    out_levels = tmp_path / "gold_levels.csv"
    df_levels = build_gold_features(output_path=out_levels, save=True, feature_type="levels")

    out_stat = tmp_path / "gold_stationary.csv"
    df_stat = build_gold_features(output_path=out_stat, save=True, feature_type="stationary")

    # Levels should not contain differenced d_* features
    assert not any(c.startswith("d_") for c in df_levels.columns)
    assert "yield_spread_10y_2y" in df_levels.columns

    # Stationary should contain only differenced d_* features
    assert all(c.startswith("d_") for c in df_stat.columns)
    assert "d_yield_spread_10y_2y" in df_stat.columns


def test_invalid_feature_type_raises_value_error():
    """Test that passing an invalid feature_type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid feature_type"):
        build_gold_features(save=False, feature_type="invalid_option")


def test_processed_gold_file_exists():
    """Test that canonical data/processed/gold_features.csv exists and is readable."""
    gold_file = PROCESSED_DIR / "gold_features.csv"
    assert gold_file.exists(), "data/processed/gold_features.csv does not exist"

    df = pd.read_csv(gold_file, parse_dates=["date"], index_col="date")
    assert not df.empty
    assert "yield_spread_10y_2y" in df.columns
    assert "d_yield_spread_10y_2y" in df.columns

