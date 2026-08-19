"""
Pytest configuration and shared fixtures for the test suite.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test requiring local data files")


@pytest.fixture
def synthetic_gold_inputs(tmp_path: Path) -> dict[str, Path]:
    """
    Generate synthetic Bank of Canada, FRED, and StatCan CPI CSV files
    in a temporary directory for unit testing without external data dependencies.
    """
    # 1. Bank of Canada synthetic daily series (40 business days)
    boc_dates = pd.bdate_range("2023-01-02", periods=40)
    boc_df = pd.DataFrame(
        {
            "date": boc_dates,
            "overnight_rate": np.linspace(2.5, 3.0, 40),
            "yield_2y": np.linspace(3.0, 3.5, 40),
            "yield_3y": np.linspace(3.1, 3.6, 40),
            "yield_5y": np.linspace(3.2, 3.7, 40),
            "yield_7y": np.linspace(3.3, 3.8, 40),
            "yield_10y": np.linspace(3.5, 4.0, 40),
            "yield_long": np.linspace(3.6, 4.1, 40),
            "usdcad": np.linspace(1.30, 1.35, 40),
        }
    )
    boc_path = tmp_path / "bank_of_canada_data.csv"
    boc_df.to_csv(boc_path, index=False)

    # 2. FRED synthetic daily series (40 business days)
    fred_df = pd.DataFrame(
        {
            "date": boc_dates,
            "us_treasury_10y": np.linspace(3.8, 4.2, 40),
            "fed_funds_rate": np.linspace(4.5, 4.75, 40),
        }
    )
    fred_path = tmp_path / "fred_rates.csv"
    fred_df.to_csv(fred_path, index=False)

    # 3. StatCan CPI synthetic monthly series (24 months: 2021-06 to 2023-05)
    cpi_months = pd.date_range("2021-06-01", periods=24, freq="MS")
    cpi_df = pd.DataFrame(
        {
            "reference_month": cpi_months,
            "release_date": [m + pd.Timedelta(days=20) for m in cpi_months],
            "cpi_all_items": [100.0 + i * 0.5 for i in range(24)],
        }
    )
    cpi_path = tmp_path / "statcan_cpi.csv"
    cpi_df.to_csv(cpi_path, index=False)

    return {
        "boc_path": boc_path,
        "fred_path": fred_path,
        "cpi_path": cpi_path,
    }
