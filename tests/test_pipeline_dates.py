"""
Unit tests for src/pipeline_dates.py and the Path A ``end_date`` / ``start_date``
override plumbing (Issue #104, Part 1).
"""

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline_dates import (
    CPI_REFERENCE_START,
    DATE_END,
    DATE_START,
    clamp_to_range,
    resolve_date_range,
)

_TODAY = date(2026, 9, 2)


# ---------------------------------------------------------------------------
# resolve_date_range
# ---------------------------------------------------------------------------

def test_resolve_defaults_to_config():
    assert resolve_date_range() == (DATE_START, DATE_END)


def test_resolve_none_falls_back_per_side():
    assert resolve_date_range(None, "2026-03-31", today=_TODAY) == (DATE_START, "2026-03-31")
    assert resolve_date_range("2015-01-05", None, today=_TODAY) == ("2015-01-05", DATE_END)


def test_resolve_is_idempotent_on_iso_strings():
    once = resolve_date_range(None, "2026-08-31", today=_TODAY)
    assert resolve_date_range(*once, today=_TODAY) == once


def test_resolve_accepts_date_and_datetime():
    assert resolve_date_range(date(2015, 1, 5), datetime(2026, 3, 31, 12), today=_TODAY) == (
        "2015-01-05",
        "2026-03-31",
    )


def test_resolve_custom_default_start_for_cpi():
    assert resolve_date_range(default_start=CPI_REFERENCE_START) == (CPI_REFERENCE_START, DATE_END)


@pytest.mark.parametrize("bad", ["2026/01/01", "garbage", "2026-13-01", ""])
def test_resolve_rejects_malformed(bad):
    with pytest.raises(ValueError, match="ISO date"):
        resolve_date_range(None, bad, today=_TODAY)


def test_resolve_rejects_inverted_range():
    with pytest.raises(ValueError, match="strictly before"):
        resolve_date_range("2026-06-30", "2026-01-01", today=_TODAY)


def test_resolve_rejects_equal_range():
    with pytest.raises(ValueError, match="strictly before"):
        resolve_date_range("2026-06-30", "2026-06-30", today=_TODAY)


def test_resolve_rejects_future_end():
    with pytest.raises(ValueError, match="in the future"):
        resolve_date_range(None, "2026-12-31", today=_TODAY)


def test_resolve_allows_end_equal_to_today():
    assert resolve_date_range("2026-01-01", "2026-09-02", today=_TODAY) == (
        "2026-01-01",
        "2026-09-02",
    )


def test_resolve_rejects_wrong_type():
    with pytest.raises(TypeError):
        resolve_date_range(None, 20260101, today=_TODAY)


# ---------------------------------------------------------------------------
# clamp_to_range
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_frame():
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", "2024-12-31", freq="D"),
            "value": range(366),
        }
    )


def test_clamp_noop_when_frame_inside_window(daily_frame):
    out = clamp_to_range(daily_frame, "date", "2020-01-01", "2030-01-01")
    pd.testing.assert_frame_equal(out, daily_frame)


def test_clamp_trims_both_ends_inclusive(daily_frame):
    out = clamp_to_range(daily_frame, "date", "2024-03-01", "2024-03-31")
    assert out["date"].min() == pd.Timestamp("2024-03-01")
    assert out["date"].max() == pd.Timestamp("2024-03-31")
    assert len(out) == 31
    assert list(out.index) == list(range(31))


def test_clamp_accepts_string_column(daily_frame):
    strung = daily_frame.assign(date=daily_frame["date"].dt.strftime("%Y-%m-%d"))
    out = clamp_to_range(strung, "date", "2024-06-01", "2024-06-30")
    assert len(out) == 30


def test_clamp_missing_column_raises(daily_frame):
    with pytest.raises(KeyError, match="reference_month"):
        clamp_to_range(daily_frame, "reference_month", "2024-01-01", "2024-12-31")


# ---------------------------------------------------------------------------
# Integration: Path A run() honours end_date in --from-cache mode.
# Requires local raw JSON cache (gitignored); skipped when absent.
# ---------------------------------------------------------------------------

def _has_cache(raw_dir: Path) -> bool:
    return raw_dir.is_dir() and any(raw_dir.glob("*.json"))


@pytest.mark.integration
def test_boc_run_from_cache_clamps_to_end_date(tmp_path, monkeypatch):
    import src.boc_data_ingestion as boc

    if not _has_cache(boc.BOC_RAW_DIR):
        pytest.skip("no local Bank of Canada raw JSON cache")

    monkeypatch.setattr(boc, "PROCESSED_FILE", tmp_path / "boc.csv")
    df = boc.run(use_cache=True, end_date="2025-06-30")

    assert df["date"].max() <= pd.Timestamp("2025-06-30")
    assert df["date"].min() >= pd.Timestamp(DATE_START)


@pytest.mark.integration
def test_statcan_run_from_cache_clamps_reference_month(tmp_path, monkeypatch):
    import src.statcan_data_ingestion as statcan

    if not _has_cache(statcan.STATCAN_RAW_DIR):
        pytest.skip("no local Statistics Canada raw JSON cache")

    monkeypatch.setattr(statcan, "PROCESSED_FILE", tmp_path / "cpi.csv")
    df = statcan.run(use_cache=True, end_date="2025-12-31")

    assert df["reference_month"].max() <= pd.Timestamp("2025-12-31")


@pytest.mark.integration
def test_run_from_cache_default_window_is_byte_identical(tmp_path, monkeypatch):
    """The clamp must be a no-op for the default window."""
    import src.boc_data_ingestion as boc

    if not _has_cache(boc.BOC_RAW_DIR):
        pytest.skip("no local Bank of Canada raw JSON cache")

    default_path = tmp_path / "default.csv"
    explicit_path = tmp_path / "explicit.csv"

    monkeypatch.setattr(boc, "PROCESSED_FILE", default_path)
    boc.run(use_cache=True)

    monkeypatch.setattr(boc, "PROCESSED_FILE", explicit_path)
    boc.run(use_cache=True, start_date=DATE_START, end_date=DATE_END)

    assert default_path.read_bytes() == explicit_path.read_bytes()
