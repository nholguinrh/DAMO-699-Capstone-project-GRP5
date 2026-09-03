"""
Unit tests for src/pipeline_runner.py (Issue #104, Part 3).

The individual Path A ``run_*`` callables and ``build_gold_features`` are
monkeypatched, so these tests do no network or file IO beyond a tmp status file.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import pipeline_runner as pr


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Patch the three source runners + gold builder with recording fakes."""
    calls: list[tuple] = []

    def make_fake(name, *, fail_live=False, fail_cache=False):
        def _fake(use_cache=False, start_date=None, end_date=None):
            calls.append((name, "cache" if use_cache else "live", start_date, end_date))
            if use_cache and fail_cache:
                raise RuntimeError(f"{name} cache broken")
            if not use_cache and fail_live:
                raise RuntimeError(f"{name} live broken")
            return pd.DataFrame({"date": pd.to_datetime(["2009-01-02"]), "v": [1.0]})

        return _fake

    monkeypatch.setattr(pr, "run_boc", make_fake("boc"))
    monkeypatch.setattr(pr, "run_fred", make_fake("fred"))
    monkeypatch.setattr(pr, "run_statcan", make_fake("statcan"))

    def _fake_gold(save=True):
        calls.append(("gold", "save" if save else "nosave", None, None))
        idx = pd.to_datetime(["2010-03-22", "2026-06-30"])
        return pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=idx)

    monkeypatch.setattr(pr, "build_gold_features", _fake_gold)
    monkeypatch.setattr(pr, "latest_mapped_month", lambda *a, **k: pd.Timestamp("2026-06-01"))
    monkeypatch.setattr(pr, "STATUS_PATH", tmp_path / "pipeline_run_status.json")

    return {"calls": calls, "make_fake": make_fake, "monkeypatch": monkeypatch, "tmp_path": tmp_path}


# ---------------------------------------------------------------------------
# run_pipeline — happy paths
# ---------------------------------------------------------------------------

def test_cache_mode_runs_all_sources_then_gold(patched):
    report = pr.run_pipeline(mode="cache")

    assert report["status"] == "success"
    assert [s["status"] for s in report["sources"].values()] == ["success"] * 3
    assert all(s["mode"] == "cache" for s in report["sources"].values())
    assert report["gold"]["status"] == "success"
    assert report["gold"]["rows"] == 2
    assert report["gold"]["date_range"] == ["2010-03-22", "2026-06-30"]
    assert report["window"] == {"start": "2009-01-02", "end": "2026-06-30"}
    # order: boc, fred, statcan, then gold
    assert [c[0] for c in patched["calls"]] == ["boc", "fred", "statcan", "gold"]


def test_end_date_is_threaded_to_every_source(patched):
    pr.run_pipeline(mode="cache", end_date="2025-12-31")
    for name, _mode, _start, end in patched["calls"][:3]:
        assert end == "2025-12-31", name


def test_status_file_written_and_readback(patched):
    report = pr.run_pipeline(mode="cache")
    assert pr.STATUS_PATH.exists()
    assert pr.read_run_status() == report


def test_write_status_can_be_disabled(patched):
    pr.run_pipeline(mode="cache", write_status=False)
    assert not pr.STATUS_PATH.exists()


# ---------------------------------------------------------------------------
# run_pipeline — failure handling
# ---------------------------------------------------------------------------

def test_bad_mode_raises(patched):
    with pytest.raises(ValueError, match="mode must be one of"):
        pr.run_pipeline(mode="turbo")


def test_future_end_date_rejected_before_any_work(patched):
    with pytest.raises(ValueError, match="in the future"):
        pr.run_pipeline(mode="cache", end_date="2099-01-01")
    assert patched["calls"] == []


def test_cache_failure_is_captured_and_gold_skipped(patched):
    patched["monkeypatch"].setattr(pr, "run_fred", patched["make_fake"]("fred", fail_cache=True))

    report = pr.run_pipeline(mode="cache")

    assert report["sources"]["fred"]["status"] == "failed"
    assert "fred cache broken" in report["sources"]["fred"]["error"]
    assert report["gold"]["status"] == "skipped"
    assert report["status"] == "error"
    assert "gold" not in [c[0] for c in patched["calls"]]


def test_live_failure_falls_back_to_cache_per_source(patched):
    patched["monkeypatch"].setattr(pr, "run_boc", patched["make_fake"]("boc", fail_live=True))

    report = pr.run_pipeline(mode="live")

    boc = report["sources"]["boc"]
    assert boc["status"] == "warning"
    assert boc["mode"] == "cached_fallback"
    assert report["sources"]["fred"]["mode"] == "live"
    assert report["gold"]["status"] == "success"
    assert report["status"] == "warning"
    # boc tried live, then cache
    boc_calls = [c for c in patched["calls"] if c[0] == "boc"]
    assert [c[1] for c in boc_calls] == ["live", "cache"]


def test_live_and_cache_both_fail_marks_source_failed(patched):
    patched["monkeypatch"].setattr(
        pr, "run_statcan", patched["make_fake"]("statcan", fail_live=True, fail_cache=True)
    )

    report = pr.run_pipeline(mode="live")

    assert report["sources"]["statcan"]["status"] == "failed"
    assert "live:" in report["sources"]["statcan"]["error"]
    assert "cache:" in report["sources"]["statcan"]["error"]
    assert report["gold"]["status"] == "skipped"
    assert report["status"] == "error"


def test_gold_failure_is_captured(patched):
    def _boom(save=True):
        raise RuntimeError("gold merge failed")

    patched["monkeypatch"].setattr(pr, "build_gold_features", _boom)

    report = pr.run_pipeline(mode="cache")

    assert report["gold"]["status"] == "failed"
    assert "gold merge failed" in report["gold"]["error"]
    assert report["status"] == "error"


def test_cpi_coverage_warning_when_map_is_behind(patched):
    patched["monkeypatch"].setattr(pr, "latest_mapped_month", lambda *a, **k: pd.Timestamp("2026-05-01"))

    report = pr.run_pipeline(mode="cache", end_date="2026-06-30")

    assert any("cpi_release_dates.csv" in w for w in report["warnings"])


def test_no_cpi_warning_when_map_covers_end(patched):
    report = pr.run_pipeline(mode="cache")
    assert report["warnings"] == []


# ---------------------------------------------------------------------------
# read_run_status robustness
# ---------------------------------------------------------------------------

def test_read_run_status_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "STATUS_PATH", tmp_path / "nope.json")
    assert pr.read_run_status() is None


def test_read_run_status_corrupt_returns_none(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(pr, "STATUS_PATH", bad)
    assert pr.read_run_status() is None


# ---------------------------------------------------------------------------
# Integration: real Path A cache run (needs local raw JSON cache)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_cache_run(monkeypatch, tmp_path):
    import src.boc_data_ingestion as boc

    raw = boc.BOC_RAW_DIR
    if not (raw.is_dir() and any(raw.glob("*.json"))):
        pytest.skip("no local raw JSON cache")

    monkeypatch.setattr(pr, "STATUS_PATH", tmp_path / "status.json")
    report = pr.run_pipeline(mode="cache", write_status=True)

    assert report["status"] == "success"
    assert report["gold"]["status"] == "success"
    assert report["gold"]["rows"] > 3000
    assert pr.read_run_status()["status"] == "success"
