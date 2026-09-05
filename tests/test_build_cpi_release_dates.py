"""
Unit tests for src/build_cpi_release_dates.py's target-end handling (Issue #110).

The scheduled refresh workflow used to call this script with no arguments,
so ``target_end`` always fell back to the module's fixed ``TARGET_END``
constant -- once real time passed that date, the workflow kept "refreshing"
the mapping without ever extending it. These tests cover the pure date-logic
functions the fix (workflow now passes an explicit, run-time-computed
``--target-end``) depends on; no network access is exercised.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import build_cpi_release_dates as bcrd  # noqa: E402


class TestCoerceTargetEnd:
    def test_none_falls_back_to_module_default(self):
        assert bcrd._coerce_target_end(None) == bcrd.TARGET_END

    def test_accepts_iso_string(self):
        assert bcrd._coerce_target_end("2027-03-01") == datetime(2027, 3, 1)

    def test_accepts_datetime_passthrough(self):
        dt = datetime(2027, 3, 1)
        assert bcrd._coerce_target_end(dt) is dt

    def test_a_later_string_extends_past_the_module_default(self):
        """The exact mechanism the workflow fix relies on: passing a
        run-time-computed date must extend coverage past TARGET_END,
        not be silently capped by it."""
        later = bcrd._coerce_target_end("2030-01-01")
        assert later > bcrd.TARGET_END


class TestMissingMonths:
    def test_no_missing_when_every_month_mapped(self):
        mapping = {
            datetime(2009, 1, 1): datetime(2009, 2, 20),
            datetime(2009, 2, 1): datetime(2009, 3, 19),
        }
        # Restrict TARGET_START implicitly via a narrow target_end
        missing = bcrd.missing_months(mapping, target_end="2009-02-01")
        assert missing == []

    def test_reports_gaps_up_to_target_end(self):
        mapping = {datetime(2009, 1, 1): datetime(2009, 2, 20)}
        missing = bcrd.missing_months(mapping, target_end="2009-03-01")
        assert missing == ["2009-02", "2009-03"]

    def test_extending_target_end_surfaces_new_gaps_not_previously_checked(self):
        """This is the actual bug: with a stale target_end, months beyond it
        are never even checked, so a workflow run against the old fixed
        TARGET_END would report "complete" while real gaps exist further out."""
        mapping = {datetime(2009, 1, 1): datetime(2009, 2, 20)}

        assert bcrd.missing_months(mapping, target_end="2009-01-01") == []
        assert bcrd.missing_months(mapping, target_end="2009-03-01") == [
            "2009-02", "2009-03",
        ]


class TestIsMappingComplete:
    def test_missing_file_is_incomplete(self, tmp_path):
        assert bcrd.is_mapping_complete(tmp_path / "nope.csv") is False

    def test_complete_up_to_its_own_target_end_but_not_a_later_one(self, tmp_path):
        csv_path = tmp_path / "cpi_release_dates.csv"
        csv_path.write_text(
            "reference_month,release_date\n"
            "2009-01-01,2009-02-20\n"
            "2009-02-01,2009-03-19\n"
        )

        assert bcrd.is_mapping_complete(csv_path, target_end="2009-02-01") is True
        # Regression guard for #110: a caller that doesn't pass a later
        # target_end than what's mapped would never notice new gaps.
        assert bcrd.is_mapping_complete(csv_path, target_end="2009-04-01") is False


class TestLatestMappedMonth:
    def test_returns_none_for_missing_file(self, tmp_path):
        assert bcrd.latest_mapped_month(tmp_path / "nope.csv") is None

    def test_returns_max_reference_month(self, tmp_path):
        csv_path = tmp_path / "cpi_release_dates.csv"
        csv_path.write_text(
            "reference_month,release_date\n"
            "2009-01-01,2009-02-20\n"
            "2026-06-01,2026-07-20\n"
            "2009-02-01,2009-03-19\n"
        )

        assert bcrd.latest_mapped_month(csv_path) == datetime(2026, 6, 1)


class TestWriteMapping:
    def test_writes_rows_through_explicit_target_end_past_module_default(self, tmp_path):
        """The write path must not silently cap output at TARGET_END when a
        later target_end is explicitly requested -- otherwise passing
        --target-end from the workflow would compute a wider window but
        still truncate the file back down."""
        csv_path = tmp_path / "cpi_release_dates.csv"
        mapping = {
            datetime(2026, 6, 1): datetime(2026, 7, 20),
            datetime(2026, 7, 1): datetime(2026, 8, 20),
            datetime(2026, 8, 1): datetime(2026, 9, 21),
        }

        bcrd.write_mapping(mapping, csv_path, target_end="2026-08-01")

        written = csv_path.read_text()
        assert "2026-07-01" in written
        assert "2026-08-01" in written

    def test_never_truncates_rows_beyond_a_narrower_target_end(self, tmp_path):
        csv_path = tmp_path / "cpi_release_dates.csv"
        mapping = {
            datetime(2026, 6, 1): datetime(2026, 7, 20),
            datetime(2026, 7, 1): datetime(2026, 8, 20),
        }

        # A narrower target_end than the mapping's own max must not delete
        # forward-known release dates.
        bcrd.write_mapping(mapping, csv_path, target_end="2026-06-01")

        written = csv_path.read_text()
        assert "2026-07-01" in written
