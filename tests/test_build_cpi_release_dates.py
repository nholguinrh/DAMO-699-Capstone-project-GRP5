"""
Unit tests for src/build_cpi_release_dates.py's target-end handling (Issue #110).

The scheduled refresh workflow used to call this script with no arguments,
so ``target_end`` always fell back to a fixed constant -- once real time
passed that date, the workflow kept "refreshing" the mapping without ever
extending it. These tests cover the pure date-logic functions the fix
(workflow now passes an explicit, run-time-computed ``--target-end``)
depends on; no network access is exercised.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import build_cpi_release_dates as bcrd  # noqa: E402


class TestDefaultTargetEnd:
    """The default horizon must be derived from the clock, never hardcoded --
    but the tests must pin that clock, or they inherit the exact staleness
    and month-boundary raciness Issue #110 was about.

    StatCan's schedule feed is a *forward* calendar (verified live against
    the real feed: a fetch today already carries a scheduled release date
    several reference months ahead), so the current month -- not a "last
    month" back-off -- is the correct, always-safe default."""

    @pytest.mark.parametrize(
        "now, expected",
        [
            (datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc), datetime(2026, 9, 1)),
            # Month boundary, both sides.
            (datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc), datetime(2026, 8, 1)),
            (datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc), datetime(2026, 9, 1)),
            # Day 29-31 following a shorter month.
            (datetime(2026, 3, 31, 0, 0, tzinfo=timezone.utc), datetime(2026, 3, 1)),
            (datetime(2027, 1, 31, 0, 0, tzinfo=timezone.utc), datetime(2027, 1, 1)),
        ],
    )
    def test_resolves_to_first_of_the_current_month(self, now, expected):
        assert bcrd._default_target_end(today=now) == expected

    def test_none_resolves_at_call_time_not_import_time(self, monkeypatch):
        """A long-lived process crossing a month boundary must pick up the
        new month, not replay a value snapshotted once at import."""
        seen = []

        def fake_default(today=None):
            seen.append(1)
            return datetime(2026, 9, 1)

        monkeypatch.setattr(bcrd, "_default_target_end", fake_default)
        assert bcrd._coerce_target_end(None) == datetime(2026, 9, 1)
        assert bcrd._coerce_target_end(None) == datetime(2026, 9, 1)
        assert len(seen) == 2  # re-derived on every call, not cached


class TestCoerceTargetEnd:
    def test_accepts_iso_string(self):
        assert bcrd._coerce_target_end("2027-03-01") == datetime(2027, 3, 1)

    def test_accepts_datetime_passthrough(self):
        dt = datetime(2027, 3, 1)
        assert bcrd._coerce_target_end(dt) is dt

    def test_normalizes_tz_aware_datetime_to_naive(self):
        """A caller passing e.g. datetime.now(timezone.utc) directly must not
        blow up comparisons against the naive datetimes used everywhere else
        in this module (TARGET_START, parsed mapping keys, the local-clock
        _default_target_end() default)."""
        aware = datetime(2027, 3, 1, 12, 30, tzinfo=timezone.utc)
        result = bcrd._coerce_target_end(aware)
        assert result == datetime(2027, 3, 1, 12, 30)
        assert result.tzinfo is None
        # And the result must actually be comparable to a naive datetime.
        assert result > bcrd.TARGET_START


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
        are never even checked, so a workflow run against a stale fixed
        target_end would report "complete" while real gaps exist further out."""
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
    def test_writes_rows_past_target_end_when_the_mapping_already_has_them(self, tmp_path):
        """The write path must not cap output at target_end when the mapping
        (e.g. from StatCan's forward-scheduling feed) already extends past
        it -- otherwise passing --target-end from the workflow would compute
        a wider window but still truncate the file back down."""
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

    def test_writes_forward_scheduled_rows_and_returns_the_true_count(self, tmp_path):
        """write_mapping must never cap at target_end -- and must report the
        row count it actually wrote, so refresh()'s log reconciles with the
        CSV rather than under-reporting relative to StatCan's forward
        schedule."""
        csv_path = tmp_path / "cpi_release_dates.csv"
        mapping = {
            datetime(2026, 6, 1): datetime(2026, 7, 20),
            datetime(2026, 7, 1): datetime(2026, 8, 19),
            datetime(2026, 8, 1): datetime(2026, 9, 15),
            datetime(2026, 9, 1): datetime(2026, 10, 20),
        }

        written = bcrd.write_mapping(mapping, csv_path, target_end="2026-07-01")

        rows = csv_path.read_text().strip().splitlines()[1:]
        assert written == len(rows) == 4  # not capped at target_end
        assert rows[-1].startswith("2026-09-01")  # forward schedule preserved
