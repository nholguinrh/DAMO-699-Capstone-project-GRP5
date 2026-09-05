"""
Unit tests for the append-only CPI observation archive (Issue #109):
StatCan's getDataFromVectorByReferencePeriodRange endpoint always returns a
fixed ~210-month rolling window regardless of the requested start/end dates,
so a single pull silently loses older history as new months publish. These
tests cover load_archive / merge_into_archive / save_archive /
rebuild_archive_from_raw_cache in src/statcan_data_ingestion.py without any
network access.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import statcan_data_ingestion as sc  # noqa: E402


def _cpi_df(months: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "reference_month": pd.to_datetime(months),
        "cpi_all_items": values,
    })


def _raw_json(months: list[str], values: list[float]) -> dict:
    """Build a minimal StatCan API response for the given reference months."""
    return [{
        "status": "SUCCESS",
        "object": {
            "vectorDataPoint": [
                {"refPer": m, "value": v} for m, v in zip(months, values)
            ]
        },
    }]


class TestLoadArchive:
    def test_missing_file_returns_empty_frame_with_correct_columns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sc, "ARCHIVE_FILE", tmp_path / "archive.csv")
        archive_df = sc.load_archive()
        assert archive_df.empty
        assert list(archive_df.columns) == sc.ARCHIVE_COLUMNS

    def test_round_trips_through_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sc, "ARCHIVE_FILE", tmp_path / "archive.csv")
        original = _cpi_df(["2020-01-01", "2020-02-01"], [100.0, 100.5])
        sc.save_archive(original)

        loaded = sc.load_archive()
        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True), original.reset_index(drop=True)
        )


class TestMergeIntoArchive:
    def test_union_preserves_history_the_new_pull_does_not_cover(self):
        """The core bug: a later pull's rolling window has slid forward and
        no longer includes the oldest reference month -- merging must not
        drop it."""
        archive = _cpi_df(["2009-01-01", "2009-02-01", "2009-03-01"], [113.0, 113.8, 114.0])
        new_pull = _cpi_df(["2009-02-01", "2009-03-01", "2009-04-01"], [113.8, 114.0, 113.9])

        merged = sc.merge_into_archive(archive, new_pull)

        assert list(merged["reference_month"].dt.strftime("%Y-%m-%d")) == [
            "2009-01-01", "2009-02-01", "2009-03-01", "2009-04-01",
        ]

    def test_new_pull_value_wins_on_overlap(self):
        """A revised value in a newer pull should overwrite the archived one."""
        archive = _cpi_df(["2020-01-01"], [100.0])
        new_pull = _cpi_df(["2020-01-01"], [100.5])

        merged = sc.merge_into_archive(archive, new_pull)

        assert len(merged) == 1
        assert merged.iloc[0]["cpi_all_items"] == pytest.approx(100.5)

    def test_no_duplicate_reference_months(self):
        archive = _cpi_df(["2020-01-01", "2020-02-01"], [100.0, 100.5])
        new_pull = _cpi_df(["2020-02-01", "2020-03-01"], [100.5, 101.0])

        merged = sc.merge_into_archive(archive, new_pull)

        assert not merged["reference_month"].duplicated().any()

    def test_result_is_sorted_chronologically(self):
        archive = _cpi_df(["2020-03-01", "2020-01-01"], [101.0, 100.0])
        new_pull = _cpi_df(["2020-02-01"], [100.5])

        merged = sc.merge_into_archive(archive, new_pull)

        assert merged["reference_month"].is_monotonic_increasing

    def test_merging_into_empty_archive_returns_new_pull(self):
        empty = pd.DataFrame(columns=sc.ARCHIVE_COLUMNS)
        new_pull = _cpi_df(["2020-01-01"], [100.0])

        merged = sc.merge_into_archive(empty, new_pull)

        assert len(merged) == 1
        assert merged.iloc[0]["reference_month"] == pd.Timestamp("2020-01-01")


class TestRebuildArchiveFromRawCache:
    def test_unions_all_cached_pulls_oldest_first(self, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)
        monkeypatch.setattr(sc, "ARCHIVE_FILE", tmp_path / "archive.csv")

        # Simulate the exact bug: an older pull covering 2009-01 - 2009-03,
        # and a newer pull whose rolling window has slid forward to
        # 2009-02 - 2009-04 and no longer includes 2009-01.
        older = raw_dir / f"cpi_{sc.VECTOR_ID}_20260101_000000.json"
        newer = raw_dir / f"cpi_{sc.VECTOR_ID}_20260201_000000.json"
        older.write_text(json.dumps(_raw_json(
            ["2009-01-01", "2009-02-01", "2009-03-01"], [113.0, 113.8, 114.0]
        )))
        newer.write_text(json.dumps(_raw_json(
            ["2009-02-01", "2009-03-01", "2009-04-01"], [113.8, 114.0, 113.9]
        )))
        # mtime ordering matters (rebuild sorts by mtime, oldest first)
        import os
        import time
        os.utime(older, (time.time() - 100, time.time() - 100))
        os.utime(newer, (time.time(), time.time()))

        rebuilt = sc.rebuild_archive_from_raw_cache()

        assert list(rebuilt["reference_month"].dt.strftime("%Y-%m-%d")) == [
            "2009-01-01", "2009-02-01", "2009-03-01", "2009-04-01",
        ]
        # Persisted, not just returned
        assert sc.ARCHIVE_FILE.exists()

    def test_raises_when_no_cached_files(self, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)

        with pytest.raises(FileNotFoundError):
            sc.rebuild_archive_from_raw_cache()


class TestRunUsesArchive:
    def test_run_with_cache_extends_past_a_single_pulls_window(
        self, tmp_path, monkeypatch
    ):
        """The regression this issue is about: even though the cached pull
        used by this run doesn't cover 2009-01, a pre-existing archive
        (seeded from an earlier pull, or committed to git) should let the
        requested window still include it."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)
        monkeypatch.setattr(sc, "ARCHIVE_FILE", tmp_path / "archive.csv")
        monkeypatch.setattr(sc, "PROCESSED_FILE", tmp_path / "statcan_cpi.csv")

        # Pre-seed the archive with a month the cached pull below won't have.
        sc.save_archive(_cpi_df(["2009-01-01"], [113.0]))

        cache_file = raw_dir / f"cpi_{sc.VECTOR_ID}_20260201_000000.json"
        cache_file.write_text(json.dumps(_raw_json(
            ["2009-02-01", "2009-03-01"], [113.8, 114.0]
        )))

        release_df = pd.DataFrame({
            "reference_month": ["2009-01-01", "2009-02-01", "2009-03-01"],
            "release_date": ["2009-02-20", "2009-03-19", "2009-04-17"],
        })
        monkeypatch.setattr(sc, "load_release_date_mapping", lambda: (
            release_df.assign(
                reference_month=pd.to_datetime(release_df["reference_month"]),
                release_date=pd.to_datetime(release_df["release_date"]),
            )
        ))

        result = sc.run(
            use_cache=True,
            cache_file=cache_file,
            start_date="2009-01-01",
            end_date="2009-03-31",
        )

        assert "2009-01-01" in result["reference_month"].dt.strftime("%Y-%m-%d").tolist()
        assert len(result) == 3
