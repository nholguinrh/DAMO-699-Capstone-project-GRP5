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


def _cpi_df(
    months: list[str], values: list[float], source: str = "test_fixture"
) -> pd.DataFrame:
    return pd.DataFrame({
        "reference_month": pd.to_datetime(months),
        "cpi_all_items": values,
        "source_pull": source,
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
            # 113.8 -> 113.9 is a plausible small revision (within
            # MAX_ARCHIVE_REVISION), unlike a rebasing-scale jump.
            ["2009-02-01", "2009-03-01", "2009-04-01"], [113.9, 114.0, 113.9]
        )))
        # Ordering must come from the filename's embedded timestamp, not
        # mtime -- set mtimes backwards (newer file "older" on disk) to
        # prove a copy/checkout that resets mtimes can't silently reorder
        # pulls and let the older pull's values win on overlap.
        import os
        os.utime(newer, (1_000_000_000, 1_000_000_000))
        os.utime(older, (2_000_000_000, 2_000_000_000))

        rebuilt = sc.rebuild_archive_from_raw_cache()

        assert list(rebuilt["reference_month"].dt.strftime("%Y-%m-%d")) == [
            "2009-01-01", "2009-02-01", "2009-03-01", "2009-04-01",
        ]
        # The newer pull's (filename-timestamp-wise) value must win on
        # overlap, despite its mtime being set older above.
        overlap = rebuilt.set_index("reference_month")["cpi_all_items"]
        assert overlap.loc[pd.Timestamp("2009-02-01")] == pytest.approx(113.9)
        # Persisted, not just returned
        assert sc.ARCHIVE_FILE.exists()

    def test_raises_when_no_cached_files(self, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)

        with pytest.raises(FileNotFoundError):
            sc.rebuild_archive_from_raw_cache()

    def test_ignores_a_pre_existing_archive_by_default(self, tmp_path, monkeypatch):
        """B4: this is the recovery/audit path, so it must derive from the
        raw pulls alone by default -- seeding from the existing archive
        would let an unattributed or corrupted row (one no raw pull
        covers) silently survive the exact operation meant to catch it."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)
        monkeypatch.setattr(sc, "ARCHIVE_FILE", tmp_path / "archive.csv")

        # A month no raw pull covers -- e.g. a corrupted or unattributed row.
        sc.save_archive(_cpi_df(["2009-01-01", "2009-02-01"], [113.0, 113.8]))
        (raw_dir / f"cpi_{sc.VECTOR_ID}_20260101_000000.json").write_text(
            json.dumps(_raw_json(["2009-01-01"], [113.0]))
        )

        rebuilt = sc.rebuild_archive_from_raw_cache()

        assert len(rebuilt) == 1, "the unattributed row must not survive a rebuild"

    def test_merge_into_existing_unions_onto_the_current_archive(
        self, tmp_path, monkeypatch
    ):
        """With merge_into_existing=True, the archive's own rows are kept
        alongside whatever the raw pulls add."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)
        monkeypatch.setattr(sc, "ARCHIVE_FILE", tmp_path / "archive.csv")

        sc.save_archive(_cpi_df(["2009-01-01"], [113.0]))
        (raw_dir / f"cpi_{sc.VECTOR_ID}_20260101_000000.json").write_text(
            json.dumps(_raw_json(["2009-02-01"], [113.8]))
        )

        rebuilt = sc.rebuild_archive_from_raw_cache(merge_into_existing=True)

        assert len(rebuilt) == 2

    def test_attaches_source_pull_provenance(self, tmp_path, monkeypatch):
        """B5: every archived row must be traceable to the raw pull it
        came from."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)
        monkeypatch.setattr(sc, "ARCHIVE_FILE", tmp_path / "archive.csv")
        filename = f"cpi_{sc.VECTOR_ID}_20260101_000000.json"
        (raw_dir / filename).write_text(json.dumps(_raw_json(["2009-01-01"], [113.0])))

        rebuilt = sc.rebuild_archive_from_raw_cache()

        assert rebuilt.iloc[0]["source_pull"] == filename


class TestMergeIntoArchiveRevisionGuard:
    """Issue #109 review, B3: an overlapping value may only change by a
    small (routine-revision-sized) amount before merge_into_archive()
    refuses to apply it."""

    def test_rebasing_scale_revision_is_rejected(self):
        archive = _cpi_df(["2009-01-01"], [113.0])
        rebased = _cpi_df(["2009-01-01"], [999.9])

        with pytest.raises(ValueError, match="rebasing or a bad pull"):
            sc.merge_into_archive(archive, rebased)

    def test_small_revision_is_accepted(self):
        archive = _cpi_df(["2009-01-01"], [113.0])
        revised = _cpi_df(["2009-01-01"], [113.2])

        merged = sc.merge_into_archive(archive, revised)

        assert merged.iloc[0]["cpi_all_items"] == pytest.approx(113.2)

    def test_null_value_never_overwrites_a_good_archived_value(self):
        archive = _cpi_df(["2009-01-01"], [113.0])
        pull_with_null = pd.DataFrame({
            "reference_month": pd.to_datetime(["2009-01-01"]),
            "cpi_all_items": [None],
            "source_pull": ["test_fixture"],
        })

        merged = sc.merge_into_archive(archive, pull_with_null)

        assert merged.iloc[0]["cpi_all_items"] == pytest.approx(113.0)


class TestValidateArchive:
    """Issue #109 review, B1: the full archive is validated before every
    persist, not just the window a given run happens to request."""

    def test_rejects_empty_archive(self):
        empty = pd.DataFrame(columns=sc.ARCHIVE_COLUMNS)
        with pytest.raises(ValueError, match="empty"):
            sc.validate_archive(empty)

    def test_rejects_null_cpi_value(self):
        bad = pd.DataFrame({
            "reference_month": pd.to_datetime(["2009-01-01"]),
            "cpi_all_items": [None],
            "source_pull": ["x"],
        })
        with pytest.raises(ValueError, match="null value"):
            sc.validate_archive(bad)

    def test_rejects_duplicate_reference_months(self):
        dupe = pd.DataFrame({
            "reference_month": pd.to_datetime(["2009-01-01", "2009-01-01"]),
            "cpi_all_items": [113.0, 113.1],
            "source_pull": ["x", "y"],
        })
        with pytest.raises(ValueError, match="duplicate"):
            sc.validate_archive(dupe)

    def test_rejects_a_gap_in_monthly_coverage(self):
        gapped = pd.DataFrame({
            "reference_month": pd.to_datetime(["2009-01-01", "2009-03-01"]),
            "cpi_all_items": [113.0, 114.0],
            "source_pull": ["x", "x"],
        })
        with pytest.raises(ValueError, match="gaps"):
            sc.validate_archive(gapped)

    def test_accepts_a_clean_continuous_archive(self):
        clean = _cpi_df(["2009-01-01", "2009-02-01"], [113.0, 113.8])
        sc.validate_archive(clean)  # must not raise

    def test_out_of_window_null_cannot_reach_the_archive(self, tmp_path, monkeypatch):
        """B1: the original defect was that run() validated a *clamped*
        view (validate_dataframe on cpi_df) but persisted the *unclamped*
        archive_df, so a null value in a reference month outside the
        requested window bypassed every check entirely. Two layers now
        prevent it: merge_into_archive() drops null values from the
        incoming pull before they can overwrite anything, and
        validate_archive() would still catch one on the full archive even
        if it got that far. The net effect: the archive's original good
        value survives untouched, not silently replaced by a null."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        archive_file = tmp_path / "archive.csv"
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)
        monkeypatch.setattr(sc, "ARCHIVE_FILE", archive_file)

        sc.save_archive(_cpi_df(["2009-01-01", "2009-02-01"], [113.0, 113.8]))

        live_file = raw_dir / f"cpi_{sc.VECTOR_ID}_20260901_000000.json"
        live_file.write_text(json.dumps(
            _raw_json(["2009-01-01", "2009-02-01"], [None, 113.8])
        ))
        monkeypatch.setattr(sc, "download_raw_data", lambda **kw: live_file)

        release_df = pd.DataFrame({
            "reference_month": ["2009-01-01", "2009-02-01"],
            "release_date": ["2009-02-20", "2009-03-19"],
        })
        monkeypatch.setattr(sc, "load_release_date_mapping", lambda: (
            release_df.assign(
                reference_month=pd.to_datetime(release_df["reference_month"]),
                release_date=pd.to_datetime(release_df["release_date"]),
            )
        ))

        # 2009-01 is OUTSIDE the requested window, so validate_dataframe()
        # never sees it at all.
        sc.run(use_cache=False, start_date="2009-02-01", end_date="2009-02-28")

        assert sc.load_archive().set_index("reference_month") \
            .loc[pd.Timestamp("2009-01-01"), "cpi_all_items"] == pytest.approx(113.0)


class TestSaveArchiveAtomicity:
    def test_rejected_write_leaves_previous_archive_intact(self, tmp_path, monkeypatch):
        """N2: a write that fails validation must not corrupt/truncate the
        existing file -- save_archive validates before it ever opens the
        real path for writing."""
        monkeypatch.setattr(sc, "ARCHIVE_FILE", tmp_path / "archive.csv")
        sc.save_archive(_cpi_df(["2009-01-01"], [113.0]))

        bad = pd.DataFrame({
            "reference_month": pd.to_datetime(["2009-01-01", "2009-01-01"]),
            "cpi_all_items": [113.0, 114.0],
            "source_pull": ["x", "y"],
        })
        with pytest.raises(ValueError):
            sc.save_archive(bad)

        assert sc.load_archive().iloc[0]["cpi_all_items"] == pytest.approx(113.0)

    def test_no_leftover_temp_file_after_a_successful_save(self, tmp_path, monkeypatch):
        archive_file = tmp_path / "archive.csv"
        monkeypatch.setattr(sc, "ARCHIVE_FILE", archive_file)
        sc.save_archive(_cpi_df(["2009-01-01"], [113.0]))

        assert not archive_file.with_suffix(".csv.tmp").exists()


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

    def test_explicit_cache_file_does_not_persist_to_shared_archive(
        self, tmp_path, monkeypatch
    ):
        """--cache-file points at one specific historical snapshot for manual
        testing/debugging, not a genuine new pull -- it must not silently
        overwrite the shared, git-tracked archive with whatever that old
        file happens to contain."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        archive_file = tmp_path / "archive.csv"
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)
        monkeypatch.setattr(sc, "ARCHIVE_FILE", archive_file)
        monkeypatch.setattr(sc, "PROCESSED_FILE", tmp_path / "statcan_cpi.csv")

        sc.save_archive(_cpi_df(["2009-01-01"], [113.0]))

        cache_file = raw_dir / f"cpi_{sc.VECTOR_ID}_20260201_000000.json"
        # A small, plausible revision -- large enough to prove the archive
        # wasn't overwritten, well within MAX_ARCHIVE_REVISION.
        cache_file.write_text(json.dumps(_raw_json(["2009-01-01"], [113.5])))

        release_df = pd.DataFrame({
            "reference_month": ["2009-01-01"],
            "release_date": ["2009-02-20"],
        })
        monkeypatch.setattr(sc, "load_release_date_mapping", lambda: (
            release_df.assign(
                reference_month=pd.to_datetime(release_df["reference_month"]),
                release_date=pd.to_datetime(release_df["release_date"]),
            )
        ))

        sc.run(
            use_cache=True,
            cache_file=cache_file,
            start_date="2009-01-01",
            end_date="2009-01-31",
        )

        untouched = sc.load_archive()
        assert untouched.iloc[0]["cpi_all_items"] == pytest.approx(113.0), \
            "The stale --cache-file pull's value must not have overwritten the archive"

    def test_archive_not_saved_when_validation_fails_downstream(
        self, tmp_path, monkeypatch
    ):
        """A run that fails after the archive merge (e.g. cpi_release_dates.csv
        doesn't cover a newly-pulled month yet) must not leave the shared
        archive file created/mutated -- only a fully-validated run persists.

        Uses use_cache=False (a live pull) so persistence is gated only by
        validation succeeding, not by B2's separate cache-mode-never-persists
        rule -- otherwise this would pass for the wrong reason regardless of
        whether the validate-before-save ordering actually holds."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        archive_file = tmp_path / "archive.csv"
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)
        monkeypatch.setattr(sc, "ARCHIVE_FILE", archive_file)

        live_file = raw_dir / f"cpi_{sc.VECTOR_ID}_20260201_000000.json"
        live_file.write_text(json.dumps(_raw_json(["2009-01-01"], [113.0])))
        monkeypatch.setattr(sc, "download_raw_data", lambda **kw: live_file)

        # No release-date mapping at all for the pulled month -> validation
        # must raise inside align_to_release_dates/validate_dataframe.
        monkeypatch.setattr(sc, "load_release_date_mapping", lambda: pd.DataFrame(
            {"reference_month": pd.Series(dtype="datetime64[ns]"),
             "release_date": pd.Series(dtype="datetime64[ns]")}
        ))

        with pytest.raises(ValueError):
            sc.run(
                use_cache=False,
                start_date="2009-01-01",
                end_date="2009-01-31",
            )

        assert not archive_file.exists(), \
            "A failed run must not create/persist the shared archive file"

    def test_cache_rebuild_with_no_explicit_file_never_persists(
        self, tmp_path, monkeypatch
    ):
        """B2: the dashboard's mode="cache" path calls run(use_cache=True,
        cache_file=None), resolved via find_latest_cached_json() -- the
        exact path that previously had zero coverage (N1). A stale local
        pull must not revert a revision the archive already holds."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        archive_file = tmp_path / "archive.csv"
        monkeypatch.setattr(sc, "STATCAN_RAW_DIR", raw_dir)
        monkeypatch.setattr(sc, "ARCHIVE_FILE", archive_file)
        monkeypatch.setattr(sc, "PROCESSED_FILE", tmp_path / "statcan_cpi.csv")

        sc.save_archive(_cpi_df(["2009-01-01"], [114.5]))  # revised value

        # Delta kept within MAX_ARCHIVE_REVISION so this test isolates B2
        # (cache mode never persists) from B3 (large-revision guard) --
        # a plausible small reversion must still not be written back.
        (raw_dir / f"cpi_{sc.VECTOR_ID}_20200101_000000.json").write_text(
            json.dumps(_raw_json(["2009-01-01"], [114.0]))  # stale local pull
        )

        release_df = pd.DataFrame({
            "reference_month": ["2009-01-01"],
            "release_date": ["2009-02-20"],
        })
        monkeypatch.setattr(sc, "load_release_date_mapping", lambda: (
            release_df.assign(
                reference_month=pd.to_datetime(release_df["reference_month"]),
                release_date=pd.to_datetime(release_df["release_date"]),
            )
        ))

        sc.run(
            use_cache=True,
            cache_file=None,
            start_date="2009-01-01",
            end_date="2009-01-31",
        )

        assert sc.load_archive().iloc[0]["cpi_all_items"] == pytest.approx(114.5), \
            "A cache-mode rebuild with no explicit file must not overwrite the archive"
