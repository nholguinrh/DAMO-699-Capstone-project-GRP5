import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from http_utils import get_with_retry
from pipeline_dates import CPI_REFERENCE_START, clamp_to_range, resolve_date_range
from project_paths import PROJECT_ROOT, PROCESSED_DIR, RAW_DIR


# ---------------------------------------------------------
# 1. Folder and file configuration
# ---------------------------------------------------------

STATCAN_RAW_DIR = RAW_DIR / "statcan"
CONFIG_DIR = PROJECT_ROOT / "config"

STATCAN_RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_FILE = PROCESSED_DIR / "statcan_cpi.csv"

RELEASE_DATE_FILE = (
    CONFIG_DIR
    / "cpi_release_dates.csv"
)

# Issue #109: the StatCan endpoint used below (getDataFromVectorByReferencePeriodRange)
# ignores endReferencePeriod and always returns a fixed ~210-observation rolling
# window anchored to the latest published month, regardless of what start/end
# dates are requested. As new months publish, the window slides forward and the
# oldest reference months silently fall out of every fresh pull. ARCHIVE_FILE is
# a git-tracked, append-only union of every reference_month ever observed across
# all pulls (seeded from the raw JSON that built the canonical dataset), so the
# ingestion window is no longer limited by what any single pull happens to cover.
ARCHIVE_FILE = CONFIG_DIR / "statcan_cpi_archive.csv"
# source_pull records the raw JSON filename each row's value came from, so
# every archived observation is independently traceable to a committed
# input (Issue #109 review, B5) rather than being an unattributed number.
ARCHIVE_COLUMNS = ["reference_month", "cpi_all_items", "source_pull"]

# Largest change in a single archived CPI value that a revision may make
# without operator sign-off (index points). StatCan revisions to already-
# published values are typically fractions of an index point; anything past
# this is far more likely a CPI basket rebasing (which shifts the *entire*
# index level and would splice two incompatible bases into one series with
# no downstream way to detect it), a bad pull, or a corrupted archive --
# never a routine revision. See merge_into_archive().
MAX_ARCHIVE_REVISION = 1.0

# Reference months StatCan's own historical archive currently cannot
# answer (both source catalog pages 500-error / timeout as of Aug 2026):
#   https://www150.statcan.gc.ca/n1/en/catalogue/62-001-X2010004
#   https://www150.statcan.gc.ca/n1/en/catalogue/62-001-X2010009
# See build_cpi_release_dates.py's MANUAL_OVERRIDES comments for the
# full investigation. This is a documented, accepted gap -- not a
# pipeline bug -- so these two months are allowed through with a null
# release_date rather than blocking the whole run. Any OTHER missing
# month still hard-fails in align_to_release_dates() below.
KNOWN_UNMAPPED_CPI_MONTHS = {
    pd.Timestamp("2010-04-01"),
    pd.Timestamp("2010-09-01"),
}


# ---------------------------------------------------------
# 2. Statistics Canada API configuration
# ---------------------------------------------------------

BASE_URL = (
    "https://www150.statcan.gc.ca/"
    "t1/wds/rest/"
    "getDataFromVectorByReferencePeriodRange"
)

VECTOR_ID = "41690973"


# ---------------------------------------------------------
# 3. Download raw CPI data
# ---------------------------------------------------------

def download_raw_data(
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    """
    Download Statistics Canada CPI observations using the
    Web Data Service API.

    The HTTP request uses synchronous retry logic supplied by
    get_with_retry().

    Parameters
    ----------
    start_date, end_date:
        Optional ISO (YYYY-MM-DD) overrides for the reference-period
        window. Default to ``config.CPI_REFERENCE_START`` / ``config.DATE_END``.

    Returns
    -------
    Path
        Location of the saved raw JSON file.
    """

    start, end = resolve_date_range(
        start_date, end_date, default_start=CPI_REFERENCE_START
    )

    params = {
        "vectorIds": VECTOR_ID,
        "startRefPeriod": start,
        "endReferencePeriod": end,
    }

    print(
        "Downloading Statistics Canada CPI data..."
    )

    response = get_with_retry(
        BASE_URL,
        params=params,
        timeout=60,
        max_retries=5,
    )

    print(
        "Statistics Canada status code:",
        response.status_code,
    )

    data = response.json()

    validate_api_response(data)

    observations = extract_observations(data)

    print(
        "Number of CPI observations:",
        len(observations),
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    raw_file = (
        STATCAN_RAW_DIR
        / f"cpi_{VECTOR_ID}_{timestamp}.json"
    )

    with raw_file.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )

    print("\nRaw Statistics Canada JSON saved to:")
    print(raw_file)

    return raw_file


# ---------------------------------------------------------
# 4. Validate the API response
# ---------------------------------------------------------

def validate_api_response(
    data: object,
) -> None:
    """
    Validate the structure and status of a Statistics Canada
    Web Data Service response.
    """

    if not isinstance(data, list):
        raise ValueError(
            "Statistics Canada response must be a list."
        )

    if not data:
        raise ValueError(
            "Statistics Canada returned an empty response."
        )

    result = data[0]

    if not isinstance(result, dict):
        raise ValueError(
            "Statistics Canada response item is not a dictionary."
        )

    status = result.get("status")

    if status != "SUCCESS":
        raise ValueError(
            "Statistics Canada request failed. "
            f"Response status: {status}. "
            f"Response: {result}"
        )

    response_object = result.get("object")

    if not isinstance(response_object, dict):
        raise ValueError(
            "Statistics Canada response does not contain "
            "a valid 'object' section."
        )

    observations = response_object.get(
        "vectorDataPoint"
    )

    if observations is None:
        raise ValueError(
            "Statistics Canada response does not contain "
            "'vectorDataPoint'."
        )

    if not observations:
        raise ValueError(
            "Statistics Canada returned no CPI observations."
        )


# ---------------------------------------------------------
# 5. Extract observations
# ---------------------------------------------------------

def extract_observations(
    data: list,
) -> list:
    """
    Extract CPI vector observations from the validated response.
    """

    return data[0]["object"]["vectorDataPoint"]


# ---------------------------------------------------------
# 6. Find cached raw JSON
# ---------------------------------------------------------

def _sorted_cached_raw_files() -> list[Path]:
    """
    Every cached raw JSON pull, oldest to newest.

    Sorts by the sortable ``%Y%m%d_%H%M%S`` timestamp embedded in each
    filename (``cpi_{VECTOR_ID}_<timestamp>.json``) rather than filesystem
    mtime -- mtimes reset whenever these files are copied, zipped, or
    checked out (they're gitignored, so that's the normal way to move them
    between machines), which would silently reorder pulls and let an older
    pull's values win over a newer one's in ``merge_into_archive``.
    """
    return sorted(STATCAN_RAW_DIR.glob(f"cpi_{VECTOR_ID}_*.json"), key=lambda p: p.name)


def find_latest_cached_json() -> Path:
    """
    Find the most recently pulled Statistics Canada CPI raw JSON file.
    """

    cached_files = _sorted_cached_raw_files()

    if not cached_files:
        raise FileNotFoundError(
            "No cached Statistics Canada CPI JSON "
            f"was found in {STATCAN_RAW_DIR}"
        )

    latest_file = cached_files[-1]

    print(
        "Using cached Statistics Canada JSON:"
    )
    print(latest_file)

    return latest_file


# ---------------------------------------------------------
# 7. Load cached raw JSON
# ---------------------------------------------------------

def load_raw_json(
    raw_file: Path,
) -> list:
    """
    Load and validate a cached Statistics Canada JSON file.
    """

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw JSON file does not exist: {raw_file}"
        )

    with raw_file.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    validate_api_response(data)

    return data


# ---------------------------------------------------------
# 8. Convert observations to a DataFrame
# ---------------------------------------------------------

def create_cpi_dataframe(
    data: list,
) -> pd.DataFrame:
    """
    Convert CPI vector observations into a DataFrame.

    The source reference period is retained as
    reference_month. It is not treated as the date on which
    the CPI value became publicly available.
    """

    observations = extract_observations(data)

    cpi_df = pd.DataFrame(observations)

    required_raw_columns = {
        "refPer",
        "value",
    }

    missing_raw_columns = (
        required_raw_columns
        - set(cpi_df.columns)
    )

    if missing_raw_columns:
        raise ValueError(
            "Statistics Canada raw data is missing columns: "
            f"{sorted(missing_raw_columns)}"
        )

    cpi_df = cpi_df[
        [
            "refPer",
            "value",
        ]
    ].copy()

    cpi_df = cpi_df.rename(
        columns={
            "refPer": "reference_month",
            "value": "cpi_all_items",
        }
    )

    cpi_df["reference_month"] = pd.to_datetime(
        cpi_df["reference_month"],
        errors="coerce",
    )

    cpi_df["cpi_all_items"] = pd.to_numeric(
        cpi_df["cpi_all_items"],
        errors="coerce",
    )

    cpi_df = (
        cpi_df
        .dropna(subset=["reference_month"])
        .drop_duplicates(
            subset=["reference_month"]
        )
        .sort_values("reference_month")
        .reset_index(drop=True)
    )

    return cpi_df


# ---------------------------------------------------------
# 8b. Append-only CPI observation archive (Issue #109)
# ---------------------------------------------------------

def load_archive() -> pd.DataFrame:
    """
    Load the persisted, append-only CPI observation archive.

    Returns an empty frame with the correct columns/dtypes if the archive
    hasn't been created yet (e.g. before the first run on a fresh clone
    that predates this fix, or after ``ARCHIVE_FILE`` is deleted).
    """
    if not ARCHIVE_FILE.exists():
        return pd.DataFrame(
            {
                "reference_month": pd.Series(dtype="datetime64[ns]"),
                "cpi_all_items": pd.Series(dtype="float64"),
                "source_pull": pd.Series(dtype="object"),
            }
        )

    archive_df = pd.read_csv(ARCHIVE_FILE)
    archive_df["reference_month"] = pd.to_datetime(archive_df["reference_month"])
    archive_df["cpi_all_items"] = pd.to_numeric(archive_df["cpi_all_items"])
    if "source_pull" not in archive_df.columns:
        archive_df["source_pull"] = "unknown (pre-provenance archive)"
    return archive_df[ARCHIVE_COLUMNS]


def merge_into_archive(
    archive_df: pd.DataFrame,
    new_df: pd.DataFrame,
    max_revision: float = MAX_ARCHIVE_REVISION,
) -> pd.DataFrame:
    """
    Union a freshly-pulled CPI dataframe into the persisted archive.

    Every pull only covers a rolling ~210-month window (Issue #109), so any
    single pull can be missing reference months an earlier pull already saw.
    Keeping the newest observation per ``reference_month`` (``new_df``'s,
    since it's appended last) preserves history the current pull doesn't
    cover while still picking up a genuine StatCan revision.

    A revision is only accepted if it is small (Issue #109 review, B3). A
    pull that moves an archived value by more than ``max_revision`` index
    points is far more likely a CPI basket rebasing (which shifts the
    *entire* index level), a bad pull, or a corrupted archive than a
    routine revision -- and rewriting only the ~210 months the rolling
    window covers would splice two incompatible index bases into one
    series with no downstream way to detect it. Raises rather than
    silently accepting it. Null values in ``new_df`` never overwrite a
    good archived value.
    """
    new_df = new_df[ARCHIVE_COLUMNS].dropna(subset=["cpi_all_items"])

    overlap = archive_df.merge(
        new_df, on="reference_month", how="inner", suffixes=("_old", "_new")
    )
    changed = overlap.loc[
        (overlap["cpi_all_items_old"] - overlap["cpi_all_items_new"]).abs() > 1e-9
    ]
    if not changed.empty:
        deltas = (changed["cpi_all_items_new"] - changed["cpi_all_items_old"]).abs()
        for _, row in changed.iterrows():
            print(
                f"REVISION: {row['reference_month'].date()} "
                f"{row['cpi_all_items_old']} -> {row['cpi_all_items_new']}"
            )
        if (deltas > max_revision).any():
            worst = changed.loc[deltas.idxmax()]
            raise ValueError(
                f"{int((deltas > max_revision).sum())} archived CPI value(s) "
                f"would change by more than {max_revision} index point(s) -- "
                f"largest: {worst['reference_month'].date()} "
                f"{worst['cpi_all_items_old']} -> {worst['cpi_all_items_new']}. "
                "This looks like a rebasing or a bad pull, not a routine "
                "revision. Reconcile manually before letting it rewrite "
                "committed history."
            )

    combined = pd.concat(
        [archive_df[ARCHIVE_COLUMNS], new_df[ARCHIVE_COLUMNS]],
        ignore_index=True,
    )
    return (
        combined
        .drop_duplicates(subset=["reference_month"], keep="last")
        .sort_values("reference_month")
        .reset_index(drop=True)
    )


def validate_archive(archive_df: pd.DataFrame) -> None:
    """
    Validate the FULL archive before it is persisted (Issue #109 review, B1).

    ``run()`` clamps to the requested window before validating against the
    release-date mapping and the other checks in ``validate_dataframe()`` --
    so without this, a reference month outside that window would reach the
    git-tracked archive with no checks at all. The archive is the upstream
    source of truth for the whole Gold frame; every row in it must be
    sound, not just the ones the current run's window happens to cover.
    """
    if archive_df.empty:
        raise ValueError("Refusing to persist an empty CPI archive.")

    if archive_df["reference_month"].isna().any():
        raise ValueError("CPI archive contains invalid reference months.")

    if archive_df["reference_month"].duplicated().any():
        dupes = archive_df.loc[
            archive_df["reference_month"].duplicated(keep=False), "reference_month"
        ]
        raise ValueError(
            "CPI archive contains duplicate reference months: "
            f"{dupes.dt.strftime('%Y-%m-%d').tolist()}"
        )

    missing = archive_df["cpi_all_items"].isna()
    if missing.any():
        raise ValueError(
            "CPI archive contains null value(s) for reference month(s): "
            f"{archive_df.loc[missing, 'reference_month'].dt.strftime('%Y-%m-%d').tolist()}"
        )

    if not archive_df["reference_month"].is_monotonic_increasing:
        raise ValueError("CPI archive reference months are not sorted ascending.")

    expected = pd.date_range(
        archive_df["reference_month"].min(),
        archive_df["reference_month"].max(),
        freq="MS",
    )
    gaps = sorted(set(expected) - set(archive_df["reference_month"]))
    if gaps:
        raise ValueError(
            "CPI archive has gaps in its monthly coverage: "
            f"{[g.strftime('%Y-%m-%d') for g in gaps]}"
        )


def save_archive(archive_df: pd.DataFrame) -> Path:
    """
    Persist the archive to ``ARCHIVE_FILE`` (git-tracked, unlike ``data/``).

    Validated first (Issue #109 review, B1) and written atomically via a
    sibling temp file + ``os.replace`` (B1/N2) -- an interrupted write must
    not truncate the only copy of the CPI history.

    Not safe against two concurrent runs racing this read-modify-write --
    each would merge into the same starting snapshot and whichever save
    lands last would win, silently dropping the other's observations. This
    is a real if uncommon scenario (the dashboard's "Rebuild data &
    features" button can trigger this path); take a lock here if it bites.
    """
    validate_archive(archive_df)

    temp_file = ARCHIVE_FILE.with_suffix(".csv.tmp")
    archive_df[ARCHIVE_COLUMNS].to_csv(temp_file, index=False)
    os.replace(temp_file, ARCHIVE_FILE)
    return ARCHIVE_FILE


def rebuild_archive_from_raw_cache(merge_into_existing: bool = False) -> pd.DataFrame:
    """
    Rebuild the archive from every cached raw JSON pull in ``STATCAN_RAW_DIR``,
    oldest pull first so the newest pull's value wins any reference-month
    overlap.

    Derived from the raw pulls ALONE by default (Issue #109 review, B4):
    this is the recovery/audit path, so seeding from the existing archive
    would let it silently pass through any corruption a raw pull doesn't
    happen to overwrite -- exactly the case this flag exists to catch --
    and would make the result depend on prior filesystem state rather than
    on the declared inputs. Pass ``merge_into_existing=True`` to union onto
    the current archive instead.
    """
    cached_files = _sorted_cached_raw_files()

    if not cached_files:
        raise FileNotFoundError(
            f"No cached Statistics Canada CPI JSON found in {STATCAN_RAW_DIR}"
        )

    if merge_into_existing:
        archive_df = load_archive()
    else:
        archive_df = pd.DataFrame(
            {
                "reference_month": pd.Series(dtype="datetime64[ns]"),
                "cpi_all_items": pd.Series(dtype="float64"),
                "source_pull": pd.Series(dtype="object"),
            }
        )

    for raw_file in cached_files:
        pull_df = create_cpi_dataframe(load_raw_json(raw_file))
        pull_df["source_pull"] = raw_file.name
        archive_df = merge_into_archive(archive_df, pull_df)

    save_archive(archive_df)
    return archive_df


# ---------------------------------------------------------
# 9. Load official CPI release-date mapping
# ---------------------------------------------------------

def load_release_date_mapping() -> pd.DataFrame:
    """
    Load the CPI publication-date mapping.

    Required file:
        config/cpi_release_dates.csv

    Required columns:
        reference_month
        release_date

    The release date must be based on an official Statistics
    Canada CPI release calendar or release publication.

    The script intentionally does not invent publication dates
    by adding a fixed number of days to the reference month.
    """

    if not RELEASE_DATE_FILE.exists():
        raise FileNotFoundError(
            "CPI release-date mapping was not found.\n"
            f"Expected file: {RELEASE_DATE_FILE}\n\n"
            "Create config/cpi_release_dates.csv with these "
            "columns:\n"
            "reference_month,release_date\n"
            "2009-01-01,YYYY-MM-DD\n"
            "2009-02-01,YYYY-MM-DD\n"
            "...\n\n"
            "Use official Statistics Canada CPI publication "
            "dates. Do not use an assumed fixed monthly offset."
        )

    release_df = pd.read_csv(
        RELEASE_DATE_FILE,
        dtype=str,
    )

    required_columns = {
        "reference_month",
        "release_date",
    }

    missing_columns = (
        required_columns
        - set(release_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "CPI release-date mapping is missing columns: "
            f"{sorted(missing_columns)}"
        )

    release_df = release_df[
        [
            "reference_month",
            "release_date",
        ]
    ].copy()

    release_df["reference_month"] = (
        pd.to_datetime(
            release_df["reference_month"],
            errors="coerce",
        )
    )

    release_df["release_date"] = pd.to_datetime(
        release_df["release_date"],
        errors="coerce",
    )

    invalid_reference_dates = (
        release_df["reference_month"].isna().sum()
    )

    invalid_release_dates = (
        release_df["release_date"].isna().sum()
    )

    if invalid_reference_dates:
        raise ValueError(
            "The release-date mapping contains "
            f"{invalid_reference_dates} invalid "
            "reference_month value(s)."
        )

    if invalid_release_dates:
        raise ValueError(
            "The release-date mapping contains "
            f"{invalid_release_dates} invalid "
            "release_date value(s)."
        )

    if release_df[
        "reference_month"
    ].duplicated().any():
        duplicate_months = release_df.loc[
            release_df[
                "reference_month"
            ].duplicated(
                keep=False
            ),
            "reference_month",
        ]

        raise ValueError(
            "Duplicate reference months were found in the "
            "CPI release-date mapping: "
            f"{duplicate_months.dt.strftime('%Y-%m-%d').tolist()}"
        )

    release_df = (
        release_df
        .sort_values("reference_month")
        .reset_index(drop=True)
    )

    return release_df


# ---------------------------------------------------------
# 10. Align CPI observations to release dates
# ---------------------------------------------------------

def align_to_release_dates(
    cpi_df: pd.DataFrame,
    release_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge CPI reference periods with official publication dates.

    The release_date field represents the first date on which the
    CPI observation may be used by downstream models.
    """

    aligned_df = cpi_df.merge(
        release_df,
        on="reference_month",
        how="left",
        validate="one_to_one",
    )

    missing_release_dates = (
        aligned_df["release_date"].isna()
    )

    missing_count = (
        missing_release_dates.sum()
    )

    if missing_count:
        missing_months_ts = aligned_df.loc[
            missing_release_dates,
            "reference_month",
        ]

        unexpected = missing_months_ts[
            ~missing_months_ts.isin(KNOWN_UNMAPPED_CPI_MONTHS)
        ]

        if not unexpected.empty:
            raise ValueError(
                f"{len(unexpected)} CPI observation(s) do not have an "
                "official release-date mapping, and are not in the "
                "documented KNOWN_UNMAPPED_CPI_MONTHS allowlist. "
                f"Missing reference months: "
                f"{unexpected.dt.strftime('%Y-%m-%d').tolist()}"
            )

        known = missing_months_ts[
            missing_months_ts.isin(KNOWN_UNMAPPED_CPI_MONTHS)
        ]
        print(
            f"NOTE: {len(known)} CPI observation(s) have no release-date "
            "mapping, but are in the documented KNOWN_UNMAPPED_CPI_MONTHS "
            "allowlist (StatCan's own archive currently can't answer "
            f"these): {known.dt.strftime('%Y-%m-%d').tolist()}. "
            "release_date will be left null for these rows; downstream "
            "cleaning/feature-engineering steps must handle this "
            "explicitly rather than assume every row has a release date."
        )

    invalid_timing = (
        aligned_df["release_date"]
        <= aligned_df["reference_month"]
    )

    if invalid_timing.any():
        invalid_rows = aligned_df.loc[
            invalid_timing,
            [
                "reference_month",
                "release_date",
            ],
        ]

        raise ValueError(
            "One or more CPI release dates are not later than "
            "their reference months:\n"
            f"{invalid_rows}"
        )

    aligned_df = aligned_df[
        [
            "reference_month",
            "release_date",
            "cpi_all_items",
        ]
    ].copy()

    # NOTE: any row in KNOWN_UNMAPPED_CPI_MONTHS has release_date = NaT
    # and will sort to the END of this dataframe (pandas' default
    # na_position="last"), out of chronological order relative to its
    # reference_month. If downstream code assumes release_date-sorted
    # rows are also reference_month-ordered, handle these rows
    # explicitly rather than relying on this sort.
    aligned_df = (
        aligned_df
        .sort_values("release_date")
        .reset_index(drop=True)
    )

    return aligned_df


# ---------------------------------------------------------
# 11. Validate processed CPI data
# ---------------------------------------------------------

def validate_dataframe(
    cpi_df: pd.DataFrame,
) -> None:
    """
    Validate the final CPI dataset.
    """

    required_columns = {
        "reference_month",
        "release_date",
        "cpi_all_items",
    }

    missing_columns = (
        required_columns
        - set(cpi_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Processed CPI dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if cpi_df.empty:
        raise ValueError(
            "The processed CPI dataset is empty."
        )

    if cpi_df[
        "reference_month"
    ].isna().any():
        raise ValueError(
            "The processed CPI dataset contains invalid "
            "reference months."
        )

    missing_release_dates = cpi_df["release_date"].isna()

    if missing_release_dates.any():
        unmapped_months = cpi_df.loc[
            missing_release_dates,
            "reference_month",
        ]

        unexpected = unmapped_months[
            ~unmapped_months.isin(KNOWN_UNMAPPED_CPI_MONTHS)
        ]

        if not unexpected.empty:
            raise ValueError(
                "The processed CPI dataset contains missing release "
                "dates that are not in the documented "
                "KNOWN_UNMAPPED_CPI_MONTHS allowlist: "
                f"{unexpected.dt.strftime('%Y-%m-%d').tolist()}"
            )

        print(
            f"NOTE (validate_dataframe): {len(unmapped_months)} row(s) "
            "with null release_date passed validation because they are "
            "in the documented KNOWN_UNMAPPED_CPI_MONTHS allowlist: "
            f"{unmapped_months.dt.strftime('%Y-%m-%d').tolist()}"
        )

    if cpi_df[
        "cpi_all_items"
    ].isna().any():
        missing_values = (
            cpi_df["cpi_all_items"].isna().sum()
        )

        raise ValueError(
            "The processed CPI dataset contains "
            f"{missing_values} missing CPI value(s)."
        )

    if cpi_df[
        "reference_month"
    ].duplicated().any():
        raise ValueError(
            "Duplicate CPI reference months remain in "
            "the processed dataset."
        )

    if cpi_df[
        "release_date"
    ].duplicated().any():
        # NOTE: pandas treats multiple NaT values as "duplicates" of each
        # other, so if both KNOWN_UNMAPPED_CPI_MONTHS rows are present
        # this will always fire. That's expected and not a real
        # duplicate-release-date problem -- check for genuine duplicate
        # *dated* releases separately if that distinction ever matters.
        print(
            "\nWarning: Multiple CPI observations have "
            "the same release date (or both have a null "
            "release date from the documented known-gap months)."
        )

    dated_rows = cpi_df["release_date"].dropna()
    if not dated_rows.is_monotonic_increasing:
        raise ValueError(
            "CPI release dates are not sorted in "
            "ascending order."
        )


# ---------------------------------------------------------
# 12. Display summary
# ---------------------------------------------------------

def display_summary(
    cpi_df: pd.DataFrame,
) -> None:
    """
    Print a summary of the processed CPI dataset.
    """

    print("\nFirst five rows:")
    print(cpi_df.head())

    print("\nLast five rows:")
    print(cpi_df.tail())

    print("\nDataset shape:")
    print(cpi_df.shape)

    print("\nColumns:")
    print(list(cpi_df.columns))

    print("\nReference-period range:")
    print(
        "Start:",
        cpi_df["reference_month"].min(),
    )
    print(
        "End:",
        cpi_df["reference_month"].max(),
    )

    print("\nRelease-date range:")
    print(
        "Start:",
        cpi_df["release_date"].min(),
    )
    print(
        "End:",
        cpi_df["release_date"].max(),
    )

    print("\nMissing values:")
    print(cpi_df.isna().sum())


# ---------------------------------------------------------
# 13. Save processed CSV
# ---------------------------------------------------------

def save_processed_data(
    cpi_df: pd.DataFrame,
) -> Path:
    """
    Save the publication-date-aligned CPI dataset.
    """

    cpi_df.to_csv(
        PROCESSED_FILE,
        index=False,
    )

    print(
        "\nProcessed Statistics Canada CPI CSV "
        "saved to:"
    )
    print(PROCESSED_FILE)

    return PROCESSED_FILE


# ---------------------------------------------------------
# 14. Reusable pipeline function
# ---------------------------------------------------------

def run(
    use_cache: bool = False,
    cache_file: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Run the complete Statistics Canada CPI pipeline.

    Parameters
    ----------
    use_cache:
        When True, rebuild the processed CSV using cached raw
        JSON rather than making a new API request.

    cache_file:
        Optional path to a specific cached raw JSON file.

    start_date, end_date:
        Optional ISO (YYYY-MM-DD) overrides for the reference-period window.
        Default to ``config.CPI_REFERENCE_START`` / ``config.DATE_END``. CPI
        reference months are clamped to this range from ``ARCHIVE_FILE`` (see
        Issue #109) -- the requested window can extend past whatever a single
        pull's ~210-month rolling window happens to cover, as long as the
        archive has accumulated that history from an earlier pull. Extending
        ``end_date`` past the coverage of ``config/cpi_release_dates.csv``
        requires refreshing that file first (see ``build_cpi_release_dates.py``).

    Returns
    -------
    pandas.DataFrame
        CPI data aligned using publication dates.
    """

    pipeline_start = time.perf_counter()
    start, end = resolve_date_range(
        start_date, end_date, default_start=CPI_REFERENCE_START
    )

    print("\n" + "=" * 60)
    print("STATISTICS CANADA CPI DATA INGESTION")
    print(f"Window: {start} to {end}")
    print("=" * 60)

    if use_cache:

        if cache_file is None:
            raw_file = find_latest_cached_json()
        else:
            raw_file = Path(cache_file).resolve()

            print(
                "Using specified Statistics Canada cache:"
            )
            print(raw_file)

    else:
        raw_file = download_raw_data(start_date=start, end_date=end)

    data = load_raw_json(raw_file)

    pull_df = create_cpi_dataframe(data)
    pull_df["source_pull"] = raw_file.name

    # Issue #109: this pull only covers StatCan's fixed ~210-month rolling
    # window, so merge it into the archive rather than trusting it alone --
    # that preserves reference months this pull doesn't cover.
    archive_df = merge_into_archive(load_archive(), pull_df)

    cpi_df = clamp_to_range(archive_df, "reference_month", start, end)

    release_df = load_release_date_mapping()

    cpi_df = align_to_release_dates(
        cpi_df,
        release_df,
    )

    validate_dataframe(cpi_df)

    display_summary(cpi_df)

    # Only a live pull may rewrite the shared, git-tracked archive (Issue
    # #109 review, B2). A cache rebuild (--from-cache, with or without an
    # explicit --cache-file) replays whatever raw JSON happens to be on
    # this machine -- gitignored, so routinely older than the archive that
    # arrived with the last `git pull` -- and letting it write back would
    # silently revert a StatCan revision the archive already holds. This
    # also restores the "offline and deterministic" contract
    # src/pipeline_runner.py's mode="cache" documents. Persisting only
    # after validate_dataframe() succeeds means a run that fails downstream
    # (e.g. cpi_release_dates.csv doesn't cover a newly-pulled month yet)
    # never leaves the shared archive mutated either way.
    if not use_cache:
        save_archive(archive_df)
    else:
        print(
            "\nNOTE: cache rebuild -- config/statcan_cpi_archive.csv was "
            "used but not rewritten. Only a live pull updates the shared "
            "archive."
        )

    save_processed_data(cpi_df)

    elapsed_time = (
        time.perf_counter()
        - pipeline_start
    )

    print(
        "\nStatistics Canada execution time: "
        f"{elapsed_time:.2f} seconds"
    )

    print(
        "\nStatistics Canada CPI pipeline "
        "completed successfully."
    )

    return cpi_df


# ---------------------------------------------------------
# 15. Command-line arguments
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Download and process Statistics Canada CPI data "
            "using official publication-date alignment."
        )
    )

    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Rebuild statcan_cpi.csv from the latest cached "
            "raw Statistics Canada JSON file."
        ),
    )

    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a specific cached Statistics "
            "Canada JSON file."
        ),
    )

    parser.add_argument(
        "--start-date",
        default=None,
        help="ISO (YYYY-MM-DD) start of the reference-period window. Defaults to config.CPI_REFERENCE_START.",
    )

    parser.add_argument(
        "--end-date",
        default=None,
        help="ISO (YYYY-MM-DD) end of the reference-period window. Defaults to config.DATE_END.",
    )

    parser.add_argument(
        "--rebuild-archive-from-raw-cache",
        action="store_true",
        help=(
            "Rebuild config/statcan_cpi_archive.csv (Issue #109) from every "
            "cached raw JSON pull in data/raw/statcan/, oldest first, instead "
            "of running the ingestion pipeline. Derives from the raw pulls "
            "alone by default -- use this to audit or recover the archive. "
            "Combine with --merge-into-existing to seed it instead."
        ),
    )

    parser.add_argument(
        "--merge-into-existing",
        action="store_true",
        help=(
            "With --rebuild-archive-from-raw-cache, union onto the existing "
            "archive instead of deriving it from the raw pulls alone. Off "
            "by default so a rebuild is reproducible from its stated "
            "inputs and actually audits/recovers rather than preserving "
            "whatever the archive already had."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------
# 16. Script entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    arguments = parse_arguments()

    if arguments.rebuild_archive_from_raw_cache:
        rebuilt = rebuild_archive_from_raw_cache(
            merge_into_existing=arguments.merge_into_existing
        )
        print(
            f"\nRebuilt {ARCHIVE_FILE} from {len(_sorted_cached_raw_files())} "
            f"cached raw pull(s): {len(rebuilt)} reference months, "
            f"{rebuilt['reference_month'].min().date()} to "
            f"{rebuilt['reference_month'].max().date()}."
        )
    else:
        if (
            arguments.cache_file is not None
            and not arguments.from_cache
        ):
            raise ValueError(
                "--cache-file must be used together with "
                "--from-cache."
            )

        run(
            use_cache=arguments.from_cache,
            cache_file=arguments.cache_file,
            start_date=arguments.start_date,
            end_date=arguments.end_date,
        )