"""
Build config/cpi_release_dates.csv, mapping each CPI reference month to
the date Statistics Canada actually published it.

Two sources are combined:

1. StatCan's structured release-schedule JSON feed (primary, reliable):
   https://www150.statcan.gc.ca/n1/dai-quo/ssi/homepage/schedule-key_indicators-eng.json
   This is StatCan's own machine-readable feed of every major economic
   indicator release since March 14, 2012, including exact dates. It's
   used for every reference month it covers.

2. StatCan's annual "Release dates" calendars (fallback, pre-2012 only):
   https://www150.statcan.gc.ca/n1/release-diffusion/{year}-eng.htm (or .pdf)
   The JSON feed doesn't go back to 2009, so this fills the gap for
   2009-01 through the JSON feed's earliest covered month.

Usage:
    pip install requests beautifulsoup4 pdfplumber --break-system-packages
    python build_cpi_release_dates.py [--target-end YYYY-MM-DD]

Output:
    config/cpi_release_dates.csv  with columns: reference_month, release_date
    Covers every month from 2009-01 through --target-end (default: the
    current calendar month -- see Issue #110 and _default_target_end()'s
    docstring for why that's safe). The scheduled GitHub Actions refresh
    also passes an explicit --target-end computed at run time, so both the
    CI job and a plain local invocation keep extending the mapping every
    month with no date ever needing to be bumped by hand.
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime
from pathlib import Path

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # only needed for the pre-2012 fallback

TARGET_START = datetime(2009, 1, 1)


def _default_target_end(today: datetime | None = None) -> datetime:
    """
    Default when --target-end is omitted: the current calendar month.

    StatCan's schedule feed (the primary source, fetch_from_json_feed) is a
    *forward* calendar, not a backward log: verified live against the real
    feed, it already carries a scheduled CPI release date for the current
    reference month, and for several months beyond it (e.g. a fetch in
    September 2026 already has a scheduled release for February 2027). The
    current month is therefore always already mapped by a fresh fetch, and
    asserting completeness through it is safe (Issue #110). A shallower
    "last month" horizon was tried and reverted -- it rests on confusing
    "has the CPI *value* been published" with "is the release *date*
    scheduled", and it made write_mapping()'s forward-preserving
    effective_end silently overshoot target_end (see write_mapping()).

    Uses the local clock (like ``pipeline_dates.resolve_date_range()``'s own
    ``date.today()`` default), not UTC. The only caller that actually needs
    this default resolved is a plain local invocation of this script; the
    scheduled GitHub Actions refresh always passes its own explicit
    ``--target-end`` computed with ``date -u`` (see refresh-cpi-dates.yml),
    so the CI path and this default never need to agree on a clock -- and
    matching the pipeline's local-time convention here means a developer
    running both around local midnight sees the same reference month from
    each, rather than a UTC/local day-boundary mismatch between them.

    Takes an explicit ``today`` so callers/tests can pin the clock instead
    of inheriting a real one, and is resolved at call time by
    ``_coerce_target_end`` -- never cached at import time -- so a long-lived
    process crossing a month boundary picks up the new month, and this
    default can never itself go stale the way a hardcoded date would.
    """
    today = today or datetime.now()
    return datetime(today.year, today.month, 1)

# This script lives in src/. config/ is a sibling of src/.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "config" / "cpi_release_dates.csv"

HEADERS = {"User-Agent": "DAMO699-capstone-data-collection/1.0"}

JSON_FEED_URL = (
    "https://www150.statcan.gc.ca/n1/dai-quo/ssi/homepage/"
    "schedule-key_indicators-eng.json"
)

# 2010 has no calendar page at release-diffusion/2010-eng.{pdf,htm} (404) --
# it appears StatCan never published one under that URL pattern, unlike
# every other year from 2009 onward. These 7 entries were each confirmed
# individually from StatCan's own "The Daily" archive (the source URL is
# next to each one) rather than scraped, since the usual fallback source
# doesn't exist for this year. The remaining months in the pre-2012 gap
# that aren't covered here or by the JSON feed/annual-calendar fallback
# will still show up in the missing-months warning -- they're a genuine,
# still-open gap, not silently invented.
MANUAL_OVERRIDES: dict[str, str] = {
    # reference_month: release_date
    # source: https://www150.statcan.gc.ca/n1/daily-quotidien/100218/dq100218a-eng.htm
    "2010-02-01": "2010-03-19",
    # source: https://www150.statcan.gc.ca/n1/en/catalogue/62-001-X2010003
    #   (dcterms.issued: 2010-04-23)
    "2010-03-01": "2010-04-23",
    # source: https://www150.statcan.gc.ca/n1/daily-quotidien/100622/dq100622a-eng.htm
    "2010-05-01": "2010-06-22",
    # source: https://www150.statcan.gc.ca/n1/daily-quotidien/100622/dq100622a-eng.htm
    #   ("The June Consumer Price Index will be released on July 23.")
    "2010-06-01": "2010-07-23",
    # source: https://www150.statcan.gc.ca/n1/daily-quotidien/100820/dq100820a-eng.htm
    "2010-07-01": "2010-08-20",
    # source: https://www150.statcan.gc.ca/n1/daily-quotidien/100921/dq100921a-eng.htm
    "2010-08-01": "2010-09-21",
    # source: https://www150.statcan.gc.ca/n1/daily-quotidien/101123/dq101123a-eng.htm
    "2010-10-01": "2010-11-23",
    # source: https://www150.statcan.gc.ca/n1/daily-quotidien/101221/dq101221a-eng.htm
    "2010-11-01": "2010-12-21",
}

PAIR_PATTERN = re.compile(
    r"([A-Z][a-z]+ \d{1,2}, \d{4})\s+([A-Z][a-z]+ \d{4})"
)


# ---------------------------------------------------------------------
# Primary source: StatCan's structured JSON release-schedule feed
# ---------------------------------------------------------------------

def fetch_from_json_feed() -> dict[datetime, datetime]:
    """
    Pull every 'Consumer Price Index' entry out of StatCan's release
    schedule JSON feed. This is the authoritative, machine-readable
    source and covers releases from March 14, 2012 onward.
    """

    print(f"Fetching JSON feed: {JSON_FEED_URL}")
    resp = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(JSON_FEED_URL, headers=HEADERS, timeout=90)
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt < 3:
                wait = 2 * attempt
                print(f"  {type(exc).__name__} on attempt {attempt}/3, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    resp.raise_for_status()
    entries = resp.json()
    print(f"  Feed returned {len(entries)} total entries "
          f"(all indicators, not just CPI)")

    # Match on a normalized, case-insensitive prefix rather than an exact
    # string. StatCan has changed indicator titles slightly over the
    # years (e.g. trailing qualifiers), and an exact match silently
    # drops any entry whose title drifted even slightly -- which is
    # much harder to notice than a loud parse failure.
    def is_cpi_entry(title: str) -> bool:
        return title.strip().lower().startswith("consumer price index")

    mapping: dict[datetime, datetime] = {}
    skipped_unparseable = 0
    matched_titles: set[str] = set()

    for entry in entries:
        title = entry.get("title", "")
        if not is_cpi_entry(title):
            continue
        matched_titles.add(title)

        release_date_str = entry["date"].split(" ")[0]  # "YYYY-MM-DD HH:MM:SS"
        release_date = datetime.strptime(release_date_str, "%Y-%m-%d")

        description = entry.get("description", "").strip()
        try:
            reference_month = datetime.strptime(description, "%B %Y")
        except ValueError:
            skipped_unparseable += 1
            print(f"  Skipping unparseable CPI entry: {entry}")
            continue

        mapping[reference_month] = release_date

    print(f"  Matched title variant(s): {sorted(matched_titles)}")
    print(f"  Parsed {len(mapping)} CPI release dates from the JSON feed "
          f"({skipped_unparseable} skipped as unparseable)")
    return mapping


# ---------------------------------------------------------------------
# Fallback source: annual release-date calendars, pre-2012 only
# ---------------------------------------------------------------------

def fetch_year_text(year: int) -> str:
    """
    Fetch the plain text of a year's StatCan release-dates page.
    Tries the .htm version first, falls back to .pdf.

    StatCan's server is occasionally slow rather than actually down --
    a single 30s timeout was getting mistaken for a permanent failure
    on years that parse fine on a retry. This retries transient network
    errors (timeouts, connection errors) up to 3 times with backoff
    before giving up; a real 404 (page doesn't exist) still fails
    immediately since retrying won't fix that.
    """

    MAX_ATTEMPTS = 3
    TIMEOUT = 60  # was 30; StatCan can be slow, not just unavailable

    def _get_with_retry(url: str):
        last_exc = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                if attempt < MAX_ATTEMPTS:
                    wait = 2 * attempt
                    print(f"    {type(exc).__name__} on attempt {attempt}/{MAX_ATTEMPTS} "
                          f"for {url}, retrying in {wait}s...")
                    time.sleep(wait)
        raise last_exc

    htm_url = f"https://www150.statcan.gc.ca/n1/release-diffusion/{year}-eng.htm"
    resp = _get_with_retry(htm_url)
    if resp.status_code == 200 and "Consumer Price Index" in resp.text and BeautifulSoup:
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator="\n")

    pdf_url = f"https://www150.statcan.gc.ca/n1/release-diffusion/{year}-eng.pdf"
    resp = _get_with_retry(pdf_url)
    resp.raise_for_status()

    import pdfplumber  # imported here so a missing install only breaks
                        # the pre-2012 fallback, not the primary JSON path

    text_parts = []
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def extract_cpi_section(full_text: str) -> str:
    """
    Isolate the text between the "Consumer Price Index" heading and the
    next indicator heading ("Employment Insurance"), since the list is
    alphabetical in every year's release calendar.

    "Consumer Price Index" appears more than once in these documents:
    once as the real table heading (immediately followed by clean
    "Release date / Reference period" pairs), and again later in the
    month-by-month cross-index at the bottom of the same document
    (e.g. "22 Consumer Price Index December 2019" -- a one-line format
    with no release-date/reference-period pair immediately after it).
    Rather than assuming a fixed occurrence number, find whichever
    occurrence is actually followed by real date pairs.
    """

    occurrences = [m.start() for m in re.finditer("Consumer Price Index", full_text)]
    if not occurrences:
        raise ValueError("Could not find 'Consumer Price Index' section")

    LOOKAHEAD = 400
    MIN_PAIRS_TO_CONFIRM = 3

    start = None
    for occ in occurrences:
        window = full_text[occ : occ + LOOKAHEAD]
        if len(PAIR_PATTERN.findall(window)) >= MIN_PAIRS_TO_CONFIRM:
            start = occ
            break

    if start is None:
        # None of the occurrences looked like a real table. Use the
        # first one and let parse_pairs() return an empty list, which
        # gets reported as a per-year parse warning rather than
        # silently mis-locating the section.
        start = occurrences[0]

    end = full_text.find("Employment Insurance", start)
    if end == -1:
        end = start + 6000  # fallback window

    return full_text[start:end]


def parse_pairs(section_text: str) -> list[tuple[datetime, datetime]]:
    results = []
    for release_str, ref_str in PAIR_PATTERN.findall(section_text):
        release_date = datetime.strptime(release_str, "%B %d, %Y")
        reference_month = datetime.strptime(ref_str, "%B %Y")
        results.append((release_date, reference_month))
    return results


def fetch_from_annual_calendars(
    years: range,
    target_end: str | datetime | None = None,
) -> dict[datetime, datetime]:
    """
    Fallback scraper for years the JSON feed doesn't cover (pre-2012).
    """

    end = _coerce_target_end(target_end)

    mapping: dict[datetime, datetime] = {}
    for year in years:
        print(f"Fetching fallback calendar for {year}...")
        try:
            text = fetch_year_text(year)
            section = extract_cpi_section(text)
            pairs = parse_pairs(section)
        except Exception as exc:
            print(f"  WARNING: could not parse {year}: {type(exc).__name__}: {exc}")
            continue

        added = 0
        for release_date, reference_month in pairs:
            if TARGET_START <= reference_month <= end:
                mapping[reference_month] = release_date
                added += 1
        print(f"  Parsed {len(pairs)} rows, {added} in target range")
        time.sleep(0.5)

    return mapping


# ---------------------------------------------------------------------
# Shared validation / write helpers
# ---------------------------------------------------------------------

def _coerce_target_end(target_end: str | datetime | None) -> datetime:
    """
    Normalise a target end (str / datetime / None) to a datetime.

    ``None`` resolves against the clock at *call* time (see
    ``_default_target_end``), not at import time, so a long-lived process
    crossing a month boundary picks up the new month rather than a snapshot
    frozen at import.

    A tz-aware ``target_end`` (e.g. a caller passing
    ``datetime.now(timezone.utc)`` directly instead of a plain string) is
    normalised to naive by dropping the tzinfo -- every other datetime in
    this module (``TARGET_START``, the parsed mapping keys, the
    ``_default_target_end()`` local-clock default) is naive, and comparing a
    naive and an aware datetime raises ``TypeError`` rather than doing
    anything useful.
    """
    if target_end is None:
        return _default_target_end()
    if isinstance(target_end, datetime):
        if target_end.tzinfo is not None:
            target_end = target_end.replace(tzinfo=None)
        return target_end
    return datetime.strptime(str(target_end).strip()[:10], "%Y-%m-%d")


def missing_months(
    mapping: dict[datetime, datetime],
    target_end: str | datetime | None = None,
) -> list[str]:
    end = _coerce_target_end(target_end)
    missing = []
    cursor = TARGET_START
    while cursor <= end:
        if cursor not in mapping:
            missing.append(cursor.strftime("%Y-%m"))
        cursor = datetime(
            cursor.year + (1 if cursor.month == 12 else 0),
            1 if cursor.month == 12 else cursor.month + 1,
            1,
        )
    return missing


def is_mapping_complete(
    csv_path: Path = OUTPUT_PATH,
    target_end: str | datetime | None = None,
) -> bool:
    """
    Lightweight, no-network check used by other pipeline code (e.g.
    run_data_collection.py) to decide whether a refresh is needed.

    ``target_end`` defaults to the current calendar month (see
    ``_default_target_end``); pass a later date to check whether the
    mapping already covers an extended ingestion window.
    """

    if not csv_path.exists():
        return False

    mapping: dict[datetime, datetime] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            ref_str, release_str = line.strip().split(",")
            mapping[datetime.strptime(ref_str, "%Y-%m-%d")] = datetime.strptime(
                release_str, "%Y-%m-%d"
            )

    return len(missing_months(mapping, target_end)) == 0


def latest_mapped_month(csv_path: Path = OUTPUT_PATH) -> datetime | None:
    """
    Newest reference month present in the release-date CSV (regardless of the
    two documented pre-2012 gaps), or ``None`` if the file is missing/empty.

    Used to decide whether an extended ingestion ``end_date`` needs a refresh --
    a more precise check than ``is_mapping_complete()``, which also reports the
    permanently-unmappable 2010-04 / 2010-09 months.
    """
    if not csv_path.exists():
        return None

    months: list[datetime] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            ref_str = line.split(",")[0]
            months.append(datetime.strptime(ref_str, "%Y-%m-%d"))

    return max(months) if months else None


def build_mapping(
    target_end: str | datetime | None = None,
) -> dict[datetime, datetime]:
    """
    Build the full mapping: JSON feed as primary source, annual
    calendar scraper as fallback only for months the JSON feed
    doesn't cover (pre-2012).

    ``target_end`` extends the coverage check forward (default: the current
    calendar month -- see ``_default_target_end``).
    """

    mapping = fetch_from_json_feed()

    still_missing = [
        m for m in missing_months(mapping, target_end)
        if datetime.strptime(m, "%Y-%m") < datetime(2012, 3, 1)
    ]
    if still_missing:
        earliest_missing_year = datetime.strptime(still_missing[0], "%Y-%m").year
        latest_missing_year = datetime.strptime(still_missing[-1], "%Y-%m").year
        print(
            f"\nJSON feed doesn't cover {len(still_missing)} early month(s) "
            f"({still_missing[0]} to {still_missing[-1]}). "
            "Falling back to annual calendars for those years."
        )
        fallback_mapping = fetch_from_annual_calendars(
            range(earliest_missing_year, latest_missing_year + 2),  # +2: overlap buffer
            target_end=target_end,
        )
        # JSON feed data always wins on overlap; fallback only fills real gaps
        for ref_month, release_date in fallback_mapping.items():
            mapping.setdefault(ref_month, release_date)

    # Manual overrides for months no automated source can reach (see
    # MANUAL_OVERRIDES docstring above). Only fills genuine gaps -- never
    # overwrites a value the automated sources already found.
    applied_overrides = []
    for ref_str, release_str in MANUAL_OVERRIDES.items():
        ref_month = datetime.strptime(ref_str, "%Y-%m-%d")
        if ref_month not in mapping:
            mapping[ref_month] = datetime.strptime(release_str, "%Y-%m-%d")
            applied_overrides.append(ref_str)
    if applied_overrides:
        print(f"\nApplied {len(applied_overrides)} manual override(s) for months "
              f"no automated source covers: {applied_overrides}")

    return mapping


def write_mapping(
    mapping: dict[datetime, datetime],
    csv_path: Path = OUTPUT_PATH,
    target_end: str | datetime | None = None,
) -> int:
    """
    Write ``mapping`` to ``csv_path`` and return the number of rows written.

    Never truncates release dates the fetch already found beyond ``end`` --
    a narrower ``target_end`` than a previous run must not delete
    forward-known release dates from the CSV. StatCan's schedule feed is a
    forward calendar (see ``_default_target_end``), so on any real run
    ``effective_end`` routinely sits past ``target_end`` -- callers that
    need to reconcile a printed row count against the file must use this
    return value, not recompute a count against ``target_end`` alone.
    """
    end = _coerce_target_end(target_end)
    # Never truncate release dates the fetch already found beyond `end` -- a
    # narrower target_end than a previous run must not delete forward-known
    # release dates from the CSV.
    effective_end = max(end, max(mapping.keys())) if mapping else end
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("reference_month,release_date\n")
        for ref_month in sorted(mapping):
            if TARGET_START <= ref_month <= effective_end:
                f.write(f"{ref_month:%Y-%m-01},{mapping[ref_month]:%Y-%m-%d}\n")
                written += 1
    return written


def refresh(
    csv_path: Path = OUTPUT_PATH,
    target_end: str | datetime | None = None,
) -> None:
    """
    Public entry point for other modules to import, e.g.:
        from build_cpi_release_dates import refresh
        refresh()

    ``target_end`` (str / datetime, default: the current calendar month --
    see ``_default_target_end``) extends the mapped window forward when the
    ingestion pipeline pulls past the current cut-off.
    """

    end = _coerce_target_end(target_end)

    mapping = build_mapping(end)

    missing = missing_months(mapping, end)
    if missing:
        print(f"\nWARNING: {len(missing)} months still missing a release date:")
        print(", ".join(missing))
        print("This is likely a genuine gap in both sources for those months --")
        print("check them manually before treating the mapping as complete.")
    else:
        print(f"\nComplete: all months from {TARGET_START:%Y-%m} to "
              f"{end:%Y-%m} are mapped.")

    written = write_mapping(mapping, csv_path, end)
    asserted = len([m for m in mapping if TARGET_START <= m <= end])

    print(f"\nWrote {written} rows to {csv_path}")
    if written > asserted:
        # Not an error: StatCan's schedule feed publishes release dates for
        # reference months beyond `end`, and write_mapping() preserves them
        # rather than truncating. Reported explicitly so this count always
        # reconciles against the committed CSV.
        print(
            f"  ({asserted} rows through the asserted target-end {end:%Y-%m}; "
            f"{written - asserted} further row(s) carry StatCan's "
            "forward-scheduled release dates.)"
        )
    print("Sources:")
    print(f"  Primary (2012-03 onward): {JSON_FEED_URL}")
    print("  Fallback (pre-2012): https://www150.statcan.gc.ca/n1/release-diffusion/"
          "{year}-eng.htm (or .pdf)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Rebuild config/cpi_release_dates.csv (CPI reference month -> "
            "StatCan release date)."
        )
    )
    parser.add_argument(
        "--target-end",
        default=None,
        help=(
            "ISO (YYYY-MM-DD) last reference month to map. Defaults to the "
            "current calendar month. Use a later date when the ingestion "
            "window is extended past the current cut-off."
        ),
    )
    args = parser.parse_args()

    refresh(target_end=args.target_end)
