import argparse
import time

from boc_data_ingestion import run as run_boc
from build_cpi_release_dates import is_mapping_complete, refresh as refresh_cpi_dates
from fred_data_ingestion import run as run_fred
from pipeline_dates import resolve_date_range
from statcan_data_ingestion import run as run_statcan


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the data collection pipeline.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the M2 data collection pipeline: Bank of Canada, "
            "FRED, and Statistics Canada, in sequence."
        )
    )

    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Rebuild all processed CSVs from cached raw JSON "
            "instead of making new API requests."
        ),
    )

    parser.add_argument(
        "--refresh-cpi-dates",
        action="store_true",
        help=(
            "Re-fetch config/cpi_release_dates.csv from Statistics Canada's "
            "official release calendars before running the pipeline. Only "
            "needed occasionally (e.g. monthly, when a new CPI release date "
            "is published) -- not required on every run."
        ),
    )

    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "ISO (YYYY-MM-DD) start of the ingestion window, passed to all "
            "three sources. Defaults to config.DATE_START "
            "(config.CPI_REFERENCE_START for StatCan)."
        ),
    )

    parser.add_argument(
        "--end-date",
        default=None,
        help=(
            "ISO (YYYY-MM-DD) end of the ingestion window, passed to all three "
            "sources. Defaults to config.DATE_END. Extending past the current "
            "CPI release-date coverage also refreshes config/cpi_release_dates.csv."
        ),
    )

    return parser.parse_args()


def main():
    """
    Execute the complete data collection pipeline.

    Path A:
        1. Bank of Canada
        2. FRED
        3. Statistics Canada

    Each source is collected synchronously, exactly once, and the
    --from-cache flag is parsed a single time and passed consistently
    to all three ingestion functions. If any source fails, the
    pipeline stops immediately with a clear error rather than
    continuing to the next source.
    """

    args = parse_arguments()
    overall_start = time.perf_counter()

    # Fail fast on a malformed / future / inverted window before doing any work.
    resolve_date_range(args.start_date, args.end_date)

    print("=" * 70)
    print("M2 DATA COLLECTION PIPELINE")
    print(f"Mode: {'--from-cache' if args.from_cache else 'live API calls'}")
    print(
        "Window: "
        f"{args.start_date or 'config.DATE_START'} to "
        f"{args.end_date or 'config.DATE_END'}"
    )
    print("=" * 70)

    if args.refresh_cpi_dates:
        print("\nRefreshing config/cpi_release_dates.csv from StatCan...")
        refresh_cpi_dates(target_end=args.end_date)
    elif not is_mapping_complete(target_end=args.end_date):
        print(
            "\nNOTE: config/cpi_release_dates.csv is missing or does not "
            "cover the requested end date.\n"
            "Run with --refresh-cpi-dates to rebuild it, or run "
            "build_cpi_release_dates.py directly. Continuing with the "
            "existing file for now (StatCan ingestion will fail if it "
            "encounters an unmapped reference month)."
        )

    try:
        print("\nSTEP 1 OF 3: Bank of Canada")
        run_boc(
            use_cache=args.from_cache,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        print("\nSTEP 2 OF 3: FRED")
        run_fred(
            use_cache=args.from_cache,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        print("\nSTEP 3 OF 3: Statistics Canada")
        run_statcan(
            use_cache=args.from_cache,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - overall_start
        print("\n" + "=" * 70)
        print("DATA COLLECTION FAILED")
        print("=" * 70)
        print(f"Pipeline stopped after {elapsed:.2f} seconds due to an error:")
        print(f"{type(exc).__name__}: {exc}")
        raise

    elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Total pipeline execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()