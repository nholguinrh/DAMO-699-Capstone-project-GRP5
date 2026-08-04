# Path A — Sequential Data Collection Pipeline

Collects Bank of Canada, FRED, and Statistics Canada data for the yield-curve
capstone project, in that order, and writes both raw and processed outputs.

## Folder layout

This pipeline assumes a fixed structure — `src/` always sits directly
inside the repo root, with `config/`, `data/`, and `.env` as its siblings:

```
DAMO-699-Capstone-project-GRP5/
├── .env                          <- real API key, gitignored, lives HERE (not in src/)
├── .env.example                  <- placeholder, safe to commit
├── config/
│   └── cpi_release_dates.csv     <- built/refreshed by build_cpi_release_dates.py
├── data/
│   ├── raw/
│   │   ├── bank_of_canada/
│   │   ├── fred/
│   │   └── statcan/
│   └── processed/
└── src/
    ├── project_paths.py
    ├── run_data_collection.py
    ├── build_cpi_release_dates.py
    ├── boc_data_ingestion.py
    ├── fred_data_ingestion.py
    └── statcan_data_ingestion.py
```

`project_paths.py` resolves `PROJECT_ROOT` as exactly one level up from
`src/` — a fixed relationship, not a marker search — so every path below
depends on this folder layout staying as-is.

## Required packages

```
pandas
requests
beautifulsoup4
pdfplumber
python-dotenv
```

Install with:

```
pip install pandas requests beautifulsoup4 pdfplumber python-dotenv
```

(`beautifulsoup4` and `pdfplumber` are only needed for
`build_cpi_release_dates.py`, not for the core BoC/FRED/StatCan pipeline.)

## API key setup

1. Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
2. Copy the template and fill in your key:

```
cp .env.example .env
```

3. Edit `.env` so it contains:

```
FRED_API_KEY=your_actual_key_here
```

**`.env` must live at the repo root** (next to `config/` and `data/`),
**not inside `src/`**. `PROJECT_ROOT` is resolved as one level up from
`src/`, so that's where `load_api_key()` looks.

`.env` is gitignored and must never be committed. Only `.env.example`
(with the placeholder) belongs in the repo.

## Running the pipeline

**Live collection** (calls BoC, FRED, and StatCan APIs):

```
python run_data_collection.py
```

**Rebuild from cache** (no API calls, uses previously saved raw JSON):

```
python run_data_collection.py --from-cache
```

Both commands run each source exactly once, in sequence (BoC → FRED →
StatCan), and stop immediately with an error if any source fails.

## CPI release-date mapping (`config/cpi_release_dates.csv`)

This file maps each CPI reference month to the date Statistics Canada
actually published it. It's built from two sources, combined by
`build_cpi_release_dates.py`:

1. **Primary — StatCan's structured release-schedule JSON feed**
   (`schedule-key_indicators-eng.json`), StatCan's own machine-readable
   record of every major-indicator release since March 2012. Used for
   every reference month it covers.
2. **Fallback — StatCan's annual "Release dates" calendars**
   (`release-diffusion/{year}-eng.htm` or `.pdf`), used only to fill the
   pre-2012 gap the JSON feed doesn't cover.

A small, explicitly documented **manual override table** inside
`build_cpi_release_dates.py` fills in 8 additional 2010 months that
have no calendar page at all under the expected URL (`2010-eng.pdf`
404s) — each entry cites the exact StatCan "Daily" archive URL it came
from.

**Known, accepted gap:** two months — **2010-04** and **2010-09** —
currently have no mapped release date anywhere. Both source catalog
pages exist (`62-001-X2010004`, `62-001-X2010009`) but were
consistently 500-erroring / timing out on StatCan's end during
investigation. This is documented, not silently dropped — see
`KNOWN_UNMAPPED_CPI_MONTHS` in `statcan_data_ingestion.py`, which lets
these two rows through with `release_date = null` rather than blocking
the whole pipeline, while still hard-failing on any *other*, unexpected
missing month. If those catalog pages come back up, add the dates to
`MANUAL_OVERRIDES` in `build_cpi_release_dates.py` and remove the two
entries from `KNOWN_UNMAPPED_CPI_MONTHS`.

**You normally don't need to touch this yourself.** Ways it gets built
or refreshed:

| Method | When to use it | Hits the network? |
|---|---|---|
| `python build_cpi_release_dates.py` | First-time setup, or a full manual rebuild | Yes |
| `python run_data_collection.py --refresh-cpi-dates` | Occasional manual refresh as part of a normal pipeline run | Yes |
| `python run_data_collection.py` (no flag) | Every normal run | No — just warns if the file is missing or incomplete, and continues with what's already there |
| `.github/workflows/refresh-cpi-dates.yml` | Fully automatic, runs monthly (21st) after StatCan's mid-month CPI release, opens a PR if anything changed | Yes, but only in CI |

Regular pipeline runs (with or without `--from-cache`) never call StatCan on
their own — the mapping is only refreshed when explicitly requested via
`--refresh-cpi-dates`, by running the build script directly, or by the
scheduled GitHub Action.

## Output locations

| Type | Path |
|---|---|
| Raw cache — Bank of Canada | `data/raw/bank_of_canada/` |
| Raw cache — FRED | `data/raw/fred/` |
| Raw cache — Statistics Canada | `data/raw/statcan/` |
| Processed — Bank of Canada | `data/processed/bank_of_canada_data.csv` |
| Processed — FRED | `data/processed/fred_rates.csv` |
| Processed — Statistics Canada | `data/processed/statcan_cpi.csv` |
| CPI release-date mapping | `config/cpi_release_dates.csv` |

## Status of this data — please read before modeling

- **These outputs are preliminary, not model-ready.**
- Missing values are **preserved as-is** during collection. No forward-fill
  or other imputation is applied at this stage.
- Trading-day / market-calendar alignment and holiday treatment are **not
  yet applied** and will be decided during cleaning and feature engineering.
- CPI release-date alignment is mapped for 208 of 210 months from
  2009-01 through 2026-06 (see above). `statcan_cpi.csv` includes both
  `reference_month` and `release_date` for every observation; **2010-04
  and 2010-09 have a null `release_date`** by design (documented known
  gap, not a pipeline bug — see `KNOWN_UNMAPPED_CPI_MONTHS` above).
  Anything consuming `statcan_cpi.csv` downstream (cleaning,
  feature engineering, the VAR baseline) must decide explicitly how to
  handle these 2 rows (drop, impute, or otherwise) rather than assume
  every row has a release date. Note also that rows with a null
  `release_date` sort to the end of the file, out of chronological
  order relative to `reference_month`.
- USD/CAD is stitched from two Bank of Canada series (`IEXE0101` for the
  legacy period, `FXUSDCAD` from the 2017-05-01 methodology change onward)
  into a single `usdcad` column. Both original source columns
  (`usdcad_legacy`, `usdcad_current`) are preserved in
  `bank_of_canada_data.csv` for traceability — use `usdcad` for modeling,
  not the two source columns directly.

## Automation

- **`--refresh-cpi-dates` flag** on `run_data_collection.py` — manual,
  on-demand refresh of the CPI release-date mapping without touching the
  rest of the pipeline.
- **`.github/workflows/refresh-cpi-dates.yml`** — scheduled monthly
  refresh, opens a PR for review rather than committing directly. See the
  workflow file for the cron schedule and required repo permissions
  (Settings → Actions → General → Workflow permissions → "Read and write").
