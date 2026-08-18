"""
Configuration for the data-collection pipeline (Issue #24, Path B).

All series identifiers, date ranges, and API settings live here so that
adding a new series is a one-line config change — no client code touched.

API keys are loaded automatically from a ``.env`` file in the project
root (via ``python-dotenv``).  See ``.env.example`` for the template.
"""

from pathlib import Path

# ── Load .env file (API keys) ───────────────────────────────────────────────
# Install once:  pip install python-dotenv
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)
except ImportError:
    # python-dotenv not installed — fall back to regular env vars
    pass


# ── Date range (proposal §4.1: Jan 2, 2009 – Jun 30, 2026) ──────────────────
DATE_START = "2009-01-02"
DATE_END = "2026-06-30"

# ── Output directory ─────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ── Bank of Canada — Valet API (no auth) ─────────────────────────────────────
# Docs: https://www.bankofcanada.ca/valet/docs
BOC_BASE_URL = "https://www.bankofcanada.ca/valet/observations"

BOC_SERIES = {
    "overnight_rate": "CBC20210",
    "yield_2y": "BD.CDN.2YR.DQ.YLD",
    "yield_3y": "BD.CDN.3YR.DQ.YLD",
    "yield_5y": "BD.CDN.5YR.DQ.YLD",
    "yield_7y": "BD.CDN.7YR.DQ.YLD",
    "yield_10y": "BD.CDN.10YR.DQ.YLD",
    "yield_long": "BD.CDN.LONG.DQ.YLD",
}

# USD/CAD requires stitching two series (proposal §4.1):
#   - Legacy noon rate IEXE0101 (2009-01-01 to 2017-04-28)
#   - Active daily avg  FXUSDCAD  (2017-05-01 to present)
BOC_USDCAD_LEGACY = "IEXE0101"
BOC_USDCAD_LEGACY_END = "2017-04-28"
BOC_USDCAD_CURRENT = "FXUSDCAD"
BOC_USDCAD_CURRENT_START = "2017-05-01"

# ── U.S. FRED API ────────────────────────────────────────────────────────────
# Docs: https://fred.stlouisfed.org/docs/api/fred/
# To obtain a free API key, register at:
#   https://fred.stlouisfed.org/docs/api/api_key.html
# Then add it to your .env file (see .env.example):
#   FRED_API_KEY=your-key-here
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY_ENV_VAR = "FRED_API_KEY"

FRED_SERIES = {
    "us_treasury_10y": "DGS10",
    "fed_funds_rate": "DFF",
}

# ── Statistics Canada — Web Data Service (no auth) ───────────────────────────
# Docs: https://www.statcan.gc.ca/eng/developers/wds
# Method: getDataFromVectorByReferencePeriodRange
STATCAN_WDS_VECTOR_URL = (
    "https://www150.statcan.gc.ca/t1/wds/rest/"
    "getDataFromVectorByReferencePeriodRange"
)

STATCAN_VECTORS = {
    "cpi_all_items": "41690973",
}
