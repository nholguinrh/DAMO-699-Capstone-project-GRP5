"""
Path A pipeline runner for the dashboard trigger (Issue #104, Part 3).

Runs bronze -> silver -> gold using the **Path A** ingestion modules -- the only
pipeline that produces ``data/processed/*.csv`` -- then writes a run summary to
``outputs/pipeline_run_status.json``.

Scope: this stops at the Gold feature store. It does **not** re-run the
forecasting models, the Clark-West battery, or the SHAP / IRF / FEVD outputs.
Those stay as committed in ``outputs/`` and can lag a fresh Gold rebuild until
the notebooks / ``src/model_comparison.py`` are re-run.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Path A modules use bare imports (src/ on sys.path). Make them importable
# whether this module is loaded as ``pipeline_runner`` or ``src.pipeline_runner``.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from boc_data_ingestion import run as run_boc  # noqa: E402
from build_cpi_release_dates import latest_mapped_month  # noqa: E402
from fred_data_ingestion import run as run_fred  # noqa: E402
from gold_feature_pipeline import build_gold_features  # noqa: E402
from pipeline_dates import DATE_END, resolve_date_range  # noqa: E402, F401  (DATE_END re-exported for the dashboard)
from project_paths import OUTPUTS_DIR  # noqa: E402
from statcan_data_ingestion import run as run_statcan  # noqa: E402

STATUS_PATH = OUTPUTS_DIR / "pipeline_run_status.json"

MODELS_NOTE = (
    "Data and Gold features only. Forecast, Clark-West, and SHAP/IRF/FEVD "
    "outputs in outputs/ were NOT re-run and may lag this refresh."
)

VALID_MODES = ("cache", "live")


def _sources() -> tuple[tuple[str, str, Callable[..., Any]], ...]:
    """(key, display name, Path A run() callable) — resolved at call time so
    tests can monkeypatch the individual ``run_*`` names."""
    return (
        ("boc", "Bank of Canada", run_boc),
        ("fred", "FRED", run_fred),
        ("statcan", "Statistics Canada", run_statcan),
    )


def _run_source(
    fetch: Callable[..., Any],
    mode: str,
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    """
    Run one Path A source and return a status dict (never raises).

    ``cache`` mode rebuilds from the local raw JSON. ``live`` mode attempts an
    API pull and, on failure, falls back to that source's local cache.

    ``start`` / ``end`` are forwarded untouched (``None`` included) so each
    source applies its own default window -- StatCan opens on
    ``config.CPI_REFERENCE_START`` (a day earlier, for month-aligned reference
    periods), BoC and FRED on ``config.DATE_START``. Passing the resolved
    ``DATE_START`` to StatCan here would clamp off reference month
    ``2009-01-01`` and shift the canonical Gold sample forward by a month.
    """
    if mode == "cache":
        try:
            fetch(use_cache=True, start_date=start, end_date=end)
            return {"status": "success", "mode": "cache"}
        except Exception as exc:  # noqa: BLE001 - report, don't crash the run
            return {"status": "failed", "mode": "cache", "error": f"{type(exc).__name__}: {exc}"}

    try:
        fetch(use_cache=False, start_date=start, end_date=end)
        return {"status": "success", "mode": "live"}
    except Exception as live_exc:  # noqa: BLE001
        try:
            fetch(use_cache=True, start_date=start, end_date=end)
            return {
                "status": "warning",
                "mode": "cached_fallback",
                "error": f"{type(live_exc).__name__}: {live_exc}",
            }
        except Exception as cache_exc:  # noqa: BLE001
            return {
                "status": "failed",
                "mode": "live",
                "error": (
                    f"live: {type(live_exc).__name__}: {live_exc}; "
                    f"cache: {type(cache_exc).__name__}: {cache_exc}"
                ),
            }


def run_pipeline(
    mode: str = "cache",
    start_date: str | None = None,
    end_date: str | None = None,
    write_status: bool = True,
) -> dict[str, Any]:
    """
    Run bronze -> silver -> gold (Path A) and return a run report.

    Parameters
    ----------
    mode : {"cache", "live"}
        ``cache`` (default) rebuilds from the local raw JSON -- offline and
        deterministic. ``live`` pulls the APIs, falling back per-source to the
        cache on failure.
    start_date, end_date : str, optional
        ISO (YYYY-MM-DD) overrides for the ingestion window
        (default ``config.DATE_START`` / ``config.DATE_END``).
    write_status : bool
        Whether to write ``outputs/pipeline_run_status.json``.

    Returns
    -------
    dict
        Run report: ``start_time``, ``end_time``, ``mode``, ``window``,
        ``sources``, ``gold``, ``warnings``, ``models_note``.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")

    # Resolved only for the run report and the CPI-coverage check below. The
    # raw start_date / end_date (None included) are what reach each source, so
    # per-source defaults (StatCan's CPI_REFERENCE_START) are preserved -- this
    # matches src/run_data_collection.py, which also forwards args untouched.
    start, end = resolve_date_range(start_date, end_date)

    report: dict[str, Any] = {
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "window": {"start": start, "end": end},
        "sources": {},
        "gold": {},
        "warnings": [],
        "models_note": MODELS_NOTE,
    }

    latest_cpi = latest_mapped_month()
    if latest_cpi is None or latest_cpi.strftime("%Y-%m") < end[:7]:
        report["warnings"].append(
            "config/cpi_release_dates.csv does not reach the requested end "
            f"date ({end}); Statistics Canada ingestion may fail on an "
            "unmapped reference month. Run "
            f"`python src/build_cpi_release_dates.py --target-end {end}` first."
        )

    for key, name, fetch in _sources():
        report["sources"][key] = {
            "name": name,
            **_run_source(fetch, mode, start_date, end_date),
        }

    any_source_failed = any(s["status"] == "failed" for s in report["sources"].values())

    if any_source_failed:
        report["gold"] = {
            "status": "skipped",
            "reason": "one or more sources failed; keeping the existing Gold dataset",
        }
    else:
        try:
            gold = build_gold_features(save=True)
            report["gold"] = {
                "status": "success",
                "rows": int(len(gold)),
                "columns": int(gold.shape[1]),
                "date_range": [
                    str(gold.index.min().date()),
                    str(gold.index.max().date()),
                ],
            }
        except Exception as exc:  # noqa: BLE001
            report["gold"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    report["end_time"] = datetime.now().isoformat(timespec="seconds")

    if any_source_failed or report["gold"].get("status") == "failed":
        report["status"] = "error"
    elif any(s["status"] == "warning" for s in report["sources"].values()):
        report["status"] = "warning"
    else:
        report["status"] = "success"

    if write_status:
        write_run_status(report)

    return report


def write_run_status(report: dict[str, Any], path: Path | None = None) -> Path:
    """Write the run report as JSON. Returns the path written."""
    path = path or STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def read_run_status(path: Path | None = None) -> dict[str, Any] | None:
    """Read the last run report, or ``None`` if it is missing or unreadable."""
    path = path or STATUS_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Path A data + Gold pipeline (bronze -> silver -> gold)."
    )
    parser.add_argument("--mode", choices=VALID_MODES, default="cache")
    parser.add_argument("--start-date", default=None, help="ISO (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="ISO (YYYY-MM-DD)")
    args = parser.parse_args()

    result = run_pipeline(
        mode=args.mode, start_date=args.start_date, end_date=args.end_date
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] != "error" else 1)
