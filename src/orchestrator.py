"""
Concurrent data-collection orchestrator (Issue #24, Path B).

Uses ``ThreadPoolExecutor`` to fetch from all three APIs in parallel.
Each client writes to its own JSON files — no shared mutable state —
so the only synchronisation is the executor's ``as_completed`` join.

Usage
-----
::

    from src.orchestrator import run_full_collection
    import src.config as cfg

    results = run_full_collection(cfg)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src import config as default_config
from src.clients.boc_client import BoCClient
from src.clients.fred_client import FREDClient
from src.clients.statcan_client import StatCanClient

logger = logging.getLogger(__name__)


def _fetch_boc(cfg: Any) -> dict[str, Any]:
    """Fetch all Bank of Canada series (including stitched USD/CAD)."""
    client = BoCClient(cache_dir=cfg.RAW_DATA_DIR)
    return client.fetch_all_boc(cfg.DATE_START, cfg.DATE_END)


def _fetch_fred(cfg: Any) -> dict[str, Any]:
    """Fetch all FRED series."""
    client = FREDClient(cache_dir=cfg.RAW_DATA_DIR)
    return client.fetch_all(cfg.FRED_SERIES, cfg.DATE_START, cfg.DATE_END)


def _fetch_statcan(cfg: Any) -> dict[str, Any]:
    """Fetch all Statistics Canada vectors."""
    client = StatCanClient(cache_dir=cfg.RAW_DATA_DIR)
    return client.fetch_all(
        cfg.STATCAN_VECTORS, cfg.DATE_START, cfg.DATE_END
    )


# Map of source name → fetcher callable
_FETCHERS: dict[str, Any] = {
    "boc": _fetch_boc,
    "fred": _fetch_fred,
    "statcan": _fetch_statcan,
}


def run_full_collection(
    cfg: Any | None = None,
    max_workers: int = 3,
) -> dict[str, dict[str, Any]]:
    """Run all three data pulls concurrently.

    Parameters
    ----------
    cfg : module, optional
        Configuration module (defaults to ``src.config``).
    max_workers : int
        Max concurrent threads (one per API source).

    Returns
    -------
    dict[str, dict]
        ``{"boc": {...}, "fred": {...}, "statcan": {...}}``

    Raises
    ------
    Exception
        Propagated from any source that fails after all retries.
    """
    if cfg is None:
        cfg = default_config

    logger.info(
        "Starting concurrent data collection "
        "(date range: %s → %s, workers: %d)",
        cfg.DATE_START, cfg.DATE_END, max_workers,
    )
    t0 = time.perf_counter()

    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, Exception] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {
            executor.submit(fetcher, cfg): source_name
            for source_name, fetcher in _FETCHERS.items()
        }

        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                results[source] = future.result()
                logger.info("✓ %s completed successfully.", source)
            except Exception as exc:
                logger.error("✗ %s failed: %s", source, exc)
                errors[source] = exc

    elapsed = time.perf_counter() - t0
    logger.info(
        "Data collection finished in %.1fs — "
        "%d succeeded, %d failed.",
        elapsed, len(results), len(errors),
    )

    if errors:
        summary = "; ".join(
            f"{src}: {type(exc).__name__}: {exc}"
            for src, exc in errors.items()
        )
        raise RuntimeError(
            f"Data collection failed for {len(errors)} source(s): {summary}"
        )

    return results
