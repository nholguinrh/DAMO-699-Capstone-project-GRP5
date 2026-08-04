"""
Bank of Canada — Valet API client.

Public REST API, no authentication required.
Docs: https://www.bankofcanada.ca/valet/docs

Handles the USD/CAD stitching logic: fetches legacy noon rate (IEXE0101,
pre-2017-04-28) and active daily average (FXUSDCAD, post-2017-05-01),
then produces a combined output cached as a single JSON file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src import config
from src.clients.base import BaseClient

logger = logging.getLogger(__name__)


class BoCClient(BaseClient):
    """Client for the Bank of Canada Valet web-services API."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__(
            source_name="boc",
            cache_dir=cache_dir or config.RAW_DATA_DIR,
        )

    def fetch_series(
        self, series_id: str, start: str, end: str
    ) -> dict[str, Any]:
        """Fetch a single BoC Valet series.

        Endpoint pattern::

            GET /valet/observations/{series_id}/json
                ?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
        """
        url = f"{config.BOC_BASE_URL}/{series_id}/json"
        params = {"start_date": start, "end_date": end}

        resp = self._request_with_backoff(url, params=params)
        data = resp.json()

        obs_count = len(data.get("observations", []))
        logger.info(
            "[boc] %s: %d observations (%s → %s)",
            series_id, obs_count, start, end,
        )
        return data

    # ── USD/CAD stitching ────────────────────────────────────────────────

    def fetch_usdcad_stitched(self, start: str, end: str) -> dict[str, Any]:
        """Fetch and stitch the two USD/CAD series.

        Per proposal §4.1, the daily USD/CAD rate is reconstructed by
        combining:
          - ``IEXE0101`` (legacy noon rate, 2009-01-01 to 2017-04-28)
          - ``FXUSDCAD``  (daily average,  2017-05-01 to present)
        """
        # Legacy series: start → 2017-04-28
        legacy_end = min(end, config.BOC_USDCAD_LEGACY_END)
        legacy_data = self.fetch_series(
            config.BOC_USDCAD_LEGACY, start, legacy_end
        )

        # Current series: 2017-05-01 → end
        current_start = max(start, config.BOC_USDCAD_CURRENT_START)
        current_data = self.fetch_series(
            config.BOC_USDCAD_CURRENT, current_start, end
        )

        # Combine observations into a single structure
        legacy_obs = legacy_data.get("observations", [])
        current_obs = current_data.get("observations", [])
        combined = {
            "stitched": True,
            "legacy_series": config.BOC_USDCAD_LEGACY,
            "current_series": config.BOC_USDCAD_CURRENT,
            "stitch_date": config.BOC_USDCAD_LEGACY_END,
            "observations": legacy_obs + current_obs,
        }

        logger.info(
            "[boc] USD/CAD stitched: %d legacy + %d current = %d total",
            len(legacy_obs), len(current_obs), len(combined["observations"]),
        )

        # Cache the stitched result
        self._cache_response("usdcad__stitched", combined)
        return combined

    def fetch_all_boc(
        self, start: str, end: str
    ) -> dict[str, dict[str, Any]]:
        """Fetch all BoC series including the stitched USD/CAD.

        This overrides the generic ``fetch_all`` to add the stitching
        step that the base class doesn't know about.
        """
        # Standard series via the base class
        results = self.fetch_all(config.BOC_SERIES, start, end)

        # USD/CAD stitched series (special case)
        results["usdcad"] = self.fetch_usdcad_stitched(start, end)

        return results
