"""
Statistics Canada — Web Data Service (WDS) client.

Public REST API, no authentication required.
Docs: https://www.statcan.gc.ca/eng/developers/wds

Uses the ``getDataFromVectorByReferencePeriodRange`` method to retrieve
time series by vector ID and date range, returning JSON.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src import config
from src.clients.base import BaseClient

logger = logging.getLogger(__name__)


class StatCanClient(BaseClient):
    """Client for the Statistics Canada Web Data Service (WDS)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__(
            source_name="statcan",
            cache_dir=cache_dir or config.RAW_DATA_DIR,
        )

    def fetch_series(
        self, vector_id: str, start: str, end: str
    ) -> dict[str, Any]:
        """Fetch a single StatCan vector via the WDS REST endpoint.

        Endpoint::

            GET /t1/wds/rest/getDataFromVectorByReferencePeriodRange
                ?vectorIds={vector_id}
                &startRefPeriod=YYYY-MM-DD
                &endReferencePeriod=YYYY-MM-DD

        Parameters
        ----------
        vector_id : str
            The StatCan vector identifier (e.g. ``"41690973"``).
        start, end : str
            ISO-8601 date strings (``YYYY-MM-DD``).

        Returns
        -------
        dict
            Parsed JSON response from the WDS.

        Notes
        -----
        The WDS ``getDataFromVectorByReferencePeriodRange`` endpoint
        expects vector IDs to be quoted in the query string (e.g.
        ``vectorIds="41690973"``).  The ``requests`` library handles
        URL-encoding, so the quotes are passed directly.
        """
        params = {
            "vectorIds": vector_id,
            "startRefPeriod": start,
            "endReferencePeriod": end,
        }

        resp = self._request_with_backoff(
            config.STATCAN_WDS_VECTOR_URL, params=params
        )
        data = resp.json()

        # The WDS wraps data in a list of objects; extract metadata
        if isinstance(data, list):
            total_obs = sum(
                len(item.get("object", {}).get("vectorDataPoint", []))
                for item in data
                if isinstance(item, dict)
            )
        else:
            total_obs = "unknown"

        logger.info(
            "[statcan] Vector %s: %s observations (%s → %s)",
            vector_id, total_obs, start, end,
        )
        return data
