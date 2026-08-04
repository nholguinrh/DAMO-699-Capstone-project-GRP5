"""
Abstract base client for API data collection.

Provides:
- Exponential back-off with jitter on HTTP 429 / 5xx (per proposal §5.1
  Bronze-layer contract: "randomized exponential back-off to comply with
  rate limits").
- Timestamped JSON caching to data/raw/.
- Structured logging.
"""

from __future__ import annotations

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class BaseClient(ABC):
    """Abstract base for per-source API clients.

    Subclasses implement ``fetch_series`` for their specific API.
    Everything else — retries, caching, logging — is handled here.
    """

    # Retry settings
    MAX_RETRIES = 5
    INITIAL_BACKOFF_S = 1.0
    BACKOFF_MULTIPLIER = 2.0
    JITTER_MAX_S = 1.0

    # HTTP status codes that trigger a retry
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self, source_name: str, cache_dir: Path) -> None:
        self.source_name = source_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "CapstoneProject-DataCollection/1.0"}
        )

    # ── HTTP with exponential back-off ───────────────────────────────────

    def _request_with_backoff(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        method: str = "GET",
    ) -> requests.Response:
        """Issue an HTTP request with exponential back-off + jitter.

        Retries on HTTP 429 (rate-limited) and 5xx (server error).
        Raises ``requests.HTTPError`` after all retries are exhausted.
        """
        backoff = self.INITIAL_BACKOFF_S

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = self._session.request(method, url, params=params,
                                             timeout=60)

                if resp.status_code not in self.RETRYABLE_STATUS_CODES:
                    resp.raise_for_status()
                    return resp

                # Retryable status — log and wait
                logger.warning(
                    "[%s] HTTP %s on attempt %d/%d for %s — "
                    "retrying in %.1fs",
                    self.source_name,
                    resp.status_code,
                    attempt,
                    self.MAX_RETRIES,
                    url,
                    backoff,
                )
            except requests.ConnectionError as exc:
                logger.warning(
                    "[%s] Connection error on attempt %d/%d for %s: %s",
                    self.source_name,
                    attempt,
                    self.MAX_RETRIES,
                    url,
                    exc,
                )

            if attempt == self.MAX_RETRIES:
                msg = (
                    f"[{self.source_name}] All {self.MAX_RETRIES} retries "
                    f"exhausted for {url}"
                )
                logger.error(msg)
                raise requests.HTTPError(msg)

            jitter = random.uniform(0, self.JITTER_MAX_S)
            time.sleep(backoff + jitter)
            backoff *= self.BACKOFF_MULTIPLIER

        # Should never reach here, but satisfy type checker
        raise requests.HTTPError("Unreachable")  # pragma: no cover

    # ── Timestamped JSON cache ───────────────────────────────────────────

    def _cache_response(self, series_id: str, data: Any) -> Path:
        """Write *data* to a timestamped JSON file in the cache directory.

        Returns the path to the written file.

        File-name format: ``{source}_{series}_{YYYYMMDD_HHMMSS}.json``
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Sanitise series_id for use in a filename
        safe_id = series_id.replace("/", "_").replace(".", "_")
        filename = f"{self.source_name}_{safe_id}_{ts}.json"
        filepath = self.cache_dir / filename

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False,
                      default=str)

        logger.info(
            "[%s] Cached %s → %s (%.1f KB)",
            self.source_name,
            series_id,
            filepath.name,
            filepath.stat().st_size / 1024,
        )
        return filepath

    # ── Public interface ─────────────────────────────────────────────────

    @abstractmethod
    def fetch_series(
        self, series_id: str, start: str, end: str
    ) -> dict[str, Any]:
        """Fetch a single series from the API.

        Parameters
        ----------
        series_id : str
            The API-specific series identifier.
        start, end : str
            ISO-8601 date strings (``YYYY-MM-DD``).

        Returns
        -------
        dict
            The parsed JSON response (or equivalent structured data).
        """

    def fetch_all(
        self,
        series_map: dict[str, str],
        start: str,
        end: str,
    ) -> dict[str, dict[str, Any]]:
        """Iterate over *series_map*, fetch each, and cache results.

        Parameters
        ----------
        series_map : dict[str, str]
            Mapping of ``{friendly_name: api_series_id}``.
        start, end : str
            ISO-8601 date strings.

        Returns
        -------
        dict[str, dict]
            ``{friendly_name: parsed_response}`` for every series.
        """
        results: dict[str, dict[str, Any]] = {}
        for friendly_name, api_id in series_map.items():
            logger.info(
                "[%s] Fetching '%s' (API id: %s) for %s → %s …",
                self.source_name,
                friendly_name,
                api_id,
                start,
                end,
            )
            data = self.fetch_series(api_id, start, end)
            self._cache_response(f"{friendly_name}__{api_id}", data)
            results[friendly_name] = data
        return results
