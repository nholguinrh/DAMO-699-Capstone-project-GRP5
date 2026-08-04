"""
U.S. Federal Reserve — FRED API client.

Requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
The key must be set as an environment variable (default: ``FRED_API_KEY``).
No key is ever hardcoded in this file or anywhere in the repository.

Docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src import config
from src.clients.base import BaseClient

logger = logging.getLogger(__name__)


class FREDClient(BaseClient):
    """Client for the U.S. Federal Reserve FRED API."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__(
            source_name="fred",
            cache_dir=cache_dir or config.RAW_DATA_DIR,
        )
        self._api_key = self._load_api_key()

    @staticmethod
    def _load_api_key() -> str:
        """Read the FRED API key from the environment.

        Raises
        ------
        EnvironmentError
            If the environment variable is not set, with clear
            instructions on how to obtain and configure a key.
        """
        key = os.environ.get(config.FRED_API_KEY_ENV_VAR)
        if not key:
            raise EnvironmentError(
                f"\n{'=' * 64}\n"
                f"FRED API key not found.\n\n"
                f"The environment variable '{config.FRED_API_KEY_ENV_VAR}' "
                f"is not set.\n\n"
                f"To obtain a free key:\n"
                f"  1. Visit https://fred.stlouisfed.org/docs/api/api_key.html\n"
                f"  2. Create an account and request an API key.\n"
                f"  3. Set the key before running the pipeline:\n\n"
                f"     PowerShell:  $env:FRED_API_KEY = \"your-key-here\"\n"
                f"     Bash/Zsh:   export FRED_API_KEY=\"your-key-here\"\n"
                f"{'=' * 64}"
            )
        return key

    def fetch_series(
        self, series_id: str, start: str, end: str
    ) -> dict[str, Any]:
        """Fetch a single FRED series.

        Endpoint::

            GET /fred/series/observations
                ?series_id=...&api_key=...
                &observation_start=YYYY-MM-DD&observation_end=YYYY-MM-DD
                &file_type=json
        """
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "observation_start": start,
            "observation_end": end,
            "file_type": "json",
        }

        resp = self._request_with_backoff(
            config.FRED_BASE_URL, params=params
        )
        data = resp.json()

        obs_count = len(data.get("observations", []))
        logger.info(
            "[fred] %s: %d observations (%s → %s)",
            series_id, obs_count, start, end,
        )
        return data
