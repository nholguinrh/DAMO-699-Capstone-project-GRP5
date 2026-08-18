"""
Unit tests for BaseClient retry logging sanitization and config cleanup.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src import config
from src.clients.base import BaseClient, _sanitize_log_str


class DummyClient(BaseClient):
    """Concrete implementation of BaseClient for testing."""

    def fetch_series(self, series_id: str, start: str, end: str):
        return {}


def test_sanitize_log_str():
    """Test that _sanitize_log_str redacts api_key and other sensitive parameters."""
    raw_url = "https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key=SECRET_KEY_12345"
    sanitized = _sanitize_log_str(raw_url)
    assert "SECRET_KEY_12345" not in sanitized
    assert "api_key=***REDACTED***" in sanitized

    raw_exc = "Connection failed for https://example.com/api?token=SUPERSECRET&foo=bar"
    sanitized_exc = _sanitize_log_str(raw_exc)
    assert "SUPERSECRET" not in sanitized_exc
    assert "token=***REDACTED***" in sanitized_exc


def test_request_with_backoff_connection_error_redaction(tmp_path, caplog):
    """Test that ConnectionError logs redact api_key from log output."""
    client = DummyClient(source_name="test_fred", cache_dir=tmp_path)
    client.MAX_RETRIES = 2
    client.INITIAL_BACKOFF_S = 0.01
    client.JITTER_MAX_S = 0.0

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": "DGS10", "api_key": "MY_PRIVATE_KEY_999"}

    exc_msg = (
        "HTTPSConnectionPool(host='api.stlouisfed.org', port=443): "
        "Max retries exceeded with url: /fred/series/observations?series_id=DGS10&api_key=MY_PRIVATE_KEY_999"
    )

    with patch.object(
        client._session,
        "request",
        side_effect=requests.ConnectionError(exc_msg),
    ):
        with caplog.at_level("WARNING"):
            with pytest.raises(requests.HTTPError):
                client._request_with_backoff(url, params=params)

    # Check logs: MY_PRIVATE_KEY_999 must NOT appear in caplog text
    assert "MY_PRIVATE_KEY_999" not in caplog.text
    assert "***REDACTED***" in caplog.text


def test_request_with_backoff_retryable_status_redaction(tmp_path, caplog):
    """Test that HTTP retry warnings redact api_key from log output."""
    client = DummyClient(source_name="test_fred", cache_dir=tmp_path)
    client.MAX_RETRIES = 1
    client.INITIAL_BACKOFF_S = 0.01

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": "DGS10", "api_key": "SENSITIVE_KEY_456"}

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.url = (
        "https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key=SENSITIVE_KEY_456"
    )

    with patch.object(client._session, "request", return_value=mock_resp):
        with caplog.at_level("WARNING"):
            with pytest.raises(requests.HTTPError):
                client._request_with_backoff(url, params=params)

    assert "SENSITIVE_KEY_456" not in caplog.text
    assert "***REDACTED***" in caplog.text


def test_statcan_wds_url_removed():
    """Test that STATCAN_WDS_URL has been removed from src.config."""
    assert not hasattr(config, "STATCAN_WDS_URL")
    assert hasattr(config, "STATCAN_WDS_VECTOR_URL")
