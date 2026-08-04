from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Mapping

import requests


# HTTP responses that are normally temporary and safe to retry for GET calls.
RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

DEFAULT_USER_AGENT = "canadian-yield-curve-data-ingestion/1.0"


def _parse_retry_after(value: str | None) -> float | None:
    """Convert a Retry-After header into a delay in seconds.

    The HTTP specification permits either:
    - an integer number of seconds; or
    - an HTTP-date.

    Invalid or past values return ``None`` so the caller can fall back to
    exponential back-off.
    """

    if value is None:
        return None

    cleaned_value = value.strip()

    if cleaned_value.isdigit():
        return max(0.0, float(cleaned_value))

    try:
        retry_datetime = parsedate_to_datetime(cleaned_value)
    except (TypeError, ValueError, OverflowError):
        return None

    if retry_datetime.tzinfo is None:
        retry_datetime = retry_datetime.replace(tzinfo=timezone.utc)

    delay = (
        retry_datetime.astimezone(timezone.utc)
        - datetime.now(timezone.utc)
    ).total_seconds()

    return max(0.0, delay) if delay > 0 else None


def _calculate_backoff_delay(
    retry_number: int,
    *,
    backoff_factor: float,
    max_backoff_seconds: float,
    jitter_seconds: float,
) -> float:
    """Calculate capped exponential back-off with optional random jitter.

    ``retry_number`` is one-based, so the default delays are approximately
    1, 2, 4, 8, and 16 seconds for retries one through five.
    """

    exponential_delay = backoff_factor ** (retry_number - 1)
    jitter = random.uniform(0.0, jitter_seconds) if jitter_seconds else 0.0

    return min(exponential_delay + jitter, max_backoff_seconds)


def get_with_retry(
    url: str,
    *,
    params: Mapping[str, object] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | tuple[float, float] = 30,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
    max_backoff_seconds: float = 60.0,
    jitter_seconds: float = 0.5,
    session: requests.Session | None = None,
) -> requests.Response:
    """Send a synchronous GET request with manual exponential back-off.

    Parameters
    ----------
    url:
        Endpoint to request.
    params:
        Optional query-string parameters.
    headers:
        Optional HTTP headers. A descriptive User-Agent is added unless one
        is supplied by the caller.
    timeout:
        Request timeout in seconds, or a ``(connect, read)`` timeout tuple.
    max_retries:
        Number of retry attempts after the initial request. For example,
        ``max_retries=5`` allows at most six total requests.
    backoff_factor:
        Exponential base used for fallback delays. The default produces
        delays of approximately 1, 2, 4, 8, and 16 seconds.
    max_backoff_seconds:
        Maximum delay permitted between attempts.
    jitter_seconds:
        Maximum random jitter added to exponential delays.
    session:
        Optional ``requests.Session``. Supplying a session enables connection
        reuse while preserving sequential execution.

    Returns
    -------
    requests.Response
        A successful HTTP response.

    Raises
    ------
    ValueError
        If retry or timing arguments are invalid.
    requests.HTTPError
        Immediately for non-retryable HTTP failures, or after all attempts
        for retryable HTTP failures.
    requests.RequestException
        After all attempts for retryable timeout or connection failures.

    Notes
    -----
    The function retries only idempotent GET requests and respects the
    server's ``Retry-After`` header when it contains either seconds or a valid
    HTTP-date.
    """

    if not url or not url.strip():
        raise ValueError("url must be a non-empty string.")

    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater.")

    if backoff_factor < 1:
        raise ValueError("backoff_factor must be at least 1.")

    if max_backoff_seconds <= 0:
        raise ValueError("max_backoff_seconds must be greater than zero.")

    if jitter_seconds < 0:
        raise ValueError("jitter_seconds must be zero or greater.")

    request_headers = dict(headers or {})
    request_headers.setdefault("User-Agent", DEFAULT_USER_AGENT)

    requester = session or requests
    total_attempts = max_retries + 1

    for attempt_number in range(1, total_attempts + 1):
        try:
            response = requester.get(
                url,
                params=params,
                headers=request_headers,
                timeout=timeout,
            )

            if response.status_code not in RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response

            if attempt_number == total_attempts:
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    raise requests.HTTPError(
                        "GET request failed after "
                        f"{total_attempts} attempt(s): {url} "
                        f"(final HTTP status {response.status_code}).",
                        response=response,
                        request=response.request,
                    ) from exc

            retry_number = attempt_number
            retry_after_delay = _parse_retry_after(
                response.headers.get("Retry-After")
            )

            if retry_after_delay is not None:
                delay = min(retry_after_delay, max_backoff_seconds)
                delay_reason = "Retry-After header"
            else:
                delay = _calculate_backoff_delay(
                    retry_number,
                    backoff_factor=backoff_factor,
                    max_backoff_seconds=max_backoff_seconds,
                    jitter_seconds=jitter_seconds,
                )
                delay_reason = "exponential back-off"

            print(
                f"Temporary HTTP {response.status_code} from {url}. "
                f"Attempt {attempt_number}/{total_attempts}; retrying in "
                f"{delay:.1f} seconds using {delay_reason}."
            )
            time.sleep(delay)

        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt_number == total_attempts:
                raise requests.RequestException(
                    "GET request failed after "
                    f"{total_attempts} attempt(s): {url}. "
                    f"Final error: {exc}"
                ) from exc

            retry_number = attempt_number
            delay = _calculate_backoff_delay(
                retry_number,
                backoff_factor=backoff_factor,
                max_backoff_seconds=max_backoff_seconds,
                jitter_seconds=jitter_seconds,
            )

            print(
                f"Temporary connection error for {url}: {exc}. "
                f"Attempt {attempt_number}/{total_attempts}; retrying in "
                f"{delay:.1f} seconds using exponential back-off."
            )
            time.sleep(delay)

    # Defensive fallback. Every normal path either returns or raises above.
    raise RuntimeError(f"Unexpected retry-loop termination for GET {url}.")