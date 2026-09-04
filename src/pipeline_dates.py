"""
Shared date-range handling for the data pipeline (Issue #104).

Every Path A ingestion ``run()`` (Bank of Canada, FRED, Statistics Canada) and
``src/run_data_collection.py`` accept optional ``start_date`` / ``end_date``
overrides. This module is the single place that validates and normalises them,
and the single place that clamps a finished frame to the requested window.

Defaults come from ``src/config.py`` (``DATE_START`` / ``DATE_END`` /
``CPI_REFERENCE_START``).
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Path A modules use bare imports (src/ on sys.path). Make ``config`` importable
# whether this module is loaded as ``pipeline_dates`` or ``src.pipeline_dates``.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import CPI_REFERENCE_START, DATE_END, DATE_START  # noqa: E402

__all__ = [
    "DATE_START",
    "DATE_END",
    "CPI_REFERENCE_START",
    "resolve_date_range",
    "clamp_to_range",
]

_ISO_FORMAT = "%Y-%m-%d"

DateLike = str | date | datetime


def _coerce(value: DateLike, field: str) -> date:
    """Normalise a str / date / datetime to a ``datetime.date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), _ISO_FORMAT).date()
        except ValueError as exc:
            raise ValueError(
                f"{field} must be an ISO date (YYYY-MM-DD); got {value!r}"
            ) from exc
    raise TypeError(
        f"{field} must be str, datetime.date, or datetime.datetime; "
        f"got {type(value).__name__}"
    )


def resolve_date_range(
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
    *,
    default_start: DateLike = DATE_START,
    default_end: DateLike = DATE_END,
    today: date | None = None,
) -> tuple[str, str]:
    """
    Validate and normalise a ``(start_date, end_date)`` pair to ISO strings.

    ``None`` on either side falls back to the corresponding default. Passing
    already-normalised ISO strings back through is idempotent.

    Raises
    ------
    ValueError
        If either date is malformed, the range is empty or inverted, or the
        end date is in the future (the sources only publish up to today).
    """
    start = _coerce(start_date if start_date is not None else default_start, "start_date")
    end = _coerce(end_date if end_date is not None else default_end, "end_date")

    if start >= end:
        raise ValueError(
            f"start_date ({start.isoformat()}) must be strictly before "
            f"end_date ({end.isoformat()})"
        )

    today = today or date.today()
    if end > today:
        raise ValueError(
            f"end_date ({end.isoformat()}) is in the future; the data sources "
            f"only publish observations up to today ({today.isoformat()})"
        )

    return start.isoformat(), end.isoformat()


def clamp_to_range(
    df: pd.DataFrame,
    date_column: str,
    start: DateLike,
    end: DateLike,
) -> pd.DataFrame:
    """
    Return the rows of ``df`` whose ``date_column`` falls within
    ``[start, end]`` inclusive, with a fresh ``RangeIndex``.

    A no-op when the frame is already inside the window (the default pipeline
    range), so ``--from-cache`` rebuilds stay byte-identical to a pre-override
    run.
    """
    if date_column not in df.columns:
        raise KeyError(
            f"clamp_to_range: {date_column!r} is not a column in the dataframe "
            f"(have: {list(df.columns)})"
        )

    stamps = pd.to_datetime(df[date_column])
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    mask = (stamps >= lo) & (stamps <= hi)
    return df.loc[mask].reset_index(drop=True)
