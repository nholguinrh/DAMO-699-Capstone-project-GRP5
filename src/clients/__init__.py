"""Data-collection client classes for the three project APIs."""

from .base import BaseClient
from .boc_client import BoCClient
from .fred_client import FREDClient
from .statcan_client import StatCanClient

__all__ = ["BaseClient", "BoCClient", "FREDClient", "StatCanClient"]
