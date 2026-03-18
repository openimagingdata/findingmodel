from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from loguru import logger as logger

# Import subpackages for attribute access (e.g., findingmodel_ai.metadata.assign_metadata)
from findingmodel_ai import authoring as authoring
from findingmodel_ai import metadata as metadata
from findingmodel_ai import observability as observability
from findingmodel_ai import search as search

from .config import settings as settings

__all__ = [
    "__version__",
    "authoring",
    "logger",
    "metadata",
    "observability",
    "search",
    "settings",
]

# Disable logging by default (library pattern)
logger.disable("findingmodel_ai")
