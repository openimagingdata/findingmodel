from loguru import logger as logger

# Import subpackages for attribute access (e.g., findingmodel_ai.enrichment.enrich_finding)
from findingmodel_ai import authoring as authoring
from findingmodel_ai import enrichment as enrichment
from findingmodel_ai import search as search

from .config import settings as settings

__all__ = [
    "authoring",
    "enrichment",
    "logger",
    "search",
    "settings",
]

# Disable logging by default (library pattern)
logger.disable("findingmodel_ai")
