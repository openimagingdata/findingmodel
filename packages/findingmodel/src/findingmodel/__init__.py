from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from loguru import logger as logger

# Re-export common types for convenience
from oidm_common.embeddings import EmbeddingCache as EmbeddingCache
from oidm_common.models import IndexCode as IndexCode
from oidm_common.models import WebReference as WebReference

import findingmodel.tools as tools

from .config import get_settings as get_settings
from .create_stub import create_model_stub_from_info as create_model_stub_from_info
from .finding_info import FindingInfo as FindingInfo
from .finding_model import ChoiceAttribute as ChoiceAttribute
from .finding_model import ChoiceAttributeIded as ChoiceAttributeIded
from .finding_model import ChoiceValue as ChoiceValue
from .finding_model import ChoiceValueIded as ChoiceValueIded
from .finding_model import FindingModelBase as FindingModelBase
from .finding_model import FindingModelFull as FindingModelFull
from .finding_model import NumericAttribute as NumericAttribute
from .finding_model import NumericAttributeIded as NumericAttributeIded
from .index import FindingModelIndex as Index

__all__ = [
    "EmbeddingCache",
    "FindingInfo",
    "FindingModelBase",
    "FindingModelFull",
    "Index",
    "IndexCode",
    "WebReference",
    "__version__",
    "create_model_stub_from_info",
    "get_settings",
    "logger",
    "tools",
]

logger.disable("findingmodel")
