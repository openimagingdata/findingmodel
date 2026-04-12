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

from findingmodel import tools

from .config import get_settings as get_settings
from .create_stub import create_model_stub_from_info as create_model_stub_from_info
from .finding_info import FindingInfo as FindingInfo
from .index import FindingModelIndex as Index
from .index import RelatedModelWeights as RelatedModelWeights
from .types.attributes import AttributeType as AttributeType
from .types.attributes import ChoiceAttribute as ChoiceAttribute
from .types.attributes import ChoiceAttributeIded as ChoiceAttributeIded
from .types.attributes import ChoiceValue as ChoiceValue
from .types.attributes import ChoiceValueIded as ChoiceValueIded
from .types.attributes import IndexCodeList as IndexCodeList
from .types.attributes import NumericAttribute as NumericAttribute
from .types.attributes import NumericAttributeIded as NumericAttributeIded
from .types.metadata import AgeProfile as AgeProfile
from .types.metadata import AgeStage as AgeStage
from .types.metadata import BodyRegion as BodyRegion
from .types.metadata import EntityType as EntityType
from .types.metadata import EtiologyCode as EtiologyCode
from .types.metadata import ExpectedDuration as ExpectedDuration
from .types.metadata import ExpectedTimeCourse as ExpectedTimeCourse
from .types.metadata import Modality as Modality
from .types.metadata import SexSpecificity as SexSpecificity
from .types.metadata import Subspecialty as Subspecialty
from .types.metadata import TimeCourseModifier as TimeCourseModifier
from .types.metadata import format_age_profile as format_age_profile
from .types.metadata import format_time_course as format_time_course
from .types.models import FindingModelBase as FindingModelBase
from .types.models import FindingModelFull as FindingModelFull

__all__ = [
    "AgeProfile",
    "AgeStage",
    "AttributeType",
    "BodyRegion",
    "ChoiceAttribute",
    "ChoiceAttributeIded",
    "ChoiceValue",
    "ChoiceValueIded",
    "EmbeddingCache",
    "EntityType",
    "EtiologyCode",
    "ExpectedDuration",
    "ExpectedTimeCourse",
    "FindingInfo",
    "FindingModelBase",
    "FindingModelFull",
    "Index",
    "IndexCode",
    "IndexCodeList",
    "Modality",
    "NumericAttribute",
    "NumericAttributeIded",
    "RelatedModelWeights",
    "SexSpecificity",
    "Subspecialty",
    "TimeCourseModifier",
    "WebReference",
    "__version__",
    "create_model_stub_from_info",
    "format_age_profile",
    "format_time_course",
    "get_settings",
    "logger",
    "tools",
]

logger.disable("findingmodel")
