from loguru import logger as logger

import findingmodel.tools as tools

from .anatomic_location import AnatomicLocation as AnatomicLocation
from .anatomic_location import AnatomicRef as AnatomicRef
from .anatomic_location import AnatomicRegion as AnatomicRegion
from .anatomic_location import BodySystem as BodySystem
from .anatomic_location import Laterality as Laterality
from .anatomic_location import LocationType as LocationType
from .anatomic_location import StructureType as StructureType
from .config import settings as settings
from .finding_info import FindingInfo as FindingInfo
from .finding_model import ChoiceAttribute as ChoiceAttribute
from .finding_model import ChoiceAttributeIded as ChoiceAttributeIded
from .finding_model import ChoiceValue as ChoiceValue
from .finding_model import ChoiceValueIded as ChoiceValueIded
from .finding_model import FindingModelBase as FindingModelBase
from .finding_model import FindingModelFull as FindingModelFull
from .finding_model import NumericAttribute as NumericAttribute
from .finding_model import NumericAttributeIded as NumericAttributeIded
from .index import DuckDBIndex as DuckDBIndex
from .index import DuckDBIndex as Index  # DuckDB is now the default Index
from .index_code import IndexCode as IndexCode
from .web_reference import WebReference as WebReference

__all__ = [
    "AnatomicLocation",
    "AnatomicRef",
    "AnatomicRegion",
    "BodySystem",
    "DuckDBIndex",
    "FindingInfo",
    "FindingModelBase",
    "FindingModelFull",
    "Index",
    "IndexCode",
    "Laterality",
    "LocationType",
    "StructureType",
    "WebReference",
    "logger",
    "settings",
    "tools",
]

logger.disable("findingmodel")
