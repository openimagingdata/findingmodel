"""Anatomic location ontology navigation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from anatomic_locations.index import AnatomicLocationIndex
from anatomic_locations.models import (
    AnatomicLocation,
    AnatomicRef,
    AnatomicRegion,
    BodySystem,
    Laterality,
    LocationType,
    StructureType,
)

__all__ = [
    "AnatomicLocation",
    "AnatomicLocationIndex",
    "AnatomicRef",
    "AnatomicRegion",
    "BodySystem",
    "Laterality",
    "LocationType",
    "StructureType",
    "__version__",
]
