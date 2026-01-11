"""Anatomic location data models with rich navigation capabilities."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, cast

from oidm_common.models import IndexCode, WebReference
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from anatomic_locations.models.enums import AnatomicRegion, BodySystem, Laterality, LocationType, StructureType

if TYPE_CHECKING:
    from anatomic_locations.index import AnatomicLocationIndex


class AnatomicRef(BaseModel):
    """Lightweight reference to another anatomic location."""

    id: str = Field(description="RID identifier")
    display: str = Field(description="Display name for quick reference")

    def resolve(self, index: AnatomicLocationIndex) -> AnatomicLocation:
        """Resolve this reference to a full AnatomicLocation.

        Args:
            index: The anatomic location index to resolve from

        Returns:
            The full AnatomicLocation object
        """
        return cast("AnatomicLocation", index.get(self.id))


class AnatomicLocation(BaseModel):
    """Rich anatomic location with navigation capabilities.

    This is a top-level Pydantic object that can be used alongside FindingModelFull.
    Navigation methods use a bound index or require an explicit index parameter.

    IMPORTANT: Locations are auto-bound to their source index via weakref.
    They must be used within the index context manager scope.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Required for weakref

    id: str = Field(description="RID identifier (e.g., 'RID10049')")
    description: str = Field(description="Human-readable name")

    # Classification
    region: AnatomicRegion | None = Field(default=None)
    location_type: LocationType = Field(default=LocationType.STRUCTURE, description="Nature of this location")
    body_system: BodySystem | None = Field(default=None, description="Body organ system")
    structure_type: StructureType | None = Field(
        default=None, description="Structure type (only for location_type=STRUCTURE)"
    )
    laterality: Laterality = Field(default=Laterality.NONLATERAL)

    # Text fields
    definition: str | None = Field(default=None)
    sex_specific: str | None = Field(default=None)
    synonyms: list[str] = Field(default_factory=list)

    # Codes (using existing IndexCode)
    codes: list[IndexCode] = Field(default_factory=list)

    # Web references (using new WebReference)
    references: list[WebReference] = Field(default_factory=list, description="Educational/documentation links")

    # Pre-computed containment hierarchy
    containment_path: str | None = Field(default=None, description="Materialized path from root")
    containment_parent: AnatomicRef | None = Field(default=None)
    containment_depth: int | None = Field(default=None)
    containment_children: list[AnatomicRef] = Field(default_factory=list)

    # Pre-computed part-of hierarchy
    partof_path: str | None = Field(default=None)
    partof_parent: AnatomicRef | None = Field(default=None)
    partof_depth: int | None = Field(default=None)
    partof_children: list[AnatomicRef] = Field(default_factory=list)

    # Laterality references
    left_variant: AnatomicRef | None = Field(default=None)
    right_variant: AnatomicRef | None = Field(default=None)
    generic_variant: AnatomicRef | None = Field(default=None)

    # Private: weakref to bound index (avoids circular reference memory leaks)
    # Note: PrivateAttr is excluded from serialization by default in Pydantic v2,
    # so this won't appear in model_dump() or JSON responses.
    _index: weakref.ReferenceType[AnatomicLocationIndex] | None = PrivateAttr(default=None)

    # =========================================================================
    # Index Binding
    # =========================================================================

    def bind(self, index: AnatomicLocationIndex) -> AnatomicLocation:
        """Bind this location to an index via weakref.

        After binding, navigation methods can be called without passing index.
        Returns self for chaining.

        Note: Uses weakref to avoid circular reference memory leaks.
        The location will fail if used after the index is closed.

        Args:
            index: The anatomic location index to bind to

        Returns:
            Self for method chaining
        """
        self._index = weakref.ref(index)
        return self

    def _get_index(self, index: AnatomicLocationIndex | None) -> AnatomicLocationIndex:
        """Get index from parameter or bound weakref.

        Raises ValueError if no index available or if bound index was garbage collected.

        Args:
            index: Optional index parameter

        Returns:
            The resolved index

        Raises:
            ValueError: If no index is available
        """
        if index is not None:
            return index
        if self._index is not None:
            idx = self._index()  # Dereference the weakref
            if idx is not None:
                return idx
        raise ValueError(
            "Index no longer available. Either pass index parameter "
            "or ensure location is used within AnatomicLocationIndex context."
        )

    # =========================================================================
    # Containment Hierarchy Navigation (uses pre-computed paths)
    # =========================================================================

    def get_containment_ancestors(self, index: AnatomicLocationIndex | None = None) -> list[AnatomicLocation]:
        """Get all ancestors in the containment hierarchy.

        Returns list ordered from immediate parent to root (body).
        Uses pre-computed containment_path for instant lookup.

        Args:
            index: Optional index (uses bound index if not provided)

        Returns:
            List of ancestor locations
        """
        return cast("list[AnatomicLocation]", self._get_index(index).get_containment_ancestors(self.id))

    def get_containment_descendants(self, index: AnatomicLocationIndex | None = None) -> list[AnatomicLocation]:
        """Get all descendants in the containment hierarchy.

        Uses pre-computed containment_path for instant LIKE query.

        Args:
            index: Optional index (uses bound index if not provided)

        Returns:
            List of descendant locations
        """
        return cast("list[AnatomicLocation]", self._get_index(index).get_containment_descendants(self.id))

    def get_containment_siblings(self, index: AnatomicLocationIndex | None = None) -> list[AnatomicLocation]:
        """Get siblings (same containment parent).

        Args:
            index: Optional index (uses bound index if not provided)

        Returns:
            List of sibling locations
        """
        if not self.containment_parent:
            return []
        return cast("list[AnatomicLocation]", self._get_index(index).get_children_of(self.containment_parent.id))

    # =========================================================================
    # Part-Of Hierarchy Navigation
    # =========================================================================

    def get_partof_ancestors(self, index: AnatomicLocationIndex | None = None) -> list[AnatomicLocation]:
        """Get all ancestors in the part-of hierarchy.

        Args:
            index: Optional index (uses bound index if not provided)

        Returns:
            List of part-of ancestors
        """
        return cast("list[AnatomicLocation]", self._get_index(index).get_partof_ancestors(self.id))

    def get_parts(self, index: AnatomicLocationIndex | None = None) -> list[AnatomicLocation]:
        """Get all parts (hasParts references).

        Args:
            index: Optional index (uses bound index if not provided)

        Returns:
            List of parts
        """
        return [ref.resolve(self._get_index(index)) for ref in self.partof_children]

    # =========================================================================
    # Laterality Navigation
    # =========================================================================

    def get_left(self, index: AnatomicLocationIndex | None = None) -> AnatomicLocation | None:
        """Get left variant if exists.

        Args:
            index: Optional index (uses bound index if not provided)

        Returns:
            Left variant location or None
        """
        if not self.left_variant:
            return None
        return self.left_variant.resolve(self._get_index(index))

    def get_right(self, index: AnatomicLocationIndex | None = None) -> AnatomicLocation | None:
        """Get right variant if exists.

        Args:
            index: Optional index (uses bound index if not provided)

        Returns:
            Right variant location or None
        """
        if not self.right_variant:
            return None
        return self.right_variant.resolve(self._get_index(index))

    def get_generic(self, index: AnatomicLocationIndex | None = None) -> AnatomicLocation | None:
        """Get generic (unsided) variant if exists.

        Args:
            index: Optional index (uses bound index if not provided)

        Returns:
            Generic variant location or None
        """
        if not self.generic_variant:
            return None
        return self.generic_variant.resolve(self._get_index(index))

    def get_laterality_variants(self, index: AnatomicLocationIndex | None = None) -> dict[Laterality, AnatomicLocation]:
        """Get all available laterality variants.

        Args:
            index: Optional index (uses bound index if not provided)

        Returns:
            Dictionary mapping Laterality to location
        """
        idx = self._get_index(index)
        variants = {}
        if self.left_variant:
            variants[Laterality.LEFT] = self.left_variant.resolve(idx)
        if self.right_variant:
            variants[Laterality.RIGHT] = self.right_variant.resolve(idx)
        if self.generic_variant:
            variants[Laterality.GENERIC] = self.generic_variant.resolve(idx)
        return variants

    # =========================================================================
    # Code Lookups (no index needed)
    # =========================================================================

    def get_code(self, system: str) -> IndexCode | None:
        """Get code for a specific system (SNOMED, FMA, MESH, UMLS, ACR).

        Args:
            system: The code system to lookup (case-insensitive)

        Returns:
            The IndexCode for the system or None
        """
        system_upper = system.upper()
        for code in self.codes:
            if code.system.upper() == system_upper:
                return code
        return None

    # =========================================================================
    # Hierarchy Predicates
    # =========================================================================

    def is_contained_in(self, ancestor_id: str, index: AnatomicLocationIndex | None = None) -> bool:
        """Check if this location is contained within the given ancestor.

        Uses pre-computed containment_path for O(1) check.

        Args:
            ancestor_id: The ID of the potential ancestor
            index: Optional index (not used, for API consistency)

        Returns:
            True if this location is contained in the ancestor
        """
        if not self.containment_path:
            return False
        return f"/{ancestor_id}/" in self.containment_path

    def is_part_of(self, ancestor_id: str, index: AnatomicLocationIndex | None = None) -> bool:
        """Check if this location is part of the given ancestor.

        Uses pre-computed partof_path for O(1) check.

        Args:
            ancestor_id: The ID of the potential ancestor
            index: Optional index (not used, for API consistency)

        Returns:
            True if this location is part of the ancestor
        """
        if not self.partof_path:
            return False
        return f"/{ancestor_id}/" in self.partof_path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_bilateral(self) -> bool:
        """True if this is a generic structure with left/right variants."""
        return self.laterality == Laterality.GENERIC

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_lateralized(self) -> bool:
        """True if this is a left or right sided structure."""
        return self.laterality in (Laterality.LEFT, Laterality.RIGHT)

    # =========================================================================
    # Conversion
    # =========================================================================

    def as_index_code(self) -> IndexCode:
        """Convert to IndexCode for use in FindingModelFull.anatomic_locations.

        Returns:
            IndexCode with system="anatomic_locations"
        """
        return IndexCode(system="anatomic_locations", code=self.id, display=self.description)

    def __str__(self) -> str:
        return f"{self.id}: {self.description}"

    def __repr__(self) -> str:
        return f"AnatomicLocation(id={self.id!r}, description={self.description!r})"
