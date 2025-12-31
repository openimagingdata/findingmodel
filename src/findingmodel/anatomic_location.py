"""Anatomic location data models with rich navigation capabilities."""

from __future__ import annotations

import weakref
from enum import Enum
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from findingmodel.index_code import IndexCode
from findingmodel.web_reference import WebReference

if TYPE_CHECKING:
    from findingmodel.anatomic_index import AnatomicLocationIndex


class AnatomicRegion(str, Enum):
    """Body regions for anatomic locations.

    Derived from existing anatomic_locations.json data.
    """

    HEAD = "Head"
    NECK = "Neck"
    THORAX = "Thorax"
    ABDOMEN = "Abdomen"
    PELVIS = "Pelvis"
    UPPER_EXTREMITY = "Upper Extremity"
    LOWER_EXTREMITY = "Lower Extremity"
    BREAST = "Breast"
    BODY = "Body"  # Whole body / systemic


class Laterality(str, Enum):
    """Laterality classification for anatomic locations."""

    GENERIC = "generic"  # Has left/right variants (e.g., "arm")
    LEFT = "left"  # Left-sided (e.g., "left arm")
    RIGHT = "right"  # Right-sided (e.g., "right arm")
    NONLATERAL = "nonlateral"  # No laterality (e.g., "heart", "spine")


class LocationType(str, Enum):
    """Classification of anatomic location by its nature.

    This is a high-level categorization that determines how the location
    is used in navigation and finding attribution. Orthogonal to StructureType,
    which describes what KIND of physical structure something is.

    Based on FMA ontology top-level organization:
    - Material anatomical entity → STRUCTURE
    - Immaterial anatomical entity → SPACE, REGION
    - Body part (macro division) → BODY_PART
    - Organ system (functional) → SYSTEM
    - Set/Collection → GROUP

    See: https://bioportal.bioontology.org/ontologies/FMA
    """

    STRUCTURE = "structure"
    # Discrete anatomical entities that can be identified as specific physical things.
    # Examples: liver, femur, aorta, biceps muscle, optic nerve, thyroid gland
    # These have a StructureType value (BONE, SOLID_ORGAN, NERVE, etc.)

    SPACE = "space"
    # Bounded volumes and cavities - anatomically defined 3D regions.
    # Examples: pleural space, peritoneal cavity, joint space, epidural space
    # Key distinction: spaces can contain things (fluid, air, masses)

    REGION = "region"
    # Spatial subdivisions of structures or body parts - directional/positional areas.
    # Examples: anterior surface, medial aspect, subareolar region, apex of lung
    # Key distinction: regions are conceptual divisions, not discrete structures

    BODY_PART = "body_part"
    # Macro anatomical divisions used for coarse localization.
    # Examples: thorax, abdomen, arm, head, pelvis, neck
    # Key distinction: composite regions containing many structures/spaces

    SYSTEM = "system"
    # Organ systems - functional groupings of related structures.
    # Examples: cardiovascular system, nervous system, musculoskeletal system
    # Key distinction: organizational concepts for navigation, not physical locations

    GROUP = "group"
    # Collections of related structures that are often referenced together.
    # Examples: cervical lymph nodes, set of ribs, cervical vertebrae, carpal bones
    # Key distinction: named sets that simplify reference to multiple structures


class BodySystem(str, Enum):
    """Body organ systems relevant to clinical imaging.

    Based on standard anatomical classification.
    See: https://www.kenhub.com/en/library/anatomy/human-body-systems
    """

    CARDIOVASCULAR = "cardiovascular"  # Heart, blood vessels
    RESPIRATORY = "respiratory"  # Lungs, airways, diaphragm
    GASTROINTESTINAL = "gastrointestinal"  # Stomach, intestines, liver, pancreas
    NERVOUS = "nervous"  # Brain, spinal cord, nerves
    MUSCULOSKELETAL = "musculoskeletal"  # Bones, joints, muscles, tendons
    GENITOURINARY = "genitourinary"  # Kidneys, bladder, reproductive organs
    LYMPHATIC = "lymphatic"  # Lymph nodes, spleen, thymus
    ENDOCRINE = "endocrine"  # Thyroid, adrenal, pituitary glands
    INTEGUMENTARY = "integumentary"  # Skin, subcutaneous tissue
    SPECIAL_SENSES = "special_senses"  # Eyes, ears, nose (sensory organs)


class StructureType(str, Enum):
    """Anatomical structure types for clinical imaging.

    Based on FMA-RadLex classification and radiology practice.
    Organized by category for easier navigation.

    See:
    - FMA-RadLex: https://pmc.ncbi.nlm.nih.gov/articles/PMC2656009/
    - Neuroanatomical domain of FMA: https://pmc.ncbi.nlm.nih.gov/articles/PMC3944952/
    - IASLC Lymph Node Map: https://pmc.ncbi.nlm.nih.gov/articles/PMC4499584/
    """

    # === MUSCULOSKELETAL ===
    BONE = "bone"  # Skeletal structures
    JOINT = "joint"  # Articulations
    MUSCLE = "muscle"  # Skeletal, smooth, cardiac muscle
    TENDON = "tendon"  # Tendinous structures
    LIGAMENT = "ligament"  # Ligamentous structures
    CARTILAGE = "cartilage"  # Cartilaginous structures
    BURSA = "bursa"  # Synovial bursae

    # === VASCULAR ===
    ARTERY = "artery"  # Arterial vessels
    VEIN = "vein"  # Venous vessels
    PORTAL_VEIN = "portal_vein"  # Portal venous system
    LYMPHATIC_VESSEL = "lymphatic_vessel"  # Lymphatic vessels

    # === NEURAL (peripheral) ===
    NERVE = "nerve"  # Peripheral nerves
    GANGLION = "ganglion"  # Nerve ganglia
    PLEXUS = "plexus"  # Nerve/vascular plexuses

    # === BRAIN-SPECIFIC (CNS) ===
    GYRUS = "gyrus"  # Cortical folds (precentral gyrus, cingulate)
    SULCUS = "sulcus"  # Cortical grooves (central sulcus, lateral)
    FISSURE = "fissure"  # Major brain divisions (Sylvian, longitudinal)
    WHITE_MATTER_TRACT = "white_matter_tract"  # White matter pathways (corticospinal)
    NUCLEUS = "nucleus"  # Deep gray matter clusters (caudate, thalamus)
    VENTRICLE = "ventricle"  # CSF-filled brain cavities (lateral, third, fourth)
    CISTERN = "cistern"  # Subarachnoid CSF spaces (cisterna magna)

    # === ORGANS ===
    SOLID_ORGAN = "solid_organ"  # Liver, spleen, kidney, pancreas
    HOLLOW_ORGAN = "hollow_organ"  # Stomach, intestines, bladder, gallbladder
    GLAND = "gland"  # Thyroid, adrenal, salivary, pituitary

    # === LYMPHATIC ===
    LYMPH_NODE = "lymph_node"  # Individual lymph nodes
    LYMPH_NODE_STATION = "lymph_node_station"  # TNM staging groups (IASLC stations, levels)

    # === ANATOMICAL ORGANIZATION ===
    COMPARTMENT = "compartment"  # Fascial-bounded spaces (mediastinal, retroperitoneal)
    MEMBRANE = "membrane"  # Serous membranes, meninges (pleura, peritoneum, dura)

    # === SPATIAL/OTHER ===
    SOFT_TISSUE = "soft_tissue"  # Fat, fascia, connective tissue
    SPACE = "space"  # Anatomical spaces
    CAVITY = "cavity"  # Body cavities (thoracic, abdominal, pelvic)
    REGION = "region"  # Body regions (not specific structures)


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
