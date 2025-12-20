# Plan: Rich Anatomic Location DuckDB with Relationships

**Status:** Planning (Rev 8)
**Related Research:** [anatomic-locations-graph-research.md](anatomic-locations-graph-research.md)
**Restructuring Plan:** [oidm-package-restructuring.md](oidm-package-restructuring.md)

> **Note:** This implementation will be built within the `findingmodel` repository first (Phase 1), then extracted to a separate `anatomic-locations` package later (Phase 3). See restructuring plan for details.

## Data Source

The anatomic locations JSON data is hosted externally and **not stored in the repository**:

```
https://oidm-public.t3.storage.dev/anatomic_locations_20251220.json
```

For future updates, new data will be provided as either:
- A new dated URL (e.g., `anatomic_locations_YYYYMMDD.json`)
- A local file path passed to the migration tool

The migration/build process will download from URL or read from file as needed.

## Database Distribution

Like the FindingModel Index, the anatomic locations DuckDB will be:

1. **Built separately** - Migration tool builds DuckDB from JSON source data
2. **Published to remote storage** - Location referenced in `manifest.json`
3. **Downloaded on first use** - `ensure_anatomic_db()` fetches if not cached locally
4. **Version-checked for updates** - Checks manifest for newer versions, re-downloads if needed
5. **Cached locally** - Stored in user's cache directory (e.g., `~/.cache/findingmodel/`)

**Manifest-based distribution (shared infrastructure):**

Both FindingModel Index and Anatomic Locations databases are referenced in a central `manifest.json`:

```json
{
  "databases": {
    "findingmodel_index": {
      "url": "https://...",
      "version": "20251220",
      "checksum": "..."
    },
    "anatomic_locations": {
      "url": "https://...",
      "version": "20251220",
      "checksum": "..."
    }
  }
}
```

The manifest handling logic will be part of `oidm-core-utils`, used by both libraries.

> **Note:** The JSON source data (e.g., `anatomic_locations_20251220.json`) is hosted on `oidm-public.t3.storage.dev`. The built DuckDB files are referenced via `manifest.json` and may be hosted elsewhere.

## Goal

1. Create a **rich Pydantic data model** (`AnatomicLocation`) at the same level as `FindingModelFull`
2. Use **pre-computed hierarchy structures** (materialized path pattern) for instant navigation
3. Extend the anatomic locations DuckDB with full relationship data and new classification fields
4. Maintain existing hybrid search capabilities (vector + BM25)

## Data Summary

From `anatomic_locations.json` (2,901 entries):

| Field | Count | Description |
|-------|-------|-------------|
| `containedByRef` | 2,901 (100%) | Spatial containment parent |
| `containsRefs` | 183 | Containment children (inverse) |
| `partOfRef` | 1,101 | Mereological parent |
| `hasPartsRefs` | 312 | Mereological children (inverse) |
| `leftRef` | 1,623 | Left laterality variant |
| `rightRef` | 1,623 | Right laterality variant |
| `unsidedRef` | 1,594 | Generic (unsided) variant |
| `codes` | varies | FMA, MESH, UMLS codes |
| `snomedId` | varies | SNOMED CT identifiers |
| `acrCommonId` | varies | ACR Common Data Element IDs |

Regions: Head (869), Lower Extremity (579), Upper Extremity (582), Neck (230), Thorax (211), Abdomen (207), Pelvis (152), Breast (46), Body (25)

---

## Key Design Decisions

### 1. Materialized Path Pattern for Hierarchies

Since the database is **completely rebuilt** when data changes, we can pre-compute hierarchy structures for **instant queries without recursive CTEs**.

**Materialized Path stores the full ancestry as a delimited string:**
```
containment_path: "/RID1/RID46/RID2660/RID2772/"
```

**This enables instant queries:**
```sql
-- Find all ancestors (no recursion!)
SELECT * FROM anatomic_locations
WHERE '/RID1/RID46/RID2660/RID2772/' LIKE containment_path || '%'
ORDER BY containment_depth;

-- Find all descendants (no recursion!)
SELECT * FROM anatomic_locations
WHERE containment_path LIKE '/RID1/RID46/RID2660/%';
```

**Sources:**
- [Materialized Path Pattern](https://adamdjellouli.com/articles/databases_notes/03_sql/09_hierarchical_data)
- [Hierarchical Data in PostgreSQL](https://www.ackee.agency/blog/hierarchical-models-in-postgresql)

### 2. Simplified Tables (4 tables)

Store hierarchy and laterality references **as fields** rather than separate tables:
- `anatomic_locations` - main table with pre-computed hierarchy fields
- `anatomic_synonyms` - for exact-match lookups
- `anatomic_codes` - for IndexCode lookups (using existing IndexCode pattern)
- `anatomic_references` - for WebReference links (educational/documentation resources)

### 3. Reuse Existing IndexCode

Instead of a new `ExternalCode` class, reuse the existing `IndexCode` from `findingmodel.index_code`:
```python
from findingmodel.index_code import IndexCode
# SNOMED, FMA, MESH, UMLS, ACR all stored the same way
```

### 4. New Shared WebReference Model

A new shared model for web resource references, designed to be compatible with Tavily search results. This will live in `findingmodel.web_reference` and be used by both anatomic locations and finding models.

```python
"""Web reference model compatible with Tavily search results."""

from datetime import date
from urllib.parse import urlparse

from pydantic import BaseModel, Field, computed_field


class WebReference(BaseModel):
    """Reference to a web resource, compatible with Tavily search results.

    Can be created manually with just url+title, or populated from
    Tavily search results with additional metadata like score and content.

    Parallels IndexCode:
    - IndexCode: system + code + display
    - WebReference: domain + url + title (with optional extras)

    Sources:
    - Tavily API Reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
    """

    # Core fields (always present)
    url: str = Field(description="The URL of the web resource")
    title: str = Field(description="The title of the page/resource")

    # Optional descriptive fields
    description: str | None = Field(
        default=None,
        description="Brief description or summary (manually added or from meta description)"
    )

    # Tavily search result fields (optional - only present if from search)
    content: str | None = Field(
        default=None,
        description="AI-extracted relevant content from the page (Tavily)"
    )
    published_date: str | None = Field(
        default=None,
        description="Publication date if available (Tavily)"
    )

    # Metadata
    accessed_date: date | None = Field(
        default=None,
        description="When this reference was accessed/verified"
    )

    @computed_field
    @property
    def domain(self) -> str:
        """Extract domain from URL (e.g., 'radiopaedia.org')."""
        parsed = urlparse(self.url)
        # Remove 'www.' prefix if present
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    @classmethod
    def from_tavily_result(cls, result: dict) -> "WebReference":
        """Create WebReference from a Tavily search result dict.

        Args:
            result: A single result from Tavily's response["results"]

        Returns:
            WebReference populated with available fields
        """
        return cls(
            url=result["url"],
            title=result["title"],
            content=result.get("content"),
            published_date=result.get("published_date"),
            accessed_date=date.today(),
        )

    def __str__(self) -> str:
        return f"{self.title} ({self.domain})"

    def __repr__(self) -> str:
        return f"WebReference(url={self.url!r}, title={self.title!r})"
```

**Usage examples:**

```python
# Manual creation
ref = WebReference(
    url="https://radiopaedia.org/articles/meniscus",
    title="Meniscus | Radiology Reference Article",
    description="Overview of meniscal anatomy and pathology"
)
print(ref.domain)  # "radiopaedia.org"

# From Tavily search
from tavily import TavilyClient
client = TavilyClient(api_key="...")
response = client.search("medial meniscus anatomy radiology")
refs = [WebReference.from_tavily_result(r) for r in response["results"]]
```

### 5. Auto-Bound `_index` with weakref

When `AnatomicLocationIndex` returns objects, they are automatically bound via **weakref** to avoid circular reference memory leaks:

```python
with AnatomicLocationIndex() as index:
    location = index.get("RID2772")  # Auto-bound via weakref
    ancestors = location.get_containment_ancestors()  # Uses bound index

    # Explicit index passing still works
    ancestors = location.get_containment_ancestors(other_index)

# IMPORTANT: Location fails if used outside index context
def get_location():
    with AnatomicLocationIndex() as index:
        return index.get("RID2772")

loc = get_location()
loc.get_containment_ancestors()  # Raises ValueError - index closed
```

**Why weakref?** Without it, `Location → Index → Location` creates circular references that delay garbage collection. The weakref breaks the cycle and enforces correct lifecycle usage.

**Sources:**
- [Circular References in Python](https://medium.com/@chipiga86/circular-references-without-memory-leaks-and-destruction-of-objects-in-python-43da57915b8d)
- [Pydantic weakref discussion](https://github.com/pydantic/pydantic/discussions/2857)

### 5. STRUCT[] Arrays for Children (not JSON or parallel arrays)

For storing child references with both ID and display name, we use native DuckDB STRUCT arrays:

```sql
containment_children STRUCT(id VARCHAR, display VARCHAR)[]
```

**Why not JSON?** Requires parsing overhead, doesn't leverage vectorized execution.

**Why not parallel VARCHAR[] arrays?** Must keep two arrays in sync, requires zip logic.

**Why STRUCT[]?** Single atomic field, native DuckDB type, maintains schema integrity, leverages vectorized engine.

**Sources:**
- [DuckDB STRUCT: Handling Nested Data](https://motherduck.com/learn-more/duckdb-struct-nested-data/)
- [Shredding Deeply Nested JSON – DuckDB](https://duckdb.org/2023/03/03/json)

---

## New Classification Enums

### Laterality (Simplified)

```python
class Laterality(str, Enum):
    """Laterality classification for anatomic locations."""
    GENERIC = "generic"       # Has left/right variants (e.g., "arm")
    LEFT = "left"             # Left-sided (e.g., "left arm")
    RIGHT = "right"           # Right-sided (e.g., "right arm")
    NONLATERAL = "nonlateral" # No laterality (e.g., "heart", "spine")
```

### Location Type (NEW)

Classifies the **nature** of the location node (orthogonal to StructureType):

```python
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
```

**Usage examples:**

| Location | LocationType | StructureType | Notes |
|----------|--------------|---------------|-------|
| femur | STRUCTURE | BONE | Physical bone structure |
| liver | STRUCTURE | SOLID_ORGAN | Physical organ |
| pleural space | SPACE | None | Cavity that can contain fluid |
| anterior surface of heart | REGION | None | Positional subdivision |
| thorax | BODY_PART | None | Macro anatomical division |
| cardiovascular system | SYSTEM | None | Functional grouping |
| cervical lymph nodes | GROUP | None | Collection of structures |

### Body System

Based on the 11 major organ systems, simplified for clinical imaging:

```python
class BodySystem(str, Enum):
    """Body organ systems relevant to clinical imaging.

    Based on standard anatomical classification.
    See: https://www.kenhub.com/en/library/anatomy/human-body-systems
    """
    CARDIOVASCULAR = "cardiovascular"       # Heart, blood vessels
    RESPIRATORY = "respiratory"             # Lungs, airways, diaphragm
    GASTROINTESTINAL = "gastrointestinal"   # Stomach, intestines, liver, pancreas
    NERVOUS = "nervous"                     # Brain, spinal cord, nerves
    MUSCULOSKELETAL = "musculoskeletal"     # Bones, joints, muscles, tendons
    GENITOURINARY = "genitourinary"         # Kidneys, bladder, reproductive organs
    LYMPHATIC = "lymphatic"                 # Lymph nodes, spleen, thymus
    ENDOCRINE = "endocrine"                 # Thyroid, adrenal, pituitary glands
    INTEGUMENTARY = "integumentary"         # Skin, subcutaneous tissue
    SPECIAL_SENSES = "special_senses"       # Eyes, ears, nose (sensory organs)
```

### Structure Type

Based on FMA-RadLex ontology and radiology imaging practice:

```python
class StructureType(str, Enum):
    """Anatomical structure types for clinical imaging.

    Based on FMA-RadLex classification and radiology practice.
    See: https://pmc.ncbi.nlm.nih.gov/articles/PMC2656009/
    """
    # Musculoskeletal structures
    BONE = "bone"                   # Skeletal structures
    JOINT = "joint"                 # Articulations
    MUSCLE = "muscle"               # Skeletal, smooth, cardiac muscle
    TENDON = "tendon"               # Tendinous structures
    LIGAMENT = "ligament"           # Ligamentous structures
    CARTILAGE = "cartilage"         # Cartilaginous structures

    # Vascular structures (subtypes important for imaging)
    ARTERY = "artery"               # Arterial vessels
    VEIN = "vein"                   # Venous vessels
    PORTAL_VEIN = "portal_vein"     # Portal venous system (connects two capillary beds)
    LYMPHATIC_VESSEL = "lymphatic_vessel"  # Lymphatic vessels

    # Neural structures
    NERVE = "nerve"                 # Peripheral nerves
    GANGLION = "ganglion"           # Nerve ganglia
    PLEXUS = "plexus"               # Nerve plexuses
    CNS = "cns"                     # Central nervous system structures

    # Organs (per Radiopaedia classification)
    SOLID_ORGAN = "solid_organ"     # Liver, spleen, kidney, pancreas
    HOLLOW_ORGAN = "hollow_organ"   # Stomach, intestines, bladder, gallbladder
    GLAND = "gland"                 # Thyroid, adrenal, salivary, pituitary

    # Other structures
    SOFT_TISSUE = "soft_tissue"     # Fat, fascia, connective tissue
    SPACE = "space"                 # Anatomical spaces, compartments
    CAVITY = "cavity"               # Body cavities (thoracic, abdominal, pelvic)
    REGION = "region"               # Body regions (not specific structures)
```

**Sources:**
- [FMA-RadLex Application Ontology](https://pmc.ncbi.nlm.nih.gov/articles/PMC2656009/)
- [Solid and Hollow Abdominal Viscera](https://radiopaedia.org/articles/solid-and-hollow-abdominal-viscera)
- [Blood Vessel Structure](https://openstax.org/books/anatomy-and-physiology-2e/pages/20-1-structure-and-function-of-blood-vessels)

---

## Database Schema (3 Tables)

### Table 1: anatomic_locations (with pre-computed hierarchy)

```sql
CREATE TABLE anatomic_locations (
    -- Core identity
    id VARCHAR PRIMARY KEY,                    -- RID format
    description VARCHAR NOT NULL,

    -- Classification
    region VARCHAR,                            -- Head, Neck, Thorax, etc.
    location_type VARCHAR NOT NULL DEFAULT 'structure',  -- NEW: structure, space, region, body_part, system, group
    body_system VARCHAR,                       -- cardiovascular, respiratory, etc.
    structure_type VARCHAR,                    -- bone, muscle, nerve, etc. (only for location_type='structure')
    laterality VARCHAR NOT NULL DEFAULT 'nonlateral',

    -- Text fields
    definition TEXT,
    sex_specific VARCHAR,
    search_text TEXT NOT NULL,
    vector FLOAT[512] NOT NULL,

    -- Pre-computed CONTAINMENT hierarchy (materialized path)
    containment_path VARCHAR,                  -- "/RID1/RID46/RID2660/" from root
    containment_parent_id VARCHAR,             -- Direct parent ID
    containment_parent_display VARCHAR,        -- Denormalized display
    containment_depth INTEGER,                 -- Depth in tree (0 = root body)
    containment_children STRUCT(id VARCHAR, display VARCHAR)[],  -- Native STRUCT array

    -- Pre-computed PART-OF hierarchy (materialized path)
    partof_path VARCHAR,
    partof_parent_id VARCHAR,
    partof_parent_display VARCHAR,
    partof_depth INTEGER,
    partof_children STRUCT(id VARCHAR, display VARCHAR)[],       -- Native STRUCT array

    -- Pre-computed LATERALITY references (fields, not a table)
    left_id VARCHAR,
    left_display VARCHAR,
    right_id VARCHAR,
    right_display VARCHAR,
    generic_id VARCHAR,                        -- The unsided/generic version
    generic_display VARCHAR,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table 2: anatomic_synonyms

```sql
CREATE TABLE anatomic_synonyms (
    location_id VARCHAR NOT NULL,
    synonym VARCHAR NOT NULL,
    PRIMARY KEY (location_id, synonym)
);

CREATE INDEX idx_synonyms_synonym ON anatomic_synonyms(synonym);
CREATE INDEX idx_synonyms_location ON anatomic_synonyms(location_id);
```

### Table 3: anatomic_codes (using IndexCode pattern)

```sql
CREATE TABLE anatomic_codes (
    location_id VARCHAR NOT NULL,
    system VARCHAR NOT NULL,              -- SNOMED, FMA, MESH, UMLS, ACR
    code VARCHAR NOT NULL,
    display VARCHAR,                      -- Optional display text
    PRIMARY KEY (location_id, system, code)
);

CREATE INDEX idx_codes_system ON anatomic_codes(system);
CREATE INDEX idx_codes_lookup ON anatomic_codes(system, code);
CREATE INDEX idx_codes_location ON anatomic_codes(location_id);
```

**Code loading strategy:** When an `AnatomicLocation` is loaded, its `codes: list[IndexCode]` are **eagerly loaded** via a JOIN on `anatomic_codes`. Since most locations have only 1-3 codes, this adds minimal overhead and avoids lazy-loading complexity.

### Table 4: anatomic_references (using WebReference pattern)

```sql
CREATE TABLE anatomic_references (
    location_id VARCHAR NOT NULL,
    url VARCHAR NOT NULL,                 -- Unique URL of the resource
    title VARCHAR NOT NULL,               -- Page title
    description VARCHAR,                  -- Brief description or summary
    content VARCHAR,                      -- AI-extracted relevant content (Tavily)
    published_date VARCHAR,               -- Publication date if available
    accessed_date DATE,                   -- When reference was accessed/verified
    PRIMARY KEY (location_id, url)
);

CREATE INDEX idx_refs_location ON anatomic_references(location_id);
CREATE INDEX idx_refs_domain ON anatomic_references(
    regexp_extract(url, 'https?://(?:www\.)?([^/]+)', 1)
);  -- Index on extracted domain
```

**Reference loading strategy:** Like codes, `references: list[WebReference]` are **eagerly loaded** via a JOIN on `anatomic_references`.

**Typical references for anatomic locations:**
- Radiopaedia articles (anatomy)
- Wikipedia anatomy pages
- FMA ontology pages
- Kenhub anatomy references

### Search Indexes

```sql
-- Full-text search (BM25)
PRAGMA create_fts_index(
    'anatomic_locations', 'id', 'description', 'definition',
    stemmer = 'porter', stopwords = 'english', lower = 0, overwrite = 1
);

-- Vector search (HNSW)
CREATE INDEX idx_anatomic_hnsw ON anatomic_locations
USING HNSW (vector)
WITH (metric = 'cosine', ef_construction = 128, ef_search = 64, M = 16);

-- Standard indexes
CREATE INDEX idx_region ON anatomic_locations(region);
CREATE INDEX idx_location_type ON anatomic_locations(location_type);
CREATE INDEX idx_body_system ON anatomic_locations(body_system);
CREATE INDEX idx_structure_type ON anatomic_locations(structure_type);
CREATE INDEX idx_laterality ON anatomic_locations(laterality);
CREATE INDEX idx_containment_parent ON anatomic_locations(containment_parent_id);
CREATE INDEX idx_containment_path ON anatomic_locations(containment_path);
CREATE INDEX idx_partof_parent ON anatomic_locations(partof_parent_id);
```

---

## Pydantic Data Model

### AnatomicLocation (Main Model)

```python
"""Anatomic location data models with rich navigation capabilities."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from findingmodel.index_code import IndexCode
from findingmodel.web_reference import WebReference

if TYPE_CHECKING:
    from findingmodel.anatomic_index import AnatomicLocationIndex


class AnatomicRef(BaseModel):
    """Lightweight reference to another anatomic location."""
    id: str = Field(description="RID identifier")
    display: str = Field(description="Display name for quick reference")

    def resolve(self, index: "AnatomicLocationIndex") -> "AnatomicLocation":
        """Resolve this reference to a full AnatomicLocation."""
        return index.get(self.id)


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
    structure_type: StructureType | None = Field(default=None, description="Structure type (only for location_type=STRUCTURE)")
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
    _index: weakref.ReferenceType["AnatomicLocationIndex"] | None = PrivateAttr(default=None)

    # =========================================================================
    # Index Binding
    # =========================================================================

    def bind(self, index: "AnatomicLocationIndex") -> "AnatomicLocation":
        """Bind this location to an index via weakref.

        After binding, navigation methods can be called without passing index.
        Returns self for chaining.

        Note: Uses weakref to avoid circular reference memory leaks.
        The location will fail if used after the index is closed.
        """
        self._index = weakref.ref(index)
        return self

    def _get_index(self, index: "AnatomicLocationIndex | None") -> "AnatomicLocationIndex":
        """Get index from parameter or bound weakref.

        Raises ValueError if no index available or if bound index was garbage collected.
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

    def get_containment_ancestors(
        self, index: "AnatomicLocationIndex | None" = None
    ) -> list["AnatomicLocation"]:
        """Get all ancestors in the containment hierarchy.

        Returns list ordered from immediate parent to root (body).
        Uses pre-computed containment_path for instant lookup.
        """
        return self._get_index(index).get_containment_ancestors(self.id)

    def get_containment_descendants(
        self, index: "AnatomicLocationIndex | None" = None
    ) -> list["AnatomicLocation"]:
        """Get all descendants in the containment hierarchy.

        Uses pre-computed containment_path for instant LIKE query.
        """
        return self._get_index(index).get_containment_descendants(self.id)

    def get_containment_siblings(
        self, index: "AnatomicLocationIndex | None" = None
    ) -> list["AnatomicLocation"]:
        """Get siblings (same containment parent)."""
        if not self.containment_parent:
            return []
        return self._get_index(index).get_children_of(self.containment_parent.id)

    # =========================================================================
    # Part-Of Hierarchy Navigation
    # =========================================================================

    def get_partof_ancestors(
        self, index: "AnatomicLocationIndex | None" = None
    ) -> list["AnatomicLocation"]:
        """Get all ancestors in the part-of hierarchy."""
        return self._get_index(index).get_partof_ancestors(self.id)

    def get_parts(
        self, index: "AnatomicLocationIndex | None" = None
    ) -> list["AnatomicLocation"]:
        """Get all parts (hasParts references)."""
        return [ref.resolve(self._get_index(index)) for ref in self.partof_children]

    # =========================================================================
    # Laterality Navigation
    # =========================================================================

    def get_left(self, index: "AnatomicLocationIndex | None" = None) -> "AnatomicLocation | None":
        """Get left variant if exists."""
        if not self.left_variant:
            return None
        return self.left_variant.resolve(self._get_index(index))

    def get_right(self, index: "AnatomicLocationIndex | None" = None) -> "AnatomicLocation | None":
        """Get right variant if exists."""
        if not self.right_variant:
            return None
        return self.right_variant.resolve(self._get_index(index))

    def get_generic(self, index: "AnatomicLocationIndex | None" = None) -> "AnatomicLocation | None":
        """Get generic (unsided) variant if exists."""
        if not self.generic_variant:
            return None
        return self.generic_variant.resolve(self._get_index(index))

    def get_laterality_variants(
        self, index: "AnatomicLocationIndex | None" = None
    ) -> dict[Laterality, "AnatomicLocation"]:
        """Get all available laterality variants."""
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
        """Get code for a specific system (SNOMED, FMA, MESH, UMLS, ACR)."""
        system_upper = system.upper()
        for code in self.codes:
            if code.system.upper() == system_upper:
                return code
        return None

    # =========================================================================
    # Hierarchy Predicates
    # =========================================================================

    def is_contained_in(
        self, ancestor_id: str, index: "AnatomicLocationIndex | None" = None
    ) -> bool:
        """Check if this location is contained within the given ancestor.

        Uses pre-computed containment_path for O(1) check.
        """
        if not self.containment_path:
            return False
        return f"/{ancestor_id}/" in self.containment_path

    def is_part_of(
        self, ancestor_id: str, index: "AnatomicLocationIndex | None" = None
    ) -> bool:
        """Check if this location is part of the given ancestor.

        Uses pre-computed partof_path for O(1) check.
        """
        if not self.partof_path:
            return False
        return f"/{ancestor_id}/" in self.partof_path

    @computed_field
    @property
    def is_bilateral(self) -> bool:
        """True if this is a generic structure with left/right variants."""
        return self.laterality == Laterality.GENERIC

    @computed_field
    @property
    def is_lateralized(self) -> bool:
        """True if this is a left or right sided structure."""
        return self.laterality in (Laterality.LEFT, Laterality.RIGHT)

    # =========================================================================
    # Conversion
    # =========================================================================

    def as_index_code(self) -> IndexCode:
        """Convert to IndexCode for use in FindingModelFull.anatomic_locations."""
        return IndexCode(system="anatomic_locations", code=self.id, display=self.description)

    def __str__(self) -> str:
        return f"{self.id}: {self.description}"

    def __repr__(self) -> str:
        return f"AnatomicLocation(id={self.id!r}, description={self.description!r})"
```

### AnatomicLocationIndex (Query Interface)

```python
class AnatomicLocationIndex:
    """Index for looking up and navigating anatomic locations.

    Wraps DuckDB connection and provides high-level navigation API.
    Uses pre-computed materialized paths for instant hierarchy queries.

    Usage:
        # Context manager (CLI/scripts)
        with AnatomicLocationIndex() as index:
            location = index.get("RID2772")

        # Explicit open/close (FastAPI lifespan)
        index = AnatomicLocationIndex()
        index.open()
        # ... use index ...
        index.close()
    """

    def __init__(self, db_path: Path | None = None):
        from findingmodel.config import ensure_anatomic_db
        self.db_path = db_path or ensure_anatomic_db()
        self._conn: duckdb.DuckDBPyConnection | None = None

    def open(self) -> "AnatomicLocationIndex":
        """Open the database connection explicitly.

        For FastAPI lifespan pattern. Returns self for chaining.
        """
        if self._conn is not None:
            return self  # Already open
        self._conn = duckdb.connect(str(self.db_path), read_only=True)
        # Load FTS and VSS extensions
        self._conn.execute("INSTALL fts; LOAD fts;")
        self._conn.execute("INSTALL vss; LOAD vss;")
        return self

    def close(self) -> None:
        """Close the database connection explicitly."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "AnatomicLocationIndex":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    # =========================================================================
    # Core Lookups (all methods auto-bind returned objects)
    # =========================================================================

    def get(self, location_id: str) -> AnatomicLocation:
        """Get a single anatomic location by ID.

        The returned object is automatically bound to this index.
        """
        ...
        # Implementation note: return location.bind(self)

    def find_by_code(self, system: str, code: str) -> list[AnatomicLocation]:
        """Find locations by external code (SNOMED, FMA, etc.).

        All returned objects are automatically bound to this index.
        """
        ...
        # Implementation note: return [loc.bind(self) for loc in locations]

    def search(self, query: str, limit: int = 10) -> list[AnatomicLocation]:
        """Hybrid search (FTS + semantic) returning ranked results.

        All returned objects are automatically bound to this index.
        """
        ...

    # =========================================================================
    # Hierarchy Navigation (using pre-computed paths, auto-binds results)
    # =========================================================================

    def get_containment_ancestors(self, location_id: str) -> list[AnatomicLocation]:
        """Get containedBy ancestors using materialized path.

        SQL: WHERE :path LIKE containment_path || '%' ORDER BY containment_depth DESC
        """
        ...

    def get_containment_descendants(self, location_id: str) -> list[AnatomicLocation]:
        """Get containment descendants using materialized path.

        SQL: WHERE containment_path LIKE :path || '%'
        """
        ...

    def get_partof_ancestors(self, location_id: str) -> list[AnatomicLocation]:
        """Get partOf ancestors using materialized path."""
        ...

    def get_children_of(self, parent_id: str) -> list[AnatomicLocation]:
        """Get direct children (containment_parent_id = ?)."""
        ...

    # =========================================================================
    # Filtering and Iteration (auto-binds results)
    # =========================================================================

    def by_region(self, region: str) -> list[AnatomicLocation]:
        """Get all locations in a region."""
        ...

    def by_location_type(self, ltype: LocationType) -> list[AnatomicLocation]:
        """Get all locations of a specific location type (structure, space, region, etc.)."""
        ...

    def by_system(self, system: BodySystem) -> list[AnatomicLocation]:
        """Get all locations in a body system."""
        ...

    def by_structure_type(self, stype: StructureType) -> list[AnatomicLocation]:
        """Get all locations of a structure type (only for location_type=STRUCTURE)."""
        ...

    def __iter__(self) -> Iterator[AnatomicLocation]:
        """Iterate over all anatomic locations."""
        ...
```

---

## Usage Examples

### CLI / Script Usage (Context Manager)

```python
from findingmodel.anatomic_location import AnatomicLocation, LocationType, BodySystem, StructureType
from findingmodel.anatomic_index import AnatomicLocationIndex

# Context manager usage - opens/closes connection
with AnatomicLocationIndex() as index:
    # Get by ID (auto-bound to index)
    meniscus = index.get("RID2772")
    print(meniscus.description)  # "medial meniscus"

    # Navigate hierarchy (uses auto-bound index)
    ancestors = meniscus.get_containment_ancestors()
    for a in ancestors:
        print(f"  contained in: {a.description}")

    # Check ancestry with O(1) path lookup (no index needed)
    is_in_knee = meniscus.is_contained_in("RID2660")  # True

    # Navigate laterality
    left_meniscus = meniscus.get_left()
    right_meniscus = meniscus.get_right()

    # Code lookups
    lung = index.find_by_code("SNOMED", "39607008")[0]
    snomed = lung.get_code("SNOMED")
    print(snomed.code if snomed else None)  # "39607008"

    # Filter by classifications
    spaces = index.by_location_type(LocationType.SPACE)     # Cavities, spaces
    systems = index.by_location_type(LocationType.SYSTEM)   # Organ systems
    bones = index.by_structure_type(StructureType.BONE)     # Physical bones
    cardio = index.by_system(BodySystem.CARDIOVASCULAR)     # Cardiovascular structures

    # Use in finding model
    from findingmodel.finding_model import FindingModelFull
    fm = FindingModelFull(
        anatomic_locations=[meniscus.as_index_code()],
        ...
    )
```

### FastAPI Usage (Lifespan Singleton)

For web applications, use the lifespan pattern to keep the index open for the app's lifetime:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request

from findingmodel.anatomic_index import AnatomicLocationIndex

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open index on startup, close on shutdown."""
    index = AnatomicLocationIndex()
    index.open()  # Explicit open (no context manager)
    app.state.anatomic_index = index
    yield
    index.close()

app = FastAPI(lifespan=lifespan)

def get_anatomic_index(request: Request) -> AnatomicLocationIndex:
    """Dependency injection for index access."""
    return request.app.state.anatomic_index

@app.get("/anatomic/{location_id}")
def get_location(
    location_id: str,
    index: AnatomicLocationIndex = Depends(get_anatomic_index)
):
    location = index.get(location_id)
    return {
        "id": location.id,
        "description": location.description,
        "ancestors": [a.description for a in location.get_containment_ancestors()]
    }
```

**Thread safety notes:**
- DuckDB read-only connections can be shared across threads
- Queries are serialized internally (MVCC)
- For high concurrency, consider a connection pool pattern

**Sources:**
- [DuckDB Concurrency](https://duckdb.org/docs/stable/connect/concurrency)
- [FastAPI Singleton Pattern](https://medium.com/@hieutrantrung.it/using-fastapi-like-a-pro-with-singleton-and-dependency-injection-patterns-28de0a833a52)
- [DuckDB FastAPI Discussion](https://github.com/duckdb/duckdb/discussions/13719)

---

## Implementation Steps

### Step 0: Create WebReference Model (Shared)

Create `src/findingmodel/web_reference.py`:
- `WebReference` model (Tavily-compatible)
- `from_tavily_result()` factory method
- `domain` computed property
- Export in `__init__.py`

This is a **shared model** that will also be used by FindingModel (future).

### Step 1: Create Enums and Data Models

Create `src/findingmodel/anatomic_location.py`:
- `AnatomicRegion` enum (existing regions)
- `Laterality` enum (generic, left, right, nonlateral)
- `LocationType` enum (structure, space, region, body_part, system, group)
- `BodySystem` enum (10 systems)
- `StructureType` enum (18 types - only for LocationType.STRUCTURE)
- `AnatomicRef` model
- `AnatomicLocation` model with all navigation methods

### Step 2: Update `anatomic_migration.py`

- New schema with pre-computed fields
- Build materialized paths during migration:
  ```python
  def compute_containment_path(record, all_records) -> str:
      """Build path by traversing containedByRef chain to root."""
      path_parts = []
      current = record
      while current.get("containedByRef"):
          parent_id = current["containedByRef"]["id"]
          path_parts.insert(0, parent_id)
          current = records_by_id[parent_id]
      return "/" + "/".join(path_parts) + "/" + record["_id"] + "/"
  ```
- Extract codes as IndexCode format
- Compute laterality fields from refs

### Step 3: Create `anatomic_index.py`

- `AnatomicLocationIndex` class
- Query methods using materialized paths
- Hybrid search (existing pattern from duckdb_search.py)

### Step 4: CLI Commands

- Update `anatomic build` for new schema
- Add `anatomic query` subcommand:
  - `anatomic query ancestors <id>`
  - `anatomic query descendants <id>`
  - `anatomic query laterality <id>`
  - `anatomic query code <system> <code>`

### Step 5: Tests

- Unit tests for Pydantic models
- Unit tests for path computation
- Integration tests for hierarchy navigation
- Integration tests for search

### Step 6: Data Enrichment (Future)

- Add `body_system` and `structure_type` to existing entries
- Could use SNOMED mappings or AI classification
- These fields are optional initially

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/findingmodel/web_reference.py` | **NEW** - WebReference model (shared, Tavily-compatible) |
| `src/findingmodel/anatomic_location.py` | **NEW** - Enums, AnatomicRef, AnatomicLocation |
| `src/findingmodel/anatomic_migration.py` | New schema, path computation, code/reference extraction |
| `src/findingmodel/anatomic_index.py` | **NEW** - AnatomicLocationIndex |
| `src/findingmodel/cli.py` | Update anatomic commands |
| `src/findingmodel/__init__.py` | Export new classes (WebReference, AnatomicLocation, etc.) |
| `test/test_web_reference.py` | **NEW** - WebReference tests |
| `test/test_anatomic_location.py` | **NEW** - Model tests |
| `test/test_anatomic_index.py` | **NEW** - Index/query tests |

---

## Design Decisions Summary

1. **Materialized path pattern** - Pre-compute hierarchy for instant queries
2. **4 tables** - locations, synonyms, codes, references
3. **Reuse IndexCode** - No new ExternalCode class
4. **New shared WebReference** - Tavily-compatible model for web resource links
5. **Auto-bound `_index` via weakref** - Avoids circular reference memory leaks, enforces context lifecycle
6. **Laterality: 4 values** - generic, left, right, nonlateral
7. **LocationType: 6 values** - structure, space, region, body_part, system, group (required field)
8. **StructureType** - Only applicable when location_type = STRUCTURE (18 specific types)
9. **BodySystem** - Optional classification by organ system (10 systems)
10. **Same search capabilities** - Vector + BM25 hybrid search
11. **No foreign keys** - Integrity enforced by build process
12. **Children as STRUCT[]** - Native DuckDB `STRUCT(id VARCHAR, display VARCHAR)[]` (not JSON)
13. **Separate timestamps** - `created_at` (first import), `updated_at` (rebuild)
14. **IndexCode system** - Use `"anatomic_locations"` for `as_index_code()`
