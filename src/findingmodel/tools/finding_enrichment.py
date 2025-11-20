"""
Finding Enrichment System

Provides structured models and types for enriching finding models with comprehensive metadata
including ontology codes, body regions, etiologies, imaging modalities, subspecialties, and anatomic locations.

This module defines the core data structures used throughout the finding enrichment workflow.
"""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal

from findingmodel import logger
from findingmodel.finding_model import FindingModelFull
from findingmodel.index import DuckDBIndex
from findingmodel.index_code import IndexCode
from findingmodel.tools.ontology_search import OntologySearchResult

# Type aliases for constrained value sets

BodyRegion = Literal["ALL", "Head", "Neck", "Chest", "Breast", "Abdomen", "Arm", "Leg"]
"""Body region categories used in radiology imaging.

- ALL: Finding applies to entire body or multiple regions
- Head: Cranial and intracranial structures
- Neck: Cervical region
- Chest: Thorax including lungs and mediastinum
- Breast: Breast tissue
- Abdomen: Abdominal cavity and retroperitoneum
- Arm: Upper extremity
- Leg: Lower extremity
"""

Modality = Literal["XR", "CT", "MR", "US", "PET", "NM", "MG", "RF", "DSA"]
"""Imaging modality codes.

- XR: Radiography (plain X-rays)
- CT: Computed Tomography
- MR: Magnetic Resonance Imaging
- US: Ultrasound (including Doppler)
- PET: Positron Emission Tomography
- NM: Nuclear Medicine (single-photon/SPECT)
- MG: Mammography
- RF: Fluoroscopy (real-time X-ray)
- DSA: Digital Subtraction Angiography
"""

Subspecialty = Literal["AB", "BR", "CA", "CH", "ER", "GI", "GU", "HN", "IR", "MI", "MK", "NR", "OB", "OI", "PD", "VI"]
"""Radiology subspecialty codes.

- AB: Abdominal Radiology
- BR: Breast Imaging
- CA: Cardiac Imaging
- CH: Chest/Thoracic Imaging
- ER: Emergency Radiology
- GI: Gastrointestinal Radiology
- GU: Genitourinary Radiology
- HN: Head & Neck Imaging
- IR: Interventional Radiology
- MI: Molecular Imaging/Nuclear Medicine
- MK: Musculoskeletal Radiology
- NR: Neuroradiology
- OB: OB/Gyn Radiology
- OI: Oncologic Imaging
- PD: Pediatric Radiology
- VI: Vascular Imaging
"""

# Etiology taxonomy - comprehensive list of finding etiology categories
ETIOLOGIES: list[str] = [
    "inflammatory:infectious",
    "inflammatory",
    "neoplastic:benign",
    "neoplastic:malignant",
    "neoplastic:metastatic",
    "neoplastic:potential",  # indeterminate lesions, incidentalomas
    "traumatic-acute",  # acute injury
    "post-traumatic",  # sequelae of prior injury
    "iatrogenic:post-operative",
    "iatrogenic:post-radiation",
    "iatrogenic:device",
    "iatrogenic:medication-related",
    "vascular:ischemic",
    "vascular:hemorrhagic",
    "vascular:thrombotic",
    "degenerative",
    "congenital",
    "metabolic",
    "toxic",
    "mechanical",  # obstruction, herniation, torsion
    "idiopathic",
    "normal-variant",
]
"""Comprehensive taxonomy of finding etiologies.

Includes hierarchical categories with colon-separated subtypes (e.g., inflammatory:infectious).
Multiple etiologies may apply to a single finding.
"""


class FindingEnrichmentResult(BaseModel):
    """Comprehensive enrichment metadata for an imaging finding.

    Contains structured metadata including ontology codes, anatomic classifications,
    etiologies, relevant modalities, and subspecialties. This result is designed for
    human review and validation before integration into finding models.
    """

    finding_name: str = Field(
        description="Name of the finding being enriched",
        min_length=1,
    )

    oifm_id: str | None = Field(
        default=None,
        description="Open Imaging Finding Model ID if the finding exists in the index (e.g., OIFM_AI_000001)",
    )

    snomed_codes: list[IndexCode] = Field(
        default_factory=list,
        description="SNOMED CT codes for this finding (disorder/finding codes only, excluding anatomic locations)",
    )

    radlex_codes: list[IndexCode] = Field(
        default_factory=list,
        description="RadLex codes for this finding (finding codes only, excluding anatomic locations)",
    )

    body_regions: Annotated[
        list[BodyRegion],
        Field(
            default_factory=list,
            description="Primary body regions where this finding occurs (from predefined BodyRegion list)",
        ),
    ]

    etiologies: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Etiology categories for this finding (from ETIOLOGIES taxonomy). "
            "Multiple categories may apply (e.g., a finding may be both infectious and vascular:thrombotic)",
        ),
    ]

    modalities: Annotated[
        list[Modality],
        Field(
            default_factory=list,
            description="Imaging modalities where this finding is typically visualized",
        ),
    ]

    subspecialties: Annotated[
        list[Subspecialty],
        Field(
            default_factory=list,
            description="Radiology subspecialties most relevant to this finding",
        ),
    ]

    anatomic_locations: list[OntologySearchResult] = Field(
        default_factory=list,
        description="Anatomic locations where this finding occurs (from ontology search with concept IDs and scores)",
    )

    enrichment_timestamp: datetime = Field(
        description="Timestamp when this enrichment was performed",
    )

    model_provider: str = Field(
        description="AI model provider used for enrichment (e.g., 'openai', 'anthropic')",
    )

    model_tier: str = Field(
        description="AI model tier used for enrichment (e.g., 'small', 'main', 'full')",
    )

    @field_validator("etiologies")
    @classmethod
    def validate_etiologies(cls, v: list[str]) -> list[str]:
        """Validate that all etiologies are in the allowed ETIOLOGIES list."""
        invalid = [etiology for etiology in v if etiology not in ETIOLOGIES]
        if invalid:
            raise ValueError(f"Invalid etiology categories: {invalid}. Must be from ETIOLOGIES list.")
        return v


async def lookup_finding_in_index(identifier: str) -> FindingModelFull | None:
    """Look up a finding model in the DuckDB index by OIFM ID or name.

    This function attempts to retrieve a finding model from the index, first trying
    to match the identifier as an OIFM ID (exact match), then falling back to name-based
    search if not found. The function properly manages DuckDB connection lifecycle
    using a context manager.

    Args:
        identifier: Either an OIFM ID (e.g., "OIFM_AI_000001") or a finding name
                   (e.g., "pneumonia"). Name matching is case-insensitive and
                   supports synonym matching.

    Returns:
        FindingModelFull object if found, None if not found.
        Returns None rather than raising an error when the finding doesn't exist,
        as missing findings are a normal case in the enrichment workflow.

    Raises:
        RuntimeError: If there are database connection issues or other system errors.
                     Does NOT raise on missing findings (returns None instead).

    Example:
        >>> # Lookup by OIFM ID
        >>> model = await lookup_finding_in_index("OIFM_AI_000001")
        >>> if model:
        ...     print(f"Found: {model.name}")
        ...
        >>> # Lookup by name
        >>> model = await lookup_finding_in_index("pneumonia")
        >>> if model is None:
        ...     print("Finding not in index")
    """
    logger.debug(f"Looking up finding in index: {identifier}")

    index = DuckDBIndex(read_only=True)

    try:
        async with index:
            # Use public API - get() handles OIFM ID resolution internally
            # Tries in order: OIFM ID match, name match, slug match, synonym match
            entry = await index.get(identifier)

            if entry is None:
                logger.debug(f"Finding not found in index: {identifier}")
                return None

            logger.debug(f"Resolved {identifier} to {entry.oifm_id}")

            # Get the full model
            model = await index.get_full(entry.oifm_id)
            logger.debug(f"Retrieved full model: {model.oifm_id} ({model.name})")
            return model

    except Exception as e:
        # Log database connection errors but re-raise for caller to handle
        logger.error(f"Error looking up finding in index: {e}")
        raise RuntimeError(f"Database error while looking up finding '{identifier}': {e}") from e
