"""Anatomic location enumeration types."""

from __future__ import annotations

from enum import Enum


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
