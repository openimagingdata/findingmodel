# Anatomic Locations Source JSON — Field Reference

The source file is `notebooks/data/anatomic_locations_noembed.json`. It is a JSON array of objects, each representing an anatomic location.

## Required Fields

### `_id` (string)
The RadLex RID identifier. For lateralized structures, a compound ID is used (see `laterality-conventions.md`).

```json
"_id": "RID10021"
"_id": "RID34566_RID5824"
```

### `description` (string)
The display name. Use lowercase, radiologist-friendly terminology.

```json
"description": "hypopharynx"
"description": "left pulmonary hilum"
```

## Code Fields

### `snomedId` (string, optional but strongly encouraged)
SNOMED CT concept ID. **Always use the "Structure of..." concept** from the SEP triad (see SNOMED guidance below).

```json
"snomedId": "46750007"
```

### `snomedDisplay` (string)
The SNOMED preferred term. Should match the official SNOMED display name.

```json
"snomedDisplay": "Structure of hilum of lung"
```

### `acrCommonId` (string, optional)
ACR Common Data Element identifier.

```json
"acrCommonId": "4013944"
```

### `codes` (array, optional)
Additional coding system references (FMA, MESH, UMLS, etc.).

```json
"codes": [
  {"system": "FMA", "code": "54880"},
  {"system": "MESH", "code": "A14.724.490"},
  {"system": "UMLS", "code": "C0014540"}
]
```

## SNOMED Coding Guidance — The SEP Triad

SNOMED CT uses a Structure-Entire-Part (SEP) triple for anatomy. Each anatomic concept may have up to 3 related concepts:

| Type | Example | When to use |
|------|---------|-------------|
| **Structure of...** | "Structure of hilum of lung" (46750007) | **Always use this one** |
| Entire... | "Entire hilum of lung" | Never — reserved for whole organ |
| ...part | "Part of hilum of lung" | Never — for sub-portions |

The "Structure" concept is what SNOMED intends for finding sites and procedure sites in clinical coding.

**Lateralized SNOMED codes**: These may not exist for every structure. If a lateralized form is not available in SNOMED, use the unsided Structure code for all three entries (generic, left, right) and note this in the PR description.

## Hierarchy Fields

### `containedByRef` (object)
The structure that **spatially contains** this one. Format: `{id, display}`.

```json
"containedByRef": {"id": "RID1301", "display": "lung"}
```

**When to use**: For spatial containment. "The hilum is contained by the lung." Most entries should have this.

### `containsRefs` (array of objects)
Structures that this entry **spatially contains**. Format: `[{id, display}, ...]`.

```json
"containsRefs": [
  {"id": "RID974", "display": "pulmonary artery"},
  {"id": "RID34918", "display": "main bronchus"}
]
```

**When to use**: When the structure is a container for other indexed structures. Not all entries need this — only add it when there are indexed structures that are contained within.

### `partOfRef` (object)
The larger structure this is a **part of** (mereological, not spatial).

```json
"partOfRef": {"id": "RID94", "display": "gastrointestinal tract"}
```

**When to use**: For part-whole relationships that aren't spatial containment. "The stomach is part of the gastrointestinal tract."

### `hasPartsRefs` (array of objects)
Structures that are **parts of** this entry.

```json
"hasPartsRefs": [
  {"id": "RID122", "display": "pylorus"},
  {"id": "RID116", "display": "gastric fundus"}
]
```

### `containedByRef` vs `partOfRef`

- **containedByRef**: spatial — "X is inside Y" (hilum is inside the lung)
- **partOfRef**: mereological — "X is a component of Y" (stomach is part of the GI tract)
- Many entries have one or both. Use containedByRef for the immediate spatial container, partOfRef for the functional/anatomical system.

## Laterality Fields

### `leftRef` / `rightRef` (object)
References to the lateralized variants. Only present on the **generic (unsided)** entry.

```json
"leftRef": {"id": "RID34566_RID5824", "display": "left pulmonary hilum"},
"rightRef": {"id": "RID34566_RID5825", "display": "right pulmonary hilum"}
```

### `unsidedRef` (object)
Back-reference to the generic entry. Only present on **left** and **right** entries.

```json
"unsidedRef": {"id": "RID34566", "display": "pulmonary hilum"}
```

See `laterality-conventions.md` for the full pattern.

## Text Fields

### `region` (string)
The body region. Common values: `"Head"`, `"Neck"`, `"Thorax"`, `"Abdomen"`, `"Pelvis"`, `"Upper Extremity"`, `"Lower Extremity"`, `"Spine"`.

```json
"region": "Thorax"
```

### `synonyms` (array of strings, optional)
Clinical shorthand that radiologists use. Include common abbreviations and alternative names.

```json
"synonyms": ["hilum", "lung hilum", "hilum of lung"]
```

### `definition` (string, optional but encouraged)
A brief anatomical definition. Prefer authoritative sources:
1. Fleischner Society glossary (for thoracic anatomy)
2. RadLex definitions
3. MeSH definitions
4. FMA definitions

Cite the source in brackets at the end.

```json
"definition": "The site on the medial aspect of the lung where the vessels and bronchi enter and leave the lung. [Fleischner Society]"
```

## Complete Entry Example

Here is the hypopharynx entry, annotated:

```json
{
  "_id": "RID10021",                          // RadLex RID
  "acrCommonId": "4013944",                   // ACR Common ID (optional)
  "snomedId": "81502006",                     // SNOMED "Structure of..." code
  "snomedDisplay": "Hypopharyngeal structure", // SNOMED display name
  "description": "hypopharynx",               // Display name (lowercase)
  "region": "Neck",                           // Body region
  "containedByRef": {                         // Spatial container
    "id": "RID7488",
    "display": "neck"
  },
  "synonyms": ["laryngopharynx"],             // Alternative names
  "partOfRef": {                              // Part-of relationship
    "id": "RID13211",
    "display": "pharynx"
  },
  "codes": [                                  // Additional codes
    {"system": "FMA", "code": "54880"},
    {"system": "MESH", "code": "A14.724.490"},
    {"system": "UMLS", "code": "C0014540"}
  ],
  "definition": "The portion of the pharynx between the inferior portion of the oropharynx and the larynx. [MeSH]"
}
```
