# Laterality Conventions

## Compound ID Pattern

Lateralized anatomic locations use RadLex modifier RIDs appended to the base RID:

| Modifier | RID | Compound ID pattern |
|----------|-----|---------------------|
| Left | `RID5824` | `{base_RID}_RID5824` |
| Right | `RID5825` | `{base_RID}_RID5825` |

Example: Pulmonary hilum (RID34566)
- Generic: `RID34566`
- Left: `RID34566_RID5824`
- Right: `RID34566_RID5825`

## Three-Entry Pattern

For every lateralized structure, create **three entries**:

### 1. Generic (unsided) entry

Has both `leftRef` and `rightRef` pointing to the lateralized variants.

```json
{
  "_id": "RID34566",
  "description": "pulmonary hilum",
  "snomedId": "46750007",
  "snomedDisplay": "Structure of hilum of lung",
  "containedByRef": {"id": "RID1301", "display": "lung"},
  "leftRef": {"id": "RID34566_RID5824", "display": "left pulmonary hilum"},
  "rightRef": {"id": "RID34566_RID5825", "display": "right pulmonary hilum"},
  "synonyms": ["hilum", "lung hilum", "hilum of lung"],
  "definition": "..."
}
```

### 2. Left entry

- ID: `{base}_RID5824`
- Has `rightRef` (counterpart) + `unsidedRef` (back to generic)
- `containedByRef` should point to the **left** version of the container (if it exists)

```json
{
  "_id": "RID34566_RID5824",
  "description": "left pulmonary hilum",
  "snomedId": "1650005",
  "snomedDisplay": "Structure of hilum of left lung",
  "containedByRef": {"id": "RID1326", "display": "left lung"},
  "rightRef": {"id": "RID34566_RID5825", "display": "right pulmonary hilum"},
  "unsidedRef": {"id": "RID34566", "display": "pulmonary hilum"},
  "synonyms": ["left hilum", "left lung hilum", "hilum of left lung"]
}
```

### 3. Right entry

- ID: `{base}_RID5825`
- Has `leftRef` (counterpart) + `unsidedRef` (back to generic)
- `containedByRef` should point to the **right** version of the container (if it exists)

```json
{
  "_id": "RID34566_RID5825",
  "description": "right pulmonary hilum",
  "snomedId": "88838000",
  "snomedDisplay": "Structure of hilum of right lung",
  "containedByRef": {"id": "RID1302", "display": "right lung"},
  "leftRef": {"id": "RID34566_RID5824", "display": "left pulmonary hilum"},
  "unsidedRef": {"id": "RID34566", "display": "pulmonary hilum"},
  "synonyms": ["right hilum", "right lung hilum", "hilum of right lung"]
}
```

## How the Build System Interprets Laterality

The `determine_laterality()` function in `oidm_maintenance/anatomic/build.py` uses this logic:

| Refs present | Laterality value | Explanation |
|-------------|-----------------|-------------|
| `leftRef` AND `rightRef` | `"generic"` | Has both sides → is the generic form |
| `leftRef` only | `"right"` | Points to left counterpart → this IS the right variant |
| `rightRef` only | `"left"` | Points to right counterpart → this IS the left variant |
| `unsidedRef` only | `"generic"` | Maps to generic |
| None | `"nonlateral"` | Not a lateralized structure |

**Key insight**: The ref points to the *counterpart*, not to self. A left entry has `rightRef` (its counterpart) and `unsidedRef` (its generic parent). It does NOT have `leftRef` — because it *is* the left variant.

## Containment for Lateralized Entries

When the containing structure is itself lateralized, each variant should point to the matching side:

- Generic entry → contained by the generic container (e.g., `lung`)
- Left entry → contained by the left container (e.g., `left lung`)
- Right entry → contained by the right container (e.g., `right lung`)

If the containing structure is not lateralized (e.g., "mediastinum"), all three entries point to the same container.

## Synonyms for Lateralized Entries

Follow a consistent pattern:

- Generic: `["hilum", "lung hilum", "hilum of lung"]`
- Left: `["left hilum", "left lung hilum", "hilum of left lung"]`
- Right: `["right hilum", "right lung hilum", "hilum of right lung"]`
