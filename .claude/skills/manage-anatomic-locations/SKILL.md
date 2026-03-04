---
name: manage-anatomic-locations
description: Add new anatomic locations or make structural edits (re-parenting, containment changes) to the anatomic locations source JSON. Use when adding missing anatomy, fixing hierarchy, or enriching entries with codes.
argument-hint: "[term-or-task] (e.g., 'add hilum' or 'fix renal containment')"
disable-model-invocation: true
allowed-tools: Bash(jq *), Bash(uv run *), Bash(python *), Read, Grep, Glob, WebSearch, WebFetch
---

# Manage Anatomic Locations

Add entries, re-parent structures, or enrich with codes in the anatomic locations source JSON (`notebooks/data/anatomic_locations_noembed.json`). This skill guides you through the full workflow: research, examine existing data, draft entries, apply changes, and verify.

**Reference files:**
- `reference/json-schema.md` — Source JSON field reference and conventions
- `reference/laterality-conventions.md` — Laterality compound ID patterns

**Helper scripts** (in `scripts/`):
- `bioontology_lookup.py` — Search RadLex + SNOMED for a term
- `sample_entries.sh` — jq snippets to inspect the source JSON
- `validate_entries.py` — Validate entries against schema and referential integrity

The source JSON path is: `notebooks/data/anatomic_locations_noembed.json`

---

## Phase 1: Research the anatomy

Before touching the JSON, build a clear picture of the structure you're adding or modifying.

### 1a. BioOntology lookup

Run the lookup script to find RadLex IDs and SNOMED codes:

```bash
uv run --env-file .env python .claude/skills/manage-anatomic-locations/scripts/bioontology_lookup.py "hilum of lung"
```

### 1b. SNOMED code selection — the SEP triad

SNOMED CT uses a Structure-Entire-Part (SEP) triple for anatomy:

- **Structure of...** (e.g., "Structure of hilum of lung" / 46750007) — **always prefer this**. This is what clinical coding uses for finding sites and procedure sites.
- **Entire...** — reserved for the whole organ. Do not use.
- **...part** — for sub-portions. Do not use.

The BioOntology lookup script flags the SEP type for each SNOMED hit. Lateralized SNOMED codes may not exist for every structure — that's OK, use the unsided code and note the gap.

### 1c. Web research

Search to verify anatomical relationships:

- What **contains** this structure? (containedByRef)
- What does it **contain**? (containsRefs)
- Is it **part of** a larger structure? (partOfRef)
- Is it **lateralized** (left/right)?
- What's the correct **clinical name** radiologists use?

Cite sources: Radiopaedia, Kenhub, Fleischner glossary, RadLex.

### 1d. Key question checklist

Before proceeding, answer:
- [ ] What is the RadLex RID? (This becomes the `_id`)
- [ ] What is the SNOMED code? (Prefer "Structure of..." concept)
- [ ] What contains this structure?
- [ ] What does it contain?
- [ ] Is it lateralized? If so, do lateralized SNOMED codes exist?
- [ ] What synonyms do radiologists actually use?
- [ ] Is there a published definition (Fleischner, RadLex, MeSH)?

---

## Phase 2: Examine existing index

### 2a. Search for existing entries

Check if entries already exist or are related:

```bash
uv run anatomic-locations search "<term>"
```

### 2b. Inspect related entries in the source JSON

Use jq to examine the neighborhood. See `scripts/sample_entries.sh` for common patterns.

```bash
# Search by description
jq '[.[] | select(.description | test("TERM"; "i"))]' notebooks/data/anatomic_locations_noembed.json

# Search by ID
jq '[.[] | select(._id == "RIDXXXX")]' notebooks/data/anatomic_locations_noembed.json

# Check containment for a structure
jq '.[] | select(._id == "RIDXXXX") | {_id, description, containedByRef, containsRefs}' notebooks/data/anatomic_locations_noembed.json
```

### 2c. Identify re-parenting needs

Look for structures that should gain or lose containment/part-of references because of the new entry. For example, if adding "pulmonary hilum", the pulmonary arteries and bronchi might need their `containedByRef` updated to point to the hilum rather than directly to the lung.

Check `containsRefs`, `containedByRef`, `partOfRef`, and `hasPartsRefs` for all affected neighbors.

---

## Phase 3: Draft the JSON entries

### 3a. Follow conventions

Read the field reference: `reference/json-schema.md`

For lateralized structures, read: `reference/laterality-conventions.md`

### 3b. Lateralized structures — create three entries

For each lateralized structure, create:

1. **Generic (unsided)** — has `leftRef` and `rightRef`, uses unsided SNOMED code
2. **Left** — ID is `{base}_RID5824`, has `rightRef` + `unsidedRef`
3. **Right** — ID is `{base}_RID5825`, has `leftRef` + `unsidedRef`

### 3c. Include essential fields

- `_id`: RadLex RID (or compound ID for lateralized)
- `description`: display name (lowercase, radiologist-friendly)
- `region`: body region (e.g., "Thorax", "Abdomen", "Head")
- `containedByRef`: `{id, display}` — the structure that contains this one
- `snomedId` + `snomedDisplay`: SNOMED code and display name
- `synonyms`: clinical shorthand radiologists actually use
- `definition`: prefer Fleischner Society or RadLex definitions

### 3d. Validate before applying

Run the validation script on the full file after mentally staging your additions:

```bash
python .claude/skills/manage-anatomic-locations/scripts/validate_entries.py notebooks/data/anatomic_locations_noembed.json
```

Or validate specific IDs:

```bash
python .claude/skills/manage-anatomic-locations/scripts/validate_entries.py notebooks/data/anatomic_locations_noembed.json --ids RID34566,RID34566_RID5824,RID34566_RID5825
```

---

## Phase 4: Apply changes with jq

### 4a. Write a jq filter

Build a jq filter that both adds new entries and modifies existing entries (e.g., updating `containedByRef` on structures being re-parented).

### 4b. Dry-run first

Show only the new/modified entries for user review before applying:

```bash
# Preview: show what would be added/changed
jq '[... filter ...] | [.[] | select(._id == "NEW_ID" or ._id == "MODIFIED_ID")]' notebooks/data/anatomic_locations_noembed.json
```

### 4c. Apply atomically

Write to a temp file, then move (atomic replacement):

```bash
jq '[... filter ...]' notebooks/data/anatomic_locations_noembed.json > /tmp/anatomic_locations_new.json
# Verify the temp file looks correct
jq 'length' /tmp/anatomic_locations_new.json
# Then replace
mv /tmp/anatomic_locations_new.json notebooks/data/anatomic_locations_noembed.json
```

---

## Phase 5: Verify and confirm

### 5a. Check counts

```bash
jq 'length' notebooks/data/anatomic_locations_noembed.json
```

Confirm the count changed by the expected number of entries (e.g., +3 for a lateralized triplet).

### 5b. Spot-check entries

```bash
jq '[.[] | select(._id | test("NEW_RID"))]' notebooks/data/anatomic_locations_noembed.json
```

### 5c. Run validation

```bash
python .claude/skills/manage-anatomic-locations/scripts/validate_entries.py notebooks/data/anatomic_locations_noembed.json
```

### 5d. Rebuild database (optional)

Ask the user if they want to rebuild the database:

```bash
uv run oidm-maintain anatomic build
```

Note: This requires an OpenAI API key for generating embeddings. The `.env` file must have `OPENAI_API_KEY` set.

### 5e. Verify search

After rebuild, verify the new entries are searchable:

```bash
uv run anatomic-locations search "<new term>"
```
