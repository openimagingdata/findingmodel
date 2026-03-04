#!/usr/bin/env bash
# =============================================================================
# jq snippets for inspecting the anatomic locations source JSON
#
# These are reference one-liners — copy-paste as needed, not meant to be run
# as a single script. Replace FILE, TERM, and RIDXXXX with actual values.
# =============================================================================

FILE="notebooks/data/anatomic_locations_noembed.json"

# --- Counting ---

# Total entry count
jq 'length' "$FILE"

# Count lateralized (left) entries
jq '[.[] | select(._id | test("_RID5824$"))] | length' "$FILE"

# Count entries with SNOMED codes
jq '[.[] | select(has("snomedId"))] | length' "$FILE"

# --- Searching ---

# Search by description (case-insensitive)
jq '[.[] | select(.description | test("TERM"; "i"))]' "$FILE"

# Search by ID (exact match)
jq '[.[] | select(._id == "RIDXXXX")]' "$FILE"

# Search by ID pattern (regex)
jq '[.[] | select(._id | test("RID34566"))]' "$FILE"

# Search by SNOMED code
jq '[.[] | select(.snomedId == "46750007")]' "$FILE"

# Search synonyms
jq '[.[] | select(.synonyms? // [] | any(test("TERM"; "i")))]' "$FILE"

# --- Inspecting hierarchy ---

# Show containment for a specific entry
jq '.[] | select(._id == "RIDXXXX") | {_id, description, containedByRef, containsRefs}' "$FILE"

# Show all entries contained by a specific structure
jq '[.[] | select(.containedByRef?.id == "RIDXXXX")] | [.[] | {_id, description}]' "$FILE"

# Show all entries that contain a specific structure
jq '[.[] | select(.containsRefs? // [] | any(.id == "RIDXXXX"))] | [.[] | {_id, description}]' "$FILE"

# Show part-of relationships
jq '.[] | select(._id == "RIDXXXX") | {_id, description, partOfRef, hasPartsRefs}' "$FILE"

# --- Inspecting laterality ---

# Show entries with laterality refs (generic entries)
jq '[.[] | select(has("leftRef") and has("rightRef"))][0:5] | [.[] | {_id, description, leftRef, rightRef}]' "$FILE"

# Show left-lateralized entries
jq '[.[] | select(._id | test("_RID5824$"))][0:5] | [.[] | {_id, description, unsidedRef}]' "$FILE"

# Show right-lateralized entries
jq '[.[] | select(._id | test("_RID5825$"))][0:5] | [.[] | {_id, description, unsidedRef}]' "$FILE"

# --- Sampling ---

# Show first N entries
jq '.[0:3]' "$FILE"

# Show a random-ish entry with all fields
jq '.[42]' "$FILE"

# Show entries in a region
jq '[.[] | select(.region == "Thorax")][0:3] | [.[] | {_id, description}]' "$FILE"

# --- Bulk inspection ---

# All unique regions
jq '[.[].region] | unique' "$FILE"

# Entries missing SNOMED codes
jq '[.[] | select(has("snomedId") | not)] | length' "$FILE"

# Entries with definitions
jq '[.[] | select(has("definition"))] | length' "$FILE"
