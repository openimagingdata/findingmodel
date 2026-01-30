# Team Update — 26 September 2025

## Highlights
- **Markdown deletion guard**: `edit_model_markdown()` now short-circuits when an edited document drops an existing attribute header or bullet. The model stays unchanged and the user receives a targeted rejection message instead of a silent mutation.
- **ID finalization on save**: `scripts/edit_finding_model.py` and the new command-focused demos finalize placeholder attribute IDs by calling `assign_real_attribute_ids()` before writing to disk, ensuring deterministic OIFMA/value codes in saved artifacts.
- **Documentation refresh**: README, CLAUDE.md, Copilot instructions, and the API contracts were updated with the new guardrails, demo scripts, and ID management workflow. CHANGELOG now tracks these updates under version 0.4.1.

## What To Know
- New attributes must continue to use `PLACEHOLDER_ATTRIBUTE_ID` during editing. Promotion to real IDs happens exclusively via `assign_real_attribute_ids()`.
- Markdown edits that omit existing headers will be rejected automatically. Encourage users to keep original sections intact when making partial edits.
- The interactive demo suite now includes: `scripts/edit_finding_model.py`, `scripts/ontology_concept_match.py`, and `scripts/anatomic_location_search.py`.

## Follow-Up Actions
- When touching the editing pipeline, run:
  - `uv run pytest test/test_model_editor.py -m "not callout"` for fast validation
  - `uv run pytest test/test_model_editor.py` if you need to exercise live callout coverage
- Confirm downstream tools that parse saved models tolerate the enriched changelog metadata added in v0.4.1.

## Lessons Learned
- Preflight validation catches most user error patterns without burning API tokens.
- Deferring ID assignment until just before persistence keeps the editing UX flexible while guaranteeing repository integrity.
