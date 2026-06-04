# Time-Course Prompt Tuning Plan

Status: Complete
Date: 2026-06-02

## Goal

Tune `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/etiology_tempo.md` to reduce
systematic over-abstention on `expected_time_course` while preserving null for genuine
non-persistent concepts.

## Scope

- Edit prompt text only.
- Do not change code, schemas, enum values, fixtures, or eval data.
- Do not paste active eval finding names or answers into the prompt.
- Keep the edits principle-based and small.

## Steps

1. [x] Review the handoff evidence and current durable field guidance.
2. [x] Add prompt rules that clarify when null etiology can still have a committed time course.
3. [x] Add compact duration-anchor refinements without turning the prompt into a case taxonomy.
4. [x] Tighten modifier selection so acute changing processes can use both `evolving` and `resolving`
   when both are characteristic, while stable/progressive choices stay distinct.
5. [x] Verify the diff contains no eval-case names and run a targeted non-callout check if
   available.

## Verification

- `rg` check for the handoff's development finding names in the prompt and plan returned no matches.
- `uv run --package findingmodel-ai pytest packages/findingmodel-ai/tests/test_metadata_etiology_tempo_eval.py -rs -m "not callout"`: 2 passed.
- Prompt growth after tightening: 96 to 101 lines, 887 to 953 words, 6,870 to 7,307 bytes.

## Alternate Review Candidate

Added `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/etiology_tempo.alternate.md` as
a non-active sibling prompt for review before scaling. It keeps the same task but changes the shape:
core rules are front-loaded, cyst/FDG/lymphadenopathy guidance stays in the middle as illustrative
specifics, and a final checklist restates the highest-priority constraints. It is 91 lines, 773
words, and 5,934 bytes. A case-name leak check against the handoff examples returned no matches.
