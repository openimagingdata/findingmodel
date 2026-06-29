# Index-Code And Anatomy Target Fixture Plan

Status: Complete
Date: 2026-06-23

## Goal

Make `index_codes` and `anatomic_locations` measurable against a human-reviewed target fixture,
while preserving non-searchable source codes that should not be rediscovered through BioOntology.

The immediate objective is eval quality, not prompt tuning. The readiness scorecard should expose
real misses in searchable SNOMED CT, RadLex, LOINC, and anatomy candidates after source-provided
GAMUTS/GMTS/RadElement/CDES codes have been carried forward deterministically.

## Background And Diagnosis

The development readiness set showed `index_codes` as the weakest metadata field. In the saved
June 4 details, the field averaged about `0.674`, below the `0.85` floor. The dominant early pattern
was under-recall: the enrichment path returned no codes, or returned some codes but missed codes
present in curator-approved records.

The first diagnosis separated misses into four buckets:

- coverage: a code system was not reachable by the ontology search;
- retrieval: the backend existed but did not return the relevant candidate;
- matching: the candidate was returned but not selected;
- abstention: the assignment path declined to return codes.

Coverage was the dominant initial issue. The ontology search path queried SNOMED CT, RadLex, and
LOINC, while many curator targets included RadElement `RDES...` identifiers. Adding only the
missing RDES targets to the saved June 4 artifact would have raised `index_codes` only to about
`0.736`, so RadElement coverage was necessary but not sufficient; matcher/abstention and searchable
code recall remained real problems.

The strategy changed after we separated two kinds of codes:

- source-provided non-searchable codes: `GAMUTS`, `GMTS`, RadElement/RDES, and `CDES`;
- searchable ontology codes: SNOMED CT, RadLex, LOINC, and local anatomic-location codes.

Because BioOntology is not the source of truth for the non-searchable source codes, the MVP should
not add a new search backend for them. The correct MVP behavior is to carry them forward from the
source `.fm.json` files and use ontology search only for systems it can actually retrieve.

## Implemented Approach

The work has three connected pieces.

First, metadata assignment now carries forward source `index_codes` from `GAMUTS`, `GMTS`,
`RADELEMENT`, and `CDES`, plus source codes with an `RDES` prefix. These are merged after ontology
decisions and after low-confidence filtering, so they do not depend on BioOntology search or the
ontology decision agent. The readiness-run blanking helper also preserves those carry-forward codes
instead of deleting all source `index_codes` before enrichment.

Second, a tolerant candidate-extraction pass generated review material for 67 records. The raw
artifact was useful provenance but not a good review surface: it included source paths, raw query
terms, and noisy search transcripts. The curated worksheet is the human-readable authority for this
MVP:

- `packages/findingmodel-ai/evals/index_code_candidate_review_curated_2026-06-05.txt`

The raw candidate dump is not the target. It should not be treated as gold and does not need to be
committed for the readiness overlay to be reproducible from human review.

Third, the curated worksheet is converted into a machine-readable target fixture:

- `packages/findingmodel-ai/evals/fixtures/index_code_review_targets.json`

The readiness runner applies that fixture as a scoring-time overlay for `index_codes` and
`anatomic_locations` only. It does not mutate `metadata_review_approved_outputs.json`, and it leaves
all other metadata fields on the existing approved-output and field-specific overlay paths.

## Reviewed Worksheet Audit

The curated worksheet audit on 2026-06-23 found:

- 67 finding sections;
- 128 reviewed `index_codes` targets in the generated fixture;
- 69 reviewed `anatomic_locations` targets in the generated fixture;
- intentionally empty `index_codes` targets for `abdominal_clips`, `basal_cistern_effacement`, and
  `breast_malignancy_risk`;
- intentionally empty `anatomic_locations` targets for `radiolucent_urinary_calculus` and
  `tunneled_catheter`;
- accepted index-code systems: `SNOMEDCT`, `RADLEX`, `LOINC`, `GAMUTS`, `GMTS`, `radelement`,
  `RADELEMENT`, and `CDES`;
- accepted anatomy system: `ANATOMICLOCATIONS` only.

One known anatomy gap remains. The worksheet has `RADLEX:RID9865 | basal cistern` under the
`basal cistern effacement` anatomy heading, but the local `anatomic_locations` index does not
currently contain basal, basilar, perimesencephalic, subarachnoid, or generic cistern entries. For
this MVP, we do not invent an `ANATOMICLOCATIONS` code. The converter records this as a known gap
and excludes the RadLex line from emitted anatomy targets.

## Source Artifacts And Commit Policy

The committed source of truth for reviewed code/anatomy targets should be the curated worksheet plus
the generated JSON fixture. The worksheet is human-readable and reviewable; the JSON fixture is what
the scorecard consumes.

The raw extraction file has unique provenance, but it is a noisy intermediate artifact. It is not
needed to reproduce the checked-in target fixture from the curated worksheet. If raw search
provenance becomes important later, commit it separately as an archived artifact with a clear reason
instead of bundling it into the MVP fixture commit.

The extraction utility is useful if we need to regenerate candidate worksheets, but it is separate
from the scoring overlay. Keeping the MVP commit focused on the curated worksheet, converter,
fixture, runner integration, tests, and docs avoids mixing the stable eval target with a one-off raw
candidate dump.

## Implementation Summary

Implemented files:

- `packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py`
- `packages/findingmodel-ai/evals/metadata_readiness_run.py`
- `packages/findingmodel-ai/evals/index_code_review_targets.py`
- `packages/findingmodel-ai/evals/fixtures/index_code_review_targets.json`
- `packages/findingmodel-ai/evals/index_code_candidate_review_curated_2026-06-05.txt`
- `packages/findingmodel-ai/tests/test_assign_metadata.py`
- `packages/findingmodel-ai/tests/test_index_code_review_targets.py`
- `docs/metadata/enrichment/evaluation.md`

The target converter:

- parses `##` headings, `Input`, `Synonyms`, `Index code candidates`, and
  `Anatomic location candidates`;
- maps each heading to `metadata_review_approved_outputs.json`;
- validates the expected 67 records;
- preserves `system`, `code`, and `display`;
- treats `[source]` as worksheet provenance only, not as part of code identity;
- rejects non-`ANATOMICLOCATIONS` anatomy targets except for the explicitly recorded basal-cistern
  known gap.

The readiness runner:

- loads reviewed code/anatomy targets once;
- overlays `index_codes` and `anatomic_locations` for matching records at scoring time;
- keeps GAMUTS/GMTS scoring exclusion behavior in `_score_codes`;
- emits Logfire spans for readiness runs and per-item scoring.

## Verification

Focused verification completed:

- `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.index_code_review_targets`
- `uv run pytest packages/findingmodel-ai/tests/test_index_code_review_targets.py packages/findingmodel-ai/tests/test_metadata_scoring.py packages/findingmodel-ai/tests/test_assign_metadata.py -q`
- `uv run ruff check packages/findingmodel-ai/evals/index_code_review_targets.py packages/findingmodel-ai/evals/metadata_readiness_run.py packages/findingmodel-ai/tests/test_index_code_review_targets.py packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py packages/findingmodel-ai/tests/test_assign_metadata.py`
- `uv run mypy`

Live development readiness run:

- command: `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_readiness_run --limit 100`
- records: 54 development findings;
- hard gates: pass;
- malignancy over-calls: 0;
- `index_codes`: `0.69` versus `0.85` floor;
- `anatomic_locations`: `0.77` versus `0.85` floor;
- Logfire trace:
  `https://logfire-us.pydantic.dev/talkasab/findingmodel/?q=trace_id%3D%27019ef6095af8629f9c07dbda56640736%27&since=2026-06-23T19%3A51%3A08.792906%2B00%3A00&until=2026-06-24T19%3A52%3A51.526156%2B00%3A00`

## Result And Next Work

The fixture, converter, and overlay are mechanically working. The reviewed targets are stricter than
the older approved-output values for code/anatomy, so the run now gives a clearer failure signal
rather than an inflated score.

The carry-forward fix solved the unreachable-source-code problem for non-searchable systems, but it
does not solve all `index_codes` misses. Remaining `index_codes` failures are mostly under-recall of
searchable SNOMED CT, RadLex, and LOINC codes after carry-forward succeeds, plus some abstention.
Remaining `anatomic_locations` failures are mostly granularity mismatches: broad parent locations
such as `lung`, `brain`, or `spine` where the reviewed target expects a more specific child
location, plus a few localizer choices that the hierarchy scorer does not treat as related.

The next tuning work should use the clearer target fixture to improve searchable-code recall and
anatomic-location specificity without weakening the other metadata fields.
