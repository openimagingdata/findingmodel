# Metadata Enrichment Evaluation

Metadata evals are quality instruments. They should expose strengths, misses, and risky failure
patterns without pretending every case has one perfect score.

## Gates Versus Scores

Gates are pass/fail checks for whether an eval result is interpretable. Gates are not score
contributors.

Gate examples:

- execution succeeded;
- output parsed;
- schema was valid;
- required `entity_type` was present;
- candidate IDs were coherent;
- fill-blanks preservation rules held.

Quality scores measure metadata judgment after gates pass.

Gate B is the destructive-cleanup gate for the tool repo. Before deleting fallback or legacy
metadata paths, the configured split-agent `assign_metadata(...)` path must pass an end-to-end smoke
eval on a real case. Unit tests with mocked model calls are useful, but they do not satisfy Gate B.

## Required Fields

`entity_type` is the only required metadata field. Other fields may be null when the model cannot
justify a value.

This is deliberate. Optional fields should not be forced just to make the output look complete. For
etiology and time course, null can mean not applicable, genuinely indeterminate from the finding,
or dependent on an unresolved cause that is not part of the finding class itself.

## Scoring Policy

- Report headline metadata quality separately from gates.
- Report per-field scores and lowest-scoring cases.
- Score set fields with precision and recall.
- Penalize unsupported additions more heavily than omissions when additions create false grouping or
  clinical relationships.
- Treat forbidden values as severe field errors.
- Give conservative abstention credit when optional fields are justifiably null.
- Evaluate source-code carry-forward differently from new source-code creation. Existing index or
  anatomic codes may be likely correct even when the enrichment tool cannot independently prove
  them; newly added unsupported codes are more risky.

The concrete scoring functions and weights live in code, not here (see
[ADR-0001](../../adr/0001-lean-metadata-docs-schema-is-spec.md)): the scoring helpers (recall-
weighted set similarity, commission-sensitive scoring, abstention credit, ordinal duration
distance) are in `evals/metadata_scoring.py`, and the per-field aggregate weights are
`QUALITY_FIELD_WEIGHTS` in `evals/metadata_assignment.py`. This doc describes the approach; the
numbers are maintained in those files.

## Component Evals

Component evals test one assignment agent at a time. Current components include:

- ontology;
- anatomy;
- entity type;
- patient applicability;
- subspecialty domain;
- modality applicability;
- etiology/time-course.

Component evals are for prompt tuning and diagnosis. They are not a claim that every expected value
is final clinical truth.

## Current Eval Sources

The current eval set is assembled from several evidence levels:

- checked-in package gold fixtures;
- human-reviewed metadata records and approved-output snapshots;
- reviewed etiology/time-course cases;
- extracted candidate guidance from feedback records;
- bounded smoke/e2e cases that exercise the configured `assign_metadata(...)` path.

Only human review is authoritative. Feedback-derived candidate hints are not gold until humans
promote them or they are converted into general prompt/eval guidance.

Current known fixture counts:

- 35 checked-in metadata gold fixtures;
- 106 etiology/time-course component cases;
- 150 unique human-reviewed records in the review evidence register;
- 180 total human review events after targeted follow-up review;
- 67 latest-approved records;
- 83 latest-feedback records;
- 57 feedback-derived candidate records preserved for later review.

## Evidence And Fixture Lifecycle

**Ontology cache.** `OntologyLookupCache` (`metadata/ontology_cache.py`) is an optional
DuckDB-backed evidence store passed to `assign_metadata(ontology_cache=...)`. It is populated
incrementally during runs (`record_ontology_result` / `record_index_code`) as candidates are looked
up, and read back by the auditor's deterministic `_ontology_evidence_flags` for code/display
evidence. It is durable and accumulates across runs; the caller owns its path. There is no automatic
rebuild — refresh by pointing at a new path (or clearing the file) when ontology sources change.
Rebuild cadence and versioning are not yet formalized; a stale cache is evidence only, never
authority.

**Regression floor.** The data-repo regression floor
(`../findingmodels-metadata/evals/regression_floor/regression-floor-v1.json` + `manifest.json`) is a
fixed set of already-acceptable cases meant to catch drift on targeted reruns. It is currently
referenced as non-authoritative in `evals/metadata_review_artifact_inventory.py` and marked "to port
or replace" — it has not yet been adopted as a tool-repo gate. When adopted, every targeted rerun
should include it and a regression in any floor case should block the change.

## Current Results To Preserve

Recent bounded verification passed structural gates and produced these quality signals:

- `task evals:metadata:smoke`: gates passed on the configured `assign_metadata(...)` path; metadata
  quality was about 0.81 in the latest smoke result.
- `task evals:metadata`: gates passed on the bounded assignment suite; assignment quality was about
  0.80.
- Component averages in the bounded suite were approximately: ontology 1.00, anatomy 1.00, entity
  type 1.00, patient applicability 0.92, subspecialty domain 0.99, modality applicability 0.95, and
  etiology/time-course 0.75.

Older etiology/time-course expanded runs are still useful because they characterize the main
failure shape:

- the expanded fixture has 106 cases;
- repeated runs showed combined quality in the low-to-mid 0.70s after commission-sensitive scoring;
- the dominant miss class was unsupported extra etiologies on broad or descriptive findings;
- time-course modifier choice remained weaker than duration choice.

These numbers are not a readiness trophy. They say the harness runs and the weak fields are
visible.

## Known Failure Classes

Current recurring failure classes include:

- over-assigning etiologies to broad descriptive findings such as generic cysts, nodules, masses,
  fluid collections, density, signal, enhancement, or distribution patterns;
- inferring congenital/developmental etiology from pediatric age context alone;
- assigning time course to findings whose persistence depends on an unresolved underlying cause;
- adding too many modalities to findings that are not truly modality-specific;
- treating some age/anatomic-location expectations inconsistently in end-to-end assignment;
- carrying old expected values whose policy may need human adjudication.

The correct response is not to paste miss cases into prompts. The evals should identify a failure
class, then prompts or schemas should change only when the fix is a general rule.

## Current Use

Use evals to identify failure classes and decide whether prompt, schema, scoring, or gold data needs
attention. Do not patch prompts by copying missed eval cases into prompt text.

Before a broad corpus run, eval reporting must make these visible:

- gate pass/fail state;
- headline metadata quality;
- per-field quality;
- lowest-scoring cases;
- miss labels such as unsupported addition, omission, forbidden value, wrong subtype, and wrong
  duration/modifier;
- enough details output to review actual versus expected values.
