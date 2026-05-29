# Metadata Enrichment Working-Tree Inventory

Status: Active inventory for cleanup/readiness plan
Date: 2026-05-24

This inventory supports
`docs/plans/metadata-enrichment-current-plan.md`.

## Summary

Two repositories are involved:

- Tool/package repo: `/Users/talkasab/repos/findingmodel-metadata`
- Corpus/data repo: `/Users/talkasab/repos/findingmodels-metadata`

Current dirty state:

- Tool repo: 31 modified, 3 deleted, 32 untracked paths.
- Data repo: 328 modified and 34 untracked paths.

The dominant risk is not the number of files. The risk is mixing four different kinds of work:

- authoritative human-review evidence;
- tool/eval implementation changes;
- historical plan/run-result documentation;
- generated corpus source output that is not authoritative.

## Tool Repo: Keep With Metadata Cleanup

These paths appear to be part of the metadata enrichment cleanup and should stay in the current
workstream unless a later review finds a specific issue.

Planning and documentation:

- `CHANGELOG.md`
- `Taskfile.yml`
- `docs/metadata-eval-scoring.md`
- `docs/metadata-legacy-classifier-rules.md`
- `docs/reviews/`
- `docs/plans/metadata-enrichment-current-plan.md`
- `docs/plans/metadata-enrichment-plan-history-2026-05-24.md`
- superseded metadata-enrichment/prompt/eval plan files under `docs/plans/`

Metadata eval and prompt implementation:

- `packages/findingmodel-ai/evals/README.md`
- `packages/findingmodel-ai/evals/fixtures/`
- `packages/findingmodel-ai/evals/gold/*.fm.json`
- `packages/findingmodel-ai/evals/metadata_anatomy_decision.py`
- `packages/findingmodel-ai/evals/metadata_assignment.py`
- `packages/findingmodel-ai/evals/metadata_entity_type_decision.py`
- `packages/findingmodel-ai/evals/metadata_etiology_tempo_decision.py`
- `packages/findingmodel-ai/evals/metadata_modality_applicability_decision.py`
- `packages/findingmodel-ai/evals/metadata_ontology_decision.py`
- `packages/findingmodel-ai/evals/metadata_patient_applicability_decision.py`
- `packages/findingmodel-ai/evals/metadata_review_approved_outputs.py`
- `packages/findingmodel-ai/evals/metadata_review_artifact_inventory.py`
- `packages/findingmodel-ai/evals/metadata_review_evidence_register.py`
- `packages/findingmodel-ai/evals/metadata_review_expected_candidates.py`
- `packages/findingmodel-ai/evals/metadata_review_feedback_summary.py`
- `packages/findingmodel-ai/evals/metadata_review_source_overlap.py`
- `packages/findingmodel-ai/evals/metadata_scoring.py`
- `packages/findingmodel-ai/evals/metadata_subspecialty_domain_decision.py`

Metadata package implementation:

- `packages/findingmodel-ai/src/findingmodel_ai/metadata/__init__.py`
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py`
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/decisions.py`
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/ontology_cache.py`
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompt_loader.py`
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/`
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/types.py`

Metadata tests:

- `packages/findingmodel-ai/tests/test_assign_metadata.py`
- `packages/findingmodel-ai/tests/test_metadata_etiology_tempo_eval.py`
- `packages/findingmodel-ai/tests/test_metadata_prompt_repair_pilot.py`
- `packages/findingmodel-ai/tests/test_metadata_review_evidence_register.py`
- `packages/findingmodel-ai/tests/test_metadata_scoring.py`

Legacy removals that need review before finalizing:

- `packages/findingmodel-ai/evals/metadata_etiology_decision.py`
- `packages/findingmodel-ai/evals/metadata_imaging_workflow_decision.py`
- `packages/findingmodel-ai/tests/test_assign_metadata_modes.py`

## Tool Repo: Move Or Keep Separate

These paths should not be bundled into the metadata cleanup commit without an explicit decision:

- `.codex/config.toml`
- `CURRENT_PROGRESS_LOG.md`
- `docs/plans/add-openrouter-provider-2026-05-19.md`
- `notebooks/data/brain_volumetry_anatomic_code_display_backfill_2026-05-10.csv`

## Tool Repo: Remove

Generated Python cache files under the new prompt directory were removed:

- `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/__pycache__/`

## Data Repo: Keep With Metadata Cleanup

These are tooling/provenance changes that should remain available until Slice 4 review decides their
final disposition:

- `docs/metadata-enrichment-setup.md`
- `docs/plans/metadata-enrichment-plan-history-2026-05-24.md`
- superseded metadata-enrichment plan/run-result files under `docs/plans/`
- `scripts/metadata_assign_batch.py`
- `scripts/metadata_audit.py`
- `scripts/metadata_ingest_review.py`
- `scripts/metadata_review_package.py`
- `scripts/metadata_review_template.html`
- `scripts/metadata_apply_dry_run_outputs.py`
- `scripts/metadata_collate_review_decisions.py`
- `scripts/metadata_compare_clean_rerun.py`
- `scripts/metadata_compare_regression_floor.py`
- `scripts/metadata_feedback_coverage.py`
- `evals/regression_floor/README.md`
- `evals/regression_floor/manifest.json`
- `evals/regression_floor/regression-floor-v1.json`

Raw `.metadata-runs/` artifacts are not listed in `git status`, but they are critical local
provenance and must be harvested or copied before any fresh-checkout-dependent process is claimed to
be complete.

## Data Repo: Quarantine Before Clearing

These generated corpus source outputs are not authoritative and must not be committed as-is:

- 160 modified `defs/*.fm.json` files.
- 160 modified `text/*.md` files.
- `index.md`.

Slice 4 resolution as of 2026-05-25:

- The full generated corpus source/text/index diff set was quarantined at
  `../findingmodels-metadata/.metadata-runs/slice4-quarantine-2026-05-25/`.
- The unapproved generated corpus diffs were cleared from the active data-repo branch.
- The approved-output application command re-landed 67 latest-approved records through
  `../findingmodels-metadata/scripts/metadata_apply_approved_outputs.py`.
- The 10 modified data-repo definitions that were not in the human review register were not
  metadata-approved; their generated semantic metadata was cleared. Nine of those plus one feedback
  record needed deterministic display-label backfills for existing model-level codes so the
  metadata-aware validator could run. Those display-only repairs are tracked in
  `../findingmodels-metadata/.metadata-runs/slice4-code-display-backfill-2026-05-25/report.json`.

Before these are cleared from the active branch, Gate A requires the tool-repo review evidence
register to preserve enough latest-approved human-reviewed evidence to reproduce the 67
latest-approved source changes through the approved-output path.

Gate A preservation is now satisfied by the review evidence register and approved-output snapshot:
67 latest-approved records are preserved from reviewed payload artifacts, not dirty source
definitions.

## Immediate Next Cleanup Actions

1. Keep the 57 expected-metadata candidate records explicit as unresolved until humans promote or
   reject them.
2. Treat the Slice 4 applied data-repo source set as 67 approved metadata records plus 11
   display-only validation repairs. Do not reintroduce the quarantined feedback/unreviewed generated
   metadata.
3. Gate B is satisfied for Slice 3. Fallback/legacy tool paths already removed from this branch
   should stay deleted unless the approved-output path exposes a concrete regression.
