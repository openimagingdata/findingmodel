# Enrichment Agent Architecture

How the metadata-assignment pipeline is built: an orchestrator, seven focused assignment agents,
two search-agent pairs, and an auditor. This describes the design and responsibilities; field
types, enums, and exact scoring weights live in code (see [ADR-0001](../../adr/0001-lean-metadata-docs-schema-is-spec.md)).

Terms used here are defined in the repo-root `CONTEXT.md` (System Context).

## Overview

`assign_metadata()` (the **orchestrator**, `packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py`)
gathers ontology and anatomic candidates, runs the seven **assignment agents**, assembles their
decisions onto the finding model, and returns a `MetadataAssignmentResult`. It never decides a
field value itself. Each assignment agent has a lean external Markdown prompt
(`metadata/prompts/*.md`, loaded by `prompt_loader.py`) and emits a typed `*Decision`. All seven
run on the single `metadata_assign` model config. See
[ADR-0002](../../adr/0002-split-agent-assignment-architecture.md) for why this is split rather than
one broad classifier.

## Pipeline flow

`_run_focused_decisions()` runs the agents in three stages (`asyncio.gather` within each):

1. **Candidate gathering** — `_gather_ontology_candidates()` (BioOntology search) and
   `_gather_anatomic_candidates()` (anatomic index + location search). Candidate lists are bounded
   by `settings.metadata_candidate_decision_limit` (default 15).
2. **Batch 1 (parallel):** ontology, anatomy assignment agents (they consume the candidates).
3. **Batch 2 (parallel):** entity_type, patient_applicability, subspecialty_domain,
   modality_applicability.
4. **Batch 3:** etiology_tempo — runs after entity_type and is validated against its result
   (`_validate_etiology_tempo_decision`).
5. **Combine + assemble** — `_combine_focused_decisions()` merges the seven decisions, then
   `_assemble_fill_blanks()` or `_assemble_reassess()` applies them to the model.

## Assignment agents

Each owns part of the structured output and has a matching component eval (see
[evaluation.md](evaluation.md)). All use the `metadata_assign` config and live in `assignment.py`
as `create_*_agent`.

| Agent | Owns | Prompt | Output | Component eval |
| --- | --- | --- | --- | --- |
| entity_type | `entity_type` | `prompts/entity_type.md` | `EntityTypeDecision` | `evals/metadata_entity_type_decision.py` |
| etiology_tempo | `etiologies`, `expected_time_course` | `prompts/etiology_tempo.md` | `EtiologyTempoDecision` | `evals/metadata_etiology_tempo_decision.py` |
| patient_applicability | `age_profile`, `sex_specificity` | `prompts/patient_applicability.md` | `PatientApplicabilityDecision` | `evals/metadata_patient_applicability_decision.py` |
| subspecialty_domain | `subspecialties` | `prompts/subspecialty_domain.md` | `SubspecialtyDomainDecision` | `evals/metadata_subspecialty_domain_decision.py` |
| modality_applicability | `applicable_modalities` | `prompts/modality_applicability.md` | `ModalityApplicabilityDecision` | `evals/metadata_modality_applicability_decision.py` |
| ontology_decision | `index_codes` (selects canonical from ontology candidates) | `prompts/ontology_decision.md` | `OntologyDecision` | `evals/metadata_ontology_decision.py` |
| anatomy_decision | `anatomic_locations`, `body_regions` (selects from anatomic candidates) | `prompts/anatomy_decision.md` | `AnatomyDecision` | `evals/metadata_anatomy_decision.py` |

The ontology and anatomy agents are selection-style: they choose among candidates produced by the
search agents rather than generating values from scratch.

## Search agents

Candidate discovery only — they propose, they do not set final metadata. They have their own model
configs (separately tunable from `metadata_assign`):

- **Ontology** (`search/ontology.py`): query generator (`ontology_search`) expands the finding name
  into alternate ontology terms; categorization agent (`ontology_match`) sorts BioOntology hits into
  exact / should-include / marginal.
- **Anatomic** (`search/anatomic.py`): query generator (`anatomic_search`) produces a region +
  search terms; selection agent (`anatomic_select`) picks the primary location and alternates from
  the anatomic index.

## Auditor

`audit_enrichment()` (`metadata/auditor.py`) runs after assembly and only emits flags — it never
reassigns metadata. Two layers:

- **Deterministic** (`_deterministic_flags`): `_ontology_evidence_flags` (missing cache evidence;
  display mismatch vs cached preferred term) and `_anatomy_sex_specificity_flags` (anatomy/sex
  conflicts).
- **Optional LLM second opinion** (`create_enrichment_auditor_agent`): high-impact sanity checks
  (related/broader/narrower index codes, anatomy/region/modality/subspecialty contradictions,
  impossible age/sex/time-course, inappropriate etiologies). Enabled by default; skippable.

Flags carry severity (low/medium/high), field, message, and evidence.

## Output contract

Defined in `metadata/types.py` (the schema is the spec — do not restate types here):

- Only `entity_type` is required (`REQUIRED_METADATA_FIELDS = ("entity_type",)`); all other
  structured fields may be null when the finding does not justify a value.
- `field_confidence` is a validated `dict[ConfidenceFieldKey, FieldConfidenceScore]` (fixed key set;
  HIGH/MEDIUM/LOW, coerced from numeric).
- There is **no `clear_fields`** mechanism; in reassess mode the agents set a field to null directly.

## Modes

- **fill_blanks_only** — populate only currently-empty fields; existing populated fields are locked
  context (`_assemble_fill_blanks`).
- **reassess** — re-evaluate all structured fields; may replace or null existing values
  (`_assemble_reassess`).

## Model configuration

The seven assignment agents resolve `settings.get_agent_model("metadata_assign")`; the search
agents use `ontology_search` / `ontology_match` / `anatomic_search` / `anatomic_select`. Per-agent
overrides via `AGENT_MODEL_OVERRIDES__<tag>` and the fallback chains in
`data/supported_models.toml`. See `docs/configuration.md`.

## Where things live

```
packages/findingmodel-ai/src/findingmodel_ai/
├── metadata/
│   ├── assignment.py        # orchestrator, assignment agents, assembly, modes
│   ├── decisions.py         # *Decision output types
│   ├── types.py             # MetadataAssignmentResult/Review, field_confidence, required fields
│   ├── auditor.py           # deterministic + optional-LLM auditor
│   ├── prompt_loader.py     # loads prompts/*.md
│   └── prompts/*.md         # one lean prompt per assignment agent
└── search/
    ├── ontology.py          # ontology query-gen + categorization
    └── anatomic.py          # anatomic query-gen + selection
```
