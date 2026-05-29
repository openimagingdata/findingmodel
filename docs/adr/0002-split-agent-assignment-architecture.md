# Split-agent assignment architecture

Metadata assignment is performed by seven focused **assignment agents** (entity_type,
etiology_tempo, patient_applicability, subspecialty_domain, modality_applicability,
ontology_decision, anatomy_decision), each with a lean external prompt and each emitting a typed
`*Decision`, all running on the single `metadata_assign` model config and assembled by
`assign_metadata()`. We chose this over one broad classifier because focused agents allow
concise/clean prompts, per-field evaluation, and targeted tuning of weak fields.

## Consequences

- More agents and orchestration: agents run in parallel batches, with etiology_tempo depending on
  entity_type; more LLM calls per finding than a single classifier.
- Each agent gets its own component eval; field quality is measured and tuned independently.
- Search agents (ontology, anatomic) feed candidates to the two selection-style assignment agents.
