# Metadata Enrichment

This context covers two vocabularies: the **system** that proposes and applies structured metadata
(the agent pipeline), and the radiology **domain** it operates on (finding models and their
metadata).

---

# System Context

The metadata-enrichment system proposes structured metadata for finding models using a set of
focused AI agents, preserves human-review evidence, and applies only reviewed changes to the corpus.

## Language

**Enrichment**:
The whole process — assignment, human review, writeback, and database publication.
_Avoid_: tagging (collides with the `tags` field)

**Assignment**:
The agent step within enrichment that proposes structured metadata field values for one finding model.
_Avoid_: classification (the old single-classifier framing)

**Assignment agent**:
One of the seven focused agents that each decide part of the structured metadata output; all run on the `metadata_assign` model config.
_Avoid_: decision agent, decision surface, rationalized prompt surface, sub-agent

**Search agent**:
A candidate-discovery agent that proposes ontology or anatomic candidates for the assignment agents to choose from; it does not set final metadata.
_Avoid_: lookup agent

**Auditor**:
A post-assembly checker (deterministic checks plus an optional LLM second-opinion pass) that emits flags for human review and changes nothing.
_Avoid_: validator (validators enforce structure; the auditor is advisory)

**Orchestrator**:
`assign_metadata()` — gathers candidates, runs the assignment agents, and assembles their decisions. It never decides a field value itself.

**Decision**:
The structured output of a single assignment agent (e.g., `EntityTypeDecision`); the seven aggregate into `MetadataAssignmentDecision`.

## Relationships

- The **Orchestrator** runs seven **Assignment agents** and assembles their **Decisions**.
- **Search agents** feed candidates to the ontology and anatomy **Assignment agents** only.
- The **Auditor** runs after assembly and emits flags; it does not change metadata.
- **Enrichment** contains **Assignment** as its first step, followed by human review, writeback, and database publication.

## Example dialogue

> **Dev:** "Does the subspecialty assignment agent pick from search-agent candidates?"
> **Maintainer:** "No — only the ontology and anatomy assignment agents consume search candidates. Subspecialty, modality, entity_type, etiology/tempo, and patient applicability decide from the finding's own content."

## Flagged ambiguities

- "decision surface" / "rationalized prompt surface" were used for what we now call **Assignment agent** — resolved: use "assignment agent."
- "sub-agent" collides with the harness's Explore/Plan subagents — resolved: use "assignment agent" / "search agent" in docs.
- "tagging" collides with the `tags` field — resolved: the process is **Enrichment**, the step is **Assignment**.

---

# Domain Context

The radiology vocabulary the system operates on: finding models and the structured metadata fields
that describe them. Several of these terms carry a project-specific meaning that differs from
casual clinical usage.

## Language

**Finding model**:
A structured definition of one radiology entity (`FindingModelBase` / `FindingModelFull`) — its name, description, attributes, and structured metadata.
_Avoid_: using bare "finding" for the whole model.

**Finding**:
The `entity_type` value for an imaging observation that needs further characterization — what is seen, not concluded.

**Diagnosis**:
The `entity_type` value for a specific pathologic entity with defined criteria — what is concluded from what is seen.

**Entity type**:
The semantic category of a finding model: finding, diagnosis, grouping, measurement, assessment, recommendation, or technique_issue. The finding-vs-diagnosis split is the most important.

**Structured metadata**:
The enrichment-assigned fields on a finding model: body_regions, subspecialties, applicable_modalities, etiologies, expected_time_course, age_profile, sex_specificity, entity_type, index_codes, anatomic_locations.

**Subspecialty**:
A radiology reader-domain (RSNA-aligned code) whose reports are concerned with the finding — the expertise to interpret it.
_Avoid_: ordering specialty (who requests the study).

**Etiology**:
A common process type that produces the finding (e.g., `neoplastic:malignant`), not an exhaustive list of possible causes.
_Avoid_: differential diagnosis.

**Expected time course**:
How long the finding stays observable on imaging and how it changes — the imaging-observable window.
_Avoid_: the clinical duration of the underlying disease.

**Sex specificity**:
Whether the finding is anatomically restricted to one sex.
_Avoid_: prevalence or sex predilection (more common in ≠ specific to).

**Age profile**:
Two parts — `applicability` (the age window the finding can occur in) and `more_common_in` (where incidence peaks).

**Index code**:
An ontology code (SNOMED CT, RadLex, LOINC, …) that is an exact match or clinically substitutable equivalent for the finding.
_Avoid_: broader, narrower, or merely related codes (those are review evidence, not index codes).

**Anatomic location**:
A RadLex-derived code (the `ANATOMICLOCATIONS` system) for where the finding is located.

**Attribute**:
A structured data element a radiologist fills out to characterize the finding in a report (e.g., severity, size, laterality) — authored content, not enrichment metadata.

## Relationships

- A **Finding model** has exactly one **Entity type** and a set of **Structured metadata** fields.
- A **Finding model** carries zero or more **Attributes** (authored, never set by enrichment).
- **Index codes** and **Anatomic locations** attach ontology identity and location to a **Finding model**.
- **Subspecialty**, **Etiology**, **Expected time course**, **Sex specificity**, and **Age profile** are **Structured metadata** fields assigned during **Assignment**.

## Example dialogue

> **Dev:** "Cardiomegaly on a chest CT — is that a finding or a diagnosis?"
> **Radiologist:** "Finding. It's an observation; the diagnosis is whatever is enlarging the heart. And don't tag it male/female just because it's more common in one — sex specificity is about anatomy, not prevalence."

## Flagged ambiguities

- "finding" means both the **Finding model** container and the `finding` **Entity type** value — resolved: "finding model" for the container, "finding" only for the entity_type value / observation.
- "subspecialty" was read as the ordering service — resolved: it is the reader-domain that interprets the finding.
