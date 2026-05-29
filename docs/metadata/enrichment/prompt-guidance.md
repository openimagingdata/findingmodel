# Metadata Enrichment Prompt Guidance

Active prompts should be concise, task-specific, and organized around general rules. Prompt text is
not the place to preserve every historical miss.

Use `docs/metadata/fields.md` and `docs/metadata/subspecialties.md` as the durable source of
review-derived judgment rules. Prompts should distill those standards into concise instructions, not
copy the whole decision record.

## Current Prompt Pattern

- One focused task per prompt.
- Clear goal at the beginning.
- General rules first.
- Examples only when they illustrate a general reusable rule.
- No examples copied from active eval fixtures.
- No long lists of case-specific exceptions.
- No obsolete `clear_fields` output contract.

## Prompt Status

All seven assignment agents have their own lean prompt under `metadata/prompts/` (see
`enrichment-agent-architecture.md`). There is no broad/aggregate assignment prompt. Prompts should
support radiologists' use of findings in reports, not maintenance of an abstract ontology for its
own sake.

The subspecialty-domain, modality-applicability, and etiology/time-course prompts have had the most
deliberate concision and rule-shaping work:

- Subspecialty domain: body-region subspecialties can coexist with horizontal domains such as
  vascular, oncologic, cardiac, quality, or procedure/device when those domains are actually
  implied.
- Modality applicability: direct modality language matters, but generic findings and generic
  artifacts should not become every modality.
- Etiology/time-course: etiologies and expected persistence are handled together because they often
  interact, but both can be null.

Still worth a tuning pass before large-scale use:

- the anatomy and index-code (ontology) selection prompts;
- any helper prompts that still carry old classifier or `clear_fields` assumptions.

## Evidence Handling

Prompts may receive source context such as a collection name, known anatomic scope, modality bias,
or curated hierarchy. Use that context to avoid over-broad assignments, but do not let it override
the finding's name and description.

Prompt rules should preserve these behaviors:

- prefer null over generic whole-body fallback when no specific region is supported;
- allow anatomy/code selection to reject weak candidate sets;
- do not force a best candidate when all candidates are poor;
- treat negative findings, technique-only concepts, device states, and postoperative states as
  distinct source-model/entity-type questions rather than ordinary disease findings.

## Examples

Examples are acceptable only when:

- they are tied to a general rule;
- they help with cases beyond the example itself;
- they are not in the active eval set.

Examples are not a place to copy missed eval cases. If a missed case matters, derive the general
rule, put the rule in the field reference if it is durable, and use the eval case to measure whether
the rule works.

## LLM Graders

Current LLM grader experiments are not part of the durable baseline. Rebuild them later only as
optional diagnostics with clear prompts and evidence of value.

If LLM review is reintroduced, it should look like a sub-agent critique task: classify misses into
actionable categories, cite evidence, and suggest whether the fix belongs in prompt rules, schema,
scoring, gold data, or source-model cleanup. It should not be a hidden numeric gate.
