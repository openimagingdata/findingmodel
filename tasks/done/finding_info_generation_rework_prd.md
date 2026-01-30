# FindingInfo Generation Rework PRD

**Date:** 2025-09-28  
**Author:** GitHub Copilot (assistant)  
**Stakeholders:** FindingModel maintainers, clinical content editors, infra tooling team

## 1. Overview

Rework the FindingInfo generation workflow so it uses Pydantic AI instead of Instructor, while updating the prompting to enforce new naming conventions and specificity guidance. The goal is to standardize agent usage across the codebase, improve maintainability, and meet newly requested clinical authoring behaviors without introducing validation logic that belongs in model definitions.

## 2. Background

- `src/findingmodel/tools/finding_description.py` currently calls `AsyncInstructor` with a Markdown prompt template to produce a `FindingInfo` instance from a finding name.
- Prompt instructions do not cover acronym expansion, specificity corrections, or stricter casing rules.
- The project already leverages Pydantic AI for other tools (e.g., `similar_finding_models.py`), making Instructor an outlier.
- Testing relies on live OpenAI integrations in callout tests; no unit-level coverage validates agent logic with fake models.

## 3. Problem Statement

- Divergent AI orchestration libraries increase maintenance overhead and complicate shared abstractions.
- Current prompt allows overly specific outputs and inconsistent acronym handling, requiring manual cleanup.
- Lack of deterministic tests makes regressions harder to catch without running external callouts.

## 4. Goals & Non-Goals

### Goals

- Replace Instructor usage with a Pydantic AI `Agent` that returns `FindingInfo` objects via structured output.
- Update prompting/instructions so the agent:
  - Expands acronyms in the canonical finding name and emits the acronym as a synonym.
  - Generalizes overly specific user inputs to an appropriate clinical granularity and documents specific examples in the description.
  - Emits lowercase names and synonyms except for proper nouns or acronyms.
- Ensure the canonical finding name is singular by default, with prompt guidance reinforcing the expectation.
- Preserve existing public interfaces (`create_info_from_name`, CLI entry points) and ensure backwards compatibility for callers.
- Add deterministic unit coverage using `TestModel`/`FunctionModel` to confirm transformations without contacting real models.

### Non-Goals

- No changes to `FindingInfo` schema or validation rules.
- No redesign of downstream tools (e.g., stub model generation, ontology search) beyond necessary touch points.
- No production rollout of new clinical policies beyond those enumerated above.

## 5. User Stories

- **Clinical author:** As a content curator, when I request "PCL tear", I receive a FindingInfo where the name is "posterior cruciate ligament tear" and the synonym list includes "PCL tear".
- **Content reviewer:** As a reviewer, when the user supplies "left lower lobe opacity", the tool standardizes to "pulmonary opacity" and mentions the specific location in the description so examples remain discoverable.
- **Engineer:** As a developer, I can run unit tests without network access and confirm that acronym expansion and specificity adjustments remain correct.

## 6. Functional Requirements

1. Introduce a Pydantic AI agent in `finding_description.py` configured with `output_type=FindingInfo` and using `get_openai_model` for provider creation.
2. Load updated prompt content (either via template or inline instructions) that explicitly captures acronym and specificity behavior, casing expectations, and inclusion of the user-provided term as a synonym if altered.
3. For inputs containing acronyms (detected via uppercase patterns or parenthetical hints), instruct the model to expand them in the canonical name and keep the acronym in synonyms.
4. For inputs that are overly specific (e.g., contain laterality, subsegment, measurement qualifiers), instruct the model to normalize to the broader finding while incorporating the specifics as examples in the description.
5. Ensure the output name and synonyms follow casing rules using agent instructions; only apply minimal post-processing needed for spacing or duplicate trimming, avoiding validation logic owned by `FindingInfo`.
6. Continue honoring existing settings for model selection and environment gating (`settings.openai_default_model`).
7. Emit an explicit log when the canonical name returned differs from the user-provided input, and surface actionable errors when the agent output cannot be validated into `FindingInfo` after limited retries.

## 7. Non-Functional Requirements

- Maintain async API.
- Keep compatibility with existing CLI workflows and notebooks.
- Reuse current logging patterns.
- Record documentation updates in `CHANGELOG.md` under the unreleased section alongside other project notes.
- Avoid telemetry beyond the required logging for canonical-name adjustments.
- Ensure tests follow project lint/type standards (`task check`).

## 8. Prompt & Instruction Updates

- Update `prompt_templates/get_finding_description.md.jinja` (or new inline instructions) to include sections covering:
  - Canonical naming at appropriate specificity.
  - Acronym expansion/synonym inclusion with concrete examples.
  - Lowercase formatting guidance, explicitly allowing proper nouns and acronyms.
  - Singular canonical naming, including an example showing plural input mapped to a singular output.
  - Direction to include examples of overly specific input phrases within the description narrative.
- Provide rationale/examples within prompt to reduce model ambiguity while keeping instructions concise.

## 9. Testing Strategy

- Add unit tests using Pydantic AI `TestModel`/`FunctionModel` with `agent.override(model=...)` to simulate responses (`ALLOW_MODEL_REQUESTS = False`).
- Cover representative cases: acronym expansion, specificity normalization, casing compliance, synonym deduplication when altered, idempotent pass-through when no adjustments required.
- Verify that logging is triggered (or a mockable hook is invoked) when the canonical name differs from the input.
- Preserve existing callout tests but adjust expectations if needed.
- Consider snapshot or inline comparisons for readability per doc guidance ([Pydantic AI Testing](https://ai.pydantic.dev/testing/)).

## 10. Dependencies & Risks

- Requires up-to-date Pydantic AI dependency (ensure version supports desired features).
- Prompt adjustments may require iterative tuning; risk mitigated via deterministic tests that assert the post-processed structure.
- Any reliance on minimal post-processing must stay within permissible normalization to avoid duplicating schema validation.

## 11. Rollout Plan

1. Implement agent migration and prompt update behind existing interface.
2. Introduce unit tests and adjust callouts.
3. Run `task check` and `task test` (callouts optional without API keys).
4. Update the unreleased section of `CHANGELOG.md` with a summary of the migration and behavioral rules.
5. Coordinate review focusing on prompt behavior, logging, and test coverage.

## 12. Open Questions

- Should synonym ordering prioritize original user term or canonicalized name aliases?
- Are there additional clinical rules (e.g., for pluralization) that need to be encoded now or later?

## 13. Appendix / References

- Pydantic AI Testing Guide: <https://ai.pydantic.dev/testing/>
- Pydantic AI Structured Output Modes: <https://ai.pydantic.dev/output/#structured-output>
- Industry best-practices overview (Perplexity summary referencing AWS article and Scrapeless blog, Sep 2025)
