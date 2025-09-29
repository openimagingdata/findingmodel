# FindingInfo Generation Rework Tasks

## Task 1 — Replace Instructor client with Pydantic AI Agent

- **Goal:** Swap `get_async_instructor_client` usage for a `pydantic_ai.Agent` that produces `FindingInfo` via structured output.
- **Where:** `src/findingmodel/tools/finding_description.py`
- **How:**
  1. Import `Agent` (and `RunContext` if tools are required) from `pydantic_ai`.
  2. Instantiate a module-level agent configured with `output_type=FindingInfo`, `model=get_openai_model(settings.openai_default_model)`, and the updated instructions from Task 2.
  3. Update `create_info_from_name` (and the deprecated wrappers) to call `agent.run` instead of the Instructor client, handling retries via agent configuration rather than manual loops.
  4. Emit an explicit log entry when the canonical name differs from the input term before returning the result.
  5. Remove Instructor-specific helpers that are no longer needed.
- **Notes:** Keep all validation inside `FindingInfo`; outside of minor whitespace trimming/deduplication, avoid additional field validation logic, and reuse existing logging patterns—no extra telemetry is required.

## Task 2 — Revise finding description prompt instructions

- **Goal:** Encode the new acronym, specificity, casing, and singular-naming rules in the agent instructions.
- **Where:** `src/findingmodel/tools/prompt_templates/get_finding_description.md.jinja` (or inline agent instructions if that proves clearer).
- **How:**
  1. Add guidance to expand acronyms in the primary name while preserving the original acronym as a synonym (e.g., "PCL tear" → name "posterior cruciate ligament tear", synonym includes "PCL tear").
  2. Direct the model to generalize overly specific requests to an appropriate clinical level and mention the specific phrasing as an example in the description (e.g., generalize "left lower lobe opacity" to "pulmonary opacity").
  3. Specify lowercase requirements for names and synonyms except for acronyms and proper nouns.
  4. Instruct the agent to produce singular canonical names even when given plural inputs, and include at least one worked example.
  5. Provide 1–2 concise examples inside the prompt so the model applies the rules consistently.
- **Notes:** Keep the prompt succinct to avoid instruction dilution.

## Task 3 — Light-touch normalization helper (optional)

- **Goal:** Ensure agent output follows casing and synonym expectations without duplicating `FindingInfo` validators.
- **Where:** `src/findingmodel/tools/finding_description.py`
- **How:**
  1. Implement a small helper that trims whitespace, removes duplicate synonyms, and ensures the user-supplied term is present if the canonical name changed.
  2. Respect the "no redundant validation" requirement—do not re-validate schema rules already enforced by Pydantic.
- **Notes:** Only perform transformations that are difficult to express directly in the prompt (e.g., deduping synonyms).

## Task 4 — Update CLI and imports

- **Goal:** Keep the CLI (`make_info` command) and any other callsites working transparently with the new agent.
- **Where:** `src/findingmodel/cli.py`, `src/findingmodel/tools/__init__.py`, notebooks referencing `create_info_from_name`.
- **How:**
  1. Confirm the CLI continues to call `describe_finding_name`/`create_info_from_name` without code changes beyond import adjustments if signatures move.
  2. Remove unused Instructor-specific imports and re-export the updated functions as before.
- **Notes:** Ensure deprecated aliases continue to warn but function correctly.

## Task 5 — Add deterministic unit tests for agent behavior

- **Goal:** Cover the new behaviors without relying on external API calls.
- **Where:** `test/test_tools.py` (or a new dedicated module if scope grows).
- **How:**
  1. Set `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False` at the module level for tests.
  2. Use `agent.override(model=TestModel())` for smoke coverage and `FunctionModel` when specific structured outputs are needed.
  3. Write tests for: acronym expansion, specificity normalization, casing enforcement, synonym deduplication, and pass-through when input already meets requirements.
  4. Assert that logging (or a mockable hook) occurs when canonical names differ from the input term.
  5. Update or extend callout tests only if expectations change, keeping them as integration checks.
- **Notes:** Consider fixtures to reuse overrides per doc guidance (<https://ai.pydantic.dev/testing/>).

## Task 6 — Documentation and verification

- **Goal:** Ensure stakeholders understand the change and quality gates pass.
- **Where:** PR, release notes, and local tooling.
- **How:**
  1. Summarize the migration in PR description, referencing this PRD and tasks list.
  2. Run `task check` and `task test` (callouts optional if keys unavailable; document any skipped steps).
  3. Update the unreleased section in `CHANGELOG.md` with a concise summary of the migration and behavioral changes.
  4. Capture notable prompt updates or conventions in `CLAUDE.md`/Serena memories as part of final review per project expectations.
- **Notes:** Highlight any residual open questions or follow-up work discovered during implementation.
