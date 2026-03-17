# Markdown Output Overhaul Plan

## Why This Exists

We are spending too much effort making Markdown behave like a transport format.
That is creating brittle code in multiple places:

- `packages/findingmodel/src/findingmodel/fm_md_template.py`
- `packages/findingmodel/src/findingmodel/finding_model.py`
- `packages/findingmodel-ai/src/findingmodel_ai/authoring/editor.py`
- `packages/findingmodel-ai/src/findingmodel_ai/authoring/markdown_in.py`

The core problem is architectural, not cosmetic:

- `FindingModel` JSON is the canonical data model.
- Markdown is useful for humans, but it is a poor canonical interchange format.
- We currently blur together:
  - pretty published output
  - editable authoring text
  - LLM prompt/input format

That makes every small schema change cascade into template churn, parser churn, and tests that mostly check formatting trivia.

## Working Decision

Treat Markdown as a one-way presentation layer by default, not a round-trip contract.

Corollaries:

- `as_markdown()` should aim to produce clean, readable output from canonical model data.
- Authoring/editing should primarily operate on structured JSON/model objects.
- Any Markdown-based editing/import should be explicitly labeled as convenience tooling, not the source of truth.
- If we keep a Markdown edit/import path, it should consume a narrow, well-defined authoring format instead of trying to infer arbitrary prose.

## Target Shape

### 1. Single Markdown rendering module

Create one renderer layer in `findingmodel` responsible for human-readable output.

Suggested shape:

- `packages/findingmodel/src/findingmodel/render_markdown.py`

Responsibilities:

- Build a small presentation model from `FindingModelBase` / `FindingModelFull`
- Render sections in a fixed order
- Centralize label formatting, list formatting, and spacing rules
- Support a small number of explicit modes, for example:
  - `published`
  - `authoring`

What should move out of call sites:

- inline Jinja condition juggling
- repeated field-by-field formatting in model methods
- ad hoc top-metadata rendering in editor code

### 2. Keep model classes thin

`FindingModelBase.as_markdown()` and `FindingModelFull.as_markdown()` should become thin wrappers around the renderer.

They should not contain presentation logic beyond selecting renderer options.

### 3. Separate published Markdown from authoring text

We should stop pretending these are the same artifact.

Published Markdown goals:

- pleasant to read
- stable enough for docs/export
- optimized for display

Authoring text goals:

- easy for humans and LLMs to edit safely
- minimal formatting surface
- deterministic enough that small edits do not require parser gymnastics

If we keep authoring text in Markdown-ish form, it should be a deliberately constrained format, not our public/published output template.

### 4. Stop investing in general Markdown round-tripping

We should not keep expanding "Markdown -> model" inference for arbitrary prose.

Preferred direction:

- structured authoring/editing on JSON or typed objects
- optional export to Markdown for review
- optional import from a constrained authoring format only

If `create_model_from_markdown()` remains, it should be treated as an AI-assisted importer from outlines, not as a guarantee of faithful round-trip recovery from exported Markdown.

## Near-Term Plan

### Phase 1: Stabilize output

- Introduce a shared renderer module for clean one-way Markdown generation.
- Refactor `FindingModelBase.as_markdown()` and `FindingModelFull.as_markdown()` to use it.
- Refactor `export_model_for_editing()` to reuse shared section/formatting helpers where appropriate.
- Keep the current visible output roughly similar unless there is a clear readability win.

### Phase 2: Narrow authoring format

- Decide whether `edit_model_markdown()` should continue to accept loose Markdown.
- If yes, define a constrained authoring format and document it explicitly.
- If no, shift authoring/editing workflows toward structured JSON plus higher-level editing commands.

### Phase 3: Reduce surface area

- Remove duplicated formatting logic from `editor.py`.
- Minimize or remove Markdown-specific parsing from importer flows.
- Keep only the smallest supported Markdown import path that still has a clear user value.

## Success Criteria

We should consider this overhaul successful when:

- There is one obvious place to change Markdown rendering.
- Schema additions do not require touching multiple unrelated Markdown code paths.
- Published output and authoring/editing output are intentionally distinct.
- Tests focus on content/sections and important invariants, not incidental whitespace behavior.
- Markdown is no longer treated as the canonical round-trip representation of a finding model.

## Recommended Next Step

Do not keep expanding Slice 2 in its current form.

Instead:

1. Finish only the minimum needed for current work to remain usable.
2. Open a new implementation slice for the renderer refactor.
3. Treat Markdown import/editing separately from published output so we can simplify both.
