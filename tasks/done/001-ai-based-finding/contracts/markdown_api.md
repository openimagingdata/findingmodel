
# API Contract: Markdown Editing

## `export_model_for_editing`

```python
def export_model_for_editing(model: FindingModelFull, *, attributes_only: bool = False) -> str
```

### Inputs
- `model`: Complete `FindingModelFull` instance with immutable IDs
- `attributes_only`: When `True`, omits the model-level metadata section

### Behavior
- Renders a stable Markdown document with top-level metadata followed by `### {attribute}` sections
- Choice attribute values render as bullet lists (`- value[: description]`)
- Numeric attribute metadata renders on a single bullet with `min`, `max`, `unit` segments (semicolon-separated)
- Output is designed for round-tripping through `edit_model_markdown`

### Output
- UTF-8 Markdown string with trailing newline

## `edit_model_markdown`

```python
async def edit_model_markdown(
		model: FindingModelFull,
		edited_markdown: str,
		*,
		agent: Agent[EditDeps, EditResult] | None = None,
) -> EditResult
```

### Inputs
- `model`: Original `FindingModelFull` that acts as the source of truth
- `edited_markdown`: User-modified Markdown produced by `export_model_for_editing`
- `agent`: Optional override for dependency injection/testing

### Behavior
- Performs a **preflight guard** before calling the LLM—if any original attribute header or bullet is missing, the function returns immediately with the untouched model and a rejection message
- Invokes a Pydantic AI agent that must:
	- Preserve all existing model, attribute, and value IDs
	- Use the literal `PLACEHOLDER_ATTRIBUTE_ID` for every brand-new attribute the user added
	- Populate numeric metadata (min/max/unit) when present in Markdown (case-insensitive keys)
- Normalizes placeholder IDs to ensure they remain placeholders until persistence-time promotion
- Validates output via `_basic_edit_validation` (missing IDs, removed attributes, incorrect suffixes) and retries the agent with detailed guidance when necessary

### Output
- Returns `EditResult` with the updated model and a `rejections` list describing forbidden edits that were skipped
- On preflight failure or agent exception, returns original model plus descriptive rejection(s)

### Post-processing Requirements
- Call `assign_real_attribute_ids()` before saving to disk or exporting. This will mint deterministic OIFMA IDs and re-number value codes for all placeholders.
- Persist changes with `model.model_dump_json(exclude_none=True)` after successful validation

### Related Tests
- `test/test_model_editor.py::test_forbidden_change_markdown_callout_real_api` (ensures deletions are rejected)
- `test/test_model_editor.py::test_assign_real_attribute_ids_*` (verifies placeholder-handling flow)