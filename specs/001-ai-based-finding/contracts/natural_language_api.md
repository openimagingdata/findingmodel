
# API Contract: Natural Language Editing

## `edit_model_natural_language`

```python
async def edit_model_natural_language(
		model: FindingModelFull,
		command: str,
		*,
		agent: Agent[EditDeps, EditResult] | None = None,
) -> EditResult
```

### Inputs
- `model`: Source `FindingModelFull`
- `command`: Natural-language request describing desired edits
- `agent`: Optional override for testing with stubbed Pydantic AI agents

### Behavior
- Delegates to a Pydantic AI agent that produces an updated model according to the safe-edit rules
- Agent prompt mirrors the Markdown workflow requirements:
	- Preserve every existing OIFM ID (model, attribute, value)
	- Use `PLACEHOLDER_ATTRIBUTE_ID` for new attributes and derive value codes as `{placeholder}.{index}`
	- Only add clinically safe content (new attributes, descriptive text) and never rename/remove existing structures
- Output validator applies `_basic_edit_validation`, retrying the agent with corrective guidance as needed
- `_normalize_new_attribute_ids` ensures any accidental agent-generated IDs are reset back to placeholders for downstream finalization

### Output
- `EditResult` containing the updated model and rejections for any forbidden sub-commands that were skipped
- In case of agent failure, returns original model with rejection message(s)

### Post-processing Requirements
- After applying accepted changes, call `assign_real_attribute_ids()` (or let the interactive demos do it on save) before persisting
- Persist via `model_dump_json(exclude_none=True)` to maintain canonical formatting

### Related Tests
- `test/test_model_editor.py::test_assign_real_attribute_ids_*`
- `test/test_model_editor.py::test_forbidden_change_markdown_callout_real_api` (ensures shared validation covers deletion attempts)