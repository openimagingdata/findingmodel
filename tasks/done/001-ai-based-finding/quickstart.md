
# Quickstart: AI-Based Finding Model Editor (Simplified)

This guide demonstrates the minimal user workflow for both editing modes and CLI.

## Prerequisites

```bash
pip install findingmodel
export OPENAI_API_KEY="your_openai_key"
```

## Natural Language Editing

```python
from findingmodel import FindingModelFull
from findingmodel.tools.model_editor import edit_model_natural_language

with open("test_data/pneumonia.fm.json") as f:
    model = FindingModelFull.model_validate_json(f.read())

result = await edit_model_natural_language(
    model=model,
    command="add severity attribute with mild, moderate, severe options"
)
```

## Markdown Editing

```python
from findingmodel.tools.model_editor import export_model_for_editing, edit_model_markdown

markdown_content = export_model_for_editing(model)
# User edits markdown_content...
result = await edit_model_markdown(
    model=model,
    edited_markdown=markdown_content
)
```
