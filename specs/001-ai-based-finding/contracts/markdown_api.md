
# API Contract: Markdown Editing (Simplified)

## Function: export_model_to_markdown
```python
def export_model_to_markdown(model: FindingModelFull) -> str
```

## Function: edit_model_markdown
```python
def edit_model_markdown(model: FindingModelFull, edited_markdown: str, source: str = "USER") -> EditResult
```