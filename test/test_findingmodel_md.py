from findingmodel import FindingModelBase, FindingModelFull

BASE_MODEL_MARKDOWN = """
# Test model

**Synonyms:** Test Synonym

**Tags:** tag1, tag2

A simple test finding model.

## Attributes

### Severity

How severe is the finding?  
*(Select one)*

- **Mild**  
- **Severe**  

### Size

Size of the finding.  
Mininum: 1  
Maximum: 10  
Unit: cm""".strip()


def test_base_model_markdown(base_model: FindingModelBase) -> None:
    md = base_model.as_markdown()
    assert md.strip() == BASE_MODEL_MARKDOWN
    # Spacing: no double blank lines between sections
    assert "\n\n\n" not in md


FULL_MODEL_MARKDOWN = """
# Test model—`OIFM_TEST_123456`

**Synonyms:** Test Synonym

**Tags:** tag1, tag2

A simple test finding model.

## Attributes

### Severity—`OIFMA_TEST_123456`

How severe is the finding?  
*(Select one)*

- **Mild**  
- **Severe**  

### Size—`OIFMA_TEST_654321`

Size of the finding.  
Mininum: 1  
Maximum: 10  
Unit: cm""".strip()


def test_full_model_markdown_with_ids(full_model: FindingModelFull) -> None:
    md = full_model.as_markdown()
    assert md.strip() == FULL_MODEL_MARKDOWN
    # Spacing: no double blank lines between sections
    assert "\n\n\n" not in md


def test_full_model_markdown_hide_ids(full_model: FindingModelFull) -> None:
    md = full_model.as_markdown(hide_ids=True)
    # IDs should not appear
    assert "OIFM_TEST_123456" not in md
    assert "OIFMA_TEST_123456" not in md
    assert "OIFMA_TEST_654321" not in md
    # Spacing: no double blank lines between sections
    assert "\n\n\n" not in md


def test_real_model_markdown(real_model: FindingModelFull, real_model_markdown: str) -> None:
    md = real_model.as_markdown()
    for generated_line, expected_line in zip(md.splitlines(), real_model_markdown.splitlines(), strict=True):
        assert generated_line.strip() == expected_line.strip()


def test_real_model_markdown_with_codes(tn_fm_json: str, tn_markdown: str) -> None:
    tn_model = FindingModelFull.model_validate_json(tn_fm_json)
    md = tn_model.as_markdown()
    for generated_line, expected_line in zip(md.splitlines(), tn_markdown.splitlines(), strict=True):
        assert generated_line.strip() == expected_line.strip()
