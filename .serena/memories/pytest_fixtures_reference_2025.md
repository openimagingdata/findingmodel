# Pytest Fixtures Reference

**Last Updated**: 2025-10-09
**File**: `test/conftest.py`

## Overview
The `test/conftest.py` file contains **reusable pytest fixtures** that provide valid test data for FindingModel tests. **ALWAYS use these existing fixtures** instead of creating new ones or using fake data.

## Critical Rules for Test Implementation

### ❌ DO NOT
- Create new fixtures for test models when existing ones already exist
- Use fake/made-up data (invalid OIFM IDs, attribute IDs, etc.)
- Create helper functions like `_make_test_model()` that duplicate fixture functionality
- Hardcode test data directly in test functions

### ✅ DO
- Use `full_model` fixture for valid FindingModelFull with proper IDs
- Use `real_model` fixture for real-world test data (pulmonary_embolism)
- Use `tmp_defs_path` fixture for directory-based tests with real .fm.json files
- Use `base_model` fixture for FindingModelBase (pre-ID assignment)
- Check conftest.py BEFORE creating any test data

## Available Fixtures

### 1. `full_model` - Valid Test Model with IDs
**Type**: `FindingModelFull`
**Use for**: Most tests requiring a complete, valid finding model

```python
def test_example(full_model: FindingModelFull):
    # full_model has valid IDs that pass validation
    assert full_model.oifm_id == "OIFM_TEST_123456"
    # Has 2 attributes with valid IDs
    assert full_model.attributes[0].oifma_id == "OIFMA_TEST_123456"
    assert full_model.attributes[1].oifma_id == "OIFMA_TEST_654321"
```

**Contents**:
- `oifm_id`: "OIFM_TEST_123456"
- `name`: "Test Model"
- `description`: "A simple test finding model."
- `synonyms`: ["Test Synonym"]
- `tags`: ["tag1", "tag2"]
- `attributes`: 2 attributes (ChoiceAttributeIded + NumericAttributeIded)
  - Severity (OIFMA_TEST_123456): choice attribute with 2 values
  - Size (OIFMA_TEST_654321): numeric attribute (1-10 cm)

**ID patterns match validation**:
- OIFM ID: `OIFM_TEST_123456` matches `^OIFM_[A-Z]{3,4}_[0-9]{6}$`
- Attribute IDs: `OIFMA_TEST_123456` matches `^OIFMA_[A-Z]{3,4}_[0-9]{6}$`

### 2. `real_model` - Real Pulmonary Embolism Model
**Type**: `FindingModelFull`
**Use for**: Integration tests, realistic data scenarios

```python
def test_with_real_data(real_model: FindingModelFull):
    # Loaded from test/data/pulmonary_embolism.fm.json
    assert real_model.name == "Pulmonary Embolism"
```

**Source**: `test/data/pulmonary_embolism.fm.json`
**Benefits**:
- Real-world complexity (multiple attributes, codes, contributors)
- Validates against actual production-like data
- Tests edge cases that simple fixtures might miss

### 3. `tmp_defs_path` - Directory of Test Files
**Type**: `Path`
**Use for**: Directory ingestion tests, batch operations

```python
@pytest.mark.asyncio
async def test_directory_ingestion(tmp_defs_path: Path):
    # tmp_defs_path contains copies of test/data/defs/*.fm.json files
    result = await index.update_from_directory(tmp_defs_path)
    assert result["added"] > 0
```

**Contents**: Temporary copy of `test/data/defs/` directory with real .fm.json files
**Benefits**:
- Isolated temporary directory (changes don't affect original files)
- Multiple real finding model files for batch testing
- Automatic cleanup after test

### 4. `base_model` - Model Without IDs
**Type**: `FindingModelBase`
**Use for**: Testing ID assignment, model creation workflows

```python
def test_id_assignment(base_model: FindingModelBase):
    # base_model has NO IDs yet (pre-assignment state)
    assert base_model.name == "Test Model"
    # Use for testing add_ids_to_model() functions
```

**Contents**: Same structure as `full_model` but without OIFM ID or attribute IDs
**Use case**: Testing ID assignment workflows, model creation from scratch

### 5. `real_model_markdown` - Markdown Source
**Type**: `str`
**Use for**: Testing markdown → model conversion

**Source**: `test/data/pulmonary_embolism.md`

### 6. `pe_fm_json` / `tn_fm_json` - Raw JSON Strings
**Type**: `str`
**Use for**: Testing JSON parsing, serialization

- `pe_fm_json`: Pulmonary embolism JSON
- `tn_fm_json`: Thyroid nodule JSON

### 7. `finding_info` - FindingInfo Object
**Type**: `FindingInfo`
**Use for**: Testing FindingInfo workflows

**Contents**:
- `name`: "test finding"
- `description`: "A test finding for testing."
- `synonyms`: ["test", "finding"]

## Common Test Patterns

### Pattern 1: Single Model CRUD Tests
```python
@pytest.mark.asyncio
async def test_add_model(full_model: FindingModelFull, tmp_path: Path):
    index = DuckDBIndex(tmp_path / "test.duckdb")
    await index.setup()
    
    # Use full_model directly - it has valid IDs
    result = await index.add_or_update_entry_from_file(
        full_model.to_file(tmp_path / "test.fm.json")
    )
    assert result == IndexReturnType.ADDED
```

### Pattern 2: Directory Batch Tests
```python
@pytest.mark.asyncio
async def test_batch_update(tmp_defs_path: Path, tmp_path: Path):
    index = DuckDBIndex(tmp_path / "test.duckdb")
    await index.setup()
    
    # tmp_defs_path has real .fm.json files
    result = await index.update_from_directory(tmp_defs_path)
    assert result["added"] > 0
```

### Pattern 3: Validation Tests
```python
@pytest.mark.asyncio
async def test_duplicate_validation(full_model: FindingModelFull, tmp_path: Path):
    index = DuckDBIndex(tmp_path / "test.duckdb")
    await index.setup()
    
    # Add once
    await index.add_or_update_entry_from_file(...)
    
    # Try to add different model with same ID - should fail validation
    errors = await index.validate_model(full_model)
    assert any("already exists" in err for err in errors)
```

## Why This Matters

### Past Problems (DO NOT REPEAT)
1. **Fake data with invalid IDs**: Tests created models with IDs like "OIFMA_UNIT_000001_001" which don't match the validation pattern `^OIFMA_[A-Z]{3,4}_[0-9]{6}$`
2. **Duplicate fixtures**: Test implementers created new `_make_test_model()` helpers instead of using `full_model`
3. **Test failures**: Invalid test data caused 48/65 tests to fail, requiring manual fixes

### Correct Approach
1. **Check conftest.py FIRST** before writing any test
2. **Use `full_model`** for 90% of tests requiring a complete model
3. **Use `real_model`** for integration/realistic scenarios
4. **Use `tmp_defs_path`** for directory/batch operations
5. **Never create fake data** - existing fixtures have valid, tested data

## ID Pattern Reference

For reference when validating or creating test data:

### OIFM ID Pattern
```regex
^OIFM_[A-Z]{3,4}_[0-9]{6}$
```
**Valid examples**:
- `OIFM_TEST_123456` ✅ (used in full_model)
- `OIFM_ACR_000001` ✅
- `OIFM_UNIT_000001` ✅

**Invalid examples**:
- `OIFM_UNIT_000001_001` ❌ (extra suffix)
- `OIFM_test_123456` ❌ (lowercase)
- `OIFM_TE_123456` ❌ (only 2 letters)

### Attribute ID Pattern
```regex
^OIFMA_[A-Z]{3,4}_[0-9]{6}$
```
**Valid examples**:
- `OIFMA_TEST_123456` ✅ (used in full_model)
- `OIFMA_TEST_654321` ✅ (used in full_model)
- `OIFMA_ACR_000001` ✅

**Invalid examples**:
- `OIFMA_UNIT_000001_001` ❌ (extra suffix)
- `OIFMA_test_123456` ❌ (lowercase)

## Test Data Files

Location: `test/data/`

**Available files**:
- `pulmonary_embolism.fm.json` - Real PE finding model
- `pulmonary_embolism.md` - Markdown source
- `thyroid_nodule_codes.fm.json` - Real thyroid nodule model
- `thyroid_nodule_codes.md` - Markdown source
- `defs/` directory - Multiple .fm.json files for batch testing

**Usage**: Access via fixtures (`real_model`, `tmp_defs_path`) rather than reading directly

## Quick Reference Card

| Need | Use Fixture | Type |
|------|-------------|------|
| Valid complete model | `full_model` | `FindingModelFull` |
| Real-world model | `real_model` | `FindingModelFull` |
| Directory of files | `tmp_defs_path` | `Path` |
| Model without IDs | `base_model` | `FindingModelBase` |
| JSON string | `pe_fm_json` | `str` |
| Markdown string | `real_model_markdown` | `str` |
| FindingInfo | `finding_info` | `FindingInfo` |

## Summary

**Golden Rule**: If you're writing a test that needs a FindingModel, check conftest.py first. The fixtures you need probably already exist, are properly validated, and are ready to use. Creating new test data is almost never necessary.
