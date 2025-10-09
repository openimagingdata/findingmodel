# Test Multi-Agent System Plan

## Goal
Verify that the multi-agent orchestration system works correctly by implementing a small test feature.

## Phase 1: Create Simple Utility Function

Create a new utility function in `src/findingmodel/tools/test_utils.py`:

```python
def format_oifm_id(source: str, number: int) -> str:
    """Format an OIFM ID from source and number.

    Args:
        source: Source code (e.g., 'MSFT', 'ACR')
        number: Six-digit number

    Returns:
        Formatted ID like 'OIFM_MSFT_123456'

    Raises:
        ValueError: If number is not 6 digits
    """
    if not (0 <= number <= 999999):
        raise ValueError(f"Number must be 0-999999, got {number}")

    return f"OIFM_{source.upper()}_{number:06d}"
```

**Success Criteria:**
- Function created in correct location
- Proper type hints
- Clear docstring
- Error handling for invalid input
- Follows code_style_conventions

## Phase 2: Add Tests for Utility Function

Create tests in `test/test_test_utils.py`:

```python
import pytest
from findingmodel.tools.test_utils import format_oifm_id

def test_format_oifm_id_basic():
    """Test basic OIFM ID formatting."""
    result = format_oifm_id("MSFT", 123456)
    assert result == "OIFM_MSFT_123456"

def test_format_oifm_id_lowercase():
    """Test that source is uppercased."""
    result = format_oifm_id("msft", 123)
    assert result == "OIFM_MSFT_000123"

def test_format_oifm_id_zero_padding():
    """Test zero padding for small numbers."""
    result = format_oifm_id("ACR", 42)
    assert result == "OIFM_ACR_000042"

def test_format_oifm_id_invalid_number():
    """Test error on invalid number."""
    with pytest.raises(ValueError, match="Number must be 0-999999"):
        format_oifm_id("MSFT", 1000000)
```

**Success Criteria:**
- All 4 tests implemented
- Tests pass when run
- Proper test naming
- Good coverage of edge cases
- Follows test patterns

## Notes

This is a minimal test to verify the orchestration system. The orchestrator should:
1. Delegate Phase 1 to python-core-implementer
2. Have python-core-evaluator assess the implementation
3. Fix any issues
4. Delegate Phase 2 to python-test-implementer
5. Have python-test-evaluator assess the tests
6. Fix any issues
7. Run tests with `task test`
8. Create git commit
9. Report success
