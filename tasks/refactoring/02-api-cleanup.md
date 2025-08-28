# API Cleanup and Deprecation Plan

## Executive Summary
The tools module has 8+ duplicate function aliases creating API confusion and maintenance burden. This refactoring will establish a clear deprecation strategy and clean up the public API.

**Impact**: Cleaner API surface, better developer experience, reduced confusion
**Risk**: Low - Changes are backward compatible with deprecation warnings
**Effort**: 3-4 days including documentation updates

## Current State Analysis

### Duplicate Functions Identified
Based on PROJECT_INDEX.json analysis:

| Old Name (Deprecated) | New Name (Canonical) | Location |
|----------------------|---------------------|----------|
| `create_info_from_name` | `create_finding_info_from_name` | tools/finding_description.py |
| `add_details_to_info` | `add_details_to_finding_info` | tools/finding_description.py |
| `describe_finding_name` | `create_finding_info_from_name` | tools/finding_description.py |
| `get_detail_on_finding` | `add_details_to_finding_info` | tools/finding_description.py |
| `create_model_from_markdown` | `create_finding_model_from_markdown` | tools/markdown_in.py |
| `create_model_stub_from_info` | `create_finding_model_stub_from_finding_info` | tools/create_stub.py |
| `add_ids_to_model` | `add_ids_to_finding_model` | tools/add_ids.py |
| `add_standard_codes_to_model` | `add_standard_codes_to_finding_model` | tools/index_codes.py |

### Problems
- **Naming Inconsistency**: Mix of abbreviated and full names
- **Multiple Aliases**: Same functionality exposed 2-3 different ways
- **Documentation Confusion**: Users unsure which function to use
- **Maintenance Burden**: Changes must be applied to multiple functions
- **Import Confusion**: Unclear what should be imported from `__init__.py`

## Target State

### Naming Convention
All public functions follow pattern: `{action}_finding_{object}_{qualifier}`

Examples:
- `create_finding_info_from_name` ✓
- `add_details_to_finding_info` ✓
- `create_finding_model_from_markdown` ✓
- `add_ids_to_finding_model` ✓

### Public API Structure
```python
# src/findingmodel/tools/__init__.py
__all__ = [
    # Finding Info Operations
    'create_finding_info_from_name',
    'add_details_to_finding_info',
    
    # Finding Model Creation
    'create_finding_model_from_markdown',
    'create_finding_model_stub_from_finding_info',
    
    # Finding Model Enhancement
    'add_ids_to_finding_model',
    'add_standard_codes_to_finding_model',
    
    # Analysis
    'find_similar_models',
    
    # Utilities
    'IdManager',
]
```

## Implementation Plan

### Phase 1: Create Deprecation Infrastructure (Day 1)

#### 1.1 Create Deprecation Utilities
```python
# src/findingmodel/tools/deprecation.py
import warnings
import functools
from typing import Callable, TypeVar, Optional
from datetime import datetime

T = TypeVar('T')

class DeprecationWarning(UserWarning):
    """Custom warning for deprecated functions."""
    pass

def deprecated(
    reason: str,
    version: str,
    removal_version: str,
    alternative: Optional[str] = None
) -> Callable[[T], T]:
    """
    Decorator to mark functions as deprecated.
    
    Args:
        reason: Why the function is deprecated
        version: Version when deprecated
        removal_version: Version when it will be removed
        alternative: Suggested alternative function
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = (
                f"{func.__module__}.{func.__name__} is deprecated since v{version} "
                f"and will be removed in v{removal_version}. {reason}"
            )
            if alternative:
                message += f" Use {alternative} instead."
            
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2
            )
            
            # Log deprecation usage for metrics
            _log_deprecation_usage(func.__name__, alternative)
            
            return func(*args, **kwargs)
        
        # Add deprecation metadata
        wrapper._deprecated = True
        wrapper._deprecation_version = version
        wrapper._removal_version = removal_version
        wrapper._alternative = alternative
        
        # Update docstring
        doc = wrapper.__doc__ or ""
        wrapper.__doc__ = f"**DEPRECATED**: {reason}\n\n{doc}"
        
        return wrapper
    return decorator

def _log_deprecation_usage(old_name: str, new_name: str):
    """Log deprecation usage for monitoring."""
    # Could write to file or send metrics
    pass
```

#### 1.2 Create Migration Helper
```python
# src/findingmodel/tools/migration.py
def create_alias(new_func: Callable, old_name: str) -> Callable:
    """Create a deprecated alias for a function."""
    
    @functools.wraps(new_func)
    def alias(*args, **kwargs):
        warnings.warn(
            f"{old_name} is deprecated. Use {new_func.__name__} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return new_func(*args, **kwargs)
    
    alias.__name__ = old_name
    alias.__doc__ = f"Deprecated alias for {new_func.__name__}.\n\n{new_func.__doc__}"
    
    return alias
```

### Phase 2: Apply Deprecations (Day 2)

#### 2.1 Mark Deprecated Functions
```python
# src/findingmodel/tools/finding_description.py
async def create_finding_info_from_name(...):
    """Create FindingInfo from a finding name using OpenAI."""
    # Current implementation
    ...

# Create deprecated aliases
create_info_from_name = deprecated(
    reason="Function renamed for consistency",
    version="0.4.0",
    removal_version="1.0.0",
    alternative="create_finding_info_from_name"
)(create_finding_info_from_name)

describe_finding_name = deprecated(
    reason="Duplicate functionality",
    version="0.4.0", 
    removal_version="1.0.0",
    alternative="create_finding_info_from_name"
)(create_finding_info_from_name)
```

#### 2.2 Update __init__.py Exports
```python
# src/findingmodel/tools/__init__.py
import warnings

# Primary exports (canonical names)
from .finding_description import (
    create_finding_info_from_name,
    add_details_to_finding_info,
)
from .markdown_in import create_finding_model_from_markdown
from .create_stub import create_finding_model_stub_from_finding_info
from .add_ids import add_ids_to_finding_model, IdManager
from .index_codes import add_standard_codes_to_finding_model
from .similar_finding_models import find_similar_models

# Deprecated aliases (will be removed in v1.0.0)
from .finding_description import (
    create_info_from_name,  # Deprecated
    add_details_to_info,    # Deprecated
    describe_finding_name,  # Deprecated
    get_detail_on_finding,  # Deprecated
)

# Show deprecation notice when importing deprecated names
def __getattr__(name):
    deprecated_mapping = {
        'create_info_from_name': 'create_finding_info_from_name',
        'add_details_to_info': 'add_details_to_finding_info',
        'describe_finding_name': 'create_finding_info_from_name',
        'get_detail_on_finding': 'add_details_to_finding_info',
    }
    
    if name in deprecated_mapping:
        warnings.warn(
            f"'{name}' is deprecated. Use '{deprecated_mapping[name]}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return globals()[deprecated_mapping[name]]
    
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    # Current API
    'create_finding_info_from_name',
    'add_details_to_finding_info',
    'create_finding_model_from_markdown',
    'create_finding_model_stub_from_finding_info',
    'add_ids_to_finding_model',
    'add_standard_codes_to_finding_model',
    'find_similar_models',
    'IdManager',
]
```

### Phase 3: Update Documentation (Day 3)

#### 3.1 Create Migration Guide
```markdown
# Migration Guide for findingmodel v0.4.0

## Breaking Changes
None - all changes are backward compatible with deprecation warnings.

## Deprecated Functions
The following functions are deprecated and will be removed in v1.0.0:

### Finding Description Functions
- `create_info_from_name()` → Use `create_finding_info_from_name()`
- `add_details_to_info()` → Use `add_details_to_finding_info()`
- `describe_finding_name()` → Use `create_finding_info_from_name()`
- `get_detail_on_finding()` → Use `add_details_to_finding_info()`

### Model Creation Functions  
- `create_model_from_markdown()` → Use `create_finding_model_from_markdown()`
- `create_model_stub_from_info()` → Use `create_finding_model_stub_from_finding_info()`

### Model Enhancement Functions
- `add_ids_to_model()` → Use `add_ids_to_finding_model()`
- `add_standard_codes_to_model()` → Use `add_standard_codes_to_finding_model()`

## How to Update Your Code

### Using sed/grep to find deprecated usage:
```bash
# Find deprecated function usage
grep -r "create_info_from_name\|add_details_to_info" --include="*.py" .

# Update imports automatically
sed -i 's/create_info_from_name/create_finding_info_from_name/g' **/*.py
sed -i 's/add_details_to_info/add_details_to_finding_info/g' **/*.py
```

### Manual Update Example:
```python
# Old code
from findingmodel.tools import create_info_from_name, add_details_to_info

info = await create_info_from_name("Pneumonia")
detailed = await add_details_to_info(info)

# New code
from findingmodel.tools import (
    create_finding_info_from_name,
    add_details_to_finding_info
)

info = await create_finding_info_from_name("Pneumonia")
detailed = await add_details_to_finding_info(info)
```
```

#### 3.2 Update README and Docs
- Update all examples to use new function names
- Add deprecation notices to old function docs
- Update API reference documentation
- Add migration guide link to changelog

### Phase 4: Codebase Migration (Day 4)

#### 4.1 Update Internal Usage
```python
# Find all internal usage
rg "create_info_from_name|add_details_to_info" src/ test/ --type py

# Update CLI commands
# src/findingmodel/cli.py
- from findingmodel.tools import create_info_from_name
+ from findingmodel.tools import create_finding_info_from_name

# Update tests
# test/test_tools.py
- result = await create_info_from_name("test")
+ result = await create_finding_info_from_name("test")
```

#### 4.2 Update External Examples
- Update notebooks/demo_find_similar.py
- Update any example scripts
- Update integration tests

## Testing Strategy

### Deprecation Tests
```python
# test/test_deprecation.py
import warnings
import pytest
from findingmodel.tools import (
    create_info_from_name,  # Deprecated
    create_finding_info_from_name,  # Current
)

def test_deprecated_function_shows_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Call deprecated function
        create_info_from_name("test")
        
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
        assert "create_finding_info_from_name" in str(w[0].message)

def test_deprecated_function_still_works():
    """Ensure deprecated functions still work correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Both should return same result
        old_result = create_info_from_name("test")
        new_result = create_finding_info_from_name("test")
        
        assert old_result == new_result
```

### Backward Compatibility Tests
- All existing tests should pass
- No breaking changes for external users
- Verify warnings are shown correctly

## Monitoring & Metrics

### Usage Tracking
```python
# src/findingmodel/tools/metrics.py
import json
from pathlib import Path
from datetime import datetime

METRICS_FILE = Path.home() / ".findingmodel" / "deprecation_usage.json"

def log_deprecation(old_name: str, new_name: str):
    """Log deprecation usage for metrics."""
    METRICS_FILE.parent.mkdir(exist_ok=True)
    
    usage = {
        "timestamp": datetime.now().isoformat(),
        "old_name": old_name,
        "new_name": new_name,
    }
    
    # Append to metrics file
    with open(METRICS_FILE, "a") as f:
        json.dump(usage, f)
        f.write("\n")

def get_deprecation_stats():
    """Get statistics on deprecated function usage."""
    if not METRICS_FILE.exists():
        return {}
    
    stats = {}
    with open(METRICS_FILE) as f:
        for line in f:
            usage = json.loads(line)
            old_name = usage["old_name"]
            stats[old_name] = stats.get(old_name, 0) + 1
    
    return stats
```

### Removal Timeline
- **v0.4.0** (Current): Add deprecation warnings
- **v0.5.0**: Show prominent warnings in CLI
- **v0.6.0**: Log usage statistics
- **v0.7.0**: Final warning before removal
- **v1.0.0**: Remove deprecated functions

## Success Metrics

### Code Quality
- [ ] All duplicate functions marked deprecated
- [ ] Clear deprecation warnings with alternatives
- [ ] Updated documentation and examples
- [ ] No internal usage of deprecated functions

### User Experience  
- [ ] Zero breaking changes
- [ ] Clear migration path
- [ ] Helpful warning messages
- [ ] Migration guide available

### Maintenance
- [ ] Single source of truth for each function
- [ ] Consistent naming convention
- [ ] Clean public API in __all__
- [ ] Metrics on deprecated function usage

## Risk Mitigation

### Gradual Deprecation
- Keep deprecated functions for 6+ months
- Multiple warning phases before removal
- Track usage metrics
- Communicate in release notes

### Clear Communication
- Deprecation warnings with alternatives
- Migration guide with examples
- Announcement in changelog
- Email to known users (if applicable)

### Rollback Plan
- If issues arise, extend deprecation period
- Can quickly restore aliases if needed
- Feature flag to disable warnings if needed

## Next Steps

1. **Review**: Get team approval on naming convention
2. **Implement Phase 1**: Create deprecation infrastructure
3. **Test**: Ensure deprecation system works correctly
4. **Apply Deprecations**: Mark all duplicate functions
5. **Update Docs**: Complete migration guide
6. **Internal Migration**: Update our own code
7. **Release v0.4.0**: With deprecation notices
8. **Monitor**: Track usage of deprecated functions
9. **Communicate**: Blog post or announcement about changes
10. **Remove in v1.0.0**: After sufficient warning period