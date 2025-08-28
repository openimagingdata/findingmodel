# Validation Framework Plan

## Executive Summary
The current validation logic is scattered across multiple methods with deep 5-level call chains. This refactoring will create a unified, extensible validation framework using the Chain of Responsibility pattern.

**Impact**: Reduced complexity, extensible validation rules, better testability
**Risk**: Medium - Core validation logic affects data integrity
**Effort**: 1 week including comprehensive testing

## Current State Analysis

### Problems with Current Validation

#### Deep Call Chains (5 levels)
```
add_or_update_entry_from_file()
  └─> validate_model()
      └─> validate_models_batch()
          └─> _get_validation_data()
              └─> get()
```

#### Scattered Validation Logic
- `_check_id_conflict()` - 10 lines
- `_check_name_conflict()` - 20 lines  
- `_check_attribute_id_conflict()` - 16 lines
- Synonym validation mixed into other checks
- No central place to add new validations

#### Tight Coupling
- Validation methods directly access database
- Hard to test individual validation rules
- Cannot reuse validation outside Index class
- Difficult to customize validation per use case

## Target Architecture

### Core Design Pattern: Chain of Responsibility

```python
# src/findingmodel/validation/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class ValidationContext:
    """Context passed through validation chain."""
    model: FindingModelFull
    existing_data: dict[str, Any]
    exclude_id: Optional[str] = None
    allow_duplicates: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str, field: str = None):
        """Add validation error with optional field context."""
        if field:
            error = f"[{field}] {error}"
        self.errors.append(error)
    
    def add_warning(self, warning: str, field: str = None):
        """Add validation warning."""
        if field:
            warning = f"[{field}] {warning}"
        self.warnings.append(warning)
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

class Validator(ABC):
    """Base validator in chain of responsibility."""
    
    def __init__(self, next_validator: Optional['Validator'] = None):
        self.next = next_validator
        self.enabled = True
        self.severity = "error"  # error, warning, info
    
    async def validate(self, context: ValidationContext) -> ValidationContext:
        """Execute validation and pass to next in chain."""
        if self.enabled:
            try:
                await self._validate_impl(context)
            except Exception as e:
                context.add_error(f"Validation failed: {str(e)}")
        
        if self.next:
            return await self.next.validate(context)
        return context
    
    @abstractmethod
    async def _validate_impl(self, context: ValidationContext) -> None:
        """Implement specific validation logic."""
        pass
    
    def set_next(self, validator: 'Validator') -> 'Validator':
        """Set next validator in chain (fluent interface)."""
        self.next = validator
        return validator
```

### Specific Validators

#### 1. ID Validator
```python
# src/findingmodel/validation/validators/id_validator.py
class IdValidator(Validator):
    """Validates OIFM ID format and uniqueness."""
    
    ID_PATTERN = re.compile(r'^OIFM_[A-Z]{3,4}_\d{6}$')
    
    async def _validate_impl(self, context: ValidationContext) -> None:
        model = context.model
        
        # Format validation
        if not self.ID_PATTERN.match(model.oifm_id):
            context.add_error(
                f"Invalid ID format: {model.oifm_id}. "
                f"Expected: OIFM_{{ORG}}_{{6-digits}}",
                field="oifm_id"
            )
        
        # Uniqueness validation
        existing_ids = context.existing_data.get('ids', {})
        if model.oifm_id in existing_ids:
            conflicting_file = existing_ids[model.oifm_id]
            if conflicting_file != context.exclude_id:
                context.add_error(
                    f"ID '{model.oifm_id}' already exists in {conflicting_file}",
                    field="oifm_id"
                )
```

#### 2. Name Validator
```python
# src/findingmodel/validation/validators/name_validator.py
class NameValidator(Validator):
    """Validates finding name constraints."""
    
    MIN_LENGTH = 3
    MAX_LENGTH = 100
    
    async def _validate_impl(self, context: ValidationContext) -> None:
        model = context.model
        
        # Length validation
        if len(model.name) < self.MIN_LENGTH:
            context.add_error(
                f"Name too short: {len(model.name)} < {self.MIN_LENGTH}",
                field="name"
            )
        
        if len(model.name) > self.MAX_LENGTH:
            context.add_warning(
                f"Name very long: {len(model.name)} > {self.MAX_LENGTH}",
                field="name"
            )
        
        # Uniqueness validation
        name_lower = model.name.lower()
        existing_names = context.existing_data.get('names', {})
        
        if name_lower in existing_names:
            conflicting_id = existing_names[name_lower]
            if conflicting_id != model.oifm_id:
                context.add_error(
                    f"Name '{model.name}' already used by {conflicting_id}",
                    field="name"
                )
```

#### 3. Attribute Validator
```python
# src/findingmodel/validation/validators/attribute_validator.py
class AttributeValidator(Validator):
    """Validates attribute constraints."""
    
    ATTR_ID_PATTERN = re.compile(r'^OIFMA_[A-Z]{3,4}_\d{6}$')
    
    async def _validate_impl(self, context: ValidationContext) -> None:
        model = context.model
        existing_attrs = context.existing_data.get('attributes', {})
        
        seen_ids = set()
        seen_names = set()
        
        for attr in model.attributes:
            # ID format validation
            if hasattr(attr, 'oifma_id'):
                if not self.ATTR_ID_PATTERN.match(attr.oifma_id):
                    context.add_error(
                        f"Invalid attribute ID: {attr.oifma_id}",
                        field=f"attributes.{attr.name}"
                    )
                
                # Duplicate within model
                if attr.oifma_id in seen_ids:
                    context.add_error(
                        f"Duplicate attribute ID: {attr.oifma_id}",
                        field=f"attributes.{attr.name}"
                    )
                seen_ids.add(attr.oifma_id)
                
                # Global uniqueness
                if attr.oifma_id in existing_attrs:
                    conflicting = existing_attrs[attr.oifma_id]
                    if conflicting != model.oifm_id:
                        context.add_error(
                            f"Attribute ID {attr.oifma_id} used by {conflicting}",
                            field=f"attributes.{attr.name}"
                        )
            
            # Name uniqueness within model
            name_lower = attr.name.lower()
            if name_lower in seen_names:
                context.add_error(
                    f"Duplicate attribute name: {attr.name}",
                    field=f"attributes.{attr.name}"
                )
            seen_names.add(name_lower)
```

#### 4. Business Rule Validators
```python
# src/findingmodel/validation/validators/business_rules.py
class RequiredAttributesValidator(Validator):
    """Ensures required attributes are present."""
    
    REQUIRED_ATTRIBUTES = ['presence', 'change_from_prior']
    
    async def _validate_impl(self, context: ValidationContext) -> None:
        model = context.model
        attr_names = {attr.name.lower() for attr in model.attributes}
        
        for required in self.REQUIRED_ATTRIBUTES:
            if required not in attr_names:
                context.add_warning(
                    f"Missing recommended attribute: {required}",
                    field="attributes"
                )

class ChoiceAttributeValidator(Validator):
    """Validates choice attribute constraints."""
    
    async def _validate_impl(self, context: ValidationContext) -> None:
        for attr in context.model.attributes:
            if isinstance(attr, ChoiceAttribute):
                # At least 2 choices
                if len(attr.values) < 2:
                    context.add_error(
                        f"Choice attribute '{attr.name}' needs at least 2 values",
                        field=f"attributes.{attr.name}"
                    )
                
                # Max selected validation
                if attr.max_selected > len(attr.values):
                    context.add_error(
                        f"max_selected ({attr.max_selected}) > values ({len(attr.values)})",
                        field=f"attributes.{attr.name}"
                    )

class CircularReferenceValidator(Validator):
    """Detects circular references in models."""
    
    async def _validate_impl(self, context: ValidationContext) -> None:
        # Check for circular references in related models
        # Implementation depends on relationship structure
        pass
```

### Validation Chain Factory

```python
# src/findingmodel/validation/factory.py
from typing import Optional

class ValidationChainFactory:
    """Factory for creating validation chains."""
    
    @staticmethod
    def create_default_chain() -> Validator:
        """Create standard validation chain for finding models."""
        return (
            IdValidator()
            .set_next(NameValidator())
            .set_next(AttributeValidator())
            .set_next(RequiredAttributesValidator())
            .set_next(ChoiceAttributeValidator())
        )
    
    @staticmethod
    def create_strict_chain() -> Validator:
        """Create strict validation with additional checks."""
        return (
            IdValidator()
            .set_next(NameValidator())
            .set_next(AttributeValidator())
            .set_next(SynonymValidator())
            .set_next(RequiredAttributesValidator())
            .set_next(ChoiceAttributeValidator())
            .set_next(NumericAttributeValidator())
            .set_next(IndexCodeValidator())
            .set_next(CircularReferenceValidator())
        )
    
    @staticmethod
    def create_minimal_chain() -> Validator:
        """Create minimal validation for drafts."""
        return (
            IdValidator()
            .set_next(NameValidator())
        )
    
    @staticmethod
    def create_custom_chain(validators: list[Validator]) -> Optional[Validator]:
        """Create custom chain from validator list."""
        if not validators:
            return None
        
        for i in range(len(validators) - 1):
            validators[i].set_next(validators[i + 1])
        
        return validators[0]
```

### Validation Service

```python
# src/findingmodel/validation/service.py
class ValidationService:
    """High-level validation service."""
    
    def __init__(self, repository: Optional[Any] = None):
        self.repository = repository
        self.cache = {}
        self.chain_factory = ValidationChainFactory()
    
    async def validate_model(
        self,
        model: FindingModelFull,
        mode: str = "default",
        exclude_id: Optional[str] = None,
        existing_data: Optional[dict] = None
    ) -> ValidationResult:
        """Validate a single model."""
        
        # Get or fetch existing data
        if existing_data is None:
            existing_data = await self._get_existing_data()
        
        # Create validation context
        context = ValidationContext(
            model=model,
            existing_data=existing_data,
            exclude_id=exclude_id
        )
        
        # Select validation chain
        chain = self._get_chain(mode)
        
        # Run validation
        context = await chain.validate(context)
        
        # Return result
        return ValidationResult(
            is_valid=context.is_valid,
            errors=context.errors,
            warnings=context.warnings,
            metadata=context.metadata
        )
    
    async def validate_batch(
        self,
        models: list[FindingModelFull],
        mode: str = "default"
    ) -> dict[str, ValidationResult]:
        """Validate multiple models efficiently."""
        
        # Fetch existing data once
        existing_data = await self._get_existing_data()
        
        # Validate each model
        results = {}
        for model in models:
            result = await self.validate_model(
                model, mode, existing_data=existing_data
            )
            results[model.oifm_id] = result
        
        return results
    
    def _get_chain(self, mode: str) -> Validator:
        """Get validation chain for mode."""
        if mode == "strict":
            return self.chain_factory.create_strict_chain()
        elif mode == "minimal":
            return self.chain_factory.create_minimal_chain()
        else:
            return self.chain_factory.create_default_chain()
    
    async def _get_existing_data(self) -> dict:
        """Fetch existing data for validation."""
        if self.repository:
            return await self.repository.get_all_validation_data()
        return {}

@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    metadata: dict[str, Any]
    
    def raise_if_invalid(self):
        """Raise exception if validation failed."""
        if not self.is_valid:
            raise ValidationError(self.errors)
```

## Implementation Plan

### Phase 1: Core Infrastructure (Day 1-2)
1. Create validation package structure:
   ```
   src/findingmodel/validation/
   ├── __init__.py
   ├── base.py              # Base classes and protocols
   ├── context.py           # ValidationContext
   ├── factory.py           # Chain factory
   ├── service.py           # ValidationService
   ├── validators/
   │   ├── __init__.py
   │   ├── id_validator.py
   │   ├── name_validator.py
   │   ├── attribute_validator.py
   │   ├── business_rules.py
   │   └── custom.py
   └── exceptions.py
   ```

2. Implement base classes and context

### Phase 2: Core Validators (Day 3-4)
1. Implement ID, Name, and Attribute validators
2. Add comprehensive unit tests
3. Ensure 100% test coverage

### Phase 3: Business Rule Validators (Day 5)
1. Implement required attributes validator
2. Implement choice/numeric validators
3. Add index code validators
4. Test business rules

### Phase 4: Integration (Day 6)
1. Update Index class to use ValidationService
2. Maintain backward compatibility
3. Run all existing tests
4. Performance benchmarks

### Phase 5: Advanced Features (Day 7)
1. Add custom validator support
2. Implement validation profiles
3. Add async validation support
4. Create validation report generator

## Testing Strategy

### Unit Tests for Each Validator
```python
# test/test_validation/test_id_validator.py
import pytest
from findingmodel.validation import IdValidator, ValidationContext

@pytest.fixture
def validator():
    return IdValidator()

@pytest.fixture
def context():
    return ValidationContext(
        model=FindingModelFull(oifm_id="OIFM_TEST_123456"),
        existing_data={}
    )

async def test_valid_id_format(validator, context):
    await validator._validate_impl(context)
    assert context.is_valid

async def test_invalid_id_format(validator, context):
    context.model.oifm_id = "BAD_ID"
    await validator._validate_impl(context)
    assert not context.is_valid
    assert "Invalid ID format" in context.errors[0]

async def test_duplicate_id(validator, context):
    context.existing_data = {
        'ids': {'OIFM_TEST_123456': 'other_file.json'}
    }
    await validator._validate_impl(context)
    assert not context.is_valid
    assert "already exists" in context.errors[0]
```

### Integration Tests
```python
# test/test_validation/test_validation_service.py
async def test_validation_service_default_chain():
    service = ValidationService()
    model = create_test_model()
    
    result = await service.validate_model(model)
    assert result.is_valid

async def test_validation_service_batch():
    service = ValidationService()
    models = [create_test_model(i) for i in range(10)]
    
    results = await service.validate_batch(models)
    assert len(results) == 10
    assert all(r.is_valid for r in results.values())
```

## Migration Strategy

### Gradual Migration
```python
# src/findingmodel/index.py (temporary during migration)
class Index:
    def __init__(self, ...):
        # Keep old validation for compatibility
        self._use_new_validation = feature_flag('new_validation')
        self.validation_service = ValidationService(self.repository)
    
    async def validate_model(self, model, ...):
        if self._use_new_validation:
            result = await self.validation_service.validate_model(model)
            return result.errors  # Match old interface
        else:
            # Old validation code
            return await self._old_validate_model(model)
```

### Feature Flags
- Start with 10% of validation using new framework
- Monitor for issues
- Gradually increase to 100%
- Remove old code after stability confirmed

## Benefits & Impact

### Immediate Benefits
- **Testability**: Each validator tested independently
- **Extensibility**: Easy to add new validators
- **Reusability**: Validation logic can be used elsewhere
- **Performance**: Batch validation optimized

### Long-term Benefits
- **Maintainability**: Clear separation of concerns
- **Customization**: Different validation modes
- **Debugging**: Better error messages with context
- **Documentation**: Self-documenting validator classes

### Performance Impact
- **Before**: 5-level call chain, multiple DB queries
- **After**: 2-level chain, single DB query
- **Expected**: 30-40% faster for batch validation

## Success Metrics

### Code Quality
- [ ] Reduce validation complexity from O(n²) to O(n)
- [ ] Each validator under 50 lines
- [ ] 100% test coverage for validators
- [ ] No validation logic in Index class

### Functionality
- [ ] All existing validations work
- [ ] New validators easy to add
- [ ] Validation modes (strict/default/minimal)
- [ ] Better error messages

### Performance
- [ ] Batch validation 30% faster
- [ ] Single validation no slower
- [ ] Memory usage stable
- [ ] Database queries minimized

## Next Steps

1. **Review**: Get feedback on design
2. **Prototype**: Build proof of concept
3. **Test**: Comprehensive testing
4. **Benchmark**: Performance comparison
5. **Integrate**: Update Index class
6. **Document**: API documentation
7. **Deploy**: Gradual rollout with monitoring