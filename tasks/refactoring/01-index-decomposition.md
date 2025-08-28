# Index Class Decomposition Plan

## Executive Summary
The Index class in `src/findingmodel/index.py` has grown to 34+ methods with 700+ lines, violating Single Responsibility Principle. This refactoring will decompose it into 5 focused classes with clear responsibilities.

**Impact**: 60% complexity reduction, improved testability, enables parallel development
**Risk**: Medium - Core functionality used throughout codebase
**Effort**: 1-2 weeks with incremental approach

## Current State Analysis

### Problems
- **God Object**: 34 methods handling database, validation, file I/O, search, and batch operations
- **Deep Call Chains**: 5-level validation chains creating tight coupling
- **Mixed Concerns**: Database operations intertwined with business logic
- **Testing Difficulty**: Hard to mock or test individual responsibilities
- **High Coupling**: 60+ calls to/from other modules based on PROJECT_INDEX analysis

### Key Methods by Responsibility
```python
# Database Operations (10 methods)
- __init__, setup_indexes, count, count_people, count_organizations
- get, get_person, get_organization, _get_existing_file_info
- _execute_batch_operations

# Validation (6 methods)  
- validate_model, validate_models_batch
- _get_validation_data, _check_id_conflict
- _check_name_conflict, _check_attribute_id_conflict

# File Operations (5 methods)
- _calculate_file_hash, _entry_from_model_file
- _get_local_file_info, _determine_operations
- update_from_directory

# Search (4 methods)
- search, search_batch
- _search_batch_combined, _search_batch_individual
- _entry_matches_query

# Other (9 methods)
- contains, add_or_update_contributors
- add_or_update_entry_from_file, remove_entry
- remove_unused_entries, _prepare_entries_for_batch
- to_markdown, _id_or_name_or_syn_query
```

## Target Architecture

### Component Breakdown

#### 1. IndexRepository (Database Layer)
```python
# src/findingmodel/index/repository.py
class IndexRepository:
    """Pure database operations for finding models."""
    
    def __init__(self, mongodb_uri: str, db_name: str, branch: str = "main")
    async def setup_indexes() -> None
    async def get_by_id(oifm_id: str) -> IndexEntry | None
    async def get_by_name(name: str) -> IndexEntry | None
    async def get_by_id_or_name_or_synonym(query: str) -> IndexEntry | None
    async def count() -> int
    async def count_people() -> int
    async def count_organizations() -> int
    async def get_person(github_username: str) -> Person | None
    async def get_organization(code: str) -> Organization | None
    async def insert_entry(entry: dict) -> str
    async def update_entry(oifm_id: str, entry: dict) -> bool
    async def delete_entry(oifm_id: str) -> bool
    async def bulk_write(operations: list) -> BulkWriteResult
    async def find(filter: dict, limit: int) -> list[IndexEntry]
    async def get_all_validation_data() -> tuple[dict, dict, dict]
```

#### 2. ModelValidator (Business Logic)
```python
# src/findingmodel/index/validator.py
class ModelValidator:
    """Validation logic for finding models."""
    
    def __init__(self, repository: IndexRepository)
    async def validate_model(model: FindingModelFull) -> list[str]
    async def validate_batch(models: list[FindingModelFull]) -> dict[str, list[str]]
    def check_id_conflicts(model, existing_data) -> list[str]
    def check_name_conflicts(model, existing_data) -> list[str]
    def check_attribute_conflicts(model, existing_data) -> list[str]
    def check_synonym_conflicts(model, existing_data) -> list[str]
```

#### 3. FileManager (File I/O)
```python
# src/findingmodel/index/file_manager.py
class FileManager:
    """File operations for finding models."""
    
    @staticmethod
    def calculate_hash(file_path: Path) -> str
    @staticmethod
    def load_model(file_path: Path) -> FindingModelFull
    @staticmethod
    def save_model(model: FindingModelFull, file_path: Path) -> None
    @staticmethod
    def scan_directory(path: Path, pattern: str = "*.fm.json") -> list[Path]
    @staticmethod
    def create_entry_from_model(model: FindingModelFull, file_path: Path) -> dict
    def get_file_info(files: list[Path]) -> dict[str, dict]
```

#### 4. SearchEngine (Query Operations)
```python
# src/findingmodel/index/search_engine.py
class SearchEngine:
    """Search operations for finding models."""
    
    def __init__(self, repository: IndexRepository)
    async def search(query: str, limit: int = 10) -> list[IndexEntry]
    async def search_batch(queries: list[str]) -> dict[str, list[IndexEntry]]
    async def search_by_field(field: str, value: str) -> list[IndexEntry]
    def build_search_filter(query: str, include_synonyms: bool) -> dict
    def matches_query(entry: IndexEntry, query: str) -> bool
```

#### 5. Index (Facade/Orchestrator)
```python
# src/findingmodel/index/index.py
class Index:
    """Facade for finding model index operations."""
    
    def __init__(self, mongodb_uri: str = None, db_name: str = None, branch: str = "main")
        self.repository = IndexRepository(...)
        self.validator = ModelValidator(self.repository)
        self.file_manager = FileManager()
        self.search_engine = SearchEngine(self.repository)
        
    # Delegate methods maintaining current API
    async def setup_indexes() -> None
    async def get(id_or_name: str) -> IndexEntry | None
    async def validate_model(model: FindingModelFull) -> list[str]
    async def search(query: str, limit: int = 10) -> list[IndexEntry]
    async def update_from_directory(directory: Path) -> dict
    async def add_or_update_entry_from_file(file: Path) -> tuple
    # ... other delegating methods
```

## Implementation Plan

### Phase 1: Setup Infrastructure (Day 1-2)
1. Create new directory structure:
   ```
   src/findingmodel/index/
   ├── __init__.py          # Re-export Index class
   ├── repository.py        # Database operations
   ├── validator.py         # Validation logic
   ├── file_manager.py      # File operations
   ├── search_engine.py     # Search functionality
   ├── index.py            # Facade class
   └── types.py            # Shared types/protocols
   ```

2. Create interfaces/protocols:
   ```python
   # src/findingmodel/index/types.py
   from typing import Protocol
   
   class RepositoryProtocol(Protocol):
       async def get_by_id(self, oifm_id: str) -> IndexEntry | None: ...
       # ... other methods
   ```

### Phase 2: Extract Repository (Day 3-4)
1. Move database operations to IndexRepository
2. Keep original methods as delegates in Index
3. Add comprehensive tests for IndexRepository
4. Run existing tests to ensure no breakage

### Phase 3: Extract Validator (Day 5-6)
1. Move validation logic to ModelValidator
2. Refactor to use repository for data access
3. Simplify validation chains
4. Update Index to use ModelValidator
5. Run full test suite

### Phase 4: Extract FileManager (Day 7)
1. Move file operations to FileManager
2. Make methods static where appropriate
3. Update Index to use FileManager
4. Test file operations independently

### Phase 5: Extract SearchEngine (Day 8)
1. Move search logic to SearchEngine
2. Optimize batch search operations
3. Update Index to use SearchEngine
4. Test search functionality

### Phase 6: Refactor Index as Facade (Day 9-10)
1. Remove all direct implementation from Index
2. Keep only delegation and orchestration logic
3. Ensure backward compatibility
4. Run full test suite

### Phase 7: Optimization & Cleanup (Day 11-12)
1. Remove dead code
2. Optimize imports
3. Update documentation
4. Performance testing
5. Create migration guide

## Testing Strategy

### Unit Tests
- Each new class gets dedicated test file
- Mock dependencies using protocols
- Test edge cases and error conditions
- Maintain 100% coverage for new code

### Integration Tests  
- Test component interactions
- Verify facade delegates correctly
- Test with real MongoDB instance
- Ensure backward compatibility

### Performance Tests
- Benchmark before/after refactoring
- Test batch operations at scale
- Monitor memory usage
- Profile database queries

## Risk Mitigation

### Backward Compatibility
- Keep Index class as public API
- All existing methods remain available
- Use facade pattern for seamless migration
- Deprecate rather than remove if needed

### Incremental Approach
- Each phase is independently deployable
- Tests run after each phase
- Rollback plan for each phase
- Feature flags for gradual rollout

### Testing Coverage
- Existing 34 tests in test_index.py
- Add tests for each new component
- Integration tests for interactions
- No production deployment until all tests pass

## Success Metrics

### Code Quality
- [ ] Reduce Index class from 700+ to <100 lines
- [ ] Each new class has single responsibility
- [ ] No method exceeds 20 lines
- [ ] Cyclomatic complexity < 5 per method

### Testing
- [ ] All existing tests pass
- [ ] New unit tests for each component
- [ ] Integration tests for facade
- [ ] Performance benchmarks show no regression

### Developer Experience
- [ ] Clear separation of concerns
- [ ] Easy to understand and modify
- [ ] Better IDE support and autocomplete
- [ ] Simplified debugging

## Migration Guide

### For Developers
```python
# Old way (still works)
from findingmodel.index import Index
index = Index()
await index.validate_model(model)

# New way (optional, for direct access)
from findingmodel.index import ModelValidator, IndexRepository
repo = IndexRepository(...)
validator = ModelValidator(repo)
await validator.validate_model(model)
```

### For Tests
```python
# Easy mocking with new structure
from unittest.mock import Mock
from findingmodel.index import ModelValidator

mock_repo = Mock(spec=IndexRepository)
mock_repo.get_all_validation_data.return_value = ({}, {}, {})
validator = ModelValidator(mock_repo)
# Test validator independently
```

## Next Steps

1. **Review & Approval**: Get team feedback on architecture
2. **Create Branch**: `feature/index-decomposition`
3. **Start Phase 1**: Set up infrastructure
4. **Daily Progress**: Update team on completion of each phase
5. **Code Review**: After each phase for early feedback
6. **Documentation**: Update as we go
7. **Performance Testing**: Before final merge

## Related Refactorings

This refactoring enables:
- Validation Framework extraction (see 03-validation-framework.md)
- Performance optimizations through better batching
- Easier testing and mocking
- Future MongoDB abstraction if needed