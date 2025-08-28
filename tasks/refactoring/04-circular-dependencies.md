# Breaking Circular Dependencies Plan

## Executive Summary
The codebase has circular dependencies between Person ↔ Organization ↔ Index classes, creating tight coupling and testing difficulties. This refactoring will break these cycles using dependency injection and the Repository pattern.

**Impact**: Independent testing, reduced coupling, improved modularity
**Risk**: Low-Medium - Changes affect contributor management
**Effort**: 1 week including testing and migration

## Current State Analysis

### Circular Dependency Map
Based on PROJECT_INDEX.json analysis:

```
Person.organization() → Organization.get() → Index.get()
                ↑                               ↓
                └──────── Person.get() ←────────┘
```

### Problems

#### 1. Class-Level Registries
```python
# Current problematic code
class Organization:
    _organizations = {}  # Class-level state
    
    @classmethod
    def get(cls, code: str) -> Optional['Organization']:
        return cls._organizations.get(code)
    
class Person:
    def organization(self) -> Organization:
        return Organization.get(self.organization_code)  # Direct dependency
```

#### 2. Direct Index Dependency
```python
# Person references Index directly
class Person:
    def organization(self) -> Organization:
        # Some implementations also call Index.get_organization()
        return Organization.get(self.organization_code)
```

#### 3. Testing Difficulties
- Cannot test Person without Organization
- Cannot test Organization without registry
- Cannot mock dependencies easily
- Global state makes tests interdependent

## Target Architecture

### Design Patterns Applied
1. **Repository Pattern**: Separate data access from domain models
2. **Dependency Injection**: Inject dependencies rather than hardcode
3. **Protocol/Interface**: Define contracts for loose coupling
4. **Registry Pattern**: Centralized registry without global state

### New Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Person    │     │ Organization │     │ContribRepo  │
│  (Model)    │     │   (Model)    │     │ (Protocol)  │
└─────────────┘     └──────────────┘     └─────────────┘
                            ↑                    ↑
                            │                    │
                    ┌───────┴────────┐   ┌──────┴──────┐
                    │ContribRegistry │   │ MongoRepo   │
                    │ (Concrete)     │   │ (Concrete)  │
                    └────────────────┘   └─────────────┘
```

## Implementation Details

### Step 1: Define Domain Models (Pure Data)

```python
# src/findingmodel/contributor/models.py
from pydantic import BaseModel, Field, EmailStr, HttpUrl
from typing import Optional

class Organization(BaseModel):
    """Pure domain model for organization."""
    name: str = Field(..., min_length=5, max_length=200)
    code: str = Field(..., pattern=r'^[A-Z]{3,4}$')
    url: Optional[HttpUrl] = None
    
    class Config:
        frozen = True  # Immutable
    
    def __hash__(self):
        return hash(self.code)
    
    def __eq__(self, other):
        if not isinstance(other, Organization):
            return False
        return self.code == other.code

class Person(BaseModel):
    """Pure domain model for person."""
    github_username: str = Field(..., min_length=1, max_length=39)
    email: Optional[EmailStr] = None
    name: str = Field(..., min_length=1, max_length=100)
    organization_code: Optional[str] = Field(None, pattern=r'^[A-Z]{3,4}$')
    url: Optional[HttpUrl] = None
    
    class Config:
        frozen = True  # Immutable
    
    def __hash__(self):
        return hash(self.github_username)
    
    def __eq__(self, other):
        if not isinstance(other, Person):
            return False
        return self.github_username == other.github_username
```

### Step 2: Define Repository Protocol

```python
# src/findingmodel/contributor/protocols.py
from typing import Protocol, Optional, AsyncIterator
from .models import Person, Organization

class ContributorRepository(Protocol):
    """Protocol for contributor data access."""
    
    async def get_person(self, github_username: str) -> Optional[Person]:
        """Retrieve person by GitHub username."""
        ...
    
    async def get_organization(self, code: str) -> Optional[Organization]:
        """Retrieve organization by code."""
        ...
    
    async def save_person(self, person: Person) -> None:
        """Save or update a person."""
        ...
    
    async def save_organization(self, org: Organization) -> None:
        """Save or update an organization."""
        ...
    
    async def delete_person(self, github_username: str) -> bool:
        """Delete a person."""
        ...
    
    async def delete_organization(self, code: str) -> bool:
        """Delete an organization."""
        ...
    
    async def list_people(self) -> AsyncIterator[Person]:
        """List all people."""
        ...
    
    async def list_organizations(self) -> AsyncIterator[Organization]:
        """List all organizations."""
        ...
    
    async def get_person_organization(
        self, person: Person
    ) -> Optional[Organization]:
        """Get organization for a person."""
        if person.organization_code:
            return await self.get_organization(person.organization_code)
        return None
```

### Step 3: Implement In-Memory Repository

```python
# src/findingmodel/contributor/memory_repository.py
from typing import Optional, AsyncIterator, Dict
from .protocols import ContributorRepository
from .models import Person, Organization

class InMemoryContributorRepository:
    """In-memory implementation of ContributorRepository."""
    
    def __init__(self):
        self._people: Dict[str, Person] = {}
        self._organizations: Dict[str, Organization] = {}
    
    async def get_person(self, github_username: str) -> Optional[Person]:
        return self._people.get(github_username)
    
    async def get_organization(self, code: str) -> Optional[Organization]:
        return self._organizations.get(code)
    
    async def save_person(self, person: Person) -> None:
        self._people[person.github_username] = person
    
    async def save_organization(self, org: Organization) -> None:
        self._organizations[org.code] = org
    
    async def delete_person(self, github_username: str) -> bool:
        if github_username in self._people:
            del self._people[github_username]
            return True
        return False
    
    async def delete_organization(self, code: str) -> bool:
        if code in self._organizations:
            del self._organizations[code]
            return True
        return False
    
    async def list_people(self) -> AsyncIterator[Person]:
        for person in self._people.values():
            yield person
    
    async def list_organizations(self) -> AsyncIterator[Organization]:
        for org in self._organizations.values():
            yield org
    
    async def get_person_organization(
        self, person: Person
    ) -> Optional[Organization]:
        if person.organization_code:
            return await self.get_organization(person.organization_code)
        return None
```

### Step 4: Implement MongoDB Repository

```python
# src/findingmodel/contributor/mongo_repository.py
from typing import Optional, AsyncIterator
from motor.motor_asyncio import AsyncIOMotorDatabase
from .protocols import ContributorRepository
from .models import Person, Organization

class MongoContributorRepository:
    """MongoDB implementation of ContributorRepository."""
    
    def __init__(self, db: AsyncIOMotorDatabase, branch: str = "main"):
        self.people_collection = db[f"people_{branch}"]
        self.orgs_collection = db[f"organizations_{branch}"]
    
    async def get_person(self, github_username: str) -> Optional[Person]:
        doc = await self.people_collection.find_one(
            {"github_username": github_username}
        )
        return Person(**doc) if doc else None
    
    async def get_organization(self, code: str) -> Optional[Organization]:
        doc = await self.orgs_collection.find_one({"code": code})
        return Organization(**doc) if doc else None
    
    async def save_person(self, person: Person) -> None:
        await self.people_collection.replace_one(
            {"github_username": person.github_username},
            person.dict(),
            upsert=True
        )
    
    async def save_organization(self, org: Organization) -> None:
        await self.orgs_collection.replace_one(
            {"code": org.code},
            org.dict(),
            upsert=True
        )
    
    async def delete_person(self, github_username: str) -> bool:
        result = await self.people_collection.delete_one(
            {"github_username": github_username}
        )
        return result.deleted_count > 0
    
    async def delete_organization(self, code: str) -> bool:
        result = await self.orgs_collection.delete_one({"code": code})
        return result.deleted_count > 0
    
    async def list_people(self) -> AsyncIterator[Person]:
        async for doc in self.people_collection.find():
            yield Person(**doc)
    
    async def list_organizations(self) -> AsyncIterator[Organization]:
        async for doc in self.orgs_collection.find():
            yield Organization(**doc)
```

### Step 5: Create Service Layer

```python
# src/findingmodel/contributor/service.py
from typing import Optional, List
from .protocols import ContributorRepository
from .models import Person, Organization

class ContributorService:
    """Service layer for contributor operations."""
    
    def __init__(self, repository: ContributorRepository):
        self.repository = repository
    
    async def get_person_with_org(
        self, github_username: str
    ) -> tuple[Optional[Person], Optional[Organization]]:
        """Get person with their organization."""
        person = await self.repository.get_person(github_username)
        if not person:
            return None, None
        
        org = await self.repository.get_person_organization(person)
        return person, org
    
    async def create_person_with_org(
        self, 
        person: Person,
        org_code: Optional[str] = None
    ) -> Person:
        """Create person and ensure organization exists."""
        if org_code and not await self.repository.get_organization(org_code):
            raise ValueError(f"Organization {org_code} not found")
        
        await self.repository.save_person(person)
        return person
    
    async def get_organization_members(
        self, org_code: str
    ) -> List[Person]:
        """Get all members of an organization."""
        members = []
        async for person in self.repository.list_people():
            if person.organization_code == org_code:
                members.append(person)
        return members
    
    async def validate_contributors(
        self, contributors: List[Person | Organization]
    ) -> List[str]:
        """Validate a list of contributors."""
        errors = []
        
        for contributor in contributors:
            if isinstance(contributor, Person):
                # Check organization exists if specified
                if contributor.organization_code:
                    org = await self.repository.get_organization(
                        contributor.organization_code
                    )
                    if not org:
                        errors.append(
                            f"Organization {contributor.organization_code} "
                            f"not found for person {contributor.github_username}"
                        )
            
            elif isinstance(contributor, Organization):
                # Check for duplicate codes
                existing = await self.repository.get_organization(
                    contributor.code
                )
                if existing and existing != contributor:
                    errors.append(
                        f"Organization code {contributor.code} already exists"
                    )
        
        return errors
```

### Step 6: Update Index Class

```python
# src/findingmodel/index.py
from .contributor.protocols import ContributorRepository
from .contributor.mongo_repository import MongoContributorRepository

class Index:
    def __init__(
        self,
        mongodb_uri: str = None,
        db_name: str = None,
        contributor_repository: Optional[ContributorRepository] = None
    ):
        # ... existing initialization ...
        
        # Use injected repository or create default
        if contributor_repository:
            self.contributor_repo = contributor_repository
        else:
            # Create MongoDB repository as default
            self.contributor_repo = MongoContributorRepository(
                self.db, self.branch
            )
    
    async def get_person(self, github_username: str) -> Optional[Person]:
        """Delegate to repository."""
        return await self.contributor_repo.get_person(github_username)
    
    async def get_organization(self, code: str) -> Optional[Organization]:
        """Delegate to repository."""
        return await self.contributor_repo.get_organization(code)
    
    async def add_or_update_contributors(
        self, contributors: List[Person | Organization]
    ) -> Optional[List[str]]:
        """Add or update contributors using repository."""
        errors = []
        
        for contributor in contributors:
            try:
                if isinstance(contributor, Person):
                    await self.contributor_repo.save_person(contributor)
                elif isinstance(contributor, Organization):
                    await self.contributor_repo.save_organization(contributor)
            except Exception as e:
                errors.append(str(e))
        
        return errors if errors else None
```

### Step 7: Migration Strategy

```python
# src/findingmodel/contributor/migration.py
"""Migration utilities for backward compatibility."""

from typing import Dict, Any
from .models import Person, Organization
from .memory_repository import InMemoryContributorRepository

# Global registry for backward compatibility
_global_registry = InMemoryContributorRepository()

def register_organization(org: Organization) -> Organization:
    """Backward compatibility for org registration."""
    import warnings
    warnings.warn(
        "register_organization is deprecated. Use ContributorRepository instead.",
        DeprecationWarning,
        stacklevel=2
    )
    import asyncio
    asyncio.run(_global_registry.save_organization(org))
    return org

def get_organization(code: str) -> Optional[Organization]:
    """Backward compatibility for org retrieval."""
    import warnings
    warnings.warn(
        "get_organization is deprecated. Use ContributorRepository instead.",
        DeprecationWarning,
        stacklevel=2
    )
    import asyncio
    return asyncio.run(_global_registry.get_organization(code))

# Temporary backward compatibility
Organization.get = classmethod(lambda cls, code: get_organization(code))
Person.get = classmethod(
    lambda cls, username: asyncio.run(_global_registry.get_person(username))
)
```

## Testing Strategy

### Unit Tests with Mocking

```python
# test/test_contributor/test_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from findingmodel.contributor import (
    ContributorService,
    Person,
    Organization,
    ContributorRepository
)

@pytest.fixture
def mock_repository():
    """Create mock repository."""
    repo = Mock(spec=ContributorRepository)
    repo.get_person = AsyncMock(return_value=None)
    repo.get_organization = AsyncMock(return_value=None)
    repo.save_person = AsyncMock()
    repo.save_organization = AsyncMock()
    repo.get_person_organization = AsyncMock(return_value=None)
    return repo

@pytest.fixture
def service(mock_repository):
    """Create service with mock repository."""
    return ContributorService(mock_repository)

async def test_get_person_with_org(service, mock_repository):
    """Test retrieving person with organization."""
    person = Person(
        github_username="johndoe",
        name="John Doe",
        organization_code="TEST"
    )
    org = Organization(name="Test Org", code="TEST")
    
    mock_repository.get_person.return_value = person
    mock_repository.get_person_organization.return_value = org
    
    result_person, result_org = await service.get_person_with_org("johndoe")
    
    assert result_person == person
    assert result_org == org
    mock_repository.get_person.assert_called_once_with("johndoe")
    mock_repository.get_person_organization.assert_called_once_with(person)

async def test_create_person_validates_org(service, mock_repository):
    """Test that creating person validates organization exists."""
    mock_repository.get_organization.return_value = None
    
    person = Person(
        github_username="johndoe",
        name="John Doe",
        organization_code="INVALID"
    )
    
    with pytest.raises(ValueError, match="Organization INVALID not found"):
        await service.create_person_with_org(person, "INVALID")
```

### Integration Tests

```python
# test/test_contributor/test_integration.py
import pytest
from findingmodel.contributor import (
    InMemoryContributorRepository,
    MongoContributorRepository,
    ContributorService,
    Person,
    Organization
)

@pytest.fixture
async def memory_repo():
    """Create in-memory repository for testing."""
    return InMemoryContributorRepository()

@pytest.fixture
async def service(memory_repo):
    """Create service with in-memory repository."""
    return ContributorService(memory_repo)

async def test_full_workflow(service):
    """Test complete contributor workflow."""
    # Create organization
    org = Organization(name="Test Organization", code="TEST")
    await service.repository.save_organization(org)
    
    # Create person with organization
    person = Person(
        github_username="johndoe",
        name="John Doe",
        organization_code="TEST"
    )
    await service.create_person_with_org(person, "TEST")
    
    # Retrieve person with organization
    retrieved_person, retrieved_org = await service.get_person_with_org("johndoe")
    
    assert retrieved_person == person
    assert retrieved_org == org
    
    # Get organization members
    members = await service.get_organization_members("TEST")
    assert len(members) == 1
    assert members[0] == person
```

## Migration Plan

### Phase 1: Create New Structure (Day 1-2)
1. Create contributor package with models, protocols, repositories
2. Implement in-memory and MongoDB repositories
3. Add comprehensive tests

### Phase 2: Add Service Layer (Day 3)
1. Implement ContributorService
2. Add validation and business logic
3. Test service layer

### Phase 3: Update Index Class (Day 4)
1. Add repository injection to Index
2. Update contributor methods to use repository
3. Maintain backward compatibility

### Phase 4: Migration Utilities (Day 5)
1. Create migration helpers
2. Add deprecation warnings
3. Update documentation

### Phase 5: Update Existing Code (Day 6-7)
1. Find and update all uses of Person.get(), Organization.get()
2. Update tests to use new structure
3. Run full test suite

## Success Metrics

### Code Quality
- [ ] No circular dependencies
- [ ] All tests can run independently
- [ ] Easy to mock dependencies
- [ ] Clear separation of concerns

### Testing
- [ ] 100% test coverage for new code
- [ ] All existing tests still pass
- [ ] New tests for repository implementations
- [ ] Integration tests for service layer

### Architecture
- [ ] Models have no dependencies
- [ ] Repository protocol clearly defined
- [ ] Service layer handles business logic
- [ ] Index class properly delegates

## Risk Mitigation

### Backward Compatibility
- Keep old methods with deprecation warnings
- Migration utilities for gradual transition
- Feature flag for new vs old implementation
- Comprehensive testing of both paths

### Data Migration
- No data structure changes needed
- Repository abstraction handles data access
- Can switch between implementations easily

### Rollback Plan
- Keep old code in separate module
- Feature flag to switch implementations
- Can revert to old code quickly if needed

## Next Steps

1. **Review**: Get team feedback on architecture
2. **Create Branch**: `feature/break-circular-deps`
3. **Implement Models**: Pure domain models first
4. **Add Repositories**: Both implementations
5. **Service Layer**: Business logic
6. **Integration**: Update Index class
7. **Migration**: Backward compatibility
8. **Testing**: Comprehensive test coverage
9. **Documentation**: Update all docs
10. **Deploy**: Gradual rollout with monitoring