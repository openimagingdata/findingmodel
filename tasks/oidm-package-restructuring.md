# Plan: OIDM Package Restructuring

**Status:** Planning
**Related:** [anatomic-locations-normalized-schema.md](anatomic-locations-normalized-schema.md)

## Goal

Restructure the OIDM ecosystem into three separate packages with shared utilities published to PyPI:

```
oidm-core-utils/          # Shared utilities (PyPI)
findingmodel/             # Finding models (PyPI) - depends on oidm-core-utils
anatomic-locations/       # Anatomic locations (PyPI) - depends on oidm-core-utils
```

## Phased Approach

### Phase 1: Implement Anatomic Locations In-Place (Current)

Build anatomic locations within the existing `findingmodel` repository first. This allows:
- Faster iteration without repo setup overhead
- Reuse existing CI/CD, testing infrastructure
- Validate the design before extracting

**Deliverables:**
- `src/findingmodel/anatomic_location.py` - Pydantic models, enums
- `src/findingmodel/anatomic_index.py` - AnatomicLocationIndex
- `src/findingmodel/anatomic_migration.py` - Database builder (updated)
- Tests and CLI commands

**See:** [anatomic-locations-normalized-schema.md](anatomic-locations-normalized-schema.md) for implementation details.

### Phase 2: Extract oidm-core-utils

Create a new repository with shared utilities. Publish to PyPI.

**What moves to oidm-core-utils:**

| Current Location | New Location | Notes |
|------------------|--------------|-------|
| `findingmodel/index_code.py` | `oidm_core_utils/index_code.py` | Core data model |
| `findingmodel/tools/embeddings.py` | `oidm_core_utils/embeddings.py` | Multi-provider embedding |
| DuckDB search patterns | `oidm_core_utils/duckdb/` | Hybrid search, FTS, HNSW |
| DuckDB download/versioning | `oidm_core_utils/database.py` | Download, version check, caching |
| Config utilities | `oidm_core_utils/config.py` | Paths, environment |

**Database management (shared by both libraries):**

Both findingmodel and anatomic-locations need to:
- Read `manifest.json` to find database URLs and versions
- Download pre-built DuckDB files from remote storage
- Check manifest for newer versions and re-download when updated
- Cache databases locally with proper paths
- Handle connection management (open/close, read-only mode)

The `manifest.json` references all OIDM databases (finding models index, anatomic locations, etc.) with their URLs, versions, and checksums. This manifest handling logic should be factored into `oidm_core_utils/database.py` with a common interface that both libraries can use.

**Dependencies for oidm-core-utils:**
```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "duckdb>=1.0",
    "openai>=1.0",        # For embeddings
    "anthropic>=0.20",    # For embeddings (optional?)
    "httpx",              # For downloads
]
```

**Steps:**
1. Create `oidm-core-utils` repository
2. Move shared code with clean interfaces
3. Write tests for extracted code
4. Publish v0.1.0 to PyPI
5. Update `findingmodel` to depend on `oidm-core-utils>=0.1.0`
6. Remove extracted code from findingmodel
7. Update imports throughout findingmodel

### Phase 3: Extract anatomic-locations

Create a new repository for anatomic locations. Publish to PyPI.

**What moves to anatomic-locations:**

| Current Location | New Location |
|------------------|--------------|
| `findingmodel/anatomic_location.py` | `anatomic_locations/models.py` |
| `findingmodel/anatomic_index.py` | `anatomic_locations/index.py` |
| `findingmodel/anatomic_migration.py` | `anatomic_locations/migration.py` |
| Anatomic CLI commands | `anatomic_locations/cli.py` |

**Dependencies for anatomic-locations:**
```toml
[project]
dependencies = [
    "oidm-core-utils>=0.1.0",
    "pydantic>=2.0",
    "duckdb>=1.0",
]
```

**Steps:**
1. Create `anatomic-locations` repository
2. Move anatomic code
3. Update imports to use `oidm_core_utils`
4. Write/move tests
5. Publish v0.1.0 to PyPI
6. Update `findingmodel` to optionally depend on `anatomic-locations`
7. Remove anatomic code from findingmodel

---

## Final Architecture

```
oidm-core-utils (PyPI)
├── oidm_core_utils/
│   ├── __init__.py
│   ├── index_code.py        # IndexCode model
│   ├── embeddings.py        # Multi-provider embeddings
│   ├── config.py            # Paths, environment
│   ├── database.py          # Download, versioning, caching
│   └── duckdb/
│       ├── __init__.py
│       ├── search.py        # Hybrid search (FTS + vector)
│       └── connection.py    # Connection management, read-only mode
└── pyproject.toml

anatomic-locations (PyPI)
├── anatomic_locations/
│   ├── __init__.py
│   ├── models.py            # AnatomicLocation, enums
│   ├── index.py             # AnatomicLocationIndex
│   ├── migration.py         # Database builder
│   └── cli.py               # CLI commands
├── pyproject.toml
└── Dependencies: oidm-core-utils

findingmodel (PyPI)
├── findingmodel/
│   ├── __init__.py
│   ├── finding_model.py     # FindingModel, FindingModelFull
│   ├── finding_info.py
│   ├── tools/               # AI workflows
│   └── ...
├── pyproject.toml
└── Dependencies: oidm-core-utils, anatomic-locations (optional)
```

---

## Version Coordination

When `oidm-core-utils` changes:

```
1. PR to oidm-core-utils → merge → release (e.g., v0.2.0)
2. PR to findingmodel → bump oidm-core-utils>=0.2.0 → merge → release
3. PR to anatomic-locations → bump oidm-core-utils>=0.2.0 → merge → release
```

Since oidm-core-utils is **stable infrastructure**, this should be infrequent.

---

## Benefits of This Approach

1. **Standard tooling** - No Una, no monorepo hacks, just pip/uv
2. **VS Code works** - Open one repo at a time, Pylance happy
3. **Independent versioning** - Each package on its own release cycle
4. **Clear boundaries** - Each repo has single responsibility
5. **Simpler CI/CD** - Per-repo pipelines, no cross-package coordination

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking changes in oidm-core-utils | Semantic versioning, careful API design |
| Version drift between packages | Dependabot/Renovate for auto-updates |
| Duplicate testing across repos | Integration tests in each consumer repo |
| Onboarding complexity (3 repos) | Good documentation, clear README in each |

---

## Open Questions

1. **Embedding provider dependency** - Should oidm-core-utils require both openai and anthropic, or make them optional extras?
2. **CLI ownership** - Does anatomic-locations have its own CLI, or does findingmodel expose anatomic commands?

---

## Timeline

No time estimates per project policy. Order of operations:

1. **Now:** Implement anatomic locations in findingmodel (Phase 1)
2. **After AL is working:** Extract oidm-core-utils (Phase 2)
3. **After oidm-core-utils is published:** Extract anatomic-locations (Phase 3)
