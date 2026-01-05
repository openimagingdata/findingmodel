# Plan: anatomic-locations Package

**Status:** Planning
**Related:** [oidm-package-restructuring.md](oidm-package-restructuring.md), [oidm-common-package-plan.md](oidm-common-package-plan.md)
**Implementation Reference:** [done/anatomic-locations-normalized-schema.md](done/anatomic-locations-normalized-schema.md)

## Purpose

Standalone package for querying and navigating anatomic locations. Provides hierarchy traversal, laterality variants, semantic search, and code lookups across medical ontologies.

## Package Structure

**Repository/package naming (PEP 8 aligned)**
- Repo: `anatomic-locations`
- Python package: `anatomic_locations`
- Source root: `src/anatomic_locations/`
```
anatomic-locations/
├── src/
│   └── anatomic_locations/
│       ├── __init__.py           # Public API exports
│       ├── models/
│       │   ├── __init__.py
│       │   ├── location.py       # AnatomicLocation, AnatomicRef
│       │   └── enums.py          # AnatomicRegion, Laterality, LocationType, etc.
│       ├── index.py              # AnatomicLocationIndex
│       ├── migration.py          # Database build from JSON source
│       ├── config.py             # Settings, ensure_db()
│       └── cli.py                # Click commands
├── tests/
│   ├── data/
│   │   ├── anatomic_sample.json
│   │   ├── anatomic_sample_embeddings.json
│   │   └── anatomic_query_embeddings.json
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_index.py
│   ├── test_index_db.py
│   ├── test_migration.py
│   └── test_cli.py
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Dependencies

```toml
[project]
name = "anatomic-locations"
requires-python = ">=3.11"
dependencies = [
    "oidm-common>=0.1",      # Shared utilities
    "duckdb>=1.0",
    "pydantic>=2.0",
    "click>=8.0",
    "rich>=13.0",            # CLI output
    "loguru>=0.7",
]

[project.optional-dependencies]
build = [
    "openai>=1.0",           # Only needed for database builds
]

[project.scripts]
anatomic = "anatomic_locations.cli:main"
```

## Tooling, Config, and Lockfiles

- **uv-first**: Use uv for install/test/build/publish. Commit `uv.lock` to ensure reproducible CI and releases.
- **Taskfile is canonical**: Prefer Task targets (e.g., `task test`, `task check`) because they include required markers/modifiers (like `-m "not callout"`) and other defaults. Use raw `uv run ...` only when no task exists or when you intentionally need different flags.
- **pyproject as source of truth**: Keep ruff/mypy/pytest/tool.uv config in `pyproject.toml`; avoid stray config files unless necessary. Taskfile commands should remain thin wrappers over uv while preserving the baked-in defaults.
- **CI**: Run uv everywhere (`uv sync --frozen`, `uv run ruff format/check`, `uv run mypy`, `uv run pytest -m "not callout"`); publish with `uv build` then `uv publish`.

## Code Migration

### From findingmodel

| Source                         | Destination          | Notes                     |
| ------------------------------ | -------------------- | ------------------------- |
| `anatomic_location.py`         | `models/location.py` | Rename, update imports    |
| `anatomic_location.py` (enums) | `models/enums.py`    | Extract enums             |
| `anatomic_index.py`            | `index.py`           | Update imports            |
| `anatomic_migration.py`        | `migration.py`       | Use oidm-common bulk_load |
| `cli.py` (anatomic commands)   | `cli.py`             | Extract, standalone       |
| `test/test_anatomic_*.py`      | `tests/`             | Update imports            |
| `test/data/anatomic_*.json`    | `tests/data/`        | Direct copy               |

### Import Updates

```python
# Old (findingmodel)
from findingmodel.anatomic_location import AnatomicLocation
from findingmodel.anatomic_index import AnatomicLocationIndex
from findingmodel.tools.duckdb_utils import setup_duckdb_connection

# New (anatomic-locations)
from anatomic_locations import AnatomicLocation, AnatomicLocationIndex
from oidm_common.duckdb import setup_connection
```

## Public API

```python
# anatomic_locations/__init__.py
from .models import (
    AnatomicLocation,
    AnatomicRef,
    AnatomicRegion,
    BodySystem,
    Laterality,
    LocationType,
    StructureType,
)
from .index import AnatomicLocationIndex

__all__ = [
    "AnatomicLocation",
    "AnatomicLocationIndex",
    "AnatomicRef",
    "AnatomicRegion",
    "BodySystem",
    "Laterality",
    "LocationType",
    "StructureType",
]
```

## Usage Examples

### Python API

```python
from anatomic_locations import AnatomicLocationIndex, Laterality

# Database auto-downloads on first use
with AnatomicLocationIndex() as index:
    # Get by ID
    kidney = index.get("RID2772")

    # Navigate hierarchy
    ancestors = kidney.get_containment_ancestors()
    descendants = kidney.get_containment_descendants()

    # Laterality variants
    variants = kidney.get_laterality_variants()
    left_kidney = variants[Laterality.LEFT]

    # Search
    results = index.search("renal cortex", limit=5)

    # Code lookup
    locations = index.find_by_code("snomed", "64033007")
```

### CLI

```bash
# Query commands
anatomic query ancestors RID2772
anatomic query descendants RID39569
anatomic query laterality RID2772
anatomic query code snomed 64033007

# Database management
anatomic stats
anatomic build --source https://example.com/data.json
anatomic validate --source /path/to/data.json
```

## Implementation Steps

### Step 1: Repository Setup

- Create `anatomic-locations` repo
- Set up pyproject.toml
- Configure CI/CD
- Add CLAUDE.md and .github/copilot-instructions.md (adapted from findingmodel templates)
- Create .claude/ and .serena/ directories
- Seed Serena memories (project_overview, code_style_conventions, suggested_commands, ai_assistant_usage_2025, anatomic_location_search_implementation)

### Step 2: Models

- Copy anatomic_location.py
- Split into location.py and enums.py
- Update imports
- Add tests

### Step 3: Index

- Copy anatomic_index.py
- Update to use oidm-common utilities
- Update imports
- Add tests

### Step 4: Migration

- Copy anatomic_migration.py
- Replace \_bulk_load_table with oidm_common.duckdb.bulk_load_table
- Update imports
- Add tests

### Step 5: Config

- Create config.py with settings
- Implement ensure_db() using oidm_common.distribution
- Configure manifest URL and keys

### Step 6: CLI

- Extract anatomic commands from findingmodel CLI
- Create standalone Click app
- Add tests

### Step 7: Test Data

- Copy test/data/anatomic\_\*.json
- Update conftest.py fixtures
- Verify all tests pass

### Step 8: Documentation

- README with installation, quick start, API overview
- Link to findingmodel docs for integration
- CLAUDE.md for AI assistants

### Step 9: Release

- Version 0.1.0
- Publish to PyPI
- Update findingmodel docs to reference new package

## Database Distribution

The anatomic locations database will be:

1. **Built separately** - Using `anatomic build` command
2. **Published to remote storage** - Same infrastructure as findingmodel
3. **Referenced in manifest** - Shared or separate manifest
4. **Downloaded on first use** - Via oidm-common distribution

### Manifest Options

**Option A: Shared manifest**

```json
{
  "databases": {
    "finding_models": { ... },
    "anatomic_locations": { ... }
  }
}
```

**Option B: Separate manifest**

- anatomic-locations has its own manifest URL
- Independent versioning

## Integration with findingmodel

**findingmodel keeps its old implementation initially.** Later:

1. Add `anatomic-locations` as optional dependency
2. Update `find_anatomic_locations()` to use new package
3. Deprecate internal anatomic code
4. This is a separate effort, not blocking

```python
# Future findingmodel/tools/anatomic.py
try:
    from anatomic_locations import AnatomicLocationIndex
    HAS_ANATOMIC = True
except ImportError:
    HAS_ANATOMIC = False

async def find_anatomic_locations(...):
    if not HAS_ANATOMIC:
        raise ImportError("Install anatomic-locations for this feature")
    ...
```

## Testing Strategy

- Unit tests with mocked dependencies
- Integration tests with real DuckDB (temp files)
- Pre-generated embeddings (no API calls)
- Test data from findingmodel test/data/

## Repository Creation & Claude Code Bootstrap

### Prerequisites

**oidm-common must be created first** (or at least planned). anatomic-locations depends on it for:

- `oidm_common.duckdb.setup_connection`
- `oidm_common.duckdb.bulk_load_table`
- `oidm_common.distribution.ensure_db_file`

During development, use editable install: `uv add --editable ../oidm-common`

### Why Fresh Repository (Not Fork)

Same rationale as oidm-common: we're restructuring code, not extracting a subdirectory. The anatomic code is spread across multiple files and needs reorganization into a cleaner module structure.

### Step-by-Step Repository Creation

```bash
# 1. Create and initialize
mkdir anatomic-locations && cd anatomic-locations
git init
uv init --lib --name anatomic-locations

# 2. Create directory structure
mkdir -p src/anatomic_locations/models
mkdir -p tests/data
mkdir -p .claude/rules
mkdir -p .serena/memories

# 3. Copy instruction files (adapt from sibling findingmodel repo)
cp ../findingmodel/CLAUDE.md ./CLAUDE.md
cp ../findingmodel/.github/copilot-instructions.md ./.github/copilot-instructions.md

# 4. Seed Serena memories (via Serena MCP)
#   project_overview, code_style_conventions, suggested_commands, ai_assistant_usage_2025, anatomic_location_search_implementation
#   Adjust scope to anatomic-locations.

# 5. Copy planning documents into new repo
cp ../findingmodel/tasks/anatomic-locations-package-plan.md ./PLAN.md
cp ../findingmodel/tasks/done/anatomic-locations-normalized-schema.md ./docs/SCHEMA.md

# 6. Copy test data
cp ../findingmodel/test/data/anatomic_*.json ./tests/data/
```

### Claude Code Bootstrap Files

#### CLAUDE.md (Create This First)

````markdown
# CLAUDE.md

## Project Snapshot

**anatomic-locations** provides anatomic location ontology navigation:

- Hierarchy traversal (containment, part-of relationships)
- Laterality variants (left/right/bilateral)
- Semantic search with embeddings
- Code lookups (RadLex, SNOMED, FMA, UBERON)
- CLI for queries and database management

**Stack:** Python 3.11+, uv, DuckDB, Pydantic v2, Click, oidm-common

**Layout:**

- `src/anatomic_locations/` - Package source
- `tests/` - pytest tests with pre-generated embeddings
- `PLAN.md` - Implementation plan
- `docs/SCHEMA.md` - Database schema documentation

## Dependencies

Depends on `oidm-common` for shared infrastructure:

```python
from oidm_common.duckdb import setup_connection, bulk_load_table
from oidm_common.distribution import ensure_db_file
```
````

## Coding Standards

- **Formatting:** Ruff, 120 char lines
- **Typing:** Strict, use `Annotated`/`Field` for constraints
- **Naming:** snake_case functions, PascalCase classes
- **Async:** Use for I/O; remove if no awaits (RUF029)

## Key Patterns

See `.claude/rules/anatomic-patterns.md` for domain-specific patterns.

## Commands

```bash
task test      # Run tests (no external calls)
task test-full # Run all tests
task check     # Format + lint + mypy
task build     # Build package
```

## Origin

Extracted from findingmodel. See PLAN.md and docs/SCHEMA.md for details.

````

#### .claude/rules/anatomic-patterns.md

Create from findingmodel docs:

```bash
# Copy and adapt
cp ../findingmodel/docs/anatomic-locations.md \
  ./.claude/rules/anatomic-patterns.md
````

Key patterns to include:

- Materialized path hierarchy navigation
- Laterality variant resolution
- Hybrid search (FTS + semantic)
- Bulk loading for database builds

#### .serena/memories/

Copy and adapt relevant memories from findingmodel:

| Memory                                    | Action                  |
| ----------------------------------------- | ----------------------- |
| `code_style_conventions`                  | Copy and adapt          |
| `anatomic_location_search_implementation` | Copy (domain knowledge) |
| `pydantic_ai_testing_best_practices`      | Copy (testing patterns) |

```bash
# Create project_overview for new package
cat > .serena/memories/project_overview.md << 'EOF'
# anatomic-locations Project Overview

Standalone package for anatomic location ontology navigation.

## Key Components

- **models/location.py** - AnatomicLocation, AnatomicRef models
- **models/enums.py** - Laterality, AnatomicRegion, BodySystem, etc.
- **index.py** - AnatomicLocationIndex with search and navigation
- **migration.py** - Database build from JSON source
- **cli.py** - Click-based CLI commands

## Domain Knowledge

- RadLex-based IDs (RID prefixes)
- Containment hierarchy via materialized paths
- Laterality: left/right/bilateral/nonlateral/generic
- Code systems: RadLex, SNOMED, FMA, UBERON

## Origin

Extracted from findingmodel. See PLAN.md and docs/SCHEMA.md.
EOF
```

### .mcp.json Configuration

```json
{
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": ["--python", "3.11", "serena-mcp-server", "--project-path", "."]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-filesystem"]
    }
  }
}
```

### Preserving Planning Context

Two key documents transfer the planning work:

1. **PLAN.md** - This file (implementation plan, migration tables)
2. **docs/SCHEMA.md** - The normalized schema design (from anatomic-locations-normalized-schema.md)

```bash
# In anatomic-locations repo
cp ../findingmodel/tasks/anatomic-locations-package-plan.md ./PLAN.md
mkdir docs
cp ../findingmodel/tasks/done/anatomic-locations-normalized-schema.md ./docs/SCHEMA.md
```

### Initial Commit Message

```bash
git add .
git commit -m "$(cat <<'EOF'
Initial anatomic-locations package structure

Bootstrap for anatomic location ontology package containing:
- Models: AnatomicLocation, AnatomicRef, enums
- Index: Search, hierarchy navigation, laterality
- Migration: Database build from JSON source
- CLI: Query and management commands

Depends on: oidm-common
Extracted from: findingmodel (commit: XXXXX)
See PLAN.md and docs/SCHEMA.md for details.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### First Claude Code Session in New Repo

When you open Claude Code in the new repo, say:

> "Read PLAN.md and docs/SCHEMA.md to understand the implementation plan. This package was extracted from findingmodel and depends on oidm-common. Start with Step 2 (Models) - copy anatomic_location.py and split it into location.py and enums.py."

Claude Code will:

1. Read CLAUDE.md automatically
2. Read PLAN.md (migration context)
3. Read docs/SCHEMA.md (schema details)
4. Know the oidm-common dependency
5. Have the code migration tables

### Development Workflow with oidm-common

During development, link oidm-common as editable:

```bash
# In anatomic-locations repo
uv add --editable ../oidm-common

# Or if oidm-common is published
uv add oidm-common>=0.1
```

Test imports work:

```python
# Verify dependency
from oidm_common.duckdb import setup_connection, bulk_load_table
from oidm_common.distribution import ensure_db_file
```

### Cross-Repo Workflow

Have all three repos accessible:

```bash
# Terminal 1: anatomic-locations (main work)
cd ~/repos/anatomic-locations
claude

# Terminal 2: oidm-common (dependency)
cd ~/repos/oidm-common

# Terminal 3: findingmodel (source)
cd ~/repos/findingmodel
```

Or use absolute paths in Claude Code:

> "Read ../findingmodel/src/findingmodel/anatomic_location.py and extract the AnatomicLocation class into src/anatomic_locations/models/location.py, updating imports to use oidm_common"

## Open Questions

1. **Package name:** `anatomic-locations` vs `oidm-anatomic` vs `anatomic-index`?
2. **Manifest strategy:** Shared with findingmodel or independent?
3. **CLI name:** `anatomic` vs `anatomic-locations`?
4. **Build dependency:** Require OpenAI for build, or support multiple providers?
