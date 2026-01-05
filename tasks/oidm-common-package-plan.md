# Plan: oidm-common Package

**Status:** Planning
**Related:** [oidm-package-restructuring.md](oidm-package-restructuring.md), [anatomic-locations-package-plan.md](anatomic-locations-package-plan.md)

## Purpose

Shared utilities for OIDM packages (findingmodel, anatomic-locations, future packages). Provides common patterns for DuckDB operations, database distribution, embeddings, and project scaffolding.

## Package Structure

**Repository/package naming (PEP 8 aligned)**
- Repo: `oidm-common`
- Python package: `oidm_common`
- Source root: `src/oidm_common/`
```
oidm-common/
├── src/
│   └── oidm_common/
│       ├── __init__.py
│       ├── duckdb/
│       │   ├── __init__.py
│       │   ├── connection.py      # setup_duckdb_connection()
│       │   ├── bulk_load.py       # bulk_load_table(), read_json patterns
│       │   ├── search.py          # normalize_scores(), weighted_fusion(), rrf_fusion()
│       │   └── indexes.py         # create_fts_index(), create_hnsw_index(), drop_search_indexes()
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── cache.py           # EmbeddingCache (DuckDB-based)
│       │   ├── provider.py        # EmbeddingProvider protocol
│       │   ├── openai.py          # OpenAI embedding provider
│       │   └── utils.py           # to_float32(), get_embedding(), batch_embeddings()
│       ├── distribution/
│       │   ├── __init__.py
│       │   ├── manifest.py        # fetch_manifest(), manifest schema
│       │   ├── download.py        # ensure_db_file(), hash verification
│       │   └── paths.py           # platform-native data directories
│       └── models/
│           ├── __init__.py
│           ├── index_code.py      # IndexCode model
│           └── web_reference.py   # WebReference model
├── tests/
│   ├── data/
│   │   └── sample_embeddings.json # Pre-generated for tests
│   ├── conftest.py
│   ├── test_duckdb_connection.py
│   ├── test_bulk_load.py
│   ├── test_search.py
│   ├── test_indexes.py
│   ├── test_embeddings.py
│   ├── test_distribution.py
│   └── test_models.py
├── .claude/
│   └── rules/
│       └── duckdb-patterns.md     # DuckDB coding standards
├── pyproject.toml
├── Taskfile.yml
├── README.md
├── CLAUDE.md
├── CHANGELOG.md
├── .gitignore
└── .env.sample
```

## Dependencies

```toml
[project]
name = "oidm-common"
requires-python = ">=3.11"
dependencies = [
    "duckdb>=1.0",
    "pydantic>=2.0",
    "pooch>=1.8",
    "platformdirs>=4.0",
    "httpx>=0.27",
    "loguru>=0.7",
]

[project.optional-dependencies]
openai = [
    "openai>=1.0",           # OpenAI embedding provider
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.4",
    "mypy>=1.8",
]
```

## Tooling, Config, and Lockfiles

- **uv-first**: All install/test/build/publish flows use uv. Commit `uv.lock` for reproducible CI and releases.
- **Taskfile is canonical**: Prefer Task targets when present (e.g., `task test`, `task check`) because they pass required modifiers/markers (like `-m "not callout"`) and consistent options. Use raw `uv run ...` only when a Task target does not exist or when intentionally deviating.
- **pyproject as source of truth**: Carry over ruff/mypy/pytest/tool.uv config into `pyproject.toml`; avoid duplicate config files unless unavoidable. Keep Taskfile commands thin wrappers over uv but with the necessary defaults baked in.
- **CI**: Use uv everywhere (`uv sync --frozen`, `uv run ruff format/check`, `uv run mypy`, `uv run pytest -m "not callout"`). Publish via `uv build` then `uv publish`.

## Code Migration

### From findingmodel

| Source                                              | Destination                | Notes                          |
| --------------------------------------------------- | -------------------------- | ------------------------------ |
| **DuckDB Utilities**                                |                            |                                |
| `tools/duckdb_utils.py:setup_duckdb_connection`     | `duckdb/connection.py`     | Direct move                    |
| `tools/duckdb_utils.py:normalize_scores`            | `duckdb/search.py`         | Direct move                    |
| `tools/duckdb_utils.py:weighted_fusion`             | `duckdb/search.py`         | Direct move                    |
| `tools/duckdb_utils.py:rrf_fusion`                  | `duckdb/search.py`         | Direct move                    |
| `tools/duckdb_utils.py:l2_to_cosine_similarity`     | `duckdb/search.py`         | Direct move                    |
| `tools/duckdb_utils.py:create_fts_index`            | `duckdb/indexes.py`        | Direct move                    |
| `tools/duckdb_utils.py:create_hnsw_index`           | `duckdb/indexes.py`        | Direct move                    |
| `tools/duckdb_utils.py:drop_search_indexes`         | `duckdb/indexes.py`        | Direct move                    |
| `anatomic_migration.py:_bulk_load_table`            | `duckdb/bulk_load.py`      | Generalize                     |
| **Embeddings**                                      |                            |                                |
| `embedding_cache.py:EmbeddingCache`                 | `embeddings/cache.py`      | Direct move                    |
| `tools/duckdb_utils.py:get_embedding_for_duckdb`    | `embeddings/utils.py`      | Rename to `get_embedding()`    |
| `tools/duckdb_utils.py:batch_embeddings_for_duckdb` | `embeddings/utils.py`      | Rename to `batch_embeddings()` |
| `tools/duckdb_utils.py:_to_float32`                 | `embeddings/utils.py`      | Rename to `to_float32()`       |
| `tools/common.py:get_embedding`                     | `embeddings/openai.py`     | Provider implementation        |
| `tools/common.py:get_embeddings_batch`              | `embeddings/openai.py`     | Provider implementation        |
| `tools/common.py:_lookup_cached_embeddings`         | `embeddings/openai.py`     | Provider implementation        |
| `tools/common.py:_fetch_and_store_embeddings`       | `embeddings/openai.py`     | Provider implementation        |
| **Distribution**                                    |                            |                                |
| `config.py:ensure_db_file`                          | `distribution/download.py` | Generalize                     |
| `config.py:fetch_manifest`                          | `distribution/manifest.py` | Direct move                    |
| `config.py:_resolve_target_path`                    | `distribution/paths.py`    | Generalize                     |
| `config.py:_verify_file_hash`                       | `distribution/download.py` | Direct move                    |
| `config.py:_download_file`                          | `distribution/download.py` | Direct move                    |
| `config.py:_download_from_manifest`                 | `distribution/download.py` | Direct move                    |
| **Models**                                          |                            |                                |
| `index_code.py`                                     | `models/index_code.py`     | Direct move                    |
| `web_reference.py`                                  | `models/web_reference.py`  | Direct move                    |

### Not Migrated (findingmodel-specific)

| Component                                             | Reason                                        |
| ----------------------------------------------------- | --------------------------------------------- |
| `tools/common.py:get_model`                           | Pydantic-AI specific, ties to agent framework |
| `tools/common.py:get_openai_model`                    | Pydantic-AI specific                          |
| `tools/common.py:get_async_tavily_client`             | Tavily-specific, used by findingmodel search  |
| `tools/common.py:get_markdown_text_from_path_or_text` | Findingmodel-specific utility                 |
| `config.py:FindingModelConfig`                        | Package-specific settings                     |
| `config.py:ModelProvider/ModelTier`                   | Pydantic-AI model selection                   |
| `common.py:normalize_name`                            | OIFM-specific naming                          |
| `common.py:model_file_name`                           | OIFM-specific file naming                     |

## API Design

### DuckDB Connection

```python
from oidm_common.duckdb import setup_connection

conn = setup_connection(
    db_path,
    read_only=True,
    extensions=["fts", "vss"],  # default
)
```

### Bulk Loading

```python
from oidm_common.duckdb import bulk_load_table

count = bulk_load_table(
    conn,
    table_name="my_table",
    data=[{"id": "1", "vector": [...], "children": [...]}],
    column_types={"id": "VARCHAR", "vector": "FLOAT[512]", "children": "STRUCT(...)[]"},
)
```

### Search Utilities

```python
from oidm_common.duckdb import normalize_scores, weighted_fusion, rrf_fusion

# Normalize BM25 scores to 0-1
normalized = normalize_scores([12.5, 8.3, 5.1])  # -> [1.0, 0.43, 0.0]

# Combine FTS and semantic results
combined = weighted_fusion(fts_results, vector_results, weight_a=0.3, weight_b=0.7)
```

### Index Management

```python
from oidm_common.duckdb import create_fts_index, create_hnsw_index, drop_search_indexes

# Create indexes
create_fts_index(conn, "locations", ["name", "description", "synonyms"])
create_hnsw_index(conn, "locations", "vector")

# Drop before schema changes
drop_search_indexes(conn, "locations")
```

### Embeddings

```python
from oidm_common.embeddings import EmbeddingCache, get_embedding, batch_embeddings

# With cache
async with EmbeddingCache() as cache:
    embedding = await get_embedding("kidney tumor", cache=cache, client=openai_client)

# Batch (single API call, automatic caching)
embeddings = await batch_embeddings(
    ["term1", "term2", "term3"],
    cache=cache,
    client=openai_client,
)
```

### Database Distribution

```python
from oidm_common.distribution import ensure_db_file, configure_app

# Configure for your app
configure_app("anatomic-locations")  # Sets data directory

# Get database (downloads if needed)
db_path = ensure_db_file(
    file_path=None,  # Managed mode
    manifest_url="https://example.com/manifest.json",
    manifest_key="anatomic_locations",
)
```

## Project Scaffolding

### Taskfile.yml (Template)

```yaml
version: "3"

tasks:
  default:
    cmds: [task -l]
    silent: true

  test:
    desc: "Run tests without external calls"
    cmds:
      - uv run pytest -rs -m "not callout" {{.CLI_ARGS}}
    silent: true

  test-full:
    desc: "Run all tests including external calls"
    cmds:
      - uv run pytest -rs {{.CLI_ARGS}}
    silent: true

  check:
    desc: "Format, lint, and type-check"
    cmds:
      - uv run ruff format
      - uv run ruff check --fix
      - uv run mypy src
    silent: true

  build:
    desc: "Build package"
    cmds:
      - uv build {{.CLI_ARGS}}
    silent: true

  publish:
    desc: "Publish to PyPI"
    cmds:
      - uv publish {{.CLI_ARGS}}
    silent: true
```

### .gitignore (Template)

Standard Python .gitignore plus:

- `.env` (secrets)
- `*.duckdb` (generated databases)
- Local cache files

### .env.sample (Template)

```bash
# OpenAI (optional - only needed for embedding generation)
OPENAI_API_KEY=

# Embedding settings
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_EMBEDDING_DIMENSIONS=512
```

### CLAUDE.md (Template)

Key sections for AI assistants:

1. Project snapshot (purpose, stack, layout)
2. Coding standards (style, typing, naming)
3. DuckDB patterns (bulk load, indexes, search)
4. Testing conventions (fixtures, mocking)
5. Suggested commands

### .claude/rules/ (Extracted Patterns)

**duckdb-patterns.md**: Extracted from docs/duckdb-development.md

- Bulk loading with read_json()
- Column type quoting for complex types
- Float32 conversion for embeddings
- HNSW persistence settings
- FTS index creation

## Implementation Steps

### Step 1: Repository Setup

- Create `oidm-common` repo
- Set up pyproject.toml with dependencies
- Configure CI/CD (GitHub Actions)
- Add Taskfile.yml, .gitignore, .env.sample
- Add CLAUDE.md and .github/copilot-instructions.md (adapted from findingmodel templates)
- Create .claude/ and .serena/ directories
- Seed Serena memories (project_overview, code_style_conventions, suggested_commands, ai_assistant_usage_2025, duckdb_development_patterns)

### Step 2: DuckDB Submodule

- Migrate connection.py from duckdb_utils.py
- Migrate search.py (normalize, fusion) from duckdb_utils.py
- Migrate indexes.py (create/drop) from duckdb_utils.py
- Generalize bulk_load.py from anatomic_migration.py
- Add comprehensive tests

### Step 3: Embeddings Submodule

- Move EmbeddingCache from embedding_cache.py
- Create EmbeddingProvider protocol
- Implement OpenAI provider
- Add utils (to_float32, get_embedding, batch_embeddings)
- Add tests with pre-generated embeddings

### Step 4: Distribution Submodule

- Generalize ensure_db_file pattern
- Make app name configurable
- Support custom manifest URLs
- Add hash verification
- Add tests with mock downloads

### Step 5: Models

- Move IndexCode
- Move WebReference
- Add tests

### Step 6: Documentation

- README with quick start, installation, usage examples
- .claude/rules/duckdb-patterns.md
- CHANGELOG.md (keep it simple)
- API documentation in docstrings

### Step 7: Release

- Version 0.1.0
- Publish to PyPI
- Announce availability

## Integration Strategy

**findingmodel keeps its own copy initially.** Later:

1. Add `oidm-common` as dependency
2. Update imports gradually
3. Remove duplicated code
4. This is a separate effort, not blocking

```python
# Future findingmodel imports
from oidm_common.duckdb import setup_connection, bulk_load_table
from oidm_common.embeddings import EmbeddingCache, get_embedding
from oidm_common.distribution import ensure_db_file
```

## Testing Strategy

- Unit tests for each module
- Integration tests with real DuckDB (temp files)
- No external API calls in tests (mock OpenAI)
- Pre-generated embeddings in test/data/
- `@pytest.mark.callout` for any real API tests

## Decision Log

### Embeddings as Submodule

**Decision:** Separate `embeddings/` submodule with protocol-based providers.
**Rationale:**

- Clean separation of concerns (cache, providers, utilities)
- Extensible to other providers (Anthropic, local models)
- Cache is reusable across any provider
- OpenAI as optional dependency

### Database Setup Patterns

**Decision:** Include all database distribution code (manifest, download, paths, hash verification).
**Rationale:**

- Identical pattern used by findingmodel and anatomic-locations
- Platform-native paths via platformdirs
- Manifest-based updates without library releases
- Hash verification for integrity

### Project Scaffolding

**Decision:** Include Taskfile.yml, .gitignore, .env.sample templates; CLAUDE.md as documentation.
**Rationale:**

- Consistent developer experience across OIDM packages
- AI assistants can follow consistent patterns
- Easy bootstrapping of new packages

### Utility Code Boundaries

**Decision:** Only infrastructure code moves to oidm-common. Domain logic stays in packages.
**Rationale:**

- `get_model()`, Tavily, Pydantic-AI specifics stay in findingmodel
- `normalize_name()`, OIFM file naming stay in findingmodel
- DuckDB, embeddings, distribution are true infrastructure

## Repository Creation & Claude Code Bootstrap

### Why Fresh Repository (Not Fork)

We're not extracting a clean subdirectory—we're gathering scattered code from multiple files and restructuring it. Git history preservation doesn't apply. A fresh repo gives:

- Clean history that makes sense for the new package
- No orphaned commits from unrelated findingmodel changes
- Initial commit references source findingmodel commit for traceability

### Step-by-Step Repository Creation

```bash
# 1. Create and initialize
mkdir oidm-common && cd oidm-common
git init
uv init --lib --name oidm-common

# 2. Create directory structure
mkdir -p src/oidm_common/{duckdb,embeddings,distribution,models}
mkdir -p tests/data
mkdir -p .claude/rules
mkdir -p .serena/memories

# 3. Copy instruction files (adapt from sibling findingmodel repo)
cp ../findingmodel/CLAUDE.md ./CLAUDE.md
cp ../findingmodel/.github/copilot-instructions.md ./.github/copilot-instructions.md

# 4. Seed Serena memories (via Serena MCP)
#   project_overview, code_style_conventions, suggested_commands, ai_assistant_usage_2025, duckdb_development_patterns
#   Adjust scope to oidm-common.

# 5. Copy planning document (this file) into new repo
cp ../findingmodel/tasks/oidm-common-package-plan.md ./PLAN.md
```

### Claude Code Bootstrap Files

#### CLAUDE.md (Create This First)

````markdown
# CLAUDE.md

## Project Snapshot

**oidm-common** provides shared infrastructure for OIDM packages:

- DuckDB utilities (connection, bulk load, search, indexes)
- Embedding management (cache, providers, float32 conversion)
- Database distribution (manifest, download, hash verification)
- Shared models (IndexCode, WebReference)

**Stack:** Python 3.11+, uv, DuckDB, Pydantic v2, OpenAI (optional)

**Layout:**

- `src/oidm_common/` - Package source
- `tests/` - pytest tests
- `PLAN.md` - Implementation plan (from findingmodel extraction)

## Coding Standards

- **Formatting:** Ruff, 120 char lines
- **Typing:** Strict, use `Annotated`/`Field` for constraints
- **Naming:** snake_case functions, PascalCase classes
- **Async:** Use for I/O; remove if no awaits (RUF029)

## Key Patterns

See `.claude/rules/duckdb-patterns.md` for DuckDB-specific patterns.

## Commands

```bash
task test      # Run tests (no external calls)
task test-full # Run all tests
task check     # Format + lint + mypy
task build     # Build package
```
````

## Origin

Extracted from findingmodel. See PLAN.md for migration details.

````

#### .claude/rules/duckdb-patterns.md

Copy from findingmodel and adapt:

```bash
# From findingmodel repo
cp docs/duckdb-development.md /path/to/oidm-common/.claude/rules/duckdb-patterns.md
````

Edit to remove findingmodel-specific references; keep:

- Bulk loading with read_json()
- Column type quoting
- Float32 conversion
- HNSW persistence settings
- FTS index creation

#### .serena/memories/ (Relevant Memories)

Copy applicable Serena memories from findingmodel:

| Memory                               | Action                        |
| ------------------------------------ | ----------------------------- |
| `code_style_conventions`             | Copy and adapt                |
| `pydantic_ai_testing_best_practices` | Copy (testing patterns apply) |
| `duckdb_development_patterns`        | Adapt to reference local docs |

```bash
# Create project_overview for new package
cat > .serena/memories/project_overview.md << 'EOF'
# oidm-common Project Overview

Shared infrastructure for OIDM packages (findingmodel, anatomic-locations).

## Modules

- **duckdb/** - Connection, bulk load, search utilities, index management
- **embeddings/** - Cache, provider protocol, OpenAI implementation
- **distribution/** - Manifest fetch, database download, path resolution
- **models/** - IndexCode, WebReference

## Origin

Extracted from findingmodel. Implementation plan in PLAN.md.
EOF
```

### .mcp.json (MCP Server Configuration)

Minimal initial configuration—no Serena until memories are populated:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-filesystem"]
    }
  }
}
```

Add Serena later once `.serena/` is populated:

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

The key insight: **this planning document becomes PLAN.md in the new repo**.

```bash
# In oidm-common repo
cp ../findingmodel/tasks/oidm-common-package-plan.md ./PLAN.md

# Reference in CLAUDE.md
echo "See PLAN.md for implementation plan and migration details." >> CLAUDE.md
```

When Claude Code opens the new repo:

1. It reads CLAUDE.md (project context)
2. CLAUDE.md points to PLAN.md
3. PLAN.md contains all migration tables, API designs, decisions

### Initial Commit Message

```bash
git add .
git commit -m "$(cat <<'EOF'
Initial oidm-common package structure

Bootstrap for shared OIDM infrastructure package containing:
- DuckDB utilities (connection, bulk load, search, indexes)
- Embeddings (cache, provider protocol, OpenAI)
- Distribution (manifest, download, paths)
- Models (IndexCode, WebReference)

Extracted from findingmodel (commit: XXXXX)
See PLAN.md for implementation details.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### First Claude Code Session in New Repo

When you open Claude Code in the new repo, say:

> "Read PLAN.md to understand the implementation plan. This package was extracted from findingmodel. Start with Step 2 (DuckDB Submodule) - copy the relevant code from findingmodel and adapt it."

Claude Code will:

1. Read CLAUDE.md automatically
2. Read PLAN.md (full migration context)
3. Have access to the code migration tables
4. Know exactly what to extract and where to put it

### Cross-Repo Workflow

When implementing, have both repos accessible:

```bash
# Terminal 1: New package
cd ~/repos/oidm-common
claude

# Terminal 2: Source repo (for reference)
cd ~/repos/findingmodel
# Keep open for copying code
```

Or use Claude Code's ability to read files from other paths:

> "Read ../findingmodel/src/findingmodel/tools/duckdb_utils.py and extract the setup_duckdb_connection function into src/oidm_common/duckdb/connection.py"

## Open Questions

1. **Package name:** `oidm-common` vs `oidm-core` vs `oidm-utils`?

   - Leaning toward `oidm-common` (clearer intent)

2. **Embedding provider interface:** Protocol-based or abstract class?

   - Leaning toward Protocol (Pythonic, no inheritance required)

3. **Manifest hosting:** Single shared manifest or per-package?

   - Leaning toward shared (easier updates, single source of truth)

4. **Multi-provider embeddings:** OpenAI-only or multi-provider from start?
   - Leaning toward OpenAI-only initially, protocol allows future extension
