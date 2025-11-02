<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased (presumed 0.5.0)]

### Added

- **Comprehensive Agent Evaluation Suites** (2025-10-18 to 2025-11-02) - Five new eval suites for AI agent quality assessment:
  - `evals/similar_models.py` - Similarity search and duplicate detection (8 cases)
  - `evals/ontology_match.py` - Multi-backend ontology concept matching (12 cases)
  - `evals/anatomic_search.py` - Two-agent anatomic location search (10 cases)
  - `evals/markdown_in.py` - Markdown to finding model parsing (8 cases)
  - `evals/finding_description.py` - Clinical description quality with LLMJudge (15 cases)
  - All use Pydantic Evals Dataset.evaluate() pattern with focused evaluators
  - Logfire observability via lazy instrumentation pattern
  - Taskfile commands: `task evals` or `task evals:agent_name`
- **LLMJudge Evaluator Support** (2025-11-02) - Built-in LLM-based quality assessment:
  - Configured for clinical description quality scoring
  - Uses cost-effective gpt-5-nano model
  - Workaround for Pydantic Evals API key bug documented
- **PerformanceEvaluator** (2025-10-29) - Reusable evaluator in `src/findingmodel/tools/evaluators.py`:
  - Configurable time limits for agent performance testing
  - Comprehensive unit tests in `test/tools/test_evaluators.py`
  - Used across all 5+ eval suites (eliminates ~190 lines duplication)
- **Enhanced Index API Methods** - Complete pagination and search capabilities without breaking abstraction:
  - `list(limit, offset, order_by, order_dir)` - Paginated browsing of all finding models
  - `search_by_slug(pattern, match_type, limit, offset)` - Pattern-based search with relevance ranking
  - `count()` and `count_search()` - Efficient counting for pagination UI
  - `get_full(oifm_id)` and `get_full_batch(oifm_ids)` - Retrieve complete FindingModelFull objects
- **Manifest-Based Database Downloads** - Runtime database version discovery:
  - Databases auto-update from remote manifest.json (no library release needed)
  - Graceful fallback to direct URL/hash for offline scenarios
  - CLI command: `db-info` to check database versions and status
- **Self-Contained Databases** - Full JSON storage in DuckDB:
  - Single .duckdb file contains metadata + embeddings + full JSON models
  - Separate `finding_model_json` table with automatic compression
  - No calls to external sites for full models
- **Local ID Generation** - Database-backed OIFM/OIFMA ID generation:
  - `generate_model_id(source)` - Random generation with collision checking
  - `generate_attribute_id(model_oifm_id, source)` - Attribute ID generation with source inference
  - No network dependency, thread-safe via DuckDB
- **Database Path Configuration** - Specify database file paths for production/Docker deployments:
  - Set `DUCKDB_INDEX_PATH` or `DUCKDB_ANATOMIC_PATH` to use pre-mounted database files
  - Default behavior unchanged (automatic manifest-based downloads)

### Changed

- **GPT-5 Model Adoption** (2025-11-02) - Updated default OpenAI models:
  - `openai_default_model`: gpt-4o-mini → gpt-5-mini (better capability)
  - `openai_default_model_small`: gpt-4.1-nano → gpt-5-nano (cost-effective)
  - `openai_default_model_full`: gpt-5 (unchanged)
- **Test Performance Optimization** (2025-11-02) - Integration tests 54% faster:
  - Tests explicitly use gpt-4o-mini for speed (278s → 127s total test suite)
  - Production code uses GPT-5 models for capability
  - Clear separation: fast models for CI/CD, capable models for production
- **Evaluator Architecture** (2025-10-29) - Clean separation of concerns:
  - Inline evaluators in eval scripts (35+ agent-specific evaluators)
  - Reusable evaluators in src/ only when genuinely shared (PerformanceEvaluator)
  - Lazy instrumentation pattern prevents Logfire noise in unit tests
- **Anatomic Location Search Enhancement** (2025-11-02) - Added model parameter:
  - `find_anatomic_locations()` accepts optional `model` parameter
  - Allows per-call model override for performance tuning
- **Index schema**: Added separate `finding_model_json` table for blob storage
- **Index schema**: Added index on `slug_name` for efficient LIKE queries
- **Code organization**: Shared helper methods eliminate duplication in list/search/count operations
- **Database configuration defaults** changed to enable custom paths (previously auto-downloaded only)

### Deprecated

- Direct access to `Index._ensure_connection()` - use new API methods instead (see migration guide)

### Removed

- **`IdManager` class** - Use `Index.add_ids_to_model()` and `Index.finalize_placeholder_attribute_ids()` instead, or the convenience function `findingmodel.tools.add_ids_to_model()`
- **`id_manager` singleton** from `findingmodel.tools` - Use convenience functions or create Index instances
- **`IdManager.load_used_ids_from_github()`** - Index queries database directly
- **`src/findingmodel/tools/add_ids.py`** module - All ID generation now handled by Index class
- **MongoDB Index backend** - DuckDB is now the only Index implementation
- **`MongoDBIndex` class** - Use `Index` (aliased to `DuckDBIndex`) instead

### Removed

- **evals/base.py** (2025-10-29) - Unused evaluators (~350 lines):
  - All 5 evaluators duplicated Pydantic Evals built-ins or were never used
  - Replaced by centralized PerformanceEvaluator and inline evaluators
- **test/test_base_evaluators.py** (2025-10-29) - Tests for removed code (~600 lines)

### Fixed

- **LLMJudge API Key Configuration** (2025-11-02) - Workaround for Pydantic Evals bug:
  - LLMJudge now reads OpenAI API key from environment variable
  - Documented workaround until upstream fix available

## [0.4.0] - 2025-10-20

### Added

- **DuckDB Index Backend**: High-performance local search with HNSW vector indexing, now the default
  - Automatic download of pre-built database files on first use (via Pooch with SHA256 verification)
  - Platform-native file storage using platformdirs (OS-appropriate cache/data directories)
  - Base contributors (4 people, 7 organizations) automatically loaded in new databases
  - Optional remote database URLs configurable via environment variables
  - Utilities for building/updating databases from directories of FM definitions
- **Anatomic Locations**: Comprehensive support for anatomic location finding and management
  - New `anatomic_locations` field in `FindingModelFull` for specifying anatomic index codes
  - Two-agent AI search tool across anatomic_locations, RadLex, and SNOMED CT ontologies
  - CLI commands for building, validating, and viewing statistics on anatomic location databases
  - Demo: `scripts/anatomic_location_search.py`
- **Ontology Concept Search Tool**: High-performance medical concept search
  - Multi-backend support: DuckDB vector search and BioOntology.org REST API (requires `BIOONTOLOGY_API_KEY`)
  - Protocol-based architecture for pluggable search backends
  - Demo: `scripts/ontology_concept_match.py`
- **Finding Model Editor Tool**: AI-assisted interactive editor with markdown and natural language workflows
- **Index API enhancements**:
  - `get_people()` method to retrieve all people from Index (sorted by name)
  - `get_organizations()` method to retrieve all organizations from Index (sorted by name)
  - Available in both DuckDBIndex and MongoDBIndex implementations
- **Documentation**: New `docs/database-management.md` guide for maintainers covering database operations, CLI commands, and write-mode Python API

### Changed

- **DuckDB is now the default index backend** (MongoDB deprecated but still available via `[mongodb]` extra)
- **Database file locations**: Files now stored in platform-native directories
  - Auto-download mitigates migration: databases automatically download to new locations on first use
- **Test suite reorganization**: Reduced from 17 to 12 test files, grouped by feature area
- **Documentation restructure**: README.md now focuses on user perspective (read-only Index operations), with maintainer/write operations moved to docs/database-management.md

### Deprecated

- **MongoDB backend**: Still functional via `pip install findingmodel[mongodb]` but no longer the default
  - DuckDB provides better performance and simpler deployment
  - MongoDB configuration commented out in config but available if needed

### Removed

- **LanceDB dependency**: Replaced by DuckDB for better performance and consistency

### Migration Notes

Upgrading from v0.3.x:

- **For most users**: No action required. Databases will auto-download to platform-native locations on first use.
- **If using MongoDB**: Your MongoDB index will continue to work, but consider migrating to DuckDB for better performance. Use `index build` CLI to create DuckDB index from your finding model definitions.
- **If using custom database paths**: Note that default paths have changed to platform-native directories. You can continue using custom paths via CLI `--index` / `--db-path` options.

## [0.3.2] - 2025-08-20

### Added

- Added comprehensive test coverage for previously untested functionality:
  - 11 tests for `Index.search()` functionality including limit, case sensitivity, and multiple terms
  - 7 integration tests for AI tools with `@pytest.mark.callout` decorator
  - 3 tests for `find_similar_models()` function including edge cases
  - 15 error handling tests for network failures, MongoDB issues, and invalid data
- Updated test documentation to reflect actual MongoDB implementation of Index
- Fixed all linting issues for clean CI/CD pipeline
- **Performance Optimization**: Optimized `similar_finding_models` tool for significantly faster execution:
  - Reduced runtime from 15-20 seconds to 4-9 seconds (up to 75% improvement)
  - Added smart model selection with fallback mechanisms
  - Implemented batch MongoDB queries to reduce round trips
  - Added lightweight term generation with intelligent fallback to full models
- **Release Automation**: Created comprehensive release automation system:
  - Added `scripts/release.py` with full release pipeline automation
  - Integrated loguru logging with version-specific log files
  - Added Taskfile commands: `task release`, `task release:check`, `task release:dry`
  - Supports dry-run mode, pre-flight checks, and automatic PyPI publishing

### Changed

- Corrected `Index` documentation in README.md and CLAUDE.md from JSONL to MongoDB implementation
- Fixed `find_similar_models()` documentation to accurately describe its purpose and signature
- Updated type annotations in test functions to satisfy ruff linting requirements
- Replaced generic Exception catches with specific exception types (JSONDecodeError, ValidationError)

### Fixed

- Fixed test validation errors by using proper 3-character source codes (TST, TES, TEX)
- Added missing pytest import in test_tools.py
- Corrected find_similar_models test signature to match actual implementation

### Removed

## [0.3.1] — 2025–07–06

### Fixed

- Dealt with a bug in the setup of OpenAI models by `find_similar_models` in `findingmodel.tools`

## [0.3.0] — 2025-07-03

### Added

- Added tests for new function names to ensure complete API coverage
- Added comprehensive examples to README.md for all tool functions

### Changed

- Renamed tool functions to remove redundant "finding" prefix for cleaner API
- Reorganized tools module into specialized files for better maintainability
- Updated README.md with new function names and complete usage examples

### Deprecated

Function names have been updated to be more concise by removing the redundant "finding" prefix. The old function names are still available but will show deprecation warnings.

| Deprecated Function | New Function | Notes |
|-------------------|-------------|-------|
| `describe_finding_name()` | `create_info_from_name()` | Creates FindingInfo from name |
| `get_detail_on_finding()` | `add_details_to_info()` | Adds details to existing FindingInfo |
| `create_finding_info_from_name()` | `create_info_from_name()` | Intermediate name, also deprecated |
| `add_details_to_finding_info()` | `add_details_to_info()` | Intermediate name, also deprecated |
| `create_finding_model_from_markdown()` | `create_model_from_markdown()` | Creates model from markdown |
| `create_finding_model_stub_from_finding_info()` | `create_model_stub_from_info()` | Creates basic model stub |
| `add_ids_to_finding_model()` | `add_ids_to_model()` | Adds OIFM IDs to model |
| `add_standard_codes_to_finding_model()` | `add_standard_codes_to_model()` | Adds standard medical codes |

### Removed

- Removed direct exposure of `load_used_ids_from_github()` from tools module (still available via `id_manager`)

## [0.2.0] – 2025-06-23

### Added

- Added facility to check on online `ids.json` structure (e.g., from a GitHub repo) to help avoid
  duplicate IDs as much as possible.
- Started doing logging with `loguru`

### Changed

- Refactored `tools` to be its own sub-module, making it easier to pull in tools
- Redid `Index` to now use a MongoDB backend, which can be either local or remote
- Fixed code field length minimum to be 2 characters

### Removed

- 🔴 Removed `SearchRepository`—this is going to be delivered directly via `Index`.

## [0.1.4] – 2025-05-08

### Added

- Added new `Person` and `Organization` classes to represent contributors to a finding model, and added
  a (non-required—yet) list of them to `FindingModelFull` as the `contributors` field.
- Added an `Index` object that can maintain an in-memory database of the metadata of finding models
  saved as `*.fm.json` in a given directory.
- Added a `SearchRepo` object that has a LanceDB-based hybrid search facility allowing for more
  sophisticated analysis of finding models for potential similarity.
- Lots of tests and test data to prevent regressions, and updating docs and notebooks.

### Changed

- Changed the way the `tools` module is imported; probably want to use `import findingmodel.tools as fmtools` or something similar.

### Removed

- Removed `SearchRepository` and associated testing/test files.

## [0.1.3] – 2025-04-26

### Added

- Fleshed out the CLI
- Added index codes, added tool to add standard index codes

### Changed

- Updated dependencies
- Dev-ex tweaks

<!-- ### Removed -->

## [0.1.2] - 2025-04-21

### Added

- Better tests for `FindingModelFull.as_markdown()`
- Script for turning FM JSON into Markdown
- Docstrings, README documation, example notebooks.

### Changed

- Improved Markdown generation
- Refactored `FindingInfo` to be a single class rather than two separate classes

## [0.1.1] - 2025-04-17

### Added

- New tests form `FindingModel[Base|Full].as_markdown()`

### Changed

- Marked improvements in `FindingModel[Base|Full].as_markdown()`

## [0.1.0] - 2025-04-16

- Initial porting of code from prior repository, first published version.
- Includes:
  - Definition of `FindingModelBase` and `FindingModelFull` pydantic models
  - Tools to get basic information and detailed information about findings from LLMs
  - Tools to generate new finding models, either as stubs or from Markdown

<!-- ## [1.0.0] - 2021-07-19
### Added
- devicely.FarosReader can both read from and write to EDF files and directories
- devicely.FarosReader has as attributes the individual dataframes (ACC, ECG, ...) and not only the joined dataframe

### Changed
- in devicely.SpacelabsReader, use xml.etree from the standard library instead of third-party "xmltodict"
- switch from setuptools to Poetry

### Removed
- removed setup.py because static project files such as pyproject.toml are preferred -->