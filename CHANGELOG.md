<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **BioOntology API Integration**: Added support for BioOntology.org REST API search
  - Access to 800+ medical ontologies including SNOMED-CT, ICD-10, LOINC, and more
  - Semantic type filtering for targeted searches
  - Pagination support for comprehensive results
  - Async/await implementation with connection pooling via httpx
- **Protocol-Based Ontology Search Architecture**: Flexible backend support system
  - `OntologySearchProtocol` defines standard interface for search providers
  - Auto-detection of available backends based on configuration
  - Parallel execution of multiple backends using `asyncio.gather`
  - Clean abstraction allows easy addition of new search providers

### Changed

- **Renamed `search_ontology_concepts()` to `match_ontology_concepts()`**: Better reflects the matching/categorization functionality
- **Multi-Backend Support**: Function now automatically uses all configured backends
  - Searches LanceDB and/or BioOntology in parallel based on configuration
  - Merges and deduplicates results from multiple sources
  - Maintains high performance with parallel execution
- **Improved Test Infrastructure**: Enhanced test patterns and organization
  - Added `test_ontology_search.py` for Protocol compliance testing
  - Proper SecretStr handling in tests
  - Fixed all linting errors with proper type annotations
  - Configured per-file ignores for test-specific patterns (ANN401 for mock functions)

### Fixed

- **SecretStr Handling**: Fixed improper use of `str(SecretStr)` which doesn't extract the actual secret value
  - BioOntologySearchClient now handles SecretStr extraction internally
  - Tests properly mock SecretStr values
- **Linting Compliance**: Fixed 33 linting errors to achieve clean `task check`
  - Combined nested `with` statements using parenthesized form (SIM117)
  - Added proper type annotations throughout test files
  - Removed `async` from functions that don't use `await` (RUF029)
  - Fixed indentation issues in test files

## [0.3.3] - 2025-09-04

### Added

- **Ontology Concept Search Tool**: New high-performance tool for searching medical ontologies
  - Refactored from 70+ second searches to ~10 second searches using optimized architecture
  - Uses programmatic query generation instead of LLM-based search for better performance
  - Implements post-processing to guarantee exact match detection
  - Proper Pydantic AI patterns with structured output validation
  - Demo script in `notebooks/demo_ontology_concept_search.py`
- **Added `anatomic_locations` to `FindingModelFull`**: Can now specify a list of index codes that 
specifically refer to anatomy to indicate where the finding is likely to occur.
- **Anatomic Location Search Tool**: New two-agent Pydantic AI tool for finding anatomic locations
  - Reusable `OntologySearchClient` for LanceDB medical terminology search
  - `OntologySearchResult` model for standardized ontology search responses
  - Two-agent architecture: search agent (generates queries) + matching agent (selects best locations)
  - Hybrid search across multiple ontology tables (anatomic_locations, radlex, snomedct)
  - Production-ready with proper error handling, logging, and connection lifecycle management
  - Demo script in `notebooks/demo_anatomic_location_search.py`
- **Testing Improvements**: Enhanced testing infrastructure following Pydantic AI best practices
  - Added API call prevention guards (`models.ALLOW_MODEL_REQUESTS = False`) to prevent accidental API calls during testing
  - Implemented proper Pydantic AI testing patterns using `TestModel` and `FunctionModel`
  - Consolidated related component tests (merged ontology_search tests into anatomic_location_search tests)
  - Fixed pytest collection issues by moving demo scripts to `notebooks/` directory
  - Added `testpaths = ["test"]` to pyproject.toml to restrict pytest scope

### Changed

- **Refactored `get_openai_model()`**: Moved from `similar_finding_models.py` to `common.py` for reusability
- **Test Philosophy**: Shifted from testing library functionality to testing actual code behavior and workflow logic
- **Project Conventions**: Established convention for demo scripts in `notebooks/` with `demo_*.py` naming
- **Test Framework Consistency**: Converted all tests from unittest.TestCase to pure pytest style for consistency
- **Ontology Concept Search Refactoring**: Major performance and architecture improvements
  - Replaced manual retry loops with Pydantic AI's built-in validation patterns
  - Simplified from 3 agents to 1 agent plus programmatic processing for 85% performance improvement
  - Changed from transformation in validators to proper post-processing functions
  - Improved text normalization to only remove trailing parenthetical content

### Fixed

- Fixed integration test API blocking issue where `models.ALLOW_MODEL_REQUESTS = False` was preventing integration tests from running
- Resolved all linting errors in test files (import order, nested with statements, unused variables)
- Fixed normalize_concept function to preserve middle parenthetical content (e.g., "Calcium (2+) level")
- Corrected Pydantic AI anti-patterns where output validators were transforming data instead of validating
- Fixed test organization by consolidating test_query_generator.py into test_ontology_concept_search.py
- Cleaned up 110+ linting errors by removing temporary debug files and fixing type annotations

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