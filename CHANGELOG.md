<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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