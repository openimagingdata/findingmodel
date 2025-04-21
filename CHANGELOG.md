<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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