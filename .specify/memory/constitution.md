<!--
Sync Impact Report:
Version change: [Initial version] → 1.0.0
Modified principles: All principles defined from template
Added sections: Medical Domain Integrity, Protocol-Based Architecture, ID Immutability, Async-First Development, Test-Driven QA
Removed sections: None (replacing template)
Templates requiring updates: ✅ All .specify/templates/ files align with constitution principles
Follow-up TODOs: None - all placeholders resolved
-->

# FindingModel Constitution

## Core Principles

### I. Medical Domain Integrity
**Non-negotiable**: All finding models must be clinically accurate and follow established medical standards.

Findings represent observable conditions in medical images. All attributes must describe measurable or categorical properties that are clinically meaningful. Standard medical vocabularies (RadLex, SNOMED-CT, ICD-10, LOINC) must be used for coding when available. Clinical validation and medical domain expertise are required for all finding model definitions.

**Rationale**: Medical data models directly impact patient care and clinical decision-making. Inaccurate or non-standard models can lead to misinterpretation of findings and potential patient harm.

### II. Protocol-Based Architecture
AI-powered tools must use flexible, protocol-based architectures that support multiple backends and graceful degradation.

All ontology search tools implement the `OntologySearchProtocol` to enable pluggable backends (LanceDB vector search, BioOntology REST API, DuckDB). Systems must handle backend failures gracefully and continue operating with available services. Parallel execution across backends is required when possible to maximize performance and reliability.

**Rationale**: Medical AI systems require high reliability and must not depend on single points of failure. Protocol-based design ensures extensibility and robustness.

### III. ID Immutability and Standardization
OIFM IDs are immutable once assigned and must follow the strict format `OIFM_{SOURCE}_{6_DIGITS}`.

Source prefixes identify contributing organizations (3-4 uppercase letters, e.g., MSFT, MGB). IDs must be validated against existing repositories to prevent conflicts. The hierarchical structure Model → Attribute → Value must be maintained. All ID generation must include duplicate checking via GitHub repository validation.

**Rationale**: Immutable IDs ensure long-term referential integrity across distributed medical systems and enable safe model sharing between organizations.

### IV. Async-First Development
All external API interactions must be asynchronous with proper error handling and rate limiting.

Functions calling OpenAI, Perplexity, MongoDB, or ontology APIs must use async/await patterns. Implement exponential backoff for API failures. Use structured outputs with Pydantic models for all LLM interactions. Provide fallback behavior when external services are unavailable.

**Rationale**: Medical applications require responsive user interfaces and must handle network failures gracefully. Async patterns prevent blocking operations and improve system reliability.

### V. Test-Driven Quality Assurance
Testing must cover core business logic with proper mocking of external services.

Use `@pytest.mark.callout` for tests requiring external APIs. Mock all external API calls in unit tests to prevent accidental charges and ensure test reliability. Focus test coverage on domain logic, validation rules, and error handling. All new features require comprehensive test coverage before merge.

**Rationale**: Medical software requires exceptional reliability. Comprehensive testing prevents regressions that could impact clinical workflows.

## Development Standards

### Code Quality Requirements
All code must pass strict type checking (MyPy), formatting (Ruff), and linting before merge. Line length limit is 120 characters. Python 3.11+ compatibility is required with expectation of 3.12+ deployment environments. All public APIs require docstrings with usage examples.

### AI Tool Integration Standards  
OpenAI models for natural language generation, Perplexity for detailed medical research, Pydantic AI for structured LLM interactions. All AI-generated content must be validated against medical standards. Rate limiting and cost monitoring are mandatory for all AI tool usage.

### Performance Requirements
Search operations must complete within 10 seconds for single-concept queries. Database operations must use efficient indexing (MongoDB, DuckDB HNSW). Memory usage must be optimized for processing large finding model repositories.

## Governance

### Amendment Process
Constitution changes require documentation of impact assessment, approval from project maintainers, and migration plan for affected code. All template files in `.specify/templates/` must be updated to maintain consistency.

### Compliance Review
All pull requests must verify compliance with medical domain integrity, ID standards, and async patterns. Code complexity must be justified and documented. Breaking changes require MAJOR version increment per semantic versioning.

### Runtime Guidance
Use `.github/copilot-instructions.md` for development guidance and architectural decisions. All AI coding assistants must follow established patterns and conventions.

**Version**: 1.0.0 | **Ratified**: 2025-09-20 | **Last Amended**: 2025-09-20