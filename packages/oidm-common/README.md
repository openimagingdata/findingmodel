# oidm-common

Shared infrastructure for OIDM packages. This is an internal package used by other packages in the workspace.

## Purpose

Provides common utilities shared across OIDM packages:

- **Database Auto-Download**: Pooch-based download with checksum verification
- **Embedding Client**: OpenAI embedding generation (optional `[openai]` extra)
- **Database Utilities**: DuckDB connection helpers and async patterns

## Installation

```bash
# Basic installation
pip install oidm-common

# With OpenAI embedding support
pip install oidm-common[openai]
```

## Usage

This package is typically used as a dependency by other OIDM packages rather than directly. See:

- [findingmodel](../findingmodel/README.md) - Core finding model library
- [anatomic-locations](../anatomic-locations/README.md) - Anatomic location queries

## Note

This package has no AI/LLM dependencies. AI tooling lives in [findingmodel-ai](../findingmodel-ai/README.md).
