# FindingModel Project Overview

## Purpose
The `findingmodel` package is a Python library for managing Open Imaging Finding Models - structured data models used to describe medical imaging findings in radiology reports. It provides tools for creating, converting, and managing these finding models with AI integration.

## Tech Stack
- **Language**: Python 3.11+
- **Build System**: uv (modern Python project management)
- **Task Runner**: Task (go-task) for development commands
- **Package Format**: Standard Python package with pyproject.toml
- **Dependencies**:
  - pydantic (v2) for data models and validation
  - pydantic-ai-slim for AI-powered tools (OpenAI, Anthropic, Google, Ollama, Gateway)
  - duckdb for index/search with HNSW vector and FTS indexes
  - click for CLI
  - rich for terminal output
  - loguru for logging
  - tavily-python for web search in AI workflows

## Project Structure
```
findingmodel/
├── src/findingmodel/       # Main package source
│   ├── tools/              # AI-powered tools for finding models
│   ├── finding_model.py    # Core data models (FindingModelBase, FindingModelFull)
│   ├── finding_info.py     # FindingInfo data model
│   ├── index.py            # DuckDB-based indexing system
│   ├── config.py           # Configuration management
│   └── cli.py              # Command-line interface
├── test/                   # Test suite
│   └── data/              # Test fixtures
├── evals/                 # Agent evaluation suites
├── notebooks/             # Example Jupyter notebooks
├── pyproject.toml         # Project configuration
├── Taskfile.yml          # Task runner commands
├── CLAUDE.md             # Project instructions for Claude Code
└── .env.sample           # Environment variables template
```

## Key Features
1. **Data Models**: Hierarchical finding model classes (FindingInfo → FindingModelBase → FindingModelFull)
2. **AI Tools**: Generate finding descriptions, create models from markdown, add medical codes
3. **Index System**: DuckDB-based lookup with HNSW vector search and full-text search
4. **CLI**: Command-line tools for model conversion and generation
5. **Evaluation System**: Pydantic Evals-based quality assessment for AI agents
