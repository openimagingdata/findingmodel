# FindingModel Project Overview

## Purpose
The `findingmodel` package is a Python library for managing Open Imaging Finding Models - structured data models used to describe medical imaging findings in radiology reports. It provides tools for creating, converting, and managing these finding models with AI integration (OpenAI and Perplexity).

## Tech Stack
- **Language**: Python 3.11+
- **Build System**: uv (modern Python project management)
- **Task Runner**: Task (go-task) for development commands
- **Package Format**: Standard Python package with pyproject.toml
- **Dependencies**:
  - pydantic (v2) for data models and validation
  - OpenAI SDK for AI-powered model generation
  - instructor for structured AI outputs
  - motor for MongoDB integration (optional)
  - click for CLI
  - rich for terminal output
  - loguru for logging
  - pydantic-ai-slim for AI tools

## Project Structure
```
findingmodel/
├── src/findingmodel/       # Main package source
│   ├── tools/              # AI-powered tools for finding models
│   ├── finding_model.py    # Core data models (FindingModelBase, FindingModelFull)
│   ├── finding_info.py     # FindingInfo data model
│   ├── index.py            # JSONL-based indexing system
│   ├── config.py           # Configuration management
│   └── cli.py              # Command-line interface
├── test/                   # Test suite
│   └── data/              # Test fixtures
├── notebooks/             # Example Jupyter notebooks
├── pyproject.toml         # Project configuration
├── Taskfile.yml          # Task runner commands
├── CLAUDE.md             # Project instructions for Claude Code
└── .env.sample           # Environment variables template
```

## Key Features
1. **Data Models**: Hierarchical finding model classes (FindingInfo → FindingModelBase → FindingModelFull)
2. **AI Tools**: Generate finding descriptions, create models from markdown, add medical codes
3. **Index System**: Fast JSONL-based lookup by ID, name, or synonym
4. **CLI**: Command-line tools for model conversion and generation
5. **MongoDB Integration**: Optional database support for indexing