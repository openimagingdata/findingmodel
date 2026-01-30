# oidm-common

Internal infrastructure package for the Open Imaging Data Model (OIDM) ecosystem.

## ⚠️ Not for Direct Use

This package provides shared infrastructure for OIDM packages. **Do not install this package directly.**

Instead, use one of the user-facing packages:
- [`findingmodel`](https://pypi.org/project/findingmodel/) - Finding model index and search
- [`anatomic-locations`](https://pypi.org/project/anatomic-locations/) - Anatomic location ontology
- [`findingmodel-ai`](https://pypi.org/project/findingmodel-ai/) - AI-powered finding model tools

## Contents

- DuckDB connection management and hybrid search
- Embedding cache and providers
- Distribution utilities (manifest, download, paths)
- Shared data models (IndexCode, WebReference)
