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

## Embedding cache

High-level helpers (`get_embedding`, `get_embeddings_batch`) use a SQLite-backed cache (via
`diskcache`) by default in the platform cache directory (for example:
`~/Library/Caches/oidm-common/embeddings.cache` on macOS).

Legacy cache data from the previous DuckDB file in the same platform cache directory
(for example: `~/Library/Caches/findingmodel/embeddings.duckdb` on macOS) and from the previous
legacy diskcache path (`~/Library/Caches/findingmodel/embeddings.cache`) is migrated once on first
setup of the default cache, assuming the current default embedding settings
(`text-embedding-3-small`, `512` dimensions). Custom cache directories stay isolated and do not
auto-import global legacy cache data.

Pass `cache=None` to disable caching, or pass an `EmbeddingCache` instance to override the source
path (for migration) and cache directory.
