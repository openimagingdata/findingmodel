# Dual-database pre/post-metadata release

We publish two DuckDB artifacts built from the same enriched source commit: `finding_models`
(current-compatible, legacy schema, readable by the currently-published `findingmodel` runtime) and
`finding_models_metadata` (metadata-aware, with structured-metadata columns, enriched JSON, and a
`database_metadata` provenance table). We chose two artifacts so existing consumers keep working
unchanged while metadata-aware consumers get the new data, rather than forcing a breaking single-DB
migration.

## Consequences

- Two DBs must be built, validated against their schema contracts, and published under two manifest
  keys from one source commit.
- A package-release gate applies: data-repo scripts move off the local wheelhouse
  (`.metadata-runs/wheelhouse/`) to released package pins before publication.
- An optional default-manifest-key flip controls when the metadata-aware DB becomes the default.
