# Documentation Update Plan

This document lists documentation updates needed to align with the current codebase behavior and CLI surface area.

## High Priority (Behavioral Mismatches)

- Root CLI docs
  - `README.md` references `findingmodel search` and `anatomic-locations search` commands; the CLIs currently expose:
    - `findingmodel`: `config`, `fm-to-markdown`, `index stats`
    - `anatomic-locations`: `stats`, `query ...` subcommands (no `search` command)
  - Update the CLI sections and examples to match `packages/findingmodel/src/findingmodel/cli.py` and
    `packages/anatomic-locations/src/anatomic_locations/cli.py`.

- Anatomic locations API docs
  - `docs/anatomic-locations.md` shows `index.search()` returning `(location, score)` tuples; the API returns
    a list of `AnatomicLocation` objects. See `packages/anatomic-locations/src/anatomic_locations/index.py`.
  - `docs/anatomic-locations.md` uses `ANATOMIC_DUCKDB_PATH`; the actual setting is `ANATOMIC_DB_PATH`
    with `ANATOMIC_REMOTE_DB_URL`, `ANATOMIC_REMOTE_DB_HASH`, and `ANATOMIC_MANIFEST_URL`.

- MCP server tool list
  - `packages/findingmodel/README.md` mentions `list_finding_model_tags`; this tool is not implemented in
    `packages/findingmodel/src/findingmodel/mcp_server.py`. Update tool list to match
    `search_finding_models`, `get_finding_model`, `count_finding_models`.

- Database configuration docs
  - `docs/configuration.md` references `DUCKDB_ANATOMIC_PATH` and `REMOTE_ANATOMIC_DB_URL/HASH`. Replace with
    the anatomic package settings from `packages/anatomic-locations/src/anatomic_locations/config.py`.
  - Add `REMOTE_MANIFEST_URL` for findingmodel (present in `packages/findingmodel/src/findingmodel/config.py`).
  - `docs/database-management.md` mentions `FINDINGMODEL_DUCKDB_INDEX_PATH` and `ANATOMIC_DUCKDB_PATH` which
    are not current settings; update to match actual env vars.

## Medium Priority (Doc Location Drift)

- Finding enrichment documentation
  - `docs/finding-enrichment-prd.md` and `docs/finding-enrichment-implementation-plan.md` reference
    `findingmodel/tools/finding_enrichment.py`, `scripts/enrich_finding.py`, and `test/test_finding_enrichment.py`.
  - Actual implementation is in `packages/findingmodel-ai/src/findingmodel_ai/enrichment/*.py`,
    `packages/findingmodel-ai/scripts/enrich_finding.py`, and
    `packages/findingmodel-ai/tests/test_finding_enrichment.py`.
  - Update file paths and any architecture diagrams or steps that cite old locations.

- Logfire guidance
  - `docs/logfire_observability_guide.md` describes per-module configuration and extra env vars that are not
    reflected in `packages/findingmodel-ai/evals/__init__.py` (current behavior is a simple package-level
    `logfire.configure(..., console=False)` and instrumentation via `ensure_instrumented()`).
  - Align doc guidance with the actual evals entry point and supported env vars.

## Low Priority (Consistency / Clarity)

- Cross-reference updates
  - Ensure `README.md` links to the correct CLI command names and doc sections after edits.
  - Keep `docs/database-management.md` and `docs/duckdb-development.md` in sync for manifest and
    distribution settings.

## Suggested Execution Order

1. Fix CLI command references in `README.md`, `packages/*/README.md`.
2. Update `docs/anatomic-locations.md` API examples + env var names.
3. Update `docs/configuration.md` and `docs/database-management.md` env vars.
4. Update enrichment docs to point to `findingmodel-ai` locations.
5. Reconcile Logfire docs with evals implementation.

## Verification Checklist

- All CLI examples match the actual command tree.
- All env var names appear in code configs and docs consistently.
- Doc references to file locations reflect current package structure.
