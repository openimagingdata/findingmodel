# Remove Local Embeddings / FastEmbed Completely

**Date:** 2026-03-22

## Summary

Delete all FastEmbed and local-embedding support from the repo.

This is not a deprecation plan and not a "leave the code unreachable" cleanup. The end state is:

- no FastEmbed implementation code
- no local embedding provider/model references
- no runtime or build-time embedding-profile choice
- no local-profile database artifact routing
- no tests for local/FastEmbed behavior
- no documentation or changelog entries describing the experiment

After this work, the repo supports exactly one embedding configuration everywhere it matters operationally:

- provider: `openai`
- model: `text-embedding-3-small`
- dimensions: `512`

Internal metadata structures may continue to describe provider/model/dimensions for the currently supported OpenAI path, but they must not preserve any FastEmbed-specific code, values, branches, or config knobs.

## Locked Decisions

- FastEmbed code must be deleted, not merely gated or made unreachable.
- No backward-compatibility work is needed for unreleased local-profile artifacts or config.
- User-facing configuration must not expose any embedding-technique choice.
- `CHANGELOG.md` should be scrubbed so the unreleased FastEmbed/local-embedding work disappears from outward-facing history.
- Generic internal metadata abstractions are acceptable only if they describe the single supported OpenAI configuration and do not retain alternate-provider behavior.

## Scope

This cleanup must cover all relevant packages and shared infrastructure:

- `findingmodel`
- `anatomic-locations`
- `oidm-common`
- `oidm-maintenance`
- shared docs, README/config samples, task docs, and `CHANGELOG.md`

It must cover:

- runtime config
- database download / artifact resolution
- database build paths
- search behavior
- test fixtures and test coverage
- public and maintainer documentation

## Workstreams

### 1. Runtime Configuration Cleanup

Remove all embedding-technique selection from package settings and env parsing.

Required outcomes:

- Delete `FINDINGMODEL_EMBEDDING_PROFILE`.
- Delete `ANATOMIC_EMBEDDING_PROFILE`.
- Remove any `auto`, `local`, or profile-selection semantics from config classes, env validation, default handling, and config tests.
- `findingmodel` must stop defaulting to local embeddings when no OpenAI key is present.
- `anatomic-locations` must stop carrying scaffolding whose only purpose was to distinguish supported OpenAI from unsupported local embeddings.
- Package docs and examples must describe only the single supported OpenAI embedding configuration.

Implementation intent:

- Keep only the minimum config needed to locate DB artifacts and provide OpenAI credentials where required.
- Do not replace removed profile knobs with new hidden equivalents.

### 2. `findingmodel` Search and Artifact Resolution

Remove all local/FastEmbed behavior from `findingmodel`.

Required outcomes:

- Search remains hybrid, but semantic search is OpenAI-only.
- Delete provider-branching whose purpose is deciding between OpenAI and FastEmbed.
- Remove local-profile artifact selection from runtime DB resolution.
- Keep validation that the active DB artifact matches the supported OpenAI embedding metadata.
- Remove all local-profile wording from MCP/server docs and user-facing package docs.

Implementation intent:

- `findingmodel` should request and operate against the single supported OpenAI embedding tuple.
- There should be no local-profile fallback path in runtime behavior.

### 3. `anatomic-locations` Cleanup

Simplify `anatomic-locations` so local embeddings are not a concept anywhere in the package.

Required outcomes:

- Remove any profile-choice language from config, runtime validation, and tests.
- Remove any code or docs that explicitly discuss local-profile support as unsupported.
- Keep only the current OpenAI-backed search path and DB validation.

Implementation intent:

- The package should simply support its current OpenAI embedding configuration without profile vocabulary.

### 4. `oidm-common` Embeddings Cleanup

Delete FastEmbed support from shared embedding infrastructure.

Required outcomes:

- Delete FastEmbed runtime creation, imports, model cache handling, temp/cache directory logic, warnings, and tests.
- Remove FastEmbed-specific constants, types, provider literals, model names, and fallback behavior.
- Remove any helper branches that exist solely to support local embeddings.
- Keep only OpenAI embedding generation/caching behavior.

Implementation intent:

- If shared helper signatures keep provider/model/dimensions fields for future online-only flexibility, they must no longer encode or implement alternate FastEmbed behavior.
- No FastEmbed string, provider value, or code path should remain in `oidm-common`.

### 5. Distribution and Database Artifact Plumbing

Remove local-profile artifact routing and associated fixtures without losing useful metadata validation.

Required outcomes:

- Remove local-profile manifest keys, alias expectations, and test fixtures such as `__local` artifact handling.
- Remove package-level behavior that selects between embedding profiles for downloads.
- Keep only the internal validation necessary to confirm that a DB artifact matches the supported OpenAI provider/model/dimensions.
- Remove docs that describe profile-aware artifact selection for local embeddings.

Implementation intent:

- It is acceptable for shared internals to retain generic metadata readers/builders, but not if they preserve any FastEmbed/local-specific values or behavior.
- Package behavior should no longer advertise or exercise profile choice.

### 6. Build / Maintenance Tooling

Ensure build scripts and maintainer workflows no longer respect or imply local embeddings.

Required outcomes:

- Remove any FastEmbed/local-provider references from maintainer docs, config samples, CLI help, and tests.
- Ensure DB build paths describe only the current OpenAI embedding generation route.
- Remove any provider/profile selection knobs that exist for the abandoned local-embedding path.
- Recheck build fixtures and manifest examples for local-profile references and delete them.

Implementation intent:

- The maintenance surface should describe one supported embedding build path and nothing else.

### 7. Test Cleanup

Delete or rewrite tests so no local/FastEmbed behavior is asserted anywhere.

Required outcomes:

- Delete tests for local config defaults and local profile selection.
- Delete tests for FastEmbed runtime creation, FastEmbed cache-path behavior, local manifest alias routing, and local artifact resolution.
- Rewrite shared/config/distribution tests to assert the single supported OpenAI embedding configuration.
- Keep coverage for OpenAI-only config, DB validation, download resolution, embedding generation/caching, and search behavior.

Implementation intent:

- This should reduce test surface, not just rename it.
- Tests must prove the absence of embedding-technique choice, not merely stop mentioning it.

### 8. Documentation and History Cleanup

Remove every outward-facing trace of the experiment.

Required outcomes:

- Remove all mentions of:
  - `fastembed`
  - `BAAI/bge-small-en-v1.5`
  - local embeddings
  - local-profile artifacts
  - `FINDINGMODEL_EMBEDDING_PROFILE`
  - `ANATOMIC_EMBEDDING_PROFILE`
  - `auto` / `local` profile values
- Update README, configuration docs, MCP docs, DuckDB/database docs, manifest docs, `.env.sample`, and any active architecture/task docs that describe current behavior.
- Scrub `CHANGELOG.md` of every unreleased FastEmbed/local-embedding note.

Implementation intent:

- After cleanup, repo documentation should read as though the local-embedding experiment never shipped and never became a supported option.

## Acceptance Criteria

The work is complete only when all of the following are true:

1. No FastEmbed implementation code remains in tracked source files.
2. No local embedding provider/model names remain in tracked source files.
3. No runtime package exposes an embedding-profile env var or profile-selection behavior.
4. No package runtime or build flow selects between OpenAI and local embeddings.
5. No package docs or samples advertise local embeddings, local-profile artifacts, or profile choice.
6. `CHANGELOG.md` contains no unreleased FastEmbed/local-embedding references.
7. Repo-wide search for the following yields no relevant tracked-file hits other than historical completed-task context that is explicitly left alone on purpose:
   - `fastembed`
   - `BAAI/bge-small-en-v1.5`
   - `FINDINGMODEL_EMBEDDING_PROFILE`
   - `ANATOMIC_EMBEDDING_PROFILE`
   - `__local`
   - `local-profile`
8. Targeted tests covering config, shared embedding helpers, distribution, maintenance, and runtime search all pass after the rewrite.

## Suggested Verification Pass

Run a focused verification pass after implementation:

- config and distribution tests for all affected packages
- shared embedding helper tests
- maintenance/build tests that cover DB generation and manifest plumbing
- runtime search tests for `findingmodel` and `anatomic-locations`
- repo-wide grep for removed terms

## Final Documentation Pass

Before marking this task complete:

- update this task file to reflect the final landed scope
- review `CHANGELOG.md` for any lingering FastEmbed/local references
- review active docs/tasks for stale references to profile selection or local artifacts
- move the task to `tasks/done/` only after the final documentation state matches the codebase

## Audit Pass - 2026-03-27

Branch audit results:

- repo-wide search found no remaining tracked-file references to `fastembed`, `BAAI/bge-small-en-v1.5`, `FINDINGMODEL_EMBEDDING_PROFILE`, `ANATOMIC_EMBEDDING_PROFILE`, `__local`, local-profile artifact routing, or maintainer embedding-profile knobs outside this task document
- package runtime config now exposes only OpenAI API key configuration; runtime embedding-profile selection is gone from `findingmodel`, `anatomic-locations`, `oidm-common`, and `oidm-maintenance`
- distribution plumbing no longer resolves profile-specific manifest keys or alias maps; managed downloads now use only the base manifest entry
- shared embedding generation is OpenAI-only; FastEmbed runtime/cache handling and related tests are removed
- `CHANGELOG.md`, `.env.sample`, README/configuration/MCP/DuckDB docs, and affected package docs were checked and no longer describe FastEmbed/local-embedding runtime support

Targeted verification run:

- `uv run pytest packages/findingmodel/tests/test_config.py` -> 16 passed
- `uv run pytest packages/anatomic-locations/tests/test_config.py` -> 17 passed
- `uv run pytest packages/oidm-common/tests/test_distribution.py packages/oidm-common/tests/test_embeddings.py` -> 62 passed
- `uv run pytest packages/oidm-maintenance/tests/test_config.py packages/oidm-maintenance/tests/test_cli.py packages/oidm-maintenance/tests/test_anatomic_build_internals.py packages/oidm-maintenance/tests/test_findingmodel_build.py` -> 101 passed

Conclusion:

- for the active code, config, docs, and targeted tests on this branch, the FastEmbed/local-embedding removal appears complete
- the only intentional remaining `fastembed` references are in this task document as historical planning context

## Publish Path Audit - 2026-03-29

Confidence for publishing a new database artifact today: moderate, not absolute.

What looks sound:

- both build paths now source embeddings from the single canonical OpenAI config in `oidm_common.embeddings.config.ACTIVE_EMBEDDING_CONFIG`
- both publish paths upload a single base artifact key and update only the base manifest entry (`finding_models` / `anatomic_locations`)
- shared S3/manifest helpers are covered by `packages/oidm-maintenance/tests/test_s3.py`
- targeted build/config/CLI tests for affected packages passed during the audit

Remaining confidence limits:

- there is no direct unit coverage for `publish_findingmodel_database()` or `publish_anatomic_database()` themselves; CLI tests mock those functions instead of exercising manifest backup/update and upload orchestration end-to-end
- current build code does not write an explicit `embedding_profile` metadata table into newly built databases; runtime still works by falling back to the canonical OpenAI config and vector dimensions, but the artifact metadata is less explicit than intended
- `findingmodel` `--no-embeddings` currently hardcodes zero vectors at `512` dimensions instead of reading `ACTIVE_EMBEDDING_CONFIG.dimensions`

Operational takeaway:

- a normal publish with embeddings enabled is likely to do the right thing on this branch
- confidence would be materially higher after adding direct publish-function tests and writing explicit `embedding_profile` metadata during database builds
