# Plan: Move BioOntology Lookup into oidm-common

Date: 2026-01-29
Owner: TBD
Status: Planned

## Goal

Move the BioOntology API client and ontology search protocol/types out of `findingmodel-ai` into `oidm-common` as shared infrastructure. This is a **hard break**: remove `findingmodel.protocols` and update all internal imports to the new `oidm_common.ontology.*` structure. BioOntology API key must be sourced **only** from the `BIOONTOLOGY_API_KEY` environment variable.

## Scope

### In scope
- New `oidm_common.ontology` package with:
  - `bioontology.py` (client + result models)
  - `protocols.py` (protocols/types + normalization)
- Remove `findingmodel.protocols` entirely.
- Update all internal imports across `findingmodel-ai` (and any other package) to new paths.
- Update `findingmodel-ai` config to remove `bioontology_api_key` (no config-based key). 
- Add unit tests to `oidm-common` for BioOntology client behavior (mocked HTTP).
- Update documentation references to the new module location.

### Out of scope
- Any backwards-compat shim (`findingmodel_ai.tools`, `findingmodel.protocols`) — explicitly **not** desired.
- Functional changes to ontology search logic beyond the module move + env-only key.

## Design Decisions

1. **Hard break**: `findingmodel.protocols` removed.
2. **Env-only key**: `BIOONTOLOGY_API_KEY` only; no config object, no parameter override.
3. **Shared protocol types** live in `oidm-common` to avoid circular dependencies and allow reuse.
4. **Logging** uses `loguru.logger` directly in oidm-common (no `findingmodel_ai.logger`).

## Implementation Steps

1. **Create new package module**
   - Add `packages/oidm-common/src/oidm_common/ontology/__init__.py`.
   - Add `bioontology.py` and `protocols.py`.
   - Move/refactor from `packages/findingmodel-ai/src/findingmodel_ai/search/bioontology.py` and `packages/findingmodel/src/findingmodel/protocols.py`.

2. **Refactor BioOntology client**
   - Remove any dependency on `findingmodel_ai.config`.
   - On init, read `BIOONTOLOGY_API_KEY` from env; raise if missing.
   - Preserve optional `httpx.AsyncClient` injection for tests.

3. **Hard break in findingmodel**
   - Delete `packages/findingmodel/src/findingmodel/protocols.py`.
   - Update all internal imports to use `oidm_common.ontology.protocols`.

4. **Update findingmodel-ai**
   - Replace imports of `BioOntologySearchClient` with `oidm_common.ontology.bioontology`.
   - Remove config guard `settings.bioontology_api_key` in `findingmodel_ai/search/ontology.py`.
   - Remove `bioontology_api_key` from `findingmodel_ai/config.py`.

5. **Tests**
   - Add mocked HTTP tests in `packages/oidm-common/tests/` using `httpx.MockTransport`.
   - Ensure `BIOONTOLOGY_API_KEY` env is set in tests via `monkeypatch`.
   - Update `findingmodel-ai` tests to import new module path.

6. **Docs**
   - Update `packages/findingmodel-ai/README.md` and any other docs referencing `findingmodel_ai.search.bioontology` or `findingmodel.protocols`.
   - Add a small section to `packages/oidm-common/README.md` describing the BioOntology client.

## Files to Touch (Initial List)

- NEW: `packages/oidm-common/src/oidm_common/ontology/__init__.py`
- NEW: `packages/oidm-common/src/oidm_common/ontology/bioontology.py`
- NEW: `packages/oidm-common/src/oidm_common/ontology/protocols.py`
- DELETE: `packages/findingmodel/src/findingmodel/protocols.py`
- UPDATE: `packages/findingmodel-ai/src/findingmodel_ai/search/ontology.py`
- UPDATE: `packages/findingmodel-ai/src/findingmodel_ai/search/__init__.py`
- UPDATE: `packages/findingmodel-ai/src/findingmodel_ai/config.py`
- TESTS: `packages/oidm-common/tests/test_bioontology.py` (new)
- DOCS: `packages/findingmodel-ai/README.md`, `packages/oidm-common/README.md`, any other references

## Acceptance Criteria

- No file `packages/findingmodel/src/findingmodel/protocols.py` exists.
- All imports resolve to `oidm_common.ontology.*`.
- BioOntology API key comes **only** from `BIOONTOLOGY_API_KEY` env var.
- `task test:oidm-common` and `task test:findingmodel-ai` pass.
- Docs updated to reference new module locations.

## Risks & Mitigations

- **Breaking imports**: hard break may affect downstream consumers. Mitigation: update docs, changelog, and internal references in the same PR.
- **Test flakiness**: external API use. Mitigation: mock HTTP requests in tests and avoid live calls.

## Verification

- `task test:oidm-common`
- `task test:findingmodel-ai`
- `task check`

