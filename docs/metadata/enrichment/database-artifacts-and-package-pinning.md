# Database Artifacts And Package Pinning

This page preserves the release/distribution plan for metadata-enriched finding models. The current
cleanup branch is not ready to publish databases yet, but the branch must keep the artifact strategy
visible so we do not lose the compatibility plan while cleaning docs and evals.

## Two Database Artifacts

The enriched source corpus must eventually produce two production DuckDB artifacts from the same
canonical `findingmodels` source commit:

- `finding_models`: current-compatible artifact for existing users.
- `finding_models_metadata`: metadata-aware artifact for metadata-capable package versions.

The current-compatible artifact exists to keep existing runtimes working while the metadata-aware
runtime and manifest key are introduced. The metadata-aware artifact exposes structured metadata
columns and stores enriched full model JSON.

Do not publish pilot-only or partial-review artifacts. Pilot and approved-baseline builds are
validation artifacts only.

## Current-Compatible DB Path

The current-compatible DB build must:

- build from the final enriched `defs/` source, not from generated review artifacts;
- use the legacy/current-compatible build script in the data repo;
- match the legacy schema contract at
  `docs/database-schemas/finding_models_legacy_2026-01-28.schema.json`;
- be readable by the current published `findingmodel` runtime;
- publish under manifest key `finding_models`;
- preserve user-facing behavior for current consumers.

During branch work, this path intentionally does not use local metadata-aware wheels. It pins the
current-compatible package set to the chosen current-main Git commit so it proves the existing
published DB contract separately from the metadata-aware path.

## Metadata-Aware DB Path

The metadata-aware DB build must:

- build from the same final enriched `defs/` source commit as the current-compatible artifact;
- use metadata-aware package versions;
- include structured metadata columns;
- store full enriched JSON;
- include the `database_metadata` provenance table;
- be readable by the metadata-aware `findingmodel` runtime;
- support representative open/query/browse/search behavior;
- publish under manifest key `finding_models_metadata`.

The `database_metadata` table should record schema name/version, source commit, build timestamp,
package versions, and embedding provider/model/dimensions when applicable.

## Local Wheelhouse During Branch Work

The data repo currently uses unpublished metadata-aware packages from this repo. During branch work,
metadata-aware scripts use PEP 723 `[tool.uv.sources]` entries pointed at local wheel files under:

```text
../findingmodels-metadata/.metadata-runs/wheelhouse/current/
```

The data-repo setup doc records the wheelhouse command and expected wheel files:

```text
../findingmodels-metadata/docs/metadata-enrichment-setup.md
```

Scripts that regenerate metadata-aware schema, docs, source-derived text, or validation output must
run against those local metadata-aware wheels during branch work. This prevents old published
packages from silently stripping metadata fields or regenerating stale schemas.

## Package Release Gate

The final data branch must not merge in a state that depends on unpublished local wheelhouse paths.
Before final data merge and DB publish:

- release or otherwise make available the metadata-aware package versions through the intended
  package channel;
- update data-repo scripts from local wheelhouse usage to released package pins;
- rerun relevant data-repo scripts with released pins;
- update package docs and `CHANGELOG.md` with concise user-facing metadata/DB behavior changes;
- change the metadata-aware runtime default manifest key to `finding_models_metadata` in the
  release-preparation work, if that remains the selected rollout behavior.

Local wheelhouse usage is an implementation bridge, not final operational state.

## Final Publish Requirements

Before publishing:

- build both DB artifacts from the same final enriched source commit;
- validate the current-compatible DB against the legacy schema contract and current runtime;
- validate the metadata-aware DB against metadata schema/provenance expectations and
  metadata-aware runtime behavior;
- preserve and back up the ontology cache used as evidence for the published metadata DB;
- add source/build metadata to both manifest entries:
  - source `findingmodels` commit;
  - tooling/package versions;
  - schema kind;
  - record count;
  - hash;
  - build timestamp;
- back up the manifest before update;
- rehearse the manifest update shape locally and confirm both artifact keys are present without
  overwriting each other;
- verify post-publish download, hash validation, runtime open/query behavior, and representative
  search/browse behavior for both artifacts.

## Current Cleanup Implication

The current cleanup milestone should not attempt to publish DBs. It should leave us with:

- reviewed and manifest-backed source changes;
- data-repo tooling that can build both validation artifacts;
- explicit package/wheel pinning documentation;
- clear gates separating local branch validation from final package release and DB publication.
