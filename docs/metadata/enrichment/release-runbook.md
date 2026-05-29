# Metadata Release Runbook

Sequenced, executable steps to release the metadata-aware libraries and publish both database
artifacts. This is the *procedure*; the *strategy and rationale* live in
[`database-artifacts-and-package-pinning.md`](database-artifacts-and-package-pinning.md) and
[ADR-0003](../../adr/0003-dual-db-pre-post-metadata-release.md).

**Do not run this until** the cleanup slices are committed, the approved source baseline (67
approved + 11 display backfills) is in place, and Gate A has passed (see
[`human-review-and-writeback.md`](human-review-and-writeback.md)).

## Preconditions

- Approved-baseline source is committed in the data repo; Gate A passed.
- Both build scripts (current-compatible and metadata-aware) exist and run in the data repo.
- One canonical `findingmodels` source commit is chosen for both DBs.
- Legacy schema contract present: `docs/database-schemas/finding_models_legacy_2026-01-28.schema.json`.

## Steps

1. **Fix the source commit.** Identify the single enriched `findingmodels` commit both artifacts
   build from. Both DBs MUST come from the same commit.

2. **Release the libraries (off the wheelhouse).**
   - Publish the metadata-aware package versions to the intended channel.
   - Update data-repo scripts from local wheelhouse (`.metadata-runs/wheelhouse/current/`) to
     released package pins.
   - Rerun the affected data-repo scripts with the released pins.
   - Update package docs and `CHANGELOG.md` with concise user-facing metadata/DB changes.
   - The final data branch must not depend on local wheelhouse paths.

3. **Build both DBs from the one commit.**
   - Current-compatible → manifest key `finding_models` (legacy build script; legacy schema).
   - Metadata-aware → manifest key `finding_models_metadata` (metadata columns + full enriched JSON
     + `database_metadata` provenance table).

4. **Validate both.**
   - Current-compatible: against the legacy schema contract AND the current published `findingmodel`
     runtime.
   - Metadata-aware: against metadata schema/provenance expectations AND the metadata-aware runtime
     (open/query/browse/search).
   - Confirm record counts agree between the two where expected.

5. **Prepare the manifest.**
   - Back up the manifest before any change.
   - Add source/build metadata to both entries: source commit, tooling/package versions, schema
     kind, record count, hash, build timestamp.
   - Preserve and back up the ontology cache used as evidence for the metadata DB.
   - Rehearse the manifest update shape locally; confirm both keys are present and neither overwrites
     the other.

6. **Publish.** Apply the manifest update with both keys. Optionally flip the metadata-aware runtime
   default manifest key to `finding_models_metadata` — only if that is the chosen rollout behavior.

7. **Post-publish verification (both artifacts).** Download, validate hash, open/query via runtime,
   and exercise representative search/browse. Do this for `finding_models` and
   `finding_models_metadata`.

## Rollback

- **Failed post-publish check:** restore the backed-up manifest (revert the key changes), leave the
  default key unflipped, and keep `finding_models` as the live default until the issue is fixed and
  re-verified. Artifacts can stay uploaded but unreferenced.
- **Bad package release:** forward-fix with a new version (do not unpublish); if a data-repo script
  is affected, pin back to the last-good version.
- **Default-key flip regret:** flip the default manifest key back to `finding_models`; the
  metadata-aware artifact remains available under its own key.

## Guardrails

- Never publish pilot-only or partial-review artifacts — those are validation builds only.
- Never publish from local-wheelhouse state; release packages first (step 2).
- Both DBs come from the same source commit.
- Never overwrite an existing manifest key; both keys coexist.
