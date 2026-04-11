# Publish Path Hardening

**Date:** 2026-04-02

## Goal

Raise confidence that publishing new `findingmodel` and `anatomic-locations` database artifacts always produces the correct database metadata, S3 object layout, and manifest entries.

## Why This Exists

Current confidence is good but not complete:

- build code uses the canonical OpenAI embedding config
- publish code updates only the base manifest entries
- shared S3 helpers are tested
- publish orchestration itself is not directly tested
- build code does not explicitly write `embedding_profile` metadata tables into published databases
- `findingmodel --no-embeddings` still hardcodes `512` for zero vectors

## Plan

### 1. Make Build Outputs Explicit

- add explicit `embedding_profile` metadata table creation to both database build paths
- write provider=`openai`, model=`text-embedding-3-small`, dimensions=`512` from the canonical shared config
- ensure `--no-embeddings` builds still write the same metadata
- remove the hardcoded `512` zero-vector dimension from `findingmodel` build

### 2. Add Direct Publish Coverage

- add unit tests for `publish_findingmodel_database()`
- add unit tests for `publish_anatomic_database()`
- verify uploaded object key, manifest backup behavior, manifest entry contents, and dry-run behavior
- verify failures short-circuit cleanly before partial manifest updates

### 3. Verify End-to-End Semantics

- add or extend tests to assert freshly built databases expose the expected `embedding_profile` metadata
- verify runtime config accepts those freshly built artifacts without relying on fallback-only behavior
- rerun targeted build/config/publish tests for `oidm-maintenance`, `findingmodel`, `anatomic-locations`, and `oidm-common`

### 4. Update Documentation

- update this plan with final status and outcomes
- update relevant maintainer docs if publish/build guarantees change materially
- review `CHANGELOG.md` and active docs for any needed outward-facing note

## Suggested Execution Order

1. write metadata table in both builders
2. remove hardcoded zero-vector dimensions
3. add direct publish-function tests
4. run targeted verification
5. update docs and mark this task complete
