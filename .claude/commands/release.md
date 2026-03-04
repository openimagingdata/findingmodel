---
description: Orchestrate a complete release with interactive confirmations at every write step
argument-hint: <package1> <package2> ... (e.g., "oidm-common anatomic-locations oidm-maintenance")
---

# Release Command

**CRITICAL**: You MUST use the `date` command to get the current date. DO NOT assume you know the date.

## Monorepo Context

This is a monorepo with 5 packages under `packages/`. Releases typically involve **multiple packages at once**. The user specifies which packages to release. Key facts:

- **Published to PyPI**: `oidm-common`, `findingmodel`, `anatomic-locations`, `findingmodel-ai`
- **NOT published to PyPI**: `oidm-maintenance` (internal tool, but still gets version bumps and tags)
- **Dependency order**: `oidm-common` → `anatomic-locations`, `findingmodel` → `findingmodel-ai`, `oidm-maintenance`
- **Tag format**: `<package>-v<version>` (e.g., `oidm-common-v0.2.6`, `anatomic-locations-v0.2.5`)
- **One combined GitHub release** per batch (not one per package)

## Process Overview

Fail-fast ordering: PyPI publish happens BEFORE git tag/GitHub release to avoid inconsistent state.

1. **Pre-flight checks** - Verify clean git, on dev branch (assumes tests/checks already run)
2. **Version & CHANGELOG** - Bump versions in pyproject.toml files, update dependency floors, `uv lock`
3. **Commit to dev** (local only, not pushed)
4. **Merge to main** (local only, not pushed)
5. **Build packages** - Create wheels and tarballs for PyPI-published packages only
6. **Verify artifacts** - Show checksums and sizes
7. **Publish to PyPI** - Publish in dependency order; most likely to fail, do FIRST
8. **Create git tags** - One per package (local only)
9. **Push main + tags + dev**
10. **GitHub release** - Single combined release with all packages
11. **Cleanup** - Return to dev, merge main back

## Instructions for Claude

**CRITICAL REMINDERS**:
- 🚨 **CHECK THE DATE** - Run `date +%Y-%m-%d` command, don't assume!
- Get user confirmation before EVERY write operation
- Show clear summaries before confirmations
- If any step fails, provide rollback instructions

### Step 1: Pre-flight Checks

```bash
date +%Y-%m-%d                           # Get current date
git branch --show-current                # Must be 'dev'
git status --porcelain                   # Must be clean (untracked OK)
git fetch origin
git rev-list HEAD...origin/dev --count   # Should be 0 (up to date)
```

If local is ahead of origin, push dev first. If not clean or not on dev, stop with instructions to fix.

### Step 2: Version & CHANGELOG

For each package being released:
1. Bump `version` in `packages/<pkg>/pyproject.toml`
2. If a package depends on another package being released, bump the dependency floor (e.g., `oidm-common>=0.2.6`)
3. Run `uv lock` to update the lockfile

Read `CHANGELOG.md` (root level, single file for all packages). Verify entries exist for each package with the target version. If entries don't have dates, add the current date.

Show user a summary table:
```
| Package            | Old   | New   | PyPI? |
|--------------------|-------|-------|-------|
| oidm-common        | 0.2.5 | 0.2.6 | Yes   |
| anatomic-locations | 0.2.4 | 0.2.5 | Yes   |
| oidm-maintenance   | 0.2.3 | 0.2.4 | No    |

Dependency floor changes:
- oidm-maintenance: oidm-common>=0.2.2 → >=0.2.6
```

### Step 3: Commit to Dev (Local)

```bash
git add packages/*/pyproject.toml CHANGELOG.md uv.lock
git commit -m "Prepare release: pkg1-X.Y.Z, pkg2-X.Y.Z, ..."
```

### Step 4: Merge to Main (Local)

```bash
git checkout main
git pull origin main
git merge --no-ff dev -m "Merge dev: pkg1-X.Y.Z, pkg2-X.Y.Z, ..."
```

### Step 5: Build Packages

Only build packages that will be published to PyPI:
```bash
rm -rf dist/
uv build --package oidm-common
uv build --package anatomic-locations
# etc. — skip oidm-maintenance
```

### Step 6: Verify Artifacts

Show checksums and sizes for all built artifacts.

### Step 7: Publish to PyPI (FIRST!)

**CRITICAL**: Publish in dependency order (e.g., oidm-common before anatomic-locations).

```bash
export $(grep UV_PUBLISH_TOKEN ../.env | xargs)
uv publish dist/oidm_common-*.tar.gz dist/oidm_common-*.whl
uv publish dist/anatomic_locations-*.tar.gz dist/anatomic_locations-*.whl
# etc.
```

**On failure**: Nothing pushed yet. Can `git reset --hard` to rollback.

### Step 8: Create Git Tags (Local)

One tag per package, format `<package>-v<version>`:
```bash
git tag -a oidm-common-v0.2.6 -m "Release oidm-common 0.2.6"
git tag -a anatomic-locations-v0.2.5 -m "Release anatomic-locations 0.2.5"
git tag -a oidm-maintenance-v0.2.4 -m "Release oidm-maintenance 0.2.4"
```

### Step 9: Push Main + Tags + Dev

```bash
git push origin main
git push origin <tag1> <tag2> <tag3>
git push origin dev
```

### Step 10: Create GitHub Release

**ONE combined release** for all packages. Use any of the new tags (typically the last one alphabetically). The title lists all packages. The body has `## package version` sections pulled from CHANGELOG.md.

**Title format**: `pkg1 X.Y.Z / pkg2 X.Y.Z / pkg3 X.Y.Z`

**Body format**: Each package gets a `## package version` header followed by its CHANGELOG sections.

**Attach all built artifacts** (wheels + tarballs for PyPI-published packages).

```bash
gh release create oidm-maintenance-v0.2.4 \
  --title "oidm-common 0.2.6 / anatomic-locations 0.2.5 / oidm-maintenance 0.2.4" \
  --notes "$(cat <<'EOF'
## oidm-common 0.2.6

### Added

- `create_fts_index()` now accepts an optional `ignore` regex for DuckDB FTS tokenization.

## anatomic-locations 0.2.5

### Changed

- Lowered semantic search minimum similarity threshold from `0.75` to `0.60`.

## oidm-maintenance 0.2.4

### Changed

- Anatomic DB build now preserves alphanumeric FTS tokens (for terms like `T12` and `C7/T1`).
EOF
)" \
  dist/oidm_common-0.2.6-py3-none-any.whl \
  dist/oidm_common-0.2.6.tar.gz \
  dist/anatomic_locations-0.2.5-py3-none-any.whl \
  dist/anatomic_locations-0.2.5.tar.gz
```

### Step 11: Cleanup - Return to Dev

```bash
git checkout dev
git merge main
git push origin dev
```

### Step 12: Summary

```
Release complete!

Published to PyPI:
  - oidm-common 0.2.6: https://pypi.org/project/oidm-common/0.2.6/
  - anatomic-locations 0.2.5: https://pypi.org/project/anatomic-locations/0.2.5/

GitHub release:
  - https://github.com/openimagingdata/findingmodel/releases/tag/<tag>

Tags created: oidm-common-v0.2.6, anatomic-locations-v0.2.5, oidm-maintenance-v0.2.4
```

## Error Recovery

**PyPI publish failure** (Step 7 - most critical):
- Nothing pushed to git yet
- Local commits on dev and main
- Recovery: `git checkout dev && git reset --hard origin/dev && git checkout main && git reset --hard origin/main`

**GitHub release failure** (Step 10):
- PyPI already published (can't undo)
- Tags pushed
- Recovery: Create release manually via GitHub UI

**General errors**:
- Show current branch: `git branch --show-current`
- Show status: `git status`
- Suggest appropriate rollback for the failed step

## Notes

- Requires: `uv`, `git`, `gh` CLI tools
- PyPI token: Set `UV_PUBLISH_TOKEN` in `../.env`
- Fail-fast design: PyPI publish before git tag ensures clean rollback on failure
- Can verify user ran tests/checks first, but don't run them here
