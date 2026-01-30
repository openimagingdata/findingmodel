# Plan 1: Versioning, Publishing, and Release

**Goal**: Get a clean release of all packages with proper versioning and publishing infrastructure.

---

## 0. Prerequisite: Separate Storage and Publish Updated Anatomic Database

### Storage Separation

Each package gets its own T3 bucket:

| Package | Bucket URL |
|---------|------------|
| findingmodel | `https://findingmodelsdata.t3.storage.dev/` (existing) |
| anatomic-locations | `https://anatomiclocationdata.t3.storage.dev/` (new) |

**Rationale:**
- Complete independence between packages
- Separate billing/quotas if needed
- Cleaner architecture for future schema versioning
- Each package manages its own distribution

### Config Changes

| File | Change |
|------|--------|
| `packages/findingmodel/src/findingmodel/config.py` | Keep current URL (already correct) |
| `packages/anatomic-locations/src/anatomic_locations/config.py` | Change manifest URL to new bucket |
| `packages/oidm-maintenance/src/oidm_maintenance/anatomic/publish.py` | Publish to new bucket |
| `packages/oidm-maintenance/src/oidm_maintenance/config.py` | Add anatomic bucket config |

### New Bucket Setup

1. Create T3 bucket `anatomiclocationdata`
2. Configure public read access
3. Set up credentials for oidm-maintenance

### Publish Updated Anatomic Database

```bash
# Build the updated anatomic database (user provides source file)
oidm-maintain anatomic build --source <USER_PROVIDED_SOURCE> --output anatomic_locations.duckdb

# Publish to new bucket and create manifest
oidm-maintain anatomic publish --db-path anatomic_locations.duckdb
```

**Note**: Source file path to be provided at execution time.

### Verification

```bash
# Verify findingmodel manifest (unchanged)
curl -s https://findingmodelsdata.t3.storage.dev/manifest.json | jq .

# Verify new anatomic bucket and manifest
curl -s https://anatomiclocationdata.t3.storage.dev/manifest.json | jq .
```

---

## 1. Add `__version__` via `importlib.metadata`

### Implementation Pattern

```python
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [..., "__version__"]
```

### Files to Modify

| Package | File | Current State |
|---------|------|---------------|
| findingmodel | `packages/findingmodel/src/findingmodel/__init__.py` | No `__version__` |
| anatomic-locations | `packages/anatomic-locations/src/anatomic_locations/__init__.py` | No `__version__` |
| findingmodel-ai | `packages/findingmodel-ai/src/findingmodel_ai/__init__.py` | No `__version__` |
| oidm-common | `packages/oidm-common/src/oidm_common/__init__.py` | Hardcoded "0.1.0" → replace |
| oidm-maintenance | `packages/oidm-maintenance/src/oidm_maintenance/__init__.py` | Hardcoded "0.1.0" → replace |

---

## 2. Publishing Workflow

### Taskfile Updates

Add tasks to `Taskfile.yml`:

```yaml
build:packages:
  desc: "Build all publishable packages"
  cmds:
    - rm -rf dist/
    - uv build --package oidm-common
    - uv build --package findingmodel
    - uv build --package anatomic-locations
    - uv build --package findingmodel-ai

verify:install:
  desc: "Build and verify packages install correctly in isolation"
  cmds:
    - uv run scripts/verify-install.py

publish:pypi:
  desc: "Publish packages to PyPI (in dependency order)"
  deps: [build:packages]
  cmds:
    - uv publish dist/oidm_common-*
    - uv publish dist/findingmodel-*
    - uv publish dist/anatomic_locations-*
    - uv publish dist/findingmodel_ai-*
```

### Package Publication Matrix

| Package | Publish to PyPI | Notes |
|---------|-----------------|-------|
| findingmodel | Yes | Primary user-facing package |
| anatomic-locations | Yes | User-facing package |
| findingmodel-ai | Yes | User-facing package |
| oidm-common | Yes (internal) | Dependency only |
| oidm-maintenance | No | Maintainer-only, not on PyPI |

### oidm-common Internal Marking

Update `packages/oidm-common/README.md` (or create if missing):
> "Internal infrastructure package for OIDM. Use `findingmodel` or `anatomic-locations` instead."

---

## 3. Manifest Verification

### Current Manifest Status

The manifest at `https://findingmodelsdata.t3.storage.dev/manifest.json` should have:

```json
{
  "manifest_version": "1.0",
  "databases": {
    "finding_models": {
      "version": "YYYY-MM-DD",
      "url": "https://...",
      "hash": "sha256:...",
      ...
    },
    "anatomic_locations": {
      "version": "YYYY-MM-DD",
      "url": "https://...",
      "hash": "sha256:...",
      ...
    }
  }
}
```

### Pre-Release Checklist

- [ ] Verify `finding_models` database is published and accessible
- [ ] Verify `anatomic_locations` database is published and accessible
- [ ] Both URLs in manifest are reachable
- [ ] Hashes match actual files
- [ ] Test download with fresh install

---

## 4. CHANGELOG Update

### Current State

CHANGELOG.md has `[Unreleased—presumed 0.7.0]` section with many features.

### Update Format

Convert to package-tagged format:

```markdown
## [Unreleased]

### findingmodel
#### Added
- `__version__` via importlib.metadata
- (existing unreleased features...)

### anatomic-locations
#### Added
- `__version__` via importlib.metadata

### findingmodel-ai
#### Added
- `__version__` via importlib.metadata

### oidm-common
#### Changed
- `__version__` now uses importlib.metadata (was hardcoded)

---

## findingmodel [0.7.0] - 2025-01-XX
(move current unreleased content here when releasing)
```

---

## 5. Version Bumps for Release

| Package | Current | Release Version | Notes |
|---------|---------|-----------------|-------|
| findingmodel | 0.6.1 | **1.0.0** | Breaking change (manifest/distribution) |
| anatomic-locations | 0.1.0 | 0.2.0 | New bucket, feature release |
| findingmodel-ai | 0.1.0 | 0.2.0 | First feature release |
| oidm-common | 0.1.0 | 0.2.0 | Distribution improvements |
| oidm-maintenance | 0.1.0 | 0.2.0 | Publishing improvements |

### Git Tag Format

`{package}-v{version}` (e.g., `findingmodel-v0.7.0`)

---

## 6. Implementation Steps

### Step 0: Separate manifests and publish updated anatomic database
- Update anatomic-locations config to use separate manifest URL
- Update oidm-maintenance to publish anatomic to separate manifest
- Build anatomic database from current source
- Publish to S3, creating new separate manifest
- Verify both manifests are correct

### Step 1: Add `__version__` to all packages
- Modify 5 `__init__.py` files
- Use importlib.metadata pattern
- Add to `__all__` exports

### Step 2: Update Taskfile
- Add `build:packages` task
- Add `verify:install` task
- Add `publish:pypi` task

### Step 3: Verify manifest and databases
- Check current manifest content
- Verify both databases are accessible
- Test download on clean environment

### Step 4: Update CHANGELOG
- Add package tags to unreleased section
- Document `__version__` additions

### Step 5: Bump versions
- Update each `pyproject.toml`

### Step 6: Test builds in isolation
- Run `task verify:install`
- Verifies packages build and install correctly without workspace

### Step 7: Release to PyPI
- Run `task publish:pypi`
- Create git tags
- Verify installs work

---

## 7. Verification

### After Step 1 (Version in Code)
```bash
uv run python -c "import findingmodel; print(findingmodel.__version__)"
uv run python -c "import anatomic_locations; print(anatomic_locations.__version__)"
uv run python -c "import findingmodel_ai; print(findingmodel_ai.__version__)"
uv run python -c "import oidm_common; print(oidm_common.__version__)"
```

### After Step 6 (Isolation Test)
```bash
task verify:install
# Verifies imports, CLI entry points, and database access
```

### After Step 7 (PyPI Release)
```bash
pip install findingmodel anatomic-locations findingmodel-ai
python -c "from findingmodel import DuckDBIndex; idx = DuckDBIndex(); print(len(idx))"
python -c "from anatomic_locations import AnatomicLocationIndex; idx = AnatomicLocationIndex(); print('OK')"
```

---

## Critical Files

| Purpose | File Path |
|---------|-----------|
| Version in code | `packages/*/src/*/__init__.py` (5 files) |
| Build/publish tasks | `Taskfile.yml` |
| Package versions | `packages/*/pyproject.toml` (5 files) |
| Changelog | `CHANGELOG.md` |
| oidm-common docs | `packages/oidm-common/README.md` |
| Anatomic bucket/manifest URL | `packages/anatomic-locations/src/anatomic_locations/config.py` |
| Anatomic publish | `packages/oidm-maintenance/src/oidm_maintenance/anatomic/publish.py` |
| Maintenance config | `packages/oidm-maintenance/src/oidm_maintenance/config.py` |
| S3 utils | `packages/oidm-maintenance/src/oidm_maintenance/s3.py` |
