---
description: Orchestrate a complete release with interactive confirmations at every write step
argument-hint: [version] (optional - extracts from pyproject.toml if not provided)
---

# Release Command

**CRITICAL**: You MUST use the `date` command to get the current date. DO NOT assume you know the date.

## Process Overview

Fail-fast ordering: PyPI publish happens BEFORE git tag/GitHub release to avoid inconsistent state.

1. **Pre-flight checks** - Verify clean git, on dev branch (assumes tests/checks already run)
2. **Version & CHANGELOG** - Extract version, update CHANGELOG with current date
3. **Commit to dev** (local only, not pushed)
4. **Merge to main** (local only, not pushed)
5. **Build packages** - Create wheel and tarball
6. **Verify artifacts** - Show checksums and sizes
7. **Publish to PyPI** - Most likely to fail, do this FIRST before any git operations
8. **Create git tag** (local only)
9. **Push main + tag** - Now safe because PyPI succeeded
10. **GitHub release** - Use the pushed tag, include CHANGELOG notes and artifacts
11. **Cleanup** - Return to dev, merge main back

## Instructions for Claude

**CRITICAL REMINDERS**:
- 🚨 **CHECK THE DATE** - Run `date +%Y-%m-%d` command, don't assume!
- Get user confirmation before EVERY write operation
- Show clear summaries before confirmations
- If any step fails, provide rollback instructions

### Step 1: Pre-flight Checks

Verify clean state (assumes tests/checks already run):

```bash
date +%Y-%m-%d                           # Get current date
git branch --show-current                # Must be 'dev'
git status --porcelain                   # Must be clean
git fetch origin
git rev-list HEAD...origin/dev --count   # Must be 0 (up to date)
```

If not clean or not on dev, stop with instructions to fix.

### Step 2: Version & CHANGELOG

Extract version from pyproject.toml. If `$1` provided, verify it matches.

Get current date: `date +%Y-%m-%d`

Read CHANGELOG.md, find section for this version. Show user:
```
Current version: X.Y.Z
CHANGELOG preview:
## [X.Y.Z]
...

Will update to: ## [X.Y.Z] - YYYY-MM-DD

Proceed?
```

If confirmed, update CHANGELOG.md. Extract full release notes (from `## [X.Y.Z]` until next `## [`).

### Step 3: Commit to Dev (Local)

Show: `git status --short`

Ask: "Commit version & CHANGELOG to dev (local only, not pushed)?"

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Prepare release vX.Y.Z"
```

**Note**: Not pushing yet - will push after PyPI succeeds.

### Step 4: Merge to Main (Local)

Show:
```
About to (LOCAL ONLY):
1. Switch to main
2. Pull latest origin/main
3. Merge dev (--no-ff)

NOT pushing yet. Proceed?
```

```bash
git checkout main
git pull origin main
git merge --no-ff dev -m "Merge dev for release vX.Y.Z"
```

### Step 5: Build Packages

```bash
rm -rf dist/
uv build
ls -lh dist/
```

Show: "Built packages in dist/"

### Step 6: Verify Artifacts

Calculate checksums and show summary:
```bash
sha256sum dist/findingmodel-X.Y.Z-py3-none-any.whl
sha256sum dist/findingmodel-X.Y.Z.tar.gz
```

Show user:
```
Artifacts ready for release:
- findingmodel-X.Y.Z-py3-none-any.whl (SIZE bytes, SHA256: ...)
- findingmodel-X.Y.Z.tar.gz (SIZE bytes, SHA256: ...)

Artifacts look correct?
```

### Step 7: Publish to PyPI (FIRST!)

**CRITICAL**: Do this BEFORE git tag/GitHub release. If this fails, we can rollback local commits.

Show:
```
About to publish to PyPI.
Requires UV_PUBLISH_TOKEN from ../.env

This is the FIRST public action. If it fails, nothing is pushed to git yet.

Proceed?
```

```bash
# Source token
export $(grep UV_PUBLISH_TOKEN ../.env | xargs)
if [ -z "$UV_PUBLISH_TOKEN" ]; then
  echo "ERROR: UV_PUBLISH_TOKEN not found in ../.env"
  exit 1
fi
uv publish
```

**On failure**: Tell user PyPI publish failed. Nothing pushed yet. Can `git reset --hard` to rollback.

### Step 8: Create Git Tag (Local)

PyPI succeeded! Now safe to tag.

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

### Step 9: Push Main + Tag

```bash
git push origin main
git push origin vX.Y.Z
git push origin dev  # Push the prep commit too
```

### Step 10: Create GitHub Release

Show:
```
Creating GitHub release vX.Y.Z with CHANGELOG notes and artifacts.

Proceed?
```

Create temp file with CHANGELOG content, then:
```bash
gh release create vX.Y.Z \
  --title "Release vX.Y.Z" \
  --notes-file /tmp/release_notes.md \
  dist/findingmodel-X.Y.Z-py3-none-any.whl \
  dist/findingmodel-X.Y.Z.tar.gz
```

### Step 11: Cleanup - Return to Dev

```bash
git checkout dev
git merge main
git push origin dev
```

### Step 12: Summary

```
✅ Release vX.Y.Z Complete!

🔗 Links:
   - PyPI: https://pypi.org/project/findingmodel/X.Y.Z/
   - GitHub: https://github.com/talkasab/findingmodel/releases/tag/vX.Y.Z

📦 Artifacts published:
   - findingmodel-X.Y.Z-py3-none-any.whl (SHA256: ...)
   - findingmodel-X.Y.Z.tar.gz (SHA256: ...)

🎉 Release complete!
```

## Error Recovery

**PyPI publish failure** (Step 7 - most critical):
- Nothing pushed to git yet
- Local commits on dev and main
- Recovery: `git checkout dev && git reset --hard origin/dev && git checkout main && git reset --hard origin/main`

**GitHub release failure** (Step 10):
- PyPI already published (can't undo)
- Tag pushed
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
