# Task Status Overview

**Last Updated:** 2025-12-31

Quick reference for active tasks and project status. For completed tasks, see [done/README.md](done/README.md).

---

## Project Health

- ✅ **All tests passing**
- ✅ **Code quality**: Clean linting, no warnings
- ✅ **Core features**: Complete and stable

---

## Active Tasks

### High Priority

#### [refactoring/02-api-cleanup.md](refactoring/02-api-cleanup.md)
**Status**: Ready to start
**Effort**: 3-4 days

Clean up 8+ duplicate function aliases in tools module. Backward compatible deprecation strategy.

**Why now**: Creating user confusion, technical debt accumulating before new features.

**Key Actions**:
- Create deprecation infrastructure
- Mark old aliases as deprecated with warnings
- Update all internal usage to new names
- Migration guide for users

---

#### [facets-implementation-plan.md](facets-implementation-plan.md)
**Status**: Blocked - needs 4 design decisions
**Effort**: 2-3 weeks once decided

Implement 8-facet classification system for finding models.

**Open Questions** (need your input):
1. Scope: All 8 facets at once, or phased implementation?
2. Required vs optional: Enforce required facets for new models?
3. Markdown format: Frontmatter, dedicated section, or inline?
4. Search performance: Add indexes upfront or measure first?

**Spec**: See [findingmodel-facets.md](findingmodel-facets.md)

---

### Medium Priority

#### [refactoring/01-index-decomposition.md](refactoring/01-index-decomposition.md)
**Status**: Phase 2 (architectural refactoring)
**Prerequisites**: DuckDB migration complete ✅

Decompose monolithic Index class (1,300+ lines) into 5 focused components:
- IndexRepository (database operations)
- ModelValidator (validation logic)
- FileManager (file I/O)
- SearchEngine (search operations)
- Index (facade)

**Benefits**: 60% complexity reduction, better testability, parallel development

**Risk**: Medium - core functionality used throughout codebase

---

#### [database_distribution_enhancements.md](database_distribution_enhancements.md)
**Status**: Deferred to v0.5.1+

Database distribution workflow improvements:
1. CLI command for manifest generation
2. S3 upload integration (stream databases, auto-generate manifests)
3. Version history and pinning in manifest.json
4. Search type parameter for graceful degradation
5. Remote DuckDB access via httpfs

**Priority**: Low - nice to have, not blocking

---

### Low Priority

#### [refactoring/03-validation-framework.md](refactoring/03-validation-framework.md)
**Status**: Planning phase

Extract validation logic into reusable framework.

---

#### [refactoring/04-circular-dependencies.md](refactoring/04-circular-dependencies.md)
**Status**: Planning phase

Address circular import issues in tools module.

---

#### [refactoring/05-performance-optimizations.md](refactoring/05-performance-optimizations.md)
**Status**: Planning phase

Performance improvements for search and batch operations.

---

## Pending Technical Debt

See [pending-fixes.md](pending-fixes.md) for details:

1. **Anatomic location rebuild optimization** (Medium priority)
   - Preserve embeddings for unchanged entries
   - Use hash-based diffing like Index does

2. **Search type parameter** (Low priority, deferred to v0.5.1+)

3. **Remote DuckDB access** (Low priority, deferred to v0.5.1+)

---

## Recently Completed (Last 3 Months)

- ✅ **Anatomic Location Models** - Rich hierarchy navigation, laterality variants, semantic search
- ✅ **Bulk-Load Migration** - 1000x faster database creation via `read_json()` pattern
- ✅ **DuckDB Index Migration** - Complete MongoDB → DuckDB backend replacement
- ✅ **Anatomic Location Cleanup** - CLI commands, common utilities, no hardcoded config
- ✅ **Anthropic Support** - Multi-provider AI with tier-based model selection
- ✅ **Agent Eval Suites** - 5 comprehensive eval suites for AI agents

See [done/README.md](done/README.md) for full history.

---

## Recommended Next Steps

1. **Review facets design questions** - Unblock implementation
2. **Start API cleanup** - Low risk, high value, clears debt
3. **Consider index decomposition** - After API cleanup, if ready for larger refactor

---

## Notes

- Task files marked ✅ have been completed but not yet moved to `done/`
- Task files in `done/` are kept for historical reference
- Always check git history for detailed implementation notes
- Branch strategy: feature branches for major work, direct to main for small fixes
