# Phase 3.9: Dependency Cleanup

**Status:** ✅ COMPLETE

**Goal:** Clean up package dependencies following the best practice that **each package declares all dependencies it directly imports**.

## Design Principle

**Best Practice:** If your code has `import foo` or `from foo import bar`, then `foo` must be in your `dependencies`.

This ensures:
1. Packages work correctly when published independently
2. Clear documentation of what each package actually requires
3. Protection from breaking changes if upstream packages modify their dependencies

**oidm-common provides shared infrastructure** but consumer packages still declare dependencies they directly import, even if oidm-common also provides them. The dependency resolver deduplicates versions.

---

## Audit Method

For each package, grep for direct imports and ensure they're declared:

```bash
grep -rh "^import \|^from " packages/<pkg>/src/**/*.py | \
  grep -v "from <pkg>" | grep -v "from \." | sort -u
```

---

## Final Dependencies

### anatomic-locations

Direct imports: `click`, `duckdb`, `loguru`, `pydantic_settings`, `rich`, `oidm_common`

```toml
dependencies = [
    "oidm-common",
    "click>=8.0",
    "duckdb>=1.0",
    "loguru>=0.7",
    "pydantic-settings>=2.9",
    "rich>=13.0",
]
```

### findingmodel

Direct imports: `click`, `duckdb`, `httpx`, `jinja2`, `loguru`, `mcp`, `openai`, `pydantic`, `pydantic_ai`, `pydantic_settings`, `rich`, `tavily`, `oidm_common`

```toml
dependencies = [
    "oidm-common",
    "click>=8.1.8",
    "duckdb>=1.0",
    "httpx>=0.27",
    "jinja2>=3.1.6",
    "loguru>=0.7",
    "mcp>=1.0.0",
    "openai>=1.76",
    "pydantic[email]>=2.0",
    "pydantic-ai-slim[openai,tavily,anthropic,google]>=0.3.2",
    "pydantic-settings>=2.9",
    "rich>=13.9.4",
    "tavily-python>=0.6.0",
]
```

### oidm-maintenance

Direct imports: `boto3`, `click`, `duckdb`, `httpx`, `loguru`, `openai`, `pooch`, `pydantic`, `pydantic_settings`, `rich`, `oidm_common`, `anatomic_locations`, `findingmodel`

```toml
dependencies = [
    "oidm-common",
    "anatomic-locations",
    "findingmodel",
    "boto3>=1.40",
    "boto3-stubs[s3]>=1.42.25",
    "click>=8.0",
    "duckdb>=1.0",
    "httpx>=0.27",
    "loguru>=0.7",
    "openai>=1.0",
    "pooch>=1.8",
    "pydantic>=2.0",
    "pydantic-settings>=2.9",
    "rich>=13.0",
]
```

### oidm-common

Provides shared infrastructure. Its dependencies are whatever IT directly imports:

```toml
dependencies = [
    "duckdb>=1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.9",
    "httpx>=0.27",
    "platformdirs>=4.0",
    "loguru>=0.7",
    "pooch>=1.8",
]
```

---

## Removed (not directly imported)

- **findingmodel**: `prompt-toolkit`, `boto3`, `boto3-stubs` (moved to oidm-maintenance)
- **anatomic-locations**: `build` optional dependency group (cruft from before oidm-maintenance existed)

---

## Verification Checklist

- [x] Each package declares all dependencies it directly imports
- [x] No package relies on transitive dependencies for code it imports
- [x] Unused dependencies removed (prompt-toolkit, boto3 from findingmodel)
- [x] All tests pass
- [x] All imports work
- [x] ruff check passes

---

## References

- [Turborepo: Managing Dependencies](https://turborepo.dev/docs/crafting-your-repository/managing-dependencies)
- [FOSSA: Direct vs Transitive Dependencies](https://fossa.com/blog/direct-dependencies-vs-transitive-dependencies/)
- [Arnica: Direct vs Transitive Dependencies](https://www.arnica.io/blog/direct-vs-transitive-dependencies-navigating-package-management-in-software-composition-analysis-sca)
