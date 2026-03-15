# Pending: Replace mypy with ty

**Status**: Blocked — waiting on ty Pydantic plugin and type system gaps
**Created**: 2026-03-14
**Tracking issues**:
- [ty #2403 — add dedicated support for pydantic](https://github.com/astral-sh/ty/issues/2403)
- [ty #1889 — type system feature overview](https://github.com/astral-sh/ty/issues/1889)
- [pydantic-ai #3970 — track ty readiness for pydantic-ai](https://github.com/pydantic/pydantic-ai/issues/3970)

## Why ty

[ty](https://docs.astral.sh/ty/) is Astral's Python type checker (same team as ruff and uv). It's 10-100x faster than mypy, has a built-in language server, and fine-grained incremental analysis. Switching would consolidate our toolchain on Astral (ruff + uv + ty).

## Current state (ty 0.0.x beta, tested 2026-03-14)

We ran `ty check` on the findingmodel codebase. Results: **95 diagnostics** (excluding notebooks/scripts).

### Real issues ty catches that mypy misses (~17)

These are genuine bugs worth fixing regardless of which checker we use:

| Count | Category | Details |
|-------|----------|---------|
| 6 | **Null safety** | `entry.tags[0]` when `tags: list[str] \| None`; `.display` is `str \| None` passed to `str` params in test mocks |
| 3 | **TypedDict strictness** | Anthropic `{"type": "disabled"}` plain dicts don't structurally match `BetaThinkingConfigDisabledParam` |
| 3 | **Return type mismatches** | `_make_index_with_row` declares `tuple[...]` return but returns bare object; generator fixture typed as return value |
| 3 | **Invalid type expressions** | `list[pytest.param]` — `param` is a function, not a type |
| 2 | **Literal narrowing** | `str` passed where `Literal["basic", "advanced"]` expected |

### False positives — Pydantic plugin gap (~29)

These all go away once ty ships Pydantic support ([ty #2403](https://github.com/astral-sh/ty/issues/2403)):

| Count | Category | Root cause |
|-------|----------|------------|
| 20 | `_env_file` unknown argument | pydantic-settings synthesizes `__init__` params via metaclass; ty doesn't see them statically |
| 9 | `ALLOW_MODEL_REQUESTS = False` | pydantic-ai declares as `Literal[True]`; tests mutate it — ty doesn't understand the mutable class var pattern |

### False positives — type narrowing gaps (~21)

| Count | Category | Root cause |
|-------|----------|------------|
| 8 | `dict.get()` with `Never` key | After `isinstance(x, dict)`, ty narrows to `dict[Unknown, Unknown]` and `.get("key")` becomes invalid |
| 6 | `.values` on union type | After `attr.type == CHOICE`, ty doesn't narrow `ChoiceAttribute \| NumericAttribute` to `ChoiceAttribute` |
| 3 | `Agent` generic defaults | pydantic-ai `Agent[Deps, OutputType]` — ty infers `Agent[Deps, str]` due to missing TypeVar default support |
| 4 | Other narrowing | isinstance-guarded ternaries, `str.join` after isinstance check |

### Conflicting suppressions (~16)

`unused-type-ignore-comment` warnings on `# type: ignore[prop-decorator]` comments that mypy needs for `@computed_field @property` patterns. These create a conflict: removing them breaks mypy, keeping them warns in ty.

## Checklist before switching

- [ ] **ty #2403 resolved** — Pydantic `__init__` synthesis understood (eliminates 20 `_env_file` false positives)
- [ ] **TypeVar defaults supported** — `Agent[Deps, OutputType]` inferred correctly (eliminates 3 `Agent` false positives)
- [ ] **Dict narrowing fixed** — `isinstance(x, dict)` followed by `.get("key")` works (eliminates 8 false positives)
- [ ] **Union narrowing after equality** — `attr.type == CHOICE` narrows union types (eliminates 6 false positives)
- [ ] **`ALLOW_MODEL_REQUESTS` pattern** — mutable class var assignment works (eliminates 9 false positives)
- [ ] **`@computed_field @property`** — no longer needs `# type: ignore[prop-decorator]` in mypy, or ty has a way to coexist
- [ ] **pydantic-ai #3970 closed** — pydantic-ai team confirms ty readiness

## How to recheck

```bash
# Run ty on the codebase (ty is installed as a uv tool)
ty check --exclude 'notebooks/' --exclude 'scripts/'

# Compare with current mypy
task check

# Count diagnostics by rule
ty check --exclude 'notebooks/' --exclude 'scripts/' 2>&1 | grep -oE '\[[-a-z]+\]' | sort | uniq -c | sort -rn
```

## Migration plan (once blockers clear)

1. Fix the ~17 real issues ty found (worth doing now, independent of migration)
2. Create `[tool.ty]` config in `pyproject.toml` with any needed rule overrides
3. Replace `uv run mypy packages/` with `ty check` in `task check`
4. Remove mypy from dev dependencies
5. Clean up now-unnecessary `# type: ignore` comments
