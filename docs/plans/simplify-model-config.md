# Plan: Model Configuration Simplification

## Part A: Deferred Plan — Config Overhaul (after findingmodel-enrich merges)

This section documents the full config simplification to execute AFTER the findingmodel-enrich branch merges back to dev. Written now so the design is ready.

### Context

The March 2026 agent performance audit proved the tier system (small/base/full) is the wrong abstraction. Each agent needs its own model × reasoning pairing. The current system is over-engineered and has a singleton propagation bug that makes benchmarking unreliable.

The findingmodel-enrich branch rewrites `search/similar.py` (new tags: `similar_plan`, `similar_select`) and replaces the legacy `enrichment/` package with `metadata/` (`metadata_assign`). All of that work still uses `ModelTier`. Doing the config simplification before that merge would create conflicts and require re-doing the work. So we wait.

### Release Posture

- Breaking major-version change for `findingmodel-ai`
- Targets `dev` after findingmodel-enrich merge
- No compatibility shims — clean removal of tier system
- Process-scoped configuration (not injectable) — benchmarks use subprocess isolation or pre-import env vars

### What Gets Removed

- `ModelTier` type
- `default_model`, `default_model_full`, `default_model_small` fields
- `default_reasoning_small`, `default_reasoning_base`, `default_reasoning_full` fields
- `get_model(tier)` method
- `_create_model_from_string(model_string, default_tier)`
- `_tier_model_string()`, `_tier_reasoning()` helpers
- `tier_fallback` concept
- `model_tier` / `default_tier` parameters on ALL agent factories and public APIs
- Old env var names: `DEFAULT_MODEL`, `DEFAULT_MODEL_FULL`, `DEFAULT_MODEL_SMALL`, `DEFAULT_REASONING_*`

### What Replaces It

- **`agents.toml`** — separate file for per-agent model+reasoning fallback chains (not mixed with model metadata in `supported_models.toml`)
- **`FMAI_MODEL__<tag>` / `FMAI_REASONING__<tag>`** — env var overrides with `FMAI_` prefix for non-secret vars
- **Secrets stay in `.env`** with standard names (OPENAI_API_KEY etc.), no prefix
- **Secret rejection in TOML** — actively scan `agents.toml` for API key names, raise on violation
- **Two settings classes** — `SecretSettings` (from .env, no prefix) and `FindingModelAIConfig` (FMAI_ prefix)
- **`get_agent_model(agent_tag)` with no tier parameter** — resolution: env override → TOML chain → ConfigurationError

### Agents In Scope (ALL — post-merge)

The final agent tag list still depends on the exact merged state, but the enrich worktree is much clearer now than when this plan was first written. The current direction is a single `metadata_assign` agent under `findingmodel_ai.metadata`, with the legacy `findingmodel_ai.enrichment` package removed. Legacy `similar_search`/`similar_assess`/`similar_term_gen` tags are likewise being replaced by `similar_plan`/`similar_select`. Slice 9 public-surface migration is still in progress there, so recheck the final merged inventory before executing this simplification.

**At simplification time, inventory the actual merged state.** The list below is the expected post-merge state based on the enrichment rewrite plan's intent:

Search domain:
- `ontology_search`, `ontology_match` — `search/ontology.py`
- `anatomic_search`, `anatomic_select` — `search/anatomic.py`
- `similar_plan`, `similar_select` — `search/similar.py` (new tags from enrich branch, replacing legacy `similar_search`/`similar_assess`/`similar_term_gen`)

Authoring domain:
- `describe_finding`, `describe_details` — `authoring/description.py`
- `edit_instructions`, `edit_markdown` — `authoring/editor.py`
- `import_markdown` — `authoring/markdown_in.py`

Metadata-assignment domain:
- `metadata_assign` — `metadata/assignment.py`

**Legacy tag cleanup**: Any tags that exist in the merged code but are not part of the new metadata-assignment/similar-search design should be removed in this simplification pass, not left as dead config. This includes old enrichment tags (`enrich_classify`, `enrich_unified`, `enrich_research`) if they somehow remain after merge, and old similar-search tags (`similar_search`, `similar_assess`, `similar_term_gen`).

### Key Design Decisions (already made)

- **Pure config** — callers never specify models at call sites
- **Process-scoped** — config is fixed at import time; runtime singleton replacement is not supported
- **Injectable config deferred** — passing config/model objects through function signatures is a separate future refactor

### Reference Documents

- `docs/agent-performance-audit-2026-03.md` — audit data backing per-agent model choices
- `docs/plans/per-agent-model-config.md` — earlier implementation plan (partially executed, to be superseded)
- `/Users/talkasab/repos/findingmodel-enrich/docs/canonical-structured-metadata-and-enrichment-rewrite.md` — enrich branch plan

---

## Part B: Cleanup Work — Do Now

Work that is safe and valuable to do right now, independent of the enrich merge.

### B1. Fix Logfire configuration bug

**Problem**: `logfire.configure()` in `evals/__init__.py` relies on `os.environ` for `LOGFIRE_TOKEN`, but `.env` is only loaded by pydantic-settings — not exported to the process environment. Logfire never actually sends traces.

**Fix**:
- `config.py` already has `logfire_token` field and `configure_logfire()` method (added this session)
- `evals/__init__.py` already updated to use `settings.configure_logfire()` (added this session)
- Verify it works: run benchmark, check Logfire MCP for traces

**Files**: `config.py` (already done), `evals/__init__.py` (already done)

### B2. Add gpt-5.4-mini and gpt-5.4-nano to model registry

**Problem**: These models were released 2026-03-17 and added to `supported_models.toml` but never properly benchmarked (the benchmark ran the wrong models due to the singleton bug).

**Fix**:
- Models are already in `supported_models.toml` (added this session)
- Write a proper benchmark script that sets env vars BEFORE import
- Run benchmarks with Logfire observability
- Verify via Logfire MCP that the correct model names appear in spans
- Update `agents.toml` / `supported_models.toml` agent chains if the new models outperform current choices

**Files**: `scripts/benchmark_models.py`, `data/supported_models.toml`

### B3. Fix benchmark script to work with process-scoped config

**Problem**: The current `scripts/benchmark_models.py` tries to replace the module-level singleton at runtime, which doesn't propagate. The old `scripts/agent_audit.py` has the same problem.

**Fix**: Rewrite `scripts/benchmark_models.py` so each model config runs as a subprocess (or set env vars before any findingmodel_ai imports). Use Logfire MCP to analyze results instead of parsing stdout.

**Files**: `scripts/benchmark_models.py`

### B4. Commit and document

- Commit the Logfire fix, new model entries, and benchmark script
- Update `docs/agent-performance-audit-2026-03.md` with corrected gpt-5.4-mini/nano results (once we have real data)
- Update CHANGELOG

---

## Execution Order for Part B

1. Verify Logfire fix works (run a single agent call, check Logfire MCP for trace)
2. Rewrite benchmark script for subprocess-based model switching
3. Run gpt-5.4-mini/nano benchmarks with Logfire verification
4. Analyze results via Logfire MCP
5. Update audit doc and TOML agent chains if warranted
6. Update docs and CHANGELOG
7. Commit
