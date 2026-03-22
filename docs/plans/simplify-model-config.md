# Plan: Merge Enrichment Branch + Model Config Simplification

## Status: ALL STEPS COMPLETE.

## Context

The `feature/enrichment` branch has been merged onto `dev`. It implemented most of what our model config simplification plan called for:

- Per-agent TOML fallback chains in `supported_models.toml`
- `FallbackModel` wrapping via `resolve_agent_config()`
- New agent tags: `metadata_assign`, `similar_plan`, `similar_select`
- Removed old enrichment package (`unified.py`, `agentic.py`)
- GPT-5.4-mini/nano as defaults, Haiku `reasoning=none`
- `observability.py` module for Logfire setup
- `disable_send_to_logfire` config field
- New `metadata/`, `facets.py`, rewritten `similar.py`, `browse()`, `related_models()`

**What the enrichment branch still has that we want to remove:**
- `ModelTier` type
- `default_model`, `default_model_full`, `default_model_small` fields
- `default_reasoning_small`, `default_reasoning_base`, `default_reasoning_full` fields
- `model_tier` / `default_tier` parameters on agent factories and public APIs
- Unprefixed env var names (`DEFAULT_MODEL`, `AGENT_MODEL_OVERRIDES__`, etc.)
- `get_model(tier)` method
- `_create_model_from_string(model_string, default_tier)` with tier parameter
- `_tier_model_string()`, `_tier_reasoning()` helpers
- `tier_fallback` in TOML entries

**What we still want to add:**
- `FMAI_` prefix for non-secret env vars
- `agents.toml` as separate file (agent config split from model metadata)
- Secret rejection in TOML
- Two settings classes (`SecretSettings` + `FindingModelAIConfig`)
- `ConfigurationError` when no models available (instead of silent tier fallback)

## Decision: Merge first, then simplify

The merge is clean and brings major new functionality. Simplify the config system as a follow-up on the merged branch, not as a pre-merge gate.

## Step 0: Update repo-located planning docs ✅ DONE

Copy this plan to `docs/plans/simplify-model-config.md` (replacing the previous version which is now outdated — most of what it planned has been done by the enrichment branch). This is the executable plan going forward.

## Step 1: Merge the enrichment branch ✅ DONE

```
git merge feature/enrichment
```

Fast-forward, no conflicts. Then verify:
- `task check` passes
- `task test` passes
- Quick smoke test of a benchmark run with Logfire verification

## Step 1.5: Assess and update model defaults for new/updated agents ✅ DONE

The enrichment branch set agent defaults before our Logfire-verified benchmarks existed. Several defaults don't reflect current data:

| Agent | Current Primary | Issue | Recommended |
|-------|----------------|-------|-------------|
| `ontology_match` | gemini-3.1-pro/low (14.2s) | gpt-5.4-mini/low is faster (8.5s) | **gpt-5.4-mini/low** as primary |
| `anatomic_select` | gpt-5.4-mini/medium | `medium` reasoning not benchmarked; `low` was 5.8s | Benchmark `low` vs `medium`, keep whichever wins |
| `metadata_assign` | gemini-3-flash/low | New agent, no benchmarks. Classification task. | Benchmark with gpt-5.4-mini and gpt-5.4-nano |
| `similar_plan` | gemini-3-flash/minimal | New agent, generative task. Reasonable default. | Benchmark to verify |
| `similar_select` | gemini-3.1-flash-lite/medium | New agent, classification task. | Benchmark to verify |

**Action**: After merge, run the benchmark script on the new agents (`metadata_assign`, `similar_plan`, `similar_select`) and update defaults based on verified data. Also update `ontology_match` primary to gpt-5.4-mini/low.

### Benchmarking Plan for New Agents

#### `metadata_assign` (highest priority — new core pipeline agent)

**What it does**: Narrow classifier that assigns structured metadata (body regions, subspecialties, etiologies, entity type, modalities, time course, age profile, sex specificity, ontology/anatomic decisions) from a FindingModelFull.

**Inputs**: Real FindingModelFull JSON from `packages/findingmodel/tests/data/defs/` (pulmonary_embolism, abdominal_aortic_aneurysm, breast_density). The full `assign_metadata()` pipeline includes ontology search, anatomic search, and the classifier agent — benchmark both the isolated classifier and the full pipeline.

**Current default**: `gemini-3-flash/low` — likely suboptimal for a classification task with rich structured output.

**Configs to test**:
| Model | Reasoning | Hypothesis |
|-------|-----------|-----------|
| gpt-5.4-nano | none | Budget floor — is nano enough for metadata classification? |
| gpt-5.4-nano | low | Budget with light reasoning |
| gpt-5.4-mini | none | Mid-tier, no reasoning overhead |
| gpt-5.4-mini | low | Mid-tier with reasoning — likely sweet spot based on ontology_match data |
| gemini-3-flash | low | Current default — baseline |
| haiku-4-5 | none | Anthropic budget option (no thinking!) |

#### `similar_plan` (generative — search term + metadata hypothesis generation)

**What it does**: Generates 2-5 search terms and metadata hypotheses (body regions, modalities, entity type, subspecialties) for a proposed finding.

**Inputs**: Finding name + optional description/synonyms. Use same 5 test findings as existing benchmarks.

**Current default**: `gemini-3-flash/minimal` — reasonable for a generative task.

**Configs to test**:
| Model | Reasoning | Hypothesis |
|-------|-----------|-----------|
| gpt-5.4-nano | none | Fastest option — ontology_search data suggests nano is strong here |
| gpt-5.4-nano | low | With light reasoning |
| gpt-5.4-mini | none | More capable, no overhead |
| gemini-3-flash | minimal | Current default — baseline |

#### `similar_select` (classification — match selection from candidates)

**What it does**: Given a proposed finding and a candidate pool, selects 0-3 matches with rejection taxonomy. Requires medical judgment about concept relationships.

**Inputs**: Need to pre-fetch candidate pools from `find_similar_models()` Phase 3, then feed to the isolated selection agent. Use 3 test findings.

**Current default**: `gemini-3.1-flash-lite/medium` — budget classification.

**Configs to test**:
| Model | Reasoning | Hypothesis |
|-------|-----------|-----------|
| gpt-5.4-mini | low | Strong classification model — likely winner based on ontology_match/anatomic_select data |
| gpt-5.4-mini | medium | With more reasoning |
| gpt-5.4-nano | medium | Budget with reasoning |
| flash-lite | medium | Current default — baseline |
| haiku-4-5 | none | Anthropic option (no thinking) |

#### `ontology_match` (already benchmarked — just needs config update)

Verified data shows gpt-5.4-mini/low (8.5s) beats current primary gemini-3.1-pro/low (14.2s). Update the TOML chain to put gpt-5.4-mini first. No new benchmarks needed.

#### Implementation

Add `metadata_assign`, `similar_plan`, and `similar_select` to `scripts/benchmark_models.py` as new agent runners in `run_one()`. Add a `new-agents` comparison set. Run with Logfire verification, analyze via Logfire MCP, update TOML defaults based on results.

## Step 2: Post-merge tier removal ✅ DONE

Tier system fully removed. What was done:

### What was done

- Removed `ModelTier` type, tier fields, `get_model()`, `model_tier` / `default_tier` parameters from all agent factories, public APIs, CLI, evals, and tests (27 files changed)
- `resolve_agent_config()` now raises `ConfigurationError` when no models available (no tier fallback)
- `validate_default_model_keys()` validates env override usability
- Updated all error messages, READMEs, evals README, and configuration.md
- `tier_fallback` removed from TOML entries
- CHANGELOG updated with breaking change entry

### What was deferred (optional polish, can be done anytime)

- `FMAI_` env prefix for non-secret vars (currently still uses `AGENT_MODEL_OVERRIDES__` / `AGENT_REASONING_OVERRIDES__`)
- Split agent config from `supported_models.toml` into separate `agents.toml`
- Secret rejection in TOML
- Two-class settings split (`SecretSettings` + `FindingModelAIConfig`)

## Step 3: Verify and document ✅ DONE

- `task check` + `task test` pass
- Benchmark a few agents with Logfire to verify model routing still works
- Update all docs
- Migration notes in CHANGELOG for removed env vars and `model_tier` parameters

## Step 4: Final documentation review ✅ DONE

- Update `docs/plans/simplify-model-config.md` to mark plan as complete
- Review `docs/configuration.md` matches final shipped behavior
- Review `docs/agent-performance-audit-2026-03.md` for any stale tier references
- Update `CHANGELOG.md` with breaking changes and migration notes
- Review `CLAUDE.md` for stale tier references

## Verification

1. After merge: `task check` + `task test`
2. After tier removal: `task check` + `task test`
3. `grep -r ModelTier packages/` returns nothing
4. `grep -r default_tier packages/` returns nothing
5. `grep -r model_tier packages/` returns nothing (except comments/docs)
6. Benchmark run with Logfire confirms model routing works without tiers
7. All docs consistent with final state
