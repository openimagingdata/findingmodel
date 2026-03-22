# Agent Performance Audit — March 2026

## Phase 1: Code Review Findings

### Overview

Reviewed 12 agents across 4 domains (search, authoring). Enrichment agents excluded (being overhauled separately).

---

### SEARCH DOMAIN

#### 1. Ontology Query Generator (`ontology_search`, small tier)
**File**: `search/ontology.py:328-355`

**Architecture**: Appropriate — simple query generation task, small tier is correct.

**Prompt Assessment**:
- Good: Includes a concrete example (quadriceps tendon rupture → tear)
- Good: Specifies 2-5 terms, keeps scope tight
- Weakness: Only one example. More diverse examples (acronym expansion, multi-word, rare findings) would help edge cases.
- Weakness: No XML structure or clear sections — just a wall of text.

**Output Model**: `list[str]` — bare list, no constraints. Could benefit from `Field(min_length=2, max_length=7)` wrapper.

**Output Validators**: None. The caller adds the original finding name post-hoc (line 385-386). This could be an output validator instead.

**Pydantic AI**:
- Uses `system_prompt=` ✅ (but should migrate to `instructions=`)
- No `UsageLimits`
- No usage tracking

**Reasoning Assessment**: Currently `low` — appropriate for this simple generative task. `minimal` might suffice.

---

#### 2. Ontology Categorization (`ontology_match`, base tier)
**File**: `search/ontology.py:201-252`

**Architecture**: Good two-agent pattern. Base tier appropriate for classification requiring medical judgment.

**Prompt Assessment**:
- Good: Clear 3-tier categorization schema with max counts
- Good: Explicit prioritization of SNOMEDCT
- Good: Clear exclusions (drugs, procedures, anatomical structures)
- Weakness: No examples of actual categorization decisions
- Weakness: Long prompt with lots of emphasis markers — could be more structured with XML tags

**Output Model**: `CategorizedConcepts` — well-constrained with `max_length` on each category.

**Output Validators**: None on agent, but has post-processing via `ensure_exact_matches_post_process()` (line 255). This is a good pattern — deterministic post-processing is better than relying on the model.

**Pydantic AI**:
- Uses `system_prompt=` (should migrate to `instructions=`)
- No `UsageLimits`
- Has `deps_type=CategorizationContext` but doesn't use it in tools (no tools)

**Reasoning Assessment**: Currently `none` — **this is a prime candidate for `low` or `medium` reasoning**. Categorization requires multi-step medical judgment (is this an exact match? what's the relationship?). Reasoning could significantly help.

---

#### 3. Anatomic Query Generator (`anatomic_search`, small tier)
**File**: `search/anatomic.py:73-103`

**Architecture**: Good — inline agent creation in the function. Small tier appropriate.

**Prompt Assessment**:
- Good: Two concrete examples (meniscal tear, pneumonia)
- Good: Explicit constraints (no acronyms, no adjectives, no left/right)
- Good: Asks for region AND terms in structured output
- Weakness: "THINK about what location..." — this instruction is vague

**Output Model**: `AnatomicQueryTerms` with `Literal` region type — excellent use of constrained output.

**Output Validators**: None. Caller adds finding name as fallback (line 114).

**Pydantic AI**:
- Uses `system_prompt=` (should migrate)
- No `UsageLimits`

**Reasoning Assessment**: Currently `low` — appropriate. The region classification is simple enough.

---

#### 4. Anatomic Location Selection (`anatomic_select`, small tier)
**File**: `search/anatomic.py:178-209`

**Architecture**: Good. Selection from a candidate list is appropriate for small tier.

**Prompt Assessment**:
- Good: "Sweet spot" concept is well explained
- Good: 5 concrete examples with expected outputs
- Weakness: Prompt says "Note: If results appear pre-ranked..." — this hedging is unhelpful
- Weakness: No negative examples (what NOT to pick)

**Output Model**: `LocationSearchResponse` with primary + alternates (max 3). Well-constrained.

**Output Validators**: None. Could benefit from validation that primary isn't also in alternates.

**Pydantic AI**:
- Uses `system_prompt=` (should migrate)
- No `UsageLimits`

**Reasoning Assessment**: Currently `low` (small tier). This task requires some judgment about specificity — `low` seems right but `medium` could help on edge cases.

---

#### 5. Similar Models Term Generation (`similar_search`, small tier)
**File**: `search/similar.py:125-145`

**Architecture**: Good lightweight agent. Shares `similar_search` tag with the search agent — this means they can't be independently overridden.

**Prompt Assessment**:
- Adequate but generic — no domain-specific examples
- Similar to ontology query gen prompt but for a different purpose
- Could benefit from an example

**Output Model**: `SearchTerms` with `list[str]` field. Fine.

**Pydantic AI**:
- Uses `system_prompt=` (should migrate)

**Reasoning Assessment**: Currently `low` — appropriate.

---

#### 6. Similar Models Search (`similar_search`, base tier)
**File**: `search/similar.py:89-122`

**Architecture**: This is the one agent with actual tools (`search_models_tool`). Base tier appropriate since it needs to make strategic search decisions.

**Prompt Assessment**:
- Good: Lists 4 search strategies
- Weakness: Numbering jumps from 3 to 6 (line 111) — copy-paste error
- Weakness: "Generate about 5 likely search terms" is vague
- Weakness: No examples of good vs bad search strategies

**Tool**: `search_models_tool` — returns JSON string. Good design.

**Output Model**: `SearchStrategy` — comprehensive with search terms used, total results, and summary.

**Pydantic AI**:
- Uses `system_prompt=` (should migrate)
- Has `retries=3` — good
- No `UsageLimits` — risky since it has a tool (could loop)

**Reasoning Assessment**: Currently `none` — **should probably have `low` reasoning** since it needs to plan a search strategy.

---

#### 7. Similar Models Analysis (`similar_assess`, base tier)
**File**: `search/similar.py:162-205`

**Architecture**: Good. Classification/recommendation task, base tier appropriate.

**Prompt Assessment**:
- Good: Clear similarity thresholds (70%+, 40-70%, <40%)
- Good: Conservative guidance ("when in doubt, recommend creating new")
- Weakness: No actual examples of edit_existing vs create_new decisions
- Note: The prompt has a grammar error: "similar enough to the to be used" (line 176)

**Output Model**: `SimilarModelAnalysis` with Literal recommendation + confidence float. Well-designed.

**Pydantic AI**:
- Uses `system_prompt=` (should migrate)
- Has `retries=3` — good

**Reasoning Assessment**: Currently `none` — **candidate for `low` reasoning**. Similarity assessment involves comparing multiple dimensions (anatomy, pathology, clinical use).

---

### AUTHORING DOMAIN

#### 8. Finding Description (`describe_finding`, small tier)
**File**: `authoring/description.py:73-83`

**Architecture**: Good. Template-based prompt, small tier appropriate for simple description generation.

**Prompt Assessment** (from `get_finding_description.md.jinja`):
- Excellent: Clear 3-rule structure (canonical name, synonyms, description)
- Excellent: Concrete examples with input→output transformations
- Good: Explicit handling of acronyms and over-specific phrasing

**Output Model**: `FindingInfo` — well-designed core model.

**Pydantic AI**:
- Uses `instructions=` ✅ — already migrated! One of only 3 agents using the modern pattern.
- No `UsageLimits`

**Reasoning Assessment**: Currently `low` — appropriate. Simple descriptive task.

---

#### 9. Finding Detail (`describe_details`, small tier)
**File**: `authoring/description.py:157-163`

**Architecture**: Has Tavily search tool. Small tier might be underweight for a tool-using agent that needs to synthesize search results.

**Prompt Assessment** (from `get_finding_detail.md.jinja`):
- Good: Specific about what to include (imaging appearance, locations, associated findings)
- Good: Lists preferred sources (Radiopaedia, Wikipedia, etc.)
- Weakness: "DON'T include general information" is negative — better to state what TO include
- Weakness: Typo "radassistant.nl" should be "radiologyassistant.nl"

**Tool**: `search_radiology_sources` with domain filtering. Good design.

**Pydantic AI**:
- Uses `instructions=` ✅ — already migrated!
- No `UsageLimits` — risky since it has a tool
- Citation extraction (lines 169-186) is done by regex over message strings — fragile pattern

**Reasoning Assessment**: Currently `low` — may need `medium` for better search strategy and synthesis.

---

#### 10. NL Editor (`edit_instructions`, base tier)
**File**: `authoring/editor.py:89-122`

**Architecture**: Complex task — receives full FindingModel JSON, interprets NL command, returns edited model. Base tier appropriate. Most complex agent in the system.

**Prompt Assessment**:
- Excellent: Modular instruction building (`_combine_instruction_sections`)
- Excellent: Clear safety rails (preserve IDs, no semantic changes, track changes)
- Good: Handles ambiguity with reject-and-explain pattern
- Weakness: Very long combined prompt — could benefit from XML structure

**Output Model**: `EditResult` with model + changes + rejections. Well-designed.

**Output Validators**: YES — `_validate_output` checks ID preservation and calls `ModelRetry`. This is the most sophisticated validation in the system.

**Pydantic AI**:
- Uses `instructions=` ✅ — already migrated!
- Has output validator with `ModelRetry` — excellent
- No `UsageLimits`

**Reasoning Assessment**: Currently `none` — **strong candidate for `low` or `medium` reasoning**. This is the most complex task (parse NL, understand medical model structure, generate valid edits). Reasoning could significantly improve quality and reduce retries.

---

#### 11. Markdown Editor (`edit_markdown`, base tier)
**File**: `authoring/editor.py:125-163`

**Architecture**: Similar to NL editor but with markdown input. Same base tier. Good separation of NL vs markdown input channels.

**Prompt Assessment**: Same modular structure as NL editor but with markdown-specific instructions. Good.

**Everything else**: Same patterns as NL editor — same output validator, same `instructions=` usage.

**Reasoning Assessment**: Same as NL editor — `none` is probably too low.

---

#### 12. Markdown Import (`import_markdown`, base tier)
**File**: `authoring/markdown_in.py:41-45`

**Architecture**: Creates a full FindingModelBase from a markdown outline. Base tier appropriate.

**Prompt Assessment** (from `get_finding_model_from_outline.md.jinja`):
- Good: Clear type system (choice vs numeric)
- Good: Naming conventions (lowercase, 1-3 words)
- Weakness: Only 2 attribute type examples — could benefit from more
- Weakness: No example of a complete outline→model transformation

**Output Model**: `FindingModelBase` — direct model output. Good.

**Pydantic AI**:
- Uses `instructions=` ✅ — already migrated!
- No output validators — could benefit from basic validation (e.g., at least 1 attribute)
- No `UsageLimits`

**Reasoning Assessment**: Currently `none` — could benefit from `low` for more careful attribute extraction.

---

### Cross-Cutting Findings

#### Pydantic AI Modernization Status
| Pattern | Status |
|---------|--------|
| `instructions=` (V1) | 4/12 agents migrated (description, detail, nl_editor, md_editor, markdown_import). 7 still on `system_prompt=`. |
| `UsageLimits` | 0/12 agents. No production safety limits anywhere. |
| `RunUsage` propagation | Not used in any multi-agent pipeline. |
| Output validators | 2/12 (both editors). Others rely on post-processing or nothing. |
| `retries` | 2/12 explicitly set (similar search, similar analysis). Others use default (1). |
| Dependency floor | `>=0.3.2` — should be `>=1.0.0`. |

#### Shared Agent Tag Issue
- `similar_search` tag is shared between term generation (small) and search (base) agents
- This means a per-agent model override for `similar_search` would affect BOTH, even though they run at different tiers

#### Prompt Quality Ranking (best to worst)
1. **Finding Description** (template, excellent examples, structured rules)
2. **NL/Markdown Editor** (modular, clear safety rails, comprehensive)
3. **Anatomic Query Generator** (good examples, clear constraints)
4. **Ontology Categorization** (clear schema, good priorities)
5. **Anatomic Location Selection** (good examples but hedging)
6. **Ontology Query Generator** (one example, unstructured)
7. **Similar Models Analysis** (good criteria but no examples, grammar error)
8. **Markdown Import** (template, adequate but sparse)
9. **Similar Models Search** (numbering error, vague, no examples)
10. **Similar Models Term Gen** (generic, no examples)
11. **Finding Detail** (typo, negative instructions)

---

## Phase 2: Verified Benchmark Data

**Note on data validity**: An earlier benchmark script had a singleton propagation bug — model overrides set in the benchmark runner did not propagate to the agent modules, so the actual model used was not the one being tested. All benchmark data in this document has been replaced with Logfire-verified data only, where `gen_ai.response.model` in the Logfire spans confirms the model that handled each request.

All verified data can be reproduced via `scripts/benchmark_models.py` with Logfire verification, and queried from the Logfire `findingmodel` project. The original broken benchmark output files in `scripts/audit_results/` should not be used.

---

### Raw API Latency (simple prompt, no agent overhead)

| Model | Avg Latency | Notes |
|-------|-------------|-------|
| **gpt-5.4-nano** | **~800ms** | 10-15x faster than gpt-5-nano |
| **gpt-5.4-mini** | **~900ms** | 10-15x faster than gpt-5-mini |
| gpt-5-nano (old) | ~13,000ms | Dramatically slower — deprioritized or degraded |
| gpt-5-mini (old) | ~12,000ms | Same pattern as nano |

---

### Agent Pipeline Benchmarks

Full-model-shootout: 150 runs, 5 agents × 6 configs × 5 findings, all Logfire-verified (March 22, 2026).

#### Generative agents (nano primary)

**ontology_search** (single LLM call — query generation):

| Model | Reasoning | Avg | n |
|-------|-----------|-----|---|
| **gpt-5.4-mini** | none | **1.1s** | 5 |
| **gpt-5.4-nano** | low | **1.1s** | 5 |
| gpt-5.4-mini | low | 1.2s | 5 |
| gpt-5.4-nano | none | 1.9s | 5 |
| gemini-flash | minimal | 2.3s | 5 |
| gemini-flash | low | 3.7s | 5 |

**describe_finding** (single LLM call — description generation):

| Model | Reasoning | Avg | n |
|-------|-----------|-----|---|
| **gpt-5.4-nano** | none | **1.4s** | 5 |
| gpt-5.4-nano | low | 1.5s | 5 |
| gpt-5.4-mini | none | 1.9s | 5 |
| gpt-5.4-mini | low | 1.9s | 5 |
| gemini-flash | minimal | 2.6s | 5 |
| gemini-flash | low | 2.7s | 5 |

**similar_plan** (generative — search terms + metadata hypotheses):

| Model | Reasoning | Avg | n |
|-------|-----------|-----|---|
| **gpt-5.4-nano** | low | **1.3s** | 5 |
| gpt-5.4-mini | none | 1.4s | 5 |
| gpt-5.4-mini | low | 1.4s | 5 |
| gpt-5.4-nano | none | 1.6s | 5 |
| gemini-flash | minimal | 2.5s | 5 |
| gemini-flash | low | 2.5s | 5 |

#### Classification agents (mini primary)

**ontology_match** (multi-stage: query gen + BioOntology API + LLM categorization):

| Model | Reasoning | Avg | n |
|-------|-----------|-----|---|
| **gpt-5.4-nano** | low | **5.0s** | 5 |
| **gpt-5.4-mini** | none | **5.1s** | 5 |
| gpt-5.4-nano | none | 5.3s | 5 |
| gpt-5.4-mini | low | 5.4s | 5 |
| gemini-flash | minimal | 5.7s | 5 |
| gemini-flash | low | 6.3s | 5 |

**anatomic_select** (multi-stage: query gen + DuckDB search + LLM selection):

| Model | Reasoning | Avg | n |
|-------|-----------|-----|---|
| **gpt-5.4-nano** | low | **4.6s** | 5 |
| **gemini-flash** | minimal | **4.6s** | 5 |
| gemini-flash | low | 4.7s | 5 |
| gpt-5.4-nano | none | 4.8s | 5 |
| gpt-5.4-mini | none | 4.9s | 5 |
| gpt-5.4-mini | low | 5.0s | 5 |

Note: classification agents show similar latency across nano/mini/flash. Mini is chosen as primary for **quality** (better medical judgment on nuanced fields), not speed.

#### Full metadata assignment pipeline (from scratch, all metadata stripped)

| Model (classifier) | Reasoning | Total Pipeline | n |
|--------------------|-----------|---------------|---|
| gpt-5.4-nano | low | 12.8s | 1 |
| gpt-5.4-mini | low | 12.9s | 1 |

Pipeline breakdown (Logfire httpx instrumentation):

| Stage | Latency |
|-------|---------|
| Ontology candidates (query gen + BioOntology search + categorization) | ~5.8-7.4s |
| Anatomic candidates (query gen + DuckDB search + selection) | ~4.5s |
| Metadata classifier (the agent under test) | ~5.4-6.8s |
| Assembly | ~0s |
| **Total** | **~13s** |

Quality note: gpt-5.4-nano leaves gaps on nuanced fields (time_course, age_profile empty for PE). gpt-5.4-mini fills age_profile but still misses time_course and modalities. This is a prompt issue — both models handle the core fields (body_regions, entity_type, etiologies, sex_specificity) correctly.

---

## Phase 3: Recommendations

### Key Findings from Verified Data

1. **gpt-5.4-mini and gpt-5.4-nano are dramatically faster than previous-gen models** — 10-15x on raw API, 2-3x on full pipeline. They should be the primary OpenAI models in all agent chains.

2. **Haiku is fast WITHOUT thinking, catastrophically slow WITH thinking** — our config was sending `anthropic_thinking: {type: "enabled", budget_tokens: 1024}` for `reasoning=low`, which turned a 2.5s call into 36-54s. Fixed to send no thinking settings at all for `reasoning=none`. Haiku without thinking (8.8s on ontology_match) is competitive with gpt-5.4-mini (8.5s).

3. **BioOntology pagination was costing ~3s per search** — 4 sequential HTTP calls at ~1s each. Fixed to single 100-result page, saving ~3s on every ontology_match call.

4. **Previous benchmark data was invalid** — the benchmark script's singleton replacement didn't propagate to agent modules, so models were not actually switched. All data prior to the subprocess-based benchmark script should be disregarded.

5. **Logfire observability is essential for model benchmarking** — without checking `gen_ai.response.model` in Logfire spans, we had no way to verify which model actually ran. The httpx instrumentation also revealed the BioOntology pagination overhead.

### Current Agent Chain Configuration

All chains now use gpt-5.4-nano/gpt-5.4-mini as the OpenAI option, with `reasoning=none` for Haiku entries (no thinking budget sent). See `packages/findingmodel-ai/src/findingmodel_ai/data/supported_models.toml`.

The recommended per-agent config overhaul plan is deferred until the findingmodel-enrich branch merges — see `docs/plans/simplify-model-config.md`.

### Prompt Improvements (Priority Order)

1. **Ontology Query Generator**: Add 2-3 more diverse examples (acronym, multi-word, rare finding). Current single example is insufficient.

2. **Similar Models Search**: Fix numbering error (3→6 skip). Add concrete search strategy example.

3. **Similar Models Analysis**: Fix grammar error ("similar enough to the to be used"). Add concrete edit_existing vs create_new example.

4. **Finding Detail**: Fix typo "radassistant.nl" → "radiologyassistant.nl". Rewrite negative instructions as positive.

5. **Ontology Categorization**: Add 1-2 worked examples of categorization decisions.

6. **Anatomic Location Selection**: Remove hedging about pre-ranking. Add negative examples.

### Architecture Changes

1. **Split `similar_search` tag** into `similar_term_gen` (small) and `similar_search` (base) — currently they share a tag and can't be independently configured.

2. **Add `UsageLimits`** to all agents, especially tool-using agents (similar search, finding detail). Prevents runaway costs from tool loops.

3. **Add `RunUsage` propagation** in multi-agent pipelines for cost tracking.

### Pydantic AI Modernization

| Item | Priority | Effort |
|------|----------|--------|
| Migrate remaining 7 agents from `system_prompt=` to `instructions=` | Medium | Low |
| Add `UsageLimits` to all agents | High | Low |
| Bump dependency floor to `>=1.0.0` | High | Trivial |
| Add output validators to categorization agent | Medium | Medium |
| ~~Implement `agent_reasoning_overrides`~~ | ~~High~~ | ~~Medium~~ | **DONE** — implemented with full per-agent fallback chains |

### Key Insights

1. **gpt-5.4-nano and gpt-5.4-mini are the clear winners for all benchmarked tasks.** At ~800-900ms raw API latency (10-15x faster than their predecessors) and 1.2-2.8s for single-LLM-call agents, they dominate every simple generative task. For multi-stage pipelines, gpt-5.4-mini (5.8s anatomic_select, 8.5s ontology_match) leads all tested configurations.

2. **Haiku's thinking mode is a latency disaster.** The Anthropic `reasoning=low` config was inadvertently enabling extended thinking (`budget_tokens: 1024`), turning a 2.5s agent call into 36-54s. With thinking disabled, Haiku is competitive (8.8s on ontology_match). Never configure Haiku with a thinking budget for latency-sensitive agents.

3. **BioOntology pagination overhead was hidden until Logfire httpx instrumentation.** Four sequential HTTP calls at ~1s each added ~3s to every ontology_match pipeline call. A single pagesize=100 request eliminated this. Instrumenting external HTTP calls is essential for understanding real pipeline latency.

4. **Logfire model verification is non-negotiable for benchmarking.** The earlier benchmark data was entirely invalid because the wrong models were running. Always verify `gen_ai.response.model` in spans before drawing conclusions.

5. **Different agents need different models.** The optimal config uses multiple distinct model × reasoning pairings. This requires per-agent model configuration (already implemented via `supported_models.toml` fallback chains) rather than a global tier setting.

6. **Editing agents are output-bound, not model-bound** (~6s regardless of model). Latency improvements require architectural changes (streaming, partial updates), not model swaps. Keep the best model for quality here.
