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

## Phase 2: Test Results Summary

**439 total runs** (364 initial + 75 supplemental fast-model runs), **0 errors across initial run, 1 failure in supplement** (flash-lite on enostosis anatomic selection).

Tested 17 distinct model × reasoning configurations across 5 findings of varying complexity.

### Finding 1: Task complexity determines whether model choice matters

**Simple tasks** (query generation, similar analysis): All models — from GPT-5-nano ($0.05/$0.40) to Claude Opus 4.6 ($5/$25) — produce equivalent quality. GPT-5-nano stood out as the fastest model for ontology query generation (2647ms) and finding description (1388ms, with the tightest variance of any config — max 1578ms vs gemini-flash's 4008ms spike). Budget models are competitive or fastest for these tasks.

**Medium tasks** (ontology categorization): Quality is identical across all 17 configs for 4/5 findings. Latency varies more — budget models are actually *slower* here (Haiku: 3091ms, Flash-Lite/low: 3374ms) vs mid-tier models (gpt-5-mini/medium: 2085ms, Gemini 3.1 Pro/low: 2028ms).

**Hard tasks** (anatomic selection on edge cases): Model choice matters significantly. The enostosis edge case exposes dramatic differences in both quality and latency tail behavior.

### Finding 2: Budget models are fastest for simple tasks, slowest for hard ones

| Agent | Fastest Config | Latency | Model Tier |
|-------|---------------|---------|-----------|
| similar_analysis | **flash-lite/medium** | 1052ms | Budget |
| similar_term_gen | **gemini-flash/minimal** | 978ms | Fast |
| anatomic_query_gen | **gemini-flash/minimal** | 980ms | Fast |
| finding_description | **gpt-5-nano/low** | 1388ms | Budget |
| ontology_categorization | **gemini-3.1-pro/low** | 2028ms | Frontier |
| anatomic_selection | **gemini-3.1-pro/medium** | 1944ms | Frontier |

For categorization and selection, budget models (Haiku, Flash-Lite) are 30-65% slower than mid/frontier models. The extra thinking on harder tasks produces longer output and more hesitant behavior.

### Finding 3: The enostosis P80 tail is the critical differentiator

Most configs handle standard findings (pneumothorax, meniscal tear, etc.) in 1.5-2.5s for anatomic selection. The enostosis edge case reveals massive divergence:

| Config | Enostosis Latency | Primary Location |
|--------|-------------------|-----------------|
| gemini-3.1-pro/medium | **1647ms** | "skeletal system" |
| anthropic:claude-haiku-4-5/low | 5714ms | "bone" |
| anthropic:claude-sonnet-4-6/low | 5762ms | "bone" |
| openai:gpt-5.4/medium | 6759ms | "bone" |
| anthropic:claude-opus-4-6/medium | 6743ms | "bone organ" |
| google-gla:gemini-3-flash-preview/medium | 7679ms | "bone" |
| openai:gpt-5.4/none | **7835ms** | **"vertebra"** (wrong) |
| openai:gpt-5-mini/high | 8555ms | "bone" |
| google-gla:gemini-3-flash-preview/low | 8830ms | "bone" |
| google-gla:gemini-3.1-flash-lite-preview/low | **9001ms** | "bone" |
| anthropic:claude-sonnet-4-6/medium | **9584ms** | "bone" |

Gemini 3.1 Pro/medium is the only config that resolves enostosis quickly (1.6s vs 5.7-9.6s for everything else). However, it chose "skeletal system" which is arguably too broad — "bone" (chosen by most others) is more appropriate.

### Finding 4: `reasoning=none` is a latency trap on base tier

GPT-5.4/none is the **slowest config** for:
- ontology categorization: 2688ms (last place out of 17)
- similar analysis: 1706ms (last place out of 17)

Adding reasoning — even just `low` — consistently reduces latency by 10-25%. The model produces shorter, more decisive output when it can think first.

### Finding 5: Quality is uniform; editing agents are latency-insensitive

All models produce identical categorization for 4/5 findings. All correctly handle the NL edit command and markdown import. Editing agents are ~6s regardless of model — bottlenecked by output size, not reasoning.

---

## Phase 3: Recommendations

### Model Configuration Recommendations (Latency-Optimized)

#### Small Tier — Match model to task simplicity

| Agent | Current | Current Avg | Recommended | Rec Avg | Savings | Why |
|-------|---------|-------------|-------------|---------|---------|-----|
| `ontology_search` | gemini-flash/low | 2872ms | **gpt-5-nano/low** | 2647ms | -8% | Fastest on this task; quality identical |
| `anatomic_search` | gemini-flash/low | 1281ms | **gemini-flash/minimal** | 980ms | -24% | Same model, less thinking overhead |
| `similar_search` (term gen) | gemini-flash/low | 1036ms | **gemini-flash/minimal** | 978ms | -6% | Marginal but free |
| `describe_finding` | gemini-flash/low | 1950ms | **gpt-5-nano/low** | 1388ms | -29% | Fastest; consistent low-variance |

#### Base Tier — Use the right model for the task complexity

| Agent | Current | Avg (Max) | Recommended | Avg (Max) | Savings | Why |
|-------|---------|-----------|-------------|-----------|---------|-----|
| `ontology_match` | gpt-5.4/none | 2688 (3274) | **gemini-3.1-pro/low** | 2028 (2235) | -25% | Fastest; tight max; quality identical |
| `anatomic_select` | gpt-5.4/none | 3272 (7835) | **gemini-3.1-pro/medium** | 1944 (2228) | **-41%** | Eliminates P80 tail spike entirely |
| `similar_assess` | gpt-5.4/none | 1706 (2554) | **flash-lite/medium** | 1052 (1424) | **-38%** | Simple task; budget model is fastest |
| `edit_instructions` | gpt-5.4/none | 6239 | **gpt-5.4/low** | 7514* | — | *See note; keep flagship, add reasoning for quality |
| `edit_markdown` | gpt-5.4/none | 6239 | **gpt-5.4/low** | — | — | Same as above |
| `import_markdown` | gpt-5.4/none | 5036 | **claude-opus-4-6/medium** | 4460 | -11% | Fastest; best quality for structured output |

*Note on editors: gemini-3.1-pro/low was fastest at 6017ms, but editing is the one task where we want to optimize for quality over latency. GPT-5.4 or Opus are the right models here — the ~6s is dominated by output size, not model capability.*

#### Pipeline-Level Impact

| Pipeline | Current | Optimized | Savings | Key Change |
|----------|---------|-----------|---------|------------|
| Ontology search | 5.6s | 4.7s | **-16%** | Gemini 3.1 Pro/low for categorization |
| Anatomic search | 4.6s | **2.9s** | **-36%** | Gemini 3.1 Pro/medium for selection |
| Similar models | 2.7s | **1.6s** | **-41%** | Flash-Lite/medium for analysis |
| **Total LLM time** | **12.9s** | **9.2s** | **-29%** | |

The anatomic pipeline P80 drops from **7.8s → 2.2s** — eliminating the worst-case latency spikes.

### Architecture Change: Replace Tier System with Per-Agent Model Profiles

The tier system (`small`/`base`/`full`) is the wrong abstraction. The audit data shows the optimal model × reasoning pairing is **task-specific**, not tier-specific. Three "base" tier agents want three completely different models at three different reasoning levels.

The current system should evolve so that **each agent declares its own model + reasoning as the primary configuration**, with tiers demoted to a fallback for agents that haven't been individually tuned.

This also needs provider-based fallback logic: when a preferred provider's API key isn't configured, the system should fall back to the best available alternative for that agent's task profile, not just a generic tier default.

---

## Appendix A: Per-Agent Benchmark Data (March 2026)

Reference data for evaluating future models against current baselines. Each agent has a **task profile** describing what it needs, a **current best config**, and **baseline metrics** to beat.

### Agent Task Profiles

#### Simple generative (query/term generation)

These agents produce short lists of medical terms. Quality is identical across all current-gen models. Optimize purely for latency.

| Agent | Tag | Task Profile | Best Config | Avg Latency | Max Latency |
|-------|-----|-------------|-------------|-------------|-------------|
| Ontology Query Gen | `ontology_search` | Generate 2-5 medical synonym terms | `gpt-5-nano/low` | 2647ms | 3365ms |
| Anatomic Query Gen | `anatomic_search` | Identify region + 3-5 anatomic terms | `gemini-3-flash/minimal` | 980ms | 1182ms |
| Similar Term Gen | `similar_search` | Generate 3-5 search terms | `gemini-3-flash/minimal` | 978ms | 1162ms |
| Finding Description | `describe_finding` | Name normalization + synonyms + 1-2 sentence description | `gpt-5-nano/low` | 1388ms | 1578ms |

GPT-5-nano is the standout for pure text generation tasks (ontology queries, descriptions). It is the fastest model tested for those two agents, and critically has the **most consistent latency** — its max (1578ms for descriptions) is far below the spikes seen with other models (gemini-flash hit 4008ms on hepatic steatosis). At $0.05/$0.40 per MTok it is also essentially free. For tasks involving structured output with constrained types (region selection, term lists), gemini-flash/minimal edges it out.

**What to look for in new models**: Sub-1s latency with structured output. Quality bar is low — any current-gen model passes. GPT-5-nano's consistency (tight max latency) is the benchmark to beat, not just average speed.

#### Medical classification (categorization/selection)

These agents classify or select from provided options. They require medical domain knowledge and multi-step judgment. Budget models are slower and less reliable here.

| Agent | Tag | Task Profile | Best Config | Avg Latency | Max Latency | Key Edge Case |
|-------|-----|-------------|-------------|-------------|-------------|---------------|
| Ontology Categorization | `ontology_match` | Categorize ~12 concepts into 3 relevance tiers | `gemini-3.1-pro/low` | 2028ms | 2235ms | enostosis (rare finding) |
| Anatomic Selection | `anatomic_select` | Select primary + alternates from search results | `gemini-3.1-pro/medium` | 1944ms | 2228ms | enostosis: **1647ms** (vs 5.7-9.6s others) |
| Similar Analysis | `similar_assess` | Assess similarity, recommend edit vs create | `flash-lite/medium` | 1052ms | 1424ms | — (simple enough for budget) |

**What to look for in new models**: Consistent sub-2.5s on categorization tasks. The key test is the **enostosis edge case for anatomic selection** — most models spike to 6-9s. A good model resolves it under 2s.

**Enostosis anatomic selection benchmark** (the hardest single test):

| Model | Reasoning | Latency | Primary Location | Quality |
|-------|-----------|---------|-----------------|---------|
| gemini-3.1-pro | medium | **1647ms** | "skeletal system" | Too broad |
| claude-haiku-4-5 | low | 5714ms | "bone" | Correct |
| claude-sonnet-4-6 | low | 5762ms | "bone" | Correct |
| gpt-5.4 | medium | 6759ms | "bone" | Correct |
| claude-opus-4-6 | medium | 6743ms | "bone organ" | Odd phrasing |
| gemini-3-flash | medium | 7679ms | "bone" | Correct |
| gpt-5.4 | none | 7835ms | "vertebra" | **Wrong** |
| gpt-5-mini | high | 8555ms | "bone" | Correct |
| gemini-3-flash | low | 8830ms | "bone" | Correct |
| flash-lite | low | 9001ms | "bone" | Correct |
| claude-sonnet-4-6 | medium | 9584ms | "bone" | Correct |

#### Complex structured output (editing/import)

These agents produce full JSON model structures. Latency is dominated by output size (~6s regardless of model). Quality and instruction-following matter most.

| Agent | Tag | Task Profile | Best Config | Avg Latency | Notes |
|-------|-----|-------------|-------------|-------------|-------|
| NL Editor | `edit_instructions` | Parse NL command, edit FindingModelFull JSON | `gpt-5.4/low` | ~6.2s | Has output validator; quality > speed |
| Markdown Editor | `edit_markdown` | Parse edited markdown, update JSON | `gpt-5.4/low` | ~6.2s | Same as above |
| Markdown Import | `import_markdown` | Convert outline to FindingModelBase | `claude-opus-4-6/medium` | 4460ms | Best structured output quality |

**What to look for in new models**: Faster structured JSON output (sub-4s) while maintaining ID preservation and change tracking accuracy. Streaming/partial output could help here architecturally.

### Provider Capability Summary (March 2026)

For configuring provider-based fallbacks when not all API keys are available:

| Provider | Models Tested | Best For | Worst For | Key Strength |
|----------|--------------|---------|-----------|-------------|
| **OpenAI** | gpt-5-nano, gpt-5-mini, gpt-5.4 | **gpt-5-nano**: fastest + most consistent for simple generative tasks. **gpt-5.4**: complex editing with reasoning. **gpt-5-mini**: solid mid-tier, good with medium/high reasoning. | gpt-5.4/none is slowest config for categorization and analysis | Widest useful model range — nano for speed, 5.4 for capability |
| **Google** | gemini-3-flash, gemini-3.1-flash-lite, gemini-3.1-pro | **gemini-3.1-pro/medium**: only model that avoids tail latency spikes on edge cases. **gemini-3-flash/minimal**: fastest for structured-output generation (region + terms). | Complex structured output (editing) | Thinking mode enables decisive, fast classification |
| **Anthropic** | claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6 | **claude-opus-4-6**: best for complex structured output (markdown import). **claude-haiku-4-5**: competitive on simple tasks. | Sonnet was never the fastest config for any agent | Best instruction-following for editing safety constraints |

**Suggested provider fallback order per task profile**:
- Simple generative: OpenAI (gpt-5-nano) → Google (gemini-3-flash/minimal) → Anthropic (haiku)
- Medical classification: Google (gemini-3.1-pro) → OpenAI (gpt-5-mini+medium) → Anthropic (sonnet)
- Complex structured output: Anthropic (claude-opus-4-6) → OpenAI (gpt-5.4) → Google (gemini-3.1-pro)

### Ontology Categorization: Full Model Comparison

All configs tested, ranked by average latency. Quality column shows categorization counts for pneumothorax / enostosis (the easy and hard cases).

| Rank | Config | Avg | Max | Pneumothorax | Enostosis |
|------|--------|-----|-----|-------------|-----------|
| 1 | gemini-3.1-pro/low | 2028 | 2235 | 5e 1i 0m | 0e 3i 2m |
| 2 | gpt-5.4/medium | 2078 | 2256 | 5e 1i 0m | 0e 3i 2m |
| 3 | gpt-5-mini/medium | 2085 | 2280 | 5e 1i 0m | 0e 3i 2m |
| 4 | gemini-3.1-pro/medium | 2126 | 2294 | 5e 1i 0m | 0e 3i 2m |
| 5 | claude-opus-4-6/low | 2113 | 2388 | 5e 1i 0m | 0e 3i 2m |
| 6 | gpt-5-mini/high | 2214 | 2465 | 5e 1i 0m | 0e 3i 2m |
| 7 | claude-sonnet-4-6/low | 2275 | 3004 | 5e 1i 0m | 0e 3i 2m |
| 8 | claude-opus-4-6/medium | 2342 | 3731 | 5e 1i 0m | 0e 3i 2m |
| 9 | claude-sonnet-4-6/medium | 2398 | 3442 | 5e 1i 0m | 0e 3i 2m |
| 10 | gpt-5.4/low | 2413 | 3034 | 5e 1i 0m | 0e 3i 2m |
| 11 | gemini-3-flash/low | 2470 | 4024 | 5e 5i 1m | 0e 3i 2m |
| 12 | gemini-3-flash/medium | 2497 | 3282 | 5e 1i 0m | 0e 3i 2m |
| 13 | claude-haiku-4-5/medium | 2637 | 3107 | 5e 4i 2m | 0e 3i 2m |
| 14 | gpt-5.4/none | 2688 | 3274 | 5e 1i 0m | 0e 3i 2m |
| 15 | flash-lite/medium | 2897 | 4195 | 5e 5i 1m | 0e 3i 2m |
| 16 | claude-haiku-4-5/low | 3091 | 3487 | 5e 4i 2m | 0e 3i 2m |
| 17 | flash-lite/low | 3374 | 8012 | 5e 5i 1m | 0e 3i 2m |

Note: Haiku, Flash, and Flash-Lite show slightly different categorization distribution (more items in should_include) but identical exact_matches. The core task is performed correctly by all models.

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

1. **Match model to task complexity, not to a fixed tier.** GPT-5-nano is fastest and most consistent for simple generative tasks (description: 1388ms avg, 1578ms max — tighter than any other model). Flash-Lite and Haiku win on simple classification (similar analysis: 1052ms). Mid-tier models (GPT-5-mini, Gemini 3.1 Pro) hit the sweet spot for medical judgment tasks. Frontier models (GPT-5.4, Opus 4.6) only earn their keep on complex editing.

2. **`reasoning=none` on base tier is a latency trap.** GPT-5.4/none is dead last on 2 of 3 base-tier search agents. Models produce shorter, more decisive output when they can think first — reasoning overhead is more than compensated.

3. **The P80 tail matters more than the average.** Anatomic selection averages 1.9-3.4s across configs, but the enostosis edge case ranges from 1.6s (Gemini 3.1 Pro/medium) to 9.6s (Sonnet/medium). Optimizing for tail latency means picking the model that handles the worst case well.

4. **Different agents need different models.** The optimal config uses 5+ distinct model × reasoning pairings. This requires implementing per-agent reasoning overrides (config.py:107 TODO) — the single most impactful infrastructure change.

5. **Editing agents are output-bound, not model-bound** (~6s regardless of model). Latency improvements require architectural changes (streaming, partial updates), not model swaps. Keep the best model for quality here.

6. **Quality is uniform across current-gen models for our tasks.** All 17 configs produce identical categorization for 4/5 findings. The differentiation is almost entirely in latency and tail behavior, not quality. This means we can optimize aggressively for speed without worrying about quality regression.
