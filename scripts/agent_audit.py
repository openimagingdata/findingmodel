#!/usr/bin/env python3
"""Agent Performance Audit Script.

Systematically tests all findingmodel-ai agents (except enrichment) across
multiple model x reasoning configurations to compare quality, latency, and cost.

Usage:
    # Run all agents with current defaults
    uv run --package findingmodel-ai python scripts/agent_audit.py --group search

    # Run a specific agent with a specific model override
    uv run --package findingmodel-ai python scripts/agent_audit.py --agent ontology_query --model "openai:gpt-5-mini" --reasoning low

    # Run all configs for a specific agent
    uv run --package findingmodel-ai python scripts/agent_audit.py --agent ontology_query --all-configs

    # Run full audit (all agents, all configs, all inputs)
    uv run --package findingmodel-ai python scripts/agent_audit.py --full

Results are written to scripts/audit_results/ as JSON files.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add packages to path for imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "findingmodel-ai" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "findingmodel" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "anatomic-locations" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "oidm-common" / "src"))

if TYPE_CHECKING:
    from findingmodel_ai.config import FindingModelAIConfig


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result from a single agent run."""

    agent_name: str
    input_name: str
    model: str
    reasoning: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    output_json: Any
    output_summary: str
    errors: list[str] = field(default_factory=list)
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "input": self.input_name,
            "model": self.model,
            "reasoning": self.reasoning,
            "latency_ms": round(self.latency_ms, 1),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "output_summary": self.output_summary,
            "output": self.output_json,
            "errors": self.errors,
            "retries": self.retries,
        }


# ---------------------------------------------------------------------------
# Pricing table (current-gen models only)
# ---------------------------------------------------------------------------

PRICING: dict[str, tuple[float, float]] = {
    # (input_per_mtok, output_per_mtok)
    "openai:gpt-5.4": (2.50, 15.00),
    "openai:gpt-5.4-mini": (0.75, 4.50),
    "openai:gpt-5.4-nano": (0.20, 1.25),
    "openai:gpt-5-mini": (0.25, 2.00),
    "openai:gpt-5-nano": (0.05, 0.40),
    "anthropic:claude-opus-4-6": (5.00, 25.00),
    "anthropic:claude-sonnet-4-6": (3.00, 15.00),
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
    "google-gla:gemini-3-flash-preview": (0.50, 3.00),
    "google-gla:gemini-3.1-pro-preview": (2.00, 12.00),  # estimated
    "google-gla:gemini-3.1-flash-lite-preview": (0.25, 1.50),
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate USD cost based on token counts."""
    if model in PRICING:
        in_rate, out_rate = PRICING[model]
        return (tokens_in * in_rate + tokens_out * out_rate) / 1_000_000
    return 0.0


# ---------------------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------------------

FINDINGS = [
    {
        "name": "pneumothorax",
        "description": "Abnormal collection of air in the pleural space between the lung and chest wall.",
    },
    {
        "name": "hepatic steatosis",
        "description": "Abnormal accumulation of fat within hepatocytes, commonly known as fatty liver.",
    },
    {"name": "meniscal tear", "description": "A tear in the meniscal cartilage of the knee joint."},
    {
        "name": "subsegmental pulmonary embolism",
        "description": "A blood clot lodged in a subsegmental branch of the pulmonary artery.",
    },
    {"name": "enostosis", "description": "A benign sclerotic bone lesion, also known as a bone island."},
]

# Model configs for small-tier agents
SMALL_CONFIGS = [
    ("google-gla:gemini-3-flash-preview", "low"),  # S-A: Current default
    ("google-gla:gemini-3-flash-preview", "minimal"),  # S-B: Less reasoning
    ("google-gla:gemini-3-flash-preview", "medium"),  # S-C: More reasoning
    ("openai:gpt-5-mini", "none"),  # S-D: GPT-5 no reasoning
    ("openai:gpt-5-mini", "low"),  # S-E: GPT-5 light reasoning
    ("openai:gpt-5-nano", "low"),  # S-F: Budget floor
    ("anthropic:claude-haiku-4-5", "low"),  # S-G: Anthropic fast
    ("google-gla:gemini-3.1-flash-lite-preview", "low"),  # S-H: Newest Google budget
]

# Model configs for base-tier agents
BASE_CONFIGS = [
    ("openai:gpt-5.4", "none"),  # B-A: Current default
    ("openai:gpt-5.4", "low"),  # B-B: Light reasoning
    ("openai:gpt-5.4", "medium"),  # B-C: Medium reasoning
    ("anthropic:claude-sonnet-4-6", "low"),  # B-D: Sonnet low
    ("anthropic:claude-sonnet-4-6", "medium"),  # B-E: Sonnet medium
    ("anthropic:claude-opus-4-6", "low"),  # B-F: Opus low
    ("anthropic:claude-opus-4-6", "medium"),  # B-G: Opus medium
    ("google-gla:gemini-3.1-pro-preview", "low"),  # B-H: Gemini Pro low
    ("google-gla:gemini-3.1-pro-preview", "medium"),  # B-I: Gemini Pro medium
    ("openai:gpt-5-mini", "medium"),  # B-J: Cheap + reasoning
    ("openai:gpt-5-mini", "high"),  # B-K: Cheap + high reasoning
    ("google-gla:gemini-3-flash-preview", "medium"),  # B-L: Flash for base tasks
]


# ---------------------------------------------------------------------------
# Model override helpers
# ---------------------------------------------------------------------------


def set_model_override(agent_tag: str, model_spec: str, reasoning: str) -> None:
    """Set environment variables to override model for an agent tag."""
    os.environ[f"AGENT_MODEL_OVERRIDES__{agent_tag}"] = model_spec
    # Reasoning is set at tier level, but we need to reload config
    # We'll handle this differently - by recreating the config


def clear_model_overrides() -> None:
    """Clear all model override environment variables."""
    for key in list(os.environ.keys()):
        if key.startswith("AGENT_MODEL_OVERRIDES__"):
            del os.environ[key]


def get_config_with_overrides(model_spec: str, reasoning: str, tier: str = "base") -> "FindingModelAIConfig":
    """Create a fresh config with the specified model and reasoning."""
    from findingmodel_ai.config import FindingModelAIConfig

    overrides: dict[str, str] = {}
    if tier == "small":
        overrides["DEFAULT_MODEL_SMALL"] = model_spec
        overrides["DEFAULT_REASONING_SMALL"] = reasoning
    elif tier == "base":
        overrides["DEFAULT_MODEL"] = model_spec
        overrides["DEFAULT_REASONING_BASE"] = reasoning
    elif tier == "full":
        overrides["DEFAULT_MODEL_FULL"] = model_spec
        overrides["DEFAULT_REASONING_FULL"] = reasoning

    # Set env vars temporarily
    old_env = {}
    for k, v in overrides.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        config = FindingModelAIConfig()
        return config
    finally:
        # Restore env
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Agent test runners
# ---------------------------------------------------------------------------


async def run_ontology_query_gen(finding: dict[str, str], model: str, reasoning: str) -> RunResult:
    """Test the ontology query generator agent."""
    # Monkey-patch the settings module to use our config
    import findingmodel_ai.config as config_mod

    old_settings = config_mod.settings
    config_mod.settings = get_config_with_overrides(model, reasoning, "small")

    try:
        from findingmodel_ai.search.ontology import generate_finding_query_terms

        start = time.perf_counter()
        result = await generate_finding_query_terms(finding["name"], finding.get("description"))
        elapsed = (time.perf_counter() - start) * 1000

        return RunResult(
            agent_name="ontology_query_gen",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=elapsed,
            tokens_in=0,  # TODO: capture from usage
            tokens_out=0,
            cost_usd=0.0,
            output_json=result,
            output_summary=f"{len(result)} terms: {result[:5]}",
        )
    except Exception as e:
        return RunResult(
            agent_name="ontology_query_gen",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=None,
            output_summary="FAILED",
            errors=[str(e)],
        )
    finally:
        config_mod.settings = old_settings


async def run_anatomic_query_gen(finding: dict[str, str], model: str, reasoning: str) -> RunResult:
    """Test the anatomic query generator agent."""
    import findingmodel_ai.config as config_mod

    old_settings = config_mod.settings
    config_mod.settings = get_config_with_overrides(model, reasoning, "small")

    try:
        from findingmodel_ai.search.anatomic import generate_anatomic_query_terms

        start = time.perf_counter()
        result = await generate_anatomic_query_terms(finding["name"], finding.get("description"))
        elapsed = (time.perf_counter() - start) * 1000

        return RunResult(
            agent_name="anatomic_query_gen",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=elapsed,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json={"region": result.region, "terms": result.terms},
            output_summary=f"region={result.region}, {len(result.terms)} terms: {result.terms}",
        )
    except Exception as e:
        return RunResult(
            agent_name="anatomic_query_gen",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=None,
            output_summary="FAILED",
            errors=[str(e)],
        )
    finally:
        config_mod.settings = old_settings


async def run_similar_term_gen(finding: dict[str, str], model: str, reasoning: str) -> RunResult:
    """Test the similar models term generation agent."""
    import findingmodel_ai.config as config_mod

    old_settings = config_mod.settings
    config_mod.settings = get_config_with_overrides(model, reasoning, "small")

    try:
        from findingmodel_ai.search.similar import create_term_generation_agent

        agent = create_term_generation_agent()
        model_info = f"Name: {finding['name']}\n"
        if finding.get("description"):
            model_info += f"Description: {finding['description']}\n"

        prompt = f"Generate 3-5 search terms for finding existing medical imaging definitions similar to this finding:\n\n{model_info}"

        start = time.perf_counter()
        result = await agent.run(prompt)
        elapsed = (time.perf_counter() - start) * 1000
        terms = result.output.search_terms

        return RunResult(
            agent_name="similar_term_gen",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=elapsed,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=terms,
            output_summary=f"{len(terms)} terms: {terms}",
        )
    except Exception as e:
        return RunResult(
            agent_name="similar_term_gen",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=None,
            output_summary="FAILED",
            errors=[str(e)],
        )
    finally:
        config_mod.settings = old_settings


async def run_finding_description(finding: dict[str, str], model: str, reasoning: str) -> RunResult:
    """Test the finding description agent."""
    import findingmodel_ai.config as config_mod

    old_settings = config_mod.settings
    config_mod.settings = get_config_with_overrides(model, reasoning, "small")

    try:
        from findingmodel_ai.authoring.description import create_info_from_name

        start = time.perf_counter()
        result = await create_info_from_name(finding["name"])
        elapsed = (time.perf_counter() - start) * 1000

        return RunResult(
            agent_name="finding_description",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=elapsed,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=result.model_dump(),
            output_summary=f"name='{result.name}', {len(result.synonyms or [])} synonyms, desc={result.description[:80] if result.description else 'None'}...",
        )
    except Exception as e:
        return RunResult(
            agent_name="finding_description",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=None,
            output_summary="FAILED",
            errors=[str(e)],
        )
    finally:
        config_mod.settings = old_settings


async def run_ontology_categorization(
    finding: dict[str, str], search_results: list, query_terms: list[str], model: str, reasoning: str
) -> RunResult:
    """Test the ontology categorization agent with pre-fetched search results."""
    import findingmodel_ai.config as config_mod

    old_settings = config_mod.settings
    config_mod.settings = get_config_with_overrides(model, reasoning, "base")

    try:
        from findingmodel_ai.search.ontology import categorize_with_validation

        start = time.perf_counter()
        result = await categorize_with_validation(
            finding_name=finding["name"],
            search_results=search_results,
            query_terms=query_terms,
        )
        elapsed = (time.perf_counter() - start) * 1000

        return RunResult(
            agent_name="ontology_categorization",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=elapsed,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json={
                "exact_matches": result.exact_matches,
                "should_include": result.should_include,
                "marginal": result.marginal,
                "rationale": result.rationale,
            },
            output_summary=f"{len(result.exact_matches)} exact, {len(result.should_include)} include, {len(result.marginal)} marginal",
        )
    except Exception as e:
        return RunResult(
            agent_name="ontology_categorization",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=None,
            output_summary="FAILED",
            errors=[str(e)],
        )
    finally:
        config_mod.settings = old_settings


async def run_anatomic_selection(
    finding: dict[str, str], search_results: list, model: str, reasoning: str
) -> RunResult:
    """Test the anatomic location selection agent with pre-fetched search results."""
    import findingmodel_ai.config as config_mod

    old_settings = config_mod.settings
    config_mod.settings = get_config_with_overrides(model, reasoning, "small")

    try:
        import json as json_mod

        from findingmodel_ai.search.anatomic import create_location_selection_agent

        agent = create_location_selection_agent()
        prompt = f"""
Finding: {finding["name"]}
Description: {finding.get("description", "Not provided")}

Search Results ({len(search_results)} locations found):
{json_mod.dumps([r.model_dump() for r in search_results], indent=2)}

Select the best primary anatomic location and 2-3 good alternates.
"""
        start = time.perf_counter()
        result = await agent.run(prompt)
        elapsed = (time.perf_counter() - start) * 1000
        output = result.output

        return RunResult(
            agent_name="anatomic_selection",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=elapsed,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json={
                "primary": {"id": output.primary_location.concept_id, "text": output.primary_location.concept_text},
                "alternates": [{"id": a.concept_id, "text": a.concept_text} for a in output.alternate_locations],
                "reasoning": output.reasoning,
            },
            output_summary=f"primary='{output.primary_location.concept_text}', {len(output.alternate_locations)} alternates",
        )
    except Exception as e:
        return RunResult(
            agent_name="anatomic_selection",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=None,
            output_summary="FAILED",
            errors=[str(e)],
        )
    finally:
        config_mod.settings = old_settings


async def run_similar_analysis(
    finding: dict[str, str], search_results_data: list[dict], model: str, reasoning: str
) -> RunResult:
    """Test the similar models analysis agent with pre-fetched search results."""
    import findingmodel_ai.config as config_mod

    old_settings = config_mod.settings
    config_mod.settings = get_config_with_overrides(model, reasoning, "base")

    try:
        from findingmodel_ai.search.similar import create_analysis_agent

        agent = create_analysis_agent()
        model_info = f"Name: {finding['name']}\n"
        if finding.get("description"):
            model_info += f"Description: {finding['description']}\n"

        analysis_prompt = f"""
Based on the search results, analyze the similarity between the proposed model and existing models.

Finding Information:
{model_info}

Existing definitions Found:
```json
{json.dumps(search_results_data, indent=2)}
```

Analyze and determine if any existing definitions are similar enough that editing them would be better.
"""
        start = time.perf_counter()
        result = await agent.run(analysis_prompt)
        elapsed = (time.perf_counter() - start) * 1000
        output = result.output

        return RunResult(
            agent_name="similar_analysis",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=elapsed,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json={
                "recommendation": output.recommendation,
                "confidence": output.confidence,
                "similar_models": [dict(m) for m in output.similar_models],
            },
            output_summary=f"rec={output.recommendation}, conf={output.confidence:.2f}, {len(output.similar_models)} similar",
        )
    except Exception as e:
        return RunResult(
            agent_name="similar_analysis",
            input_name=finding["name"],
            model=model,
            reasoning=reasoning,
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=None,
            output_summary="FAILED",
            errors=[str(e)],
        )
    finally:
        config_mod.settings = old_settings


async def run_nl_editor(model_json: str, command: str, model: str, reasoning: str) -> RunResult:
    """Test the natural language editor agent."""
    import findingmodel_ai.config as config_mod

    old_settings = config_mod.settings
    config_mod.settings = get_config_with_overrides(model, reasoning, "base")

    try:
        from findingmodel import FindingModelFull
        from findingmodel_ai.authoring.editor import edit_model_natural_language

        fm = FindingModelFull.model_validate_json(model_json)

        start = time.perf_counter()
        result = await edit_model_natural_language(fm, command)
        elapsed = (time.perf_counter() - start) * 1000

        return RunResult(
            agent_name="nl_editor",
            input_name=command[:50],
            model=model,
            reasoning=reasoning,
            latency_ms=elapsed,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json={
                "changes": result.changes,
                "rejections": result.rejections,
                "n_attributes": len(result.model.attributes),
            },
            output_summary=f"{len(result.changes)} changes, {len(result.rejections)} rejections",
        )
    except Exception as e:
        return RunResult(
            agent_name="nl_editor",
            input_name=command[:50],
            model=model,
            reasoning=reasoning,
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=None,
            output_summary="FAILED",
            errors=[str(e)],
        )
    finally:
        config_mod.settings = old_settings


async def run_markdown_import(finding_name: str, markdown_text: str, model: str, reasoning: str) -> RunResult:
    """Test the markdown import agent."""
    import findingmodel_ai.config as config_mod

    old_settings = config_mod.settings
    config_mod.settings = get_config_with_overrides(model, reasoning, "base")

    try:
        from findingmodel.finding_info import FindingInfo
        from findingmodel_ai.authoring.markdown_in import create_model_from_markdown

        info = FindingInfo(name=finding_name, description=f"A radiology finding: {finding_name}")

        start = time.perf_counter()
        result = await create_model_from_markdown(info, markdown_text=markdown_text)
        elapsed = (time.perf_counter() - start) * 1000

        return RunResult(
            agent_name="markdown_import",
            input_name=finding_name,
            model=model,
            reasoning=reasoning,
            latency_ms=elapsed,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json={
                "name": result.name,
                "n_attributes": len(result.attributes),
                "attributes": [{"name": a.name, "type": a.type} for a in result.attributes],
            },
            output_summary=f"name='{result.name}', {len(result.attributes)} attributes",
        )
    except Exception as e:
        return RunResult(
            agent_name="markdown_import",
            input_name=finding_name,
            model=model,
            reasoning=reasoning,
            latency_ms=0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            output_json=None,
            output_summary="FAILED",
            errors=[str(e)],
        )
    finally:
        config_mod.settings = old_settings


# ---------------------------------------------------------------------------
# Test data for authoring agents
# ---------------------------------------------------------------------------

EDIT_COMMAND = "Add a 'size' attribute with range 0-10 cm. Also add 'chronic' as a synonym."

PNEUMOTHORAX_OUTLINE = """## Attributes

### size
- small (less than 2cm apical)
- moderate (2-4cm)
- large (greater than 4cm or complete collapse)

### location
- apical
- basilar
- lateral
- medial

### type
- simple
- tension
- loculated

### associated findings
- subcutaneous emphysema
- rib fracture
- pleural effusion
- chest tube
"""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def print_result(result: RunResult) -> None:
    """Print a single result in a readable format."""
    status = "OK" if not result.errors else "FAIL"
    print(
        f"  [{status}] {result.agent_name} | {result.input_name[:30]:30s} | "
        f"{result.model:45s} | reason={result.reasoning:7s} | "
        f"{result.latency_ms:8.0f}ms | {result.output_summary}"
    )
    if result.errors:
        for err in result.errors:
            print(f"         ERROR: {err[:120]}")


async def run_small_tier_agents(findings: list[dict[str, str]], configs: list[tuple[str, str]]) -> list[RunResult]:
    """Run all small-tier agents across findings and configs."""
    results: list[RunResult] = []

    runners = [
        ("ontology_query_gen", run_ontology_query_gen),
        ("anatomic_query_gen", run_anatomic_query_gen),
        ("similar_term_gen", run_similar_term_gen),
        ("finding_description", run_finding_description),
    ]

    for config_model, config_reasoning in configs:
        print(f"\n--- Config: {config_model} / reasoning={config_reasoning} ---")
        for _agent_name, runner in runners:
            for finding in findings:
                result = await runner(finding, config_model, config_reasoning)
                print_result(result)
                results.append(result)

    return results


async def run_base_tier_agents_needing_context(
    findings: list[dict[str, str]],
    configs: list[tuple[str, str]],
    cached_contexts: dict[str, dict[str, Any]],
) -> list[RunResult]:
    """Run base-tier agents that need upstream context (categorization, selection, analysis)."""
    results: list[RunResult] = []

    for config_model, config_reasoning in configs:
        print(f"\n--- Config: {config_model} / reasoning={config_reasoning} ---")
        for finding in findings:
            ctx = cached_contexts.get(finding["name"], {})

            # Ontology categorization
            if ctx.get("ontology_results") and ctx.get("query_terms"):
                result = await run_ontology_categorization(
                    finding,
                    ctx["ontology_results"],
                    ctx["query_terms"],
                    config_model,
                    config_reasoning,
                )
                print_result(result)
                results.append(result)

            # Anatomic selection
            if ctx.get("anatomic_results"):
                result = await run_anatomic_selection(
                    finding,
                    ctx["anatomic_results"],
                    config_model,
                    config_reasoning,
                )
                print_result(result)
                results.append(result)

            # Similar models analysis
            if ctx.get("similar_results"):
                result = await run_similar_analysis(
                    finding,
                    ctx["similar_results"],
                    config_model,
                    config_reasoning,
                )
                print_result(result)
                results.append(result)

    return results


async def run_authoring_agents(configs: list[tuple[str, str]], model_json: str) -> list[RunResult]:
    """Run authoring/editing agents."""
    results: list[RunResult] = []

    for config_model, config_reasoning in configs:
        print(f"\n--- Config: {config_model} / reasoning={config_reasoning} ---")

        # NL Editor
        result = await run_nl_editor(model_json, EDIT_COMMAND, config_model, config_reasoning)
        print_result(result)
        results.append(result)

        # Markdown import
        result = await run_markdown_import("pneumothorax", PNEUMOTHORAX_OUTLINE, config_model, config_reasoning)
        print_result(result)
        results.append(result)

    return results


async def prefetch_contexts(findings: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Pre-fetch search results for downstream agents using current defaults.

    Returns cached context per finding name.
    """
    contexts: dict[str, dict[str, Any]] = {}

    for finding in findings:
        print(f"  Pre-fetching context for '{finding['name']}'...")
        ctx: dict[str, Any] = {}

        # Ontology search
        try:
            from findingmodel_ai.search.ontology import execute_ontology_search, generate_finding_query_terms

            query_terms = await generate_finding_query_terms(finding["name"], finding.get("description"))
            ctx["query_terms"] = query_terms
            search_results = await execute_ontology_search(query_terms=query_terms)
            ctx["ontology_results"] = search_results
            print(f"    Ontology: {len(query_terms)} terms, {len(search_results)} results")
        except Exception as e:
            print(f"    Ontology FAILED: {e}")

        # Anatomic search
        try:
            from anatomic_locations import AnatomicLocationIndex
            from findingmodel_ai.search.anatomic import execute_anatomic_search, generate_anatomic_query_terms

            query_info = await generate_anatomic_query_terms(finding["name"], finding.get("description"))
            async with AnatomicLocationIndex() as index:
                anatomic_results = await execute_anatomic_search(query_info, index)
            ctx["anatomic_results"] = anatomic_results
            print(f"    Anatomic: {len(anatomic_results)} results (region={query_info.region})")
        except Exception as e:
            print(f"    Anatomic FAILED: {e}")

        # Similar models search
        try:
            from findingmodel.index import FindingModelIndex

            index = FindingModelIndex()
            search_terms = [finding["name"]]
            batch_results = await index.search_batch(search_terms, limit=5)
            similar_data = []
            for _query, results in batch_results.items():
                for r in results:
                    similar_data.append({"oifm_id": r.oifm_id, "name": r.name})
            ctx["similar_results"] = similar_data
            print(f"    Similar: {len(similar_data)} results")
        except Exception as e:
            print(f"    Similar FAILED: {e}")

        contexts[finding["name"]] = ctx

    return contexts


def load_test_model_json() -> str:
    """Load a test FindingModelFull JSON for editing tests."""
    test_data = REPO_ROOT / "packages" / "findingmodel" / "tests" / "data" / "defs" / "pulmonary_embolism.fm.json"
    return test_data.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


async def run_full_audit() -> None:
    """Run the complete audit: all agents, all configs, all inputs."""
    output_dir = REPO_ROOT / "scripts" / "audit_results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    all_results: list[RunResult] = []

    print("=" * 80)
    print("AGENT PERFORMANCE AUDIT")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 80)

    # Step 1: Pre-fetch contexts for downstream agents
    print("\n### PHASE 1: Pre-fetching contexts for downstream agents ###")
    cached_contexts = await prefetch_contexts(FINDINGS)

    # Step 2: Small-tier agents
    print("\n### PHASE 2: Small-tier agents (query generation, descriptions) ###")
    small_results = await run_small_tier_agents(FINDINGS, SMALL_CONFIGS)
    all_results.extend(small_results)

    # Step 3: Base-tier context-dependent agents
    print("\n### PHASE 3: Base-tier agents needing context (categorization, selection, analysis) ###")
    base_context_results = await run_base_tier_agents_needing_context(FINDINGS, BASE_CONFIGS, cached_contexts)
    all_results.extend(base_context_results)

    # Step 4: Authoring agents
    print("\n### PHASE 4: Authoring agents (editor, markdown import) ###")
    model_json = load_test_model_json()
    authoring_results = await run_authoring_agents(BASE_CONFIGS, model_json)
    all_results.extend(authoring_results)

    # Save results
    output_file = output_dir / f"audit_{timestamp}.json"
    results_data = {
        "timestamp": timestamp,
        "total_runs": len(all_results),
        "total_errors": sum(1 for r in all_results if r.errors),
        "results": [r.to_dict() for r in all_results],
    }
    output_file.write_text(json.dumps(results_data, indent=2, default=str))
    print(f"\n{'=' * 80}")
    print(f"AUDIT COMPLETE: {len(all_results)} runs, {sum(1 for r in all_results if r.errors)} errors")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}")


async def run_quick_test(agent_name: str | None = None, model: str | None = None, reasoning: str | None = None) -> None:
    """Run a quick test for a single agent with current defaults or specified override."""
    findings = FINDINGS[:2]  # Just first 2 for quick test

    configs = [(model, reasoning)] if model and reasoning else [("google-gla:gemini-3-flash-preview", "low")]

    print(f"Quick test: agent={agent_name or 'all'}, configs={configs}")

    if agent_name in (None, "ontology_query"):
        results = await run_small_tier_agents(findings, configs)
        for r in results:
            print_result(r)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agent Performance Audit")
    parser.add_argument("--full", action="store_true", help="Run full audit (all agents, all configs)")
    parser.add_argument("--agent", type=str, help="Run specific agent only")
    parser.add_argument("--model", type=str, help="Model override (e.g., 'openai:gpt-5-mini')")
    parser.add_argument("--reasoning", type=str, default="low", help="Reasoning level override")
    parser.add_argument(
        "--group", type=str, choices=["search", "authoring", "all"], default="all", help="Agent group to test"
    )
    args = parser.parse_args()

    if args.full:
        asyncio.run(run_full_audit())
    else:
        asyncio.run(run_quick_test(args.agent, args.model, args.reasoning))


if __name__ == "__main__":
    main()
