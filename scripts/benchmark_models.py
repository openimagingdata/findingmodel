#!/usr/bin/env python3
"""Model benchmarking with Logfire observability.

Runs specific agents with specific model configurations and sends all telemetry
to Logfire. Results are analyzed via Logfire MCP queries, not stdout parsing.

IMPORTANT: Model overrides are process-scoped. The --comparison mode launches
a subprocess per (agent, model, reasoning) combination to ensure the config
singleton picks up the correct env vars before any imports.

Usage:
    # Benchmark a specific model on a specific agent (single process)
    uv run --package findingmodel-ai python scripts/benchmark_models.py \
        --agent ontology_search \
        --model openai:gpt-5.4-nano \
        --reasoning low

    # Run a comparison set (launches subprocesses)
    uv run --package findingmodel-ai python scripts/benchmark_models.py \
        --comparison openai-nano-generations

    # List available comparisons
    uv run --package findingmodel-ai python scripts/benchmark_models.py --list

    # Internal: run a single (agent, model, finding) — called by comparison mode
    uv run --package findingmodel-ai python scripts/benchmark_models.py \
        --run-one ontology_search openai:gpt-5.4-nano low pneumothorax
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Test findings
# ---------------------------------------------------------------------------

FINDINGS = {
    "pneumothorax": "Abnormal collection of air in the pleural space.",
    "hepatic steatosis": "Abnormal accumulation of fat within hepatocytes.",
    "meniscal tear": "A tear in the meniscal cartilage of the knee joint.",
    "subsegmental pulmonary embolism": "Blood clot in a subsegmental pulmonary artery.",
    "enostosis": "A benign sclerotic bone lesion, also known as a bone island.",
}

# ---------------------------------------------------------------------------
# Predefined comparison sets
# ---------------------------------------------------------------------------

COMPARISONS = {
    "openai-nano-generations": {
        "description": "Compare gpt-5-nano vs gpt-5.4-nano across reasoning levels",
        "agents": ["ontology_search", "describe_finding", "ontology_match", "anatomic_select"],
        "configs": [
            ("openai:gpt-5-nano", "none"),
            ("openai:gpt-5-nano", "low"),
            ("openai:gpt-5.4-nano", "none"),
            ("openai:gpt-5.4-nano", "low"),
            ("openai:gpt-5.4-nano", "medium"),
        ],
    },
    "openai-mini-generations": {
        "description": "Compare gpt-5-mini vs gpt-5.4-mini across reasoning levels",
        "agents": ["ontology_match", "anatomic_select", "edit_instructions", "import_markdown"],
        "configs": [
            ("openai:gpt-5-mini", "medium"),
            ("openai:gpt-5.4-mini", "none"),
            ("openai:gpt-5.4-mini", "low"),
            ("openai:gpt-5.4-mini", "medium"),
        ],
    },
    "classification-shootout": {
        "description": "All providers on classification tasks",
        "agents": ["ontology_match", "anatomic_select"],
        "configs": [
            ("google-gla:gemini-3.1-pro-preview", "medium"),
            ("openai:gpt-5.4-nano", "medium"),
            ("openai:gpt-5.4-mini", "low"),
            ("openai:gpt-5-mini", "medium"),
            ("anthropic:claude-sonnet-4-6", "low"),
            ("anthropic:claude-haiku-4-5", "low"),
        ],
    },
    "missing-data": {
        "description": "Fill gaps: simple generative with 5.4 models + Anthropic on classification",
        "agents": ["ontology_search", "describe_finding"],
        "configs": [
            ("openai:gpt-5.4-nano", "none"),
            ("openai:gpt-5.4-nano", "low"),
            ("openai:gpt-5.4-mini", "none"),
            ("openai:gpt-5.4-mini", "low"),
            ("anthropic:claude-haiku-4-5", "low"),
        ],
    },
    "anthropic-classification": {
        "description": "Anthropic models on classification tasks",
        "agents": ["ontology_match", "anatomic_select"],
        "configs": [
            ("anthropic:claude-haiku-4-5", "low"),
            ("anthropic:claude-sonnet-4-6", "low"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Single-run mode: called as subprocess with env vars already set
# ---------------------------------------------------------------------------

def run_one(agent_tag: str, model: str, reasoning: str, finding_name: str) -> None:
    """Execute a single agent call. Env vars must be set BEFORE this process starts."""
    import asyncio
    import os

    # Set path for imports
    sys.path.insert(0, str(REPO_ROOT / "packages" / "findingmodel-ai" / "src"))
    sys.path.insert(0, str(REPO_ROOT / "packages" / "findingmodel-ai" / "evals"))
    sys.path.insert(0, str(REPO_ROOT / "packages" / "findingmodel" / "src"))
    sys.path.insert(0, str(REPO_ROOT / "packages" / "anatomic-locations" / "src"))
    sys.path.insert(0, str(REPO_ROOT / "packages" / "oidm-common" / "src"))

    # Configure logfire BEFORE any agent imports — this is the correct order
    from findingmodel_ai.config import settings
    settings.configure_logfire()

    import logfire

    # Verify the override took effect
    chain = settings.resolve_agent_config(agent_tag)  # type: ignore[arg-type]
    resolved = chain[0]

    async def _run() -> None:
        description = FINDINGS.get(finding_name, "")

        with logfire.span(
            "benchmark {agent_tag} | {model}/{reasoning} | {finding}",
            agent_tag=agent_tag,
            model=model,
            reasoning=reasoning,
            finding=finding_name,
            resolved_model=resolved.model_string,
            resolved_reasoning=resolved.reasoning,
        ):
            if agent_tag == "ontology_search":
                from findingmodel_ai.search.ontology import generate_finding_query_terms
                result = await generate_finding_query_terms(finding_name, description)
                logfire.info("result: {n} terms: {terms}", n=len(result), terms=result)

            elif agent_tag == "describe_finding":
                from findingmodel_ai.authoring.description import create_info_from_name
                result = await create_info_from_name(finding_name)
                logfire.info("result: name={name} synonyms={synonyms}",
                             name=result.name, synonyms=result.synonyms)

            elif agent_tag == "ontology_match":
                from findingmodel_ai.search.ontology import (
                    categorize_with_validation,
                    execute_ontology_search,
                    generate_finding_query_terms,
                )
                query_terms = await generate_finding_query_terms(finding_name, description)
                search_results = await execute_ontology_search(query_terms=query_terms)
                result = await categorize_with_validation(finding_name, search_results, query_terms)
                logfire.info("result: {e}e {i}i {m}m",
                             e=len(result.exact_matches), i=len(result.should_include),
                             m=len(result.marginal))

            elif agent_tag == "anatomic_select":
                from anatomic_locations import AnatomicLocationIndex
                from findingmodel_ai.search.anatomic import (
                    create_location_selection_agent,
                    execute_anatomic_search,
                    generate_anatomic_query_terms,
                )
                query_info = await generate_anatomic_query_terms(finding_name, description)
                async with AnatomicLocationIndex() as index:
                    search_results = await execute_anatomic_search(query_info, index)
                agent = create_location_selection_agent()
                prompt = (
                    f"Finding: {finding_name}\n"
                    f"Search Results ({len(search_results)} locations):\n"
                    f"{json.dumps([r.model_dump() for r in search_results], indent=2)}\n"
                    f"Select the best primary anatomic location and 2-3 alternates."
                )
                r = await agent.run(prompt)
                output = r.output
                logfire.info("result: primary={primary} alternates={alts}",
                             primary=output.primary_location.concept_text,
                             alts=[a.concept_text for a in output.alternate_locations])

            elif agent_tag == "edit_instructions":
                from findingmodel.finding_model import FindingModelFull
                from findingmodel_ai.authoring.editor import edit_model_natural_language
                test_json = (REPO_ROOT / "packages" / "findingmodel" / "tests" / "data" / "defs" / "pulmonary_embolism.fm.json").read_text()
                fm = FindingModelFull.model_validate_json(test_json)
                result = await edit_model_natural_language(fm, "Add a 'size' attribute with range 0-10 cm. Also add 'chronic' as a synonym.")
                logfire.info("result: {c} changes, {r} rejections",
                             c=len(result.changes), r=len(result.rejections))

            elif agent_tag == "import_markdown":
                from findingmodel.finding_info import FindingInfo
                from findingmodel_ai.authoring.markdown_in import create_model_from_markdown
                info = FindingInfo(name="pneumothorax", description="Air in pleural space.")
                outline = "## Attributes\n\n### size\n- small\n- moderate\n- large\n\n### location\n- apical\n- basilar\n- lateral\n"
                result = await create_model_from_markdown(info, markdown_text=outline)
                logfire.info("result: {n} attributes", n=len(result.attributes))

            else:
                logfire.warn("Unknown agent tag: {tag}", tag=agent_tag)

    asyncio.run(_run())
    print(f"  [{agent_tag}] {finding_name} | {resolved.model_string}/{resolved.reasoning} | done")


# ---------------------------------------------------------------------------
# Comparison mode: launch subprocesses
# ---------------------------------------------------------------------------

def run_comparison(name: str) -> None:
    """Launch subprocesses for each (agent, model, reasoning, finding) combo."""
    comp = COMPARISONS[name]
    print(f"\n{'=' * 60}")
    print(f"Comparison: {name}")
    print(f"  {comp['description']}")
    print(f"  Agents: {comp['agents']}")
    print(f"  Configs: {len(comp['configs'])}")
    total = len(comp["configs"]) * len(comp["agents"]) * len(FINDINGS)
    print(f"  Total runs: {total}")
    print(f"{'=' * 60}\n")

    succeeded = 0
    failed = 0
    timed_out = 0

    for model, reasoning in comp["configs"]:
        for agent_tag in comp["agents"]:
            print(f"\n--- {agent_tag} | {model} / {reasoning} ---")
            for finding_name in FINDINGS:
                env = dict(__import__("os").environ)
                env[f"AGENT_MODEL_OVERRIDES__{agent_tag}"] = model
                env[f"AGENT_REASONING_OVERRIDES__{agent_tag}"] = reasoning

                try:
                    result = subprocess.run(
                        [
                            sys.executable, str(Path(__file__)),
                            "--run-one", agent_tag, model, reasoning, finding_name,
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.strip().splitlines():
                            print(line)
                        succeeded += 1
                    else:
                        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown"
                        print(f"  [{agent_tag}] {finding_name} | FAILED: {err[:120]}")
                        failed += 1
                except subprocess.TimeoutExpired:
                    print(f"  [{agent_tag}] {finding_name} | TIMEOUT (180s)")
                    timed_out += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {succeeded} succeeded, {failed} failed, {timed_out} timed out")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Single-agent mode: set env vars, then run in-process
# ---------------------------------------------------------------------------

def run_single(agent_tag: str, model: str, reasoning: str, finding: str | None) -> None:
    """Run a single agent with overrides. Sets env vars before importing config."""
    import os
    os.environ[f"AGENT_MODEL_OVERRIDES__{agent_tag}"] = model
    os.environ[f"AGENT_REASONING_OVERRIDES__{agent_tag}"] = reasoning

    findings = [finding] if finding else list(FINDINGS.keys())
    for finding_name in findings:
        run_one(agent_tag, model, reasoning, finding_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Model benchmarking with Logfire observability")
    parser.add_argument("--agent", type=str, help="Agent tag to benchmark")
    parser.add_argument("--model", type=str, help="Model spec (e.g., openai:gpt-5.4-nano)")
    parser.add_argument("--reasoning", type=str, default="low", help="Reasoning level")
    parser.add_argument("--finding", type=str, help="Single finding to test (default: all)")
    parser.add_argument("--comparison", type=str, help="Run a predefined comparison set")
    parser.add_argument("--list", action="store_true", help="List available comparisons")
    parser.add_argument("--run-one", nargs=4, metavar=("AGENT", "MODEL", "REASONING", "FINDING"),
                        help="Internal: run a single agent call (used by comparison mode)")
    args = parser.parse_args()

    if args.list:
        print("Available comparisons:")
        for cname, comp in COMPARISONS.items():
            print(f"  {cname}: {comp['description']}")
            print(f"    agents: {comp['agents']}")
            print(f"    configs: {len(comp['configs'])} model x reasoning combos")
        return

    if args.run_one:
        agent_tag, model, reasoning, finding_name = args.run_one
        run_one(agent_tag, model, reasoning, finding_name)
        return

    if args.comparison:
        if args.comparison not in COMPARISONS:
            print(f"Unknown comparison: {args.comparison}")
            print(f"Available: {', '.join(COMPARISONS.keys())}")
            sys.exit(1)
        run_comparison(args.comparison)
    elif args.agent and args.model:
        run_single(args.agent, args.model, args.reasoning, args.finding)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
