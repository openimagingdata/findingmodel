#!/usr/bin/env python
"""
Simple script to enrich a finding with metadata.

Usage:
    python scripts/enrich_finding.py "pneumonia"
    python scripts/enrich_finding.py "pneumonia" --model anthropic:claude-sonnet-4-5
    python scripts/enrich_finding.py OIFM_AI_000001
"""

import argparse
import asyncio

from findingmodel_ai.enrichment.agentic import enrich_finding_agentic as enrich_finding

from findingmodel import logger

# Enable loguru logger (disabled by default in __init__.py)
logger.enable("findingmodel")
logger.enable("findingmodel_ai")


async def main() -> None:
    """Run finding enrichment and display results as JSON."""
    parser = argparse.ArgumentParser(description="Enrich a finding with metadata")
    parser.add_argument("finding", help="Finding name or OIFM ID")
    parser.add_argument(
        "--model",
        help="AI model string (e.g., 'openai:gpt-5', 'anthropic:claude-sonnet-4-5'). If not specified, uses configured default.",
    )
    args = parser.parse_args()

    try:
        result = await enrich_finding(args.finding, model=args.model)
        # Output as clean JSON
        print("\n" + result.model_dump_json(exclude_none=True, indent=2))

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())
