#!/usr/bin/env python
"""
Simple script to enrich a finding with metadata.

Usage:
    python scripts/enrich_finding.py "pneumonia"
    python scripts/enrich_finding.py "pneumonia" --provider anthropic
    python scripts/enrich_finding.py OIFM_AI_000001
"""

import argparse
import asyncio

from findingmodel import logger
from findingmodel.tools.finding_enrichment import enrich_finding

# Enable loguru logger (disabled by default in __init__.py)
logger.enable("findingmodel")


async def main() -> None:
    """Run finding enrichment and display results as JSON."""
    parser = argparse.ArgumentParser(description="Enrich a finding with metadata")
    parser.add_argument("finding", help="Finding name or OIFM ID")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="AI model provider (default: from settings)",
    )
    args = parser.parse_args()

    try:
        result = await enrich_finding(args.finding, provider=args.provider)
        # Output as clean JSON
        print("\n" + result.model_dump_json(exclude_none=True, indent=2))

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())
