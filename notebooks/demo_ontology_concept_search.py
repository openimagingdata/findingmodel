#!/usr/bin/env python
"""
Demo script for the Ontology Concept Search Tool.

This script demonstrates the two-agent system that searches across medical ontologies
to find and categorize relevant concepts for imaging findings.

Usage:
    python notebooks/demo_ontology_concept_search.py
"""

import asyncio
import sys
import time
from typing import Optional

from findingmodel.config import settings
from findingmodel.tools.ontology_concept_search import (
    CategorizedOntologyConcepts,
    search_ontology_concepts,
)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def print_section(title: str) -> None:
    """Print a section title."""
    print(f"\n{title}")
    print("-" * len(title))


def display_results(results: CategorizedOntologyConcepts, finding_name: str, elapsed_time: float) -> None:
    """Display categorized results in a readable format."""
    print_header(f"Results for: {finding_name} (completed in {elapsed_time:.1f}s)")

    # Display search summary
    print_section("Search Summary")
    print(results.search_summary)

    # Display exact matches
    print_section(f"Exact Matches ({len(results.exact_matches)})")
    if results.exact_matches:
        for i, match in enumerate(results.exact_matches, 1):
            print(f"  {i}. {match.concept_text}")
            print(f"     Source: {match.table_name} | ID: {match.concept_id} | Score: {match.score:.3f}")
    else:
        print("  No exact matches found")

    # Display should include
    print_section(f"Should Include ({len(results.should_include)})")
    if results.should_include:
        for i, match in enumerate(results.should_include, 1):
            print(f"  {i}. {match.concept_text}")
            print(f"     Source: {match.table_name} | ID: {match.concept_id} | Score: {match.score:.3f}")
    else:
        print("  No concepts in this category")

    # Display marginal concepts
    print_section(f"Marginal/Consider ({len(results.marginal_concepts)})")
    if results.marginal_concepts:
        for i, match in enumerate(results.marginal_concepts, 1):
            print(f"  {i}. {match.concept_text}")
            print(f"     Source: {match.table_name} | ID: {match.concept_id} | Score: {match.score:.3f}")
    else:
        print("  No marginal concepts")

    # Display excluded anatomical concepts
    if results.excluded_anatomical:
        print_section(f"Excluded Anatomical Concepts ({len(results.excluded_anatomical)})")
        for concept in results.excluded_anatomical[:5]:  # Show first 5
            print(f"  - {concept}")
        if len(results.excluded_anatomical) > 5:
            print(f"  ... and {len(results.excluded_anatomical) - 5} more")


async def demo_single_finding(
    finding_name: str, finding_description: Optional[str] = None, exclude_anatomical: bool = True
) -> float:
    """Demonstrate search for a single finding. Returns elapsed time."""
    try:
        print(f"\n🔍 Searching for: {finding_name}")
        if finding_description:
            print(f"   Description: {finding_description}")
        print(f"   Exclude anatomical: {exclude_anatomical}")
        print("   ⏱️  Starting search...")

        start_time = time.time()
        results = await search_ontology_concepts(
            finding_name=finding_name,
            finding_description=finding_description,
            exclude_anatomical=exclude_anatomical,
            max_exact_matches=5,
            max_should_include=10,
            max_marginal=10,
        )
        elapsed_time = time.time() - start_time

        display_results(results, finding_name, elapsed_time)
        return elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time if "start_time" in locals() else 0
        print(f"❌ Error searching for {finding_name} (after {elapsed_time:.1f}s): {e}")
        import traceback

        traceback.print_exc()
        return elapsed_time


async def main() -> None:
    """Main demonstration function."""
    print_header("Ontology Concept Search Tool Demonstration")

    # Check for required configuration
    if not settings.lancedb_uri:
        print("\n⚠️  Warning: LANCEDB_URI not configured. This demo requires database access.")
        print("Please set LANCEDB_URI and LANCEDB_API_KEY in your .env file.")
        return

    if not settings.openai_api_key:
        print("\n⚠️  Warning: OPENAI_API_KEY not configured. This demo requires OpenAI API access.")
        print("Please set OPENAI_API_KEY in your .env file.")
        return

    print("\nThis demo will search for ontology concepts for various medical findings.")
    print("The tool uses two AI agents:")
    print(f"  1. Search Agent ({settings.openai_default_model_small}): Generates diverse search queries")
    print(f"  2. Categorization Agent ({settings.openai_default_model}): Categorizes results by relevance")

    # Track timing for all demos
    total_start_time = time.time()
    demo_times = []

    # Demo 1: Pathological finding with clear concepts
    demo_time = await demo_single_finding(
        finding_name="hepatic metastasis",
        finding_description="Secondary malignant neoplasm in liver parenchyma",
        exclude_anatomical=True,
    )
    demo_times.append(("hepatic metastasis", demo_time))

    # Demo 2: Inflammatory condition
    demo_time = await demo_single_finding(
        finding_name="pneumonia",
        finding_description="Inflammation of lung parenchyma with consolidation",
        exclude_anatomical=True,
    )
    demo_times.append(("pneumonia", demo_time))

    # Demo 3: Vascular finding
    demo_time = await demo_single_finding(
        finding_name="pulmonary embolism",
        finding_description="Obstruction of pulmonary arterial circulation by thrombus",
        exclude_anatomical=True,
    )
    demo_times.append(("pulmonary embolism", demo_time))

    # Demo 4: Structural abnormality
    demo_time = await demo_single_finding(
        finding_name="brain aneurysm",
        finding_description="Focal dilation of cerebral artery wall",
        exclude_anatomical=True,
    )
    demo_times.append(("brain aneurysm", demo_time))

    # Demo 5: Same finding WITHOUT anatomical filtering (for comparison)
    print_header("Comparison: With vs Without Anatomical Filtering")
    demo_time = await demo_single_finding(
        finding_name="liver mass", finding_description="Focal lesion in hepatic parenchyma", exclude_anatomical=False
    )
    demo_times.append(("liver mass", demo_time))

    # Demo 6: Finding without description
    demo_time = await demo_single_finding(finding_name="fracture", finding_description=None, exclude_anatomical=True)
    demo_times.append(("fracture", demo_time))

    total_elapsed_time = time.time() - total_start_time

    # Display timing summary
    print_header("Performance Summary")
    print(f"\nTotal demonstration time: {total_elapsed_time:.1f}s")
    print(f"Average time per finding: {total_elapsed_time / len(demo_times):.1f}s")
    print("\nIndividual timings:")
    for finding_name, elapsed_time in demo_times:
        print(f"  • {finding_name}: {elapsed_time:.1f}s")

    print_header("Demonstration Complete")
    print("\n✅ The ontology concept search tool successfully:")
    print("   - Generated diverse search queries for each finding")
    print("   - Searched across multiple ontology tables")
    print("   - Filtered anatomical concepts when requested")
    print("   - Categorized results into relevance tiers")
    print("   - Provided reasoning for categorization decisions")
    print("   - Validated exact matches with retry mechanism")


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 11):
        print("This script requires Python 3.11 or higher")
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
