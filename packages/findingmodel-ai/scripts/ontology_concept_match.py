"""Command-line tool for testing ontology concept matching."""

import argparse
import asyncio
import sys
import time
from typing import Any

from findingmodel.finding_info import FindingInfo
from findingmodel_ai.authoring.description import create_info_from_name
from findingmodel_ai.config import settings
from findingmodel_ai.search.ontology import (
    build_final_output,
    categorize_with_validation,
    execute_ontology_search,
    generate_finding_query_terms,
)

from findingmodel import logger


async def perform_search_stages(finding_info: FindingInfo) -> dict[str, Any]:
    """Perform the search in stages, returning results and timing for each."""
    stages_timing = {}

    # Stage 1: Generate query terms
    query_terms_start = time.perf_counter()
    query_terms = await generate_finding_query_terms(finding_info.name, finding_info.description)
    stages_timing["query_terms"] = time.perf_counter() - query_terms_start

    # Stage 2: BioOntology Search
    bio_start = time.perf_counter()
    search_results = await execute_ontology_search(
        query_terms=query_terms,
        exclude_anatomical=True,
        base_limit=30,
        max_results=12,
    )
    stages_timing["bio_search"] = time.perf_counter() - bio_start

    # Stage 3: AI Categorization
    ai_start = time.perf_counter()
    categorized = await categorize_with_validation(
        finding_name=finding_info.name,
        search_results=search_results,
        query_terms=query_terms,
    )
    stages_timing["ai_categorization"] = time.perf_counter() - ai_start

    # Stage 4: Build final output
    build_start = time.perf_counter()
    final_response = build_final_output(
        categorized=categorized,
        search_results=search_results,
        max_exact_matches=5,
        max_should_include=10,
        max_marginal=10,
    )
    stages_timing["build_output"] = time.perf_counter() - build_start

    return {
        "query_terms": query_terms,
        "search_results": search_results,
        "categorized": categorized,
        "final_response": final_response,
        "timing": stages_timing,
    }


def _print_finding_info_stage(finding_info: FindingInfo, source: str, timing: float) -> None:
    """Print the finding info stage output."""
    print("Stage 1: Creating Finding Info")
    print("-" * 30)
    print(f"  Time: {timing:.3f}s ({source})")
    print(f"  Name: {finding_info.name}")
    if finding_info.description:
        desc_preview = (
            finding_info.description[:100] + "..." if len(finding_info.description) > 100 else finding_info.description
        )
        print(f"  Description: {desc_preview}")
    print()


def _print_query_terms_stage(results: dict[str, Any]) -> None:
    """Print query terms generation stage."""
    print("Stage 2: Generating Query Terms")
    print("-" * 30)
    print(f"  Time: {results['timing']['query_terms']:.3f}s")
    print(f"  Terms: {', '.join(results['query_terms'])}")
    print()


def _print_bio_search_stage(results: dict[str, Any]) -> None:
    """Print BioOntology search stage."""
    print("Stage 2: BioOntology API Search")
    print("-" * 30)
    print(f"  Time: {results['timing']['bio_search']:.3f}s")
    print(f"  Results found: {len(results['search_results'])}")
    if results["search_results"]:
        print("  Top 10 results:")
        for i, result in enumerate(results["search_results"][:10], 1):
            score_str = f"{result.score:.4f}" if result.score < 1.0 else "1.0000"
            source = result.table_name if hasattr(result, "table_name") else "unknown"
            print(f"    {i:2}. {result.concept_text[:35]:35} ({result.concept_id:15}) [{source:10}] - {score_str}")
    print()


def _print_categorization_stage(results: dict[str, Any]) -> None:
    """Print AI categorization stage."""
    print("Stage 3: AI Categorization")
    print("-" * 30)
    print(f"  Time: {results['timing']['ai_categorization']:.3f}s")
    print("  Categorized concepts:")
    print(f"    Exact matches: {len(results['categorized'].exact_matches)}")
    print(f"    Should include: {len(results['categorized'].should_include)}")
    print(f"    Marginal: {len(results['categorized'].marginal)}")
    print(
        f"  Rationale: {results['categorized'].rationale[:100]}{'...' if len(results['categorized'].rationale) > 100 else ''}"
    )
    print()


def _print_final_output_stage(results: dict[str, Any]) -> None:
    """Print final output building stage."""
    print("Stage 4: Build Final Output")
    print("-" * 30)
    print(f"  Time: {results['timing']['build_output']:.3f}s")
    print("  Final results:")

    if results["final_response"].exact_matches:
        print(f"  Exact Matches ({len(results['final_response'].exact_matches)}):")
        for i, match in enumerate(results["final_response"].exact_matches, 1):
            source = match.table_name if hasattr(match, "table_name") else "unknown"
            print(f"    {i}. {match.concept_text} ({match.concept_id}) [{source}]")

    if results["final_response"].should_include:
        print(f"  Should Include ({len(results['final_response'].should_include)}):")
        for i, match in enumerate(results["final_response"].should_include[:3], 1):
            source = match.table_name if hasattr(match, "table_name") else "unknown"
            print(f"    {i}. {match.concept_text} ({match.concept_id}) [{source}]")
        if len(results["final_response"].should_include) > 3:
            print(f"    ... and {len(results['final_response'].should_include) - 3} more")

    if results["final_response"].marginal_concepts:
        print(f"  Marginal ({len(results['final_response'].marginal_concepts)}):")
        for i, match in enumerate(results["final_response"].marginal_concepts[:3], 1):
            source = match.table_name if hasattr(match, "table_name") else "unknown"
            print(f"    {i}. {match.concept_text} ({match.concept_id}) [{source}]")
        if len(results["final_response"].marginal_concepts) > 3:
            print(f"    ... and {len(results['final_response'].marginal_concepts) - 3} more")
    print()


def _print_verbose_stages(results: dict[str, Any]) -> None:
    """Print all verbose stage outputs."""
    _print_query_terms_stage(results)
    _print_bio_search_stage(results)
    _print_categorization_stage(results)
    _print_final_output_stage(results)


def _print_summary(finding_name: str, results: dict[str, Any], total_time: float) -> None:
    """Print non-verbose summary output."""
    print(f"Finding: {finding_name}")
    print(f"Exact Matches ({len(results['final_response'].exact_matches)}):")
    if results["final_response"].exact_matches:
        for i, match in enumerate(results["final_response"].exact_matches, 1):
            source = match.table_name if hasattr(match, "table_name") else "unknown"
            print(f"  {i}. {match.concept_text} ({match.concept_id}) [{source}]")
    else:
        print("  None")

    print(f"Should Include ({len(results['final_response'].should_include)}):")
    if results["final_response"].should_include:
        for i, match in enumerate(results["final_response"].should_include[:5], 1):
            source = match.table_name if hasattr(match, "table_name") else "unknown"
            print(f"  {i}. {match.concept_text} ({match.concept_id}) [{source}]")
        if len(results["final_response"].should_include) > 5:
            print(f"  ... and {len(results['final_response'].should_include) - 5} more")
    else:
        print("  None")

    print(f"Total time: {total_time:.3f}s")


async def test_ontology_concept_match(finding_name: str, description: str | None = None, verbose: bool = False) -> None:
    """Test ontology concept matching with timing information."""
    total_start = time.perf_counter()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Testing: {finding_name}")
        print(f"{'=' * 60}")
        print()

    try:
        # Stage 1: Create Finding Info
        finding_info_start = time.perf_counter()
        if description:
            finding_info = FindingInfo(name=finding_name, description=description)
            finding_info_source = "provided by user"
        else:
            finding_info = await create_info_from_name(finding_name)
            finding_info_source = "generated by AI"
        finding_info_time = time.perf_counter() - finding_info_start

        if verbose:
            _print_finding_info_stage(finding_info, finding_info_source, finding_info_time)

        # Perform search stages
        results = await perform_search_stages(finding_info)
        total_time = time.perf_counter() - total_start

        # Display output
        if verbose:
            _print_verbose_stages(results)
            print(f"Total time: {total_time:.3f}s")
        else:
            _print_summary(finding_name, results, total_time)

    except Exception as e:
        total_time = time.perf_counter() - total_start
        print(f"Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        print(f"Total time: {total_time:.3f}s")


def check_configuration() -> bool:
    """Check if the configuration is valid for running the tool."""
    try:
        # Check required configuration
        if not settings.openai_api_key.get_secret_value():
            print("Error: OPENAI_API_KEY is required but not set")
            return False

        if not settings.bioontology_api_key or not settings.bioontology_api_key.get_secret_value():
            print("Error: BIOONTOLOGY_API_KEY is required but not set")
            print("Please set BIOONTOLOGY_API_KEY in your .env file")
            return False

        print("Configuration:")
        print(f"  Default Model: {settings.default_model}")
        print("  BioOntology API: Configured")
        print()
        return True

    except Exception as e:
        print(f"Configuration error: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test ontology concept matching for medical findings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "pneumonia"
  %(prog)s "hepatic metastasis" --verbose
  %(prog)s "pulmonary embolism" -d "Obstruction of pulmonary arterial circulation" -v
""",
    )

    parser.add_argument("finding_name", help="Name of the medical finding to search for")

    parser.add_argument("-d", "--description", help="Optional description of the finding")

    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output with timing for each stage")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging from underlying tools")

    return parser.parse_args()


async def main() -> None:
    """Main entry point for the ontology concept matching tool."""
    args = parse_args()

    print("Ontology Concept Matching Tool")
    print("=" * 40)
    print()

    # Configure loguru logging if debug is enabled
    if args.debug:
        logger.enable("findingmodel")
        logger.info("Debug logging enabled")
    else:
        logger.disable("findingmodel")

    # Check configuration
    if not check_configuration():
        sys.exit(1)

    # Run the search
    await test_ontology_concept_match(args.finding_name, args.description, args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
