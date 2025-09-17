"""Command-line tool for testing anatomic location search."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.finding_info import FindingInfo
from findingmodel.tools import create_info_from_name
from findingmodel.tools.anatomic_location_search import (
    create_location_selection_agent,
    execute_anatomic_search,
    generate_anatomic_query_terms,
)
from findingmodel.tools.duckdb_search import DuckDBOntologySearchClient
from findingmodel.tools.ontology_search import rerank_with_cohere


async def perform_search_stages(finding_info: FindingInfo) -> dict:
    """Perform the search in stages, returning results and timing for each."""
    stages_timing = {}

    # Stage 1: Generate query terms
    query_terms_start = time.perf_counter()
    query_info = await generate_anatomic_query_terms(finding_info.name, finding_info.description)
    stages_timing["query_terms"] = time.perf_counter() - query_terms_start

    # Stage 2: DuckDB Search
    duckdb_start = time.perf_counter()
    async with DuckDBOntologySearchClient() as client:
        search_results = await execute_anatomic_search(query_info, client)
    stages_timing["duckdb_search"] = time.perf_counter() - duckdb_start

    # Keep original results before reranking
    original_results = search_results.copy()

    # Stage 3: Cohere Reranking (if enabled and available)
    reranked_results = search_results
    cohere_enabled = settings.use_cohere_in_anatomic_location_search
    cohere_available = bool(settings.cohere_api_key) and cohere_enabled
    rerank_query = None

    if cohere_available and search_results:
        cohere_start = time.perf_counter()
        # Build a focused query using just the primary search term
        # The query should be simple and direct for best reranking results
        primary_term = query_info.terms[0] if query_info.terms else finding_info.name
        rerank_query = f"anatomic location: {primary_term}"

        reranked_results = await rerank_with_cohere(
            query=rerank_query,
            documents=search_results,
            retry_attempts=1,
        )
        stages_timing["cohere_rerank"] = time.perf_counter() - cohere_start
    else:
        stages_timing["cohere_rerank"] = 0.0

    # Stage 4: AI Selection
    ai_start = time.perf_counter()
    selection_agent = create_location_selection_agent()

    # Build structured prompt for the agent
    prompt = f"""
Finding: {finding_info.name}
Description: {finding_info.description or "Not provided"}

Search Results ({len(reranked_results)} locations found):
{json.dumps([r.model_dump() for r in reranked_results], indent=2)}

Select the best primary anatomic location and 2-3 good alternates.
The goal is to find the "sweet spot" where it's as specific as possible,
but still encompassing the locations where the finding can occur.
"""

    result = await selection_agent.run(prompt)
    final_response = result.output
    stages_timing["ai_selection"] = time.perf_counter() - ai_start

    return {
        "query_info": query_info,
        "original_results": original_results,
        "reranked_results": reranked_results,
        "final_response": final_response,
        "timing": stages_timing,
        "cohere_available": cohere_available,
        "rerank_query": rerank_query,
    }


def display_results_comparison(original: list, reranked: list, limit: int = 10) -> None:
    """Display comparison of results before and after reranking."""
    # Check if order changed
    order_changed = False
    if len(original) == len(reranked):
        for i in range(min(limit, len(original))):
            if original[i].concept_id != reranked[i].concept_id:
                order_changed = True
                break

    if order_changed:
        print("  Before reranking | After reranking:")
        print("  " + "-" * 60)
        for i in range(min(limit, len(original))):
            orig = original[i] if i < len(original) else None
            rerank = reranked[i] if i < len(reranked) else None

            orig_text = f"{orig.concept_text[:25]:25}" if orig else " " * 25
            rerank_text = f"{rerank.concept_text[:25]:25}" if rerank else " " * 25

            if orig and rerank and orig.concept_id != rerank.concept_id:
                # Highlight changes with arrow
                print(f"  {i + 1:2}. {orig_text} → {rerank_text}")
            else:
                print(f"  {i + 1:2}. {orig_text}   {rerank_text}")
    else:
        print("  Order unchanged (or no reranking applied)")


def _create_finding_info(finding_name: str, description: Optional[str]) -> tuple[FindingInfo, str, float]:
    """Create FindingInfo and return info, source, and timing."""
    import asyncio

    async def _async_create_finding_info() -> tuple[FindingInfo, str, float]:
        finding_info_start = time.perf_counter()
        if description:
            # Use provided description
            finding_info = FindingInfo(name=finding_name, description=description)
            finding_info_source = "provided by user"
        else:
            # Generate FindingInfo using AI
            finding_info = await create_info_from_name(finding_name)
            finding_info_source = "generated by AI"
        finding_info_time = time.perf_counter() - finding_info_start
        return finding_info, finding_info_source, finding_info_time

    return asyncio.get_event_loop().run_until_complete(_async_create_finding_info())


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


def _print_verbose_stages(results: dict) -> None:
    """Print all verbose stage outputs."""
    # Stage 2: Query Terms
    print("Stage 2: Generating Query Terms")
    print("-" * 30)
    print(f"  Time: {results['timing']['query_terms']:.3f}s")
    print(f"  Region: {results['query_info'].region or 'None (no specific region)'}")
    print(f"  Terms: {', '.join(results['query_info'].terms)}")
    print()

    # Stage 3: DuckDB Search
    print("Stage 3: DuckDB Search")
    print("-" * 30)
    print(f"  Time: {results['timing']['duckdb_search']:.3f}s")
    print(f"  Results found: {len(results['original_results'])}")
    if results["original_results"]:
        print("  Top 10 results (before reranking):")
        for i, result in enumerate(results["original_results"][:10], 1):
            score_str = f"{result.score:.4f}" if result.score < 1.0 else "1.0000"
            print(f"    {i:2}. {result.concept_text[:45]:45} ({result.concept_id}) - {score_str}")
    print()

    # Stage 4: Cohere Reranking
    _print_cohere_stage(results)

    # Stage 5: AI Selection
    _print_ai_selection_stage(results)


def _print_cohere_stage(results: dict) -> None:
    """Print Cohere reranking stage output."""
    if results["cohere_available"]:
        print("Stage 4: Cohere Reranking")
        print("-" * 30)
        print(f"  Time: {results['timing']['cohere_rerank']:.3f}s")
        if results.get("rerank_query"):
            print(f"  Query: {results['rerank_query']}")
        display_results_comparison(results["original_results"], results["reranked_results"])
        print()
    else:
        print("Stage 4: Cohere Reranking")
        print("-" * 30)
        if not settings.cohere_api_key:
            print("  Skipped (Cohere API key not configured)")
        elif not settings.use_cohere_in_anatomic_location_search:
            print("  Skipped (disabled for anatomic search - see use_cohere_in_anatomic_location_search)")
        else:
            print("  Skipped (no results to rerank)")
        print()


def _print_ai_selection_stage(results: dict) -> None:
    """Print AI selection stage output."""
    print("Stage 5: AI Selection")
    print("-" * 30)
    print(f"  Time: {results['timing']['ai_selection']:.3f}s")
    print(
        f"  Primary: {results['final_response'].primary_location.concept_text} ({results['final_response'].primary_location.concept_id})"
    )
    print(f"  Alternates: {len(results['final_response'].alternate_locations)} locations")
    if results["final_response"].alternate_locations:
        for i, alt in enumerate(results["final_response"].alternate_locations, 1):
            print(f"    {i}. {alt.concept_text} ({alt.concept_id})")
    print(
        f"  Reasoning: {results['final_response'].reasoning[:100]}{'...' if len(results['final_response'].reasoning) > 100 else ''}"
    )
    print()


def _print_summary(finding_name: str, results: dict, total_time: float) -> None:
    """Print non-verbose summary output."""
    print(f"Finding: {finding_name}")
    print(
        f"Primary Location: {results['final_response'].primary_location.concept_text} ({results['final_response'].primary_location.concept_id})"
    )
    print(f"Alternates ({len(results['final_response'].alternate_locations)}):")
    if results["final_response"].alternate_locations:
        for i, alt in enumerate(results["final_response"].alternate_locations, 1):
            print(f"  {i}. {alt.concept_text} ({alt.concept_id})")
    else:
        print("  None")
    print(f"Total time: {total_time:.3f}s")


async def test_anatomic_location_search(
    finding_name: str, description: Optional[str] = None, verbose: bool = False
) -> None:
    """Test anatomic location search with timing information."""
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
            # Use provided description
            finding_info = FindingInfo(name=finding_name, description=description)
            finding_info_source = "provided by user"
        else:
            # Generate FindingInfo using AI
            finding_info = await create_info_from_name(finding_name)
            finding_info_source = "generated by AI"
        finding_info_time = time.perf_counter() - finding_info_start

        if verbose:
            _print_finding_info_stage(finding_info, finding_info_source, finding_info_time)

        # Perform search stages
        results = await perform_search_stages(finding_info)

        # Display output
        total_time = time.perf_counter() - total_start
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
        if not settings.openai_api_key:
            print("Error: OPENAI_API_KEY is required but not set")
            return False

        # Check DuckDB database exists
        if not Path(settings.duckdb_anatomic_path).exists():
            print(f"Error: DuckDB database not found at {settings.duckdb_anatomic_path}")
            print("Please ensure the anatomic locations database has been created")
            return False

        print("Configuration:")
        print(f"  OpenAI Model (small): {settings.openai_default_model_small}")
        print(f"  OpenAI Model (main): {settings.openai_default_model}")
        print(f"  DuckDB Database: {settings.duckdb_anatomic_path} (exists)")

        cohere_status = "Not configured"
        if settings.cohere_api_key:
            if settings.use_cohere_in_anatomic_location_search:
                cohere_status = "Configured and ENABLED for anatomic search"
            else:
                cohere_status = "Configured but DISABLED for anatomic search (default)"
        print(f"  Cohere API: {cohere_status}")
        print()
        return True

    except Exception as e:
        print(f"Configuration error: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test anatomic location search for medical findings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "pneumonia"
  %(prog)s "myocardial infarction" --verbose
  %(prog)s "brain tumor" -d "Malignant growth in brain tissue" -v
""",
    )

    parser.add_argument("finding_name", help="Name of the medical finding to search for")

    parser.add_argument("-d", "--description", help="Optional description of the finding")

    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output with timing for each stage")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging from underlying tools")

    return parser.parse_args()


async def main() -> None:
    """Main entry point for the anatomic location search tool."""
    args = parse_args()

    print("Anatomic Location Search Tool")
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
    await test_anatomic_location_search(args.finding_name, args.description, args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
