#!/usr/bin/env python
"""
Demo script for the BioOntology Search Tool.

This script demonstrates how to use the async BioOntology REST API search client
to find medical concepts and codes across multiple ontologies.

Usage:
    python notebooks/demo_bioontology_search.py
"""

import asyncio
import sys
import time

from findingmodel.config import settings
from findingmodel.tools.ontology_search import BioOntologySearchClient, BioOntologySearchResult


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def print_section(title: str) -> None:
    """Print a section title."""
    print(f"\n{title}")
    print("-" * len(title))


def display_results(results: list[BioOntologySearchResult], query: str, ontologies: list[str]) -> None:
    """Display search results in a readable format."""
    print_header(f"Results for: '{query}' in {', '.join(ontologies)}")
    print(f"Found {len(results)} results")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.pref_label}")
        print(f"   Ontology: {result.ontology}")
        print(f"   ID: {result.concept_id}")
        if result.definition:
            print(f"   Definition: {result.definition[:100]}...")
        if result.synonyms:
            print(f"   Synonyms: {', '.join(result.synonyms[:3])}")
        if result.semantic_types:
            print(f"   Semantic Types: {', '.join(result.semantic_types)}")
        print(f"   UI Link: {result.ui_link}")


async def demo_basic_search() -> None:
    """Demonstrate basic search functionality."""
    print_section("Basic Search: Pneumonia")

    async with BioOntologySearchClient() as client:
        results = await client.search_all_pages(query="pneumonia", ontologies=["SNOMEDCT", "RADLEX"], max_results=5)

    display_results(results, "pneumonia", ["SNOMEDCT", "RADLEX"])


async def demo_detailed_search() -> None:
    """Demonstrate detailed search with specific parameters."""
    print_section("Detailed Search: Hepatic Metastasis")

    async with BioOntologySearchClient() as client:
        search_results = await client.search(
            query="hepatic metastasis", ontologies=["SNOMEDCT", "RADLEX", "LOINC"], page_size=10, page=1
        )

    print(f"Query: {search_results.query}")
    print(f"Total Available: {search_results.total_count}")
    print(f"Pages: {search_results.page_count}")
    print(f"Current Page: {search_results.current_page}")
    print(f"Results on this page: {len(search_results.results)}")

    display_results(search_results.results[:5], "hepatic metastasis", ["SNOMEDCT", "RADLEX", "LOINC"])


async def demo_semantic_type_filter() -> None:
    """Demonstrate filtering by semantic type."""
    print_section("Semantic Type Filter: Liver Diseases (T047)")

    async with BioOntologySearchClient() as client:
        # Search for liver-related disease concepts (T047 = Disease or Syndrome)
        disease_results = await client.search(
            query="liver",
            ontologies=["SNOMEDCT"],
            page_size=10,
            semantic_types=["T047"],  # Disease or syndrome
        )

    print(f"Disease concepts found: {disease_results.total_count}")

    if disease_results.results:
        print("\nFirst 5 disease-related liver concepts:")
        for i, result in enumerate(disease_results.results[:5], 1):
            print(f"{i}. {result.pref_label} ({result.ontology})")


async def demo_multiple_ontologies() -> None:
    """Demonstrate searching across multiple ontologies."""
    print_section("Multi-Ontology Search: Fracture")

    # Search across all default ontologies
    async with BioOntologySearchClient() as client:
        results = await client.search_all_pages(query="fracture", max_results=20)

    # Group by ontology
    by_ontology = {}
    for result in results:
        if result.ontology not in by_ontology:
            by_ontology[result.ontology] = []
        by_ontology[result.ontology].append(result)

    print(f"Total results: {len(results)}")
    print("\nResults by ontology:")
    for ontology, ont_results in by_ontology.items():
        print(f"  {ontology}: {len(ont_results)} results")
        for result in ont_results[:2]:  # Show first 2 from each
            print(f"    - {result.pref_label}")


async def demo_pagination() -> None:
    """Demonstrate pagination handling."""
    print_section("Pagination: Getting Multiple Pages")

    async with BioOntologySearchClient() as client:
        # Get first page
        page1 = await client.search(query="cancer", ontologies=["SNOMEDCT"], page_size=5, page=1)

        print(f"Page 1: {len(page1.results)} results")
        print(f"Total available: {page1.total_count}")
        print(f"Total pages: {page1.page_count}")

        # Get second page if available
        if page1.page_count > 1:
            page2 = await client.search(query="cancer", ontologies=["SNOMEDCT"], page_size=5, page=2)
            print(f"Page 2: {len(page2.results)} results")

            # Show that results are different
            page1_ids = {r.concept_id for r in page1.results}
            page2_ids = {r.concept_id for r in page2.results}
            print(f"Unique results: {len(page1_ids.intersection(page2_ids)) == 0}")

        # Demonstrate automatic pagination
        print("\nAutomatic pagination (getting 15 results):")
        all_results = await client.search_all_pages(query="cancer", ontologies=["SNOMEDCT"], max_results=15)
        print(f"Retrieved {len(all_results)} results across multiple pages")


async def main() -> None:
    """Main demonstration function."""
    print_header("BioOntology Search Tool Demonstration")

    # Check for required configuration
    if not getattr(settings, "bioontology_api_key", None):
        print("\n⚠️  Warning: BIOONTOLOGY_API_KEY not configured.")
        print("Please set BIOONTOLOGY_API_KEY in your .env file.")
        print("\nYou can get an API key from: https://bioportal.bioontology.org/account")
        return

    print("\nThis demo will show various ways to search for medical concepts")
    print("using the BioOntology REST API.")

    # Track timing
    start_time = time.time()

    try:
        # Run demos
        await demo_basic_search()
        await demo_detailed_search()
        await demo_semantic_type_filter()
        await demo_multiple_ontologies()
        await demo_pagination()

        elapsed_time = time.time() - start_time

        print_header("Demonstration Complete")
        print(f"\nTotal demonstration time: {elapsed_time:.1f}s")
        print("\n✅ The BioOntology search tool successfully:")
        print("   - Searched for concepts across multiple ontologies")
        print("   - Retrieved detailed concept information")
        print("   - Filtered by semantic types")
        print("   - Handled pagination automatically")
        print("   - Provided direct links to BioPortal UI")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


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
