#!/usr/bin/env python3
"""Search BioOntology.org for RadLex and SNOMED codes for an anatomic term.

Usage:
    uv run --env-file .env python .claude/skills/manage-anatomic-locations/scripts/bioontology_lookup.py "hilum of lung"

Requires BIOONTOLOGY_API_KEY in .env or environment.
"""

from __future__ import annotations

import os
import re
import sys

import httpx

API_BASE_URL = "https://data.bioontology.org"
ONTOLOGIES = ["RADLEX", "SNOMEDCT"]
INCLUDE_FIELDS = "prefLabel,synonym,definition,semanticType"

# Patterns to detect SNOMED SEP triad type
SEP_PATTERNS = {
    "Structure": re.compile(r"^Structure of\b", re.IGNORECASE),
    "Entire": re.compile(r"^Entire\b", re.IGNORECASE),
    "Part": re.compile(r"\bpart\b", re.IGNORECASE),
}


def classify_snomed_sep(label: str) -> str:
    """Classify a SNOMED label by its SEP triad type."""
    if SEP_PATTERNS["Structure"].search(label):
        return "Structure"
    if SEP_PATTERNS["Entire"].search(label):
        return "Entire"
    if SEP_PATTERNS["Part"].search(label):
        return "Part"
    return "Other"


def extract_radlex_rid(concept_id: str) -> str | None:
    """Extract RID from a RadLex concept URI."""
    match = re.search(r"(RID\d+)", concept_id)
    return match.group(1) if match else None


def extract_snomed_code(concept_id: str) -> str | None:
    """Extract numeric SNOMED code from a concept URI."""
    match = re.search(r"/(\d+)$", concept_id)
    return match.group(1) if match else None


def format_result(item: dict, ontology: str) -> None:
    """Print a formatted result for a single BioOntology hit."""
    pref_label = item.get("prefLabel", "")
    concept_id = item.get("@id", "")
    synonyms = item.get("synonym", [])
    definition_raw = item.get("definition", [])

    # Handle definition as list or string
    if isinstance(definition_raw, list):
        definition = definition_raw[0] if definition_raw else None
    elif isinstance(definition_raw, str):
        definition = definition_raw
    else:
        definition = None

    if ontology == "RADLEX":
        rid = extract_radlex_rid(concept_id)
        print(f"  RadLex: {rid or concept_id}")
        print(f"  Label: {pref_label}")
    elif ontology == "SNOMEDCT":
        code = extract_snomed_code(concept_id)
        sep_type = classify_snomed_sep(pref_label)

        # Flag SEP type
        if sep_type == "Structure":
            flag = "  ** PREFERRED (Structure concept) **"
        elif sep_type == "Entire":
            flag = "  !! AVOID (Entire concept) !!"
        elif sep_type == "Part":
            flag = "  !! AVOID (Part concept) !!"
        else:
            flag = ""

        print(f"  SNOMED: {code or concept_id}{flag}")
        print(f"  Label: {pref_label}")
        if sep_type != "Other":
            print(f"  SEP Type: {sep_type}")
    else:
        print(f"  ID: {concept_id}")
        print(f"  Label: {pref_label}")

    if synonyms:
        print(f"  Synonyms: {', '.join(synonyms[:8])}")
        if len(synonyms) > 8:
            print(f"    ... and {len(synonyms) - 8} more")
    if definition:
        print(f"  Definition: {definition[:200]}")
        if len(definition) > 200:
            print(f"    ... ({len(definition)} chars)")


def search_ontology(client: httpx.Client, api_key: str, query: str, ontology: str) -> list[dict]:
    """Search a single ontology and return raw results."""
    params = {
        "q": query,
        "ontologies": ontology,
        "pagesize": 20,
        "page": 1,
        "include": INCLUDE_FIELDS,
    }
    headers = {"Authorization": f"apikey token={api_key}"}

    response = client.get(f"{API_BASE_URL}/search", params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    return data.get("collection", [])


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python bioontology_lookup.py <search_term>")
        print('Example: python bioontology_lookup.py "hilum of lung"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    api_key = os.environ.get("BIOONTOLOGY_API_KEY", "")

    if not api_key:
        print("Error: BIOONTOLOGY_API_KEY not set.")
        print("Run with: uv run --env-file .env python <script> <term>")
        sys.exit(1)

    print(f"Searching BioOntology for: {query!r}")
    print("=" * 60)

    with httpx.Client(timeout=30) as client:
        for ontology in ONTOLOGIES:
            print(f"\n--- {ontology} ---\n")
            try:
                results = search_ontology(client, api_key, query, ontology)
            except httpx.HTTPError as e:
                print(f"  Error querying {ontology}: {e}")
                continue

            if not results:
                print("  No results found.")
                continue

            for i, item in enumerate(results, 1):
                print(f"  [{i}]")
                format_result(item, ontology)
                print()


if __name__ == "__main__":
    main()
