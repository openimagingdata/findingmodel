#!/usr/bin/env python3
"""Batch enrichment script for finding models.

Takes an input JSON file with finding names (like ipl_finding_models.json) and produces
an enriched output file with OIFM IDs as keys and full enrichment data.

Usage:
    python scripts/enrich_findings_batch.py scripts/ipl_finding_models.json scripts/enriched_findings.json
"""

import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

from findingmodel import logger
from findingmodel.index import DuckDBIndex
from findingmodel.tools.finding_enrichment import enrich_finding_unified

# Enable loguru logger
logger.enable("findingmodel")


def load_input_file(input_path: Path) -> dict:
    """Load the input JSON file with finding names."""
    with open(input_path) as f:
        return json.load(f)


def load_output_file(output_path: Path) -> dict:
    """Load existing output file if it exists, otherwise return empty dict."""
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return {}


def save_output_file(output_path: Path, data: dict) -> None:
    """Save the output JSON file."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def is_already_enriched(finding_name: str, existing_output: dict) -> str | None:
    """Check if a finding is already enriched in the output.

    Returns the OIFM ID if found, None otherwise.
    """
    for oifm_id, data in existing_output.items():
        if data.get("name") == finding_name:
            return oifm_id
    return None


async def enrich_single_finding(finding_name: str, index: DuckDBIndex) -> dict | None:
    """Enrich a single finding and return the formatted output.

    Returns None if the finding model doesn't exist in the index.
    """
    # First, get the existing model from the index
    entry = await index.get(finding_name)
    if not entry:
        logger.warning(f"No existing model found for '{finding_name}' - skipping")
        return None

    existing_model = await index.get_full(entry.oifm_id)

    # Run enrichment
    enrichment = await enrich_finding_unified(finding_name)

    # Build the output structure
    result = {
        "oifm_id": existing_model.oifm_id,
        "name": existing_model.name,
        "description": existing_model.description,
        "synonyms": list(existing_model.synonyms) if existing_model.synonyms else [],
        "attributes": [attr.name for attr in existing_model.attributes],
        # Enrichment data
        "ontology_codes": [
            {"system": code.system, "code": code.code, "display": code.display}
            for code in (enrichment.ontology_codes.exact_matches + enrichment.ontology_codes.should_include)
        ],
        "anatomic_location": {
            "id": enrichment.anatomic_location.location.concept_id,
            "text": enrichment.anatomic_location.location.concept_text,
        },
        "body_regions": list(enrichment.body_regions),
        "etiologies": enrichment.etiologies,
        "modalities": list(enrichment.modalities),
        "subspecialties": list(enrichment.subspecialties),
    }

    return result


async def main(input_path: Path, output_path: Path) -> None:
    """Process all findings from input file and write enriched output."""

    # Load files
    input_data = load_input_file(input_path)
    output_data = load_output_file(output_path)

    finding_names = list(input_data.keys())
    total = len(finding_names)

    logger.info(f"Processing {total} findings from {input_path}")
    logger.info(f"Output will be written to {output_path}")

    # Count already processed
    already_done = sum(1 for name in finding_names if is_already_enriched(name, output_data))
    logger.info(f"Already enriched: {already_done}/{total}")

    processed = 0
    failed = 0
    skipped = 0

    total_start = perf_counter()

    # Open index once for all lookups
    index = DuckDBIndex(read_only=True)
    async with index:
        for i, finding_name in enumerate(finding_names, 1):
            # Check if already enriched
            existing_id = is_already_enriched(finding_name, output_data)
            if existing_id:
                logger.debug(f"[{i}/{total}] Skipping '{finding_name}' - already enriched as {existing_id}")
                skipped += 1
                continue

            logger.info(f"[{i}/{total}] Enriching: {finding_name}")
            start = perf_counter()

            try:
                result = await enrich_single_finding(finding_name, index)

                if result:
                    oifm_id = result.pop("oifm_id")
                    output_data[oifm_id] = result
                    processed += 1

                    elapsed = perf_counter() - start
                    logger.info(f"  ✓ Completed in {elapsed:.1f}s -> {oifm_id}")

                    # Save after each successful enrichment (in case of interruption)
                    save_output_file(output_path, output_data)
                else:
                    failed += 1
                    logger.warning("  ✗ No model found")

            except Exception as e:
                failed += 1
                logger.error(f"  ✗ Failed: {e}")

            # Brief pause between API calls
            await asyncio.sleep(0.5)

    total_time = perf_counter() - total_start

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH ENRICHMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total findings: {total}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped (already done): {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time:.1f}s")
    if processed > 0:
        logger.info(f"Avg time per finding: {total_time / processed:.1f}s")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/enrich_findings_batch.py <input.json> <output.json>")
        print(
            "Example: python scripts/enrich_findings_batch.py scripts/ipl_finding_models.json scripts/enriched_findings.json"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    asyncio.run(main(input_path, output_path))
