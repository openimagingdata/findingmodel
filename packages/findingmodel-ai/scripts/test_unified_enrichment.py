#!/usr/bin/env python3
"""Test script for unified enrichment approach.

Tests the enrich_finding_unified() function on 5 sample findings from the IPL dataset
and compares results against ground truth body regions.

This script measures:
- Timing for each finding
- Quality of classifications (ontology codes, anatomic locations, body regions, etc.)
- Accuracy of body_regions against ground truth from ipl_finding_models.json

Output is saved to scripts/enrichment_comparison/unified_results.json for comparison
against the original 5-call pipeline results.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from findingmodel import logger
from findingmodel_ai.tools.finding_enrichment import enrich_finding_unified

# Enable loguru logger (disabled by default in __init__.py)
logger.enable("findingmodel")
logger.enable("findingmodel_ai")

# Test findings - same 5 used in original comparison
TEST_FINDINGS = [
    "pulmonary segmental consolidation",
    "calcification in a tendon",
    "abnormal intracranial enhancement",
    "thoracic spine degenerative change",
    "enotosis",
]


def load_ground_truth() -> dict[str, list[str] | str]:
    """Load ground truth body regions from ipl_finding_models.json.

    Returns:
        Dict mapping finding names to their ground truth body regions.
        Body regions may be a list of strings or the string "ALL".
    """
    script_dir = Path(__file__).parent
    ground_truth_path = script_dir / "ipl_finding_models.json"

    with open(ground_truth_path) as f:
        data = json.load(f)

    # Extract body regions for each finding
    ground_truth: dict[str, list[str] | str] = {}
    for finding, info in data.items():
        regions = info.get("regions", [])
        ground_truth[finding] = regions

    return ground_truth


def compare_body_regions(predicted: list[str], ground_truth: list[str] | str) -> dict[str, Any]:
    """Compare predicted body regions against ground truth.

    Args:
        predicted: List of predicted body regions
        ground_truth: Ground truth regions (list or "ALL")

    Returns:
        Dict with comparison metrics:
        - match: True if predicted matches ground truth exactly
        - predicted: Predicted regions
        - ground_truth: Ground truth regions
        - missing: Regions in ground truth but not predicted
        - extra: Regions predicted but not in ground truth
    """
    # Normalize ground truth to list
    if ground_truth == "ALL":
        gt_set = {"ALL"}
    else:
        gt_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}

    pred_set = set(predicted)

    # Exact match check
    match = pred_set == gt_set

    # Calculate missing and extra
    missing = list(gt_set - pred_set)
    extra = list(pred_set - gt_set)

    return {
        "match": match,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "missing": missing,
        "extra": extra,
    }


async def test_finding(finding_name: str, ground_truth: dict[str, list[str] | str]) -> dict[str, Any]:
    """Test unified enrichment on a single finding.

    Args:
        finding_name: Name of the finding to test
        ground_truth: Ground truth body regions mapping

    Returns:
        Dict with test results including timing and classification data
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Testing: {finding_name}")
    logger.info(f"{'=' * 80}")

    start_time = perf_counter()

    try:
        result = await enrich_finding_unified(finding_name)
        elapsed_time = perf_counter() - start_time

        logger.info(f"✓ Completed in {elapsed_time:.1f}s")

        # Extract ontology codes (combined SNOMED + RadLex)
        exact_codes = result.ontology_codes.exact_matches
        should_include_codes = result.ontology_codes.should_include

        ontology_codes = [
            {"system": code.system, "code": code.code, "display": code.display}
            for code in exact_codes + should_include_codes
        ]

        # Extract anatomic location (single location now)
        location = result.anatomic_location.location
        anatomic_location = {"id": location.concept_id, "text": location.concept_text}

        # Compare body regions against ground truth
        gt_regions = ground_truth.get(finding_name, [])
        region_comparison = compare_body_regions(list(result.body_regions), gt_regions)

        # Log comparison results
        if region_comparison["match"]:
            logger.info(f"✓ Body regions match ground truth: {result.body_regions}")
        else:
            logger.warning("✗ Body regions mismatch:")
            logger.warning(f"  Predicted: {result.body_regions}")
            logger.warning(f"  Ground truth: {gt_regions}")
            if region_comparison["missing"]:
                logger.warning(f"  Missing: {region_comparison['missing']}")
            if region_comparison["extra"]:
                logger.warning(f"  Extra: {region_comparison['extra']}")

        return {
            "finding": finding_name,
            "time": round(elapsed_time, 1),
            "ontology_codes": ontology_codes,
            "anatomic_location": anatomic_location,
            "body_regions": result.body_regions,
            "etiologies": result.etiologies,
            "modalities": result.modalities,
            "subspecialties": result.subspecialties,
            "reasoning": result.reasoning,
            "body_region_comparison": region_comparison,
        }

    except Exception as e:
        elapsed_time = perf_counter() - start_time
        logger.error(f"✗ Failed after {elapsed_time:.1f}s: {e}")

        return {
            "finding": finding_name,
            "time": round(elapsed_time, 1),
            "error": str(e),
            "ontology_codes": [],
            "anatomic_location": None,
            "body_regions": [],
            "etiologies": [],
            "modalities": [],
            "subspecialties": [],
            "reasoning": "",
            "body_region_comparison": {
                "match": False,
                "predicted": [],
                "ground_truth": ground_truth.get(finding_name, []),
                "missing": [],
                "extra": [],
            },
        }


async def main() -> None:
    """Run unified enrichment tests on all sample findings."""
    logger.info("Starting unified enrichment test suite")
    logger.info(f"Testing {len(TEST_FINDINGS)} findings")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Load ground truth
    ground_truth = load_ground_truth()
    logger.info(f"Loaded ground truth for {len(ground_truth)} findings")

    # Test each finding
    results = []
    total_start = perf_counter()

    for finding in TEST_FINDINGS:
        result = await test_finding(finding, ground_truth)
        results.append(result)

        # Brief pause between tests to avoid rate limiting
        await asyncio.sleep(1)

    total_time = perf_counter() - total_start

    # Calculate summary statistics
    successful_tests = [r for r in results if "error" not in r]
    failed_tests = [r for r in results if "error" in r]
    matching_regions = [r for r in results if r["body_region_comparison"]["match"]]

    logger.info(f"\n{'=' * 80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Average time per finding: {total_time / len(TEST_FINDINGS):.1f}s")
    logger.info(f"Successful tests: {len(successful_tests)}/{len(TEST_FINDINGS)}")
    logger.info(f"Failed tests: {len(failed_tests)}/{len(TEST_FINDINGS)}")
    logger.info(
        f"Body region accuracy: {len(matching_regions)}/{len(TEST_FINDINGS)} "
        f"({100 * len(matching_regions) / len(TEST_FINDINGS):.1f}%)"
    )

    # Save results to JSON
    output_dir = Path(__file__).parent / "enrichment_comparison"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "unified_results.json"

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_time": round(total_time, 1),
            "avg_time": round(total_time / len(TEST_FINDINGS), 1),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "body_region_accuracy": round(100 * len(matching_regions) / len(TEST_FINDINGS), 1),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Print body region comparison table
    logger.info(f"\n{'=' * 80}")
    logger.info("BODY REGION COMPARISON")
    logger.info(f"{'=' * 80}")
    for result in results:
        finding = result["finding"]
        comparison = result["body_region_comparison"]
        status = "✓" if comparison["match"] else "✗"

        logger.info(f"\n{status} {finding}")
        logger.info(f"  Predicted:     {comparison['predicted']}")
        logger.info(f"  Ground Truth:  {comparison['ground_truth']}")
        if not comparison["match"]:
            if comparison["missing"]:
                logger.info(f"  Missing:       {comparison['missing']}")
            if comparison["extra"]:
                logger.info(f"  Extra:         {comparison['extra']}")


if __name__ == "__main__":
    asyncio.run(main())
