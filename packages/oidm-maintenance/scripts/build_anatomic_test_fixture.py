#!/usr/bin/env python3
"""Build anatomic-locations test database fixture with real pre-generated embeddings.

This script builds the DuckDB test database from sample anatomic location data
using pre-generated real embeddings (no API calls at build time). The resulting
database is committed to the repository for use by tests.

Usage:
    uv run python packages/oidm-maintenance/scripts/build_anatomic_test_fixture.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from oidm_maintenance.anatomic.build import build_anatomic_database
from oidm_maintenance.config import MaintenanceSettings
from pydantic import SecretStr
from rich.console import Console

console = Console()

REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Source data and embeddings (real, pre-generated)
SAMPLE_JSON = REPO_ROOT / "packages" / "anatomic-locations" / "tests" / "data" / "anatomic_sample.json"
SAMPLE_EMBEDDINGS = REPO_ROOT / "packages" / "anatomic-locations" / "tests" / "data" / "anatomic_sample_embeddings.json"

# Also used by oidm-maintenance tests (symlinked/copied data)
MAINTENANCE_SAMPLE_JSON = REPO_ROOT / "packages" / "oidm-maintenance" / "tests" / "data" / "anatomic_sample.json"
MAINTENANCE_SAMPLE_EMBEDDINGS = (
    REPO_ROOT / "packages" / "oidm-maintenance" / "tests" / "data" / "anatomic_sample_embeddings.json"
)

# Output
OUTPUT_DB = REPO_ROOT / "packages" / "anatomic-locations" / "tests" / "data" / "anatomic_test.duckdb"


def _load_embeddings_by_id(embeddings_path: Path) -> dict[str, list[float]]:
    """Load pre-generated embeddings keyed by record ID."""
    with open(embeddings_path) as f:
        return json.load(f)


def _load_sample_ids(sample_path: Path) -> list[str]:
    """Load record IDs from sample JSON to map searchable_texts back to IDs."""
    with open(sample_path) as f:
        records = json.load(f)
    return [r["_id"] for r in records]


async def build_anatomic_test_fixture() -> None:
    """Build the anatomic test DuckDB database using real pre-generated embeddings."""
    console.print("\n[bold blue]Building anatomic test fixture database[/bold blue]")
    console.print(f"  Source: {SAMPLE_JSON}")
    console.print(f"  Embeddings: {SAMPLE_EMBEDDINGS}")
    console.print(f"  Output: {OUTPUT_DB}\n")

    if not SAMPLE_JSON.exists():
        raise FileNotFoundError(f"Sample data not found: {SAMPLE_JSON}")
    if not SAMPLE_EMBEDDINGS.exists():
        raise FileNotFoundError(f"Sample embeddings not found: {SAMPLE_EMBEDDINGS}")

    # Load pre-generated real embeddings
    embeddings_by_id = _load_embeddings_by_id(SAMPLE_EMBEDDINGS)
    sample_ids = _load_sample_ids(SAMPLE_JSON)

    # Track which records have been processed to map searchable_texts → embeddings
    call_counter = 0

    def _mock_get_embeddings_batch(
        texts: list[str],
        *,
        api_key: str,
        model: str | None = None,
        dimensions: int = 512,
        cache: object | None = None,
    ) -> list[list[float]]:
        """Return real pre-generated embeddings in order matching the build pipeline."""
        nonlocal call_counter
        _ = (api_key, model, cache)

        # The build pipeline calls this once with all searchable_texts in the same
        # order as location_rows, which corresponds to the order of successful records.
        # We return embeddings by matching the order of sample_ids.
        result: list[list[float]] = []
        for i, _text in enumerate(texts):
            record_id = sample_ids[call_counter + i]
            if record_id in embeddings_by_id:
                result.append(embeddings_by_id[record_id])
            else:
                # Fallback: zero vector (shouldn't happen with aligned data)
                console.print(f"[yellow]Warning: No embedding for {record_id}, using zero vector[/yellow]")
                result.append([0.0] * dimensions)

        call_counter += len(texts)
        return result

    # Remove existing DB if present
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    # Build with real embeddings via mock (no API calls)
    settings = MaintenanceSettings(openai_api_key=SecretStr("fake-key-not-used"))
    with (
        patch("oidm_maintenance.anatomic.build.get_settings", return_value=settings),
        patch(
            "oidm_maintenance.anatomic.build.get_embeddings_batch",
            new_callable=AsyncMock,
            side_effect=_mock_get_embeddings_batch,
        ),
    ):
        db_path = await build_anatomic_database(
            source_json=SAMPLE_JSON,
            output_path=OUTPUT_DB,
            generate_embeddings=True,
        )

    console.print(f"\n[bold green]Test fixture built successfully:[/bold green] {db_path}")
    console.print("[dim]This database uses real pre-generated embeddings from anatomic_sample_embeddings.json[/dim]")
    console.print("Commit this file to the repository for use by tests.")


async def main() -> None:
    """Main entry point."""
    try:
        await build_anatomic_test_fixture()
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
