#!/usr/bin/env python3
"""Build test database fixtures with mocked embeddings.

This script builds a DuckDB test database from sample finding models using
deterministic hash-based embeddings (no API calls). The resulting database
is committed to the repository for use by tests.

Usage:
    uv run python packages/oidm-maintenance/scripts/build_test_fixtures.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from oidm_maintenance.config import MaintenanceSettings
from oidm_maintenance.findingmodel.build import build_findingmodel_database
from pydantic import SecretStr
from rich.console import Console

console = Console()


async def _fake_embedding_deterministic(
    text: str,
    *,
    client: object | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:
    """Generate deterministic fake embedding based on text hash.

    This matches the pattern used in test_duckdb_index.py to ensure
    consistent behavior between the pre-built fixture and tests.

    Args:
        text: Text to embed
        client: Ignored (for API compatibility)
        model: Ignored (for API compatibility)
        dimensions: Target embedding dimensions (default 512)

    Returns:
        Deterministic embedding vector based on text hash
    """
    _ = (client, model)  # Explicitly ignore unused parameters
    target_dims = dimensions or 512
    await asyncio.sleep(0)  # Yield control to event loop
    # Use simple hash-based embedding for determinism
    hash_val = sum(ord(c) for c in text)
    return [(hash_val % 100) / 100.0] * target_dims


async def _fake_get_embeddings_batch(
    texts: list[str],
    *,
    api_key: str,
    model: str | None = None,
    dimensions: int = 512,
    cache: object | None = None,
) -> list[list[float]]:
    """Generate deterministic fake embeddings matching get_embeddings_batch signature.

    This function replaces the real get_embeddings_batch in the build module
    to avoid requiring API keys during test fixture generation.

    Args:
        texts: List of texts to embed
        api_key: Ignored (for API compatibility)
        model: Ignored (for API compatibility)
        dimensions: Target embedding dimensions (default 512)
        cache: Ignored (for API compatibility)

    Returns:
        List of deterministic embedding vectors
    """
    _ = (api_key, model, cache)
    return [await _fake_embedding_deterministic(text, dimensions=dimensions) for text in texts]


async def build_test_findingmodel_database() -> None:
    """Build findingmodel test database with deterministic embeddings.

    This function:
    1. Patches the embedding generation to use deterministic hash-based vectors
    2. Builds the database from test model definitions
    3. Creates HNSW and FTS indexes
    4. Saves the result as a committed test fixture

    The resulting database can be used by tests without requiring API calls.
    """
    # Define paths
    repo_root = Path(__file__).parent.parent.parent.parent
    source_dir = repo_root / "packages" / "findingmodel" / "tests" / "data" / "defs"
    output_path = repo_root / "packages" / "findingmodel" / "tests" / "data" / "test_index.duckdb"

    console.print("\n[bold blue]Building test database with mocked embeddings[/bold blue]")
    console.print(f"Source: {source_dir}")
    console.print(f"Output: {output_path}\n")

    # Verify source directory exists
    if not source_dir.exists():
        console.print(f"[bold red]Error: Source directory not found: {source_dir}[/bold red]")
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Patch the embedding generation function
    # This bypasses the API key check and generates deterministic embeddings
    with (
        patch(
            "oidm_maintenance.findingmodel.build.get_settings",
            return_value=MaintenanceSettings(openai_api_key=SecretStr("fake-key")),
        ),
        patch(
            "oidm_maintenance.findingmodel.build.get_embeddings_batch",
            new_callable=AsyncMock,
            side_effect=_fake_get_embeddings_batch,
        ),
    ):
        # Build database with mocked embeddings
        db_path = await build_findingmodel_database(
            source_dir=source_dir,
            output_path=output_path,
            generate_embeddings=True,  # Use mocked embeddings
        )

    console.print(f"\n[bold green]Test database built successfully:[/bold green] {db_path}")
    console.print("\n[yellow]Note:[/yellow] This database uses deterministic hash-based embeddings.")
    console.print("Commit this file to the repository for use by tests.")


async def main() -> None:
    """Main entry point for the script."""
    try:
        await build_test_findingmodel_database()
    except Exception as e:
        console.print(f"\n[bold red]Error building test database:[/bold red] {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
