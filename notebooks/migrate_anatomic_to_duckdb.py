#!/usr/bin/env python
"""Migrate anatomic locations from JSON to DuckDB database.

This script loads anatomic location data from a JSON file and creates a DuckDB
database with full-text search and vector similarity search capabilities.

Usage:
    uv run notebooks/migrate_anatomic_to_duckdb.py
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import duckdb
from openai import AsyncOpenAI

from findingmodel.tools.common import get_embeddings_batch

logger = logging.getLogger(__name__)


def create_searchable_text(record: dict[str, Any]) -> str:
    """Create a searchable text representation of an anatomic location record.

    Combines description, synonyms, and definition for embedding.

    Args:
        record: Anatomic location record

    Returns:
        Combined text for embedding
    """
    parts = []

    # Always include description
    if desc := record.get("description"):
        parts.append(desc)

    # Add synonyms if available
    if (synonyms := record.get("synonyms")) and isinstance(synonyms, list) and synonyms:
        parts.append(f"also known as: {', '.join(synonyms[:5])}")  # Limit to 5 synonyms

    # Add definition if available (truncate if too long)
    if definition := record.get("definition"):
        # Truncate definition to first 200 chars to keep embeddings focused
        if len(definition) > 200:
            definition = definition[:197] + "..."
        parts.append(definition)

    return " | ".join(parts)


def determine_sided(record: dict[str, Any]) -> str:
    """Determine the 'sided' value based on ref properties.

    Logic:
    - If has leftRef AND rightRef: "generic"
    - If has leftRef only: "left"
    - If has rightRef only: "right"
    - If has unsidedRef only: "unsided"
    - Otherwise: "nonlateral" (instead of NULL)

    Args:
        record: JSON record with potential ref properties

    Returns:
        Sided value (never None)
    """
    has_left = "leftRef" in record
    has_right = "rightRef" in record
    has_unsided = "unsidedRef" in record

    if has_left and has_right:
        return "generic"
    elif has_left and not has_right:
        return "left"
    elif has_right and not has_left:
        return "right"
    elif has_unsided and not has_left and not has_right:
        return "unsided"
    else:
        return "nonlateral"  # Default for items with no laterality


def create_database(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create DuckDB database with required extensions and schema.

    Args:
        db_path: Path to the database file

    Returns:
        DuckDB connection object
    """
    logger.info(f"Creating database at {db_path}")

    conn = duckdb.connect(str(db_path))

    # Install and load required extensions
    logger.info("Installing DuckDB extensions...")
    conn.execute("INSTALL fts")
    conn.execute("LOAD fts")
    conn.execute("INSTALL vss")
    conn.execute("LOAD vss")

    # Enable experimental persistence for HNSW index
    conn.execute("SET hnsw_enable_experimental_persistence = true")

    # Create the anatomic_locations table with FLOAT[512] for text-embedding-3-small
    logger.info("Creating anatomic_locations table...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS anatomic_locations (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            region TEXT,
            sided TEXT,
            synonyms TEXT[],
            definition TEXT,
            vector FLOAT[512]
        )
    """)

    return conn


def load_json_data(json_path: Path) -> list[dict[str, Any]]:
    """Load and validate JSON data from file.

    Args:
        json_path: Path to the JSON file

    Returns:
        List of anatomic location records
    """
    logger.info(f"Loading data from {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of records, got {type(data)}")

    logger.info(f"Loaded {len(data)} records")
    return data


async def process_and_insert_data(
    conn: duckdb.DuckDBPyConnection,
    records: list[dict[str, Any]],
    openai_client: AsyncOpenAI,
    batch_size: int = 50,  # Smaller batches for embedding API
) -> tuple[int, int]:
    """Process records and insert into database in batches.

    Args:
        conn: DuckDB connection
        records: List of anatomic location records
        openai_client: OpenAI client for generating embeddings
        batch_size: Number of records to insert per batch

    Returns:
        Tuple of (successful_count, failed_count)
    """
    logger.info(f"Processing {len(records)} records...")

    successful = 0
    failed = 0
    batch_records = []
    batch_texts = []

    for i, record in enumerate(records, 1):
        try:
            # Extract and validate required fields
            record_id = record.get("_id")
            if not record_id:
                logger.warning(f"Record {i} missing _id, skipping")
                failed += 1
                continue

            description = record.get("description", "")
            if not description:
                logger.warning(f"Record {record_id} missing description, skipping")
                failed += 1
                continue

            # Extract other fields
            region = record.get("region")
            sided = determine_sided(record)
            synonyms = record.get("synonyms", [])
            definition = record.get("definition")

            # Create searchable text for embedding
            searchable_text = create_searchable_text(record)

            # Add to batch
            batch_records.append({
                "id": record_id,
                "description": description,
                "region": region,
                "sided": sided,
                "synonyms": synonyms if synonyms else [],
                "definition": definition,
            })
            batch_texts.append(searchable_text)

            # Process batch when full
            if len(batch_records) >= batch_size:
                # Generate embeddings for batch
                logger.info(f"Generating embeddings for batch {successful // batch_size + 1}...")
                embeddings = await get_embeddings_batch(batch_texts, openai_client)

                # Prepare data for insertion
                batch_data = []
                for rec, embedding in zip(batch_records, embeddings, strict=False):
                    batch_data.append((
                        rec["id"],
                        rec["description"],
                        rec["region"],
                        rec["sided"],
                        rec["synonyms"],
                        rec["definition"],
                        embedding,  # Will be None if embedding failed
                    ))

                _insert_batch(conn, batch_data)
                successful += len(batch_data)
                logger.info(f"Inserted {successful}/{len(records)} records...")

                batch_records = []
                batch_texts = []

        except Exception as e:
            logger.error(f"Error processing record {i}: {e}")
            failed += 1

    # Insert remaining records
    if batch_records:
        # Generate embeddings for final batch
        logger.info("Generating embeddings for final batch...")
        embeddings = await get_embeddings_batch(batch_texts, openai_client)

        # Prepare data for insertion
        batch_data = []
        for rec, embedding in zip(batch_records, embeddings, strict=False):
            batch_data.append((
                rec["id"],
                rec["description"],
                rec["region"],
                rec["sided"],
                rec["synonyms"],
                rec["definition"],
                embedding,
            ))

        _insert_batch(conn, batch_data)
        successful += len(batch_data)

    # Commit all changes at once
    conn.commit()

    logger.info(f"Insertion complete: {successful} successful, {failed} failed")
    return successful, failed


def _insert_batch(conn: duckdb.DuckDBPyConnection, batch: list[tuple]) -> None:
    """Insert a batch of records into the database.

    Args:
        conn: DuckDB connection
        batch: List of record tuples
    """
    conn.executemany(
        """
        INSERT INTO anatomic_locations 
            (id, description, region, sided, synonyms, definition, vector)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )


def create_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Create FTS, HNSW, and other indexes for efficient searching.

    Args:
        conn: DuckDB connection
    """
    logger.info("Creating indexes...")

    # Create FTS index on searchable text fields
    logger.info("Creating FTS index...")
    conn.execute("""
        PRAGMA create_fts_index(
            'anatomic_locations', 
            'id',
            'description', 
            'definition',
            overwrite=1
        )
    """)

    # Create HNSW index for vector similarity search
    # This is done AFTER data loading for better performance
    logger.info("Creating HNSW index for vector similarity search...")
    try:
        conn.execute("""
            CREATE INDEX idx_anatomic_hnsw 
            ON anatomic_locations 
            USING HNSW (vector)
            WITH (metric = 'cosine', ef_construction = 128, ef_search = 64, M = 16)
        """)
        logger.info("HNSW index created successfully")
    except Exception as e:
        logger.warning(f"Could not create HNSW index: {e}")
        logger.warning("Vector search will use brute force instead of index")

    # Create standard indexes
    logger.info("Creating standard indexes...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_region ON anatomic_locations(region)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sided ON anatomic_locations(sided)")

    # Create indexes for exact matching
    logger.info("Creating indexes for exact matching...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_description ON anatomic_locations(description)")
    # For synonyms array, we can't directly index, but we can use GIN-like functionality if available
    # or rely on the FTS index for synonym searching

    conn.commit()


def verify_database(conn: duckdb.DuckDBPyConnection) -> None:
    """Verify database contents and print summary statistics.

    Args:
        conn: DuckDB connection
    """
    logger.info("Verifying database...")

    # Get total count
    result = conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()
    total = result[0] if result else 0
    logger.info(f"Total records: {total}")

    # Get sided distribution
    sided_dist = conn.execute("""
        SELECT sided, COUNT(*) as count 
        FROM anatomic_locations 
        GROUP BY sided 
        ORDER BY count DESC
    """).fetchall()

    logger.info("Sided distribution:")
    for sided, count in sided_dist:
        sided_label = sided if sided else "NULL"
        logger.info(f"  {sided_label}: {count}")

    # Get region distribution (top 10)
    region_dist = conn.execute("""
        SELECT region, COUNT(*) as count 
        FROM anatomic_locations 
        WHERE region IS NOT NULL
        GROUP BY region 
        ORDER BY count DESC
        LIMIT 10
    """).fetchall()

    logger.info("Top 10 regions:")
    for region, count in region_dist:
        logger.info(f"  {region}: {count}")

    # Check vector completeness
    vector_count = conn.execute("""
        SELECT COUNT(*) 
        FROM anatomic_locations 
        WHERE vector IS NOT NULL
    """).fetchone()[0]
    logger.info(f"Records with vectors: {vector_count}/{total}")

    # Sample FTS search
    logger.info("\nSample FTS search for 'lung':")
    results = conn.execute("""
        SELECT id, description, fts_main_anatomic_locations.match_bm25(id, 'lung') as score
        FROM anatomic_locations
        WHERE fts_main_anatomic_locations.match_bm25(id, 'lung') IS NOT NULL
        ORDER BY score DESC
        LIMIT 3
    """).fetchall()

    for id, desc, score in results:
        logger.info(f"  {id}: {desc[:50]}... (score: {score:.2f})")


async def main() -> None:
    """Main migration function."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Define paths
    project_root = Path(__file__).parent.parent
    json_path = project_root / "notebooks" / "data" / "anatomic_locations.json"
    db_path = project_root / "data" / "anatomic_locations.duckdb"

    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if database already exists
    if db_path.exists():
        logger.warning(f"Database already exists at {db_path}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != "y":
            logger.info("Migration cancelled")
            return
        db_path.unlink()

    # Check for OpenAI API key
    from findingmodel.config import settings

    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY not configured in settings")
        return

    try:
        # Create OpenAI client for embeddings
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

        # Load JSON data
        records = load_json_data(json_path)

        # Create database
        conn = create_database(db_path)

        # Process and insert data with embeddings
        successful, failed = await process_and_insert_data(conn, records, openai_client)

        # Create indexes
        create_indexes(conn)

        # Verify and summarize
        verify_database(conn)

        # Close connection
        conn.close()

        logger.info("\n✅ Migration complete!")
        logger.info(f"Database location: {db_path}")
        logger.info(f"Records migrated: {successful}")
        if failed > 0:
            logger.warning(f"Records failed: {failed}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
