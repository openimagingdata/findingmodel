"""Anatomic location database migration and management functions.

This module provides functions for building, validating, and managing
anatomic location databases with DuckDB.
"""

import json
from pathlib import Path
from typing import Any

import duckdb
import httpx
from openai import AsyncOpenAI

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.tools.duckdb_utils import (
    batch_embeddings_for_duckdb,
    create_fts_index,
    create_hnsw_index,
    setup_duckdb_connection,
)


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


async def load_anatomic_data(source: str) -> list[dict[str, Any]]:
    """Load anatomic location data from URL or file.

    Args:
        source: URL (starts with http:// or https://) or file path

    Returns:
        List of anatomic location records
    """
    if source.startswith("http://") or source.startswith("https://"):
        # Download from URL
        logger.info(f"Downloading data from {source}")
        async with httpx.AsyncClient() as client:
            response = await client.get(source, follow_redirects=True)
            response.raise_for_status()
            data = response.json()
    else:
        # Load from file
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        logger.info(f"Loading data from {source_path}")
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of records, got {type(data)}")

    logger.info(f"Loaded {len(data)} records")
    return data


def validate_anatomic_record(record: dict[str, Any]) -> list[str]:
    """Validate an anatomic location record.

    Args:
        record: Anatomic location record to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    if not record.get("_id"):
        errors.append("Missing required field: _id")

    if not record.get("description"):
        errors.append("Missing required field: description")

    # Validate synonyms field if present
    synonyms = record.get("synonyms")
    if synonyms is not None and not isinstance(synonyms, list):
        errors.append("Field 'synonyms' must be a list")

    return errors


async def create_anatomic_database(
    db_path: Path,
    records: list[dict[str, Any]],
    client: AsyncOpenAI,
    batch_size: int = 50,
) -> tuple[int, int]:
    """Create anatomic location database with embeddings and indexes.

    Args:
        db_path: Path to the database file to create
        records: List of anatomic location records
        client: OpenAI client for generating embeddings
        batch_size: Number of records to process per batch

    Returns:
        Tuple of (successful_count, failed_count)
    """
    logger.info(f"Creating database at {db_path}")

    # Create database with common utilities
    conn = setup_duckdb_connection(db_path, read_only=False)

    try:
        # Create the anatomic_locations table
        logger.info("Creating anatomic_locations table...")
        dimensions = settings.openai_embedding_dimensions
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS anatomic_locations (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                region TEXT,
                sided TEXT,
                synonyms TEXT[],
                definition TEXT,
                vector FLOAT[{dimensions}]
            )
        """)

        # Process and insert data
        successful, failed = await _process_and_insert_data(conn, records, client, batch_size)

        # Create indexes
        _create_indexes(conn, dimensions)

        # Verify and summarize
        _verify_database(conn)

        return successful, failed

    finally:
        conn.close()


async def _process_and_insert_data(
    conn: duckdb.DuckDBPyConnection,
    records: list[dict[str, Any]],
    client: AsyncOpenAI,
    batch_size: int,
) -> tuple[int, int]:
    """Process records and insert into database in batches.

    Args:
        conn: DuckDB connection
        records: List of anatomic location records
        client: OpenAI client for generating embeddings
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
            # Validate record
            errors = validate_anatomic_record(record)
            if errors:
                logger.warning(f"Record {i} validation errors: {errors}")
                failed += 1
                continue

            # Extract fields
            record_id = record["_id"]
            description = record["description"]
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
                await _insert_batch(conn, batch_records, batch_texts, client)
                successful += len(batch_records)
                logger.info(f"Inserted {successful}/{len(records)} records...")

                batch_records = []
                batch_texts = []

        except Exception as e:
            logger.error(f"Error processing record {i}: {e}")
            failed += 1

    # Insert remaining records
    if batch_records:
        await _insert_batch(conn, batch_records, batch_texts, client)
        successful += len(batch_records)

    # Commit all changes at once
    conn.commit()

    logger.info(f"Insertion complete: {successful} successful, {failed} failed")
    return successful, failed


async def _insert_batch(
    conn: duckdb.DuckDBPyConnection,
    batch_records: list[dict[str, Any]],
    batch_texts: list[str],
    client: AsyncOpenAI,
) -> None:
    """Insert a batch of records with embeddings.

    Args:
        conn: DuckDB connection
        batch_records: List of record dicts
        batch_texts: List of texts to embed
        client: OpenAI client
    """
    # Generate embeddings for batch using common utility
    logger.info(f"Generating embeddings for {len(batch_texts)} records...")
    embeddings = await batch_embeddings_for_duckdb(batch_texts, client=client)

    # Prepare data for insertion
    batch_data = []
    for rec, embedding in zip(batch_records, embeddings, strict=True):
        batch_data.append((
            rec["id"],
            rec["description"],
            rec["region"],
            rec["sided"],
            rec["synonyms"],
            rec["definition"],
            embedding,  # Will be None if embedding failed
        ))

    conn.executemany(
        """
        INSERT INTO anatomic_locations 
            (id, description, region, sided, synonyms, definition, vector)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        batch_data,
    )


def _create_indexes(conn: duckdb.DuckDBPyConnection, dimensions: int) -> None:
    """Create FTS, HNSW, and other indexes for efficient searching.

    Args:
        conn: DuckDB connection
        dimensions: Vector dimensions for HNSW index
    """
    logger.info("Creating indexes...")

    # Create FTS index on searchable text fields
    create_fts_index(
        conn,
        "anatomic_locations",
        "id",
        "description",
        "definition",
        stemmer="porter",
        stopwords="english",
        lower=0,
        overwrite=True,
    )

    # Create HNSW index for vector similarity search (optional, will fall back to brute force)
    try:
        create_hnsw_index(
            conn,
            table="anatomic_locations",
            column="vector",
            index_name="idx_anatomic_hnsw",
            metric="cosine",
            ef_construction=128,
            ef_search=64,
            m=16,
        )
    except Exception:
        # Utility logged the specific error; continuing without index
        logger.info("Anatomic location search will continue without HNSW index")

    # Create standard indexes
    logger.info("Creating standard indexes...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_region ON anatomic_locations(region)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sided ON anatomic_locations(sided)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_description ON anatomic_locations(description)")

    conn.commit()


def _verify_database(conn: duckdb.DuckDBPyConnection) -> None:
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
    vector_result = conn.execute("""
        SELECT COUNT(*)
        FROM anatomic_locations
        WHERE vector IS NOT NULL
    """).fetchone()
    vector_count = vector_result[0] if vector_result else 0
    logger.info(f"Records with vectors: {vector_count}/{total}")


def get_database_stats(db_path: Path) -> dict[str, Any]:
    """Get statistics about an anatomic location database.

    Args:
        db_path: Path to the database file

    Returns:
        Dictionary with database statistics
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        # Get counts
        result = conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()
        total_records = result[0] if result else 0

        result = conn.execute("SELECT COUNT(*) FROM anatomic_locations WHERE vector IS NOT NULL").fetchone()
        vector_count = result[0] if result else 0

        # Get region count
        result = conn.execute(
            "SELECT COUNT(DISTINCT region) FROM anatomic_locations WHERE region IS NOT NULL"
        ).fetchone()
        region_count = result[0] if result else 0

        # Get sided distribution
        sided_dist = conn.execute("""
            SELECT sided, COUNT(*) as count 
            FROM anatomic_locations 
            GROUP BY sided 
            ORDER BY count DESC
        """).fetchall()

        return {
            "total_records": total_records,
            "records_with_vectors": vector_count,
            "unique_regions": region_count,
            "sided_distribution": dict(sided_dist),
            "file_size_mb": db_path.stat().st_size / (1024 * 1024),
        }

    finally:
        conn.close()
