"""Build anatomic-locations database from source data.

This module provides functions for creating the anatomic location DuckDB database
from source JSON data, including embedding generation and index creation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import httpx
from loguru import logger
from oidm_common.duckdb import create_fts_index, create_hnsw_index, setup_duckdb_connection
from oidm_common.embeddings import generate_embeddings_batch
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from oidm_maintenance.config import get_settings

if TYPE_CHECKING:
    from openai import AsyncOpenAI

console = Console()


# Column type definitions for bulk JSON loading
SYNONYM_COLUMNS = {
    "location_id": "VARCHAR",
    "synonym": "VARCHAR",
}

CODE_COLUMNS = {
    "location_id": "VARCHAR",
    "system": "VARCHAR",
    "code": "VARCHAR",
    "display": "VARCHAR",
}


def _get_location_columns(dimensions: int = 512) -> dict[str, str]:
    """Get column type definitions for anatomic_locations table.

    Args:
        dimensions: Embedding vector dimensions (default: 512 for text-embedding-3-small)

    Returns:
        Dictionary mapping column names to DuckDB type strings
    """
    return {
        "id": "VARCHAR",
        "description": "VARCHAR",
        "region": "VARCHAR",
        "location_type": "VARCHAR",
        "body_system": "VARCHAR",
        "structure_type": "VARCHAR",
        "laterality": "VARCHAR",
        "definition": "VARCHAR",
        "sex_specific": "VARCHAR",
        "search_text": "VARCHAR",
        "vector": f"FLOAT[{dimensions}]",
        "containment_path": "VARCHAR",
        "containment_parent_id": "VARCHAR",
        "containment_parent_display": "VARCHAR",
        "containment_depth": "INTEGER",
        "containment_children": "STRUCT(id VARCHAR, display VARCHAR)[]",
        "partof_path": "VARCHAR",
        "partof_parent_id": "VARCHAR",
        "partof_parent_display": "VARCHAR",
        "partof_depth": "INTEGER",
        "partof_children": "STRUCT(id VARCHAR, display VARCHAR)[]",
        "left_id": "VARCHAR",
        "left_display": "VARCHAR",
        "right_id": "VARCHAR",
        "right_display": "VARCHAR",
        "generic_id": "VARCHAR",
        "generic_display": "VARCHAR",
    }


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


def determine_laterality(record: dict[str, Any]) -> str:
    """Determine the laterality value based on ref properties.

    Logic:
    - If has leftRef AND rightRef: "generic"
    - If has leftRef only: "left"
    - If has rightRef only: "right"
    - If has unsidedRef only: "unsided" (maps to generic)
    - Otherwise: "nonlateral"

    Args:
        record: JSON record with potential ref properties

    Returns:
        Laterality value (never None)
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
        return "generic"  # unsided is a variant of generic
    else:
        return "nonlateral"


def compute_containment_path(record: dict[str, Any], records_by_id: dict[str, dict[str, Any]]) -> str:
    """Compute materialized path for containment hierarchy.

    Traverses containedByRef chain to root and builds path.

    Args:
        record: Current record
        records_by_id: Dictionary of all records by ID

    Returns:
        Path string like "/RID1/RID46/RID2660/"
    """
    path_parts: list[str] = []
    current = record
    current_id = record["_id"]
    visited: set[str] = {current_id}  # Start with self to detect self-references

    while current.get("containedByRef"):
        parent_id = current["containedByRef"]["id"]

        # Detect cycles (including self-references)
        if parent_id in visited:
            # Self-reference (parent_id == current_id) is normal for root nodes - don't warn
            # Only warn for true cycles back to a non-self ancestor
            if parent_id != current_id:
                logger.warning(f"Circular containment reference at {parent_id} for {record['_id']}")
            break
        visited.add(parent_id)

        path_parts.insert(0, parent_id)

        # Look up parent
        if parent_id not in records_by_id:
            logger.warning(f"Parent {parent_id} not found for {record['_id']}")
            break
        current = records_by_id[parent_id]
        current_id = parent_id

    # Add self at the end
    path_parts.append(record["_id"])

    return "/" + "/".join(path_parts) + "/"


def compute_partof_path(record: dict[str, Any], records_by_id: dict[str, dict[str, Any]]) -> str:
    """Compute materialized path for part-of hierarchy.

    Traverses partOfRef chain to root and builds path.

    Args:
        record: Current record
        records_by_id: Dictionary of all records by ID

    Returns:
        Path string like "/RID1/RID46/RID2660/"
    """
    path_parts: list[str] = []
    current = record
    current_id = record["_id"]
    visited: set[str] = {current_id}  # Start with self to detect self-references

    while current.get("partOfRef"):
        parent_id = current["partOfRef"]["id"]

        # Detect cycles (including self-references)
        if parent_id in visited:
            # Self-reference (parent_id == current_id) is normal for root nodes - don't warn
            # Only warn for true cycles back to a non-self ancestor
            if parent_id != current_id:
                logger.warning(f"Circular partOf reference at {parent_id} for {record['_id']}")
            break
        visited.add(parent_id)

        path_parts.insert(0, parent_id)

        # Look up parent
        if parent_id not in records_by_id:
            logger.warning(f"Part-of parent {parent_id} not found for {record['_id']}")
            break
        current = records_by_id[parent_id]
        current_id = parent_id

    # Add self at the end
    path_parts.append(record["_id"])

    return "/" + "/".join(path_parts) + "/"


def extract_codes(record: dict[str, Any]) -> list[dict[str, str | None]]:
    """Extract all codes from a record.

    Extracts from:
    - codes field (list of {system, code, display?})
    - snomedId/snomedDisplay
    - acrCommonId

    Args:
        record: Anatomic location record

    Returns:
        List of code dictionaries with system, code, display
    """
    codes = []

    # Extract from codes array
    if "codes" in record and isinstance(record["codes"], list):
        for code_obj in record["codes"]:
            if isinstance(code_obj, dict) and "system" in code_obj and "code" in code_obj:
                codes.append({
                    "system": code_obj["system"],
                    "code": code_obj["code"],
                    "display": code_obj.get("display"),
                })

    # Extract SNOMED
    if "snomedId" in record:
        codes.append({
            "system": "SNOMED",
            "code": record["snomedId"],
            "display": record.get("snomedDisplay"),
        })

    # Extract ACR
    if "acrCommonId" in record:
        codes.append({
            "system": "ACR",
            "code": record["acrCommonId"],
            "display": None,
        })

    return codes


def build_children_struct(refs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert a list of refs to a list of dicts for STRUCT[] storage.

    Args:
        refs: List of {id, display} dictionaries

    Returns:
        List of {id, display} dicts (empty list if no valid refs)
    """
    if not refs or not isinstance(refs, list):
        return []

    children = []
    for ref in refs:
        if isinstance(ref, dict) and "id" in ref and "display" in ref:
            children.append({"id": ref["id"], "display": ref["display"]})
    return children


def _bulk_load_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    data: list[dict[str, Any]],
    column_types: dict[str, str],
) -> int:
    """Bulk load data into table via temp JSON file.

    Args:
        conn: DuckDB connection
        table_name: Target table name
        data: List of row dicts
        column_types: Map of column name to DuckDB type string

    Returns:
        Number of rows inserted
    """
    if not data:
        return 0

    # Create temp file for JSON data - using context manager
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as temp_file:
        temp_path = temp_file.name
        # Write data to temp file
        json.dump(data, temp_file, ensure_ascii=False)

    try:
        # Build column specification for read_json
        # Complex types like FLOAT[512] and STRUCT(...)[] need to be quoted
        columns_spec = ", ".join(f"{name}: '{dtype}'" for name, dtype in column_types.items())
        column_names = ", ".join(column_types.keys())

        # Execute bulk insert via read_json
        # Specify column names explicitly so DB defaults are used for created_at, updated_at
        sql = f"""
            INSERT INTO {table_name} ({column_names})
            SELECT * FROM read_json('{temp_path}', columns={{{columns_spec}}})
        """
        conn.execute(sql)

        return len(data)

    finally:
        # Clean up temp file
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {temp_path}: {e}")


async def load_anatomic_data(source: str | Path) -> list[dict[str, Any]]:
    """Load anatomic location data from URL or file.

    Args:
        source: URL (starts with http:// or https://) or file path

    Returns:
        List of anatomic location records (deduplicated by _id)
    """
    source_str = str(source)

    if source_str.startswith("http://") or source_str.startswith("https://"):
        # Download from URL
        logger.info(f"Downloading data from {source_str}")
        async with httpx.AsyncClient() as client:
            response = await client.get(source_str, follow_redirects=True)
            response.raise_for_status()
            data = response.json()
    else:
        # Load from file
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        logger.info(f"Loading data from {source_path}")
        with open(source_path, encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of records, got {type(data)}")

    # Deduplicate by _id (keep first occurrence)
    seen_ids: set[str] = set()
    unique_records: list[dict[str, Any]] = []
    duplicates = 0
    for record in data:
        record_id = record.get("_id")
        if record_id and record_id not in seen_ids:
            seen_ids.add(record_id)
            unique_records.append(record)
        elif record_id:
            duplicates += 1
            logger.debug(f"Skipping duplicate record ID: {record_id}")

    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate record IDs in source data (skipped)")

    logger.info(f"Loaded {len(unique_records)} unique records")
    return unique_records


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


def _create_schema(conn: duckdb.DuckDBPyConnection, dimensions: int = 512) -> None:
    """Create the 4-table schema for anatomic locations.

    Args:
        conn: DuckDB connection
        dimensions: Embedding vector dimensions (default: 512)
    """
    # Table 1: anatomic_locations (main table with pre-computed hierarchy)
    conn.execute(f"""
        CREATE TABLE anatomic_locations (
            -- Core identity
            id VARCHAR PRIMARY KEY,
            description VARCHAR NOT NULL,

            -- Classification
            region VARCHAR,
            location_type VARCHAR NOT NULL DEFAULT 'structure',
            body_system VARCHAR,
            structure_type VARCHAR,
            laterality VARCHAR NOT NULL DEFAULT 'nonlateral',

            -- Text fields
            definition TEXT,
            sex_specific VARCHAR,
            search_text TEXT NOT NULL,
            vector FLOAT[{dimensions}] NOT NULL,

            -- Pre-computed CONTAINMENT hierarchy (materialized path)
            containment_path VARCHAR,
            containment_parent_id VARCHAR,
            containment_parent_display VARCHAR,
            containment_depth INTEGER,
            containment_children STRUCT(id VARCHAR, display VARCHAR)[],

            -- Pre-computed PART-OF hierarchy (materialized path)
            partof_path VARCHAR,
            partof_parent_id VARCHAR,
            partof_parent_display VARCHAR,
            partof_depth INTEGER,
            partof_children STRUCT(id VARCHAR, display VARCHAR)[],

            -- Pre-computed LATERALITY references
            left_id VARCHAR,
            left_display VARCHAR,
            right_id VARCHAR,
            right_display VARCHAR,
            generic_id VARCHAR,
            generic_display VARCHAR,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 2: anatomic_synonyms
    conn.execute("""
        CREATE TABLE anatomic_synonyms (
            location_id VARCHAR NOT NULL,
            synonym VARCHAR NOT NULL,
            PRIMARY KEY (location_id, synonym)
        )
    """)

    # Table 3: anatomic_codes
    conn.execute("""
        CREATE TABLE anatomic_codes (
            location_id VARCHAR NOT NULL,
            system VARCHAR NOT NULL,
            code VARCHAR NOT NULL,
            display VARCHAR,
            PRIMARY KEY (location_id, system, code)
        )
    """)

    # Table 4: anatomic_references
    conn.execute("""
        CREATE TABLE anatomic_references (
            location_id VARCHAR NOT NULL,
            url VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            description VARCHAR,
            content VARCHAR,
            published_date VARCHAR,
            accessed_date DATE,
            PRIMARY KEY (location_id, url)
        )
    """)


def _prepare_record_for_insert(
    record: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    record_num: int,
) -> tuple[dict[str, Any], str, list[tuple[str, str]], list[tuple[str, str, str, str | None]]] | None:
    """Prepare a single record for database insertion.

    Args:
        record: Anatomic location record
        records_by_id: Dictionary of all records by ID for path computation
        record_num: Record number for logging

    Returns:
        Tuple of (record_data, searchable_text, synonyms, codes) or None if validation fails
    """
    # Validate record
    errors = validate_anatomic_record(record)
    if errors:
        logger.warning(f"Record {record_num} validation errors: {errors}")
        return None

    # Extract core fields
    record_id = record["_id"]
    description = record["description"]
    region = record.get("region")
    laterality = determine_laterality(record)
    definition = record.get("definition")
    sex_specific = record.get("sex_specific")

    # Create searchable text for embedding
    searchable_text = create_searchable_text(record)

    # Compute containment hierarchy
    containment_path = compute_containment_path(record, records_by_id)
    containment_parent_id = record.get("containedByRef", {}).get("id")
    containment_parent_display = record.get("containedByRef", {}).get("display")
    containment_depth = containment_path.count("/") - 2  # Subtract leading and trailing /
    containment_children = build_children_struct(record.get("containsRefs", []))

    # Compute part-of hierarchy
    partof_path = compute_partof_path(record, records_by_id)
    partof_parent_id = record.get("partOfRef", {}).get("id")
    partof_parent_display = record.get("partOfRef", {}).get("display")
    partof_depth = partof_path.count("/") - 2
    partof_children = build_children_struct(record.get("hasPartsRefs", []))

    # Extract laterality references
    left_id = record.get("leftRef", {}).get("id")
    left_display = record.get("leftRef", {}).get("display")
    right_id = record.get("rightRef", {}).get("id")
    right_display = record.get("rightRef", {}).get("display")
    generic_id = record.get("unsidedRef", {}).get("id")
    generic_display = record.get("unsidedRef", {}).get("display")

    # Build record data
    record_data = {
        "id": record_id,
        "description": description,
        "region": region,
        "location_type": "structure",  # Default, can be enhanced later
        "body_system": None,  # To be populated later
        "structure_type": None,  # To be populated later
        "laterality": laterality,
        "definition": definition,
        "sex_specific": sex_specific,
        "search_text": searchable_text,
        "containment_path": containment_path,
        "containment_parent_id": containment_parent_id,
        "containment_parent_display": containment_parent_display,
        "containment_depth": containment_depth,
        "containment_children": containment_children,
        "partof_path": partof_path,
        "partof_parent_id": partof_parent_id,
        "partof_parent_display": partof_parent_display,
        "partof_depth": partof_depth,
        "partof_children": partof_children,
        "left_id": left_id,
        "left_display": left_display,
        "right_id": right_id,
        "right_display": right_display,
        "generic_id": generic_id,
        "generic_display": generic_display,
    }

    # Collect synonyms (deduplicated)
    seen_synonyms: set[str] = set()
    synonyms: list[tuple[str, str]] = []
    raw_synonyms = record.get("synonyms", [])
    if raw_synonyms and isinstance(raw_synonyms, list):
        for synonym in raw_synonyms:
            if synonym not in seen_synonyms:
                seen_synonyms.add(synonym)
                synonyms.append((record_id, synonym))

    # Collect codes (deduplicated by system+code)
    seen_codes: set[tuple[str, str]] = set()
    codes: list[tuple[str, str, str, str | None]] = []
    for code in extract_codes(record):
        system, code_val = code["system"], code["code"]
        # Skip codes with missing system or code
        if system is None or code_val is None:
            continue
        code_key = (system, code_val)
        if code_key not in seen_codes:
            seen_codes.add(code_key)
            codes.append((record_id, system, code_val, code.get("display")))

    return record_data, searchable_text, synonyms, codes


def _prepare_all_records(
    records: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]], list[dict[str, str | None]]]:
    """Prepare all records for bulk insertion.

    Args:
        records: List of anatomic location records
        records_by_id: Dictionary of all records by ID for path computation

    Returns:
        Tuple of (location_rows, searchable_texts, synonym_rows, code_rows)
    """
    location_rows: list[dict[str, Any]] = []
    searchable_texts: list[str] = []
    synonym_rows: list[dict[str, str]] = []
    code_rows: list[dict[str, str | None]] = []

    failed_count = 0

    for i, record in enumerate(records, 1):
        try:
            # Validate and prepare record
            processed = _prepare_record_for_insert(record, records_by_id, i)
            if processed is None:
                failed_count += 1
                continue

            record_data, searchable_text, synonyms, codes = processed

            # Build location row without vector (will be added later)
            location_row = dict(record_data)
            location_rows.append(location_row)
            searchable_texts.append(searchable_text)

            # Collect synonyms as dicts
            for location_id, synonym in synonyms:
                synonym_rows.append({"location_id": location_id, "synonym": synonym})

            # Collect codes as dicts
            for location_id, system, code, display in codes:
                code_rows.append({
                    "location_id": location_id,
                    "system": system,
                    "code": code,
                    "display": display,
                })

        except Exception as e:
            logger.error(f"Error processing record {i}: {e}")
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"Failed to prepare {failed_count} records")

    return location_rows, searchable_texts, synonym_rows, code_rows


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

    # Create standard indexes on main table
    logger.info("Creating standard indexes on anatomic_locations...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_region ON anatomic_locations(region)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_location_type ON anatomic_locations(location_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_body_system ON anatomic_locations(body_system)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_structure_type ON anatomic_locations(structure_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_laterality ON anatomic_locations(laterality)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_containment_parent ON anatomic_locations(containment_parent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_containment_path ON anatomic_locations(containment_path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_partof_parent ON anatomic_locations(partof_parent_id)")

    # Create indexes on synonym table
    logger.info("Creating indexes on anatomic_synonyms...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_synonyms_synonym ON anatomic_synonyms(synonym)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_synonyms_location ON anatomic_synonyms(location_id)")

    # Create indexes on codes table
    logger.info("Creating indexes on anatomic_codes...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_codes_system ON anatomic_codes(system)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_codes_lookup ON anatomic_codes(system, code)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_codes_location ON anatomic_codes(location_id)")

    # Create indexes on references table
    logger.info("Creating indexes on anatomic_references...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_location ON anatomic_references(location_id)")

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

    # Get laterality distribution
    laterality_dist = conn.execute("""
        SELECT laterality, COUNT(*) as count
        FROM anatomic_locations
        GROUP BY laterality
        ORDER BY count DESC
    """).fetchall()

    logger.info("Laterality distribution:")
    for laterality, count in laterality_dist:
        laterality_label = laterality if laterality else "NULL"
        logger.info(f"  {laterality_label}: {count}")

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

    # Check synonym count
    synonym_result = conn.execute("SELECT COUNT(*) FROM anatomic_synonyms").fetchone()
    synonym_count = synonym_result[0] if synonym_result else 0
    logger.info(f"Total synonyms: {synonym_count}")

    # Check code count
    code_result = conn.execute("SELECT COUNT(*) FROM anatomic_codes").fetchone()
    code_count = code_result[0] if code_result else 0
    logger.info(f"Total codes: {code_count}")

    # Check code system distribution
    code_dist = conn.execute("""
        SELECT system, COUNT(*) as count
        FROM anatomic_codes
        GROUP BY system
        ORDER BY count DESC
    """).fetchall()

    logger.info("Code system distribution:")
    for system, count in code_dist:
        logger.info(f"  {system}: {count}")


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

        # Get laterality distribution
        laterality_dist = conn.execute("""
            SELECT laterality, COUNT(*) as count
            FROM anatomic_locations
            GROUP BY laterality
            ORDER BY count DESC
        """).fetchall()

        # Get synonym and code counts
        result = conn.execute("SELECT COUNT(*) FROM anatomic_synonyms").fetchone()
        synonym_count = result[0] if result else 0

        result = conn.execute("SELECT COUNT(*) FROM anatomic_codes").fetchone()
        code_count = result[0] if result else 0

        return {
            "total_records": total_records,
            "records_with_vectors": vector_count,
            "unique_regions": region_count,
            "laterality_distribution": dict(laterality_dist),
            "total_synonyms": synonym_count,
            "total_codes": code_count,
            "file_size_mb": db_path.stat().st_size / (1024 * 1024),
        }

    finally:
        conn.close()


async def create_anatomic_database(
    db_path: Path,
    records: list[dict[str, Any]],
    client: AsyncOpenAI,
    batch_size: int = 50,
    dimensions: int = 1536,
) -> tuple[int, int]:
    """Create anatomic location database from in-memory records.

    This is a test utility that creates a database from pre-loaded records
    rather than loading from a file. Used by test fixtures.

    Args:
        db_path: Path to the database file to create
        records: List of anatomic location records
        client: OpenAI client for generating embeddings (AsyncOpenAI)
        batch_size: Number of records to embed per batch
        dimensions: Embedding vector dimensions

    Returns:
        Tuple of (successful_count, failed_count)
    """
    conn = setup_duckdb_connection(db_path, read_only=False)

    try:
        # Drop existing tables
        conn.execute("DROP TABLE IF EXISTS anatomic_references")
        conn.execute("DROP TABLE IF EXISTS anatomic_codes")
        conn.execute("DROP TABLE IF EXISTS anatomic_synonyms")
        conn.execute("DROP TABLE IF EXISTS anatomic_locations")

        # Create schema
        _create_schema(conn, dimensions)

        # Build record index
        records_by_id = {record["_id"]: record for record in records}

        # Prepare records
        location_rows, searchable_texts, synonym_rows, code_rows = _prepare_all_records(records, records_by_id)

        successful = len(location_rows)
        failed = len(records) - successful

        if successful == 0:
            return 0, failed

        # Generate embeddings
        embeddings: list[list[float] | None] = []
        for i in range(0, len(searchable_texts), batch_size):
            batch = searchable_texts[i : i + batch_size]
            batch_embeddings = await generate_embeddings_batch(
                batch, client, model="text-embedding-3-small", dimensions=dimensions
            )
            embeddings.extend(batch_embeddings)

        # Merge embeddings
        for location_row, embedding in zip(location_rows, embeddings, strict=True):
            location_row["vector"] = embedding

        # Bulk load
        _bulk_load_table(conn, "anatomic_locations", location_rows, _get_location_columns(dimensions))
        _bulk_load_table(conn, "anatomic_synonyms", synonym_rows, SYNONYM_COLUMNS)
        _bulk_load_table(conn, "anatomic_codes", code_rows, CODE_COLUMNS)

        conn.commit()
        _create_indexes(conn, dimensions)
        _verify_database(conn)

        return successful, failed

    finally:
        conn.close()


async def build_anatomic_database(
    source_json: str | Path,
    output_path: Path,
    generate_embeddings: bool = True,
    batch_size: int = 50,
) -> Path:
    """Build the anatomic-locations DuckDB database.

    Args:
        source_json: Path to source JSON with anatomic location data (or URL)
        output_path: Path for output DuckDB file
        generate_embeddings: Whether to generate OpenAI embeddings
        batch_size: Number of records to embed per batch (for rate limiting)

    Returns:
        Path to the created database file
    """
    settings = get_settings()

    # Determine dimensions from settings
    dimensions = settings.openai_embedding_dimensions

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Load data
        load_task = progress.add_task("Loading anatomic data...", total=None)
        records = await load_anatomic_data(source_json)
        progress.update(load_task, completed=1, total=1)

        # Create database connection
        logger.info(f"Creating database at {output_path}")
        conn = setup_duckdb_connection(output_path, read_only=False)

        try:
            # Drop existing tables if they exist (idempotent migration)
            schema_task = progress.add_task("Creating schema...", total=4)
            logger.info("Dropping existing tables if present...")
            conn.execute("DROP TABLE IF EXISTS anatomic_references")
            progress.advance(schema_task)
            conn.execute("DROP TABLE IF EXISTS anatomic_codes")
            progress.advance(schema_task)
            conn.execute("DROP TABLE IF EXISTS anatomic_synonyms")
            progress.advance(schema_task)
            conn.execute("DROP TABLE IF EXISTS anatomic_locations")
            progress.advance(schema_task)

            # Create the 4-table schema
            logger.info("Creating new schema...")
            _create_schema(conn, dimensions)
            progress.remove_task(schema_task)

            # Build index of all records by ID for path computation
            index_task = progress.add_task("Building record index...", total=None)
            logger.info("Building record index...")
            records_by_id = {record["_id"]: record for record in records}
            progress.update(index_task, completed=1, total=1)
            progress.remove_task(index_task)

            # Prepare all records for insertion
            prepare_task = progress.add_task(f"Preparing {len(records)} records...", total=len(records))
            logger.info(f"Preparing {len(records)} records...")
            location_rows, searchable_texts, synonym_rows, code_rows = _prepare_all_records(records, records_by_id)
            progress.update(prepare_task, completed=len(records))
            progress.remove_task(prepare_task)

            successful = len(location_rows)
            failed = len(records) - successful

            if successful == 0:
                logger.warning("No valid records to insert")
                return output_path

            # Generate embeddings if requested
            if generate_embeddings:
                if not settings.openai_api_key:
                    raise ValueError("OpenAI API key required for embedding generation (OIDM_MAINTAIN_OPENAI_API_KEY)")

                # Import OpenAI lazily for build
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

                embed_task = progress.add_task(
                    f"Generating embeddings ({len(searchable_texts)} records)...",
                    total=len(searchable_texts),
                )

                embeddings: list[list[float] | None] = []
                for i in range(0, len(searchable_texts), batch_size):
                    batch = searchable_texts[i : i + batch_size]
                    logger.info(f"Generating embeddings for batch {i // batch_size + 1} ({len(batch)} records)...")
                    batch_embeddings = await generate_embeddings_batch(
                        batch,
                        client,
                        model=settings.openai_embedding_model,
                        dimensions=dimensions,
                    )
                    embeddings.extend(batch_embeddings)
                    progress.update(embed_task, completed=i + len(batch))

                progress.remove_task(embed_task)

                # Merge embeddings into location rows
                logger.info("Merging embeddings into location data...")
                for location_row, embedding in zip(location_rows, embeddings, strict=True):
                    location_row["vector"] = embedding
            else:
                # If no embeddings, fill with zero vectors
                logger.warning("Skipping embedding generation - database will have zero vectors")
                zero_vector = [0.0] * dimensions
                for location_row in location_rows:
                    location_row["vector"] = zero_vector

            # Bulk load all tables
            load_task = progress.add_task("Loading data into database...", total=3)
            logger.info("Bulk loading anatomic_locations table...")
            _bulk_load_table(conn, "anatomic_locations", location_rows, _get_location_columns(dimensions))
            progress.advance(load_task)

            logger.info(f"Bulk loading {len(synonym_rows)} synonyms...")
            _bulk_load_table(conn, "anatomic_synonyms", synonym_rows, SYNONYM_COLUMNS)
            progress.advance(load_task)

            logger.info(f"Bulk loading {len(code_rows)} codes...")
            _bulk_load_table(conn, "anatomic_codes", code_rows, CODE_COLUMNS)
            progress.advance(load_task)

            # Commit all changes
            conn.commit()
            logger.info(f"Successfully inserted {successful} records")
            progress.remove_task(load_task)

            # Create indexes after data load
            index_task = progress.add_task("Creating indexes...", total=None)
            _create_indexes(conn, dimensions)
            progress.update(index_task, completed=1, total=1)
            progress.remove_task(index_task)

            # Verify and summarize
            verify_task = progress.add_task("Verifying database...", total=None)
            _verify_database(conn)
            progress.update(verify_task, completed=1, total=1)
            progress.remove_task(verify_task)

            console.print(f"[green]Successfully created database: {output_path}")
            console.print(f"[green]Loaded {successful} records ({failed} failed)")

            return output_path

        finally:
            conn.close()
