"""Build findingmodel database from source models.

This module extracts the database build logic from findingmodel.index.FindingModelIndex
and provides standalone functions for use by oidm-maintenance CLI tools.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import duckdb
from findingmodel.common import normalize_name
from findingmodel.contributor import Organization, Person
from findingmodel.finding_model import FindingModelFull
from oidm_common.duckdb import create_fts_index, create_hnsw_index, setup_duckdb_connection
from oidm_common.embeddings import get_embeddings_batch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from oidm_maintenance.config import get_settings

console = Console()
DEFAULT_CONTRIBUTOR_ROLE = "contributor"

# Schema statements copied from findingmodel.index.FindingModelIndex
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS finding_models (
        oifm_id VARCHAR PRIMARY KEY,
        slug_name VARCHAR NOT NULL UNIQUE,
        name VARCHAR NOT NULL UNIQUE,
        filename VARCHAR NOT NULL UNIQUE,
        file_hash_sha256 VARCHAR NOT NULL,
        description TEXT,
        search_text TEXT NOT NULL,
        embedding FLOAT[512] NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS people (
        github_username VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        email VARCHAR NOT NULL,
        organization_code VARCHAR,
        url VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS organizations (
        code VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        url VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_people (
        oifm_id VARCHAR NOT NULL,
        person_id VARCHAR NOT NULL,
        role VARCHAR NOT NULL DEFAULT 'contributor',
        display_order INTEGER,
        PRIMARY KEY (oifm_id, person_id, role)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_organizations (
        oifm_id VARCHAR NOT NULL,
        organization_id VARCHAR NOT NULL,
        role VARCHAR NOT NULL DEFAULT 'contributor',
        display_order INTEGER,
        PRIMARY KEY (oifm_id, organization_id, role)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS synonyms (
        oifm_id VARCHAR NOT NULL,
        synonym VARCHAR NOT NULL,
        PRIMARY KEY (oifm_id, synonym)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS attributes (
        attribute_id VARCHAR PRIMARY KEY,
        oifm_id VARCHAR NOT NULL,
        model_name VARCHAR NOT NULL,
        attribute_name VARCHAR NOT NULL,
        attribute_type VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tags (
        oifm_id VARCHAR NOT NULL,
        tag VARCHAR NOT NULL,
        PRIMARY KEY (oifm_id, tag)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS finding_model_json (
        oifm_id VARCHAR PRIMARY KEY,
        model_json TEXT NOT NULL
    )
    """,
)

# Index statements copied from findingmodel.index.FindingModelIndex
_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_finding_models_name ON finding_models(name)",
    "CREATE INDEX IF NOT EXISTS idx_finding_models_slug_name ON finding_models(slug_name)",
    "CREATE INDEX IF NOT EXISTS idx_finding_models_filename ON finding_models(filename)",
    "CREATE INDEX IF NOT EXISTS idx_synonyms_synonym ON synonyms(synonym)",
    "CREATE INDEX IF NOT EXISTS idx_synonyms_model ON synonyms(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)",
    "CREATE INDEX IF NOT EXISTS idx_tags_model ON tags(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_people_model ON model_people(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_people_person ON model_people(person_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_orgs_model ON model_organizations(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_orgs_org ON model_organizations(organization_id)",
    "CREATE INDEX IF NOT EXISTS idx_attributes_model ON attributes(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_attributes_name ON attributes(attribute_name)",
)


async def build_findingmodel_database(
    source_dir: Path,
    output_path: Path,
    generate_embeddings: bool = True,
) -> Path:
    """Build the findingmodel DuckDB database from .fm.json files.

    Args:
        source_dir: Directory containing .fm.json files.
        output_path: Path for output DuckDB file.
        generate_embeddings: Whether to generate OpenAI embeddings.

    Returns:
        Path to the created database file.

    Raises:
        ValueError: If source_dir is not a valid directory
        FileNotFoundError: If no .fm.json files found in source_dir
        RuntimeError: If embedding generation fails

    Example:
        from pathlib import Path
        from oidm_maintenance.findingmodel.build import build_findingmodel_database

        db_path = await build_findingmodel_database(
            source_dir=Path("models"),
            output_path=Path("index.duckdb"),
            generate_embeddings=True,
        )
    """
    if not source_dir.is_dir():
        raise ValueError(f"{source_dir} is not a valid directory")

    # Discover all .fm.json files
    fm_files = _discover_fm_files(source_dir)
    if not fm_files:
        raise FileNotFoundError(f"No .fm.json files found in {source_dir}")

    console.print(f"[bold blue]Building findingmodel database from {len(fm_files)} models[/bold blue]")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database if present
    if output_path.exists():
        output_path.unlink()
        console.print(f"[yellow]Removed existing database: {output_path}[/yellow]")

    settings = get_settings()

    # Create connection
    conn = setup_duckdb_connection(output_path, read_only=False)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Create tables
            task = progress.add_task("Creating database schema...", total=None)
            _create_tables(conn)
            progress.update(task, completed=1)

            # Load and validate models
            task = progress.add_task(f"Loading {len(fm_files)} models...", total=None)
            models_data = _load_models(fm_files)
            progress.update(task, completed=1)

            # Generate embeddings if requested
            if generate_embeddings:
                task = progress.add_task("Generating embeddings...", total=None)
                if not settings.openai_api_key:
                    raise RuntimeError("OPENAI_API_KEY not configured in environment")

                embedding_texts = [_build_embedding_text(model) for model, _, _, _, _ in models_data]
                raw_embeddings = await get_embeddings_batch(
                    list(embedding_texts),
                    api_key=settings.openai_api_key.get_secret_value(),
                    model=settings.openai_embedding_model,
                    dimensions=settings.openai_embedding_dimensions,
                )

                embeddings: list[list[float]] = []
                for i, embedding in enumerate(raw_embeddings):
                    if embedding is None:
                        raise RuntimeError(f"Failed to generate embedding for model at index {i}")
                    embeddings.append(embedding)
                progress.update(task, completed=1)
            else:
                # Create dummy embeddings (all zeros)
                console.print("[yellow]Skipping embedding generation (using zero vectors)[/yellow]")
                embeddings = [[0.0] * 512 for _ in models_data]

            # Insert models
            task = progress.add_task(f"Inserting {len(models_data)} models...", total=None)
            _insert_models(conn, models_data, embeddings)
            progress.update(task, completed=1)

            # Create indexes
            task = progress.add_task("Creating standard indexes...", total=None)
            _create_standard_indexes(conn)
            progress.update(task, completed=1)

            # Create search indexes (FTS and HNSW)
            task = progress.add_task("Creating search indexes...", total=None)
            _create_search_indexes(conn)
            progress.update(task, completed=1)

        console.print(f"[bold green]Database built successfully: {output_path}[/bold green]")

    finally:
        conn.close()

    return output_path


def _discover_fm_files(source_dir: Path) -> list[Path]:
    """Find all .fm.json files in source directory.

    Args:
        source_dir: Directory to search

    Returns:
        Sorted list of .fm.json file paths
    """
    return sorted(source_dir.glob("*.fm.json"))


def _load_models(
    fm_files: Sequence[Path],
) -> list[tuple[FindingModelFull, Path, str, str, str]]:
    """Load and validate finding models from files.

    Args:
        fm_files: List of .fm.json file paths

    Returns:
        List of (model, file_path, file_hash, search_text, json_text) tuples

    Raises:
        ValueError: If model validation fails
    """
    models_data: list[tuple[FindingModelFull, Path, str, str, str]] = []

    for file_path in fm_files:
        # Read JSON
        json_text = file_path.read_text(encoding="utf-8")

        # Parse and validate
        model = FindingModelFull.model_validate_json(json_text)

        # Calculate file hash
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        # Build search text
        search_text = _build_search_text(model)

        models_data.append((model, file_path, file_hash, search_text, json_text))

    return models_data


def _create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all database tables.

    Args:
        conn: Active database connection
    """
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)


def _create_standard_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Create standard B-tree indexes.

    Args:
        conn: Active database connection
    """
    for statement in _INDEX_STATEMENTS:
        conn.execute(statement)


def _create_search_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Create FTS and HNSW vector search indexes.

    Args:
        conn: Active database connection
    """
    # Create HNSW vector index
    create_hnsw_index(
        conn,
        table="finding_models",
        column="embedding",
        index_name="finding_models_embedding_hnsw",
        metric="l2sq",
    )

    # Create FTS index
    create_fts_index(
        conn,
        "finding_models",
        "oifm_id",
        "search_text",
        stemmer="porter",
        stopwords="english",
        lower=1,
        overwrite=True,
    )


def _insert_models(
    conn: duckdb.DuckDBPyConnection,
    models_data: Sequence[tuple[FindingModelFull, Path, str, str, str]],
    embeddings: Sequence[list[float]],
) -> None:
    """Insert all models and related data into database.

    Args:
        conn: Active database connection
        models_data: List of model tuples from _load_models
        embeddings: List of embedding vectors (one per model)
    """
    # Prepare row data
    model_rows: list[tuple[object, ...]] = []
    synonym_rows: list[tuple[str, str]] = []
    tag_rows: list[tuple[str, str]] = []
    attribute_rows: list[tuple[str, str, str, str, str]] = []
    people_rows_dict: dict[str, tuple[str, str, str, str, str | None]] = {}
    model_people_rows: list[tuple[str, str, str, int]] = []
    organization_rows_dict: dict[str, tuple[str, str, str | None]] = {}
    model_organization_rows: list[tuple[str, str, str, int]] = []
    json_rows: list[tuple[str, str]] = []

    for (model, file_path, file_hash, search_text, json_text), embedding in zip(models_data, embeddings, strict=True):
        # Main model row
        model_rows.append((
            model.oifm_id,
            normalize_name(model.name),
            model.name,
            file_path.name,
            file_hash,
            model.description,
            search_text,
            embedding,
        ))

        # JSON storage
        json_rows.append((model.oifm_id, json_text))

        # Synonyms (deduplicated)
        unique_synonyms = list(dict.fromkeys(model.synonyms or []))
        synonym_rows.extend((model.oifm_id, synonym) for synonym in unique_synonyms)

        # Tags (deduplicated)
        unique_tags = list(dict.fromkeys(model.tags or []))
        tag_rows.extend((model.oifm_id, tag) for tag in unique_tags)

        # Attributes
        attribute_rows.extend(
            (
                attribute.oifma_id,
                model.oifm_id,
                model.name,
                attribute.name,
                str(attribute.type),
            )
            for attribute in model.attributes
        )

        # Contributors (people and organizations)
        for order, contributor in enumerate(model.contributors or []):
            if isinstance(contributor, Person):
                people_rows_dict[contributor.github_username] = (
                    contributor.github_username,
                    contributor.name,
                    str(contributor.email),
                    contributor.organization_code,
                    str(contributor.url) if contributor.url else None,
                )
                model_people_rows.append((
                    model.oifm_id,
                    contributor.github_username,
                    DEFAULT_CONTRIBUTOR_ROLE,
                    order,
                ))
            elif isinstance(contributor, Organization):
                organization_rows_dict[contributor.code] = (
                    contributor.code,
                    contributor.name,
                    str(contributor.url) if contributor.url else None,
                )
                model_organization_rows.append((
                    model.oifm_id,
                    contributor.code,
                    DEFAULT_CONTRIBUTOR_ROLE,
                    order,
                ))

    # Insert all data
    _insert_batch(
        conn,
        model_rows,
        synonym_rows,
        tag_rows,
        attribute_rows,
        people_rows_dict,
        model_people_rows,
        organization_rows_dict,
        model_organization_rows,
        json_rows,
    )


def _insert_batch(
    conn: duckdb.DuckDBPyConnection,
    model_rows: Sequence[tuple[object, ...]],
    synonym_rows: Sequence[tuple[str, str]],
    tag_rows: Sequence[tuple[str, str]],
    attribute_rows: Sequence[tuple[str, str, str, str, str]],
    people_rows_dict: dict[str, tuple[str, str, str, str, str | None]],
    model_people_rows: Sequence[tuple[str, str, str, int]],
    organization_rows_dict: dict[str, tuple[str, str, str | None]],
    model_organization_rows: Sequence[tuple[str, str, str, int]],
    json_rows: Sequence[tuple[str, str]],
) -> None:
    """Execute batch inserts for all data.

    Args:
        conn: Active database connection
        model_rows: Main finding_models table rows
        synonym_rows: Synonyms table rows
        tag_rows: Tags table rows
        attribute_rows: Attributes table rows
        people_rows_dict: People table rows (deduplicated by github_username)
        model_people_rows: Model-person junction table rows
        organization_rows_dict: Organizations table rows (deduplicated by code)
        model_organization_rows: Model-organization junction table rows
        json_rows: JSON storage table rows
    """
    # Insert finding models
    if model_rows:
        conn.executemany(
            """
            INSERT INTO finding_models (
                oifm_id,
                slug_name,
                name,
                filename,
                file_hash_sha256,
                description,
                search_text,
                embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            model_rows,
        )

    # Insert synonyms
    if synonym_rows:
        conn.executemany("INSERT INTO synonyms (oifm_id, synonym) VALUES (?, ?)", synonym_rows)

    # Insert tags
    if tag_rows:
        conn.executemany("INSERT INTO tags (oifm_id, tag) VALUES (?, ?)", tag_rows)

    # Insert attributes
    if attribute_rows:
        conn.executemany(
            """
            INSERT INTO attributes (
                attribute_id,
                oifm_id,
                model_name,
                attribute_name,
                attribute_type
            ) VALUES (?, ?, ?, ?, ?)
            """,
            attribute_rows,
        )

    # Insert people
    people_rows = list(people_rows_dict.values())
    if people_rows:
        conn.executemany(
            """
            INSERT INTO people (
                github_username,
                name,
                email,
                organization_code,
                url
            ) VALUES (?, ?, ?, ?, ?)
            """,
            people_rows,
        )

    # Insert model-people relationships
    if model_people_rows:
        conn.executemany(
            "INSERT INTO model_people (oifm_id, person_id, role, display_order) VALUES (?, ?, ?, ?)",
            model_people_rows,
        )

    # Insert organizations
    organization_rows = list(organization_rows_dict.values())
    if organization_rows:
        conn.executemany(
            """
            INSERT INTO organizations (
                code,
                name,
                url
            ) VALUES (?, ?, ?)
            """,
            organization_rows,
        )

    # Insert model-organization relationships
    if model_organization_rows:
        conn.executemany(
            "INSERT INTO model_organizations (oifm_id, organization_id, role, display_order) VALUES (?, ?, ?, ?)",
            model_organization_rows,
        )

    # Insert JSON storage
    if json_rows:
        conn.executemany(
            """
            INSERT INTO finding_model_json (oifm_id, model_json)
            VALUES (?, ?)
            """,
            json_rows,
        )


def _build_search_text(model: FindingModelFull) -> str:
    """Build search text for FTS index.

    Args:
        model: The finding model

    Returns:
        Combined search text from name, description, synonyms, tags, and attributes
    """
    parts: list[str] = [model.name]
    if model.description:
        parts.append(model.description)
    if model.synonyms:
        parts.extend(model.synonyms)
    if model.tags:
        parts.extend(model.tags)
    parts.extend(attribute.name for attribute in model.attributes)
    return "\n".join(part for part in parts if part)


def _build_embedding_text(model: FindingModelFull) -> str:
    """Build text for embedding generation.

    Args:
        model: The finding model

    Returns:
        Formatted text for embedding with structured context
    """
    parts: list[str] = [model.name]
    if model.description:
        parts.append(model.description)
    if model.synonyms:
        parts.append("Synonyms: " + ", ".join(model.synonyms))
    if model.tags:
        parts.append("Tags: " + ", ".join(model.tags))
    attribute_lines = [
        f"Attribute {attribute.name}: {attribute.description or attribute.type}" for attribute in model.attributes
    ]
    parts.extend(attribute_lines)
    return "\n".join(part for part in parts if part)


def _verify_database(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Verify database integrity and return counts.

    Args:
        conn: Active database connection

    Returns:
        Dictionary with counts for each table

    Example:
        counts = _verify_database(conn)
        print(f"Models: {counts['finding_models']}")
    """
    tables = [
        "finding_models",
        "synonyms",
        "tags",
        "attributes",
        "people",
        "organizations",
        "model_people",
        "model_organizations",
        "finding_model_json",
    ]

    counts: dict[str, int] = {}
    for table in tables:
        result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = int(result[0]) if result else 0

    return counts


__all__ = ["build_findingmodel_database"]
