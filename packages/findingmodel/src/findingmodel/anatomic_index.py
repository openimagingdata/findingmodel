"""Index for looking up and navigating anatomic locations.

Provides high-level query interface over DuckDB anatomic locations database.
All returned AnatomicLocation objects are automatically bound to the index.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import duckdb
from oidm_common.duckdb import setup_duckdb_connection
from oidm_common.models import IndexCode, WebReference

from findingmodel.anatomic_location import (
    AnatomicLocation,
    AnatomicRef,
    AnatomicRegion,
    BodySystem,
    Laterality,
    LocationType,
    StructureType,
)

if TYPE_CHECKING:
    from typing_extensions import Self


class AnatomicLocationIndex:
    """Index for looking up and navigating anatomic locations.

    Wraps DuckDB connection and provides high-level navigation API.
    Uses pre-computed materialized paths for instant hierarchy queries.

    All returned AnatomicLocation objects are automatically bound to this index
    via weakref, allowing navigation methods to be called without passing the index.

    Usage:
        # Context manager (CLI/scripts)
        with AnatomicLocationIndex() as index:
            location = index.get("RID2772")
            ancestors = location.get_containment_ancestors()  # Uses bound index

        # Explicit open/close (FastAPI lifespan)
        index = AnatomicLocationIndex()
        index.open()
        # ... use index ...
        index.close()
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the index with optional custom database path.

        Args:
            db_path: Optional path to DuckDB file. If not provided,
                     uses ensure_anatomic_db() to locate/download standard database.
        """
        if db_path is not None:
            self.db_path = Path(db_path).expanduser()
        else:
            from findingmodel.config import ensure_anatomic_db

            self.db_path = ensure_anatomic_db()

        self.conn: duckdb.DuckDBPyConnection | None = None

    def open(self) -> Self:
        """Open the database connection explicitly.

        For FastAPI lifespan pattern. Returns self for chaining.

        Returns:
            Self for method chaining
        """
        if self.conn is not None:
            return self  # Already open
        self.conn = setup_duckdb_connection(self.db_path, read_only=True)
        return self

    def close(self) -> None:
        """Close the database connection explicitly."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> Self:
        """Enter context manager, opening connection.

        Returns:
            Self for context manager pattern
        """
        return self.open()

    def __exit__(self, *_args: object) -> None:
        """Exit context manager, closing connection."""
        self.close()

    # =========================================================================
    # Core Lookups (all methods auto-bind returned objects)
    # =========================================================================

    def get(self, location_id: str) -> AnatomicLocation:
        """Get a single anatomic location by ID.

        The returned object is automatically bound to this index.

        Args:
            location_id: RID identifier (e.g., "RID2772")

        Returns:
            AnatomicLocation bound to this index

        Raises:
            KeyError: If location_id not found
        """
        conn = self._ensure_connection()
        row = conn.execute(
            "SELECT * FROM anatomic_locations WHERE id = ?",
            [location_id],
        ).fetchone()

        if not row:
            raise KeyError(f"Anatomic location not found: {location_id}")

        return self._row_to_location(row)

    def find_by_code(self, system: str, code: str) -> list[AnatomicLocation]:
        """Find locations by external code (SNOMED, FMA, etc.).

        All returned objects are automatically bound to this index.

        Args:
            system: Code system (e.g., "SNOMED", "FMA")
            code: Code value in that system

        Returns:
            List of matching AnatomicLocation objects (may be empty)
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            """
            SELECT al.*
            FROM anatomic_locations al
            JOIN anatomic_codes alc ON al.id = alc.location_id
            WHERE UPPER(alc.system) = UPPER(?) AND alc.code = ?
            """,
            [system, code],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[AnatomicLocation]:
        """Hybrid search (FTS + semantic) returning ranked results.

        All returned objects are automatically bound to this index.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching AnatomicLocation objects sorted by relevance
        """
        conn = self._ensure_connection()

        # For now, implement FTS-only search
        # TODO: Add semantic search when embeddings are available
        rows = conn.execute(
            """
            WITH candidates AS (
                SELECT
                    al.id,
                    fts_main_anatomic_locations.match_bm25(al.id, ?) AS bm25_score
                FROM anatomic_locations AS al
            )
            SELECT al.*
            FROM candidates c
            JOIN anatomic_locations al ON c.id = al.id
            WHERE c.bm25_score IS NOT NULL
            ORDER BY c.bm25_score DESC
            LIMIT ?
            """,
            [query, limit],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    # =========================================================================
    # Hierarchy Navigation (using pre-computed paths, auto-binds results)
    # =========================================================================

    def get_containment_ancestors(self, location_id: str) -> list[AnatomicLocation]:
        """Get containedBy ancestors using materialized path.

        Returns list ordered from immediate parent to root (body).

        Args:
            location_id: RID identifier

        Returns:
            List of ancestor locations (may be empty if location has no parent)
        """
        conn = self._ensure_connection()

        # Get the location's path
        path_row = conn.execute(
            "SELECT containment_path FROM anatomic_locations WHERE id = ?",
            [location_id],
        ).fetchone()

        if not path_row or not path_row[0]:
            return []

        path = path_row[0]

        # Find all ancestors by matching their paths as prefixes
        # Order by depth descending (immediate parent first, root last)
        rows = conn.execute(
            """
            SELECT *
            FROM anatomic_locations
            WHERE ? LIKE containment_path || '%' AND id != ?
            ORDER BY containment_depth DESC
            """,
            [path, location_id],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def get_containment_descendants(self, location_id: str) -> list[AnatomicLocation]:
        """Get containment descendants using materialized path.

        Args:
            location_id: RID identifier

        Returns:
            List of descendant locations (may be empty)
        """
        conn = self._ensure_connection()

        # Get the location's path
        path_row = conn.execute(
            "SELECT containment_path FROM anatomic_locations WHERE id = ?",
            [location_id],
        ).fetchone()

        if not path_row or not path_row[0]:
            return []

        path = path_row[0]

        # Find all descendants by matching path prefix
        rows = conn.execute(
            """
            SELECT *
            FROM anatomic_locations
            WHERE containment_path LIKE ? || '%' AND id != ?
            ORDER BY containment_depth
            """,
            [path, location_id],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def get_partof_ancestors(self, location_id: str) -> list[AnatomicLocation]:
        """Get partOf ancestors using materialized path.

        Args:
            location_id: RID identifier

        Returns:
            List of part-of ancestors (may be empty)
        """
        conn = self._ensure_connection()

        # Get the location's part-of path
        path_row = conn.execute(
            "SELECT partof_path FROM anatomic_locations WHERE id = ?",
            [location_id],
        ).fetchone()

        if not path_row or not path_row[0]:
            return []

        path = path_row[0]

        # Find all part-of ancestors by matching their paths as prefixes
        rows = conn.execute(
            """
            SELECT *
            FROM anatomic_locations
            WHERE ? LIKE partof_path || '%' AND id != ?
            ORDER BY partof_depth DESC
            """,
            [path, location_id],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def get_children_of(self, parent_id: str) -> list[AnatomicLocation]:
        """Get direct children (containment_parent_id = parent_id).

        Args:
            parent_id: RID identifier of parent

        Returns:
            List of child locations (may be empty)
        """
        conn = self._ensure_connection()

        # Query for direct children via parent reference
        rows = conn.execute(
            """
            SELECT *
            FROM anatomic_locations
            WHERE containment_parent_id = ?
            ORDER BY description
            """,
            [parent_id],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    # =========================================================================
    # Filtering and Iteration (auto-binds results)
    # =========================================================================

    def by_region(self, region: str) -> list[AnatomicLocation]:
        """Get all locations in a region.

        Args:
            region: Region name (e.g., "Head", "Thorax")

        Returns:
            List of locations in that region
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            "SELECT * FROM anatomic_locations WHERE region = ? ORDER BY description",
            [region],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def by_location_type(self, ltype: LocationType) -> list[AnatomicLocation]:
        """Get all locations of a specific location type.

        Args:
            ltype: LocationType enum value (STRUCTURE, SPACE, REGION, etc.)

        Returns:
            List of locations with that type
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            "SELECT * FROM anatomic_locations WHERE location_type = ? ORDER BY description",
            [ltype.value],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def by_system(self, system: BodySystem) -> list[AnatomicLocation]:
        """Get all locations in a body system.

        Args:
            system: BodySystem enum value

        Returns:
            List of locations in that system
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            "SELECT * FROM anatomic_locations WHERE body_system = ? ORDER BY description",
            [system.value],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def by_structure_type(self, stype: StructureType) -> list[AnatomicLocation]:
        """Get all locations of a structure type.

        Only returns locations where location_type=STRUCTURE.

        Args:
            stype: StructureType enum value

        Returns:
            List of locations with that structure type
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            "SELECT * FROM anatomic_locations WHERE structure_type = ? ORDER BY description",
            [stype.value],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def __iter__(self) -> Iterator[AnatomicLocation]:
        """Iterate over all anatomic locations.

        Yields:
            AnatomicLocation objects bound to this index
        """
        conn = self._ensure_connection()
        rows = conn.execute("SELECT * FROM anatomic_locations ORDER BY description").fetchall()

        for row in rows:
            yield self._row_to_location(row)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _ensure_connection(self) -> duckdb.DuckDBPyConnection:
        """Ensure connection is open, raising if not.

        Returns:
            Active DuckDB connection

        Raises:
            RuntimeError: If connection not open
        """
        if self.conn is None:
            raise RuntimeError("AnatomicLocationIndex connection not open. Use context manager or call open() first.")
        return self.conn

    def _row_to_location(self, row: tuple[object, ...]) -> AnatomicLocation:
        """Convert a database row to an AnatomicLocation object.

        Eagerly loads codes and synonyms via JOIN queries.
        Automatically binds the location to this index.

        Args:
            row: Database row from anatomic_locations table

        Returns:
            AnatomicLocation object bound to this index
        """
        conn = self._ensure_connection()

        # Map row columns to fields (based on table schema)
        # Schema: id, description, region, location_type, body_system, structure_type,
        #         laterality, definition, sex_specific, search_text, vector,
        #         containment_path, containment_parent_id, containment_parent_display,
        #         containment_depth, containment_children,
        #         partof_path, partof_parent_id, partof_parent_display,
        #         partof_depth, partof_children,
        #         left_id, left_display, right_id, right_display, generic_id, generic_display,
        #         created_at, updated_at
        location_id = str(row[0])
        description = str(row[1])
        region_str = row[2]
        location_type_str = row[3]
        body_system_str = row[4]
        structure_type_str = row[5]
        laterality_str = row[6]
        definition = str(row[7]) if row[7] else None
        sex_specific = str(row[8]) if row[8] else None
        # Skip search_text (row[9]) and vector (row[10])
        containment_path = str(row[11]) if row[11] else None
        containment_parent_id = str(row[12]) if row[12] else None
        containment_parent_display = str(row[13]) if row[13] else None
        containment_depth = cast(int, row[14]) if row[14] is not None else None
        containment_children_raw = cast(list[dict[str, str]], row[15]) if row[15] else []
        partof_path = str(row[16]) if row[16] else None
        partof_parent_id = str(row[17]) if row[17] else None
        partof_parent_display = str(row[18]) if row[18] else None
        partof_depth = cast(int, row[19]) if row[19] is not None else None
        partof_children_raw = cast(list[dict[str, str]], row[20]) if row[20] else []
        left_id = str(row[21]) if row[21] else None
        left_display = str(row[22]) if row[22] else None
        right_id = str(row[23]) if row[23] else None
        right_display = str(row[24]) if row[24] else None
        generic_id = str(row[25]) if row[25] else None
        generic_display = str(row[26]) if row[26] else None
        # Skip created_at (row[27]) and updated_at (row[28])

        # Load codes
        code_rows = conn.execute(
            "SELECT system, code, display FROM anatomic_codes WHERE location_id = ?",
            [location_id],
        ).fetchall()
        codes = [IndexCode(system=str(r[0]), code=str(r[1]), display=str(r[2]) if r[2] else None) for r in code_rows]

        # Load synonyms
        synonym_rows = conn.execute(
            "SELECT synonym FROM anatomic_synonyms WHERE location_id = ? ORDER BY synonym",
            [location_id],
        ).fetchall()
        synonyms = [str(r[0]) for r in synonym_rows]

        # Load references
        ref_rows = conn.execute(
            "SELECT url, title, description FROM anatomic_references WHERE location_id = ?",
            [location_id],
        ).fetchall()
        references = [
            WebReference(
                url=str(r[0]),
                title=str(r[1]),  # title is required in WebReference
                description=str(r[2]) if r[2] else None,
            )
            for r in ref_rows
            if r[1]  # Skip rows without title
        ]

        # Parse containment parent from denormalized columns
        containment_parent = None
        if containment_parent_id and containment_parent_display:
            containment_parent = AnatomicRef(id=containment_parent_id, display=containment_parent_display)

        # Parse containment children from STRUCT array (DuckDB returns list of dicts)
        containment_children = [
            AnatomicRef(id=str(child["id"]), display=str(child["display"]))
            for child in containment_children_raw
            if isinstance(child, dict) and "id" in child and "display" in child
        ]

        # Parse part-of parent from denormalized columns
        partof_parent = None
        if partof_parent_id and partof_parent_display:
            partof_parent = AnatomicRef(id=partof_parent_id, display=partof_parent_display)

        # Parse part-of children from STRUCT array (DuckDB returns list of dicts)
        partof_children = [
            AnatomicRef(id=str(child["id"]), display=str(child["display"]))
            for child in partof_children_raw
            if isinstance(child, dict) and "id" in child and "display" in child
        ]

        # Parse laterality variants from denormalized columns
        left_variant = None
        right_variant = None
        generic_variant = None

        if left_id and left_display:
            left_variant = AnatomicRef(id=left_id, display=left_display)
        if right_id and right_display:
            right_variant = AnatomicRef(id=right_id, display=right_display)
        if generic_id and generic_display:
            generic_variant = AnatomicRef(id=generic_id, display=generic_display)

        # Convert enum strings to enums
        region = AnatomicRegion(region_str) if region_str else None
        location_type = LocationType(location_type_str) if location_type_str else LocationType.STRUCTURE
        body_system = BodySystem(body_system_str) if body_system_str else None
        structure_type = StructureType(structure_type_str) if structure_type_str else None
        laterality = Laterality(laterality_str) if laterality_str else Laterality.NONLATERAL

        # Create and bind location
        location = AnatomicLocation(
            id=location_id,
            description=description,
            region=region,
            location_type=location_type,
            body_system=body_system,
            structure_type=structure_type,
            laterality=laterality,
            definition=definition,
            sex_specific=sex_specific,
            synonyms=synonyms,
            codes=codes,
            references=references,
            containment_path=containment_path,
            containment_parent=containment_parent,
            containment_depth=containment_depth,
            containment_children=containment_children,
            partof_path=partof_path,
            partof_parent=partof_parent,
            partof_depth=partof_depth,
            partof_children=partof_children,
            left_variant=left_variant,
            right_variant=right_variant,
            generic_variant=generic_variant,
        )

        return location.bind(self)


__all__ = ["AnatomicLocationIndex"]
