# src/findingmodel/tools/ontology_search.py

import re

import lancedb
from pydantic import BaseModel

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.index_code import IndexCode

# Table constants
ONTOLOGY_TABLES = ["anatomic_locations", "radlex", "snomedct"]
TABLE_TO_INDEX_CODE_SYSTEM = {"anatomic_locations": "ANATOMICLOCATIONS", "radlex": "RADLEX", "snomedct": "SNOMEDCT"}


class OntologySearchResult(BaseModel):
    """Search result from LanceDB"""

    concept_id: str
    concept_text: str
    score: float
    table_name: str

    def as_index_code(self) -> IndexCode:
        """Convert to IndexCode format"""

        # Concept text should chop off parenthetical info at end if present
        display_text = self.concept_text.split("\n")[0]
        if m := re.match(r"^(.*?)(\s+\(.+\))?$", display_text):
            display_text = m.group(1)
        return IndexCode(
            system=TABLE_TO_INDEX_CODE_SYSTEM.get(self.table_name, self.table_name),
            code=self.concept_id,
            display=display_text,
        )


class OntologySearchClient:
    """Production-ready LanceDB client for medical terminology search"""

    def __init__(self, lancedb_uri: str | None = None, api_key: str | None = None) -> None:
        self._db_conn: lancedb.AsyncConnection | None = None
        self._tables: dict[str, lancedb.AsyncTable] = {}
        self._uri = lancedb_uri or settings.lancedb_uri
        self._api_key = api_key or (settings.lancedb_api_key.get_secret_value() if settings.lancedb_api_key else None)

    @property
    def connected(self) -> bool:
        """Check if connected to LanceDB."""
        return self._db_conn is not None

    async def connect(self) -> None:
        """Connect to LanceDB and cache table references."""
        if self._db_conn:
            return  # Already connected

        try:
            assert self._uri is not None, "LanceDB URI must be provided"

            self._db_conn = await lancedb.connect_async(
                uri=self._uri,
                api_key=self._api_key,
            )

            # Load and cache all available tables
            table_names = list(await self._db_conn.table_names())
            for table_name in table_names:
                self._tables[table_name] = await self._db_conn.open_table(table_name)

            logger.info(f"Connected to LanceDB with {len(self._tables)} tables: {list(self._tables.keys())}")

        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            # Ensure connection status is clean on failure
            self._db_conn = None
            self._tables.clear()
            raise

    def disconnect(self) -> None:
        """Disconnect from LanceDB (synchronous)."""
        if self.connected and self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None
            self._tables.clear()
            logger.info("Disconnected from LanceDB")

    async def search_tables(
        self, query: str, tables: list[str] | None = None, limit_per_table: int = 10
    ) -> dict[str, list[OntologySearchResult]]:
        """Search across specified tables using hybrid search."""
        if not self.connected:
            raise RuntimeError("Must be connected to LanceDB before searching")

        # Use specified tables or search all available tables
        search_tables = tables if tables is not None else list(self._tables.keys())

        results: dict[str, list[OntologySearchResult]] = {}

        for table_name in search_tables:
            if table_name not in self._tables:
                logger.warning(f"Table '{table_name}' not found, skipping")
                continue

            try:
                table = self._tables[table_name]
                cursor = await table.search(query=query, query_type="hybrid")
                search_results = await cursor.limit(limit_per_table).to_list()

                # Convert to OntologySearchResult objects
                ontology_results = []
                for row in search_results:
                    result = OntologySearchResult(
                        concept_id=row["concept_id"],
                        concept_text=row["concept_text"],
                        score=row.get("_relevance_score", 0.0),
                        table_name=table_name,
                    )
                    ontology_results.append(result)

                results[table_name] = ontology_results
                logger.info(f"Found {len(ontology_results)} results in table '{table_name}' for query: '{query}'")

            except Exception as e:
                logger.error(f"Search failed for table '{table_name}': {e}")
                # Continue with other tables
                continue

        return results
