# src/findingmodel/tools/ontology_search.py

import asyncio

import lancedb
from pydantic import BaseModel

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.index_code import IndexCode

# Table constants
ONTOLOGY_TABLES = ["anatomic_locations", "radlex", "snomedct"]
TABLE_TO_INDEX_CODE_SYSTEM = {"anatomic_locations": "ANATOMICLOCATIONS", "radlex": "RADLEX", "snomedct": "SNOMEDCT"}


def normalize_concept(text: str) -> str:
    """
    Normalize concept text for deduplication by removing semantic tags and trailing parenthetical content.

    Args:
        text: Original concept text

    Returns:
        Normalized text for comparison
    """
    # Take only the first line if multi-line
    normalized = text.split("\n")[0]

    # Remove everything after colon (common in RadLex results like "berry aneurysm: description...")
    if ":" in normalized:
        normalized = normalized.split(":")[0]

    # Remove TRAILING parenthetical content only (e.g., "Liver (organ)" -> "Liver")
    # But preserve middle parenthetical content (e.g., "Calcium (2+) level" stays as is)
    normalized = normalized.strip()

    # Check if string ends with parentheses
    if normalized.endswith(")"):
        # Find the matching opening parenthesis for the trailing group
        paren_count = 0
        start_pos = -1

        # Work backwards from the end
        for i in range(len(normalized) - 1, -1, -1):
            if normalized[i] == ")":
                paren_count += 1
            elif normalized[i] == "(":
                paren_count -= 1
                if paren_count == 0:
                    start_pos = i
                    break

        # If we found a matching opening parenthesis, check if it's trailing
        # (i.e., only whitespace between the opening paren and what comes before)
        if start_pos > 0:
            # Get text before the parenthesis
            before_paren = normalized[:start_pos].rstrip()
            # If there's text before and it doesn't end with another closing paren,
            # this is a trailing parenthetical expression
            if before_paren and not before_paren.endswith(")"):
                normalized = before_paren

    # Normalize whitespace (but preserve case)
    normalized = " ".join(normalized.split())

    return normalized


class OntologySearchResult(BaseModel):
    """Search result from LanceDB"""

    concept_id: str
    concept_text: str
    score: float
    table_name: str

    def as_index_code(self) -> IndexCode:
        """Convert to IndexCode format"""
        return IndexCode(
            system=TABLE_TO_INDEX_CODE_SYSTEM.get(self.table_name, self.table_name),
            code=self.concept_id,
            display=normalize_concept(self.concept_text),
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

    async def search_parallel(
        self,
        queries: list[str],
        tables: list[str] | None = None,
        limit_per_query: int = 10,
        filter_anatomical: bool = False,
    ) -> list[OntologySearchResult]:
        """
        Execute multiple search queries in parallel and return merged results.

        Args:
            queries: List of search queries to execute
            tables: Optional list of table names to search. If None, searches all available tables
            limit_per_query: Maximum number of results per query (distributed across tables)
            filter_anatomical: If True, filters out purely anatomical concepts

        Returns:
            Deduplicated list of search results sorted by score
        """
        if not self.connected:
            raise RuntimeError("Must be connected to LanceDB before searching")

        if not queries:
            return []

        # Calculate limit per table to distribute the per-query limit across tables
        search_tables = tables if tables is not None else list(self._tables.keys())
        limit_per_table = max(1, limit_per_query // len(search_tables)) if search_tables else limit_per_query

        # Execute all queries in parallel
        search_tasks = [self.search_tables(query, tables=tables, limit_per_table=limit_per_table) for query in queries]

        try:
            results_per_query = await asyncio.gather(*search_tasks)
            logger.info(f"Completed {len(queries)} parallel searches")
        except Exception as e:
            logger.error(f"Parallel search failed: {e}")
            raise

        # Flatten results from all queries and tables
        all_results = []
        for query_results in results_per_query:
            for table_results in query_results.values():
                all_results.extend(table_results)

        # Deduplicate results
        deduplicated_results = self._deduplicate_results(all_results)

        # Apply anatomical filtering if requested
        if filter_anatomical:
            filtered_results = [result for result in deduplicated_results if not self._is_anatomical_concept(result)]
            logger.info(f"Filtered {len(deduplicated_results) - len(filtered_results)} anatomical concepts")
            return filtered_results

        return deduplicated_results

    def _deduplicate_results(self, results: list[OntologySearchResult]) -> list[OntologySearchResult]:
        """
        Remove duplicate results based on normalized concept text.

        Args:
            results: List of search results to deduplicate

        Returns:
            Deduplicated list sorted by score (highest first)
        """
        if not results:
            return []

        # Group results by normalized concept text
        concept_groups: dict[str, list[OntologySearchResult]] = {}

        for result in results:
            normalized = normalize_concept(result.concept_text)
            if normalized not in concept_groups:
                concept_groups[normalized] = []
            concept_groups[normalized].append(result)

        # Keep the highest scoring version of each concept
        deduplicated = []
        for group in concept_groups.values():
            # Sort by score descending and take the best one
            best_result = max(group, key=lambda x: x.score)
            deduplicated.append(best_result)

        # Sort final results by score descending
        deduplicated.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)} unique concepts")
        return deduplicated

    def _is_anatomical_concept(self, result: OntologySearchResult) -> bool:
        """
        Determine if a concept is purely anatomical and should be filtered.

        Args:
            result: Search result to check

        Returns:
            True if the concept is purely anatomical
        """
        concept_text = result.concept_text.lower()

        # For SNOMED-CT, check for body structure tag
        if result.table_name == "snomedct" and "(body structure)" in concept_text:
            # Preserve pathological anatomy with these keywords
            pathological_keywords = [
                "metastasis",
                "tumor",
                "cancer",
                "lesion",
                "abnormal",
                "malignant",
                "benign",
                "cyst",
                "mass",
                "neoplasm",
            ]

            # If it contains pathological keywords, don't filter it out
            # Pure anatomical structure should be filtered unless it has pathological keywords
            return all(keyword not in concept_text for keyword in pathological_keywords)

        # For other tables, we don't filter anatomical concepts
        return False
