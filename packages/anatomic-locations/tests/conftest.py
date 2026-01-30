from pathlib import Path

import pytest

# =============================================================================
# Anatomic location test fixtures
# =============================================================================


@pytest.fixture(scope="session")
def anatomic_sample_data() -> list[dict[str, object]]:
    """Load real sample data from test/data/anatomic_sample.json."""
    import json

    sample_path = Path(__file__).parent / "data" / "anatomic_sample.json"
    with open(sample_path) as f:
        data: list[dict[str, object]] = json.load(f)
        return data


@pytest.fixture(scope="session")
def anatomic_sample_embeddings() -> dict[str, list[float]]:
    """Load pre-generated embeddings from test/data/anatomic_sample_embeddings.json."""
    import json

    emb_path = Path(__file__).parent / "data" / "anatomic_sample_embeddings.json"
    with open(emb_path) as f:
        embeddings: dict[str, list[float]] = json.load(f)
        return embeddings


@pytest.fixture(scope="session")
def anatomic_query_embeddings() -> dict[str, list[float]]:
    """Load pre-generated query embeddings for search tests."""
    import json

    query_path = Path(__file__).parent / "data" / "anatomic_query_embeddings.json"
    with open(query_path) as f:
        queries: dict[str, list[float]] = json.load(f)
        return queries


@pytest.fixture(scope="session")
def anatomic_records_by_id(anatomic_sample_data: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Index sample data by _id for path computation tests."""
    return {r["_id"]: r for r in anatomic_sample_data}  # type: ignore[misc]


@pytest.fixture
def temp_duckdb_path(tmp_path: Path) -> Path:
    """Temporary DuckDB file path for integration tests."""
    return tmp_path / "test_anatomic.duckdb"


@pytest.fixture(scope="session")
def prebuilt_db_path() -> Path:
    """Path to pre-built test database (committed to repo).

    This database is built using oidm-maintenance and committed to the repo.
    To rebuild, run from workspace root:
        uv run python packages/oidm-maintenance/scripts/build_anatomic_test_fixture.py
    """
    db_path = Path(__file__).parent / "data" / "anatomic_test.duckdb"
    if not db_path.exists():
        pytest.skip(
            "Pre-built test database not found. "
            "Run: uv run python packages/oidm-maintenance/scripts/build_anatomic_test_fixture.py"
        )
    return db_path
