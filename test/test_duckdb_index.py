"""Tests for the DuckDB-backed index implementation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import duckdb
import pytest

from findingmodel import duckdb_index
from findingmodel.config import settings
from findingmodel.duckdb_index import DuckDBIndex, IndexReturnType
from findingmodel.finding_model import FindingModelFull


def _fake_openai_client(*_: Any, **__: Any) -> object:  # pragma: no cover - test helper
    """Return a dummy OpenAI client for patched calls."""

    return object()


def _write_model_file(path: Path, data: FindingModelFull) -> None:
    path.write_text(data.model_dump_json(indent=2))


def _make_test_model() -> FindingModelFull:
    return FindingModelFull.model_validate({
        "oifm_id": "OIFM_UNIT_000001",
        "name": "Unit Test Model",
        "description": "Model used for unit testing behavior.",
        "synonyms": ["UTM"],
        "tags": ["unit"],
        "attributes": [
            {
                "oifma_id": "OIFMA_UNIT_000001",
                "name": "Lesion size",
                "description": "Numeric size attribute.",
                "type": "numeric",
                "minimum": 0,
                "maximum": 10,
                "unit": "cm",
            }
        ],
    })


def _table_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


@pytest.mark.asyncio
async def test_setup_creates_hnsw_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_client(self: DuckDBIndex) -> object:  # pragma: no cover - test helper
        await asyncio.sleep(0)
        return _fake_openai_client()

    monkeypatch.setattr(DuckDBIndex, "_ensure_openai_client", fake_client, raising=False)

    db_path = tmp_path / "index.duckdb"
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()

    conn = index._ensure_connection()
    rows = conn.execute("SELECT index_name FROM duckdb_indexes() WHERE table_name = 'finding_models'").fetchall()
    assert any(row[0] == "finding_models_embedding_hnsw" for row in rows)

    if index.conn is not None:
        index.conn.close()


@pytest.mark.asyncio
async def test_semantic_search_returns_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dims = settings.openai_embedding_dimensions

    async def fake_embedding(
        text: str,
        *,
        client: object | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[float]:  # pragma: no cover - test helper
        _ = (text, client, model)
        target_dims = dimensions or dims
        await asyncio.sleep(0)
        return [0.01] * target_dims

    async def fake_client(self: DuckDBIndex) -> object:  # pragma: no cover - test helper
        await asyncio.sleep(0)
        return _fake_openai_client()

    monkeypatch.setattr(duckdb_index, "get_embedding_for_duckdb", fake_embedding)
    monkeypatch.setattr(DuckDBIndex, "_ensure_openai_client", fake_client, raising=False)

    db_path = tmp_path / "index.duckdb"
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()

    model = _make_test_model()
    file_path = tmp_path / "model.fm.json"
    _write_model_file(file_path, model)

    status = await index.add_or_update_entry_from_file(file_path, model)
    assert status is IndexReturnType.ADDED

    entry = await index.get("OIFM_UNIT_000001")
    assert entry is not None
    assert entry.name == "Unit Test Model"

    results = await index.search("unrelated query", limit=5)
    assert [res.oifm_id for res in results] == ["OIFM_UNIT_000001"]

    filtered = await index.search("unrelated query", limit=5, tags=["unit"])
    assert [res.oifm_id for res in filtered] == ["OIFM_UNIT_000001"]

    no_results = await index.search("unrelated query", limit=5, tags=["other"])
    assert no_results == []

    if index.conn is not None:
        index.conn.close()


@pytest.mark.asyncio
async def test_remove_entry_clears_related_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dims = settings.openai_embedding_dimensions

    async def fake_embedding(
        text: str,
        *,
        client: object | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[float]:  # pragma: no cover - test helper
        _ = (text, client, model)
        target_dims = dimensions or dims
        await asyncio.sleep(0)
        return [0.02] * target_dims

    async def fake_client(self: DuckDBIndex) -> object:  # pragma: no cover - test helper
        await asyncio.sleep(0)
        return _fake_openai_client()

    monkeypatch.setattr(duckdb_index, "get_embedding_for_duckdb", fake_embedding)
    monkeypatch.setattr(DuckDBIndex, "_ensure_openai_client", fake_client, raising=False)

    db_path = tmp_path / "index.duckdb"
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()

    model = _make_test_model()
    file_path = tmp_path / "model.fm.json"
    _write_model_file(file_path, model)

    await index.add_or_update_entry_from_file(file_path, model)

    conn = index._ensure_connection()
    pre_delete_counts = {
        "synonyms": _table_count(conn, "synonyms"),
        "tags": _table_count(conn, "tags"),
        "attributes": _table_count(conn, "attributes"),
    }
    assert pre_delete_counts == {"synonyms": 1, "tags": 1, "attributes": 1}

    removed = await index.remove_entry(model.oifm_id)
    assert removed is True

    post_delete_counts = {
        "synonyms": _table_count(conn, "synonyms"),
        "tags": _table_count(conn, "tags"),
        "attributes": _table_count(conn, "attributes"),
    }
    assert post_delete_counts == {"synonyms": 0, "tags": 0, "attributes": 0}

    if index.conn is not None:
        index.conn.close()


@pytest.mark.asyncio
async def test_write_operations_rebuild_search_indexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dims = settings.openai_embedding_dimensions

    async def fake_embedding(
        text: str,
        *,
        client: object | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[float]:  # pragma: no cover - test helper
        _ = (text, client, model)
        target_dims = dimensions or dims
        await asyncio.sleep(0)
        return [0.03] * target_dims

    async def fake_client(self: DuckDBIndex) -> object:  # pragma: no cover - test helper
        await asyncio.sleep(0)
        return _fake_openai_client()

    monkeypatch.setattr(duckdb_index, "get_embedding_for_duckdb", fake_embedding)
    monkeypatch.setattr(DuckDBIndex, "_ensure_openai_client", fake_client, raising=False)

    db_path = tmp_path / "index.duckdb"
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()

    model = _make_test_model()
    file_path = tmp_path / "model.fm.json"
    _write_model_file(file_path, model)

    await index.add_or_update_entry_from_file(file_path, model)

    conn = index._ensure_connection()
    hnsw_rows = conn.execute("SELECT index_name FROM duckdb_indexes() WHERE table_name = 'finding_models'").fetchall()
    assert any(row[0] == "finding_models_embedding_hnsw" for row in hnsw_rows)

    conn.execute("SELECT fts_main_finding_models.match_bm25(oifm_id, 'unit') FROM finding_models LIMIT 1").fetchall()

    await index.remove_entry(model.oifm_id)

    hnsw_rows_after = conn.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'finding_models'"
    ).fetchall()
    assert any(row[0] == "finding_models_embedding_hnsw" for row in hnsw_rows_after)

    conn.execute("SELECT fts_main_finding_models.match_bm25(oifm_id, 'unit') FROM finding_models LIMIT 1").fetchall()

    if index.conn is not None:
        index.conn.close()
