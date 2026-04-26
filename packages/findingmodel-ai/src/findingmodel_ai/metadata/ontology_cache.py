"""Durable ontology lookup evidence cache for metadata enrichment workflows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb
from findingmodel.protocols import OntologySearchResult
from oidm_common.duckdb import setup_duckdb_connection
from oidm_common.models import IndexCode
from pydantic import BaseModel, Field

OntologyEvidenceUsage = Literal[
    "canonical_selected",
    "related_candidate",
    "rejected_candidate",
    "fact_check_evidence",
]


class OntologyLookupEvidence(BaseModel):
    """Stored evidence for one ontology code lookup or candidate consideration."""

    system: str
    code: str
    preferred_display: str
    labels: list[str] = Field(default_factory=list)
    source_service: str
    source_url: str | None = None
    concept_uri: str | None = None
    lookup_timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    raw_response: dict[str, Any] | None = None
    usage: OntologyEvidenceUsage
    query: str | None = None
    relationship: str | None = None
    rejection_reason: str | None = None


class OntologyLookupCache:
    """DuckDB-backed cache of ontology lookup evidence.

    The cache is intentionally simple and durable. It records enough normalized evidence for review,
    auditing, and later fact-checking without coupling callers to a specific live ontology service.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._setup_done = False

    def __enter__(self) -> OntologyLookupCache:
        self.setup()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._setup_done = False

    def setup(self) -> None:
        if self._setup_done:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_lookup_evidence (
                system TEXT NOT NULL,
                code TEXT NOT NULL,
                preferred_display TEXT NOT NULL,
                labels_json TEXT NOT NULL,
                source_service TEXT NOT NULL,
                source_url TEXT,
                concept_uri TEXT,
                lookup_timestamp TEXT NOT NULL,
                raw_response_json TEXT,
                usage TEXT NOT NULL,
                query TEXT,
                relationship TEXT,
                rejection_reason TEXT,
                PRIMARY KEY (system, code, source_service)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ontology_lookup_usage
            ON ontology_lookup_evidence(usage)
            """
        )
        self._setup_done = True

    def upsert(self, evidence: OntologyLookupEvidence) -> None:
        self.setup()
        conn = self._connect()
        conn.execute(
            """
            DELETE FROM ontology_lookup_evidence
            WHERE system = ? AND code = ? AND source_service = ?
            """,
            [
                evidence.system,
                evidence.code,
                evidence.source_service,
            ],
        )
        conn.execute(
            """
            INSERT INTO ontology_lookup_evidence (
                system,
                code,
                preferred_display,
                labels_json,
                source_service,
                source_url,
                concept_uri,
                lookup_timestamp,
                raw_response_json,
                usage,
                query,
                relationship,
                rejection_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                evidence.system,
                evidence.code,
                evidence.preferred_display,
                json.dumps(evidence.labels, sort_keys=True),
                evidence.source_service,
                evidence.source_url,
                evidence.concept_uri,
                evidence.lookup_timestamp.isoformat(),
                json.dumps(evidence.raw_response, sort_keys=True) if evidence.raw_response is not None else None,
                evidence.usage,
                evidence.query,
                evidence.relationship,
                evidence.rejection_reason,
            ],
        )

    def record_ontology_result(
        self,
        result: OntologySearchResult,
        *,
        usage: OntologyEvidenceUsage,
        source_service: str = "metadata_assignment",
        query: str | None = None,
        relationship: str | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        code = result.as_index_code()
        self.upsert(
            OntologyLookupEvidence(
                system=code.system,
                code=code.code,
                preferred_display=code.display or result.concept_text,
                labels=[result.concept_text],
                source_service=source_service,
                concept_uri=_concept_uri(code),
                raw_response=result.model_dump(mode="json"),
                usage=usage,
                query=query,
                relationship=relationship,
                rejection_reason=rejection_reason,
            )
        )

    def record_index_code(
        self,
        code: IndexCode,
        *,
        usage: OntologyEvidenceUsage,
        source_service: str = "metadata_assignment",
        query: str | None = None,
        relationship: str | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        self.upsert(
            OntologyLookupEvidence(
                system=code.system,
                code=code.code,
                preferred_display=code.display or code.code,
                labels=[code.display] if code.display else [],
                source_service=source_service,
                concept_uri=_concept_uri(code),
                raw_response=code.model_dump(mode="json"),
                usage=usage,
                query=query,
                relationship=relationship,
                rejection_reason=rejection_reason,
            )
        )

    def get(self, system: str, code: str, *, source_service: str = "metadata_assignment") -> OntologyLookupEvidence | None:
        self.setup()
        conn = self._connect()
        row = conn.execute(
            """
            SELECT
                system,
                code,
                preferred_display,
                labels_json,
                source_service,
                source_url,
                concept_uri,
                lookup_timestamp,
                raw_response_json,
                usage,
                query,
                relationship,
                rejection_reason
            FROM ontology_lookup_evidence
            WHERE system = ? AND code = ? AND source_service = ?
            """,
            [system, code, source_service],
        ).fetchone()
        if row is None:
            return None
        return OntologyLookupEvidence(
            system=row[0],
            code=row[1],
            preferred_display=row[2],
            labels=json.loads(row[3]),
            source_service=row[4],
            source_url=row[5],
            concept_uri=row[6],
            lookup_timestamp=datetime.fromisoformat(row[7]),
            raw_response=json.loads(row[8]) if row[8] else None,
            usage=row[9],
            query=row[10],
            relationship=row[11],
            rejection_reason=row[12],
        )

    def get_many(
        self, codes: list[IndexCode], *, source_service: str = "metadata_assignment"
    ) -> dict[tuple[str, str], OntologyLookupEvidence]:
        """Return cache evidence for a list of codes keyed by ``(system, code)``."""
        evidence: dict[tuple[str, str], OntologyLookupEvidence] = {}
        for code in codes:
            item = self.get(code.system, code.code, source_service=source_service)
            if item is not None:
                evidence[code.system, code.code] = item
        return evidence

    def _connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = setup_duckdb_connection(self.path, read_only=False)
        return self._conn


def _concept_uri(code: IndexCode) -> str | None:
    system = code.system.lower()
    if system == "snomedct":
        return f"http://purl.bioontology.org/ontology/SNOMEDCT/{code.code}"
    if system == "radlex":
        return f"http://purl.bioontology.org/ontology/RADLEX/{code.code}"
    if system == "loinc":
        return f"http://purl.bioontology.org/ontology/LNC/{code.code}"
    return None


__all__ = ["OntologyEvidenceUsage", "OntologyLookupCache", "OntologyLookupEvidence"]
