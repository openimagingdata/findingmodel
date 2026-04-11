#!/usr/bin/env python3
"""Sanity-check anatomic DB embeddings for zero-filled vectors.

Usage:
  uv run python scripts/check_anatomic_embeddings.py
  uv run python scripts/check_anatomic_embeddings.py --db dist/anatomic_locations.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate anatomic embedding vectors are non-zero.")
    parser.add_argument(
        "--db",
        action="append",
        dest="db_paths",
        help="Path to an anatomic_locations DuckDB file. Repeatable.",
    )
    parser.add_argument(
        "--min-nonzero-ratio",
        type=float,
        default=0.95,
        help="Minimum required ratio of non-zero vectors among non-null vectors (default: 0.95).",
    )
    return parser.parse_args()


def default_db_paths() -> list[Path]:
    return [
        Path("dist/anatomic_locations.duckdb"),
    ]


def check_one_db(db_path: Path, min_nonzero_ratio: float) -> tuple[bool, str]:
    if not db_path.exists():
        return False, f"{db_path}: missing file"

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        # Read embedding profile if available (optional metadata)
        profile_label = "unknown"
        has_profile = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'embedding_profile'"
        ).fetchone()
        if has_profile and int(has_profile[0]) > 0:
            profile = conn.execute(
                "SELECT provider, model, dimensions FROM embedding_profile LIMIT 1"
            ).fetchone()
            if profile is not None:
                profile_label = f"{profile[0]}/{profile[1]}/{profile[2]}"

        total = conn.execute(
            "SELECT COUNT(*) FROM anatomic_locations WHERE vector IS NOT NULL"
        ).fetchone()[0]
        nonzero = conn.execute(
            """
            SELECT COUNT(*)
            FROM anatomic_locations
            WHERE vector IS NOT NULL
              AND (list_min(vector) <> 0 OR list_max(vector) <> 0)
            """
        ).fetchone()[0]

        ratio = (nonzero / total) if total else 0.0
        ok = total > 0 and ratio >= min_nonzero_ratio
        status = "PASS" if ok else "FAIL"
        return (
            ok,
            (
                f"{status} {db_path} | embedding={profile_label} "
                f"| nonzero={nonzero}/{total} ({ratio:.1%})"
            ),
        )
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    db_paths = [Path(p) for p in args.db_paths] if args.db_paths else default_db_paths()

    failures = 0
    for db_path in db_paths:
        ok, message = check_one_db(db_path, args.min_nonzero_ratio)
        print(message)
        if not ok:
            failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
