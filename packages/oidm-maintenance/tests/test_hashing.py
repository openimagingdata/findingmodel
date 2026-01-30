"""Tests for hashing module."""

from pathlib import Path

from oidm_maintenance.hashing import compute_file_hash


def test_compute_file_hash(tmp_path: Path) -> None:
    """Test that file hashing works."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    hash_result = compute_file_hash(test_file)

    assert hash_result.startswith("sha256:")
    assert len(hash_result) == 71  # "sha256:" + 64 hex chars


def test_compute_file_hash_consistency(tmp_path: Path) -> None:
    """Test that same content produces same hash."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("consistent content")

    hash1 = compute_file_hash(test_file)
    hash2 = compute_file_hash(test_file)

    assert hash1 == hash2


def test_compute_file_hash_different_content(tmp_path: Path) -> None:
    """Test that different content produces different hashes."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    file1.write_text("content one")
    file2.write_text("content two")

    hash1 = compute_file_hash(file1)
    hash2 = compute_file_hash(file2)

    assert hash1 != hash2
