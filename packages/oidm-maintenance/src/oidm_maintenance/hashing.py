"""File hashing utilities."""

from pathlib import Path

import pooch


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Uses pooch.file_hash() to compute the hash in a format compatible with
    pooch's verification system.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hash string in format "sha256:hexdigest"

    Example:
        hash_str = compute_file_hash(Path("database.duckdb"))
        # Returns: "sha256:abc123def456..."
    """
    digest = pooch.file_hash(str(file_path), alg="sha256")
    return f"sha256:{digest}"


__all__ = ["compute_file_hash"]
