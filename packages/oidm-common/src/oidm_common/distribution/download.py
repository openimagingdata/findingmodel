"""File download and verification utilities."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _verify_file_hash(file_path: Path, expected_hash: str) -> bool:
    """Verify file hash matches expected value using Pooch.

    Args:
        file_path: Path to file to verify
        expected_hash: Expected hash in format "algorithm:hexdigest" (e.g., "sha256:abc123...")

    Returns:
        True if hash matches, False otherwise
    """
    import pooch

    # Parse "algorithm:hexdigest" format
    algorithm, expected_digest = expected_hash.split(":", 1)

    # Use Pooch's file_hash function
    actual_digest: str = pooch.file_hash(str(file_path), alg=algorithm)

    return actual_digest == expected_digest


def download_file(target_path: Path, url: str, hash_value: str) -> Path:
    """Download file using Pooch with hash verification.

    Args:
        target_path: Target path for downloaded file
        url: Download URL
        hash_value: Expected hash in format "algorithm:hexdigest"

    Returns:
        Path to downloaded file

    Raises:
        Exception: If download or verification fails
    """
    import pooch

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading database file from {url}")
    downloaded = pooch.retrieve(url=url, known_hash=hash_value, path=target_path.parent, fname=target_path.name)
    logger.info(f"Database file ready at {downloaded}")
    return Path(downloaded)
