from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import openai
from platformdirs import user_data_dir
from pydantic import BeforeValidator, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

# Module-level cache for manifest (cleared on process restart)
_manifest_cache: dict[str, Any] | None = None


class ConfigurationError(RuntimeError):
    pass


def strip_quotes(value: str) -> str:
    return value.strip("\"'")


def strip_quotes_secret(value: str | SecretStr) -> str:
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    return strip_quotes(value)


QuoteStrippedStr = Annotated[str, BeforeValidator(strip_quotes)]


QuoteStrippedSecretStr = Annotated[SecretStr, BeforeValidator(strip_quotes_secret)]


class FindingModelConfig(BaseSettings):
    # OpenAI API
    openai_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
    openai_default_model: str = Field(default="gpt-5-mini")
    openai_default_model_full: str = Field(default="gpt-5")
    openai_default_model_small: str = Field(default="gpt-5-nano")

    # Perplexity API
    perplexity_base_url: HttpUrl = Field(default=HttpUrl("https://api.perplexity.ai"))
    perplexity_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
    perplexity_default_model: str = Field(default="sonar-pro")

    # BioOntology API
    bioontology_api_key: QuoteStrippedSecretStr | None = Field(default=None, description="BioOntology.org API key")

    # Logfire configuration (observability platform)
    logfire_token: QuoteStrippedSecretStr | None = Field(
        default=None,
        description="Logfire.dev write token for cloud tracing (optional)",
    )
    disable_send_to_logfire: bool = Field(
        default=False,
        description="Disable sending data to Logfire platform (local-only mode)",
    )
    logfire_verbose: bool = Field(
        default=False,
        description="Enable verbose Logfire console logging",
    )

    # DuckDB configuration
    duckdb_anatomic_path: str | None = Field(
        default=None,
        description="Path to anatomic locations database (absolute, relative to user data dir, or None for default)",
    )
    duckdb_index_path: str | None = Field(
        default=None,
        description="Path to finding models index database (absolute, relative to user data dir, or None for default)",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI model for generating embeddings"
    )
    openai_embedding_dimensions: int = Field(
        default=512, description="Embedding dimensions (512 for text-embedding-3-small reduced, 1536 for full)"
    )

    # Optional remote DuckDB download URLs
    remote_anatomic_db_url: str | None = Field(
        default=None,
        description="URL to download anatomic locations database",
    )
    remote_anatomic_db_hash: str | None = Field(
        default=None,
        description="SHA256 hash for anatomic DB (e.g. 'sha256:abc...')",
    )
    remote_index_db_url: str | None = Field(
        default=None,
        description="URL to download finding models index database",
    )
    remote_index_db_hash: str | None = Field(
        default=None,
        description="SHA256 hash for index DB (e.g. 'sha256:def...')",
    )
    remote_manifest_url: str | None = Field(
        default="https://findingmodelsdata.t3.storage.dev/manifest.json",
        description="URL to JSON manifest for database versions",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def validate_remote_db_config(self) -> Self:
        """Validate that remote URL and hash are provided together (or neither)."""
        # Check anatomic database config
        if (self.remote_anatomic_db_url is None) != (self.remote_anatomic_db_hash is None):
            raise ValueError(
                "Must provide both REMOTE_ANATOMIC_DB_URL and REMOTE_ANATOMIC_DB_HASH, or neither. "
                f"Got URL={'set' if self.remote_anatomic_db_url else 'unset'}, "
                f"hash={'set' if self.remote_anatomic_db_hash else 'unset'}"
            )

        # Check index database config
        if (self.remote_index_db_url is None) != (self.remote_index_db_hash is None):
            raise ValueError(
                "Must provide both REMOTE_INDEX_DB_URL and REMOTE_INDEX_DB_HASH, or neither. "
                f"Got URL={'set' if self.remote_index_db_url else 'unset'}, "
                f"hash={'set' if self.remote_index_db_hash else 'unset'}"
            )

        return self

    def check_ready_for_openai(self) -> Literal[True]:
        if not self.openai_api_key.get_secret_value():
            raise ConfigurationError("OpenAI API key is not set")
        return True

    def check_ready_for_perplexity(self) -> Literal[True]:
        if not self.perplexity_api_key.get_secret_value():
            raise ConfigurationError("Perplexity API key is not set")
        return True


settings = FindingModelConfig()
openai.api_key = settings.openai_api_key.get_secret_value()


def _resolve_target_path(file_path: str | Path | None, manifest_key: str) -> Path:
    """Resolve database file path to absolute Path.

    Args:
        file_path: User-specified path (absolute, relative, or None)
        manifest_key: Key in manifest for default filename (e.g., 'finding_models')

    Returns:
        Resolved absolute Path
    """
    data_dir = Path(user_data_dir(appname="findingmodel", appauthor="openimagingdata", ensure_exists=True))

    if file_path is None:
        # Default: {manifest_key}.duckdb in user data dir
        return data_dir / f"{manifest_key}.duckdb"

    path = Path(file_path)
    if path.is_absolute():
        return path
    else:
        # Relative path: resolve to user_data_dir
        return data_dir / path


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


def _download_file(target_path: Path, url: str, hash_value: str) -> Path:
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

    from findingmodel import logger

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading database file from {url}")
    downloaded = pooch.retrieve(url=url, known_hash=hash_value, path=target_path.parent, fname=target_path.name)
    logger.info(f"Database file ready at {downloaded}")
    return Path(downloaded)


def _download_from_manifest(target_path: Path, manifest_key: str) -> Path:
    """Download file using manifest information.

    Args:
        target_path: Target path for downloaded file
        manifest_key: Key in manifest databases section

    Returns:
        Path to downloaded file

    Raises:
        ConfigurationError: If manifest fetch or download fails
    """
    from findingmodel import logger

    try:
        manifest = fetch_manifest()
        db_info = manifest["databases"][manifest_key]
        url = db_info["url"]
        hash_value = db_info["hash"]
        version = db_info.get("version", "unknown")
        logger.info(f"Using manifest version {version} for {manifest_key}")
        return _download_file(target_path, url, hash_value)
    except Exception as e:
        raise ConfigurationError(
            f"Cannot download {target_path.name}: manifest fetch/download failed ({e}). "
            f"Either fix manifest connectivity or set explicit DUCKDB_*_PATH and REMOTE_*_DB_URL/HASH."
        ) from e


def ensure_db_file(
    file_path: str | Path | None,
    remote_url: str | None,
    remote_hash: str | None,
    manifest_key: str,
) -> Path:
    """Ensure database file is available with flexible configuration priority.

    Priority logic:
        1. If file_path exists and no URL/hash: use file directly (no verification)
        2. If file_path exists and URL/hash provided: verify hash
           - Hash matches: use file
           - Hash mismatch: download from URL
        3. If file doesn't exist and URL/hash provided: download using URL/hash
        4. If file doesn't exist and no URL/hash: download using manifest (fallback)

    Args:
        file_path: Database file path (absolute, relative to user data dir, or None for default)
        remote_url: Optional download URL (must provide both URL and hash, or neither)
        remote_hash: Optional hash for verification (e.g., 'sha256:abc...')
        manifest_key: Key in manifest JSON databases section (e.g., 'finding_models', 'anatomic_locations')

    Returns:
        Path to the database file

    Raises:
        ConfigurationError: If configuration is invalid or download fails

    Examples:
        # Docker production: use pre-mounted file
        db_path = ensure_db_file("/mnt/data/finding_models.duckdb", None, None, "finding_models")

        # Development: use manifest
        db_path = ensure_db_file(None, None, None, "finding_models")

        # Custom URL with verification
        db_path = ensure_db_file(
            "my_db.duckdb",
            "https://example.com/db.duckdb",
            "sha256:abc123...",
            "finding_models"
        )
    """
    from findingmodel import logger

    # Resolve target path
    target = _resolve_target_path(file_path, manifest_key)

    # Check if we have explicit remote config
    has_explicit_remote = remote_url is not None and remote_hash is not None

    # Check if file exists
    if target.exists():
        if has_explicit_remote:
            # Type narrowing: has_explicit_remote means both are not None
            assert remote_url is not None and remote_hash is not None
            # Verify hash
            logger.info(f"Verifying existing file {target}")
            if _verify_file_hash(target, remote_hash):
                logger.info(f"File hash verified, using existing file: {target}")
                return target
            else:
                logger.warning(f"File hash mismatch, re-downloading from {remote_url}")
                return _download_file(target, remote_url, remote_hash)
        else:
            # No hash to verify, use file as-is
            logger.debug(f"Using existing database file: {target}")
            return target

    # File doesn't exist, need to download
    if has_explicit_remote:
        # Type narrowing: has_explicit_remote means both are not None
        assert remote_url is not None and remote_hash is not None
        return _download_file(target, remote_url, remote_hash)
    else:
        # Fall back to manifest
        return _download_from_manifest(target, manifest_key)


def ensure_index_db() -> Path:
    """Ensure finding models index database is available.

    Uses settings to determine path, URL, and hash configuration.
    Automatically downloads from manifest if no local file exists.

    Returns:
        Path to the finding models index database
    """
    return ensure_db_file(
        settings.duckdb_index_path,
        settings.remote_index_db_url,
        settings.remote_index_db_hash,
        manifest_key="finding_models",
    )


def ensure_anatomic_db() -> Path:
    """Ensure anatomic locations database is available.

    Uses settings to determine path, URL, and hash configuration.
    Automatically downloads from manifest if no local file exists.

    Returns:
        Path to the anatomic locations database
    """
    return ensure_db_file(
        settings.duckdb_anatomic_path,
        settings.remote_anatomic_db_url,
        settings.remote_anatomic_db_hash,
        manifest_key="anatomic_locations",
    )


def fetch_manifest() -> dict[str, Any]:
    """Fetch and parse the remote manifest JSON with session caching.

    Returns:
        Parsed manifest with database version info

    Raises:
        ConfigurationError: If manifest URL not configured
        httpx.HTTPError: If fetch fails

    Example:
        manifest = fetch_manifest()
        db_info = manifest["finding_models"]
        # {"version": "2025-01-24", "url": "...", "hash": "sha256:..."}
    """
    from findingmodel import logger

    global _manifest_cache

    # Return cached manifest if available
    if _manifest_cache is not None:
        logger.debug("Using cached manifest")
        return _manifest_cache

    settings = FindingModelConfig()
    if not settings.remote_manifest_url:
        raise ConfigurationError("Manifest URL not configured")

    logger.info(f"Fetching manifest from {settings.remote_manifest_url}")
    response = httpx.get(settings.remote_manifest_url, timeout=10.0)
    response.raise_for_status()

    manifest_data: dict[str, Any] = response.json()
    _manifest_cache = manifest_data
    logger.debug(f"Manifest cached with keys: {list(manifest_data.keys())}")
    return manifest_data


def clear_manifest_cache() -> None:
    """Clear the manifest cache (for testing)."""
    global _manifest_cache
    _manifest_cache = None
