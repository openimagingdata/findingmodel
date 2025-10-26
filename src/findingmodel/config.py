from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import openai
from platformdirs import user_data_dir
from pydantic import BeforeValidator, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    openai_default_model: str = Field(default="gpt-4o-mini")
    openai_default_model_full: str = Field(default="gpt-5")
    openai_default_model_small: str = Field(default="gpt-4.1-nano")

    # Perplexity API
    perplexity_base_url: HttpUrl = Field(default=HttpUrl("https://api.perplexity.ai"))
    perplexity_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
    perplexity_default_model: str = Field(default="sonar-pro")

    # DEPRECATED: MongoDB is no longer the default index backend
    # Use DuckDB instead (see duckdb_* settings below)
    # To use MongoDB, install with: pip install findingmodel[mongodb]
    # mongodb_uri: QuoteStrippedSecretStr = Field(default=SecretStr("mongodb://localhost:27017"))
    # mongodb_db: str = Field(default="findingmodels")
    # mongodb_index_collection_base: str = Field(default="index_entries")
    # mongodb_organizations_collection_base: str = Field(default="organizations")
    # mongodb_people_collection_base: str = Field(default="people")

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
    duckdb_anatomic_path: str = Field(
        default="anatomic_locations.duckdb",
        description="Filename for anatomic locations database in user data directory",
    )
    duckdb_index_path: str = Field(
        default="finding_models.duckdb",
        description="Filename for finding models index database in user data directory",
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


def ensure_db_file(
    filename: str,
    remote_url: str | None,
    remote_hash: str | None,
    manifest_key: str,
) -> Path:
    """Download DB file to user data directory with manifest support.

    Pooch will automatically re-download if the local file's hash doesn't match the expected hash.

    Priority:
        1. Use existing local file if present and hash matches (Pooch verifies this)
        2. Try manifest fetch (always attempted, manifest_key always provided)
        3. Fall back to explicit environment config (remote_url/remote_hash) if not None
        4. Error if manifest fails and no explicit environment config

    Args:
        filename: Database filename (e.g., 'anatomic_locations.duckdb')
        remote_url: Optional direct URL to download from (explicit env config fallback)
        remote_hash: Optional hash for verification (e.g., 'sha256:abc...')
        manifest_key: Key in manifest JSON databases section (e.g., 'finding_models')

    Returns:
        Path to the database file (may not exist if download not configured)

    Raises:
        ConfigurationError: If manifest fails and no explicit environment config provided

    Example:
        # Prefer manifest, fall back to explicit env config
        db_path = ensure_db_file(
            "finding_models.duckdb",
            remote_url="https://example.com/db.duckdb",
            remote_hash="sha256:abc123...",
            manifest_key="finding_models"
        )
    """
    from findingmodel import logger

    # Get user data directory (platform-specific)
    data_dir = Path(user_data_dir(appname="findingmodel", appauthor="openimagingdata", ensure_exists=True))
    db_path = data_dir / filename

    # Always try manifest first (manifest_key always provided now)
    url_to_use = None
    hash_to_use = None
    version = None

    try:
        manifest = fetch_manifest()
        db_info = manifest["databases"][manifest_key]
        url_to_use = db_info["url"]
        hash_to_use = db_info["hash"]
        version = db_info.get("version")
        version_str = version if version else "unknown"
        logger.info(f"Using manifest version {version_str} for {manifest_key}")
    except Exception as e:
        # Manifest failed - check for explicit environment config
        if remote_url is None or remote_hash is None:
            raise ConfigurationError(
                f"Cannot download {filename}: manifest fetch failed ({e}) "
                f"and no explicit REMOTE_*_DB_URL/REMOTE_*_DB_HASH configured in environment. "
                f"Either fix manifest connectivity or set explicit environment variables."
            ) from e
        # User explicitly configured URL/hash - use as fallback
        logger.warning(
            f"Manifest fetch failed ({e}), using explicit environment config: url={remote_url}, hash={remote_hash}"
        )
        url_to_use = remote_url
        hash_to_use = remote_hash

    # Download using Pooch if URL/hash available
    if url_to_use and hash_to_use:
        import pooch

        # Pooch will check if file exists and verify hash
        # If hash mismatches, it will automatically re-download
        logger.info(f"Ensuring database file '{filename}' is available (will download/update if needed)")
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            downloaded = pooch.retrieve(url=url_to_use, known_hash=hash_to_use, path=data_dir, fname=filename)
            logger.info(f"Database file ready at {downloaded}")
            return Path(downloaded)
        except Exception as e:
            logger.error(f"Failed to download/verify database file '{filename}': {e}")
            raise

    # No remote URL configured - check if local file exists
    if db_path.exists():
        logger.debug(f"Using local database file: {db_path}")
        return db_path

    logger.debug(f"No remote URL configured for '{filename}', returning local path: {db_path}")
    return db_path  # Return path even if doesn't exist (existing error handling will catch it)


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
