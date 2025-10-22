from pathlib import Path
from typing import Annotated, Literal

import openai
from platformdirs import user_data_dir
from pydantic import BeforeValidator, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        default="https://findingmodelsdata.t3.storage.dev/anatomic_locations_20251016.duckdb",
        description="URL to download anatomic locations database",
    )
    remote_anatomic_db_hash: str | None = Field(
        default="sha256:b69c9b072b3661241e858abb307c4eff7d5d074d27fdda61fb87b0efec0dd65b",
        description="SHA256 hash for anatomic DB (e.g. 'sha256:abc...')",
    )
    remote_index_db_url: str | None = Field(
        default="https://findingmodelsdata.t3.storage.dev/finding_models_20251017.duckdb",
        description="URL to download finding models index database",
    )
    remote_index_db_hash: str | None = Field(
        default="sha256:86e52f7cddfa015464f6b8f0947dd8c63d50d55b9e0376c6bdc15be66fcee25f",
        description="SHA256 hash for index DB (e.g. 'sha256:def...')",
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


def ensure_db_file(filename: str, remote_url: str | None, remote_hash: str | None) -> Path:
    """Download DB file to user data directory if it doesn't exist and remote URL is configured.

    Pooch will automatically re-download if the local file's hash doesn't match the expected hash.

    Args:
        filename: Database filename (e.g., 'anatomic_locations.duckdb')
        remote_url: Optional URL to download from
        remote_hash: Optional hash for verification (e.g., 'sha256:abc...')

    Returns:
        Path to the database file (may not exist if download not configured)
    """
    from findingmodel import logger

    # Get user data directory (platform-specific)
    data_dir = Path(user_data_dir(appname="findingmodel", appauthor="openimagingdata", ensure_exists=True))
    db_path = data_dir / filename

    if remote_url and remote_hash:
        import pooch

        # Pooch will check if file exists and verify hash
        # If hash mismatches, it will automatically re-download
        logger.info(f"Ensuring database file '{filename}' is available (will download/update if needed)")
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            downloaded = pooch.retrieve(url=remote_url, known_hash=remote_hash, path=data_dir, fname=filename)
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
    return db_path  # Return path even if doesn't exist (existing error handling will catch it)  # Return path even if doesn't exist (existing error handling will catch it)
