from typing import Annotated, Literal

import openai
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
    openai_default_model: str = Field(default="gpt-4o")
    openai_default_model_small: str = Field(default="gpt-4o-mini")

    # Perplexity API
    perplexity_base_url: HttpUrl = Field(default=HttpUrl("https://api.perplexity.ai"))
    perplexity_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
    perplexity_default_model: str = Field(default="sonar-pro")

    # Mongo DB
    mongodb_uri: QuoteStrippedSecretStr = Field(default=SecretStr("mongodb://localhost:27017"))
    mongodb_db: str = Field(default="findingmodels")
    mongodb_index_collection_base: str = Field(default="index_entries")
    mongodb_organizations_collection_base: str = Field(default="organizations")
    mongodb_people_collection_base: str = Field(default="people")

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
