"""Common utility functions for OIDM packages."""

from pydantic import SecretStr


def strip_quotes(value: str) -> str:
    """Strip leading and trailing quotes from a string.

    Useful for environment variables that may be quoted.
    """
    return value.strip("\"'")


def strip_quotes_secret(value: str | SecretStr) -> str:
    """Strip leading and trailing quotes from a string or SecretStr.

    Handles both plain strings and Pydantic SecretStr values.
    """
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    return strip_quotes(value)
