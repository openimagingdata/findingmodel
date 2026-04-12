"""ID generation utilities for OIFM and OIFMA identifiers."""

import random

ID_LENGTH = 6


def _random_digits(length: int) -> str:
    """Generate a string of random decimal digits of the given length."""
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def generate_oifm_id(source: str) -> str:
    """Generate a new OIFM model ID for the given source code."""
    return f"OIFM_{source.upper()}_{_random_digits(ID_LENGTH)}"


def generate_oifma_id(source: str) -> str:
    """Generate a new OIFMA attribute ID for the given source code."""
    return f"OIFMA_{source.upper()}_{_random_digits(ID_LENGTH)}"
