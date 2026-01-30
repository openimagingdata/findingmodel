import re


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Model name must be a string, not {type(name)}")
    if len(name) < 3:
        raise ValueError(f"Model name must be at least 3 characters long, not {name}")
    return re.sub(r"[^a-zA-Z0-9]", "_", name).lower()


def model_file_name(name: str) -> str:
    """Convert a finding model name to a file name."""
    return normalize_name(name) + ".fm.json"


__all__ = ["model_file_name", "normalize_name"]
