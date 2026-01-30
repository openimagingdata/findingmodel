"""OIDM Maintenance Tools - Database build and publish utilities."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
