"""Anatomic location migration has moved to the anatomic-locations package.

Import from anatomic_locations.migration instead:
    from anatomic_locations.migration import (
        create_anatomic_database,
        load_anatomic_data,
        validate_anatomic_record,
        get_database_stats,
    )
"""

raise ImportError(
    "Anatomic migration functions have moved to the anatomic-locations package. "
    "Use: from anatomic_locations.migration import create_anatomic_database, ..."
)
