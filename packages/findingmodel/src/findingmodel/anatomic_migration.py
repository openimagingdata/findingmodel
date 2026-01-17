"""Anatomic location migration has moved to the oidm-maintenance package.

Import from oidm_maintenance.anatomic.build instead:
    from oidm_maintenance.anatomic.build import (
        create_anatomic_database,
        build_anatomic_database,
        load_anatomic_data,
        validate_anatomic_record,
    )

For database statistics:
    from anatomic_locations.index import get_database_stats
"""

raise ImportError(
    "Anatomic migration/build functions have moved to the oidm-maintenance package. "
    "Use: from oidm_maintenance.anatomic.build import build_anatomic_database, ..."
)
