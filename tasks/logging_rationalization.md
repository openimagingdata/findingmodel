# Standardize Loguru Usage Across Packages

## Goal
Make logging behavior consistent across all packages:
- Libraries do not configure handlers by default.
- Library logs are disabled on import.
- Internal modules import Loguru directly (`from loguru import logger`), not via package base.

## Scope
Packages: `findingmodel`, `findingmodel-ai`, `anatomic-locations`, `oidm-common`, `oidm-maintenance`.

## Plan
1. **Audit current usage**
   - Search for imports like:
     - `from <pkg> import logger`
     - `from <pkg> import logger as logger`
     - `from <pkg> import *` (if it pulls logger via `__all__`)
   - Search for package-level re-exports of `logger` and `logger.disable(...)`.

2. **Standardize package defaults**
   - In each package `__init__.py`, add:
     - `from loguru import logger`
     - `logger.disable("<package_name>")`
   - Do **not** re-export `logger` from `__init__.py` unless explicitly required.
   - Ensure the package name used in `disable()` matches the top-level module name.

3. **Remove re-exports**
   - Remove `logger` from `__all__` where present.
   - If any docs or scripts import `logger` from package base, update them to import from Loguru directly.

4. **Update internal imports**
   - Change internal code to use:
     - `from loguru import logger`
   - Remove any indirect imports from package base.

5. **Docs / examples**
   - Add a short logging note in docs (or existing configuration docs):
     - Users enable via `logger.enable("findingmodel")`, etc.
     - Example of enabling multiple packages.

6. **Smoke check**
   - Ensure no broken imports.
   - Optional: add a tiny doctest/snippet in docs (no runtime tests required).

## Acceptance Criteria
- Every package disables its own logger on import using its package name.
- No production code imports `logger` from package base.
- Package `__all__` does not export `logger`.
- Docs show how to enable package logs with Loguru.

## Suggested Implementation Hints
- Search patterns:
  - `rg -n "from (findingmodel|findingmodel_ai|anatomic_locations|oidm_common|oidm_maintenance) import logger"`
  - `rg -n "logger\\.disable\\(" packages`
  - `rg -n "__all__.*logger" packages`
- Files likely to touch:
  - `packages/*/src/*/__init__.py`
  - scripts in `packages/findingmodel-ai/scripts/`
  - docs that mention logging
- Use `logger.disable("<pkg>")` with package name matching module root (e.g., `findingmodel`, `findingmodel_ai`, `anatomic_locations`, `oidm_common`, `oidm_maintenance`).
