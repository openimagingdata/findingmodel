# Phase 3.7: Create oidm-maintenance Package

**Status:** ✅ COMPLETE

**Goal:** Create a dedicated maintenance package for database build and publish operations. Make user-facing index classes read-only.

## Architecture Decision

Database building and publishing are **maintainer operations**, not user features. Rather than scattering this code across packages or using pseudo-package structures, we create a proper `oidm-maintenance` package that:
- Contains ALL build and publish logic in one place
- Has proper package structure (testable, versionable)
- Keeps heavy dependencies (boto3, openai) out of user packages
- Can be published to PyPI later if needed (start as workspace-only)

**Key insight:** Users never build databases—they download pre-built ones via `ensure_db_file()` and query them. All build/publish operations are maintainer-only.

## Design Rationale

**Strengths of this approach:**
- **Proper Python package**: Real pyproject.toml, tests, versioning—follows community best practices
- **No duplication**: Shared S3/manifest code in one place, used by both anatomic and findingmodel operations
- **Clean dependency isolation**: boto3, openai (for builds) never appear in user package dependencies
- **Simple mental model**: "Users install findingmodel; maintainers also install oidm-maintenance"
- **Testable**: Standard pytest structure, can mock S3 operations
- **Discoverable**: `oidm-maintain --help` shows all available commands

**Potential issues to mitigate:**
- **Extra package to maintain**: Mitigated by starting workspace-only (no PyPI releases until needed)
- **Coupling to user packages**: oidm-maintenance depends on findingmodel and anatomic-locations; schema changes require coordinated updates. Mitigated by keeping build logic close to the data models it serializes.
- **CI complexity**: Build/publish workflows need oidm-maintenance installed. Mitigated by Taskfile targets that handle this transparently.

## Target Structure

```
packages/
  ├── oidm-common/              # Shared infrastructure (download, duckdb, embeddings)
  ├── anatomic-locations/       # READ-ONLY: search, get, query
  ├── findingmodel/             # READ-ONLY: search, get, query, MCP
  ├── findingmodel-ai/          # AI tools (Phase 4)
  └── oidm-maintenance/         # NEW: Build + publish (maintainers only)
      ├── pyproject.toml
      ├── src/oidm_maintenance/
      │   ├── __init__.py
      │   ├── config.py         # Settings (S3 credentials, endpoints)
      │   ├── s3.py             # S3 client, upload, manifest operations
      │   ├── hashing.py        # File hash computation
      │   ├── anatomic/
      │   │   ├── __init__.py
      │   │   ├── build.py      # Build anatomic database
      │   │   └── publish.py    # Publish anatomic database
      │   ├── findingmodel/
      │   │   ├── __init__.py
      │   │   ├── build.py      # Build findingmodel database
      │   │   └── publish.py    # Publish findingmodel database
      │   └── cli.py            # oidm-maintain command
      └── tests/
          ├── test_s3.py
          ├── test_hashing.py
          ├── test_anatomic_build.py
          └── test_findingmodel_build.py
```

## CLI Usage After Implementation

```bash
# Run via uv (workspace-only, not published yet)
uv run --package oidm-maintenance oidm-maintain --help

# Build databases
uv run --package oidm-maintenance oidm-maintain anatomic build --source data/anatomic.csv --output anatomic.duckdb
uv run --package oidm-maintenance oidm-maintain findingmodel build --source models/ --output findingmodel.duckdb

# Publish databases
uv run --package oidm-maintenance oidm-maintain anatomic publish anatomic.duckdb
uv run --package oidm-maintenance oidm-maintain findingmodel publish findingmodel.duckdb

# Or via Taskfile (simpler)
task maintain:anatomic:build
task maintain:anatomic:publish
task maintain:findingmodel:build
task maintain:findingmodel:publish
```

---

## Sub-phase 3.7.1: Create Package Scaffolding

**What:** Create the oidm-maintenance package directory structure and pyproject.toml.

**Steps:**

1. Create directory structure:
   ```bash
   mkdir -p packages/oidm-maintenance/src/oidm_maintenance/anatomic
   mkdir -p packages/oidm-maintenance/src/oidm_maintenance/findingmodel
   mkdir -p packages/oidm-maintenance/tests
   ```

2. Create `packages/oidm-maintenance/pyproject.toml`:
   ```toml
   [project]
   name = "oidm-maintenance"
   version = "0.1.0"
   description = "Maintenance tools for OIDM packages (database build and publish)"
   requires-python = ">=3.11"
   dependencies = [
       "oidm-common",
       "anatomic-locations",
       "findingmodel",
       "boto3>=1.40",
       "openai>=1.0",
       "rich>=13.0",
       "click>=8.0",
   ]

   [project.scripts]
   oidm-maintain = "oidm_maintenance.cli:main"

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["src/oidm_maintenance"]

   [tool.uv.sources]
   oidm-common = { workspace = true }
   anatomic-locations = { workspace = true }
   findingmodel = { workspace = true }
   ```

3. Create `packages/oidm-maintenance/src/oidm_maintenance/__init__.py`:
   ```python
   """OIDM Maintenance Tools - Database build and publish utilities."""
   __version__ = "0.1.0"
   ```

4. Create empty `__init__.py` files:
   - `packages/oidm-maintenance/src/oidm_maintenance/anatomic/__init__.py`
   - `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/__init__.py`

**Verify:** `uv sync` completes without errors.

---

## Sub-phase 3.7.2: Create Shared Infrastructure Modules

**What:** Create the config, S3, and hashing modules that are shared by both anatomic and findingmodel operations.

**Steps:**

1. Create `packages/oidm-maintenance/src/oidm_maintenance/config.py`:

   ```python
   """Configuration for maintenance operations."""
   from pydantic import SecretStr
   from pydantic_settings import BaseSettings, SettingsConfigDict


   class MaintenanceSettings(BaseSettings):
       """Settings for database maintenance operations."""

       model_config = SettingsConfigDict(env_prefix="OIDM_MAINTAIN_", extra="ignore")

       # S3/Tigris settings
       s3_endpoint_url: str = "https://fly.storage.tigris.dev"
       s3_bucket: str = "findingmodelsdata"
       aws_access_key_id: SecretStr | None = None
       aws_secret_access_key: SecretStr | None = None

       # Manifest settings
       manifest_key: str = "manifest.json"
       manifest_backup_prefix: str = "manifests/archive/"

       # OpenAI for embeddings during build
       openai_api_key: SecretStr | None = None
       openai_embedding_model: str = "text-embedding-3-small"
       openai_embedding_dimensions: int = 512


   _settings: MaintenanceSettings | None = None


   def get_settings() -> MaintenanceSettings:
       """Get singleton settings instance."""
       global _settings
       if _settings is None:
           _settings = MaintenanceSettings()
       return _settings
   ```

2. Create `packages/oidm-maintenance/src/oidm_maintenance/hashing.py`:

   ```python
   """File hashing utilities."""
   from pathlib import Path

   import pooch


   def compute_file_hash(file_path: Path) -> str:
       """Compute SHA256 hash of a file.

       Args:
           file_path: Path to the file to hash.

       Returns:
           Hash string in format "sha256:abc123..."
       """
       return pooch.file_hash(str(file_path), alg="sha256")
   ```

3. Create `packages/oidm-maintenance/src/oidm_maintenance/s3.py`:

   **Source:** Copy and adapt from `packages/findingmodel/src/findingmodel/db_publish.py`

   The module should contain these functions (read the source file for implementation):
   - `create_s3_client(settings: MaintenanceSettings) -> boto3.client`
   - `upload_file_to_s3(client, bucket: str, key: str, local_path: Path) -> str`
   - `load_manifest_from_s3(client, bucket: str, key: str) -> dict`
   - `backup_manifest(client, bucket: str, manifest: dict, backup_prefix: str) -> str`
   - `save_manifest_to_s3(client, bucket: str, key: str, manifest: dict) -> None`
   - `update_manifest_entry(manifest: dict, db_key: str, entry: dict) -> dict`

   **Key changes from original:**
   - Use `MaintenanceSettings` instead of `FindingModelConfig`
   - Remove findingmodel-specific code (sanity checks, model queries)
   - Make functions pure utilities (no Rich prompts here)

**Verify:** `uv run --package oidm-maintenance python -c "from oidm_maintenance import config, s3, hashing; print('OK')"`

---

## Sub-phase 3.7.3: Create Anatomic Build Module

**What:** Move database build logic from `anatomic_locations/migration.py` to oidm-maintenance.

**Steps:**

1. Read the current implementation:
   - `packages/anatomic-locations/src/anatomic_locations/migration.py`

2. Create `packages/oidm-maintenance/src/oidm_maintenance/anatomic/build.py`:

   ```python
   """Build anatomic-locations database from source data."""
   from pathlib import Path

   import duckdb
   from rich.console import Console
   from rich.progress import Progress

   from oidm_maintenance.config import get_settings

   console = Console()


   def build_anatomic_database(
       source_csv: Path,
       output_path: Path,
       generate_embeddings: bool = True,
   ) -> Path:
       """Build the anatomic-locations DuckDB database.

       Args:
           source_csv: Path to source CSV with anatomic location data.
           output_path: Path for output DuckDB file.
           generate_embeddings: Whether to generate OpenAI embeddings.

       Returns:
           Path to the created database file.
       """
       # Implementation moved from anatomic_locations/migration.py:
       # 1. Load CSV data
       # 2. Create DuckDB connection
       # 3. Create tables (anatomic_locations with id, description, synonyms, region, sided, vector)
       # 4. Insert data
       # 5. Generate embeddings if requested (using oidm_common.embeddings)
       # 6. Create FTS index
       # 7. Create HNSW vector index
       # 8. Return output path
       ...
   ```

   **Copy the actual implementation from `migration.py`, adapting:**
   - Import `get_settings()` from `oidm_maintenance.config`
   - Use `oidm_common.embeddings.generate_embeddings_batch()` for embeddings
   - Add Rich progress bars for user feedback
   - Remove any CLI decorators (those go in cli.py)

**Verify:** Unit test can import the module without errors.

---

## Sub-phase 3.7.4: Create Anatomic Publish Module

**What:** Create publish logic for anatomic-locations database.

**Steps:**

1. Create `packages/oidm-maintenance/src/oidm_maintenance/anatomic/publish.py`:

   ```python
   """Publish anatomic-locations database to S3/Tigris."""
   from pathlib import Path

   import duckdb
   from rich.console import Console
   from rich.prompt import Confirm
   from rich.table import Table

   from oidm_maintenance.config import get_settings
   from oidm_maintenance.hashing import compute_file_hash
   from oidm_maintenance.s3 import (
       create_s3_client,
       load_manifest_from_s3,
       update_manifest_entry,
       backup_manifest,
       save_manifest_to_s3,
       upload_file_to_s3,
   )

   console = Console()


   def get_anatomic_stats(db_path: Path) -> dict:
       """Get statistics about an anatomic database.

       Returns dict with: record_count, sample_descriptions, regions, etc.
       """
       conn = duckdb.connect(str(db_path), read_only=True)
       try:
           count = conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()[0]
           samples = conn.execute(
               "SELECT description FROM anatomic_locations LIMIT 5"
           ).fetchall()
           regions = conn.execute(
               "SELECT DISTINCT region FROM anatomic_locations"
           ).fetchall()
           return {
               "record_count": count,
               "sample_descriptions": [s[0] for s in samples],
               "regions": [r[0] for r in regions],
           }
       finally:
           conn.close()


   def display_anatomic_stats(stats: dict) -> None:
       """Display database statistics using Rich."""
       table = Table(title="Anatomic Database Statistics")
       table.add_column("Metric", style="cyan")
       table.add_column("Value", style="green")
       table.add_row("Record Count", str(stats["record_count"]))
       table.add_row("Regions", ", ".join(stats["regions"]))
       table.add_row("Sample Descriptions", "\n".join(stats["sample_descriptions"][:3]))
       console.print(table)


   def publish_anatomic_database(
       db_path: Path,
       version: str | None = None,
       dry_run: bool = False,
   ) -> bool:
       """Publish anatomic database to S3.

       Args:
           db_path: Path to the DuckDB file to publish.
           version: Version string (default: YYYY-MM-DD).
           dry_run: If True, show what would happen without uploading.

       Returns:
           True if publish succeeded, False if cancelled.
       """
       settings = get_settings()

       # Step 1: Compute hash and gather stats
       console.print("[bold]Step 1:[/bold] Analyzing database...")
       file_hash = compute_file_hash(db_path)
       stats = get_anatomic_stats(db_path)
       display_anatomic_stats(stats)

       console.print(f"\nFile hash: [cyan]{file_hash}[/cyan]")

       if not Confirm.ask("Proceed with upload?"):
           return False

       if dry_run:
           console.print("[yellow]Dry run - no changes made[/yellow]")
           return True

       # Step 2: Upload to S3
       console.print("\n[bold]Step 2:[/bold] Uploading to S3...")
       client = create_s3_client(settings)
       # ... upload logic ...

       # Step 3: Update manifest
       console.print("\n[bold]Step 3:[/bold] Updating manifest...")
       # ... manifest logic ...

       console.print("\n[bold green]✓ Publish complete![/bold green]")
       return True
   ```

**Verify:** Unit test can import the module without errors.

---

## Sub-phase 3.7.5: Create FindingModel Build Module

**What:** Move database build logic from `findingmodel/index.py` write methods to oidm-maintenance.

**Steps:**

1. Read the current implementation:
   - `packages/findingmodel/src/findingmodel/index.py` (look for `setup()`, `ingest()`, `_populate_*` methods)

2. Create `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py`:

   ```python
   """Build findingmodel database from source models."""
   from pathlib import Path

   import duckdb
   from rich.console import Console
   from rich.progress import Progress

   from findingmodel import FindingModel
   from oidm_maintenance.config import get_settings

   console = Console()


   def build_findingmodel_database(
       source_dir: Path,
       output_path: Path,
       generate_embeddings: bool = True,
   ) -> Path:
       """Build the findingmodel DuckDB database.

       Args:
           source_dir: Directory containing .fm.json files.
           output_path: Path for output DuckDB file.
           generate_embeddings: Whether to generate OpenAI embeddings.

       Returns:
           Path to the created database file.
       """
       # Implementation moved from findingmodel/index.py:
       # 1. Discover all .fm.json files in source_dir
       # 2. Load and validate each FindingModel
       # 3. Create DuckDB connection
       # 4. Create tables (finding_models, finding_model_json, etc.)
       # 5. Insert data with embeddings
       # 6. Create FTS index
       # 7. Create HNSW vector index
       # 8. Return output path
       ...
   ```

   **Copy implementation from `index.py`, adapting:**
   - Extract `setup()`, `_create_tables()`, `ingest()`, `_populate_*` methods
   - Use standalone functions instead of class methods
   - Import settings from `oidm_maintenance.config`
   - Add Rich progress bars

**Verify:** Unit test can import the module without errors.

---

## Sub-phase 3.7.6: Create FindingModel Publish Module

**What:** Move publish logic from `findingmodel/db_publish.py` to oidm-maintenance.

**Steps:**

1. Read the current implementation:
   - `packages/findingmodel/src/findingmodel/db_publish.py`

2. Create `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/publish.py`:

   ```python
   """Publish findingmodel database to S3/Tigris."""
   from pathlib import Path

   import duckdb
   from rich.console import Console
   from rich.prompt import Confirm
   from rich.table import Table

   from oidm_maintenance.config import get_settings
   from oidm_maintenance.hashing import compute_file_hash
   from oidm_maintenance.s3 import (
       create_s3_client,
       load_manifest_from_s3,
       update_manifest_entry,
       backup_manifest,
       save_manifest_to_s3,
       upload_file_to_s3,
   )

   console = Console()


   def get_findingmodel_stats(db_path: Path) -> dict:
       """Get statistics about a findingmodel database.

       Returns dict with: model_count, sample_oifm_ids, json_roundtrip_ok, etc.
       """
       # Adapted from db_publish.py: get_database_stats(), run_sanity_check()
       ...


   def display_findingmodel_stats(stats: dict) -> None:
       """Display database statistics using Rich."""
       # Adapted from db_publish.py: prompt_user_confirmation()
       ...


   def publish_findingmodel_database(
       db_path: Path,
       version: str | None = None,
       dry_run: bool = False,
   ) -> bool:
       """Publish findingmodel database to S3.

       Args:
           db_path: Path to the DuckDB file to publish.
           version: Version string (default: YYYY-MM-DD).
           dry_run: If True, show what would happen without uploading.

       Returns:
           True if publish succeeded, False if cancelled.
       """
       # Similar structure to anatomic/publish.py
       # Uses findingmodel-specific stats and sanity checks
       ...
   ```

   **Copy implementation from `db_publish.py`, adapting:**
   - Move `DatabaseStats`, `SanityCheckResult` dataclasses here
   - Move `get_database_stats()`, `run_sanity_check()`, `prompt_user_confirmation()`
   - Use shared S3 functions from `oidm_maintenance.s3`
   - Use `MaintenanceSettings` instead of `FindingModelConfig`

**Verify:** Unit test can import the module without errors.

---

## Sub-phase 3.7.7: Create CLI Module

**What:** Create the unified CLI for all maintenance operations.

**Steps:**

1. Create `packages/oidm-maintenance/src/oidm_maintenance/cli.py`:

   ```python
   """CLI for OIDM maintenance operations."""
   from pathlib import Path

   import click
   from rich.console import Console

   console = Console()


   @click.group()
   @click.version_option()
   def main() -> None:
       """OIDM Maintenance Tools - Build and publish databases."""
       pass


   @main.group()
   def anatomic() -> None:
       """Anatomic-locations database operations."""
       pass


   @anatomic.command()
   @click.option("--source", "-s", type=click.Path(exists=True, path_type=Path), required=True,
                 help="Source CSV file with anatomic location data")
   @click.option("--output", "-o", type=click.Path(path_type=Path), required=True,
                 help="Output DuckDB file path")
   @click.option("--no-embeddings", is_flag=True, help="Skip embedding generation")
   def build(source: Path, output: Path, no_embeddings: bool) -> None:
       """Build anatomic-locations database from source CSV."""
       from oidm_maintenance.anatomic.build import build_anatomic_database

       console.print(f"[bold]Building anatomic database...[/bold]")
       console.print(f"  Source: {source}")
       console.print(f"  Output: {output}")

       result = build_anatomic_database(
           source_csv=source,
           output_path=output,
           generate_embeddings=not no_embeddings,
       )
       console.print(f"\n[bold green]✓ Created:[/bold green] {result}")


   @anatomic.command()
   @click.argument("db_path", type=click.Path(exists=True, path_type=Path))
   @click.option("--version", "-v", help="Version string (default: YYYY-MM-DD)")
   @click.option("--dry-run", is_flag=True, help="Show what would happen without uploading")
   def publish(db_path: Path, version: str | None, dry_run: bool) -> None:
       """Publish anatomic-locations database to S3."""
       from oidm_maintenance.anatomic.publish import publish_anatomic_database

       success = publish_anatomic_database(db_path, version=version, dry_run=dry_run)
       if not success:
           raise SystemExit(1)


   @main.group()
   def findingmodel() -> None:
       """FindingModel database operations."""
       pass


   @findingmodel.command()
   @click.option("--source", "-s", type=click.Path(exists=True, path_type=Path), required=True,
                 help="Source directory containing .fm.json files")
   @click.option("--output", "-o", type=click.Path(path_type=Path), required=True,
                 help="Output DuckDB file path")
   @click.option("--no-embeddings", is_flag=True, help="Skip embedding generation")
   def build(source: Path, output: Path, no_embeddings: bool) -> None:
       """Build findingmodel database from source models."""
       from oidm_maintenance.findingmodel.build import build_findingmodel_database

       console.print(f"[bold]Building findingmodel database...[/bold]")
       console.print(f"  Source: {source}")
       console.print(f"  Output: {output}")

       result = build_findingmodel_database(
           source_dir=source,
           output_path=output,
           generate_embeddings=not no_embeddings,
       )
       console.print(f"\n[bold green]✓ Created:[/bold green] {result}")


   @findingmodel.command()
   @click.argument("db_path", type=click.Path(exists=True, path_type=Path))
   @click.option("--version", "-v", help="Version string (default: YYYY-MM-DD)")
   @click.option("--dry-run", is_flag=True, help="Show what would happen without uploading")
   def publish(db_path: Path, version: str | None, dry_run: bool) -> None:
       """Publish findingmodel database to S3."""
       from oidm_maintenance.findingmodel.publish import publish_findingmodel_database

       success = publish_findingmodel_database(db_path, version=version, dry_run=dry_run)
       if not success:
           raise SystemExit(1)


   if __name__ == "__main__":
       main()
   ```

**Verify:**
```bash
uv run --package oidm-maintenance oidm-maintain --help
uv run --package oidm-maintenance oidm-maintain anatomic --help
uv run --package oidm-maintenance oidm-maintain findingmodel --help
```

---

## Sub-phase 3.7.8: Strip Index Classes to Read-Only

**What:** Remove write methods from user-facing index classes.

**Steps:**

1. **Update `packages/anatomic-locations/src/anatomic_locations/index.py`:**

   Remove these methods (they now live in oidm-maintenance):
   - `setup()` or `create_tables()`
   - `ingest()` or `add_entry()`
   - Any method that writes to the database

   Keep only:
   - `__init__()` - opens connection in read-only mode
   - `search()` - hybrid FTS + vector search
   - `get()` or `get_by_id()` - retrieve by ID
   - `close()` or context manager methods

   Update `__init__` to enforce read-only:
   ```python
   def __init__(self, db_path: Path | None = None) -> None:
       if db_path is None:
           from anatomic_locations.config import ensure_anatomic_db
           db_path = ensure_anatomic_db()
       self.conn = duckdb.connect(str(db_path), read_only=True)  # Always read-only
   ```

2. **Update `packages/findingmodel/src/findingmodel/index.py`:**

   Remove these methods:
   - `setup()`
   - `ingest()`
   - `add_model()`
   - `_create_tables()`
   - `_populate_finding_models()`
   - `_populate_finding_model_json()`
   - `_populate_denormalized_tables()`
   - `_rebuild_indexes()`

   Keep only:
   - `__init__()` - opens connection in read-only mode
   - `search()` - search for finding models
   - `get_full()` - get complete model by ID
   - `validate()` - validate a model against the index
   - Context manager methods

   Update `__init__` to enforce read-only:
   ```python
   def __init__(self, db_path: Path | None = None, read_only: bool = True) -> None:
       # ... existing download logic ...
       self.conn = duckdb.connect(str(self.db_path), read_only=True)  # Always read-only
   ```

**Verify:**
```bash
uv run --package anatomic-locations pytest
uv run --package findingmodel pytest
```

---

## Sub-phase 3.7.9: Remove Old Code from Packages

**What:** Delete the build/publish code that has been moved to oidm-maintenance.

**Steps:**

1. **Delete `packages/findingmodel/src/findingmodel/db_publish.py`**

2. **Delete `packages/anatomic-locations/src/anatomic_locations/migration.py`** (if it exists as separate file)

3. **Update `packages/findingmodel/src/findingmodel/__init__.py`:**
   - Remove any exports of `db_publish` functions

4. **Update `packages/findingmodel/src/findingmodel/cli.py`:**
   - Remove any `publish` or `build` commands that referenced db_publish

5. **Update `packages/anatomic-locations/src/anatomic_locations/cli.py`:**
   - Remove `anatomic build` command
   - Keep only: `anatomic search`, `anatomic show`

6. **Update `packages/findingmodel/pyproject.toml`:**
   - Remove `boto3` from dependencies (if present)
   - Remove `boto3-stubs` from dependencies (if present)

7. **Update `packages/anatomic-locations/pyproject.toml`:**
   - Remove `openai` from dependencies (only needed for builds)
   - Remove `[build]` optional dependency group if it exists

**Verify:**
```bash
uv sync
uv run --package anatomic-locations pytest
uv run --package findingmodel pytest
# Ensure boto3 is NOT installed when only user packages are installed
```

---

## Sub-phase 3.7.10: Update Taskfile

**What:** Add convenient task targets for maintenance operations.

**Steps:**

1. Add to `Taskfile.yml`:

   ```yaml
   # Maintenance tasks
   maintain:anatomic:build:
     desc: "Build anatomic-locations database"
     cmds:
       - uv run --package oidm-maintenance oidm-maintain anatomic build {{.CLI_ARGS}}

   maintain:anatomic:publish:
     desc: "Publish anatomic-locations database to S3"
     cmds:
       - uv run --package oidm-maintenance oidm-maintain anatomic publish {{.CLI_ARGS}}

   maintain:findingmodel:build:
     desc: "Build findingmodel database"
     cmds:
       - uv run --package oidm-maintenance oidm-maintain findingmodel build {{.CLI_ARGS}}

   maintain:findingmodel:publish:
     desc: "Publish findingmodel database to S3"
     cmds:
       - uv run --package oidm-maintenance oidm-maintain findingmodel publish {{.CLI_ARGS}}
   ```

**Verify:**
```bash
task maintain:anatomic:build --help
task maintain:findingmodel:build --help
```

---

## Sub-phase 3.7.11: Create Tests for oidm-maintenance

**What:** Add tests for the maintenance package.

**Steps:**

1. Create `packages/oidm-maintenance/tests/conftest.py`:
   ```python
   """Test fixtures for oidm-maintenance."""
   import pytest
   from pathlib import Path
   import tempfile


   @pytest.fixture
   def temp_db_path(tmp_path: Path) -> Path:
       """Temporary path for test databases."""
       return tmp_path / "test.duckdb"
   ```

2. Create `packages/oidm-maintenance/tests/test_hashing.py`:
   ```python
   """Tests for hashing module."""
   from pathlib import Path
   from oidm_maintenance.hashing import compute_file_hash


   def test_compute_file_hash(tmp_path: Path) -> None:
       """Test that file hashing works."""
       test_file = tmp_path / "test.txt"
       test_file.write_text("hello world")

       hash_result = compute_file_hash(test_file)

       assert hash_result.startswith("sha256:")
       assert len(hash_result) == 71  # "sha256:" + 64 hex chars
   ```

3. Create `packages/oidm-maintenance/tests/test_s3.py`:
   ```python
   """Tests for S3 module (mocked)."""
   from unittest.mock import MagicMock, patch

   from oidm_maintenance.s3 import create_s3_client
   from oidm_maintenance.config import MaintenanceSettings


   def test_create_s3_client_with_credentials() -> None:
       """Test S3 client creation with credentials."""
       settings = MaintenanceSettings(
           aws_access_key_id="test-key",
           aws_secret_access_key="test-secret",
       )

       with patch("boto3.client") as mock_client:
           create_s3_client(settings)

           mock_client.assert_called_once()
           call_kwargs = mock_client.call_args.kwargs
           assert call_kwargs["service_name"] == "s3"
           assert call_kwargs["endpoint_url"] == settings.s3_endpoint_url
   ```

4. Create `packages/oidm-maintenance/tests/test_anatomic_build.py`:
   ```python
   """Tests for anatomic build module."""
   import pytest
   from pathlib import Path
   from unittest.mock import patch


   def test_build_anatomic_database_creates_file(tmp_path: Path) -> None:
       """Test that build creates a database file."""
       # Create minimal test CSV
       source_csv = tmp_path / "source.csv"
       source_csv.write_text("id,description,region,sided\n1,Test Location,head,nonlateral\n")

       output_path = tmp_path / "anatomic.duckdb"

       with patch("oidm_maintenance.anatomic.build.generate_embeddings_batch", return_value=[[0.0] * 512]):
           from oidm_maintenance.anatomic.build import build_anatomic_database
           result = build_anatomic_database(source_csv, output_path, generate_embeddings=False)

       assert result.exists()
       assert result == output_path
   ```

**Verify:**
```bash
uv run --package oidm-maintenance pytest
```

---

## Sub-phase 3.7.12: Final Verification

**What:** Ensure everything works together.

**Verification checklist:**

1. **Package installation:**
   ```bash
   uv sync
   # All packages should install without errors
   ```

2. **User packages are read-only:**
   ```bash
   # These should work (read operations)
   uv run --package anatomic-locations python -c "from anatomic_locations import AnatomicLocationIndex; print('OK')"
   uv run --package findingmodel python -c "from findingmodel import DuckDBIndex; print('OK')"

   # Verify no write methods exist
   uv run --package anatomic-locations python -c "
   from anatomic_locations.index import AnatomicLocationIndex
   idx = AnatomicLocationIndex.__new__(AnatomicLocationIndex)
   assert not hasattr(idx, 'setup'), 'setup() should not exist'
   assert not hasattr(idx, 'ingest'), 'ingest() should not exist'
   print('OK - no write methods')
   "
   ```

3. **Maintenance CLI works:**
   ```bash
   uv run --package oidm-maintenance oidm-maintain --help
   uv run --package oidm-maintenance oidm-maintain anatomic --help
   uv run --package oidm-maintenance oidm-maintain anatomic build --help
   uv run --package oidm-maintenance oidm-maintain findingmodel --help
   uv run --package oidm-maintenance oidm-maintain findingmodel build --help
   ```

4. **All tests pass:**
   ```bash
   uv run --package oidm-common pytest
   uv run --package anatomic-locations pytest
   uv run --package findingmodel pytest
   uv run --package oidm-maintenance pytest
   ```

5. **Heavy dependencies isolated:**
   ```bash
   # boto3 should NOT be in user package dependencies
   grep -r "boto3" packages/findingmodel/pyproject.toml && echo "FAIL: boto3 in findingmodel" || echo "OK"
   grep -r "boto3" packages/anatomic-locations/pyproject.toml && echo "FAIL: boto3 in anatomic" || echo "OK"

   # boto3 SHOULD be in oidm-maintenance
   grep "boto3" packages/oidm-maintenance/pyproject.toml && echo "OK" || echo "FAIL: boto3 missing from oidm-maintenance"
   ```

6. **Taskfile works:**
   ```bash
   task maintain:anatomic:build -- --help
   task maintain:findingmodel:build -- --help
   ```

---

## Summary

| Sub-phase | Description | Key Files |
|-----------|-------------|-----------|
| 3.7.1 | Create package scaffolding | `pyproject.toml`, `__init__.py` |
| 3.7.2 | Create shared infrastructure | `config.py`, `s3.py`, `hashing.py` |
| 3.7.3 | Create anatomic build module | `anatomic/build.py` |
| 3.7.4 | Create anatomic publish module | `anatomic/publish.py` |
| 3.7.5 | Create findingmodel build module | `findingmodel/build.py` |
| 3.7.6 | Create findingmodel publish module | `findingmodel/publish.py` |
| 3.7.7 | Create CLI module | `cli.py` |
| 3.7.8 | Strip index classes to read-only | `index.py` in both packages |
| 3.7.9 | Remove old code from packages | Delete `db_publish.py`, `migration.py` |
| 3.7.10 | Update Taskfile | `Taskfile.yml` |
| 3.7.11 | Create tests | `tests/*.py` |
| 3.7.12 | Final verification | Run all checks |

**Total files to create:** ~15 new files in oidm-maintenance
**Total files to modify:** ~6 files in existing packages
**Total files to delete:** ~2 files
