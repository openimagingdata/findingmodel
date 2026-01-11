"""Tests for oidm-maintenance CLI commands.

This module tests the Click CLI commands for building and publishing
findingmodel and anatomic-location databases. Tests use Click's CliRunner
for isolated command invocation and mock external dependencies.

All tests use mocked embeddings to avoid API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from oidm_maintenance.cli import main
from pydantic_ai import models

# Block all AI model requests - embeddings are mocked
models.ALLOW_MODEL_REQUESTS = False


# =============================================================================
# Fixtures
# =============================================================================


def _fake_embeddings_deterministic(texts: list[str]) -> list[list[float]]:
    """Generate deterministic fake embeddings based on text hash.

    Args:
        texts: List of texts to embed

    Returns:
        List of deterministic embedding vectors based on text hash
    """
    return [[(sum(ord(c) for c in text) % 100) / 100.0] * 512 for text in texts]


@pytest.fixture
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def findingmodel_source_dir() -> Path:
    """Path to findingmodel test data."""
    return Path(__file__).parent.parent.parent / "findingmodel" / "tests" / "data" / "defs"


@pytest.fixture
def anatomic_source_file() -> Path:
    """Path to anatomic-locations test data."""
    return Path(__file__).parent.parent.parent / "anatomic-locations" / "tests" / "data" / "anatomic_sample.json"


# =============================================================================
# FindingModel Build Command Tests
# =============================================================================


class TestFindingModelBuildCommand:
    """Tests for findingmodel build CLI command."""

    def test_findingmodel_build_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
        findingmodel_source_dir: Path,
    ) -> None:
        """Build command creates database with mocked embeddings."""
        output = tmp_path / "test.duckdb"

        with patch(
            "oidm_maintenance.findingmodel.build._generate_embeddings_async",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ):
            result = runner.invoke(
                main,
                [
                    "findingmodel",
                    "build",
                    "--source",
                    str(findingmodel_source_dir),
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()
        assert "Building findingmodel database" in result.output
        assert "Created:" in result.output

    def test_findingmodel_build_with_no_embeddings_flag(
        self,
        runner: CliRunner,
        tmp_path: Path,
        findingmodel_source_dir: Path,
    ) -> None:
        """Build command with --no-embeddings creates database with zero vectors."""
        output = tmp_path / "test.duckdb"

        result = runner.invoke(
            main,
            [
                "findingmodel",
                "build",
                "--source",
                str(findingmodel_source_dir),
                "--output",
                str(output),
                "--no-embeddings",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()
        assert "Building findingmodel database" in result.output

    def test_findingmodel_build_missing_source(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Build command fails when source directory does not exist."""
        nonexistent_source = tmp_path / "nonexistent"
        output = tmp_path / "test.duckdb"

        result = runner.invoke(
            main,
            [
                "findingmodel",
                "build",
                "--source",
                str(nonexistent_source),
                "--output",
                str(output),
            ],
        )

        # Click validates path exists, so should fail at CLI level
        assert result.exit_code != 0
        # Click will report path doesn't exist
        assert "does not exist" in result.output or "Invalid value" in result.output

    def test_findingmodel_build_short_options(
        self,
        runner: CliRunner,
        tmp_path: Path,
        findingmodel_source_dir: Path,
    ) -> None:
        """Build command works with short option flags -s and -o."""
        output = tmp_path / "test.duckdb"

        with patch(
            "oidm_maintenance.findingmodel.build._generate_embeddings_async",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ):
            result = runner.invoke(
                main,
                [
                    "findingmodel",
                    "build",
                    "-s",
                    str(findingmodel_source_dir),
                    "-o",
                    str(output),
                ],
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

    def test_findingmodel_build_creates_parent_directories(
        self,
        runner: CliRunner,
        tmp_path: Path,
        findingmodel_source_dir: Path,
    ) -> None:
        """Build command creates parent directories if they don't exist."""
        nested_output = tmp_path / "nested" / "path" / "test.duckdb"

        with patch(
            "oidm_maintenance.findingmodel.build._generate_embeddings_async",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ):
            result = runner.invoke(
                main,
                [
                    "findingmodel",
                    "build",
                    "-s",
                    str(findingmodel_source_dir),
                    "-o",
                    str(nested_output),
                ],
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert nested_output.exists()
        assert nested_output.parent.exists()


# =============================================================================
# FindingModel Publish Command Tests
# =============================================================================


class TestFindingModelPublishCommand:
    """Tests for findingmodel publish CLI command."""

    def test_findingmodel_publish_dry_run(
        self,
        runner: CliRunner,
        tmp_path: Path,
        findingmodel_source_dir: Path,
    ) -> None:
        """Publish command with --dry-run shows what would happen without uploading."""
        # First create a database to publish
        db_path = tmp_path / "test.duckdb"

        with patch(
            "oidm_maintenance.findingmodel.build._generate_embeddings_async",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ):
            build_result = runner.invoke(
                main,
                [
                    "findingmodel",
                    "build",
                    "-s",
                    str(findingmodel_source_dir),
                    "-o",
                    str(db_path),
                    "--no-embeddings",
                ],
            )

        assert build_result.exit_code == 0

        # Mock the publish function to verify it's called correctly
        with patch("oidm_maintenance.findingmodel.publish.publish_findingmodel_database") as mock_publish:
            mock_publish.return_value = True

            result = runner.invoke(
                main,
                [
                    "findingmodel",
                    "publish",
                    str(db_path),
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0
        # Verify publish function was called with correct arguments
        mock_publish.assert_called_once()
        args, kwargs = mock_publish.call_args
        assert args[0] == db_path
        assert kwargs["dry_run"] is True

    def test_findingmodel_publish_with_version(
        self,
        runner: CliRunner,
        tmp_path: Path,
        findingmodel_source_dir: Path,
    ) -> None:
        """Publish command accepts custom version string."""
        # First create a database to publish
        db_path = tmp_path / "test.duckdb"

        with patch(
            "oidm_maintenance.findingmodel.build._generate_embeddings_async",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ):
            build_result = runner.invoke(
                main,
                [
                    "findingmodel",
                    "build",
                    "-s",
                    str(findingmodel_source_dir),
                    "-o",
                    str(db_path),
                    "--no-embeddings",
                ],
            )

        assert build_result.exit_code == 0

        # Mock the publish function
        with patch("oidm_maintenance.findingmodel.publish.publish_findingmodel_database") as mock_publish:
            mock_publish.return_value = True

            result = runner.invoke(
                main,
                [
                    "findingmodel",
                    "publish",
                    str(db_path),
                    "--version",
                    "2025-01-15",
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0
        # Verify version was passed
        _args, kwargs = mock_publish.call_args
        assert kwargs["version"] == "2025-01-15"

    def test_findingmodel_publish_fails_with_missing_db(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Publish command fails when database file does not exist."""
        nonexistent_db = tmp_path / "nonexistent.duckdb"

        result = runner.invoke(
            main,
            [
                "findingmodel",
                "publish",
                str(nonexistent_db),
                "--dry-run",
            ],
        )

        # Click validates path exists, so should fail at CLI level
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Invalid value" in result.output

    def test_findingmodel_publish_propagates_failure(
        self,
        runner: CliRunner,
        tmp_path: Path,
        findingmodel_source_dir: Path,
    ) -> None:
        """Publish command exits with code 1 when publish function fails."""
        # First create a database to publish
        db_path = tmp_path / "test.duckdb"

        with patch(
            "oidm_maintenance.findingmodel.build._generate_embeddings_async",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ):
            build_result = runner.invoke(
                main,
                [
                    "findingmodel",
                    "build",
                    "-s",
                    str(findingmodel_source_dir),
                    "-o",
                    str(db_path),
                    "--no-embeddings",
                ],
            )

        assert build_result.exit_code == 0

        # Mock publish to fail
        with patch("oidm_maintenance.findingmodel.publish.publish_findingmodel_database") as mock_publish:
            mock_publish.return_value = False

            result = runner.invoke(
                main,
                [
                    "findingmodel",
                    "publish",
                    str(db_path),
                    "--dry-run",
                ],
            )

        assert result.exit_code == 1


# =============================================================================
# Anatomic Build Command Tests
# =============================================================================


class TestAnatomicBuildCommand:
    """Tests for anatomic build CLI command."""

    def test_anatomic_build_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
        anatomic_source_file: Path,
    ) -> None:
        """Anatomic build command creates database with mocked embeddings."""
        if not anatomic_source_file.exists():
            pytest.skip("Anatomic test data not available")

        output = tmp_path / "anatomic_test.duckdb"

        # Mock settings to provide fake API key
        mock_settings = MagicMock()
        mock_settings.openai_api_key = MagicMock()
        mock_settings.openai_api_key.get_secret_value.return_value = "fake-key"
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.openai_embedding_dimensions = 512

        # Generate fake embeddings for the expected number of records
        def fake_batch_embeddings(texts: list[str], client: Any, model: str, dimensions: int) -> list[list[float]]:
            return _fake_embeddings_deterministic(texts)

        with (
            patch("oidm_maintenance.anatomic.build.get_settings", return_value=mock_settings),
            patch("oidm_maintenance.anatomic.build.generate_embeddings_batch", side_effect=fake_batch_embeddings),
        ):
            result = runner.invoke(
                main,
                [
                    "anatomic",
                    "build",
                    "--source",
                    str(anatomic_source_file),
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()
        assert "Building anatomic database" in result.output
        assert "Created:" in result.output

    def test_anatomic_build_with_no_embeddings_flag(
        self,
        runner: CliRunner,
        tmp_path: Path,
        anatomic_source_file: Path,
    ) -> None:
        """Anatomic build command with --no-embeddings creates database with zero vectors."""
        if not anatomic_source_file.exists():
            pytest.skip("Anatomic test data not available")

        output = tmp_path / "anatomic_test.duckdb"

        result = runner.invoke(
            main,
            [
                "anatomic",
                "build",
                "--source",
                str(anatomic_source_file),
                "--output",
                str(output),
                "--no-embeddings",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()
        assert "Building anatomic database" in result.output

    def test_anatomic_build_missing_source(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Anatomic build command fails when source file does not exist."""
        nonexistent_source = tmp_path / "nonexistent.json"
        output = tmp_path / "anatomic_test.duckdb"

        result = runner.invoke(
            main,
            [
                "anatomic",
                "build",
                "--source",
                str(nonexistent_source),
                "--output",
                str(output),
            ],
        )

        # Click validates path exists, so should fail at CLI level
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Invalid value" in result.output

    def test_anatomic_build_short_options(
        self,
        runner: CliRunner,
        tmp_path: Path,
        anatomic_source_file: Path,
    ) -> None:
        """Anatomic build command works with short option flags -s and -o."""
        if not anatomic_source_file.exists():
            pytest.skip("Anatomic test data not available")

        output = tmp_path / "anatomic_test.duckdb"

        # Mock settings to provide fake API key
        mock_settings = MagicMock()
        mock_settings.openai_api_key = MagicMock()
        mock_settings.openai_api_key.get_secret_value.return_value = "fake-key"
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.openai_embedding_dimensions = 512

        def fake_batch_embeddings(texts: list[str], client: Any, model: str, dimensions: int) -> list[list[float]]:
            return _fake_embeddings_deterministic(texts)

        with (
            patch("oidm_maintenance.anatomic.build.get_settings", return_value=mock_settings),
            patch("oidm_maintenance.anatomic.build.generate_embeddings_batch", side_effect=fake_batch_embeddings),
        ):
            result = runner.invoke(
                main,
                [
                    "anatomic",
                    "build",
                    "-s",
                    str(anatomic_source_file),
                    "-o",
                    str(output),
                ],
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()


# =============================================================================
# Anatomic Publish Command Tests
# =============================================================================


class TestAnatomicPublishCommand:
    """Tests for anatomic publish CLI command."""

    def test_anatomic_publish_dry_run(
        self,
        runner: CliRunner,
        tmp_path: Path,
        anatomic_source_file: Path,
    ) -> None:
        """Anatomic publish command with --dry-run shows what would happen without uploading."""
        if not anatomic_source_file.exists():
            pytest.skip("Anatomic test data not available")

        # First create a database to publish
        db_path = tmp_path / "anatomic_test.duckdb"

        build_result = runner.invoke(
            main,
            [
                "anatomic",
                "build",
                "-s",
                str(anatomic_source_file),
                "-o",
                str(db_path),
                "--no-embeddings",
            ],
        )

        assert build_result.exit_code == 0

        # Mock the publish function
        with patch("oidm_maintenance.anatomic.publish.publish_anatomic_database") as mock_publish:
            mock_publish.return_value = True

            result = runner.invoke(
                main,
                [
                    "anatomic",
                    "publish",
                    str(db_path),
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0
        # Verify publish function was called with correct arguments
        mock_publish.assert_called_once()
        args, kwargs = mock_publish.call_args
        assert args[0] == db_path
        assert kwargs["dry_run"] is True

    def test_anatomic_publish_with_version(
        self,
        runner: CliRunner,
        tmp_path: Path,
        anatomic_source_file: Path,
    ) -> None:
        """Anatomic publish command accepts custom version string."""
        if not anatomic_source_file.exists():
            pytest.skip("Anatomic test data not available")

        # First create a database to publish
        db_path = tmp_path / "anatomic_test.duckdb"

        build_result = runner.invoke(
            main,
            [
                "anatomic",
                "build",
                "-s",
                str(anatomic_source_file),
                "-o",
                str(db_path),
                "--no-embeddings",
            ],
        )

        assert build_result.exit_code == 0

        # Mock the publish function
        with patch("oidm_maintenance.anatomic.publish.publish_anatomic_database") as mock_publish:
            mock_publish.return_value = True

            result = runner.invoke(
                main,
                [
                    "anatomic",
                    "publish",
                    str(db_path),
                    "--version",
                    "2025-01-15",
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0
        # Verify version was passed
        _args, kwargs = mock_publish.call_args
        assert kwargs["version"] == "2025-01-15"

    def test_anatomic_publish_propagates_failure(
        self,
        runner: CliRunner,
        tmp_path: Path,
        anatomic_source_file: Path,
    ) -> None:
        """Anatomic publish command exits with code 1 when publish function fails."""
        if not anatomic_source_file.exists():
            pytest.skip("Anatomic test data not available")

        # First create a database to publish
        db_path = tmp_path / "anatomic_test.duckdb"

        build_result = runner.invoke(
            main,
            [
                "anatomic",
                "build",
                "-s",
                str(anatomic_source_file),
                "-o",
                str(db_path),
                "--no-embeddings",
            ],
        )

        assert build_result.exit_code == 0

        # Mock publish to fail
        with patch("oidm_maintenance.anatomic.publish.publish_anatomic_database") as mock_publish:
            mock_publish.return_value = False

            result = runner.invoke(
                main,
                [
                    "anatomic",
                    "publish",
                    str(db_path),
                    "--dry-run",
                ],
            )

        assert result.exit_code == 1


# =============================================================================
# Version and Help Tests
# =============================================================================


class TestCLIMetadata:
    """Tests for CLI metadata and help output."""

    def test_main_help(self, runner: CliRunner) -> None:
        """Main command shows help text."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "OIDM Maintenance Tools" in result.output
        assert "findingmodel" in result.output
        assert "anatomic" in result.output

    def test_findingmodel_help(self, runner: CliRunner) -> None:
        """Findingmodel subcommand shows help text."""
        result = runner.invoke(main, ["findingmodel", "--help"])

        assert result.exit_code == 0
        assert "FindingModel database operations" in result.output
        assert "build" in result.output
        assert "publish" in result.output

    def test_anatomic_help(self, runner: CliRunner) -> None:
        """Anatomic subcommand shows help text."""
        result = runner.invoke(main, ["anatomic", "--help"])

        assert result.exit_code == 0
        assert "Anatomic-locations database operations" in result.output
        assert "build" in result.output
        assert "publish" in result.output

    def test_findingmodel_build_help(self, runner: CliRunner) -> None:
        """Findingmodel build command shows help text."""
        result = runner.invoke(main, ["findingmodel", "build", "--help"])

        assert result.exit_code == 0
        assert "--source" in result.output
        assert "--output" in result.output
        assert "--no-embeddings" in result.output

    def test_anatomic_build_help(self, runner: CliRunner) -> None:
        """Anatomic build command shows help text."""
        result = runner.invoke(main, ["anatomic", "build", "--help"])

        assert result.exit_code == 0
        assert "--source" in result.output
        assert "--output" in result.output
        assert "--no-embeddings" in result.output

    def test_version_option(self, runner: CliRunner) -> None:
        """Main command shows version with --version."""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        # Version output should contain some version info (format may vary)
        assert len(result.output) > 0
