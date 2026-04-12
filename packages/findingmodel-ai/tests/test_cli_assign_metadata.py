"""Tests for `findingmodel-ai assign-metadata` CLI command."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from findingmodel import FindingModelFull
from findingmodel_ai.cli import cli
from findingmodel_ai.metadata.types import MetadataAssignmentResult, MetadataAssignmentReview


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_model_path() -> Path:
    return Path(__file__).parent / "data" / "pulmonary_embolism.fm.json"


def _result_for(model: FindingModelFull, *, trace_id: str | None = None) -> MetadataAssignmentResult:
    return MetadataAssignmentResult(
        model=model,
        review=MetadataAssignmentReview(
            oifm_id=model.oifm_id,
            finding_name=model.name,
            assignment_timestamp=datetime.now(tz=UTC),
            model_used="openai:gpt-5-mini",
            logfire_trace_id=trace_id,
        ),
    )


def test_assign_metadata_cli_no_logfire_by_default(runner: CliRunner, sample_model_path: Path) -> None:
    model = FindingModelFull.model_validate_json(sample_model_path.read_text())
    result_obj = _result_for(model)

    with (
        patch("findingmodel_ai.cli.settings") as mock_settings,
        patch("findingmodel_ai.cli.assign_metadata") as mock_assign_metadata,
        patch("findingmodel_ai.cli.ensure_logfire_configured") as mock_ensure_logfire,
    ):
        mock_settings.validate_default_model_keys = MagicMock()
        mock_assign_metadata.return_value = result_obj

        result = runner.invoke(cli, ["assign-metadata", str(sample_model_path)])

    assert result.exit_code == 0
    mock_assign_metadata.assert_awaited_once()
    mock_ensure_logfire.assert_not_called()
    assert '"name": "pulmonary embolism"' in result.output


def test_assign_metadata_cli_enables_logfire_and_writes_outputs(runner: CliRunner, sample_model_path: Path) -> None:
    model = FindingModelFull.model_validate_json(sample_model_path.read_text())
    result_obj = _result_for(model, trace_id="trace-123")

    with runner.isolated_filesystem():
        output_path = Path("updated.fm.json")
        review_path = Path("updated.metadata-review.json")

        with (
            patch("findingmodel_ai.cli.settings") as mock_settings,
            patch("findingmodel_ai.cli.assign_metadata") as mock_assign_metadata,
            patch("findingmodel_ai.cli.ensure_logfire_configured") as mock_ensure_logfire,
        ):
            mock_settings.validate_default_model_keys = MagicMock()
            mock_assign_metadata.return_value = result_obj

            result = runner.invoke(
                cli,
                [
                    "assign-metadata",
                    str(sample_model_path),
                    "--output",
                    str(output_path),
                    "--review-output",
                    str(review_path),
                    "--logfire",
                ],
            )

        assert result.exit_code == 0
        mock_ensure_logfire.assert_called_once_with(console=False)
        assert output_path.exists()
        assert review_path.exists()
        assert "Logfire trace_id: trace-123" in result.stderr
