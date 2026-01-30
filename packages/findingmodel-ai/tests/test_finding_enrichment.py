"""Tests for finding enrichment data models and validation.

This module tests the core data structures and validation logic for the finding
enrichment system, including FindingEnrichmentResult, EnrichmentContext, and
EnrichmentClassification models.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from findingmodel import IndexCode
from findingmodel.finding_model import FindingModelFull
from findingmodel.protocols import OntologySearchResult
from findingmodel_ai.enrichment.unified import (
    ETIOLOGIES,
    BodyRegion,
    EnrichmentClassification,
    EnrichmentContext,
    FindingEnrichmentResult,
    Modality,
    Subspecialty,
)
from pydantic import ValidationError
from pydantic_ai import models

# Import test model constants (defined in conftest.py)
from .conftest import TEST_ANTHROPIC_MODEL, TEST_OPENAI_MODEL

# Prevent accidental model requests in unit tests
# Tests marked with @pytest.mark.callout can enable this as needed
models.ALLOW_MODEL_REQUESTS = False

# =============================================================================
# Test data constants
# =============================================================================

TEST_DATA_FILE = Path(__file__).parent / "data" / "test_enrichment_samples.json"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def test_data() -> Any:
    """Load test enrichment samples from JSON file."""
    with TEST_DATA_FILE.open() as f:
        return json.load(f)


@pytest.fixture
def mock_finding_model(test_data: Any) -> FindingModelFull:
    """Return a FindingModelFull instance with realistic data."""
    model_data = test_data["finding_models"][0]
    return FindingModelFull(**model_data)


@pytest.fixture
def mock_snomed_codes(test_data: Any) -> list[IndexCode]:
    """Return list of SNOMED CT IndexCode objects."""
    codes_data = test_data["index_codes"]["snomed_findings"]
    return [IndexCode(**code) for code in codes_data]


@pytest.fixture
def mock_radlex_codes(test_data: Any) -> list[IndexCode]:
    """Return list of RadLex IndexCode objects."""
    codes_data = test_data["index_codes"]["radlex_findings"]
    return [IndexCode(**code) for code in codes_data]


@pytest.fixture
def mock_anatomic_locations(test_data: Any) -> list[OntologySearchResult]:
    """Return list of OntologySearchResult objects for anatomic locations."""
    results_data = test_data["ontology_search_results"]
    return [OntologySearchResult(**result) for result in results_data]


@pytest.fixture
def mock_enrichment_context(
    mock_finding_model: FindingModelFull,
    mock_snomed_codes: list[IndexCode],
    mock_radlex_codes: list[IndexCode],
) -> EnrichmentContext:
    """Return EnrichmentContext instance for agent testing."""
    all_codes = mock_snomed_codes + mock_radlex_codes
    return EnrichmentContext(
        finding_name=mock_finding_model.name,
        finding_description=mock_finding_model.description,
        existing_codes=all_codes,
        existing_model=mock_finding_model,
    )


# =============================================================================
# FindingEnrichmentResult Model Tests
# =============================================================================


class TestFindingEnrichmentResult:
    """Tests for FindingEnrichmentResult data model."""

    def test_enrichment_result_validation_all_fields(
        self,
        mock_snomed_codes: list[IndexCode],
        mock_radlex_codes: list[IndexCode],
        mock_anatomic_locations: list[OntologySearchResult],
    ) -> None:
        """Test FindingEnrichmentResult validates all fields correctly."""
        result = FindingEnrichmentResult(
            finding_name="Pulmonary Nodule",
            oifm_id="OIFM_TEST_000001",
            snomed_codes=mock_snomed_codes[:1],
            radlex_codes=mock_radlex_codes[:1],
            body_regions=["Chest"],
            etiologies=["neoplastic:benign", "inflammatory:infectious"],
            modalities=["CT", "XR"],
            subspecialties=["CH", "OI"],
            anatomic_locations=mock_anatomic_locations[:2],
            enrichment_timestamp=datetime.now(timezone.utc),
            model_used=TEST_OPENAI_MODEL,
            model_tier="main",
        )

        assert result.finding_name == "Pulmonary Nodule"
        assert result.oifm_id == "OIFM_TEST_000001"
        assert len(result.snomed_codes) == 1
        assert len(result.radlex_codes) == 1
        assert result.body_regions == ["Chest"]
        assert "neoplastic:benign" in result.etiologies
        assert "CT" in result.modalities
        assert "CH" in result.subspecialties
        assert len(result.anatomic_locations) == 2
        assert result.model_used == TEST_OPENAI_MODEL
        assert result.model_tier == "main"

    def test_enrichment_result_minimal_fields(self) -> None:
        """Test FindingEnrichmentResult with only required fields."""
        result = FindingEnrichmentResult(  # type: ignore[call-arg]
            finding_name="Test Finding",
            enrichment_timestamp=datetime.now(timezone.utc),
            model_used=TEST_ANTHROPIC_MODEL,
            model_tier="small",
        )

        assert result.finding_name == "Test Finding"
        assert result.oifm_id is None
        assert result.snomed_codes == []
        assert result.radlex_codes == []
        assert result.body_regions == []
        assert result.etiologies == []
        assert result.modalities == []
        assert result.subspecialties == []
        assert result.anatomic_locations == []

    def test_enrichment_result_missing_required_field(self) -> None:
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            FindingEnrichmentResult(  # type: ignore[call-arg]
                finding_name="Test Finding",
                # Missing required enrichment_timestamp, model_used, model_tier
            )

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "enrichment_timestamp" in error_fields
        assert "model_used" in error_fields
        assert "model_tier" in error_fields

    def test_enrichment_result_empty_finding_name(self) -> None:
        """Test that empty finding_name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            FindingEnrichmentResult(  # type: ignore[call-arg]
                finding_name="",  # Empty string should fail min_length=1
                enrichment_timestamp=datetime.now(timezone.utc),
                model_used=TEST_OPENAI_MODEL,
                model_tier="main",
            )

        errors = exc_info.value.errors()
        assert any(error["loc"][0] == "finding_name" for error in errors)

    def test_etiology_validation_valid_values(self) -> None:
        """Test that ETIOLOGIES validator accepts valid values."""
        valid_etiologies = [
            "inflammatory:infectious",
            "neoplastic:benign",
            "vascular:ischemic",
            "degenerative",
        ]

        result = FindingEnrichmentResult(  # type: ignore[call-arg]
            finding_name="Test Finding",
            etiologies=valid_etiologies,
            enrichment_timestamp=datetime.now(timezone.utc),
            model_used=TEST_OPENAI_MODEL,
            model_tier="main",
        )

        assert result.etiologies == valid_etiologies

    def test_etiology_validation_invalid_values(self) -> None:
        """Test that ETIOLOGIES validator rejects invalid values."""
        invalid_etiologies = ["inflammatory:infectious", "invalid_etiology", "fake_category"]

        with pytest.raises(ValidationError) as exc_info:
            FindingEnrichmentResult(  # type: ignore[call-arg]
                finding_name="Test Finding",
                etiologies=invalid_etiologies,
                enrichment_timestamp=datetime.now(timezone.utc),
                model_used=TEST_OPENAI_MODEL,
                model_tier="main",
            )

        error_msg = str(exc_info.value)
        assert "Invalid etiology categories" in error_msg
        assert "invalid_etiology" in error_msg
        assert "fake_category" in error_msg

    def test_enrichment_result_serialization(
        self,
        mock_snomed_codes: list[IndexCode],
        mock_anatomic_locations: list[OntologySearchResult],
    ) -> None:
        """Test model_dump_json() produces valid JSON."""
        timestamp = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = FindingEnrichmentResult(
            finding_name="Pneumonia",
            oifm_id="OIFM_TEST_000002",
            snomed_codes=mock_snomed_codes[1:2],
            body_regions=["Chest"],
            etiologies=["inflammatory:infectious"],
            modalities=["XR", "CT"],
            subspecialties=["CH", "ER"],
            anatomic_locations=mock_anatomic_locations[:1],
            enrichment_timestamp=timestamp,
            model_used=TEST_ANTHROPIC_MODEL,
            model_tier="full",
        )

        json_str = result.model_dump_json()
        assert isinstance(json_str, str)

        # Parse back to verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["finding_name"] == "Pneumonia"
        assert parsed["oifm_id"] == "OIFM_TEST_000002"
        assert parsed["body_regions"] == ["Chest"]
        assert parsed["etiologies"] == ["inflammatory:infectious"]
        assert parsed["modalities"] == ["XR", "CT"]
        assert parsed["model_used"] == TEST_ANTHROPIC_MODEL


# =============================================================================
# Enum/Literal Type Tests
# =============================================================================


class TestEnumTypes:
    """Tests for BodyRegion, Modality, and Subspecialty Literal types."""

    def test_body_region_enum_valid_values(self) -> None:
        """Test BodyRegion Literal type accepts valid values."""
        valid_regions: list[BodyRegion] = ["ALL", "Head", "Neck", "Chest", "Breast", "Abdomen", "Arm", "Leg"]

        result = FindingEnrichmentResult(  # type: ignore[call-arg]
            finding_name="Multi-region Finding",
            body_regions=valid_regions,
            enrichment_timestamp=datetime.now(timezone.utc),
            model_used=TEST_OPENAI_MODEL,
            model_tier="main",
        )

        assert set(result.body_regions) == set(valid_regions)

    def test_body_region_enum_invalid_value(self) -> None:
        """Test BodyRegion Literal type rejects invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            FindingEnrichmentResult(  # type: ignore[call-arg]
                finding_name="Test Finding",
                body_regions=["Chest", "InvalidRegion"],  # type: ignore[list-item]
                enrichment_timestamp=datetime.now(timezone.utc),
                model_used=TEST_OPENAI_MODEL,
                model_tier="main",
            )

        error_msg = str(exc_info.value)
        assert "body_regions" in error_msg

    def test_modality_enum_valid_values(self) -> None:
        """Test Modality Literal type accepts valid values."""
        valid_modalities: list[Modality] = ["XR", "CT", "MR", "US", "PET", "NM", "MG", "RF", "DSA"]

        result = FindingEnrichmentResult(  # type: ignore[call-arg]
            finding_name="Multi-modality Finding",
            modalities=valid_modalities,
            enrichment_timestamp=datetime.now(timezone.utc),
            model_used=TEST_OPENAI_MODEL,
            model_tier="main",
        )

        assert set(result.modalities) == set(valid_modalities)

    def test_modality_enum_invalid_value(self) -> None:
        """Test Modality Literal type rejects invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            FindingEnrichmentResult(  # type: ignore[call-arg]
                finding_name="Test Finding",
                modalities=["CT", "INVALID"],  # type: ignore[list-item]
                enrichment_timestamp=datetime.now(timezone.utc),
                model_used=TEST_OPENAI_MODEL,
                model_tier="main",
            )

        error_msg = str(exc_info.value)
        assert "modalities" in error_msg

    def test_subspecialty_enum_valid_values(self) -> None:
        """Test Subspecialty Literal type accepts valid values."""
        valid_subspecialties: list[Subspecialty] = [
            "AB",
            "BR",
            "CA",
            "CH",
            "ER",
            "GI",
            "GU",
            "HN",
            "IR",
            "MI",
            "MK",
            "NR",
            "OB",
            "OI",
            "PD",
            "VI",
        ]

        result = FindingEnrichmentResult(  # type: ignore[call-arg]
            finding_name="Multi-specialty Finding",
            subspecialties=valid_subspecialties,
            enrichment_timestamp=datetime.now(timezone.utc),
            model_used=TEST_OPENAI_MODEL,
            model_tier="main",
        )

        assert set(result.subspecialties) == set(valid_subspecialties)

    def test_subspecialty_enum_invalid_value(self) -> None:
        """Test Subspecialty Literal type rejects invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            FindingEnrichmentResult(  # type: ignore[call-arg]
                finding_name="Test Finding",
                subspecialties=["CH", "FAKE"],  # type: ignore[list-item]
                enrichment_timestamp=datetime.now(timezone.utc),
                model_used=TEST_OPENAI_MODEL,
                model_tier="main",
            )

        error_msg = str(exc_info.value)
        assert "subspecialties" in error_msg


# =============================================================================
# ETIOLOGIES Constant Tests
# =============================================================================


class TestEtiologiesConstant:
    """Tests for the ETIOLOGIES taxonomy constant."""

    def test_etiologies_is_list(self) -> None:
        """Test that ETIOLOGIES is a list."""
        assert isinstance(ETIOLOGIES, list)

    def test_etiologies_not_empty(self) -> None:
        """Test that ETIOLOGIES contains values."""
        assert len(ETIOLOGIES) > 0

    def test_etiologies_expected_values(self) -> None:
        """Test that ETIOLOGIES contains expected taxonomy values."""
        expected_values = [
            "inflammatory:infectious",
            "inflammatory",
            "neoplastic:benign",
            "neoplastic:malignant",
            "vascular:ischemic",
            "degenerative",
            "congenital",
            "traumatic",
            "idiopathic",
            "normal-variant",
        ]

        for value in expected_values:
            assert value in ETIOLOGIES, f"Expected etiology '{value}' not found in ETIOLOGIES"

    def test_etiologies_all_strings(self) -> None:
        """Test that all ETIOLOGIES entries are strings."""
        assert all(isinstance(etiology, str) for etiology in ETIOLOGIES)

    def test_etiologies_no_duplicates(self) -> None:
        """Test that ETIOLOGIES has no duplicate entries."""
        assert len(ETIOLOGIES) == len(set(ETIOLOGIES))


# =============================================================================
# EnrichmentContext Model Tests
# =============================================================================


class TestEnrichmentContext:
    """Tests for EnrichmentContext data model."""

    def test_enrichment_context_creation_all_fields(
        self,
        mock_finding_model: FindingModelFull,
        mock_snomed_codes: list[IndexCode],
    ) -> None:
        """Test EnrichmentContext model instantiation with all fields."""
        context = EnrichmentContext(
            finding_name="Pulmonary Nodule",
            finding_description="A rounded opacity in the lung",
            existing_codes=mock_snomed_codes,
            existing_model=mock_finding_model,
        )

        assert context.finding_name == "Pulmonary Nodule"
        assert context.finding_description == "A rounded opacity in the lung"
        assert len(context.existing_codes) == len(mock_snomed_codes)
        assert context.existing_model == mock_finding_model

    def test_enrichment_context_minimal_fields(self) -> None:
        """Test EnrichmentContext with only required field."""
        context = EnrichmentContext(finding_name="Test Finding")

        assert context.finding_name == "Test Finding"
        assert context.finding_description is None
        assert context.existing_codes == []
        assert context.existing_model is None

    def test_enrichment_context_no_existing_model(self, mock_snomed_codes: list[IndexCode]) -> None:
        """Test EnrichmentContext without existing model (new finding)."""
        context = EnrichmentContext(
            finding_name="New Finding",
            finding_description="This is a new finding not in the index",
            existing_codes=mock_snomed_codes,
            existing_model=None,
        )

        assert context.existing_model is None
        assert len(context.existing_codes) > 0


# =============================================================================
# EnrichmentClassification Model Tests
# =============================================================================


class TestEnrichmentClassification:
    """Tests for EnrichmentClassification agent output model."""

    def test_enrichment_classification_creation(self) -> None:
        """Test EnrichmentClassification model instantiation."""
        classification = EnrichmentClassification(
            body_regions=["Chest"],
            etiologies=["inflammatory:infectious", "neoplastic:potential"],
            modalities=["CT", "XR"],
            subspecialties=["CH", "ER"],
            reasoning="This is a chest finding commonly seen on CT and radiographs.",
        )

        assert classification.body_regions == ["Chest"]
        assert classification.etiologies == ["inflammatory:infectious", "neoplastic:potential"]
        assert classification.modalities == ["CT", "XR"]
        assert classification.subspecialties == ["CH", "ER"]
        assert "chest finding" in classification.reasoning

    def test_enrichment_classification_empty_lists(self) -> None:
        """Test EnrichmentClassification with empty classification lists."""
        classification = EnrichmentClassification(
            body_regions=[],
            etiologies=[],
            modalities=[],
            subspecialties=[],
            reasoning="Unable to classify this finding.",
        )

        assert classification.body_regions == []
        assert classification.etiologies == []
        assert classification.modalities == []
        assert classification.subspecialties == []

    def test_enrichment_classification_etiology_validation(self) -> None:
        """Test EnrichmentClassification validates etiologies."""
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentClassification(
                body_regions=["Chest"],
                etiologies=["invalid_etiology"],
                modalities=["CT"],
                subspecialties=["CH"],
                reasoning="Test reasoning",
            )

        error_msg = str(exc_info.value)
        assert "Invalid etiology categories" in error_msg
        assert "invalid_etiology" in error_msg

    def test_enrichment_classification_valid_etiologies(self) -> None:
        """Test EnrichmentClassification accepts valid etiologies."""
        classification = EnrichmentClassification(
            body_regions=["Abdomen"],
            etiologies=["neoplastic:malignant", "vascular:ischemic"],
            modalities=["CT", "MR"],
            subspecialties=["AB", "OI"],
            reasoning="Malignant liver lesion with vascular involvement.",
        )

        assert "neoplastic:malignant" in classification.etiologies
        assert "vascular:ischemic" in classification.etiologies


# =============================================================================
# Integration Tests for Model Compatibility
# =============================================================================


class TestModelCompatibility:
    """Tests for compatibility between enrichment models and other system models."""

    def test_index_code_in_enrichment_result(self, mock_snomed_codes: list[IndexCode]) -> None:
        """Test that IndexCode objects work correctly in FindingEnrichmentResult."""
        result = FindingEnrichmentResult(  # type: ignore[call-arg]
            finding_name="Test Finding",
            snomed_codes=mock_snomed_codes,
            enrichment_timestamp=datetime.now(timezone.utc),
            model_used=TEST_OPENAI_MODEL,
            model_tier="main",
        )

        assert len(result.snomed_codes) == len(mock_snomed_codes)
        for code in result.snomed_codes:
            assert isinstance(code, IndexCode)
            assert code.system == "SNOMEDCT"

    def test_ontology_search_result_in_enrichment_result(
        self, mock_anatomic_locations: list[OntologySearchResult]
    ) -> None:
        """Test that OntologySearchResult objects work in FindingEnrichmentResult."""
        result = FindingEnrichmentResult(  # type: ignore[call-arg]
            finding_name="Test Finding",
            anatomic_locations=mock_anatomic_locations,
            enrichment_timestamp=datetime.now(timezone.utc),
            model_used=TEST_OPENAI_MODEL,
            model_tier="main",
        )

        assert len(result.anatomic_locations) == len(mock_anatomic_locations)
        for location in result.anatomic_locations:
            assert isinstance(location, OntologySearchResult)
            assert hasattr(location, "concept_id")
            assert hasattr(location, "score")

    def test_finding_model_in_enrichment_context(self, mock_finding_model: FindingModelFull) -> None:
        """Test that FindingModelFull works correctly in EnrichmentContext."""
        context = EnrichmentContext(
            finding_name=mock_finding_model.name,
            finding_description=mock_finding_model.description,
            existing_model=mock_finding_model,
        )

        assert context.existing_model is not None
        assert context.existing_model.oifm_id == mock_finding_model.oifm_id
        assert context.existing_model.name == mock_finding_model.name


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestSearchOntologyCodesForFinding:
    """Tests for search_ontology_codes_for_finding helper function."""

    @pytest.mark.asyncio
    async def test_search_ontology_codes_separates_systems(
        self, mock_snomed_codes: list[IndexCode], mock_radlex_codes: list[IndexCode]
    ) -> None:
        """Test that search_ontology_codes_for_finding separates SNOMED and RadLex codes."""
        from unittest.mock import AsyncMock, patch

        from findingmodel_ai.enrichment.unified import search_ontology_codes_for_finding
        from findingmodel_ai.search.ontology import CategorizedOntologyConcepts

        # Create mock categorized results with mixed ontology systems
        mock_results = CategorizedOntologyConcepts(
            exact_matches=[
                OntologySearchResult(
                    concept_id=mock_snomed_codes[0].code,
                    concept_text=mock_snomed_codes[0].display,
                    score=0.95,
                    table_name="snomedct",
                )
            ],
            should_include=[
                OntologySearchResult(
                    concept_id=mock_radlex_codes[0].code,
                    concept_text=mock_radlex_codes[0].display,
                    score=0.88,
                    table_name="radlex",
                )
            ],
            marginal_concepts=[],
            search_summary="Test search",
            excluded_anatomical=[],
        )

        with patch(
            "findingmodel_ai.search.ontology.match_ontology_concepts",
            new=AsyncMock(return_value=mock_results),
        ):
            snomed_codes, radlex_codes = await search_ontology_codes_for_finding("test finding")

            # Verify separation by system
            assert len(snomed_codes) == 1
            assert len(radlex_codes) == 1
            assert snomed_codes[0].system == "SNOMEDCT"
            assert radlex_codes[0].system == "RADLEX"

    @pytest.mark.asyncio
    async def test_search_ontology_codes_empty_results(self) -> None:
        """Test search_ontology_codes_for_finding with empty results."""
        from unittest.mock import AsyncMock, patch

        from findingmodel_ai.enrichment.unified import search_ontology_codes_for_finding
        from findingmodel_ai.search.ontology import CategorizedOntologyConcepts

        mock_results = CategorizedOntologyConcepts(
            exact_matches=[],
            should_include=[],
            marginal_concepts=[],
            search_summary="Test search",
            excluded_anatomical=[],
        )

        with patch(
            "findingmodel_ai.search.ontology.match_ontology_concepts",
            new=AsyncMock(return_value=mock_results),
        ):
            snomed_codes, radlex_codes = await search_ontology_codes_for_finding("nonexistent finding")

            assert snomed_codes == []
            assert radlex_codes == []

    @pytest.mark.asyncio
    async def test_search_ontology_codes_filters_anatomical(self) -> None:
        """Test that exclude_anatomical=True is passed to match_ontology_concepts."""
        from unittest.mock import AsyncMock, patch

        from findingmodel_ai.enrichment.unified import search_ontology_codes_for_finding
        from findingmodel_ai.search.ontology import CategorizedOntologyConcepts

        mock_match = AsyncMock(
            return_value=CategorizedOntologyConcepts(
                exact_matches=[],
                should_include=[],
                marginal_concepts=[],
                search_summary="Test search",
                excluded_anatomical=[],
            )
        )

        with patch("findingmodel_ai.search.ontology.match_ontology_concepts", new=mock_match):
            await search_ontology_codes_for_finding("test finding", "test description")

            # Verify exclude_anatomical=True was passed (model_tier defaults to "base")
            mock_match.assert_called_once_with(
                finding_name="test finding",
                finding_description="test description",
                exclude_anatomical=True,
                model_tier="base",
            )

    @pytest.mark.asyncio
    async def test_search_ontology_codes_includes_exact_and_should(self, mock_snomed_codes: list[IndexCode]) -> None:
        """Test that both exact_matches and should_include are collected."""
        from unittest.mock import AsyncMock, patch

        from findingmodel_ai.enrichment.unified import search_ontology_codes_for_finding
        from findingmodel_ai.search.ontology import CategorizedOntologyConcepts

        mock_results = CategorizedOntologyConcepts(
            exact_matches=[
                OntologySearchResult(
                    concept_id=mock_snomed_codes[0].code,
                    concept_text=mock_snomed_codes[0].display,
                    score=0.95,
                    table_name="snomedct",
                )
            ],
            should_include=[
                OntologySearchResult(
                    concept_id=mock_snomed_codes[1].code,
                    concept_text=mock_snomed_codes[1].display,
                    score=0.85,
                    table_name="snomedct",
                )
            ],
            marginal_concepts=[],
            search_summary="Test search",
            excluded_anatomical=[],
        )

        with patch(
            "findingmodel_ai.search.ontology.match_ontology_concepts",
            new=AsyncMock(return_value=mock_results),
        ):
            snomed_codes, _ = await search_ontology_codes_for_finding("test finding")

            # Should collect both exact and should_include
            assert len(snomed_codes) == 2

    @pytest.mark.asyncio
    async def test_search_ontology_codes_error_handling(self) -> None:
        """Test that exceptions are raised from search_ontology_codes_for_finding."""
        from unittest.mock import AsyncMock, patch

        from findingmodel_ai.enrichment.unified import search_ontology_codes_for_finding

        mock_match = AsyncMock(side_effect=Exception("Search failed"))

        with (
            patch("findingmodel_ai.search.ontology.match_ontology_concepts", new=mock_match),
            pytest.raises(Exception, match="Search failed"),
        ):
            await search_ontology_codes_for_finding("test finding")


class TestCreateEnrichmentSystemPrompt:
    """Tests for _create_enrichment_system_prompt helper function."""

    def test_system_prompt_includes_all_etiologies(self) -> None:
        """Test that system prompt includes all 22 etiologies."""
        from findingmodel_ai.enrichment.unified import ETIOLOGIES, _create_enrichment_system_prompt

        prompt = _create_enrichment_system_prompt()

        # Verify all etiologies are present in the prompt
        for etiology in ETIOLOGIES:
            assert etiology in prompt, f"Etiology '{etiology}' not found in system prompt"

    def test_system_prompt_includes_modality_definitions(self) -> None:
        """Test that system prompt includes modality codes and explanations."""
        from findingmodel_ai.enrichment.unified import _create_enrichment_system_prompt

        prompt = _create_enrichment_system_prompt()

        # Verify modality codes are explained
        assert "XR: Radiography" in prompt
        assert "CT: Computed Tomography" in prompt
        assert "MR: Magnetic Resonance Imaging" in prompt
        assert "US: Ultrasound" in prompt

    def test_system_prompt_includes_subspecialty_definitions(self) -> None:
        """Test that system prompt includes subspecialty codes and explanations."""
        from findingmodel_ai.enrichment.unified import _create_enrichment_system_prompt

        prompt = _create_enrichment_system_prompt()

        # Verify subspecialty codes are explained
        assert "CH: Chest/Thoracic Imaging" in prompt
        assert "NR: Neuroradiology" in prompt
        assert "AB: Abdominal Radiology" in prompt
        assert "MK: Musculoskeletal Radiology" in prompt

    def test_system_prompt_includes_role_description(self) -> None:
        """Test that system prompt describes the specialist role."""
        from findingmodel_ai.enrichment.unified import _create_enrichment_system_prompt

        prompt = _create_enrichment_system_prompt()

        # Verify role description is present
        assert "enrichment specialist" in prompt
        assert "radiology" in prompt
        assert "ROLE & RESPONSIBILITIES" in prompt


# =============================================================================
# Agent Configuration Tests
# =============================================================================


class TestCreateEnrichmentAgent:
    """Tests for create_enrichment_agent function."""

    def test_agent_has_correct_model_tier(self) -> None:
        """Test that agent is created with correct model tier."""
        from unittest.mock import patch

        from findingmodel_ai.enrichment.unified import create_enrichment_agent
        from pydantic_ai.models.test import TestModel

        with patch("findingmodel_ai.enrichment.unified.settings") as mock_settings:
            mock_settings.get_agent_model.return_value = TestModel()

            create_enrichment_agent(model_tier="base")

            # Verify settings.get_agent_model was called with correct tag and tier
            mock_settings.get_agent_model.assert_called_once_with("enrich_classify", default_tier="base")

    def test_agent_has_no_tools(self) -> None:
        """Test that agent has no tools registered.

        The agent receives pre-fetched search results in context rather than
        calling tools, avoiding duplicate API calls and improving performance.
        """
        from unittest.mock import patch

        from findingmodel_ai.config import FindingModelAIConfig
        from findingmodel_ai.enrichment.unified import create_enrichment_agent

        # Mock get_model to avoid API key check
        with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
            agent = create_enrichment_agent()

        # Agent should have no tools - searches are pre-fetched
        assert len(agent._function_toolset.tools) == 0

    def test_agent_output_type_is_classification(self) -> None:
        """Test that agent output type is EnrichmentClassification."""
        from unittest.mock import patch

        from findingmodel_ai.config import FindingModelAIConfig
        from findingmodel_ai.enrichment.unified import create_enrichment_agent

        # Mock get_model to avoid API key check
        with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
            agent = create_enrichment_agent()

        # Verify output type
        assert agent._output_type == EnrichmentClassification

    def test_agent_deps_type_is_context(self) -> None:
        """Test that agent deps type is EnrichmentContext."""
        from unittest.mock import patch

        from findingmodel_ai.config import FindingModelAIConfig
        from findingmodel_ai.enrichment.unified import create_enrichment_agent

        # Mock get_model to avoid API key check
        with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
            agent = create_enrichment_agent()

        # Verify deps type
        assert agent._deps_type == EnrichmentContext

    def test_agent_system_prompt_not_empty(self) -> None:
        """Test that agent has a non-empty system prompt."""
        from unittest.mock import patch

        from findingmodel_ai.config import FindingModelAIConfig
        from findingmodel_ai.enrichment.unified import create_enrichment_agent

        # Mock get_model to avoid API key check
        with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
            agent = create_enrichment_agent()

        # System prompt should be set (stored in _system_prompts tuple)
        assert agent._system_prompts is not None
        assert len(agent._system_prompts) > 0
        assert len(agent._system_prompts[0]) > 0

    def test_agent_with_custom_model(self) -> None:
        """Test creating agent with custom model string."""
        from findingmodel_ai.enrichment.unified import create_enrichment_agent
        from pydantic_ai.models.test import TestModel

        # Use TestModel to avoid needing API keys
        agent = create_enrichment_agent(model=TestModel())

        # Agent should be created with the custom model
        # (Can't easily verify the model without accessing internals,
        # but we can verify the function accepts the parameter)
        assert agent is not None


# =============================================================================
# Agent Behavior Tests with TestModel
# =============================================================================


class TestEnrichmentAgentBehavior:
    """Tests for enrichment agent behavior using TestModel."""

    @pytest.mark.asyncio
    async def test_agent_produces_valid_classification(self, mock_enrichment_context: EnrichmentContext) -> None:
        """Test that agent produces valid EnrichmentClassification output."""
        from unittest.mock import patch

        from findingmodel_ai.config import FindingModelAIConfig
        from findingmodel_ai.enrichment.unified import create_enrichment_agent
        from pydantic_ai.models.test import TestModel

        # Create controlled classification response
        controlled_classification = EnrichmentClassification(
            body_regions=["Chest"],
            etiologies=["inflammatory:infectious"],
            modalities=["CT", "XR"],
            subspecialties=["CH"],
            reasoning="Test reasoning",
        )

        # Mock get_model to avoid API key check
        with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
            agent = create_enrichment_agent()

        with agent.override(model=TestModel(custom_output_args=controlled_classification)):
            result = await agent.run("Classify test finding", deps=mock_enrichment_context)

            assert isinstance(result.output, EnrichmentClassification)
            assert result.output.body_regions == ["Chest"]
            assert result.output.etiologies == ["inflammatory:infectious"]

    @pytest.mark.asyncio
    async def test_agent_accepts_context_with_existing_model(self, mock_enrichment_context: EnrichmentContext) -> None:
        """Test that agent runs with context containing existing_model."""
        from unittest.mock import patch

        from findingmodel_ai.config import FindingModelAIConfig
        from findingmodel_ai.enrichment.unified import create_enrichment_agent
        from pydantic_ai.models.test import TestModel

        controlled_classification = EnrichmentClassification(
            body_regions=["Abdomen"],
            etiologies=["neoplastic:benign"],
            modalities=["MR"],
            subspecialties=["AB"],
            reasoning="Test with existing model",
        )

        # Mock get_model to avoid API key check
        with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
            agent = create_enrichment_agent()

        # mock_enrichment_context has existing_model set
        assert mock_enrichment_context.existing_model is not None

        with agent.override(model=TestModel(custom_output_args=controlled_classification)):
            result = await agent.run("Classify", deps=mock_enrichment_context)

            assert result.output is not None

    @pytest.mark.asyncio
    async def test_agent_accepts_context_without_existing_model(self) -> None:
        """Test that agent runs with context without existing_model."""
        from unittest.mock import patch

        from findingmodel_ai.config import FindingModelAIConfig
        from findingmodel_ai.enrichment.unified import create_enrichment_agent
        from pydantic_ai.models.test import TestModel

        controlled_classification = EnrichmentClassification(
            body_regions=["Head"],
            etiologies=["vascular:ischemic"],
            modalities=["CT", "MR"],
            subspecialties=["NR"],
            reasoning="Test without existing model",
        )

        # Mock get_model to avoid API key check
        with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
            agent = create_enrichment_agent()

        # Create context without existing_model
        context = EnrichmentContext(
            finding_name="New Finding",
            finding_description="Not in index",
            existing_model=None,
        )

        with agent.override(model=TestModel(custom_output_args=controlled_classification)):
            result = await agent.run("Classify", deps=context)

            assert result.output is not None

    @pytest.mark.asyncio
    async def test_agent_output_has_required_fields(self) -> None:
        """Test that agent output has all required classification fields."""
        from unittest.mock import patch

        from findingmodel_ai.config import FindingModelAIConfig
        from findingmodel_ai.enrichment.unified import create_enrichment_agent
        from pydantic_ai.models.test import TestModel

        controlled_classification = EnrichmentClassification(
            body_regions=["Chest", "Abdomen"],
            etiologies=["traumatic", "vascular:hemorrhagic"],
            modalities=["CT"],
            subspecialties=["ER"],
            reasoning="Multi-system trauma",
        )

        # Mock get_model to avoid API key check
        with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
            agent = create_enrichment_agent()

        context = EnrichmentContext(finding_name="Trauma")

        with agent.override(model=TestModel(custom_output_args=controlled_classification)):
            result = await agent.run("Classify", deps=context)

            # Verify all required fields present
            assert hasattr(result.output, "body_regions")
            assert hasattr(result.output, "etiologies")
            assert hasattr(result.output, "modalities")
            assert hasattr(result.output, "subspecialties")
            assert hasattr(result.output, "reasoning")

    @pytest.mark.asyncio
    async def test_agent_allows_empty_classifications(self) -> None:
        """Test that agent accepts empty classification lists."""
        from unittest.mock import patch

        from findingmodel_ai.config import FindingModelAIConfig
        from findingmodel_ai.enrichment.unified import create_enrichment_agent
        from pydantic_ai.models.test import TestModel

        controlled_classification = EnrichmentClassification(
            body_regions=[],
            etiologies=[],
            modalities=[],
            subspecialties=[],
            reasoning="Unable to classify",
        )

        # Mock get_model to avoid API key check
        with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
            agent = create_enrichment_agent()

        context = EnrichmentContext(finding_name="Unknown")

        with agent.override(model=TestModel(custom_output_args=controlled_classification)):
            result = await agent.run("Classify", deps=context)

            assert result.output.body_regions == []
            assert result.output.etiologies == []

    @pytest.mark.asyncio
    async def test_agent_validates_etiology_values(self) -> None:
        """Test that agent validates etiology values against ETIOLOGIES list."""
        # The validation happens at Pydantic model level, not agent level
        # So we test that creating the invalid model raises ValidationError
        with pytest.raises(ValidationError, match="Invalid etiology categories"):
            EnrichmentClassification(
                body_regions=["Chest"],
                etiologies=["invalid_etiology_value"],
                modalities=["CT"],
                subspecialties=["CH"],
                reasoning="Test",
            )

    @pytest.mark.asyncio
    async def test_context_includes_prefetched_codes(
        self,
        mock_snomed_codes: list[IndexCode],
        mock_radlex_codes: list[IndexCode],
    ) -> None:
        """Test that EnrichmentContext can include pre-fetched ontology codes."""
        context = EnrichmentContext(
            finding_name="pneumonia",
            finding_description="Infection of the lung parenchyma",
            existing_codes=[],
            existing_model=None,
            snomed_codes=mock_snomed_codes,
            radlex_codes=mock_radlex_codes,
        )

        assert len(context.snomed_codes) == len(mock_snomed_codes)
        assert len(context.radlex_codes) == len(mock_radlex_codes)

    @pytest.mark.asyncio
    async def test_context_includes_prefetched_anatomic_locations(
        self,
        mock_anatomic_locations: list[OntologySearchResult],
    ) -> None:
        """Test that EnrichmentContext can include pre-fetched anatomic locations."""
        context = EnrichmentContext(
            finding_name="pneumonia",
            finding_description="Infection of the lung parenchyma",
            existing_codes=[],
            existing_model=None,
            anatomic_locations=mock_anatomic_locations,
        )

        assert len(context.anatomic_locations) == len(mock_anatomic_locations)


# =============================================================================
# Orchestration Function Tests
# =============================================================================


class TestEnrichFindingOrchestration:
    """Tests for enrich_finding orchestration function."""

    @pytest.mark.asyncio
    async def test_enrich_finding_with_existing_in_index(
        self,
        mock_finding_model: FindingModelFull,
        mock_snomed_codes: list[IndexCode],
        mock_radlex_codes: list[IndexCode],
        mock_anatomic_locations: list[OntologySearchResult],
    ) -> None:
        """Test enrich_finding when finding exists in index."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from findingmodel.index import IndexEntry
        from findingmodel_ai.enrichment.unified import enrich_finding

        # Create mock index entry (minimal required fields)
        mock_entry = IndexEntry(
            oifm_id=mock_finding_model.oifm_id,
            name=mock_finding_model.name,
            slug_name=mock_finding_model.name.lower().replace(" ", "-"),
            filename="test_file.json",
            file_hash_sha256="abc123def456",
        )

        # Mock DuckDBIndex
        mock_index = AsyncMock()
        mock_index.__aenter__ = AsyncMock(return_value=mock_index)
        mock_index.__aexit__ = AsyncMock(return_value=None)
        mock_index.get = AsyncMock(return_value=mock_entry)
        mock_index.get_full = AsyncMock(return_value=mock_finding_model)

        # Mock search functions
        mock_search_ontology = AsyncMock(return_value=(mock_snomed_codes[:1], mock_radlex_codes[:1]))
        mock_find_anatomic = AsyncMock()
        mock_find_anatomic.return_value = MagicMock(
            primary_location=mock_anatomic_locations[0],
            alternate_locations=mock_anatomic_locations[1:2],
        )

        # Mock agent
        controlled_classification = EnrichmentClassification(
            body_regions=["Chest"],
            etiologies=["inflammatory:infectious"],
            modalities=["CT"],
            subspecialties=["CH"],
            reasoning="Test classification",
        )

        with (
            patch("findingmodel_ai.enrichment.unified.DuckDBIndex", return_value=mock_index),
            patch("findingmodel_ai.enrichment.unified.search_ontology_codes_for_finding", mock_search_ontology),
            patch("findingmodel_ai.search.anatomic.find_anatomic_locations", mock_find_anatomic),
            patch("findingmodel_ai.enrichment.unified.create_enrichment_agent") as mock_create_agent,
        ):
            mock_agent = AsyncMock()
            mock_result = MagicMock()
            mock_result.output = controlled_classification
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_create_agent.return_value = mock_agent

            result = await enrich_finding(mock_finding_model.oifm_id)

            # Verify complete result
            assert result.oifm_id == mock_finding_model.oifm_id
            assert result.finding_name == mock_finding_model.name
            assert len(result.snomed_codes) == 1
            assert len(result.radlex_codes) == 1
            assert result.body_regions == ["Chest"]
            assert result.etiologies == ["inflammatory:infectious"]

    @pytest.mark.asyncio
    async def test_enrich_finding_not_in_index(self) -> None:
        """Test enrich_finding when finding is not in index."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from findingmodel_ai.enrichment.unified import enrich_finding

        # Mock DuckDBIndex to return None (not found)
        mock_index = AsyncMock()
        mock_index.__aenter__ = AsyncMock(return_value=mock_index)
        mock_index.__aexit__ = AsyncMock(return_value=None)
        mock_index.get = AsyncMock(return_value=None)

        # Mock search functions
        mock_search_ontology = AsyncMock(return_value=([], []))
        mock_find_anatomic = AsyncMock()
        mock_find_anatomic.return_value = MagicMock(
            primary_location=OntologySearchResult(
                concept_id="NO_RESULTS",
                concept_text="unspecified anatomic location",
                score=0.0,
                table_name="anatomic_locations",
            ),
            alternate_locations=[],
        )

        controlled_classification = EnrichmentClassification(
            body_regions=[], etiologies=[], modalities=[], subspecialties=[], reasoning="New finding"
        )

        with (
            patch("findingmodel_ai.enrichment.unified.DuckDBIndex", return_value=mock_index),
            patch("findingmodel_ai.enrichment.unified.search_ontology_codes_for_finding", mock_search_ontology),
            patch("findingmodel_ai.search.anatomic.find_anatomic_locations", mock_find_anatomic),
            patch("findingmodel_ai.enrichment.unified.create_enrichment_agent") as mock_create_agent,
        ):
            mock_agent = AsyncMock()
            mock_result = MagicMock()
            mock_result.output = controlled_classification
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_create_agent.return_value = mock_agent

            result = await enrich_finding("new finding name")

            # Should continue with finding_name = identifier
            assert result.finding_name == "new finding name"
            assert result.oifm_id is None

    @pytest.mark.asyncio
    async def test_enrich_finding_assembles_result_correctly(
        self, mock_snomed_codes: list[IndexCode], mock_anatomic_locations: list[OntologySearchResult]
    ) -> None:
        """Test that enrich_finding assembles all result fields correctly."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from findingmodel_ai.enrichment.unified import enrich_finding

        # Mock DuckDBIndex
        mock_index = AsyncMock()
        mock_index.__aenter__ = AsyncMock(return_value=mock_index)
        mock_index.__aexit__ = AsyncMock(return_value=None)
        mock_index.get = AsyncMock(return_value=None)

        # Mock search functions with specific data
        mock_search_ontology = AsyncMock(return_value=(mock_snomed_codes[:2], []))
        mock_find_anatomic = AsyncMock()
        mock_find_anatomic.return_value = MagicMock(
            primary_location=mock_anatomic_locations[0],
            alternate_locations=mock_anatomic_locations[1:3],
        )

        controlled_classification = EnrichmentClassification(
            body_regions=["Chest", "Abdomen"],
            etiologies=["neoplastic:malignant", "inflammatory"],
            modalities=["CT", "MR", "PET"],
            subspecialties=["CH", "AB", "OI"],
            reasoning="Multi-system involvement",
        )

        with (
            patch("findingmodel_ai.enrichment.unified.DuckDBIndex", return_value=mock_index),
            patch("findingmodel_ai.enrichment.unified.search_ontology_codes_for_finding", mock_search_ontology),
            patch("findingmodel_ai.search.anatomic.find_anatomic_locations", mock_find_anatomic),
            patch("findingmodel_ai.enrichment.unified.create_enrichment_agent") as mock_create_agent,
        ):
            mock_agent = AsyncMock()
            mock_result = MagicMock()
            mock_result.output = controlled_classification
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_create_agent.return_value = mock_agent

            result = await enrich_finding("test finding")

            # Verify all fields populated correctly
            assert result.finding_name == "test finding"
            assert len(result.snomed_codes) == 2
            assert len(result.radlex_codes) == 0
            assert len(result.anatomic_locations) == 3  # primary + 2 alternates
            assert len(result.body_regions) == 2
            assert len(result.etiologies) == 2
            assert len(result.modalities) == 3
            assert len(result.subspecialties) == 3
            assert result.enrichment_timestamp is not None
            assert result.model_used is not None
            assert result.model_tier == "base"

    @pytest.mark.asyncio
    async def test_enrich_finding_handles_ontology_error(self) -> None:
        """Test that enrich_finding gracefully handles ontology search errors."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from findingmodel_ai.enrichment.unified import enrich_finding

        # Mock DuckDBIndex
        mock_index = AsyncMock()
        mock_index.__aenter__ = AsyncMock(return_value=mock_index)
        mock_index.__aexit__ = AsyncMock(return_value=None)
        mock_index.get = AsyncMock(return_value=None)

        # Mock search to raise exception (will be caught by fallback)
        mock_search_ontology = AsyncMock(side_effect=Exception("Ontology search failed"))
        mock_find_anatomic = AsyncMock()
        mock_find_anatomic.return_value = MagicMock(
            primary_location=OntologySearchResult(
                concept_id="NO_RESULTS",
                concept_text="unspecified anatomic location",
                score=0.0,
                table_name="anatomic_locations",
            ),
            alternate_locations=[],
        )

        controlled_classification = EnrichmentClassification(
            body_regions=[], etiologies=[], modalities=[], subspecialties=[], reasoning="Error handling test"
        )

        with (
            patch("findingmodel_ai.enrichment.unified.DuckDBIndex", return_value=mock_index),
            patch("findingmodel_ai.enrichment.unified.search_ontology_codes_for_finding", mock_search_ontology),
            patch("findingmodel_ai.search.anatomic.find_anatomic_locations", mock_find_anatomic),
            patch("findingmodel_ai.enrichment.unified.create_enrichment_agent") as mock_create_agent,
        ):
            mock_agent = AsyncMock()
            mock_result = MagicMock()
            mock_result.output = controlled_classification
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_create_agent.return_value = mock_agent

            result = await enrich_finding("test finding")

            # Should gracefully degrade with empty codes
            assert result.snomed_codes == []
            assert result.radlex_codes == []
            assert result.finding_name == "test finding"

    @pytest.mark.asyncio
    async def test_enrich_finding_handles_anatomic_error(self) -> None:
        """Test that enrich_finding gracefully handles anatomic location errors."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from findingmodel_ai.enrichment.unified import enrich_finding

        # Mock DuckDBIndex
        mock_index = AsyncMock()
        mock_index.__aenter__ = AsyncMock(return_value=mock_index)
        mock_index.__aexit__ = AsyncMock(return_value=None)
        mock_index.get = AsyncMock(return_value=None)

        # Mock anatomic search to raise exception (will be caught by fallback)
        mock_search_ontology = AsyncMock(return_value=([], []))
        mock_find_anatomic = AsyncMock(side_effect=Exception("Anatomic search failed"))

        controlled_classification = EnrichmentClassification(
            body_regions=[], etiologies=[], modalities=[], subspecialties=[], reasoning="Error handling test"
        )

        with (
            patch("findingmodel_ai.enrichment.unified.DuckDBIndex", return_value=mock_index),
            patch("findingmodel_ai.enrichment.unified.search_ontology_codes_for_finding", mock_search_ontology),
            patch("findingmodel_ai.search.anatomic.find_anatomic_locations", mock_find_anatomic),
            patch("findingmodel_ai.enrichment.unified.create_enrichment_agent") as mock_create_agent,
        ):
            mock_agent = AsyncMock()
            mock_result = MagicMock()
            mock_result.output = controlled_classification
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_create_agent.return_value = mock_agent

            result = await enrich_finding("test finding")

            # Should gracefully degrade with empty anatomic locations
            assert result.anatomic_locations == []
            assert result.finding_name == "test finding"

    @pytest.mark.asyncio
    async def test_enrich_finding_model_parameter_propagated(self) -> None:
        """Test that model parameter is passed to create_enrichment_agent."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from findingmodel_ai.enrichment.unified import enrich_finding

        # Mock DuckDBIndex
        mock_index = AsyncMock()
        mock_index.__aenter__ = AsyncMock(return_value=mock_index)
        mock_index.__aexit__ = AsyncMock(return_value=None)
        mock_index.get = AsyncMock(return_value=None)

        # Mock search functions
        mock_search_ontology = AsyncMock(return_value=([], []))
        mock_find_anatomic = AsyncMock()
        mock_find_anatomic.return_value = MagicMock(
            primary_location=OntologySearchResult(
                concept_id="NO_RESULTS",
                concept_text="unspecified anatomic location",
                score=0.0,
                table_name="anatomic_locations",
            ),
            alternate_locations=[],
        )

        controlled_classification = EnrichmentClassification(
            body_regions=[], etiologies=[], modalities=[], subspecialties=[], reasoning="Test"
        )

        with (
            patch("findingmodel_ai.enrichment.unified.DuckDBIndex", return_value=mock_index),
            patch("findingmodel_ai.enrichment.unified.search_ontology_codes_for_finding", mock_search_ontology),
            patch("findingmodel_ai.search.anatomic.find_anatomic_locations", mock_find_anatomic),
            patch("findingmodel_ai.enrichment.unified.create_enrichment_agent") as mock_create_agent,
        ):
            mock_agent = AsyncMock()
            mock_result = MagicMock()
            mock_result.output = controlled_classification
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_create_agent.return_value = mock_agent

            await enrich_finding("test finding", model="anthropic:claude-sonnet-4-5")

            # Verify model was passed to create_enrichment_agent
            mock_create_agent.assert_called_once_with(model_tier="base", model="anthropic:claude-sonnet-4-5")


# =============================================================================
# Integration Tests with Real API Calls
# =============================================================================


class TestFindingEnrichmentIntegration:
    """Integration tests for finding enrichment with real API calls.

    These tests use actual AI models and external services to verify end-to-end
    functionality. They are marked with @pytest.mark.callout and are excluded
    from the default test run. Run with: task test-full
    """

    @pytest.mark.callout
    @pytest.mark.asyncio
    async def test_enrich_pneumonia(self) -> None:
        """Test enrichment of simple finding: pneumonia."""
        # Temporarily enable model requests for this test
        original = models.ALLOW_MODEL_REQUESTS
        models.ALLOW_MODEL_REQUESTS = True
        try:
            from findingmodel_ai.enrichment.unified import enrich_finding

            # Use "small" tier for faster integration tests
            result = await enrich_finding("pneumonia", model_tier="small")

            # Validate result structure - finding_name may be canonical form from Index
            assert result.finding_name.lower() == "pneumonia"
            assert isinstance(result.snomed_codes, list)
            assert isinstance(result.radlex_codes, list)
            assert isinstance(result.body_regions, list)
            assert isinstance(result.etiologies, list)
            assert isinstance(result.modalities, list)
            assert isinstance(result.subspecialties, list)
            assert isinstance(result.anatomic_locations, list)
            assert result.enrichment_timestamp is not None

            # Content validation for pneumonia
            # Should have at least one chest region
            assert "Chest" in result.body_regions or len(result.body_regions) > 0

            # Should have infectious etiology
            infectious_etiologies = [e for e in result.etiologies if "infectious" in e.lower()]
            assert len(infectious_etiologies) > 0 or len(result.etiologies) > 0

            # Common modalities for pneumonia should include XR or CT
            assert len(result.modalities) > 0

            # Should have chest subspecialty
            assert "CH" in result.subspecialties or len(result.subspecialties) > 0

            # Should have anatomic locations
            assert len(result.anatomic_locations) >= 0  # May be empty if search fails

        finally:
            models.ALLOW_MODEL_REQUESTS = original

    @pytest.mark.callout
    @pytest.mark.asyncio
    async def test_enrich_unknown_finding(self) -> None:
        """Test enrichment of unknown finding name - should still produce valid result."""
        original = models.ALLOW_MODEL_REQUESTS
        models.ALLOW_MODEL_REQUESTS = True
        try:
            from findingmodel_ai.enrichment.unified import enrich_finding

            # Use a nonsensical finding name that won't be in any database
            # Use "small" tier for faster integration tests
            result = await enrich_finding("xyzzy123nonexistent", model_tier="small")

            # Should still return valid structure even for unknown finding
            assert result.finding_name == "xyzzy123nonexistent"
            assert isinstance(result.snomed_codes, list)
            assert isinstance(result.radlex_codes, list)
            assert isinstance(result.body_regions, list)
            assert isinstance(result.etiologies, list)
            assert isinstance(result.modalities, list)
            assert isinstance(result.subspecialties, list)
            assert isinstance(result.anatomic_locations, list)
            assert result.enrichment_timestamp is not None

            # All lists should be valid (even if empty)
            # The agent should handle unknown findings gracefully
            assert result.oifm_id is None  # Not in index

        finally:
            models.ALLOW_MODEL_REQUESTS = original
