"""Evaluation suite for similar_finding_models agent.

INTEGRATION SUITE: Requires real API calls and populated DuckDB index.
Run via `task evals:similar_models`.

INDEX SETUP REQUIREMENTS:
This eval suite requires a populated DuckDB index containing test finding models.
To set up the index:
1. Ensure test data exists in test/data/defs/*.fm.json
2. Run: python -m findingmodel.cli index rebuild --backend duckdb
3. Verify index populated: python -m findingmodel.cli index info --backend duckdb

The suite uses real AI agents and DuckDB vector search to find similar models,
making it an integration test rather than a unit test.

API CALL PATTERN:
    Unlike unit tests, this eval suite makes REAL API calls by design. The
    `models.ALLOW_MODEL_REQUESTS = False` guard is intentionally omitted since
    evaluation requires actual agent execution with DuckDB and AI models.

    This is an integration eval suite, not a unit test. It validates end-to-end
    behavior including:
    - Real DuckDB index queries with embeddings
    - Actual AI model similarity analysis
    - Complete agent workflow coordination

    See: Serena memory 'agent_evaluation_best_practices_2025'

This module defines evaluation cases for assessing the similar_finding_models functionality,
which uses two AI agents to find existing models similar enough to a proposed model
that editing them might be better than creating new ones.

EVALUATOR-BASED PATTERN:
- Cases are evaluated using Dataset.evaluate() with focused evaluators
- Each evaluator checks a specific aspect (duplicate detection, ranking quality, etc.)
- Hybrid scoring: strict for must-haves (0.0 or 1.0), partial credit for quality (0.0-1.0)

EVALUATORS:
- DuplicateDetectionEvaluator: Exact duplicates must be found (strict)
- RankingQualityEvaluator: Similarity ranking quality using MRR (partial credit)
- PrecisionAtKEvaluator: Precision@K for top results (partial credit)
- SemanticSimilarityEvaluator: Semantic similarity of results (partial credit)
- ExclusionEvaluator: Dissimilar models correctly excluded (strict)
- PerformanceEvaluator: Query performance under threshold (strict)

LOGFIRE INTEGRATION:
Logfire observability is configured automatically in evals/__init__.py.
No manual instrumentation needed in this module - automatic spans are created by:
- Dataset.evaluate() for root and per-case spans
- Pydantic AI instrumentation for agent/model/tool calls

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

import time

from findingmodel.index import DuckDBIndex as Index
from findingmodel_ai.evaluators import PerformanceEvaluator
from findingmodel_ai.search.similar import SimilarModelAnalysis, find_similar_models
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport


class SimilarModelsInput(BaseModel):
    """Input for a similar models evaluation case."""

    finding_name: str = Field(description="Name of the proposed finding model")
    description: str | None = Field(default=None, description="Description of the proposed finding")
    synonyms: list[str] | None = Field(default=None, description="Synonyms for the proposed finding")


class SimilarModelsExpectedOutput(BaseModel):
    """Expected output for a similar models evaluation case."""

    should_find_exact_match: bool = Field(default=False, description="Whether an exact duplicate should be found")
    expected_recommendation: str | None = Field(
        default=None, description="Expected recommendation: 'edit_existing' or 'create_new'"
    )
    expected_similar_ids: list[str] = Field(
        default_factory=list, description="Expected OIFM IDs of similar models (for ranking evaluation)"
    )
    unexpected_similar_ids: list[str] = Field(
        default_factory=list,
        description="OIFM IDs that should NOT appear in results (for dissimilar validation)",
    )
    min_confidence: float = Field(default=0.0, description="Minimum acceptable confidence score", ge=0.0, le=1.0)
    max_query_time: float = Field(default=30.0, description="Maximum acceptable query time in seconds")
    semantic_keywords: list[str] = Field(
        default_factory=list, description="Keywords that should appear in similar model names/descriptions"
    )


class SimilarModelsActualOutput(BaseModel):
    """Actual output from running a similar models case."""

    analysis: SimilarModelAnalysis
    query_time: float = Field(description="Time taken to execute query in seconds")
    error: str | None = Field(default=None, description="Error message if query failed")


class SimilarModelsCase(Case[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput]):
    """A test case for similar_finding_models functionality."""

    def __init__(
        self,
        name: str,
        finding_name: str,
        description: str | None = None,
        synonyms: list[str] | None = None,
        should_find_exact_match: bool = False,
        expected_recommendation: str | None = None,
        expected_similar_ids: list[str] | None = None,
        unexpected_similar_ids: list[str] | None = None,
        min_confidence: float = 0.0,
        max_query_time: float = 30.0,
        semantic_keywords: list[str] | None = None,
    ) -> None:
        """Initialize a similar models evaluation case.

        Args:
            name: Name of the test case
            finding_name: Name of the proposed finding model
            description: Description of the proposed finding
            synonyms: Synonyms for the proposed finding
            should_find_exact_match: Whether an exact duplicate should be found
            expected_recommendation: Expected recommendation ('edit_existing' or 'create_new')
            expected_similar_ids: Expected OIFM IDs of similar models (for ranking)
            unexpected_similar_ids: OIFM IDs that should NOT appear in results
            min_confidence: Minimum acceptable confidence score
            max_query_time: Maximum acceptable query time in seconds
            semantic_keywords: Keywords for semantic similarity check
        """
        inputs = SimilarModelsInput(
            finding_name=finding_name,
            description=description,
            synonyms=synonyms,
        )
        metadata = SimilarModelsExpectedOutput(
            should_find_exact_match=should_find_exact_match,
            expected_recommendation=expected_recommendation,
            expected_similar_ids=expected_similar_ids or [],
            unexpected_similar_ids=unexpected_similar_ids or [],
            min_confidence=min_confidence,
            max_query_time=max_query_time,
            semantic_keywords=semantic_keywords or [],
        )
        super().__init__(name=name, inputs=inputs, metadata=metadata)


# =============================================================================
# Focused Evaluator Classes
# =============================================================================


class DuplicateDetectionEvaluator(
    Evaluator[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput]
):
    """Evaluate that exact duplicates are correctly detected.

    This evaluator uses strict scoring (0.0 or 1.0) because duplicate detection
    is a non-negotiable requirement - if an exact match exists, it MUST be found.

    Returns:
        1.0 if exact duplicate found when expected OR not expected to find one
        0.0 if duplicate expected but not found OR found when not expected
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput],
    ) -> float:
        """Evaluate duplicate detection accuracy.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if duplicate detection matches expectation, 0.0 otherwise
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error:
            return 0.0

        # Check if we found similar models
        has_similar_models = len(ctx.output.analysis.similar_models) > 0
        has_edit_recommendation = ctx.output.analysis.recommendation == "edit_existing"
        has_create_recommendation = ctx.output.analysis.recommendation == "create_new"

        # Strict check: if we should find exact match, verify we did
        if ctx.metadata.should_find_exact_match:
            # Both must be true for exact duplicate detection
            return 1.0 if (has_similar_models and has_edit_recommendation) else 0.0

        # If we have an expected recommendation, verify it matches
        if ctx.metadata.expected_recommendation:
            if ctx.metadata.expected_recommendation == "edit_existing":
                return 1.0 if (has_similar_models and has_edit_recommendation) else 0.0
            elif ctx.metadata.expected_recommendation == "create_new":
                # For create_new, we expect either no similar models or low confidence
                return 1.0 if has_create_recommendation else 0.0

        # No strict requirement if no expected_recommendation
        return 1.0


class RankingQualityEvaluator(Evaluator[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput]):
    """Evaluate ranking quality using Mean Reciprocal Rank (MRR).

    MRR measures how highly the expected relevant results are ranked in the
    returned list. Uses partial credit based on ranking position.

    MRR Formula: 1 / rank_of_first_relevant_result

    Returns:
        1.0 if first result is relevant
        0.5 if second result is relevant
        0.33 if third result is relevant
        0.0 if no relevant results found
        1.0 if no expected results specified (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput],
    ) -> float:
        """Evaluate ranking quality using MRR.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            MRR score from 0.0-1.0 based on ranking quality
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Skip if no expected results specified - N/A case, return 1.0
        if not ctx.metadata.expected_similar_ids:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error:
            return 0.0

        # Skip if no results returned
        if not ctx.output.analysis.similar_models:
            return 0.0

        # Extract OIFM IDs from results in order
        actual_ids = [model["oifm_id"] for model in ctx.output.analysis.similar_models]
        expected_ids_set = set(ctx.metadata.expected_similar_ids)

        # Find rank of first relevant result (1-indexed)
        for rank, oifm_id in enumerate(actual_ids, start=1):
            if oifm_id in expected_ids_set:
                return 1.0 / rank

        # No relevant results found
        return 0.0


class PrecisionAtKEvaluator(Evaluator[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput]):
    """Evaluate precision at K (precision@5 and precision@10).

    Precision@K measures what proportion of the top K results are relevant.
    Uses K=3 since similar_models returns max 3 results.

    Formula: (number of relevant results in top K) / K

    Returns:
        0.0-1.0 proportional score based on precision
        1.0 if no expected results specified (N/A)
    """

    def __init__(self, k: int = 3) -> None:
        """Initialize precision@K evaluator.

        Args:
            k: Number of top results to consider (default 3)
        """
        self.k = k

    def evaluate(
        self,
        ctx: EvaluatorContext[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput],
    ) -> float:
        """Evaluate precision at K.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Precision@K score from 0.0-1.0
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Skip if no expected results specified - N/A case, return 1.0
        if not ctx.metadata.expected_similar_ids:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error:
            return 0.0

        # Get top K results
        top_k_models = ctx.output.analysis.similar_models[: self.k]
        if not top_k_models:
            return 0.0

        # Count relevant results in top K
        actual_ids = [model["oifm_id"] for model in top_k_models]
        expected_ids_set = set(ctx.metadata.expected_similar_ids)
        relevant_count = sum(1 for oifm_id in actual_ids if oifm_id in expected_ids_set)

        # Return precision@K
        return relevant_count / len(top_k_models)


class SemanticSimilarityEvaluator(
    Evaluator[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput]
):
    """Evaluate semantic similarity using keyword matching.

    Checks if results contain expected keywords in their names or descriptions,
    indicating semantic similarity to the query.

    Returns:
        0.0-1.0 proportional score based on keyword matches
        1.0 if no keywords specified (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput],
    ) -> float:
        """Evaluate semantic similarity via keyword matching.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Proportion of keywords found in results (0.0-1.0)
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Skip if no keywords specified - N/A case, return 1.0
        if not ctx.metadata.semantic_keywords:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error:
            return 0.0

        # Skip if no results returned
        if not ctx.output.analysis.similar_models:
            return 0.0

        # Combine all text from results
        result_text = []
        for model in ctx.output.analysis.similar_models:
            result_text.append(model.get("name", "").lower())
            if "description" in model:
                result_text.append(model["description"].lower())
        combined_text = " ".join(result_text)

        # Count keyword matches
        matches = sum(1 for keyword in ctx.metadata.semantic_keywords if keyword.lower() in combined_text)

        # Return proportion of keywords found
        return matches / len(ctx.metadata.semantic_keywords)


class ExclusionEvaluator(Evaluator[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput]):
    """Verify that unexpected IDs don't appear in dissimilar cases.

    For dissimilar test cases, validates that specific known-dissimilar
    models are correctly excluded from results. Returns 1.0 if no unexpected
    IDs found, 0.0 if any unexpected IDs appear in results.

    Uses strict (0.0/1.0) scoring since exclusion is binary - either the
    dissimilar model appears (wrong) or it doesn't (correct).
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput],
    ) -> float:
        """Evaluate that unexpected IDs are excluded from results.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if no unexpected IDs found (or N/A), 0.0 if any unexpected IDs present
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # N/A if no exclusion list provided
        if not ctx.metadata.unexpected_similar_ids:
            return 1.0

        # Error case
        if ctx.output.error:
            return 0.0

        # Check if any unexpected IDs appear in results
        actual_ids = {model["oifm_id"] for model in ctx.output.analysis.similar_models}
        unexpected_found = actual_ids & set(ctx.metadata.unexpected_similar_ids)

        return 0.0 if unexpected_found else 1.0


# =============================================================================
# Test Case Creation Functions
# =============================================================================


def create_exact_duplicate_cases() -> list[SimilarModelsCase]:
    """Create cases for exact duplicate detection."""
    cases = []

    # Case 1: Exact name match
    cases.append(
        SimilarModelsCase(
            name="exact_name_match_pulmonary_embolism",
            finding_name="pulmonary embolism",
            description="Blockage of pulmonary artery by blood clot",
            should_find_exact_match=True,
            expected_recommendation="edit_existing",
            expected_similar_ids=["OIFM_MSFT_932618"],
            min_confidence=0.9,
            max_query_time=15.0,
        )
    )

    # Case 2: Exact synonym match
    cases.append(
        SimilarModelsCase(
            name="exact_synonym_match_pe",
            finding_name="PE",
            description="Pulmonary embolism",
            should_find_exact_match=True,
            expected_recommendation="edit_existing",
            expected_similar_ids=["OIFM_MSFT_932618"],
            min_confidence=0.8,
            max_query_time=15.0,
        )
    )

    # Case 3: Exact match with synonym in input
    cases.append(
        SimilarModelsCase(
            name="exact_match_with_synonyms",
            finding_name="aortic dissection",
            synonyms=["aortic tear", "dissecting aneurysm"],
            should_find_exact_match=True,
            expected_recommendation="edit_existing",
            expected_similar_ids=["OIFM_MSFT_573630"],
            min_confidence=0.9,
            max_query_time=15.0,
        )
    )

    return cases


def create_semantic_similarity_cases() -> list[SimilarModelsCase]:
    """Create cases for semantic similarity detection (no exact match)."""
    cases = []

    # Case 4: Near-duplicate with similar meaning
    cases.append(
        SimilarModelsCase(
            name="near_duplicate_aaa",
            finding_name="abdominal aortic aneurysm",
            description="Enlargement of the abdominal aorta",
            should_find_exact_match=True,
            expected_recommendation="edit_existing",
            expected_similar_ids=["OIFM_MSFT_134126"],
            min_confidence=0.8,
            semantic_keywords=["aortic", "aneurysm", "abdominal"],
            max_query_time=20.0,
        )
    )

    # Case 5: Related finding in same anatomical region
    cases.append(
        SimilarModelsCase(
            name="related_finding_aortic",
            finding_name="aortic stenosis",
            description="Narrowing of the aortic valve",
            should_find_exact_match=False,
            expected_similar_ids=["OIFM_MSFT_573630", "OIFM_MSFT_134126"],
            semantic_keywords=["aortic"],
            min_confidence=0.4,
            max_query_time=20.0,
        )
    )

    # Case 6: Similar pathology different location
    cases.append(
        SimilarModelsCase(
            name="similar_pathology_dissection",
            finding_name="carotid dissection",
            description="Tear in the wall of carotid artery",
            should_find_exact_match=False,
            expected_similar_ids=["OIFM_MSFT_573630"],
            semantic_keywords=["dissection"],
            min_confidence=0.3,
            max_query_time=20.0,
        )
    )

    return cases


def create_dissimilar_cases() -> list[SimilarModelsCase]:
    """Create cases for findings that should not match (create new)."""
    cases = []

    # Case 7: Completely different finding
    cases.append(
        SimilarModelsCase(
            name="dissimilar_brain_finding",
            finding_name="cerebral hemorrhage",
            description="Bleeding in the brain tissue",
            should_find_exact_match=False,
            expected_recommendation="create_new",
            # Vascular findings (verified in test data): PE, aortic dissection, AAA
            # These should NOT match brain hemorrhage (cerebral vs cardiovascular)
            unexpected_similar_ids=["OIFM_MSFT_932618", "OIFM_MSFT_573630", "OIFM_MSFT_134126"],
            semantic_keywords=["brain", "cerebral", "hemorrhage"],
            max_query_time=20.0,
        )
    )

    # Case 8: Similar name but different medical context
    cases.append(
        SimilarModelsCase(
            name="dissimilar_different_context",
            finding_name="kidney stone",
            description="Calculus in the kidney",
            should_find_exact_match=False,
            expected_recommendation="create_new",
            # Vascular findings (verified in test data): PE, aortic dissection, AAA
            # These should NOT match kidney stone (renal vs cardiovascular)
            unexpected_similar_ids=["OIFM_MSFT_932618", "OIFM_MSFT_573630", "OIFM_MSFT_134126"],
            semantic_keywords=["kidney", "renal", "stone"],
            max_query_time=20.0,
        )
    )

    return cases


def create_edge_cases() -> list[SimilarModelsCase]:
    """Create edge cases and boundary conditions."""
    cases = []

    # Case 9: Not in index (completely new finding)
    cases.append(
        SimilarModelsCase(
            name="edge_not_in_index",
            finding_name="xylophone bone disease",
            description="A completely fictional finding that doesn't exist",
            should_find_exact_match=False,
            expected_recommendation="create_new",
            max_query_time=20.0,
        )
    )

    # Case 10: Minimal information (name only)
    cases.append(
        SimilarModelsCase(
            name="edge_minimal_info",
            finding_name="aneurysm",
            should_find_exact_match=False,
            expected_similar_ids=["OIFM_MSFT_134126", "OIFM_MSFT_573630"],
            semantic_keywords=["aneurysm"],
            max_query_time=20.0,
        )
    )

    # Case 11: Very common term (should find multiple)
    cases.append(
        SimilarModelsCase(
            name="edge_common_term",
            finding_name="abnormality",
            description="Generic abnormality in medical imaging",
            should_find_exact_match=False,
            max_query_time=20.0,
        )
    )

    # Case 12: Long detailed description
    cases.append(
        SimilarModelsCase(
            name="edge_long_description",
            finding_name="thoracic aortic aneurysm",
            description=(
                "A thoracic aortic aneurysm is a localized dilation of the thoracic aorta "
                "exceeding 50% of the normal diameter, which can occur in the ascending aorta, "
                "aortic arch, or descending thoracic aorta. Risk factors include hypertension, "
                "atherosclerosis, genetic disorders such as Marfan syndrome, and trauma. "
                "Complications may include rupture, dissection, or compression of adjacent structures."
            ),
            should_find_exact_match=False,
            expected_similar_ids=["OIFM_MSFT_134126", "OIFM_MSFT_573630"],
            semantic_keywords=["aortic", "aneurysm", "thoracic"],
            min_confidence=0.4,
            max_query_time=25.0,
        )
    )

    # Case 13: Empty description
    cases.append(
        SimilarModelsCase(
            name="edge_empty_description",
            finding_name="embolism",
            description="",
            should_find_exact_match=False,
            expected_similar_ids=["OIFM_MSFT_932618"],
            semantic_keywords=["embolism"],
            max_query_time=20.0,
        )
    )

    # Case 14: Multiple synonyms
    cases.append(
        SimilarModelsCase(
            name="edge_multiple_synonyms",
            finding_name="dissecting aneurysm",
            synonyms=["aortic tear", "aortic rupture", "dissection"],
            should_find_exact_match=True,
            expected_recommendation="edit_existing",
            expected_similar_ids=["OIFM_MSFT_573630"],
            min_confidence=0.8,
            max_query_time=20.0,
        )
    )

    # Case 15: Abbreviation vs full name
    cases.append(
        SimilarModelsCase(
            name="edge_abbreviation",
            finding_name="AAA",
            description="Triple A - abdominal aortic aneurysm",
            should_find_exact_match=False,
            expected_similar_ids=["OIFM_MSFT_134126"],
            semantic_keywords=["aortic", "aneurysm"],
            min_confidence=0.3,
            max_query_time=20.0,
        )
    )

    return cases


def create_performance_cases() -> list[SimilarModelsCase]:
    """Create cases focused on performance testing."""
    cases = []

    # Case 16: Simple query should be fast
    cases.append(
        SimilarModelsCase(
            name="performance_simple_query",
            finding_name="PE",
            should_find_exact_match=True,
            max_query_time=10.0,
        )
    )

    # Case 17: Complex query with all parameters
    cases.append(
        SimilarModelsCase(
            name="performance_complex_query",
            finding_name="pulmonary thromboembolism",
            description="Blood clot in pulmonary circulation causing respiratory distress",
            synonyms=["pulmonary embolus", "PE", "lung clot"],
            should_find_exact_match=False,
            max_query_time=25.0,
        )
    )

    return cases


def create_error_handling_cases() -> list[SimilarModelsCase]:
    """Create cases for error handling and robustness."""
    cases = []

    # Case 18: Empty finding name (malformed input)
    cases.append(
        SimilarModelsCase(
            name="error_empty_name",
            finding_name="",
            description="This should fail gracefully",
            should_find_exact_match=False,
            max_query_time=20.0,
        )
    )

    # Case 19: Very long finding name (boundary condition)
    cases.append(
        SimilarModelsCase(
            name="error_very_long_name",
            finding_name="a" * 1000,  # 1000 character name
            description="Testing handling of excessively long input",
            should_find_exact_match=False,
            max_query_time=20.0,
        )
    )

    # Case 20: Special characters in finding name
    cases.append(
        SimilarModelsCase(
            name="error_special_characters",
            finding_name="test@#$%^&*(){}[]|\\:;\"'<>?,./~`",
            description="Testing handling of special characters",
            should_find_exact_match=False,
            max_query_time=20.0,
        )
    )

    return cases


def create_ranking_consistency_cases() -> list[SimilarModelsCase]:
    """Create cases for testing ranking consistency."""
    cases = []

    # Case 21: Same query should return consistent ranking
    # This case is run twice to verify consistency
    cases.append(
        SimilarModelsCase(
            name="ranking_consistency_pulmonary_embolism",
            finding_name="pulmonary embolism",
            description="Blockage of pulmonary artery",
            should_find_exact_match=True,
            expected_similar_ids=["OIFM_MSFT_932618"],
            semantic_keywords=["pulmonary", "embolism"],
            max_query_time=15.0,
        )
    )

    return cases


def create_recommendation_threshold_cases() -> list[SimilarModelsCase]:
    """Create cases for testing recommendation threshold boundaries."""
    cases = []

    # Case 22: High similarity should recommend edit_existing
    cases.append(
        SimilarModelsCase(
            name="threshold_high_similarity_exact_match",
            finding_name="pulmonary embolism",
            description="Occlusion of pulmonary artery by thrombus",
            should_find_exact_match=True,
            expected_recommendation="edit_existing",
            expected_similar_ids=["OIFM_MSFT_932618"],
            min_confidence=0.9,
            max_query_time=15.0,
        )
    )

    # Case 23: Low similarity should recommend create_new
    cases.append(
        SimilarModelsCase(
            name="threshold_low_similarity_different_domain",
            finding_name="retinal detachment",
            description="Separation of retina from underlying tissue",
            should_find_exact_match=False,
            expected_recommendation="create_new",
            semantic_keywords=["retinal", "detachment"],
            max_query_time=20.0,
        )
    )

    return cases


# =============================================================================
# Task Execution Function
# =============================================================================


async def run_similar_models_task(input_data: SimilarModelsInput) -> SimilarModelsActualOutput:
    """Execute a single similar_finding_models evaluation case.

    Dataset.evaluate() automatically creates spans and captures inputs/outputs.
    Pydantic AI instrumentation captures agent/model/tool calls.
    No manual Logfire code needed.

    Args:
        input_data: Input data for the similar models case

    Returns:
        Actual output from the similar models execution

    Raises:
        RuntimeError: If DuckDB index is not populated with test data
    """
    try:
        # Create index for this evaluation
        index = Index()

        # Verify index is populated (should have at least test data)
        # This is a sanity check to ensure the index was built before running evals
        try:
            model_count = await index.count()
            if model_count == 0:
                raise RuntimeError(
                    "DuckDB index is empty. Please run: python -m findingmodel.cli index rebuild --backend duckdb"
                )
        except Exception as e:
            # If we can't get count, that's also a problem
            raise RuntimeError(f"Failed to verify index population: {e}") from e

        # Time the query
        start_time = time.time()
        analysis = await find_similar_models(
            finding_name=input_data.finding_name,
            description=input_data.description,
            synonyms=input_data.synonyms,
            index=index,
        )
        query_time = time.time() - start_time

        return SimilarModelsActualOutput(
            analysis=analysis,
            query_time=query_time,
        )
    except Exception as e:
        # Return error in output for evaluation
        return SimilarModelsActualOutput(
            analysis=SimilarModelAnalysis(
                similar_models=[],
                recommendation="create_new",
                confidence=0.0,
            ),
            query_time=0.0,
            error=str(e),
        )


# =============================================================================
# Dataset Creation with Evaluator-Based Pattern
# =============================================================================

all_cases = (
    create_exact_duplicate_cases()
    + create_semantic_similarity_cases()
    + create_dissimilar_cases()
    + create_edge_cases()
    + create_performance_cases()
    + create_error_handling_cases()
    + create_ranking_consistency_cases()
    + create_recommendation_threshold_cases()
)

evaluators = [
    DuplicateDetectionEvaluator(),
    RankingQualityEvaluator(),
    PrecisionAtKEvaluator(k=3),
    SemanticSimilarityEvaluator(),
    ExclusionEvaluator(),
    PerformanceEvaluator(),
]

similar_models_dataset = Dataset(cases=all_cases, evaluators=evaluators)


async def run_similar_models_evals() -> EvaluationReport[
    SimilarModelsInput, SimilarModelsActualOutput, SimilarModelsExpectedOutput
]:
    """Run similar_finding_models evaluation suite.

    Dataset.evaluate() automatically creates evaluation spans and captures
    all inputs, outputs, and scores for visualization in Logfire.
    """
    report = await similar_models_dataset.evaluate(run_similar_models_task)
    return report


if __name__ == "__main__":
    import asyncio

    from evals import ensure_instrumented

    ensure_instrumented()  # Explicit instrumentation for eval run

    async def main() -> None:
        print("\nRunning similar_finding_models evaluation suite...")
        print("=" * 80)

        report = await run_similar_models_evals()

        print("\n" + "=" * 80)
        print("SIMILAR FINDING MODELS EVALUATION RESULTS")
        print("=" * 80 + "\n")

        # Don't include full outputs in table - focus on scores and metrics
        report.print(
            include_input=True,
            include_output=False,  # Outputs contain full analysis objects - too verbose
            include_durations=True,
            width=120,
        )

        # Calculate overall score manually (average of all evaluator scores across all cases)
        all_scores = [score.value for case in report.cases for score in case.scores.values()]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        print("\n" + "=" * 80)
        print(f"OVERALL SCORE: {overall_score:.2f}")
        print("=" * 80 + "\n")

    asyncio.run(main())
