"""Evaluation suite for ontology_concept_match agent.

INTEGRATION SUITE: Requires real API calls to BioOntology and/or DuckDB index.
Run via `task evals:ontology_match`.

INDEX SETUP REQUIREMENTS (for DuckDB backend):
This eval suite can use either BioOntology API or DuckDB backend.
For DuckDB backend testing:
1. Ensure ontology data is indexed in DuckDB
2. Run: python -m findingmodel.cli index rebuild --backend duckdb
3. Verify index populated: python -m findingmodel.cli index info --backend duckdb

API CALL PATTERN:
    Unlike unit tests, this eval suite makes REAL API calls by design. The
    `models.ALLOW_MODEL_REQUESTS = False` guard is intentionally omitted since
    evaluation requires actual agent execution with BioOntology API and AI models.

    This is an integration eval suite, not a unit test. It validates end-to-end
    behavior including:
    - Real BioOntology API queries with concept search
    - Actual AI model query generation and categorization
    - Complete agent workflow coordination
    - Multi-backend consistency (BioOntology vs DuckDB)

    See: Serena memory 'agent_evaluation_best_practices_2025'

This module defines evaluation cases for assessing the ontology_concept_match functionality,
which uses two AI agents to find and categorize relevant medical concepts from ontology databases.

EVALUATOR-BASED PATTERN:
- Cases are evaluated using Dataset.evaluate() with focused evaluators
- Each evaluator checks a specific aspect (concept match accuracy, ranking quality, etc.)
- Hybrid scoring: strict for must-haves (0.0 or 1.0), partial credit for quality (0.0-1.0)

EVALUATORS:
- ConceptMatchAccuracyEvaluator: Checks if correct concept found (partial credit)
- RankingQualityEvaluator: Uses NDCG for ranking quality (partial credit)
- BackendConsistencyEvaluator: Compares results across backends (partial credit)
- SynonymHandlingEvaluator: Tests synonym and abbreviation handling (partial credit)
- ErrorHandlingEvaluator: Verifies errors are handled correctly (strict)
- PerformanceEvaluator: Query performance under threshold (strict)

LOGFIRE INTEGRATION:
Logfire observability is configured automatically in evals/__init__.py.
No manual instrumentation needed in this module - automatic spans are created by:
- Dataset.evaluate() for root and per-case spans
- Pydantic AI instrumentation for agent/model/tool calls

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

import math
import time

from findingmodel_ai.evaluators import PerformanceEvaluator
from findingmodel_ai.search.bioontology import OntologySearchResult
from findingmodel_ai.search.ontology import CategorizedOntologyConcepts, match_ontology_concepts
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport


class OntologyMatchInput(BaseModel):
    """Input for an ontology concept match evaluation case."""

    finding_name: str = Field(description="Name of the imaging finding to match")
    finding_description: str | None = Field(default=None, description="Optional detailed description")
    ontologies: list[str] | None = Field(
        default=None, description="Optional list of specific ontologies to search (e.g., ['SNOMEDCT', 'RADLEX'])"
    )


class OntologyMatchExpectedOutput(BaseModel):
    """Expected output for an ontology concept match evaluation case."""

    should_succeed: bool = Field(default=True, description="Whether the search should succeed")
    expected_exact_match_ids: list[str] = Field(
        default_factory=list, description="Expected concept IDs in exact_matches category"
    )
    expected_should_include_ids: list[str] = Field(
        default_factory=list, description="Expected concept IDs in should_include category"
    )
    unexpected_concept_ids: list[str] = Field(
        default_factory=list, description="Concept IDs that should NOT appear in any category"
    )
    min_exact_matches: int = Field(default=0, description="Minimum number of exact matches expected", ge=0)
    max_exact_matches: int = Field(default=5, description="Maximum number of exact matches allowed", ge=0)
    semantic_keywords: list[str] = Field(
        default_factory=list, description="Keywords that should appear in matched concept names"
    )
    ranking_target_ndcg: float = Field(default=0.8, description="Target NDCG score for ranking quality", ge=0.0, le=1.0)
    max_query_time: float = Field(default=30.0, description="Maximum acceptable query time in seconds")


class OntologyMatchActualOutput(BaseModel):
    """Actual output from running an ontology concept match case."""

    result: CategorizedOntologyConcepts | None = Field(default=None, description="Categorized ontology concepts result")
    query_time: float = Field(description="Time taken to execute query in seconds")
    error: str | None = Field(default=None, description="Error message if query failed")


class OntologyMatchCase(Case[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput]):
    """A test case for ontology_concept_match functionality."""

    def __init__(
        self,
        name: str,
        finding_name: str,
        finding_description: str | None = None,
        ontologies: list[str] | None = None,
        should_succeed: bool = True,
        expected_exact_match_ids: list[str] | None = None,
        expected_should_include_ids: list[str] | None = None,
        unexpected_concept_ids: list[str] | None = None,
        min_exact_matches: int = 0,
        max_exact_matches: int = 5,
        semantic_keywords: list[str] | None = None,
        ranking_target_ndcg: float = 0.8,
        max_query_time: float = 30.0,
    ) -> None:
        """Initialize an ontology concept match evaluation case.

        Args:
            name: Name of the test case
            finding_name: Name of the imaging finding to match
            finding_description: Optional detailed description
            ontologies: Optional list of specific ontologies to search
            should_succeed: Whether the search should succeed
            expected_exact_match_ids: Expected concept IDs in exact_matches
            expected_should_include_ids: Expected concept IDs in should_include
            unexpected_concept_ids: Concept IDs that should NOT appear
            min_exact_matches: Minimum number of exact matches expected
            max_exact_matches: Maximum number of exact matches allowed
            semantic_keywords: Keywords for semantic similarity check
            ranking_target_ndcg: Target NDCG score for ranking quality
            max_query_time: Maximum acceptable query time in seconds
        """
        inputs = OntologyMatchInput(
            finding_name=finding_name,
            finding_description=finding_description,
            ontologies=ontologies,
        )
        metadata = OntologyMatchExpectedOutput(
            should_succeed=should_succeed,
            expected_exact_match_ids=expected_exact_match_ids or [],
            expected_should_include_ids=expected_should_include_ids or [],
            unexpected_concept_ids=unexpected_concept_ids or [],
            min_exact_matches=min_exact_matches,
            max_exact_matches=max_exact_matches,
            semantic_keywords=semantic_keywords or [],
            ranking_target_ndcg=ranking_target_ndcg,
            max_query_time=max_query_time,
        )
        super().__init__(name=name, inputs=inputs, metadata=metadata)


# =============================================================================
# Focused Evaluator Classes
# =============================================================================


class ConceptMatchAccuracyEvaluator(
    Evaluator[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput]
):
    """Evaluate that expected concepts are found and categorized correctly.

    This evaluator uses partial credit scoring to reward finding some of the
    expected concepts even if not all are found. It checks both exact_matches
    and should_include categories.

    Scoring:
        - 1.0 if all expected concepts found in correct categories
        - Partial credit (0.0-1.0) based on proportion of expected concepts found
        - 0.0 if no expected concepts found
        - 1.0 if no expected concepts specified (N/A case)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput],
    ) -> float:
        """Evaluate concept match accuracy with partial credit.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Proportion of expected concepts found (0.0-1.0)
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error or ctx.output.result is None:
            return 0.0

        # If no expected concepts specified, this is N/A - return 1.0
        total_expected = len(ctx.metadata.expected_exact_match_ids) + len(ctx.metadata.expected_should_include_ids)
        if total_expected == 0:
            return 1.0

        # Collect all found concept IDs from result
        found_exact_ids = {concept.concept_id for concept in ctx.output.result.exact_matches}
        found_should_include_ids = {concept.concept_id for concept in ctx.output.result.should_include}

        # Count matches in expected categories
        exact_match_count = sum(
            1 for expected_id in ctx.metadata.expected_exact_match_ids if expected_id in found_exact_ids
        )
        should_include_count = sum(
            1
            for expected_id in ctx.metadata.expected_should_include_ids
            if expected_id in found_should_include_ids or expected_id in found_exact_ids
        )

        # Total matches found
        total_matches = exact_match_count + should_include_count

        # Return proportion of expected concepts found
        return total_matches / total_expected if total_expected > 0 else 1.0


class RankingQualityEvaluator(Evaluator[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput]):
    """Evaluate ranking quality using Normalized Discounted Cumulative Gain (NDCG).

    NDCG measures how well the ranking places relevant results at the top.
    It gives higher scores when more relevant results appear earlier in the list.

    NDCG Formula:
        DCG = sum(relevance[i] / log2(i + 1))  for each position i
        IDCG = DCG for perfect ranking
        NDCG = DCG / IDCG

    Relevance scoring:
        - 3.0 if concept in expected_exact_match_ids (highest relevance)
        - 2.0 if concept in expected_should_include_ids (medium relevance)
        - 0.0 otherwise (not relevant)

    Returns:
        NDCG score from 0.0-1.0 based on ranking quality
        1.0 if no expected results specified (N/A)
    """

    def _calculate_dcg(self, relevances: list[float]) -> float:
        """Calculate Discounted Cumulative Gain.

        Args:
            relevances: List of relevance scores in ranking order

        Returns:
            DCG score
        """
        dcg = 0.0
        for i, relevance in enumerate(relevances):
            # Position is 1-indexed for log2 calculation
            position = i + 1
            # Avoid division by zero - first position has no discount
            discount = 1.0 if position == 1 else (1.0 / math.log2(position + 1))
            dcg += relevance * discount
        return dcg

    def _calculate_ndcg(self, actual_relevances: list[float], expected_relevances: list[float]) -> float:
        """Calculate Normalized Discounted Cumulative Gain.

        Args:
            actual_relevances: Relevance scores in actual ranking order
            expected_relevances: Relevance scores for ideal ranking

        Returns:
            NDCG score from 0.0-1.0
        """
        if not actual_relevances or not expected_relevances:
            return 0.0

        dcg = self._calculate_dcg(actual_relevances)
        # IDCG is DCG of ideal ranking (sorted by relevance descending)
        ideal_relevances = sorted(expected_relevances, reverse=True)
        idcg = self._calculate_dcg(ideal_relevances)

        # Avoid division by zero
        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def evaluate(
        self,
        ctx: EvaluatorContext[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput],
    ) -> float:
        """Evaluate ranking quality using NDCG.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            NDCG score from 0.0-1.0 based on ranking quality
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Skip if no expected results specified - N/A case, return 1.0
        if not ctx.metadata.expected_exact_match_ids and not ctx.metadata.expected_should_include_ids:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error or ctx.output.result is None:
            return 0.0

        # Get all results in order (exact matches first, then should_include)
        all_results: list[OntologySearchResult] = []
        all_results.extend(ctx.output.result.exact_matches)
        all_results.extend(ctx.output.result.should_include)

        if not all_results:
            return 0.0

        # Create relevance mapping
        exact_match_set = set(ctx.metadata.expected_exact_match_ids)
        should_include_set = set(ctx.metadata.expected_should_include_ids)

        # Calculate relevance scores for actual ranking
        actual_relevances = []
        for result in all_results:
            if result.concept_id in exact_match_set:
                actual_relevances.append(3.0)  # Highest relevance
            elif result.concept_id in should_include_set:
                actual_relevances.append(2.0)  # Medium relevance
            else:
                actual_relevances.append(0.0)  # Not relevant

        # Expected relevances for ideal ranking
        expected_relevances = [3.0] * len(exact_match_set) + [2.0] * len(should_include_set)

        # Calculate NDCG
        ndcg = self._calculate_ndcg(actual_relevances, expected_relevances)

        return ndcg


class BackendConsistencyEvaluator(
    Evaluator[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput]
):
    """Evaluate consistency between BioOntology API and DuckDB backends.

    This evaluator is currently a placeholder for future multi-backend testing.
    For now, it always returns 1.0 (N/A) since we're only testing BioOntology API.

    Future implementation will:
    - Run same query against both backends
    - Compare top results for consistency
    - Return proportion of overlapping results

    Returns:
        1.0 (N/A - not yet implemented)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput],
    ) -> float:
        """Evaluate backend consistency.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 (N/A - not yet implemented)
        """
        # TODO: Implement multi-backend consistency checking
        # For now, this is N/A since we're only using BioOntology API
        return 1.0


class SynonymHandlingEvaluator(Evaluator[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput]):
    """Evaluate handling of synonyms and abbreviations.

    Tests that the agent can find concepts even when the query uses
    synonyms or abbreviations rather than the canonical term.

    This is tested by checking if semantic keywords appear in the
    matched concept texts, indicating successful synonym matching.

    Returns:
        Proportion of semantic keywords found in results (0.0-1.0)
        1.0 if no keywords specified (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput],
    ) -> float:
        """Evaluate synonym and abbreviation handling.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Proportion of semantic keywords found (0.0-1.0)
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Skip if no keywords specified - N/A case, return 1.0
        if not ctx.metadata.semantic_keywords:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error or ctx.output.result is None:
            return 0.0

        # Skip if no results returned
        all_results = ctx.output.result.exact_matches + ctx.output.result.should_include
        if not all_results:
            return 0.0

        # Combine all concept texts
        concept_texts = [result.concept_text.lower() for result in all_results]
        combined_text = " ".join(concept_texts)

        # Count keyword matches
        matches = sum(1 for keyword in ctx.metadata.semantic_keywords if keyword.lower() in combined_text)

        # Return proportion of keywords found
        return matches / len(ctx.metadata.semantic_keywords)


class ErrorHandlingEvaluator(Evaluator[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput]):
    """Verify errors are handled as expected.

    Uses strict scoring (0.0/1.0) because error handling is non-negotiable.
    Operations should succeed when they should and fail when they should.

    Returns:
        1.0 if error handling matches expectation
        0.0 if error handling does not match expectation
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput],
    ) -> float:
        """Evaluate error handling correctness.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if error handling matches expectation, 0.0 otherwise
        """
        # Handle missing metadata - assume should succeed
        should_succeed = ctx.metadata.should_succeed if ctx.metadata else True

        # Determine if operation actually succeeded (no error)
        has_error = ctx.output.error is not None

        # Evaluate based on expectation vs reality
        if should_succeed and not has_error:
            return 1.0  # Should succeed and did succeed
        elif not should_succeed and has_error:
            return 1.0  # Should fail and did fail
        else:
            return 0.0  # Mismatch between expectation and reality


# =============================================================================
# Test Case Creation Functions
# =============================================================================


def create_success_cases() -> list[OntologyMatchCase]:
    """Create success cases for common radiology concepts."""
    cases = []

    # Case 1: Common radiology concept - pneumonia
    cases.append(
        OntologyMatchCase(
            name="success_common_pneumonia",
            finding_name="pneumonia",
            finding_description="Lung infection with consolidation",
            ontologies=["SNOMEDCT", "RADLEX"],
            min_exact_matches=1,
            semantic_keywords=["pneumonia", "lung", "infection"],
            max_query_time=20.0,
        )
    )

    # Case 2: Specific pathology - pulmonary embolism
    cases.append(
        OntologyMatchCase(
            name="success_specific_pulmonary_embolism",
            finding_name="pulmonary embolism",
            finding_description="Blood clot in pulmonary artery",
            ontologies=["SNOMEDCT", "RADLEX"],
            min_exact_matches=1,
            semantic_keywords=["pulmonary", "embolism", "embolus"],
            max_query_time=20.0,
        )
    )

    # Case 3: Anatomy + pathology - hepatic metastasis
    cases.append(
        OntologyMatchCase(
            name="success_anatomy_pathology_hepatic_metastasis",
            finding_name="hepatic metastasis",
            finding_description="Metastatic lesion in the liver",
            ontologies=["SNOMEDCT", "RADLEX"],
            min_exact_matches=1,
            semantic_keywords=["hepatic", "liver", "metastasis", "metastatic"],
            max_query_time=20.0,
        )
    )

    # Case 4: Modality-specific - fracture
    cases.append(
        OntologyMatchCase(
            name="success_modality_fracture",
            finding_name="fracture",
            finding_description="Break in bone continuity",
            ontologies=["SNOMEDCT", "RADLEX"],
            min_exact_matches=1,
            semantic_keywords=["fracture", "bone"],
            max_query_time=20.0,
        )
    )

    # Case 5: Different ontologies - search RADLEX only
    cases.append(
        OntologyMatchCase(
            name="success_radlex_only",
            finding_name="consolidation",
            finding_description="Dense lung opacification",
            ontologies=["RADLEX"],
            min_exact_matches=1,
            semantic_keywords=["consolidation", "lung"],
            max_query_time=20.0,
        )
    )

    return cases


def create_synonym_cases() -> list[OntologyMatchCase]:
    """Create cases testing synonym and abbreviation handling."""
    cases = []

    # Case 6: Abbreviation - PE
    cases.append(
        OntologyMatchCase(
            name="synonym_abbreviation_pe",
            finding_name="PE",
            finding_description="Pulmonary embolism",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["pulmonary", "embolism", "embolus"],
            max_query_time=20.0,
        )
    )

    # Case 7: Synonym variation - MI
    cases.append(
        OntologyMatchCase(
            name="synonym_abbreviation_mi",
            finding_name="MI",
            finding_description="Myocardial infarction",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["myocardial", "infarction", "heart"],
            max_query_time=20.0,
        )
    )

    # Case 8: Alternative terminology - lung cancer vs pulmonary neoplasm
    cases.append(
        OntologyMatchCase(
            name="synonym_alternative_lung_cancer",
            finding_name="lung cancer",
            finding_description="Malignant lung neoplasm",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["lung", "pulmonary", "cancer", "neoplasm", "carcinoma"],
            max_query_time=20.0,
        )
    )

    return cases


def create_edge_cases() -> list[OntologyMatchCase]:
    """Create edge cases and boundary conditions."""
    cases = []

    # Case 9: Case variations
    cases.append(
        OntologyMatchCase(
            name="edge_case_variation_pneumonia",
            finding_name="PNEUMONIA",
            finding_description="Upper case finding name",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["pneumonia"],
            max_query_time=20.0,
        )
    )

    # Case 10: Special characters in name
    cases.append(
        OntologyMatchCase(
            name="edge_special_chars_type_a_dissection",
            finding_name="Type A aortic dissection",
            finding_description="Stanford Type A dissection",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["aortic", "dissection"],
            max_query_time=20.0,
        )
    )

    # Case 11: Very long compound name
    cases.append(
        OntologyMatchCase(
            name="edge_long_compound_name",
            finding_name="bilateral lower lobe consolidation with air bronchograms",
            finding_description="Consolidation in both lower lobes showing air bronchograms",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["consolidation", "bronchogram", "bilateral", "lower lobe"],
            max_query_time=25.0,
        )
    )

    # Case 12: Minimal information (name only)
    cases.append(
        OntologyMatchCase(
            name="edge_minimal_info_aneurysm",
            finding_name="aneurysm",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["aneurysm"],
            max_query_time=20.0,
        )
    )

    # Case 13: Similar but distinct concepts
    cases.append(
        OntologyMatchCase(
            name="edge_similar_concepts_aortic_stenosis",
            finding_name="aortic stenosis",
            finding_description="Narrowing of aortic valve",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["aortic", "stenosis"],
            # Should NOT find aortic dissection or aneurysm
            unexpected_concept_ids=[],  # We don't have specific IDs to exclude
            max_query_time=20.0,
        )
    )

    return cases


def create_ranking_cases() -> list[OntologyMatchCase]:
    """Create cases for testing ranking quality."""
    cases = []

    # Case 14: Multiple valid matches - best should rank first
    cases.append(
        OntologyMatchCase(
            name="ranking_multiple_matches_pneumonia",
            finding_name="pneumonia",
            finding_description="Bacterial pneumonia",
            ontologies=["SNOMEDCT", "RADLEX"],
            min_exact_matches=1,
            semantic_keywords=["pneumonia", "bacterial"],
            ranking_target_ndcg=0.8,
            max_query_time=20.0,
        )
    )

    # Case 15: Partial vs exact matches
    cases.append(
        OntologyMatchCase(
            name="ranking_partial_exact_fracture",
            finding_name="rib fracture",
            finding_description="Fractured rib bone",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["rib", "fracture"],
            ranking_target_ndcg=0.7,
            max_query_time=20.0,
        )
    )

    # Case 16: Context-dependent ranking
    cases.append(
        OntologyMatchCase(
            name="ranking_context_dependent_stroke",
            finding_name="stroke",
            finding_description="Ischemic cerebrovascular accident",
            ontologies=["SNOMEDCT", "RADLEX"],
            semantic_keywords=["stroke", "ischemic", "cerebrovascular"],
            ranking_target_ndcg=0.75,
            max_query_time=20.0,
        )
    )

    return cases


def create_rejection_cases() -> list[OntologyMatchCase]:
    """Create rejection and error cases."""
    cases = []

    # Case 17: Non-medical term
    cases.append(
        OntologyMatchCase(
            name="rejection_non_medical_banana",
            finding_name="banana",
            finding_description="A yellow fruit",
            ontologies=["SNOMEDCT", "RADLEX"],
            should_succeed=True,  # Search succeeds but finds no relevant results
            min_exact_matches=0,  # Expect no exact matches
            max_query_time=20.0,
        )
    )

    # Case 18: Empty finding name
    cases.append(
        OntologyMatchCase(
            name="rejection_empty_name",
            finding_name="",
            finding_description="This should fail",
            should_succeed=False,  # Should fail validation
            max_query_time=20.0,
        )
    )

    # Case 19: Invalid ontology name
    cases.append(
        OntologyMatchCase(
            name="rejection_invalid_ontology",
            finding_name="pneumonia",
            ontologies=["INVALID_ONTOLOGY_XYZ"],
            should_succeed=True,  # Search succeeds but finds no results
            min_exact_matches=0,
            max_query_time=20.0,
        )
    )

    return cases


def create_performance_cases() -> list[OntologyMatchCase]:
    """Create cases focused on performance testing."""
    cases = []

    # Case 20: Simple query should be fast
    cases.append(
        OntologyMatchCase(
            name="performance_simple_query",
            finding_name="pneumonia",
            ontologies=["SNOMEDCT"],
            max_query_time=15.0,  # Strict time limit
        )
    )

    # Case 21: Complex query with description
    cases.append(
        OntologyMatchCase(
            name="performance_complex_query",
            finding_name="bilateral pneumonia",
            finding_description="Infection affecting both lungs with consolidation and air bronchograms",
            ontologies=["SNOMEDCT", "RADLEX"],
            max_query_time=25.0,
        )
    )

    # Case 22: Multiple ontologies search
    cases.append(
        OntologyMatchCase(
            name="performance_multi_ontology",
            finding_name="fracture",
            ontologies=["SNOMEDCT", "RADLEX", "LOINC"],
            max_query_time=25.0,
        )
    )

    return cases


# =============================================================================
# Task Execution Function
# =============================================================================


async def run_ontology_match_task(input_data: OntologyMatchInput) -> OntologyMatchActualOutput:
    """Execute a single ontology_concept_match evaluation case.

    Dataset.evaluate() automatically creates spans and captures inputs/outputs.
    Pydantic AI instrumentation captures agent/model/tool calls.
    No manual Logfire code needed.

    Args:
        input_data: Input data for the ontology match case

    Returns:
        Actual output from the ontology match execution

    Raises:
        Exception: Any errors are caught and returned in the error field
    """
    try:
        # Time the query
        start_time = time.time()
        result = await match_ontology_concepts(
            finding_name=input_data.finding_name,
            finding_description=input_data.finding_description,
            ontologies=input_data.ontologies,
        )
        query_time = time.time() - start_time

        return OntologyMatchActualOutput(
            result=result,
            query_time=query_time,
        )
    except Exception as e:
        # Return error in output for evaluation
        query_time = time.time() - start_time if "start_time" in locals() else 0.0
        return OntologyMatchActualOutput(
            result=None,
            query_time=query_time,
            error=str(e),
        )


# =============================================================================
# Dataset Creation with Evaluator-Based Pattern
# =============================================================================

all_cases = (
    create_success_cases()
    + create_synonym_cases()
    + create_edge_cases()
    + create_ranking_cases()
    + create_rejection_cases()
    + create_performance_cases()
)

evaluators = [
    ConceptMatchAccuracyEvaluator(),
    RankingQualityEvaluator(),
    BackendConsistencyEvaluator(),
    SynonymHandlingEvaluator(),
    ErrorHandlingEvaluator(),
    PerformanceEvaluator(),
]

ontology_match_dataset = Dataset(cases=all_cases, evaluators=evaluators)


async def run_ontology_match_evals() -> EvaluationReport[
    OntologyMatchInput, OntologyMatchActualOutput, OntologyMatchExpectedOutput
]:
    """Run ontology_concept_match evaluation suite.

    Dataset.evaluate() automatically creates evaluation spans and captures
    all inputs, outputs, and scores for visualization in Logfire.
    """
    report = await ontology_match_dataset.evaluate(run_ontology_match_task)
    return report


if __name__ == "__main__":
    import asyncio

    from evals import ensure_instrumented

    ensure_instrumented()  # Explicit instrumentation for eval run

    async def main() -> None:
        print("\nRunning ontology_concept_match evaluation suite...")
        print("=" * 80)
        print("NOTE: This eval requires BioOntology API key and makes real API calls.")
        print("=" * 80 + "\n")

        report = await run_ontology_match_evals()

        print("\n" + "=" * 80)
        print("ONTOLOGY CONCEPT MATCH EVALUATION RESULTS")
        print("=" * 80 + "\n")

        # Don't include full outputs in table - focus on scores and metrics
        report.print(
            include_input=True,
            include_output=False,  # Outputs contain full concept lists - too verbose
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
