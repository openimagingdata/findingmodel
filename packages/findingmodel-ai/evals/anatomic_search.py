"""Evaluation suite for anatomic_location_search agent.

INTEGRATION SUITE: Requires real API calls and populated DuckDB anatomic database.
Run via `task evals:anatomic_search`.

DATABASE SETUP REQUIREMENTS:
This eval suite requires a populated DuckDB anatomic locations database.
To set up the database:
1. Ensure anatomic location data exists at configured URL or local path
2. Run: python -m findingmodel anatomic build
3. Verify database populated: python -m findingmodel anatomic stats

The suite uses real AI agents and DuckDB vector/hybrid search to find anatomic locations,
making it an integration test rather than a unit test.

API CALL PATTERN:
    Unlike unit tests, this eval suite makes REAL API calls by design. The
    `models.ALLOW_MODEL_REQUESTS = False` guard is intentionally omitted since
    evaluation requires actual agent execution with DuckDB and AI models.

    This is an integration eval suite, not a unit test. It validates end-to-end
    behavior including:
    - Real DuckDB index queries with embeddings
    - Actual AI model query generation and location selection
    - Complete two-agent workflow coordination
    - Backend fallback behavior (MongoDB → DuckDB)

    See: Serena memory 'agent_evaluation_best_practices_2025'

SECURITY NOTE:
    This eval suite intentionally does NOT include `models.ALLOW_MODEL_REQUESTS = False`
    because it requires real API calls to OpenAI and DuckDB. This is appropriate for
    integration eval suites (not unit tests). For unit tests with mocked responses,
    always include the ALLOW_MODEL_REQUESTS guard.

This module defines evaluation cases for assessing the anatomic_location_search functionality,
which uses two AI agents to find relevant anatomic locations for imaging findings:
- Agent 1: Query generation - generates search terms and identifies anatomic region
- Agent 2: Location selection - analyzes results and selects best locations

EVALUATOR-BASED PATTERN:
- Cases are evaluated using Dataset.evaluate() with focused evaluators
- Each evaluator checks a specific aspect (search accuracy, hierarchy, fallback, ranking)
- Hybrid scoring: strict for must-haves (0.0 or 1.0), partial credit for quality (0.0-1.0)

EVALUATORS:
- SearchAccuracyEvaluator: Checks if correct anatomic location found (partial credit)
- HierarchyEvaluator: Verifies hierarchical relationships preserved (partial credit)
- BackendFallbackEvaluator: Tests MongoDB → DuckDB fallback (strict)
- RankingQualityEvaluator: Assesses result ranking quality using MRR (partial credit)
- PerformanceEvaluator: Query performance under threshold (strict)

LOGFIRE INTEGRATION:
Logfire observability is configured automatically in evals/__init__.py.
No manual instrumentation needed in this module - automatic spans are created by:
- Dataset.evaluate() for root and per-case spans
- Pydantic AI instrumentation for agent/model/tool calls

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

import time

from findingmodel_ai.tools.anatomic_location_search import LocationSearchResponse, find_anatomic_locations
from findingmodel_ai.tools.evaluators import PerformanceEvaluator
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport


class AnatomicSearchInput(BaseModel):
    """Input for an anatomic location search evaluation case."""

    finding_name: str = Field(description="Name of the imaging finding to search")
    description: str | None = Field(default=None, description="Optional detailed description of the finding")


class AnatomicSearchExpectedOutput(BaseModel):
    """Expected output for an anatomic location search evaluation case."""

    should_succeed: bool = Field(default=True, description="Whether the search should succeed")
    expected_primary_ids: list[str] = Field(
        default_factory=list, description="Expected concept IDs that could be primary location"
    )
    expected_alternate_ids: list[str] = Field(
        default_factory=list, description="Expected concept IDs in alternate locations"
    )
    unexpected_location_ids: list[str] = Field(
        default_factory=list, description="Concept IDs that should NOT appear in results"
    )
    expected_region: str | None = Field(default=None, description="Expected anatomic region classification")
    min_alternates: int = Field(default=0, description="Minimum number of alternate locations expected", ge=0)
    max_alternates: int = Field(default=3, description="Maximum number of alternate locations allowed", ge=0, le=3)
    semantic_keywords: list[str] = Field(
        default_factory=list, description="Keywords that should appear in location names"
    )
    hierarchy_parent_id: str | None = Field(
        default=None, description="Expected parent location in hierarchy (e.g., lung for pulmonary)"
    )
    max_query_time: float = Field(default=30.0, description="Maximum acceptable query time in seconds")


class AnatomicSearchActualOutput(BaseModel):
    """Actual output from running an anatomic location search case."""

    result: LocationSearchResponse | None = Field(default=None, description="Location search result with selections")
    query_time: float = Field(description="Time taken to execute query in seconds")
    error: str | None = Field(default=None, description="Error message if query failed")


class AnatomicSearchCase(Case[AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput]):
    """A test case for anatomic_location_search functionality."""

    def __init__(
        self,
        name: str,
        finding_name: str,
        description: str | None = None,
        should_succeed: bool = True,
        expected_primary_ids: list[str] | None = None,
        expected_alternate_ids: list[str] | None = None,
        unexpected_location_ids: list[str] | None = None,
        expected_region: str | None = None,
        min_alternates: int = 0,
        max_alternates: int = 3,
        semantic_keywords: list[str] | None = None,
        hierarchy_parent_id: str | None = None,
        max_query_time: float = 30.0,
    ) -> None:
        """Initialize an anatomic location search evaluation case.

        Args:
            name: Name of the test case
            finding_name: Name of the imaging finding to search
            description: Optional detailed description
            should_succeed: Whether the search should succeed
            expected_primary_ids: Expected concept IDs for primary location
            expected_alternate_ids: Expected concept IDs in alternates
            unexpected_location_ids: Concept IDs that should NOT appear
            expected_region: Expected anatomic region classification
            min_alternates: Minimum number of alternate locations
            max_alternates: Maximum number of alternate locations
            semantic_keywords: Keywords for semantic similarity check
            hierarchy_parent_id: Expected parent in hierarchy
            max_query_time: Maximum acceptable query time in seconds
        """
        inputs = AnatomicSearchInput(
            finding_name=finding_name,
            description=description,
        )
        metadata = AnatomicSearchExpectedOutput(
            should_succeed=should_succeed,
            expected_primary_ids=expected_primary_ids or [],
            expected_alternate_ids=expected_alternate_ids or [],
            unexpected_location_ids=unexpected_location_ids or [],
            expected_region=expected_region,
            min_alternates=min_alternates,
            max_alternates=max_alternates,
            semantic_keywords=semantic_keywords or [],
            hierarchy_parent_id=hierarchy_parent_id,
            max_query_time=max_query_time,
        )
        super().__init__(name=name, inputs=inputs, metadata=metadata)


# =============================================================================
# Focused Evaluator Classes
# =============================================================================


class SearchAccuracyEvaluator(Evaluator[AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput]):
    """Evaluate that expected anatomic locations are found.

    This evaluator uses partial credit scoring to reward finding some of the
    expected locations even if not all are found. It checks both primary and
    alternate location lists.

    Scoring:
        - 1.0 if all expected locations found in correct categories
        - Partial credit (0.0-1.0) based on proportion of expected locations found
        - 0.0 if no expected locations found
        - 1.0 if no expected locations specified (N/A case)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput],
    ) -> float:
        """Evaluate search accuracy with partial credit.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Proportion of expected locations found (0.0-1.0)
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error or ctx.output.result is None:
            return 0.0

        # If no expected locations specified, this is N/A - return 1.0
        total_expected = len(ctx.metadata.expected_primary_ids) + len(ctx.metadata.expected_alternate_ids)
        if total_expected == 0:
            return 1.0

        # Collect all found location IDs from result
        found_primary_id = ctx.output.result.primary_location.concept_id
        found_alternate_ids = {loc.concept_id for loc in ctx.output.result.alternate_locations}

        # Count matches in expected categories
        primary_match_count = sum(
            1 for expected_id in ctx.metadata.expected_primary_ids if expected_id == found_primary_id
        )
        alternate_match_count = sum(
            1
            for expected_id in ctx.metadata.expected_alternate_ids
            if expected_id in found_alternate_ids or expected_id == found_primary_id
        )

        # Total matches found
        total_matches = primary_match_count + alternate_match_count

        # Return proportion of expected locations found
        return total_matches / total_expected if total_expected > 0 else 1.0


class HierarchyEvaluator(Evaluator[AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput]):
    """Evaluate that hierarchical relationships are preserved.

    Checks if selected locations maintain appropriate hierarchical relationships.
    For example, if searching for "pulmonary embolism", the location should be
    "lung" or more specific, not overly broad like "body" or unrelated like "liver".

    Scoring:
        - 1.0 if hierarchy relationships correct or not specified
        - 0.5 if hierarchy partially correct (e.g., correct parent but wrong specificity)
        - 0.0 if hierarchy completely wrong
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput],
    ) -> float:
        """Evaluate hierarchical relationship preservation.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Score from 0.0-1.0 based on hierarchy correctness
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Skip if no hierarchy parent specified - N/A case, return 1.0
        if not ctx.metadata.hierarchy_parent_id:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error or ctx.output.result is None:
            return 0.0

        # Check primary location for hierarchy relationship
        primary_text = ctx.output.result.primary_location.concept_text.lower()

        # Extract expected parent keywords from ID and metadata
        expected_parent_keywords = []
        if ctx.metadata.hierarchy_parent_id:
            expected_parent_keywords.append(ctx.metadata.hierarchy_parent_id.lower())
        if ctx.metadata.semantic_keywords:
            expected_parent_keywords.extend(kw.lower() for kw in ctx.metadata.semantic_keywords[:2])

        # Check if primary location contains expected hierarchy keywords
        hierarchy_matches = sum(1 for keyword in expected_parent_keywords if keyword in primary_text)

        if hierarchy_matches == 0:
            # No hierarchy keywords found - likely wrong location
            return 0.0
        elif hierarchy_matches >= len(expected_parent_keywords) // 2:
            # Most hierarchy keywords found - good hierarchy
            return 1.0
        else:
            # Some hierarchy keywords found - partial credit
            return 0.5


class BackendFallbackEvaluator(
    Evaluator[AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput]
):
    """Evaluate backend fallback behavior (MongoDB → DuckDB).

    This evaluator is currently a placeholder since the tool only uses DuckDB backend.
    For now, it verifies that the search completes successfully with DuckDB.

    TODO: Implement full MongoDB fallback testing once MongoDB backend is added
    Currently returns 1.0 for all non-error cases as placeholder

    Future implementation will test:
    - MongoDB availability detection
    - Graceful fallback to DuckDB when MongoDB unavailable
    - Consistent results across backends

    Returns:
        1.0 if search succeeds (correct backend usage)
        0.0 if search fails with backend error
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput],
    ) -> float:
        """Evaluate backend fallback behavior.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if backend handling is correct, 0.0 otherwise
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Check if search succeeded without backend errors
        if ctx.output.error:
            # Check if error is backend-related
            error_text = ctx.output.error.lower()
            if "database" in error_text or "connection" in error_text or "duckdb" in error_text:
                return 0.0  # Backend error
            # Other errors not related to backend - return 1.0 (N/A for backend eval)
            return 1.0

        # Search succeeded - backend handling correct
        return 1.0


class RankingQualityEvaluator(Evaluator[AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput]):
    """Evaluate ranking quality using Mean Reciprocal Rank (MRR).

    MRR measures how highly the expected relevant results are ranked.
    For anatomic locations, we check if expected locations appear as primary
    or in alternates (with primary ranked higher).

    MRR Formula: 1 / rank_of_first_relevant_result

    Returns:
        1.0 if expected location is primary
        0.5 if expected location is first alternate
        0.33 if expected location is second alternate
        0.25 if expected location is third alternate
        0.0 if no expected locations found
        1.0 if no expected locations specified (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput],
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
        all_expected = ctx.metadata.expected_primary_ids + ctx.metadata.expected_alternate_ids
        if not all_expected:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error or ctx.output.result is None:
            return 0.0

        # Build ranked list: [primary, alternate1, alternate2, alternate3]
        ranked_ids = [ctx.output.result.primary_location.concept_id]
        ranked_ids.extend(loc.concept_id for loc in ctx.output.result.alternate_locations)

        # Create set of expected IDs
        expected_ids_set = set(all_expected)

        # Find rank of first relevant result (1-indexed)
        for rank, location_id in enumerate(ranked_ids, start=1):
            if location_id in expected_ids_set:
                return 1.0 / rank

        # No relevant results found
        return 0.0


# =============================================================================
# Test Case Creation Functions
# =============================================================================


def create_success_common_terms() -> list[AnatomicSearchCase]:
    """Create success cases for common anatomic terms."""
    cases = []

    # Case 1: Heart - common cardiovascular term
    cases.append(
        AnatomicSearchCase(
            name="success_common_heart",
            finding_name="heart",
            description="Central organ of cardiovascular system",
            expected_primary_ids=["RID1385"],  # heart in RadLex
            expected_region="Thorax",
            min_alternates=1,
            semantic_keywords=["heart", "cardiac"],
            max_query_time=20.0,
        )
    )

    # Case 2: Lung - common thoracic term
    cases.append(
        AnatomicSearchCase(
            name="success_common_lung",
            finding_name="lung",
            description="Respiratory organ",
            expected_primary_ids=["RID1301"],  # lung in RadLex
            expected_region="Thorax",
            min_alternates=1,
            semantic_keywords=["lung", "pulmonary"],
            hierarchy_parent_id="RID1301",
            max_query_time=20.0,
        )
    )

    # Case 3: Liver - common abdominal term
    cases.append(
        AnatomicSearchCase(
            name="success_common_liver",
            finding_name="liver",
            description="Largest abdominal organ",
            expected_primary_ids=["RID58"],  # liver in RadLex
            expected_region="Abdomen",
            min_alternates=1,
            semantic_keywords=["liver", "hepatic"],
            max_query_time=20.0,
        )
    )

    # Case 4: Brain - common neurological term
    cases.append(
        AnatomicSearchCase(
            name="success_common_brain",
            finding_name="brain",
            description="Central nervous system organ",
            expected_primary_ids=["RID6434"],  # brain in RadLex
            expected_region="Head",
            min_alternates=1,
            semantic_keywords=["brain", "cerebral"],
            max_query_time=20.0,
        )
    )

    # Case 5: Kidney - paired abdominal organ
    cases.append(
        AnatomicSearchCase(
            name="success_common_kidney",
            finding_name="kidney",
            description="Paired retroperitoneal organ",
            expected_primary_ids=["RID205", "RID30325"],  # kidney structures
            expected_region="Abdomen",
            min_alternates=1,
            semantic_keywords=["kidney", "renal"],
            max_query_time=20.0,
        )
    )

    return cases


def create_success_specific_locations() -> list[AnatomicSearchCase]:
    """Create success cases for specific anatomic locations."""
    cases = []

    # Case 6: Left ventricle - specific cardiac structure
    cases.append(
        AnatomicSearchCase(
            name="success_specific_left_ventricle",
            finding_name="left ventricle",
            description="Left lower chamber of heart",
            expected_primary_ids=["RID1390"],  # left ventricle
            expected_region="Thorax",
            semantic_keywords=["ventricle", "left", "heart"],
            hierarchy_parent_id="RID1385",  # heart
            max_query_time=20.0,
        )
    )

    # Case 7: Right upper lobe - specific lung segment
    cases.append(
        AnatomicSearchCase(
            name="success_specific_right_upper_lobe",
            finding_name="right upper lobe",
            description="Superior lobe of right lung",
            expected_primary_ids=["RID1327"],  # right upper lobe
            expected_region="Thorax",
            semantic_keywords=["lobe", "upper", "right"],
            hierarchy_parent_id="RID1301",  # lung
            max_query_time=20.0,
        )
    )

    # Case 8: Ascending aorta - specific vascular structure
    cases.append(
        AnatomicSearchCase(
            name="success_specific_ascending_aorta",
            finding_name="ascending aorta",
            description="Initial segment of aorta from heart",
            expected_primary_ids=["RID480"],  # ascending aorta
            expected_region="Thorax",
            semantic_keywords=["aorta", "ascending"],
            max_query_time=20.0,
        )
    )

    return cases


def create_success_hierarchical() -> list[AnatomicSearchCase]:
    """Create success cases testing hierarchical relationships."""
    cases = []

    # Case 9: Mitral valve → Heart hierarchy
    cases.append(
        AnatomicSearchCase(
            name="success_hierarchy_mitral_valve",
            finding_name="mitral valve",
            description="Valve between left atrium and left ventricle",
            expected_primary_ids=["RID1391"],  # mitral valve
            expected_region="Thorax",
            semantic_keywords=["mitral", "valve", "heart"],
            hierarchy_parent_id="RID1385",  # heart
            max_query_time=20.0,
        )
    )

    # Case 10: Pulmonary artery → Thorax hierarchy
    cases.append(
        AnatomicSearchCase(
            name="success_hierarchy_pulmonary_artery",
            finding_name="pulmonary artery",
            description="Artery carrying blood from heart to lungs",
            expected_primary_ids=["RID1398"],  # pulmonary artery
            expected_region="Thorax",
            semantic_keywords=["pulmonary", "artery"],
            hierarchy_parent_id="RID1301",  # lung area
            max_query_time=20.0,
        )
    )

    return cases


def create_success_synonyms() -> list[AnatomicSearchCase]:
    """Create success cases testing synonym handling."""
    cases = []

    # Case 11: PA vs pulmonary artery
    cases.append(
        AnatomicSearchCase(
            name="success_synonym_pa",
            finding_name="PA",
            description="Pulmonary artery abbreviation",
            expected_primary_ids=["RID1398"],  # pulmonary artery
            expected_region="Thorax",
            semantic_keywords=["pulmonary", "artery"],
            max_query_time=20.0,
        )
    )

    # Case 12: RV vs right ventricle
    cases.append(
        AnatomicSearchCase(
            name="success_synonym_rv",
            finding_name="RV",
            description="Right ventricle abbreviation",
            expected_primary_ids=["RID1389"],  # right ventricle
            expected_region="Thorax",
            semantic_keywords=["ventricle", "right"],
            max_query_time=20.0,
        )
    )

    # Case 13: CNS vs central nervous system
    cases.append(
        AnatomicSearchCase(
            name="success_synonym_cns",
            finding_name="CNS",
            description="Central nervous system",
            expected_region="Head",
            semantic_keywords=["nervous", "central", "brain"],
            max_query_time=20.0,
        )
    )

    return cases


def create_success_systems() -> list[AnatomicSearchCase]:
    """Create success cases for anatomic systems."""
    cases = []

    # Case 14: Cardiovascular system
    cases.append(
        AnatomicSearchCase(
            name="success_system_cardiovascular",
            finding_name="cardiovascular system",
            description="Heart and blood vessels",
            expected_region="Thorax",
            semantic_keywords=["heart", "cardiovascular", "vascular"],
            max_query_time=20.0,
        )
    )

    # Case 15: Respiratory system
    cases.append(
        AnatomicSearchCase(
            name="success_system_respiratory",
            finding_name="respiratory system",
            description="Lungs and airways",
            expected_region="Thorax",
            semantic_keywords=["lung", "respiratory", "airway"],
            max_query_time=20.0,
        )
    )

    return cases


def create_rejection_non_anatomic() -> list[AnatomicSearchCase]:
    """Create rejection cases for non-anatomic terms."""
    cases = []

    # Case 16: Disease term - diabetes
    cases.append(
        AnatomicSearchCase(
            name="rejection_non_anatomic_diabetes",
            finding_name="diabetes",
            description="Metabolic disease, not anatomic location",
            should_succeed=True,  # Search succeeds but returns generic/no results
            expected_primary_ids=[],
            unexpected_location_ids=["RID58", "RID1385", "RID1301"],  # Should not match specific organs
            max_query_time=20.0,
        )
    )

    # Case 17: Protocol term
    cases.append(
        AnatomicSearchCase(
            name="rejection_non_anatomic_protocol",
            finding_name="CT protocol",
            description="Imaging protocol, not anatomic structure",
            should_succeed=True,  # Search succeeds but returns generic/no results
            expected_primary_ids=[],
            max_query_time=20.0,
        )
    )

    # Case 18: Procedure term
    cases.append(
        AnatomicSearchCase(
            name="rejection_non_anatomic_surgery",
            finding_name="surgery",
            description="Medical procedure, not anatomic location",
            should_succeed=True,  # Search succeeds but returns generic/no results
            expected_primary_ids=[],
            max_query_time=20.0,
        )
    )

    return cases


def create_edge_cases() -> list[AnatomicSearchCase]:
    """Create edge cases and boundary conditions."""
    cases = []

    # Case 19: Ambiguous term - trunk
    cases.append(
        AnatomicSearchCase(
            name="edge_ambiguous_trunk",
            finding_name="trunk",
            description="Could be torso or vessel trunk",
            expected_region="Body",
            semantic_keywords=["trunk", "body"],
            max_query_time=20.0,
        )
    )

    # Case 20: Very long compound name
    cases.append(
        AnatomicSearchCase(
            name="edge_long_name",
            finding_name="left posterior descending branch of left coronary artery",
            description="Very specific and long anatomic name",
            expected_region="Thorax",
            semantic_keywords=["coronary", "artery", "left"],
            hierarchy_parent_id="RID1385",  # heart
            max_query_time=25.0,
        )
    )

    # Case 21: Case sensitivity test
    cases.append(
        AnatomicSearchCase(
            name="edge_case_sensitivity",
            finding_name="HEART",
            description="All caps anatomic term",
            expected_primary_ids=["RID1385"],
            expected_region="Thorax",
            semantic_keywords=["heart"],
            max_query_time=20.0,
        )
    )

    # Case 22: Special characters
    cases.append(
        AnatomicSearchCase(
            name="edge_special_chars",
            finding_name="C7-T1",
            description="Spine level with special characters",
            expected_region="Neck",
            semantic_keywords=["spine", "cervical", "thoracic"],
            max_query_time=20.0,
        )
    )

    # Case 23: Misspelling
    cases.append(
        AnatomicSearchCase(
            name="edge_misspelling",
            finding_name="luang",  # Misspelling of "lung"
            description="Common misspelling",
            should_succeed=True,  # May or may not find correct location
            semantic_keywords=["lung", "pulmonary"],  # Should still try to match
            max_query_time=20.0,
        )
    )

    # Case 24: Empty description
    cases.append(
        AnatomicSearchCase(
            name="edge_empty_description",
            finding_name="spleen",
            description="",  # Empty description
            expected_primary_ids=["RID86"],  # spleen
            expected_region="Abdomen",
            semantic_keywords=["spleen"],
            max_query_time=20.0,
        )
    )

    # Case 25: Multiple interpretations
    cases.append(
        AnatomicSearchCase(
            name="edge_multiple_interpretations",
            finding_name="mass",
            description="Could refer to anatomic mass or pathologic mass",
            should_succeed=True,
            max_query_time=20.0,
        )
    )

    return cases


def create_performance_cases() -> list[AnatomicSearchCase]:
    """Create cases focused on performance testing."""
    cases = []

    # Case 26: Simple query should be fast
    cases.append(
        AnatomicSearchCase(
            name="performance_simple_query",
            finding_name="heart",
            expected_primary_ids=["RID1385"],
            max_query_time=15.0,  # Strict time limit for simple query
        )
    )

    # Case 27: Complex query with full description
    cases.append(
        AnatomicSearchCase(
            name="performance_complex_query",
            finding_name="anterior interventricular septum",
            description="The anterior portion of the wall between the left and right ventricles",
            expected_region="Thorax",
            semantic_keywords=["septum", "interventricular", "heart"],
            max_query_time=25.0,
        )
    )

    return cases


# =============================================================================
# Task Execution Function
# =============================================================================


def verify_database_populated() -> None:
    """Verify that the anatomic database is populated with expected test data.

    Checks for a few key RadLex IDs used in test cases. If the database is missing
    or not populated, raises a clear error directing the user to run database setup.

    Raises:
        RuntimeError: If database is not found or not populated with required test data
    """
    from anatomic_locations.config import ensure_anatomic_db
    from oidm_common.duckdb import setup_duckdb_connection

    # Get database path
    try:
        db_path = ensure_anatomic_db()
    except Exception as e:
        raise RuntimeError(
            f"Failed to locate anatomic database: {e}\nPlease run: python -m findingmodel anatomic build"
        ) from e

    if not db_path.exists():
        raise RuntimeError(
            f"Anatomic database not found at {db_path}\nPlease run: python -m findingmodel anatomic build"
        )

    # Check that database has records with key test IDs
    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        # Check total record count
        result = conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()
        total_records = result[0] if result else 0

        if total_records == 0:
            raise RuntimeError(
                f"Anatomic database at {db_path} is empty\nPlease run: python -m findingmodel anatomic build"
            )

        # Check for a few key RadLex IDs used in test cases
        test_ids = ["RID1385", "RID1301", "RID58"]  # heart, lung, liver
        for test_id in test_ids:
            result = conn.execute("SELECT COUNT(*) FROM anatomic_locations WHERE id = ?", [test_id]).fetchone()
            count = result[0] if result else 0
            if count == 0:
                raise RuntimeError(
                    f"Anatomic database at {db_path} is missing expected test data (e.g., {test_id})\n"
                    "The database may be incomplete or corrupted.\n"
                    "Please run: python -m findingmodel anatomic build --force"
                )

    finally:
        conn.close()


async def run_anatomic_search_task(input_data: AnatomicSearchInput) -> AnatomicSearchActualOutput:
    """Execute a single anatomic_location_search evaluation case.

    Dataset.evaluate() automatically creates spans and captures inputs/outputs.
    Pydantic AI instrumentation captures agent/model/tool calls.
    No manual Logfire code needed.

    Args:
        input_data: Input data for the anatomic search case

    Returns:
        Actual output from the anatomic search execution

    Raises:
        Exception: Any errors are caught and returned in the error field
    """
    try:
        # Time the query
        start_time = time.time()
        result = await find_anatomic_locations(
            finding_name=input_data.finding_name,
            description=input_data.description,
            use_duckdb=True,  # Always use DuckDB backend
        )
        query_time = time.time() - start_time

        return AnatomicSearchActualOutput(
            result=result,
            query_time=query_time,
        )
    except Exception as e:
        # Return error in output for evaluation
        query_time = time.time() - start_time if "start_time" in locals() else 0.0
        return AnatomicSearchActualOutput(
            result=None,
            query_time=query_time,
            error=str(e),
        )


# =============================================================================
# Dataset Creation with Evaluator-Based Pattern
# =============================================================================

all_cases = (
    create_success_common_terms()
    + create_success_specific_locations()
    + create_success_hierarchical()
    + create_success_synonyms()
    + create_success_systems()
    + create_rejection_non_anatomic()
    + create_edge_cases()
    + create_performance_cases()
)

evaluators = [
    SearchAccuracyEvaluator(),
    HierarchyEvaluator(),
    BackendFallbackEvaluator(),
    RankingQualityEvaluator(),
    PerformanceEvaluator(),
]

anatomic_search_dataset = Dataset(cases=all_cases, evaluators=evaluators)


async def run_anatomic_search_evals() -> EvaluationReport[
    AnatomicSearchInput, AnatomicSearchActualOutput, AnatomicSearchExpectedOutput
]:
    """Run anatomic_location_search evaluation suite.

    Dataset.evaluate() automatically creates evaluation spans and captures
    all inputs, outputs, and scores for visualization in Logfire.

    Raises:
        RuntimeError: If database is not populated with required test data
    """
    # Verify database is populated before running evals
    verify_database_populated()

    report = await anatomic_search_dataset.evaluate(run_anatomic_search_task)
    return report


if __name__ == "__main__":
    import asyncio

    from evals import ensure_instrumented

    ensure_instrumented()  # Explicit instrumentation for eval run

    async def main() -> None:
        print("\nRunning anatomic_location_search evaluation suite...")
        print("=" * 80)
        print("NOTE: This eval requires DuckDB anatomic database and OpenAI API key.")
        print("Ensure database is built: python -m findingmodel anatomic build")
        print("=" * 80 + "\n")

        report = await run_anatomic_search_evals()

        print("\n" + "=" * 80)
        print("ANATOMIC LOCATION SEARCH EVALUATION RESULTS")
        print("=" * 80 + "\n")

        # Don't include full outputs in table - focus on scores and metrics
        report.print(
            include_input=True,
            include_output=False,  # Outputs contain full location objects - too verbose
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
