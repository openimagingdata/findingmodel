"""Evaluation suite for finding description generation agent.

INTEGRATION SUITE: Requires real API calls to OpenAI and Tavily.
Run via `task evals:finding_description`.

API CALL PATTERN:
    Unlike unit tests, this eval suite makes REAL API calls by design. The
    `models.ALLOW_MODEL_REQUESTS = False` guard is intentionally omitted since
    evaluation requires actual agent execution with real AI models.

    This is an integration eval suite, not a unit test. It validates end-to-end
    behavior including:
    - Real OpenAI API calls for finding description generation
    - Real Tavily API calls for detailed information retrieval
    - Actual medical terminology usage and clinical accuracy
    - Complete FindingInfo construction from finding names

    See: Serena memory 'agent_evaluation_best_practices_2025'

This module defines evaluation cases for assessing the finding description generation
functionality, which uses AI agents to create clinical descriptions for finding models.

TOOLS UNDER TEST:
- create_info_from_name: Creates FindingInfo from a finding name using OpenAI
- add_details_to_info: Adds detailed description and citations using Tavily

EVALUATOR-BASED PATTERN:
- Cases are evaluated using Dataset.evaluate() with focused evaluators
- Each evaluator checks a specific aspect (length, terminology, quality, etc.)
- Hybrid scoring: strict for must-haves (0.0 or 1.0), partial credit for quality (0.0-1.0)

EVALUATORS:
- LengthAppropriatenessEvaluator: Checks description length (not too short/long)
- TerminologyEvaluator: Verifies proper medical terminology usage
- SynonymQualityEvaluator: Evaluates quality of generated synonyms
- CanonicalNameEvaluator: Checks canonical name correctness
- CitationQualityEvaluator: Verifies citation presence and format
- ReadabilityEvaluator: Checks readability for clinical professionals
- ConsistencyEvaluator: Compares similar findings for consistency
- LLMJudge: Uses OpenAI to assess clinical accuracy and quality (0.0-1.0 score)
- PerformanceEvaluator: Validates execution time

LOGFIRE INTEGRATION:
Logfire observability is configured automatically in evals/__init__.py.
No manual instrumentation needed in this module - automatic spans are created by:
- Dataset.evaluate() for root and per-case spans
- Pydantic AI instrumentation for agent/model/tool calls

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

import asyncio
import os
import re
import time

from findingmodel.config import settings
from findingmodel.finding_info import FindingInfo
from findingmodel_ai.authoring.description import add_details_to_info, create_info_from_name
from findingmodel_ai.evaluators import PerformanceEvaluator
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge
from pydantic_evals.reporting import EvaluationReport

# WORKAROUND: LLMJudge has a bug - it doesn't accept api_key parameter
# and only reads from OPENAI_API_KEY environment variable.
# For eval files using LLMJudge, we set the environment variable from settings.
if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()


class FindingDescriptionInput(BaseModel):
    """Input for a finding description evaluation case."""

    finding_name: str = Field(description="Name of the finding to describe")
    fetch_details: bool = Field(default=False, description="Whether to fetch detailed information via search API")


class FindingDescriptionExpectedOutput(BaseModel):
    """Expected output for a finding description evaluation case."""

    expected_canonical_name: str | None = Field(default=None, description="Expected canonical name after normalization")
    expected_synonyms: list[str] = Field(default_factory=list, description="Expected synonyms to be included")
    unexpected_synonyms: list[str] = Field(
        default_factory=list, description="Synonyms that should NOT be included (incorrect)"
    )
    expected_keywords: list[str] = Field(
        default_factory=list, description="Keywords that should appear in description/detail"
    )
    unexpected_keywords: list[str] = Field(
        default_factory=list, description="Keywords that should NOT appear (incorrect medical terms)"
    )
    min_description_length: int = Field(default=20, description="Minimum acceptable description length", ge=0)
    max_description_length: int = Field(
        default=500, description="Maximum acceptable description length (concise)", ge=0
    )
    min_detail_length: int = Field(default=100, description="Minimum acceptable detail length when using Tavily", ge=0)
    requires_citations: bool = Field(default=False, description="Whether citations are required (Tavily only)")
    medical_terminology_level: str = Field(
        default="professional", description="Expected terminology level: 'professional' or 'layperson'"
    )
    consistency_group: str | None = Field(
        default=None, description="Group name for consistency checking across similar findings"
    )
    reference_description: str | None = Field(default=None, description="Reference description for quality comparison")
    max_query_time: float = Field(default=30.0, description="Maximum acceptable query time in seconds")


class FindingDescriptionActualOutput(BaseModel):
    """Actual output from running a finding description case."""

    finding_info: FindingInfo | None = Field(default=None, description="Generated finding information")
    query_time: float = Field(description="Time taken to execute query in seconds")
    error: str | None = Field(default=None, description="Error message if generation failed")


class FindingDescriptionCase(
    Case[FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput]
):
    """A test case for finding description generation functionality."""

    def __init__(
        self,
        name: str,
        finding_name: str,
        fetch_details: bool = False,
        expected_canonical_name: str | None = None,
        expected_synonyms: list[str] | None = None,
        unexpected_synonyms: list[str] | None = None,
        expected_keywords: list[str] | None = None,
        unexpected_keywords: list[str] | None = None,
        min_description_length: int = 20,
        max_description_length: int = 500,
        min_detail_length: int = 100,
        requires_citations: bool = False,
        medical_terminology_level: str = "professional",
        consistency_group: str | None = None,
        reference_description: str | None = None,
        max_query_time: float = 30.0,
    ) -> None:
        """Initialize a finding description evaluation case.

        Args:
            name: Name of the test case
            finding_name: Name of the finding to describe
            fetch_details: Whether to fetch detailed information via search API
            expected_canonical_name: Expected canonical name after normalization
            expected_synonyms: Expected synonyms to be included
            unexpected_synonyms: Synonyms that should NOT be included
            expected_keywords: Keywords that should appear in description/detail
            unexpected_keywords: Keywords that should NOT appear
            min_description_length: Minimum acceptable description length
            max_description_length: Maximum acceptable description length
            min_detail_length: Minimum acceptable detail length when fetching details
            requires_citations: Whether citations are required
            medical_terminology_level: Expected terminology level
            consistency_group: Group name for consistency checking
            reference_description: Reference description for quality comparison
            max_query_time: Maximum acceptable query time in seconds
        """
        inputs = FindingDescriptionInput(
            finding_name=finding_name,
            fetch_details=fetch_details,
        )
        metadata = FindingDescriptionExpectedOutput(
            expected_canonical_name=expected_canonical_name,
            expected_synonyms=expected_synonyms or [],
            unexpected_synonyms=unexpected_synonyms or [],
            expected_keywords=expected_keywords or [],
            unexpected_keywords=unexpected_keywords or [],
            min_description_length=min_description_length,
            max_description_length=max_description_length,
            min_detail_length=min_detail_length,
            requires_citations=requires_citations,
            medical_terminology_level=medical_terminology_level,
            consistency_group=consistency_group,
            reference_description=reference_description,
            max_query_time=max_query_time,
        )
        super().__init__(name=name, inputs=inputs, metadata=metadata)


# =============================================================================
# Focused Evaluator Classes
# =============================================================================


class LengthAppropriatenessEvaluator(
    Evaluator[FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput]
):
    """Evaluate description length appropriateness.

    Uses strict scoring for must-have length constraints (description must be
    present and within bounds). Clinical descriptions should be concise but
    informative - too short lacks detail, too long loses focus.

    Returns:
        1.0 if description length is appropriate
        0.0 if description is missing, too short, or too long
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput
        ],
    ) -> float:
        """Evaluate length appropriateness.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if length appropriate, 0.0 otherwise
        """
        if ctx.metadata is None or ctx.output.error or ctx.output.finding_info is None:
            return 0.0

        # Check description length
        description = ctx.output.finding_info.description or ""
        desc_len = len(description)

        if desc_len < ctx.metadata.min_description_length or desc_len > ctx.metadata.max_description_length:
            return 0.0

        # If fetching details, also check detail length
        if ctx.inputs.fetch_details:
            detail = ctx.output.finding_info.detail or ""
            detail_len = len(detail)

            if detail_len < ctx.metadata.min_detail_length:
                return 0.0

        return 1.0


class TerminologyEvaluator(
    Evaluator[FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput]
):
    """Verify proper medical terminology usage.

    Checks for expected medical keywords and absence of unexpected terms.
    Uses partial credit based on keyword matches.

    Returns:
        0.0-1.0 based on proportion of expected keywords found and unexpected keywords avoided
        1.0 if no keyword constraints specified (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput
        ],
    ) -> float:
        """Evaluate medical terminology usage.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Proportion of correct terminology usage (0.0-1.0)
        """
        if ctx.metadata is None or ctx.output.error or ctx.output.finding_info is None:
            return 0.0

        # Combine all text for searching
        description = (ctx.output.finding_info.description or "").lower()
        detail = (ctx.output.finding_info.detail or "").lower()
        combined_text = f"{description} {detail}"

        # Track scores
        scores = []

        # Check expected keywords (if any)
        if ctx.metadata.expected_keywords:
            expected_matches = sum(1 for keyword in ctx.metadata.expected_keywords if keyword.lower() in combined_text)
            scores.append(expected_matches / len(ctx.metadata.expected_keywords))

        # Check unexpected keywords (if any) - these should NOT be present
        if ctx.metadata.unexpected_keywords:
            unexpected_found = sum(
                1 for keyword in ctx.metadata.unexpected_keywords if keyword.lower() in combined_text
            )
            # Invert: 1.0 if none found, 0.0 if all found
            scores.append(1.0 - (unexpected_found / len(ctx.metadata.unexpected_keywords)))

        # Return average of scores, or 1.0 if no constraints
        return sum(scores) / len(scores) if scores else 1.0


class ConsistencyEvaluator(
    Evaluator[FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput]
):
    """Evaluate consistency across similar findings.

    Stores outputs by consistency_group and compares them to ensure similar
    findings receive similar treatment (terminology, structure, detail level).

    Returns:
        0.0-1.0 based on consistency with other findings in same group
        1.0 if no consistency group specified (N/A)
    """

    def __init__(self) -> None:
        """Initialize consistency evaluator with empty group storage."""
        self.groups: dict[str, list[FindingInfo]] = {}

    def evaluate(
        self,
        ctx: EvaluatorContext[
            FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput
        ],
    ) -> float:
        """Evaluate consistency with similar findings.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Consistency score (0.0-1.0) based on similarity to group
        """
        if ctx.metadata is None or ctx.output.error or ctx.output.finding_info is None:
            return 1.0

        # Skip if no consistency group specified
        if not ctx.metadata.consistency_group:
            return 1.0

        group = ctx.metadata.consistency_group

        # Initialize group if first entry
        if group not in self.groups:
            self.groups[group] = []

        current_info = ctx.output.finding_info

        # If first in group, just add and return 1.0
        if not self.groups[group]:
            self.groups[group].append(current_info)
            return 1.0

        # Compare with existing entries in group
        consistency_scores = []

        for existing_info in self.groups[group]:
            score = 0.0

            # Check description length similarity (within 50% range)
            desc_len = len(current_info.description or "")
            existing_len = len(existing_info.description or "")
            if existing_len > 0:
                length_ratio = min(desc_len, existing_len) / max(desc_len, existing_len)
                score += length_ratio * 0.5

            # Check for common terminology (shared words)
            current_words = set((current_info.description or "").lower().split())
            existing_words = set((existing_info.description or "").lower().split())
            if current_words and existing_words:
                overlap = len(current_words & existing_words) / len(current_words | existing_words)
                score += overlap * 0.5

            consistency_scores.append(score)

        # Add to group for future comparisons
        self.groups[group].append(current_info)

        # Return average consistency with group
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0


class ReadabilityEvaluator(
    Evaluator[FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput]
):
    """Check readability for clinical professionals.

    Evaluates readability appropriate for medical professionals using simple
    heuristics: sentence count, average sentence length, and avoiding overly
    complex sentences.

    Returns:
        0.0-1.0 score based on readability metrics
        1.0 if no text to evaluate (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput
        ],
    ) -> float:
        """Evaluate readability.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Readability score (0.0-1.0)
        """
        if ctx.metadata is None or ctx.output.error or ctx.output.finding_info is None:
            return 0.0

        description = ctx.output.finding_info.description or ""
        if not description:
            return 0.0

        # Split into sentences
        sentences = [s.strip() for s in re.split(r"[.!?]+", description) if s.strip()]
        if not sentences:
            return 0.0

        # Calculate metrics
        sentence_count = len(sentences)
        word_count = len(description.split())
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        score = 1.0

        # Professional medical descriptions should be 1-3 sentences
        if sentence_count < 1 or sentence_count > 3:
            score -= 0.3

        # Average sentence should be 10-30 words
        if avg_sentence_length < 10 or avg_sentence_length > 30:
            score -= 0.3

        # Check for overly long sentences (>50 words)
        for sentence in sentences:
            if len(sentence.split()) > 50:
                score -= 0.2
                break

        # Ensure score stays in bounds
        return max(0.0, min(1.0, score))


class SynonymQualityEvaluator(
    Evaluator[FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput]
):
    """Evaluate synonym quality and correctness.

    Checks that expected synonyms are included and unexpected (incorrect)
    synonyms are excluded. Uses partial credit.

    Returns:
        0.0-1.0 based on synonym accuracy
        1.0 if no synonym constraints specified (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput
        ],
    ) -> float:
        """Evaluate synonym quality.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Synonym quality score (0.0-1.0)
        """
        if ctx.metadata is None or ctx.output.error or ctx.output.finding_info is None:
            return 0.0

        synonyms = ctx.output.finding_info.synonyms or []
        synonyms_lower = [s.lower() for s in synonyms]

        scores = []

        # Check expected synonyms (if any)
        if ctx.metadata.expected_synonyms:
            expected_found = sum(1 for syn in ctx.metadata.expected_synonyms if syn.lower() in synonyms_lower)
            scores.append(expected_found / len(ctx.metadata.expected_synonyms))

        # Check unexpected synonyms (if any) - these should NOT be present
        if ctx.metadata.unexpected_synonyms:
            unexpected_found = sum(1 for syn in ctx.metadata.unexpected_synonyms if syn.lower() in synonyms_lower)
            # Invert: 1.0 if none found, 0.0 if all found
            scores.append(1.0 - (unexpected_found / len(ctx.metadata.unexpected_synonyms)))

        # Return average of scores, or 1.0 if no constraints
        return sum(scores) / len(scores) if scores else 1.0


class CanonicalNameEvaluator(
    Evaluator[FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput]
):
    """Evaluate canonical name normalization.

    Checks that the canonical name matches expected normalization rules:
    - Acronyms expanded
    - Overly specific terms generalized
    - Singular form used
    - Proper capitalization

    Returns:
        1.0 if canonical name matches expectation (or N/A)
        0.0 if canonical name does not match expectation
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput
        ],
    ) -> float:
        """Evaluate canonical name quality.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if canonical name correct, 0.0 otherwise
        """
        if ctx.metadata is None or ctx.output.error or ctx.output.finding_info is None:
            return 0.0

        # Skip if no expected canonical name specified
        if not ctx.metadata.expected_canonical_name:
            return 1.0

        actual_name = ctx.output.finding_info.name.lower()
        expected_name = ctx.metadata.expected_canonical_name.lower()

        return 1.0 if actual_name == expected_name else 0.0


class CitationQualityEvaluator(
    Evaluator[FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput]
):
    """Evaluate citation quality for Tavily-enhanced descriptions.

    Checks that citations are present when required and are valid URLs.

    Returns:
        1.0 if citations meet requirements
        0.0 if citations missing or invalid when required
        1.0 if citations not required (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput
        ],
    ) -> float:
        """Evaluate citation quality.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if citations meet requirements, 0.0 otherwise
        """
        if ctx.metadata is None or ctx.output.error or ctx.output.finding_info is None:
            return 0.0

        # Skip if citations not required
        if not ctx.metadata.requires_citations:
            return 1.0

        citations = ctx.output.finding_info.citations or []

        # Must have at least one citation when required
        if not citations:
            return 0.0

        # Check that citations are valid URLs
        valid_count = sum(1 for citation in citations if citation.startswith(("http://", "https://")))

        # Return proportion of valid URLs
        return valid_count / len(citations) if citations else 0.0


# =============================================================================
# Test Case Creation Functions
# =============================================================================


def create_common_findings_cases() -> list[FindingDescriptionCase]:
    """Create cases for common medical findings with clear descriptions."""
    cases = []

    # Case 1: Common finding with well-known description
    cases.append(
        FindingDescriptionCase(
            name="common_pneumothorax",
            finding_name="pneumothorax",
            expected_canonical_name="pneumothorax",
            expected_synonyms=["ptx", "collapsed lung"],
            expected_keywords=["air", "pleural", "space", "lung"],
            min_description_length=30,
            max_description_length=200,
            medical_terminology_level="professional",
            consistency_group="pulmonary",
            reference_description="Presence of air in the pleural space between the lung and chest wall, causing lung collapse.",
            max_query_time=20.0,
        )
    )

    # Case 2: Finding with acronym expansion
    cases.append(
        FindingDescriptionCase(
            name="common_pe_acronym",
            finding_name="PE",
            expected_canonical_name="pulmonary embolism",
            expected_synonyms=["pe"],
            expected_keywords=["pulmonary", "embolism", "blood", "clot"],
            min_description_length=30,
            max_description_length=200,
            consistency_group="vascular",
            reference_description="Blockage of pulmonary artery by a blood clot, typically originating from deep vein thrombosis.",
            max_query_time=20.0,
        )
    )

    # Case 3: Common fracture finding
    cases.append(
        FindingDescriptionCase(
            name="common_fracture",
            finding_name="fracture",
            expected_canonical_name="fracture",
            expected_keywords=["bone", "break", "discontinuity"],
            min_description_length=20,
            max_description_length=150,
            medical_terminology_level="professional",
            reference_description="Break in bone continuity, which may be complete or incomplete.",
            max_query_time=20.0,
        )
    )

    # Case 4: Specific anatomic location finding
    cases.append(
        FindingDescriptionCase(
            name="common_kidney_stone",
            finding_name="kidney stone",
            expected_canonical_name="kidney stone",
            expected_synonyms=["renal calculus", "nephrolithiasis"],
            expected_keywords=["kidney", "stone", "calculus"],
            min_description_length=30,
            max_description_length=200,
            consistency_group="genitourinary",
            reference_description="Crystalline mineral deposit in the kidney that may cause obstruction and pain.",
            max_query_time=20.0,
        )
    )

    return cases


def create_complex_findings_cases() -> list[FindingDescriptionCase]:
    """Create cases for complex findings requiring detailed descriptions."""
    cases = []

    # Case 5: Complex pathology with multiple aspects
    cases.append(
        FindingDescriptionCase(
            name="complex_aortic_dissection",
            finding_name="aortic dissection",
            expected_canonical_name="aortic dissection",
            expected_keywords=["aorta", "tear", "intimal", "blood"],
            unexpected_keywords=["infection", "tumor"],  # Wrong pathology
            min_description_length=40,
            max_description_length=300,
            medical_terminology_level="professional",
            consistency_group="vascular",
            reference_description="Tear in the intimal layer of the aorta allowing blood to dissect between the layers of the aortic wall.",
            max_query_time=25.0,
        )
    )

    # Case 6: Finding with grading/classification
    cases.append(
        FindingDescriptionCase(
            name="complex_brain_tumor",
            finding_name="glioblastoma",
            expected_keywords=["brain", "tumor", "malignant", "glioma"],
            min_description_length=40,
            max_description_length=300,
            medical_terminology_level="professional",
            max_query_time=25.0,
        )
    )

    # Case 7: Multi-system finding
    cases.append(
        FindingDescriptionCase(
            name="complex_pulmonary_edema",
            finding_name="pulmonary edema",
            expected_keywords=["lung", "fluid", "alveoli"],
            min_description_length=40,
            max_description_length=250,
            consistency_group="pulmonary",
            reference_description="Accumulation of fluid in the pulmonary interstitium and alveoli, often due to cardiac or non-cardiac causes.",
            max_query_time=25.0,
        )
    )

    return cases


def create_normalization_cases() -> list[FindingDescriptionCase]:
    """Create cases testing name normalization (acronyms, specificity, plurals)."""
    cases = []

    # Case 8: Overly specific anatomic location
    cases.append(
        FindingDescriptionCase(
            name="normalize_specific_location",
            finding_name="left lower lobe opacity",
            expected_canonical_name="pulmonary opacity",
            expected_keywords=["lung", "opacity"],
            min_description_length=20,
            max_description_length=200,
            consistency_group="pulmonary",
            max_query_time=20.0,
        )
    )

    # Case 9: Plural to singular normalization
    cases.append(
        FindingDescriptionCase(
            name="normalize_plural",
            finding_name="pulmonary nodules",
            expected_canonical_name="pulmonary nodule",
            expected_keywords=["lung", "nodule"],
            min_description_length=20,
            max_description_length=200,
            consistency_group="pulmonary",
            max_query_time=20.0,
        )
    )

    # Case 10: Acronym expansion with synonym
    cases.append(
        FindingDescriptionCase(
            name="normalize_pcl_tear",
            finding_name="PCL tear",
            expected_canonical_name="posterior cruciate ligament tear",
            expected_synonyms=["pcl tear"],
            expected_keywords=["ligament", "knee", "tear"],
            min_description_length=30,
            max_description_length=200,
            max_query_time=20.0,
        )
    )

    # Case 11: Mixed case normalization
    cases.append(
        FindingDescriptionCase(
            name="normalize_mixed_case",
            finding_name="Pneumonia",
            expected_canonical_name="pneumonia",
            expected_keywords=["lung", "infection", "inflammation"],
            min_description_length=30,
            max_description_length=200,
            consistency_group="pulmonary",
            max_query_time=20.0,
        )
    )

    return cases


def create_tavily_detail_cases() -> list[FindingDescriptionCase]:
    """Create cases testing detail enhancement with citations via search API."""
    cases = []

    # Case 12: Common finding with detail
    cases.append(
        FindingDescriptionCase(
            name="detail_pneumothorax",
            finding_name="pneumothorax",
            fetch_details=True,
            expected_keywords=["air", "pleural", "lung", "imaging", "treatment"],
            min_description_length=30,
            min_detail_length=200,
            requires_citations=True,
            medical_terminology_level="professional",
            max_query_time=45.0,
        )
    )

    # Case 13: Complex pathology with detail
    cases.append(
        FindingDescriptionCase(
            name="detail_stroke",
            finding_name="acute ischemic stroke",
            fetch_details=True,
            expected_keywords=["brain", "ischemia", "vessel", "occlusion"],
            min_description_length=40,
            min_detail_length=250,
            requires_citations=True,
            medical_terminology_level="professional",
            max_query_time=45.0,
        )
    )

    return cases


def create_edge_cases() -> list[FindingDescriptionCase]:
    """Create edge cases and boundary conditions."""
    cases = []

    # Case 14: Very short finding name
    cases.append(
        FindingDescriptionCase(
            name="edge_short_name",
            finding_name="AAA",
            expected_canonical_name="abdominal aortic aneurysm",
            expected_synonyms=["aaa"],
            expected_keywords=["aorta", "aneurysm", "abdominal"],
            min_description_length=30,
            max_query_time=20.0,
        )
    )

    # Case 15: Finding with special characters
    cases.append(
        FindingDescriptionCase(
            name="edge_special_chars",
            finding_name="Type A dissection",
            expected_keywords=["aortic", "dissection", "ascending"],
            min_description_length=30,
            max_query_time=20.0,
        )
    )

    # Case 16: Rare/uncommon finding
    cases.append(
        FindingDescriptionCase(
            name="edge_rare_finding",
            finding_name="pneumomediastinum",
            expected_keywords=["air", "mediastinum"],
            min_description_length=20,
            max_description_length=300,
            max_query_time=25.0,
        )
    )

    # Case 17: Very common simple finding
    cases.append(
        FindingDescriptionCase(
            name="edge_simple_finding",
            finding_name="opacity",
            expected_keywords=["density", "imaging"],
            min_description_length=15,
            max_description_length=150,
            max_query_time=20.0,
        )
    )

    return cases


def create_terminology_validation_cases() -> list[FindingDescriptionCase]:
    """Create cases for medical terminology validation."""
    cases = []

    # Case 18: Proper cardiovascular terminology
    cases.append(
        FindingDescriptionCase(
            name="terminology_cardiac",
            finding_name="myocardial infarction",
            expected_canonical_name="myocardial infarction",
            expected_synonyms=["mi", "heart attack"],
            expected_keywords=["myocardial", "infarction", "heart"],
            unexpected_keywords=["heart failure"],  # Different condition
            min_description_length=30,
            medical_terminology_level="professional",
            consistency_group="cardiac",
            max_query_time=20.0,
        )
    )

    # Case 19: Proper neurological terminology
    cases.append(
        FindingDescriptionCase(
            name="terminology_neuro",
            finding_name="subdural hematoma",
            expected_keywords=["subdural", "blood", "hematoma", "brain"],
            unexpected_keywords=["epidural"],  # Different location
            min_description_length=30,
            medical_terminology_level="professional",
            max_query_time=20.0,
        )
    )

    # Case 20: Proper musculoskeletal terminology
    cases.append(
        FindingDescriptionCase(
            name="terminology_msk",
            finding_name="rotator cuff tear",
            expected_keywords=["shoulder", "tendon", "rotator cuff"],
            min_description_length=30,
            medical_terminology_level="professional",
            max_query_time=20.0,
        )
    )

    return cases


# =============================================================================
# Task Execution Function
# =============================================================================


async def run_finding_description_task(input_data: FindingDescriptionInput) -> FindingDescriptionActualOutput:
    """Execute a single finding description evaluation case.

    Dataset.evaluate() automatically creates spans and captures inputs/outputs.
    Pydantic AI instrumentation captures agent/model/tool calls.
    No manual Logfire code needed.

    Args:
        input_data: Input data for the finding description case

    Returns:
        Actual output from the finding description execution
    """
    try:
        # Time the query
        start_time = time.time()

        # Create basic info from name
        finding_info = await create_info_from_name(input_data.finding_name)

        # Add detailed info if requested
        if input_data.fetch_details:
            detailed_info = await add_details_to_info(finding_info)
            if detailed_info:
                finding_info = detailed_info

        query_time = time.time() - start_time

        return FindingDescriptionActualOutput(
            finding_info=finding_info,
            query_time=query_time,
        )
    except Exception as e:
        query_time = time.time() - start_time if "start_time" in locals() else 0.0
        return FindingDescriptionActualOutput(
            finding_info=None,
            query_time=query_time,
            error=str(e),
        )


# =============================================================================
# Dataset Creation with Evaluator-Based Pattern
# =============================================================================

all_cases = (
    create_common_findings_cases()
    + create_complex_findings_cases()
    + create_normalization_cases()
    + create_tavily_detail_cases()
    + create_terminology_validation_cases()
    + create_edge_cases()
)

evaluators = [
    LengthAppropriatenessEvaluator(),
    TerminologyEvaluator(),
    SynonymQualityEvaluator(),
    CanonicalNameEvaluator(),
    CitationQualityEvaluator(),
    ReadabilityEvaluator(),
    ConsistencyEvaluator(),
    LLMJudge(
        rubric="""You are a medical expert evaluating the quality of clinical finding descriptions.

Rate the quality of the generated description on a scale from 0.0 to 1.0 based on:
1. Clinical accuracy (40%): Medical facts are correct and appropriate
2. Terminology usage (30%): Proper medical terms used correctly
3. Completeness (20%): Adequate detail for clinical use
4. Clarity (10%): Clear and well-structured

- 1.0 = Excellent: Clinically accurate, professional terminology, complete, clear
- 0.8 = Good: Minor issues with detail or clarity
- 0.6 = Acceptable: Some terminology issues or missing detail
- 0.4 = Poor: Significant inaccuracies or inappropriate terminology
- 0.2 = Very Poor: Major clinical errors or unusable
- 0.0 = Failed: Completely incorrect or nonsensical

Be strict but fair. Medical professionals will use these descriptions.""",
        include_input=True,
        include_expected_output=True,
        score={"evaluation_name": "quality", "include_reason": True},
        assertion=False,
        model=settings.get_model("small"),  # Use small model from settings
    ),
    PerformanceEvaluator(time_limit=45.0),  # Accommodate Tavily cases which take longest
]

finding_description_dataset = Dataset(cases=all_cases, evaluators=evaluators)


async def run_finding_description_evals() -> EvaluationReport[
    FindingDescriptionInput, FindingDescriptionActualOutput, FindingDescriptionExpectedOutput
]:
    """Run finding description generation evaluation suite.

    Dataset.evaluate() automatically creates evaluation spans and captures
    all inputs, outputs, and scores for visualization in Logfire.
    """
    report = await finding_description_dataset.evaluate(run_finding_description_task)
    return report


if __name__ == "__main__":
    import asyncio

    from evals import ensure_instrumented

    ensure_instrumented()  # Explicit instrumentation for eval run

    async def main() -> None:
        print("\nRunning finding description generation evaluation suite...")
        print("=" * 80)
        print("NOTE: This eval requires OpenAI and Tavily API keys and makes real API calls.")
        print("=" * 80 + "\n")

        report = await run_finding_description_evals()

        print("\n" + "=" * 80)
        print("FINDING DESCRIPTION GENERATION EVALUATION RESULTS")
        print("=" * 80 + "\n")

        # Don't include full outputs in table - focus on scores and metrics
        report.print(
            include_input=True,
            include_output=False,  # Outputs contain full FindingInfo objects - too verbose
            include_durations=True,
            width=120,
        )

        # Calculate overall score manually (average of all evaluator scores across all cases)
        all_scores = [score.value for case in report.cases for score in case.scores.values()]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        print("\n" + "=" * 80)
        print(f"OVERALL SCORE: {overall_score:.2f}")
        print("=" * 80 + "\n")

        # Quality threshold check
        quality_threshold = 0.80
        if overall_score >= quality_threshold:
            print(f"✓ PASSED: Overall score {overall_score:.2%} meets threshold {quality_threshold:.0%}")
        else:
            print(f"✗ FAILED: Overall score {overall_score:.2%} below threshold {quality_threshold:.0%}")

    asyncio.run(main())
