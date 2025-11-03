"""Evaluation suite for create_model_from_markdown agent.

INTEGRATION SUITE: Requires real API calls to OpenAI.
Run via `task evals:markdown_in`.

API CALL PATTERN:
    Unlike unit tests, this eval suite makes REAL API calls by design. The
    `models.ALLOW_MODEL_REQUESTS = False` guard is intentionally omitted since
    evaluation requires actual agent execution with real OpenAI API.

    This is an integration eval suite, not a unit test. It validates end-to-end
    behavior including:
    - Real OpenAI API calls with structured output generation
    - Actual AI model markdown parsing and interpretation
    - Complete FindingModelBase construction from markdown

    See: Serena memory 'agent_evaluation_best_practices_2025'

This module defines evaluation cases for assessing the create_model_from_markdown functionality,
which uses an AI agent to parse markdown text into structured finding model representations.

MARKDOWN FORMAT SPECIFICATION:
The expected markdown format for finding models is:

```markdown
# Finding Name Attributes

## Attribute 1 Name
- Value 1: Description
- Value 2: Description
- Value 3: Description

## Attribute 2 Name
- Value A
- Value B
```

For numeric attributes:
```markdown
## Size
Range: 1-10 cm
```

EVALUATOR-BASED PATTERN:
- Cases are evaluated using Dataset.evaluate() with focused evaluators
- Each evaluator checks a specific aspect (structural validity, attribute preservation, etc.)
- Hybrid scoring: strict for must-haves (0.0 or 1.0), partial credit for quality (0.0-1.0)

EVALUATORS:
- StructuralValidityEvaluator: Checks parsed model has correct structure (strict)
- AttributePreservationEvaluator: Verifies all attributes parsed correctly (partial credit)
- TypeCorrectnessEvaluator: Checks attribute types match markdown specification (partial credit)
- ErrorMessageQualityEvaluator: Assesses clarity of error messages (partial credit)
- RoundTripEvaluator: Model → markdown → model should be equivalent (partial credit)

LOGFIRE INTEGRATION:
Logfire observability is configured automatically in evals/__init__.py.
No manual instrumentation needed in this module - automatic spans are created by:
- Dataset.evaluate() for root and per-case spans
- Pydantic AI instrumentation for agent/model/tool calls

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

import time

from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from findingmodel.finding_info import FindingInfo
from findingmodel.finding_model import FindingModelBase
from findingmodel.tools.evaluators import PerformanceEvaluator
from findingmodel.tools.markdown_in import create_model_from_markdown


class MarkdownInInput(BaseModel):
    """Input for a markdown input evaluation case."""

    finding_name: str = Field(description="Name of the finding model")
    finding_description: str = Field(description="Description of the finding")
    synonyms: list[str] | None = Field(default=None, description="Synonyms for the finding")
    markdown_text: str = Field(description="Markdown text containing the finding model structure")


class MarkdownInExpectedOutput(BaseModel):
    """Expected output for a markdown input evaluation case."""

    should_succeed: bool = Field(default=True, description="Whether parsing should succeed")
    expected_attribute_names: list[str] = Field(
        default_factory=list, description="Expected attribute names (case-insensitive)"
    )
    expected_attribute_types: dict[str, str] = Field(
        default_factory=dict, description="Expected attribute types by attribute name (choice/numeric)"
    )
    min_attributes: int = Field(default=0, description="Minimum number of attributes expected", ge=0)
    max_attributes: int = Field(default=100, description="Maximum number of attributes allowed", ge=0)
    semantic_keywords: list[str] = Field(
        default_factory=list, description="Keywords that should appear in model or attribute descriptions"
    )
    error_keywords: list[str] = Field(
        default_factory=list, description="Keywords that should appear in error messages when failure expected"
    )
    max_query_time: float = Field(default=30.0, description="Maximum acceptable query time in seconds")
    round_trip_score_threshold: float = Field(
        default=0.99, description="Minimum acceptable round-trip preservation score", ge=0.0, le=1.0
    )


class MarkdownInActualOutput(BaseModel):
    """Actual output from running a markdown input case."""

    model: FindingModelBase | None = Field(default=None, description="Parsed finding model")
    query_time: float = Field(description="Time taken to execute query in seconds")
    error: str | None = Field(default=None, description="Error message if parsing failed")


class MarkdownInCase(Case[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput]):
    """A test case for create_model_from_markdown functionality."""

    def __init__(
        self,
        name: str,
        finding_name: str,
        finding_description: str,
        markdown_text: str,
        synonyms: list[str] | None = None,
        should_succeed: bool = True,
        expected_attribute_names: list[str] | None = None,
        expected_attribute_types: dict[str, str] | None = None,
        min_attributes: int = 0,
        max_attributes: int = 100,
        semantic_keywords: list[str] | None = None,
        error_keywords: list[str] | None = None,
        max_query_time: float = 30.0,
        round_trip_score_threshold: float = 0.99,
    ) -> None:
        """Initialize a markdown input evaluation case.

        Args:
            name: Name of the test case
            finding_name: Name of the finding model
            finding_description: Description of the finding
            markdown_text: Markdown text containing the finding model structure
            synonyms: Synonyms for the finding
            should_succeed: Whether parsing should succeed
            expected_attribute_names: Expected attribute names (case-insensitive)
            expected_attribute_types: Expected attribute types by name (choice/numeric)
            min_attributes: Minimum number of attributes expected
            max_attributes: Maximum number of attributes allowed
            semantic_keywords: Keywords for semantic similarity check
            error_keywords: Keywords that should appear in error messages
            max_query_time: Maximum acceptable query time in seconds
            round_trip_score_threshold: Minimum acceptable round-trip preservation score
        """
        inputs = MarkdownInInput(
            finding_name=finding_name,
            finding_description=finding_description,
            synonyms=synonyms,
            markdown_text=markdown_text,
        )
        metadata = MarkdownInExpectedOutput(
            should_succeed=should_succeed,
            expected_attribute_names=expected_attribute_names or [],
            expected_attribute_types=expected_attribute_types or {},
            min_attributes=min_attributes,
            max_attributes=max_attributes,
            semantic_keywords=semantic_keywords or [],
            error_keywords=error_keywords or [],
            max_query_time=max_query_time,
            round_trip_score_threshold=round_trip_score_threshold,
        )
        super().__init__(name=name, inputs=inputs, metadata=metadata)


# =============================================================================
# Focused Evaluator Classes
# =============================================================================


class StructuralValidityEvaluator(Evaluator[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput]):
    """Evaluate that parsed model has correct structure.

    This evaluator uses strict scoring (0.0 or 1.0) because structural validity
    is a non-negotiable requirement - the model must be a valid FindingModelBase
    with all required fields.

    Returns:
        1.0 if model has correct structure and required fields
        0.0 if model is missing or has invalid structure
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput],
    ) -> float:
        """Evaluate structural validity.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if structure valid, 0.0 otherwise
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # If should fail, we don't check structure (error handling evaluated separately)
        if not ctx.metadata.should_succeed:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error or ctx.output.model is None:
            return 0.0

        # Check that model is a FindingModelBase
        if not isinstance(ctx.output.model, FindingModelBase):
            return 0.0

        # Check required fields are present and valid
        if not ctx.output.model.name or not ctx.output.model.description:
            return 0.0

        # Check attributes list exists and has minimum count
        if not hasattr(ctx.output.model, "attributes") or ctx.output.model.attributes is None:
            return 0.0

        # Check attribute count constraints
        attr_count = len(ctx.output.model.attributes)
        if attr_count < ctx.metadata.min_attributes or attr_count > ctx.metadata.max_attributes:
            return 0.0

        return 1.0


class AttributePreservationEvaluator(Evaluator[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput]):
    """Verify all expected attributes are parsed correctly.

    Uses partial credit scoring to reward finding some of the expected attributes
    even if not all are found.

    Returns:
        0.0-1.0 proportional score based on how many expected attributes found
        1.0 if no expected attributes specified (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput],
    ) -> float:
        """Evaluate attribute preservation with partial credit.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Proportion of expected attributes found (0.0-1.0)
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Skip if no expected attributes specified - N/A case, return 1.0
        if not ctx.metadata.expected_attribute_names:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error or ctx.output.model is None:
            return 0.0

        # Get actual attribute names (case-insensitive)
        actual_names = {attr.name.lower() for attr in ctx.output.model.attributes}

        # Count matches (case-insensitive)
        matches = sum(
            1 for expected_name in ctx.metadata.expected_attribute_names if expected_name.lower() in actual_names
        )

        # Return proportion of expected attributes found
        return matches / len(ctx.metadata.expected_attribute_names)


class TypeCorrectnessEvaluator(Evaluator[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput]):
    """Check attribute types match markdown specification.

    Uses partial credit scoring to reward correct typing of some attributes
    even if not all types are correct.

    Returns:
        0.0-1.0 proportional score based on correct attribute types
        1.0 if no expected types specified (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput],
    ) -> float:
        """Evaluate type correctness with partial credit.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Proportion of attributes with correct types (0.0-1.0)
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Skip if no expected types specified - N/A case, return 1.0
        if not ctx.metadata.expected_attribute_types:
            return 1.0

        # Execution errors are failures - return 0.0
        if ctx.output.error or ctx.output.model is None:
            return 0.0

        # Build mapping of actual attribute names to types (case-insensitive)
        actual_types = {attr.name.lower(): attr.type.value for attr in ctx.output.model.attributes}

        # Count matches (case-insensitive names)
        correct_count = 0
        for expected_name, expected_type in ctx.metadata.expected_attribute_types.items():
            actual_type = actual_types.get(expected_name.lower())
            if actual_type and actual_type == expected_type.lower():
                correct_count += 1

        # Return proportion of correct types
        return correct_count / len(ctx.metadata.expected_attribute_types)


class ErrorMessageQualityEvaluator(Evaluator[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput]):
    """Assess clarity of error messages.

    Uses partial credit scoring to reward error messages that contain relevant
    keywords indicating what went wrong.

    Returns:
        0.0-1.0 proportional score based on error keyword presence
        1.0 if no error expected (N/A)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput],
    ) -> float:
        """Evaluate error message quality.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Proportion of error keywords found in error message (0.0-1.0)
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # If should succeed, error message quality is N/A - return 1.0
        if ctx.metadata.should_succeed:
            return 1.0

        # Skip if no error keywords specified - N/A case, return 1.0
        if not ctx.metadata.error_keywords:
            return 1.0

        # If no error occurred when one was expected, return 0.0
        if not ctx.output.error:
            return 0.0

        # Count keyword matches in error message (case-insensitive)
        error_text = ctx.output.error.lower()
        matches = sum(1 for keyword in ctx.metadata.error_keywords if keyword.lower() in error_text)

        # Return proportion of keywords found
        return matches / len(ctx.metadata.error_keywords)


class RoundTripEvaluator(Evaluator[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput]):
    """Verify model → markdown → model round-trip preservation.

    Tests that converting a parsed model back to markdown and re-parsing it
    produces an equivalent model. Uses partial credit based on how many
    attributes are preserved.

    Returns:
        0.0-1.0 score based on attribute preservation in round-trip
        1.0 if no model parsed (N/A for failed cases)
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput],
    ) -> float:
        """Evaluate round-trip preservation.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Round-trip preservation score (0.0-1.0)
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # If should fail or execution error, round-trip is N/A - return 1.0
        if not ctx.metadata.should_succeed or ctx.output.error or ctx.output.model is None:
            return 1.0

        # Convert model to markdown
        try:
            markdown_output = ctx.output.model.as_markdown()
        except Exception:
            # If markdown conversion fails, round-trip fails
            return 0.0

        # Compare original and round-trip markdown
        # Count preserved attributes (basic heuristic: check attribute names appear in markdown)
        original_attrs = {attr.name.lower() for attr in ctx.output.model.attributes}
        preserved_count = sum(1 for attr_name in original_attrs if attr_name in markdown_output.lower())

        # Return proportion of attributes preserved in markdown
        if len(original_attrs) == 0:
            return 1.0  # No attributes to preserve
        return preserved_count / len(original_attrs)


# =============================================================================
# Test Case Creation Functions
# =============================================================================


def create_success_cases() -> list[MarkdownInCase]:
    """Create success cases for well-formed markdown."""
    cases = []

    # Case 1: Simple choice attributes
    cases.append(
        MarkdownInCase(
            name="success_simple_choice_attributes",
            finding_name="pneumothorax",
            finding_description="Air in the pleural space causing lung collapse",
            markdown_text="""
# Pneumothorax Attributes

## Size
- Small: Less than 2cm
- Moderate: 2-4cm
- Large: Greater than 4cm

## Location
- Apical
- Basilar
- Complete

## Tension
- Present
- Absent
""",
            expected_attribute_names=["size", "location", "tension"],
            expected_attribute_types={"size": "choice", "location": "choice", "tension": "choice"},
            min_attributes=3,
            semantic_keywords=["pneumothorax", "air", "pleural"],
            max_query_time=25.0,
        )
    )

    # Case 2: Numeric attributes with units
    cases.append(
        MarkdownInCase(
            name="success_numeric_with_units",
            finding_name="pulmonary nodule",
            finding_description="Focal rounded opacity in lung parenchyma",
            markdown_text="""
# Pulmonary Nodule Attributes

## Size
Range: 1-30 mm

## Number of nodules
Range: 1-10

## Density
- Solid
- Ground glass
- Part-solid
""",
            expected_attribute_names=["size", "number of nodules", "density"],
            expected_attribute_types={"size": "numeric", "number of nodules": "numeric", "density": "choice"},
            min_attributes=3,
            semantic_keywords=["nodule", "lung", "opacity"],
            max_query_time=25.0,
        )
    )

    # Case 3: Multiple choice attributes
    cases.append(
        MarkdownInCase(
            name="success_multiple_choice_attributes",
            finding_name="liver lesion",
            finding_description="Focal abnormality in liver parenchyma",
            markdown_text="""
# Liver Lesion Attributes

## Morphology
- Cystic
- Solid
- Mixed

## Enhancement pattern
- Hyperenhancing
- Hypoenhancing
- No enhancement

## Location
- Right lobe
- Left lobe
- Caudate
- Quadrate

## Size category
- Small (<1cm)
- Medium (1-5cm)
- Large (>5cm)
""",
            expected_attribute_names=["morphology", "enhancement pattern", "location", "size category"],
            expected_attribute_types={
                "morphology": "choice",
                "enhancement pattern": "choice",
                "location": "choice",
                "size category": "choice",
            },
            min_attributes=4,
            semantic_keywords=["liver", "lesion", "abnormality"],
            max_query_time=25.0,
        )
    )

    # Case 4: Mixed attribute types
    cases.append(
        MarkdownInCase(
            name="success_mixed_attribute_types",
            finding_name="aortic aneurysm",
            finding_description="Focal dilation of the aorta",
            markdown_text="""
# Aortic Aneurysm Attributes

## Maximum diameter
Range: 3-10 cm

## Location
- Ascending
- Arch
- Descending
- Abdominal

## Morphology
- Fusiform
- Saccular

## Thrombus present
- Yes
- No
""",
            expected_attribute_names=["maximum diameter", "location", "morphology", "thrombus present"],
            expected_attribute_types={
                "maximum diameter": "numeric",
                "location": "choice",
                "morphology": "choice",
                "thrombus present": "choice",
            },
            min_attributes=4,
            semantic_keywords=["aortic", "aneurysm", "dilation"],
            max_query_time=25.0,
        )
    )

    # Case 5: Attributes with detailed descriptions
    cases.append(
        MarkdownInCase(
            name="success_detailed_descriptions",
            finding_name="fracture",
            finding_description="Break in bone continuity",
            markdown_text="""
# Fracture Attributes

## Fracture type
- Simple: Single fracture line without fragments
- Comminuted: Multiple bone fragments
- Compound: Fracture with skin break

## Displacement
- Non-displaced: Fragments in anatomic alignment
- Displaced: Fragments not in anatomic alignment

## Angulation
Range: 0-90 degrees
""",
            expected_attribute_names=["fracture type", "displacement", "angulation"],
            expected_attribute_types={"fracture type": "choice", "displacement": "choice", "angulation": "numeric"},
            min_attributes=3,
            semantic_keywords=["fracture", "bone", "break"],
            max_query_time=25.0,
        )
    )

    return cases


def create_edge_cases() -> list[MarkdownInCase]:
    """Create edge cases and boundary conditions."""
    cases = []

    # Case 6: Very long descriptions
    cases.append(
        MarkdownInCase(
            name="edge_long_descriptions",
            finding_name="brain tumor",
            finding_description="Mass lesion in brain parenchyma with various characteristics",
            markdown_text="""
# Brain Tumor Attributes

## Tumor type
- Glioblastoma: Highly aggressive primary brain tumor with rapid growth, irregular borders, central necrosis, and extensive surrounding edema
- Meningioma: Typically benign tumor arising from meninges with well-defined borders and homogeneous enhancement
- Metastasis: Secondary tumor from distant primary malignancy showing variable appearance depending on origin

## Size
Range: 0.5-10 cm
""",
            expected_attribute_names=["tumor type", "size"],
            expected_attribute_types={"tumor type": "choice", "size": "numeric"},
            min_attributes=2,
            semantic_keywords=["tumor", "brain", "lesion"],
            max_query_time=30.0,
        )
    )

    # Case 7: Special characters in attribute names
    cases.append(
        MarkdownInCase(
            name="edge_special_characters",
            finding_name="Type A dissection",
            finding_description="Stanford Type A aortic dissection",
            markdown_text="""
# Type A Dissection Attributes

## Dissection extent
- Type A (ascending)
- Type B (descending)

## Valve involvement
- Yes
- No

## Branch vessel involvement
- Left subclavian
- Brachiocephalic
- Left carotid
- None
""",
            expected_attribute_names=["dissection extent", "valve involvement", "branch vessel involvement"],
            expected_attribute_types={
                "dissection extent": "choice",
                "valve involvement": "choice",
                "branch vessel involvement": "choice",
            },
            min_attributes=3,
            semantic_keywords=["dissection", "aortic", "type"],
            max_query_time=25.0,
        )
    )

    # Case 8: Extra whitespace and formatting variations
    cases.append(
        MarkdownInCase(
            name="edge_whitespace_formatting",
            finding_name="pleural effusion",
            finding_description="Fluid in pleural space",
            markdown_text="""


# Pleural Effusion Attributes


## Side

- Left
- Right
- Bilateral


## Size

- Small
- Moderate
- Large


""",
            expected_attribute_names=["side", "size"],
            expected_attribute_types={"side": "choice", "size": "choice"},
            min_attributes=2,
            semantic_keywords=["pleural", "effusion", "fluid"],
            max_query_time=25.0,
        )
    )

    # Case 9: Minimal attribute count (single attribute)
    cases.append(
        MarkdownInCase(
            name="edge_minimal_attributes",
            finding_name="atelectasis",
            finding_description="Lung collapse or incomplete expansion",
            markdown_text="""
# Atelectasis Attributes

## Severity
- Mild
- Moderate
- Severe
""",
            expected_attribute_names=["severity"],
            expected_attribute_types={"severity": "choice"},
            min_attributes=1,
            semantic_keywords=["atelectasis", "lung", "collapse"],
            max_query_time=20.0,
        )
    )

    # Case 10: Many attributes (boundary test)
    cases.append(
        MarkdownInCase(
            name="edge_many_attributes",
            finding_name="comprehensive lung finding",
            finding_description="Detailed lung pathology assessment",
            markdown_text="""
# Comprehensive Lung Finding Attributes

## Size
Range: 1-10 cm

## Location
- Upper lobe
- Middle lobe
- Lower lobe

## Laterality
- Left
- Right
- Bilateral

## Density
- Solid
- Ground glass
- Cavitary

## Margins
- Well-defined
- Ill-defined

## Enhancement
- Yes
- No

## Calcification
- Present
- Absent

## Associated findings
- Pleural effusion
- Lymphadenopathy
- None
""",
            expected_attribute_names=[
                "size",
                "location",
                "laterality",
                "density",
                "margins",
                "enhancement",
                "calcification",
                "associated findings",
            ],
            min_attributes=8,
            semantic_keywords=["lung", "finding"],
            max_query_time=30.0,
        )
    )

    # Case 11: Mixed case in markdown headers
    cases.append(
        MarkdownInCase(
            name="edge_mixed_case_headers",
            finding_name="kidney stone",
            finding_description="Calculus in the kidney",
            markdown_text="""
# Kidney Stone Attributes

## STONE SIZE
- Small (<5mm)
- Medium (5-10mm)
- Large (>10mm)

## Location in Kidney
- Upper pole
- Mid pole
- Lower pole

## stone_composition
- Calcium
- Uric acid
- Struvite
""",
            expected_attribute_names=["stone size", "location in kidney", "stone composition"],
            min_attributes=3,
            semantic_keywords=["kidney", "stone", "calculus"],
            max_query_time=25.0,
        )
    )

    # Case 12: Attributes with ranges and units variations
    cases.append(
        MarkdownInCase(
            name="edge_range_unit_variations",
            finding_name="mass",
            finding_description="Solid mass lesion",
            markdown_text="""
# Mass Attributes

## Diameter
Range: 0.5 to 15 centimeters

## Volume
Range: 1-100 cubic cm

## Growth rate
Range: 0-50 percent per year

## Morphology
- Round
- Irregular
- Lobulated
""",
            expected_attribute_names=["diameter", "volume", "growth rate", "morphology"],
            min_attributes=3,  # At least 3 should be parsed (might vary based on AI interpretation)
            semantic_keywords=["mass", "lesion"],
            max_query_time=25.0,
        )
    )

    return cases


def create_error_handling_cases() -> list[MarkdownInCase]:
    """Create error handling and robustness cases."""
    cases = []

    # Case 13: Empty markdown
    cases.append(
        MarkdownInCase(
            name="error_empty_markdown",
            finding_name="test finding",
            finding_description="Test description",
            markdown_text="",
            should_succeed=False,
            error_keywords=["empty", "invalid", "parse", "error"],
            max_query_time=20.0,
        )
    )

    # Case 14: Markdown with no attributes
    cases.append(
        MarkdownInCase(
            name="error_no_attributes",
            finding_name="test finding",
            finding_description="Test description",
            markdown_text="""
# Test Finding Attributes

Just some text but no actual attributes defined.
""",
            should_succeed=False,
            error_keywords=["attribute", "required", "missing"],
            max_query_time=20.0,
        )
    )

    # Case 15: Malformed markdown (no headers)
    cases.append(
        MarkdownInCase(
            name="error_malformed_no_headers",
            finding_name="test finding",
            finding_description="Test description",
            markdown_text="""
This is just plain text without any markdown structure.
No headers, no lists, nothing.
Size: small, medium, large
Location: left, right
""",
            should_succeed=False,
            error_keywords=["format", "parse", "invalid"],
            max_query_time=20.0,
        )
    )

    # Case 16: Attribute with no values
    cases.append(
        MarkdownInCase(
            name="error_attribute_no_values",
            finding_name="test finding",
            finding_description="Test description",
            markdown_text="""
# Test Finding Attributes

## Severity

## Location
- Left
- Right
""",
            should_succeed=False,
            error_keywords=["value", "required", "empty"],
            max_query_time=20.0,
        )
    )

    return cases


def create_complex_structure_cases() -> list[MarkdownInCase]:
    """Create cases with complex attribute structures."""
    cases = []

    # Case 17: Hierarchical attribute names
    cases.append(
        MarkdownInCase(
            name="complex_hierarchical_names",
            finding_name="coronary artery disease",
            finding_description="Atherosclerotic disease of coronary arteries",
            markdown_text="""
# Coronary Artery Disease Attributes

## Left main stenosis
- None
- Mild (<50%)
- Moderate (50-70%)
- Severe (>70%)

## LAD stenosis
- None
- Mild (<50%)
- Moderate (50-70%)
- Severe (>70%)

## Circumflex stenosis
- None
- Mild (<50%)
- Moderate (50-70%)
- Severe (>70%)

## RCA stenosis
- None
- Mild (<50%)
- Moderate (50-70%)
- Severe (>70%)
""",
            expected_attribute_names=["left main stenosis", "lad stenosis", "circumflex stenosis", "rca stenosis"],
            min_attributes=4,
            semantic_keywords=["coronary", "artery", "stenosis"],
            max_query_time=30.0,
        )
    )

    # Case 18: Attributes with constraints and qualifiers
    cases.append(
        MarkdownInCase(
            name="complex_constraints_qualifiers",
            finding_name="stroke",
            finding_description="Acute cerebrovascular accident",
            markdown_text="""
# Stroke Attributes

## ASPECTS score
Range: 0-10 (Alberta Stroke Program Early CT Score)

## Vessel occlusion site
- M1 segment
- M2 segment
- M3 segment
- ICA terminus
- Basilar artery
- None

## Hemorrhagic transformation
- None
- HI-1 (small petechiae)
- HI-2 (confluent petechiae)
- PH-1 (blood clot <30%)
- PH-2 (blood clot >30%)

## Time from onset
Range: 0-24 hours
""",
            expected_attribute_names=[
                "aspects score",
                "vessel occlusion site",
                "hemorrhagic transformation",
                "time from onset",
            ],
            min_attributes=4,
            semantic_keywords=["stroke", "cerebrovascular", "occlusion"],
            max_query_time=30.0,
        )
    )

    return cases


# =============================================================================
# Task Execution Function
# =============================================================================


async def run_markdown_in_task(input_data: MarkdownInInput) -> MarkdownInActualOutput:
    """Execute a single create_model_from_markdown evaluation case.

    Dataset.evaluate() automatically creates spans and captures inputs/outputs.
    Pydantic AI instrumentation captures agent/model/tool calls.
    No manual Logfire code needed.

    Args:
        input_data: Input data for the markdown input case

    Returns:
        Actual output from the markdown parsing execution

    Raises:
        Exception: Any errors are caught and returned in the error field
    """
    # Time the query
    start_time = time.time()
    try:
        # Create FindingInfo from inputs
        finding_info = FindingInfo(
            name=input_data.finding_name,
            description=input_data.finding_description,
            synonyms=input_data.synonyms,
        )

        model = await create_model_from_markdown(
            finding_info,
            markdown_text=input_data.markdown_text,
        )
        query_time = time.time() - start_time

        return MarkdownInActualOutput(
            model=model,
            query_time=query_time,
        )
    except Exception as e:
        # Return error in output for evaluation
        query_time = time.time() - start_time if "start_time" in locals() else 0.0
        return MarkdownInActualOutput(
            model=None,
            query_time=query_time,
            error=str(e),
        )


# =============================================================================
# Dataset Creation with Evaluator-Based Pattern
# =============================================================================

all_cases = (
    create_success_cases() + create_edge_cases() + create_error_handling_cases() + create_complex_structure_cases()
)

evaluators = [
    StructuralValidityEvaluator(),
    AttributePreservationEvaluator(),
    TypeCorrectnessEvaluator(),
    ErrorMessageQualityEvaluator(),
    RoundTripEvaluator(),
    PerformanceEvaluator(time_limit=30.0),
]

markdown_in_dataset = Dataset(cases=all_cases, evaluators=evaluators)


async def run_markdown_in_evals() -> EvaluationReport[
    MarkdownInInput, MarkdownInActualOutput, MarkdownInExpectedOutput
]:
    """Run create_model_from_markdown evaluation suite.

    Dataset.evaluate() automatically creates evaluation spans and captures
    all inputs, outputs, and scores for visualization in Logfire.
    """
    report = await markdown_in_dataset.evaluate(run_markdown_in_task)
    return report


if __name__ == "__main__":
    import asyncio

    from evals import ensure_instrumented

    ensure_instrumented()  # Explicit instrumentation for eval run

    async def main() -> None:
        print("\nRunning create_model_from_markdown evaluation suite...")
        print("=" * 80)
        print("NOTE: This eval requires OpenAI API key and makes real API calls.")
        print("=" * 80 + "\n")

        report = await run_markdown_in_evals()

        print("\n" + "=" * 80)
        print("MARKDOWN INPUT PARSER EVALUATION RESULTS")
        print("=" * 80 + "\n")

        # Don't include full outputs in table - focus on scores and metrics
        report.print(
            include_input=True,
            include_output=False,  # Outputs contain full models - too verbose
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
