"""Reusable evaluator classes for findingmodel agents.

This module provides generic evaluator classes that can be used across different
agent evaluation suites. Each evaluator follows the Pydantic AI Evals framework
and returns continuous scores (0.0-1.0) to enable nuanced evaluation and partial credit.

All evaluators inherit from pydantic_evals.evaluators.Evaluator and are designed
to be reusable across different input and output types through generics.
"""

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

# Type variables for generic evaluators
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ExactMatchEvaluator(Evaluator[InputT, str], Generic[InputT]):
    """Check if output exactly matches expected output.

    Returns 1.0 if the output exactly matches the expected output, 0.0 otherwise.
    This evaluator is strict and does not allow for partial matches or variations.

    Example usage:
        >>> from pydantic_evals import Case, Dataset
        >>> from pydantic import BaseModel
        >>>
        >>> class MyInput(BaseModel):
        ...     query: str
        >>>
        >>> class MyExpectedOutput(BaseModel):
        ...     text: str
        >>>
        >>> evaluator = ExactMatchEvaluator()
        >>> case = Case(
        ...     name="test_exact_match",
        ...     inputs=MyInput(query="test"),
        ...     expected_output=MyExpectedOutput(text="expected text")
        ... )
        >>> # Assuming output is "expected text"
        >>> # score = 1.0
    """

    def evaluate(self, ctx: EvaluatorContext[InputT, str]) -> float:
        """Evaluate exact string match.

        Args:
            ctx: Evaluation context containing output and expected_output

        Returns:
            1.0 if output exactly matches expected_output, 0.0 otherwise
        """
        # Handle both string expected_output and objects with text attribute
        expected_text: str
        if isinstance(ctx.expected_output, str):
            expected_text = ctx.expected_output
        elif ctx.expected_output is not None and hasattr(ctx.expected_output, "text"):
            expected_text = str(ctx.expected_output.text)
        else:
            expected_text = str(ctx.expected_output) if ctx.expected_output is not None else ""

        return 1.0 if ctx.output == expected_text else 0.0


class ContainsEvaluator(Evaluator[InputT, str], Generic[InputT]):
    """Check if output contains expected substring.

    Returns 1.0 if the output contains the expected substring, 0.0 otherwise.
    Supports case-sensitive and case-insensitive matching.

    Example usage:
        >>> from pydantic_evals import Case, Dataset
        >>> from pydantic import BaseModel
        >>>
        >>> class MyInput(BaseModel):
        ...     query: str
        >>>
        >>> class MyExpectedOutput(BaseModel):
        ...     substring: str
        >>>
        >>> # Case-insensitive matching (default)
        >>> evaluator = ContainsEvaluator(case_sensitive=False)
        >>> case = Case(
        ...     name="test_contains",
        ...     inputs=MyInput(query="test"),
        ...     expected_output=MyExpectedOutput(substring="KEYWORD")
        ... )
        >>> # Assuming output is "This contains keyword text"
        >>> # score = 1.0 (found "keyword" ignoring case)
        >>>
        >>> # Case-sensitive matching
        >>> evaluator = ContainsEvaluator(case_sensitive=True)
        >>> # score = 0.0 (exact case "KEYWORD" not found)
    """

    def __init__(self, case_sensitive: bool = False) -> None:
        """Initialize the contains evaluator.

        Args:
            case_sensitive: If True, perform case-sensitive matching. Default False.
        """
        self.case_sensitive = case_sensitive

    def evaluate(self, ctx: EvaluatorContext[InputT, str]) -> float:
        """Evaluate substring containment.

        Args:
            ctx: Evaluation context containing output and expected_output

        Returns:
            1.0 if expected substring is found in output, 0.0 otherwise
        """
        # Handle both string expected_output and objects with substring attribute
        expected_substring: str
        if isinstance(ctx.expected_output, str):
            expected_substring = ctx.expected_output
        elif ctx.expected_output is not None and hasattr(ctx.expected_output, "substring"):
            expected_substring = str(ctx.expected_output.substring)
        else:
            expected_substring = str(ctx.expected_output) if ctx.expected_output is not None else ""

        # Apply case sensitivity setting
        output_text = ctx.output if self.case_sensitive else ctx.output.lower()
        expected_text = expected_substring if self.case_sensitive else expected_substring.lower()

        return 1.0 if expected_text in output_text else 0.0


class KeywordMatchEvaluator(Evaluator[InputT, OutputT], Generic[InputT, OutputT]):
    """Check for multiple keywords in extracted text with partial credit.

    Evaluates how many of the expected keywords appear in the output text.
    Returns a score proportional to the number of matches, enabling partial credit.

    Example usage:
        >>> from pydantic_evals import Case, Dataset
        >>> from pydantic import BaseModel
        >>>
        >>> class MyInput(BaseModel):
        ...     query: str
        >>>
        >>> class MyExpectedOutput(BaseModel):
        ...     expected_keywords: list[str]
        >>>
        >>> class MyOutput(BaseModel):
        ...     result: str
        ...     error: str | None = None
        >>>
        >>> # Define text extractor function
        >>> def extract_text(output: MyOutput) -> str:
        ...     return output.result
        >>>
        >>> evaluator = KeywordMatchEvaluator(
        ...     keyword_field="expected_keywords",
        ...     text_extractor=extract_text,
        ...     partial_credit=True
        ... )
        >>>
        >>> case = Case(
        ...     name="test_keywords",
        ...     inputs=MyInput(query="test"),
        ...     expected_output=MyExpectedOutput(expected_keywords=["severity", "added", "attribute"])
        ... )
        >>> # If output.result contains "severity" and "attribute" but not "added"
        >>> # score = 2/3 = 0.67 (partial credit)
        >>>
        >>> # With partial_credit=False
        >>> # score = 0.0 (all keywords must be present)
    """

    def __init__(
        self,
        keyword_field: str,
        text_extractor: Callable[[OutputT], str],
        partial_credit: bool = True,
    ) -> None:
        """Initialize the keyword match evaluator.

        Args:
            keyword_field: Name of the field in expected_output that contains the list of keywords
            text_extractor: Function to extract searchable text from the output object
            partial_credit: If True, return proportion of matches. If False, require all matches.
        """
        self.keyword_field = keyword_field
        self.text_extractor = text_extractor
        self.partial_credit = partial_credit

    def evaluate(self, ctx: EvaluatorContext[InputT, OutputT]) -> float:
        """Evaluate keyword matching with partial credit.

        Args:
            ctx: Evaluation context containing output and expected_output

        Returns:
            If partial_credit=True: proportion of keywords found (0.0-1.0)
            If partial_credit=False: 1.0 if all keywords found, 0.0 otherwise
            Returns 1.0 if no keywords expected (N/A case)
        """
        keywords = getattr(ctx.expected_output, self.keyword_field, [])

        # Handle empty keyword list - return 1.0 (N/A)
        if not keywords:
            return 1.0

        # Extract text from output
        text = self.text_extractor(ctx.output).lower()

        # Count matches
        matches = sum(1 for keyword in keywords if keyword.lower() in text)

        # Return score based on partial_credit setting
        if self.partial_credit:
            return matches / len(keywords)
        else:
            return 1.0 if matches == len(keywords) else 0.0


class StructuralValidityEvaluator(Evaluator[InputT, BaseModel], Generic[InputT]):
    """Check if output has required fields.

    Validates that a Pydantic model output contains all required fields.
    Useful for ensuring structured outputs have expected attributes.

    Example usage:
        >>> from pydantic_evals import Case, Dataset
        >>> from pydantic import BaseModel
        >>>
        >>> class MyInput(BaseModel):
        ...     query: str
        >>>
        >>> class MyExpectedOutput(BaseModel):
        ...     pass  # No expected output needed for structural checks
        >>>
        >>> class MyOutput(BaseModel):
        ...     model_id: str
        ...     attributes: list
        ...     description: str
        >>>
        >>> # Check for specific required fields
        >>> evaluator = StructuralValidityEvaluator(
        ...     required_fields=["model_id", "attributes"]
        ... )
        >>> # If output has both fields: score = 1.0
        >>> # If output has only model_id: score = 0.5
        >>> # If output has neither: score = 0.0
        >>>
        >>> # Check that output is valid Pydantic model (no specific fields)
        >>> evaluator = StructuralValidityEvaluator()
        >>> # Returns 1.0 if output is a BaseModel instance
    """

    def __init__(self, required_fields: list[str] | None = None) -> None:
        """Initialize the structural validity evaluator.

        Args:
            required_fields: List of field names that must be present in the output.
                           If None or empty, just checks that output is a valid BaseModel.
        """
        self.required_fields = required_fields or []

    def evaluate(self, ctx: EvaluatorContext[InputT, BaseModel]) -> float:
        """Evaluate structural validity and field presence.

        Args:
            ctx: Evaluation context containing output

        Returns:
            If required_fields specified: proportion of required fields present (0.0-1.0)
            If no required_fields: 1.0 if output is BaseModel, 0.0 otherwise
            Returns 1.0 if no fields required (N/A case)
        """
        # Check if output is a Pydantic model
        if not isinstance(ctx.output, BaseModel):
            return 0.0

        # If no specific fields required, just validate it's a BaseModel
        if not self.required_fields:
            return 1.0

        # Check which required fields are present
        model_dict = ctx.output.model_dump()
        present_count = sum(1 for field in self.required_fields if field in model_dict)

        # Return proportion of required fields present
        return present_count / len(self.required_fields)


class ErrorHandlingEvaluator(Evaluator[InputT, OutputT], Generic[InputT, OutputT]):
    """Verify errors are handled as expected.

    Checks that operations succeed when they should and fail when they should.
    Uses the expected_output.should_succeed field and output.error field to determine correctness.

    Example usage:
        >>> from pydantic_evals import Case, Dataset
        >>> from pydantic import BaseModel
        >>>
        >>> class MyInput(BaseModel):
        ...     query: str
        >>>
        >>> class MyExpectedOutput(BaseModel):
        ...     should_succeed: bool
        >>>
        >>> class MyOutput(BaseModel):
        ...     result: str | None = None
        ...     error: str | None = None
        >>>
        >>> evaluator = ErrorHandlingEvaluator(error_field="error")
        >>>
        >>> # Case 1: Should succeed and does (no error)
        >>> case1 = Case(
        ...     name="success_case",
        ...     inputs=MyInput(query="valid"),
        ...     expected_output=MyExpectedOutput(should_succeed=True)
        ... )
        >>> # If output.error is None: score = 1.0
        >>> # If output.error is set: score = 0.0
        >>>
        >>> # Case 2: Should fail and does (has error)
        >>> case2 = Case(
        ...     name="rejection_case",
        ...     inputs=MyInput(query="invalid"),
        ...     expected_output=MyExpectedOutput(should_succeed=False)
        ... )
        >>> # If output.error is set: score = 1.0
        >>> # If output.error is None: score = 0.0
    """

    def __init__(self, error_field: str = "error") -> None:
        """Initialize the error handling evaluator.

        Args:
            error_field: Name of the field in output that contains error information.
                        If this field is None, operation succeeded. If set, it failed.
        """
        self.error_field = error_field

    def evaluate(self, ctx: EvaluatorContext[InputT, OutputT]) -> float:
        """Evaluate error handling correctness.

        Args:
            ctx: Evaluation context containing output and expected_output

        Returns:
            1.0 if error handling matches expectation (success/failure as expected)
            0.0 if error handling does not match expectation
        """
        # Determine if operation should succeed
        should_succeed = getattr(ctx.expected_output, "should_succeed", True)

        # Determine if operation actually succeeded (no error)
        error_value = getattr(ctx.output, self.error_field, None)
        has_error = error_value is not None

        # Evaluate based on expectation vs reality
        if should_succeed and not has_error:
            return 1.0  # Should succeed and did succeed
        elif not should_succeed and has_error:
            return 1.0  # Should fail and did fail
        else:
            return 0.0  # Mismatch between expectation and reality


# Type variables for the evaluation suite base class
ExpectedT = TypeVar("ExpectedT")
ActualT = TypeVar("ActualT")


class AgentEvaluationSuite(ABC, Generic[InputT, ExpectedT, ActualT]):
    """Abstract base class for creating agent evaluation suites.

    This class provides a standardized structure for building evaluation suites
    for any findingmodel agent. Subclasses must implement methods to create
    different categories of test cases (successful, failure, edge cases) and
    provide the logic to execute the agent being tested.

    The suite uses Pydantic AI Evals framework to evaluate non-deterministic
    agent behavior with continuous scoring (0.0-1.0) and partial credit.

    Type Parameters:
        InputT: The input model type for the agent (e.g., ModelEditorInput)
        ExpectedT: The expected output model type (e.g., ModelEditorExpectedOutput)
        ActualT: The actual output model type from the agent (e.g., ModelEditorActualOutput)

    Example usage:
        >>> from pydantic import BaseModel
        >>> from pydantic_evals import Case, Dataset
        >>> from pydantic_ai import Agent
        >>>
        >>> # Define your data models
        >>> class MyAgentInput(BaseModel):
        ...     query: str
        ...     model_json: str
        >>>
        >>> class MyAgentExpectedOutput(BaseModel):
        ...     should_succeed: bool
        ...     expected_keywords: list[str] = []
        >>>
        >>> class MyAgentActualOutput(BaseModel):
        ...     result: str
        ...     modified_model: dict | None = None
        ...     error: str | None = None
        >>>
        >>> # Create your evaluation suite
        >>> class MyAgentEvaluationSuite(AgentEvaluationSuite[MyAgentInput, MyAgentExpectedOutput, MyAgentActualOutput]):
        ...     def __init__(self, agent: Agent):
        ...         self.agent = agent
        ...
        ...     def create_successful_cases(self) -> list[Case[MyAgentInput, MyAgentExpectedOutput]]:
        ...         \"\"\"Create cases where agent should succeed.\"\"\"
        ...         return [
        ...             Case(
        ...                 name="add_attribute",
        ...                 inputs=MyAgentInput(query="Add severity attribute", model_json="{}"),
        ...                 expected_output=MyAgentExpectedOutput(
        ...                     should_succeed=True,
        ...                     expected_keywords=["added", "severity"]
        ...                 )
        ...             )
        ...         ]
        ...
        ...     def create_failure_cases(self) -> list[Case[MyAgentInput, MyAgentExpectedOutput]]:
        ...         \"\"\"Create cases where agent should fail gracefully.\"\"\"
        ...         return [
        ...             Case(
        ...                 name="reject_delete",
        ...                 inputs=MyAgentInput(query="Delete all attributes", model_json="{}"),
        ...                 expected_output=MyAgentExpectedOutput(
        ...                     should_succeed=False,
        ...                     expected_keywords=["reject", "not allowed"]
        ...                 )
        ...             )
        ...         ]
        ...
        ...     def create_edge_cases(self) -> list[Case[MyAgentInput, MyAgentExpectedOutput]]:
        ...         \"\"\"Create edge cases and boundary conditions.\"\"\"
        ...         return [
        ...             Case(
        ...                 name="empty_query",
        ...                 inputs=MyAgentInput(query="", model_json="{}"),
        ...                 expected_output=MyAgentExpectedOutput(should_succeed=False)
        ...             )
        ...         ]
        ...
        ...     async def execute_agent(self, input_data: MyAgentInput) -> MyAgentActualOutput:
        ...         \"\"\"Execute the agent being tested.\"\"\"
        ...         try:
        ...             result = await self.agent.run(input_data.query, model_json=input_data.model_json)
        ...             return MyAgentActualOutput(
        ...                 result=result.data.summary,
        ...                 modified_model=result.data.model
        ...             )
        ...         except Exception as e:
        ...             return MyAgentActualOutput(result="", error=str(e))
        ...
        >>> # Use the evaluation suite
        >>> suite = MyAgentEvaluationSuite(my_agent)
        >>> evaluators = [ErrorHandlingEvaluator(), KeywordMatchEvaluator(...)]
        >>> dataset = suite.build_dataset(evaluators)
        >>> report = await dataset.evaluate_async(suite.execute_agent)
        >>> report.print(include_input=True, include_output=True)
        >>> all_scores = [score.value for case in report.cases for score in case.scores.values()]
        >>> overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        >>> assert overall_score >= 0.9, f"Only {overall_score:.1%} passing"
    """

    @abstractmethod
    def create_successful_cases(self) -> list[Case[InputT, ExpectedT]]:
        """Create test cases where the agent should succeed.

        These are positive test cases where the agent should successfully complete
        the requested operation and return a valid result without errors.

        Returns:
            List of Case objects representing successful scenarios
        """
        pass

    @abstractmethod
    def create_failure_cases(self) -> list[Case[InputT, ExpectedT]]:
        """Create test cases where the agent should fail gracefully.

        These are cases where the agent should recognize that the request is
        invalid, unsafe, or not supported, and should reject it appropriately
        with a helpful error message.

        Returns:
            List of Case objects representing scenarios requiring rejection
        """
        pass

    @abstractmethod
    def create_edge_cases(self) -> list[Case[InputT, ExpectedT]]:
        """Create edge cases and boundary conditions.

        These test the agent's behavior at the boundaries of its capabilities,
        such as empty inputs, very large inputs, unusual formatting, or
        uncommon combinations of parameters.

        Returns:
            List of Case objects representing edge cases
        """
        pass

    @abstractmethod
    async def execute_agent(self, input_data: InputT) -> ActualT:
        """Execute the agent being tested.

        This method wraps the agent execution with appropriate error handling
        to capture both successful results and errors in the ActualT output model.

        Args:
            input_data: The input data to pass to the agent

        Returns:
            The actual output from the agent, including any error information
        """
        pass

    def get_all_cases(self) -> list[Case[InputT, ExpectedT]]:
        """Get all test cases combined from all categories.

        This helper method combines successful cases, failure cases, and edge cases
        into a single list for easy dataset creation.

        Returns:
            Complete list of all test cases
        """
        return self.create_successful_cases() + self.create_failure_cases() + self.create_edge_cases()

    def build_dataset(self, evaluators: list[Evaluator]) -> Dataset:
        """Build a Dataset with all cases and the provided evaluators.

        This helper method creates a complete Dataset ready for evaluation by
        combining all test cases with the specified evaluators.

        Args:
            evaluators: List of Evaluator instances to apply to all cases

        Returns:
            Dataset configured with all cases and evaluators
        """
        return Dataset(cases=self.get_all_cases(), evaluators=evaluators)
