

# Research: AI-Based Finding Model Editor (Expanded)

## Technical Context
- Use `Agent[Input, Output]` from Pydantic AI for both editing modes, with clear system prompts and output types.
- All logic is implemented using the current findingmodel package, no new abstractions.
- Only two editing modes: natural language and markdown.
- All changes must preserve OIFM IDs and use the ID manager for new IDs.

## Pydantic AI Agent Usage Patterns
- Construct agents with `Agent[InputType, OutputType]`, e.g.:
	```python
	agent = Agent[None, AnatomicQueryTerms](
			model=get_openai_model(model),
			output_type=AnatomicQueryTerms,
			system_prompt="..."
	)
	```
- Use `await agent.run(prompt)` for async agent calls.
- Use `deps_type` and `RunContext` for agents needing external dependencies.
- Define clear, domain-specific system prompts.

## Testing Pydantic AI Agents
- Use `pytest` as the test runner, with `pytest.mark.asyncio` or `pytest.mark.anyio` for async tests.
- Set `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False` globally in tests to block real LLM calls.
- Use `TestModel` from `pydantic_ai.models.test` to replace the agent's model in tests:
	```python
	from pydantic_ai.models.test import TestModel
	with agent.override(model=TestModel()):
			result = await agent.run("test prompt")
			assert result.output is not None
	```
- Use `FunctionModel` for custom tool call logic in tests.
- Use `override` context manager to swap agent models during tests.
- Use pytest fixtures to apply model overrides across multiple tests.
- Use `capture_run_messages()` to inspect agent/model message exchanges.
- Use `patch` and `AsyncMock` to mock dependencies or agent methods as needed.

## Example: Async Agent Test
```python
import pytest
from pydantic_ai.models.test import TestModel
from your_module import your_agent

@pytest.mark.asyncio
async def test_agent_behavior():
		with your_agent.override(model=TestModel()):
				result = await your_agent.run("test prompt")
				assert result.output is not None
```

## Patterns from Existing Code
- Both ontology search and anatomic location search use async agents, structured output types, and are tested with `pytest.mark.asyncio` and `TestModel`.
- Tests use `patch` to replace model selection and agent methods, and `AsyncMock` for async dependencies.
- Fallbacks are provided in agent logic for error cases, and tests assert correct fallback behavior.

## Rationale
- Simplicity and maintainability are prioritized.
- No multi-agent validation, advanced error handling, or performance optimizations beyond what is already present in the codebase.