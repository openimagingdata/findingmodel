# Pydantic AI Testing Best Practices

## Core Philosophy
Test actual code behavior and workflow logic, NOT library implementation details. Don't test that Pydantic AI agents have a `run` method or that Pydantic models validate - test YOUR code's logic.

## Essential Patterns

### 1. Prevent Accidental API Calls
Always add at module level in test files:
```python
from pydantic_ai import models
models.ALLOW_MODEL_REQUESTS = False
```

### 2. Use TestModel for Simple Testing
For deterministic agent testing with controlled responses:
```python
from pydantic_ai.models.test import TestModel

async def test_agent_behavior():
    agent = create_my_agent()
    response = MyResponse(field="value")
    
    with agent.override(model=TestModel(custom_output_args=response)):
        result = await agent.run("test prompt")
        assert result.output == response
```

### 3. Use FunctionModel for Complex Behavior
For testing complex workflows with controlled logic:
```python
from pydantic_ai.models.function import FunctionModel

async def test_complex_workflow():
    def controlled_behavior(messages, info):
        # Return controlled response based on input
        if "error" in messages[-1].content:
            raise Exception("Controlled error")
        return MyResponse(...)
    
    with agent.override(model=FunctionModel(controlled_behavior)):
        result = await agent.run("test prompt")
```

### 4. Integration Test Pattern
For tests that need real API calls:
```python
@pytest.mark.callout
async def test_integration():
    # Temporarily enable for this test only
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    
    try:
        result = await real_api_call()
        assert result.is_valid()
    finally:
        # Always restore original setting
        models.ALLOW_MODEL_REQUESTS = original
```

## Anti-Patterns to Avoid

### ❌ Testing Library Functionality
```python
# BAD - tests Pydantic AI, not your code
def test_agent_has_run():
    agent = create_agent()
    assert hasattr(agent, 'run')
```

### ❌ Over-Mocking Everything
```python
# BAD - too much mocking obscures actual behavior
with patch('module.create_agent') as mock:
    mock.return_value.run = AsyncMock()
    # Lost track of what we're actually testing
```

### ❌ Testing Basic Validation
```python
# BAD - tests Pydantic, not your logic
def test_model_validation():
    with pytest.raises(ValidationError):
        MyModel(invalid_field="value")
```

## Good Patterns

### ✅ Test Configuration
```python
# GOOD - tests your agent setup
async def test_agent_configuration():
    agent = create_search_agent("gpt-4")
    with agent.override(model=TestModel()):
        assert my_tool in agent.tools
        assert "expected prompt" in agent.system_prompt
```

### ✅ Test Workflow Logic
```python
# GOOD - tests actual workflow behavior
async def test_two_agent_workflow():
    search_response = SearchResults(...)
    match_response = MatchedItems(...)
    
    with (
        search_agent.override(model=TestModel(custom_output_args=search_response)),
        match_agent.override(model=TestModel(custom_output_args=match_response)),
    ):
        result = await my_workflow()
        assert result.primary_item is not None
```

### ✅ Test Error Handling
```python
# GOOD - tests cleanup and error propagation
async def test_cleanup_on_error():
    mock_connection = MagicMock()
    
    with patch('module.connect', return_value=mock_connection):
        with pytest.raises(ProcessingError):
            await process_with_connection()
        
        # Verify cleanup happened
        mock_connection.disconnect.assert_called_once()
```

## Project-Specific Conventions

- Use `@pytest.mark.callout` for tests requiring external services
- Demo scripts in `notebooks/` with `demo_*.py` naming
- Consolidate related component tests together
- Add `testpaths = ["test"]` to pyproject.toml to restrict pytest scope

## Key Takeaways

1. **Focus on behavior, not implementation**
2. **Use Pydantic AI's testing utilities (TestModel, FunctionModel)**
3. **Prevent accidental API calls by default**
4. **Test workflows and error handling**
5. **Keep tests simple and meaningful**