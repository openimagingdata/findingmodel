from findingmodel import main


def test_main_function_exists() -> None:
    """Test that findingmodel defines a main function."""
    assert callable(main), "main should be a callable function"
