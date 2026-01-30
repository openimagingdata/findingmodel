"""Tests for WebReference model."""

from datetime import date

from findingmodel import WebReference


class TestWebReference:
    """Tests for WebReference model."""

    def test_creation_with_required_fields(self) -> None:
        """Test creating WebReference with only required fields."""
        ref = WebReference(url="https://example.com/page", title="Example Page")

        assert ref.url == "https://example.com/page"
        assert ref.title == "Example Page"
        assert ref.description is None
        assert ref.content is None
        assert ref.published_date is None
        assert ref.accessed_date is None

    def test_creation_with_all_fields(self) -> None:
        """Test creating WebReference with all fields."""
        accessed = date(2025, 12, 20)
        ref = WebReference(
            url="https://radiopaedia.org/articles/pulmonary-embolism",
            title="Pulmonary embolism",
            description="Comprehensive article on PE",
            content="Pulmonary embolism is a blood clot in the lungs...",
            published_date="2024-01-15",
            accessed_date=accessed,
        )

        assert ref.url == "https://radiopaedia.org/articles/pulmonary-embolism"
        assert ref.title == "Pulmonary embolism"
        assert ref.description == "Comprehensive article on PE"
        assert ref.content == "Pulmonary embolism is a blood clot in the lungs..."
        assert ref.published_date == "2024-01-15"
        assert ref.accessed_date == accessed

    def test_computed_field_domain(self) -> None:
        """Test domain computed property extracts domain correctly."""
        ref = WebReference(url="https://www.radiopaedia.org/articles/test", title="Test")

        # Should remove www. prefix
        assert ref.domain == "radiopaedia.org"

    def test_domain_without_www(self) -> None:
        """Test domain extraction without www prefix."""
        ref = WebReference(url="https://example.com/page", title="Test")

        assert ref.domain == "example.com"

    def test_domain_with_subdomain(self) -> None:
        """Test domain extraction with non-www subdomain."""
        ref = WebReference(url="https://docs.example.com/page", title="Test")

        # Should preserve non-www subdomains
        assert ref.domain == "docs.example.com"

    def test_domain_case_insensitive(self) -> None:
        """Test domain is lowercased."""
        ref = WebReference(url="https://Example.COM/page", title="Test")

        assert ref.domain == "example.com"

    def test_from_tavily_result(self) -> None:
        """Test creating WebReference from Tavily search result dict."""
        tavily_result = {
            "url": "https://radiopaedia.org/articles/lung-cancer",
            "title": "Lung cancer",
            "content": "Lung cancer is a malignant tumor...",
            "published_date": "2023-05-10",
        }

        ref = WebReference.from_tavily_result(tavily_result)

        assert ref.url == "https://radiopaedia.org/articles/lung-cancer"
        assert ref.title == "Lung cancer"
        assert ref.content == "Lung cancer is a malignant tumor..."
        assert ref.published_date == "2023-05-10"
        assert ref.accessed_date == date.today()
        assert ref.description is None  # Not in Tavily result

    def test_from_tavily_result_minimal(self) -> None:
        """Test creating WebReference from minimal Tavily result."""
        tavily_result = {
            "url": "https://example.com",
            "title": "Example",
        }

        ref = WebReference.from_tavily_result(tavily_result)

        assert ref.url == "https://example.com"
        assert ref.title == "Example"
        assert ref.content is None
        assert ref.published_date is None
        assert ref.accessed_date == date.today()

    def test_string_representation(self) -> None:
        """Test __str__ method."""
        ref = WebReference(url="https://www.radiopaedia.org/test", title="Test Article")

        assert str(ref) == "Test Article (radiopaedia.org)"

    def test_repr_representation(self) -> None:
        """Test __repr__ method."""
        ref = WebReference(url="https://example.com", title="Test")

        assert repr(ref) == "WebReference(url='https://example.com', title='Test')"

    def test_model_dump_excludes_none(self) -> None:
        """Test that model_dump can exclude None values."""
        ref = WebReference(url="https://example.com", title="Test")

        dumped = ref.model_dump(exclude_none=True)

        assert "url" in dumped
        assert "title" in dumped
        assert "description" not in dumped
        assert "content" not in dumped
        assert "published_date" not in dumped
        assert "accessed_date" not in dumped

    def test_model_dump_includes_computed_field(self) -> None:
        """Test that model_dump includes computed field domain."""
        ref = WebReference(url="https://www.example.com", title="Test")

        dumped = ref.model_dump()

        assert dumped["domain"] == "example.com"

    def test_serialization_round_trip(self) -> None:
        """Test serialization and deserialization round trip."""
        original = WebReference(
            url="https://radiopaedia.org/test",
            title="Test Article",
            description="Test description",
            accessed_date=date(2025, 12, 20),
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        restored = WebReference.model_validate_json(json_str)

        assert restored.url == original.url
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.accessed_date == original.accessed_date
        assert restored.domain == original.domain
