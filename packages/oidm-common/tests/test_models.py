"""Tests for oidm-common models (IndexCode and WebReference)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from oidm_common.models.index_code import IndexCode
from oidm_common.models.web_reference import WebReference
from pydantic import ValidationError


class TestIndexCode:
    """Tests for IndexCode model."""

    def test_index_code_creation_with_display(self) -> None:
        """Test creating an IndexCode with all fields."""
        code = IndexCode(system="SNOMED", code="123456", display="Example Finding")

        assert code.system == "SNOMED"
        assert code.code == "123456"
        assert code.display == "Example Finding"

    def test_index_code_creation_without_display(self) -> None:
        """Test creating an IndexCode without display name."""
        code = IndexCode(system="LOINC", code="12345-6")

        assert code.system == "LOINC"
        assert code.code == "12345-6"
        assert code.display is None

    def test_index_code_str_with_display(self) -> None:
        """Test __str__ method includes display when present."""
        code = IndexCode(system="RadLex", code="RID001", display="Anatomical Structure")

        assert str(code) == "RadLex RID001 Anatomical Structure"

    def test_index_code_str_without_display(self) -> None:
        """Test __str__ method excludes display when not present."""
        code = IndexCode(system="SNOMED", code="789012")

        assert str(code) == "SNOMED 789012"

    def test_index_code_system_validation_min_length(self) -> None:
        """Test system field validation requires minimum 3 characters."""
        with pytest.raises(ValidationError) as exc_info:
            IndexCode(system="AB", code="12345")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("system",) for error in errors)

    def test_index_code_code_validation_min_length(self) -> None:
        """Test code field validation requires minimum 2 characters."""
        with pytest.raises(ValidationError) as exc_info:
            IndexCode(system="SNOMED", code="1")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("code",) for error in errors)

    def test_index_code_display_validation_min_length(self) -> None:
        """Test display field validation requires minimum 3 characters when provided."""
        with pytest.raises(ValidationError) as exc_info:
            IndexCode(system="SNOMED", code="123", display="AB")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("display",) for error in errors)


class TestWebReference:
    """Tests for WebReference model."""

    def test_web_reference_creation_minimal(self) -> None:
        """Test creating a WebReference with only required fields."""
        ref = WebReference(url="https://example.com", title="Example Page")

        assert ref.url == "https://example.com"
        assert ref.title == "Example Page"
        assert ref.description is None
        assert ref.content is None
        assert ref.published_date is None
        assert ref.accessed_date is None

    def test_web_reference_creation_full(self) -> None:
        """Test creating a WebReference with all fields."""
        ref = WebReference(
            url="https://radiopaedia.org/articles/test",
            title="Test Article",
            description="A test description",
            content="Extracted content",
            published_date="2024-12-01",
            accessed_date=date(2025, 1, 5),
        )

        assert ref.url == "https://radiopaedia.org/articles/test"
        assert ref.title == "Test Article"
        assert ref.description == "A test description"
        assert ref.content == "Extracted content"
        assert ref.published_date == "2024-12-01"
        assert ref.accessed_date == date(2025, 1, 5)

    def test_web_reference_domain_property_basic(self) -> None:
        """Test domain property extracts domain from URL."""
        ref = WebReference(url="https://radiopaedia.org/articles/test", title="Test")

        assert ref.domain == "radiopaedia.org"

    def test_web_reference_domain_property_removes_www(self) -> None:
        """Test domain property removes www prefix."""
        ref = WebReference(url="https://www.example.com/page", title="Test")

        assert ref.domain == "example.com"

    def test_web_reference_domain_property_subdomain(self) -> None:
        """Test domain property handles subdomains correctly."""
        ref = WebReference(url="https://docs.example.com/page", title="Test")

        assert ref.domain == "docs.example.com"

    def test_web_reference_domain_property_uppercase(self) -> None:
        """Test domain property lowercases domain."""
        ref = WebReference(url="https://Example.COM/page", title="Test")

        assert ref.domain == "example.com"

    def test_web_reference_from_tavily_result(self, tavily_search_result: dict[str, Any]) -> None:
        """Test creating WebReference from Tavily search result."""
        ref = WebReference.from_tavily_result(tavily_search_result)

        assert ref.url == "https://example.com/article"
        assert ref.title == "Medical Imaging Reference"
        assert ref.content == "Extracted content about medical imaging findings"
        assert ref.published_date == "2024-06-15"
        assert ref.accessed_date == date.today()

    def test_web_reference_from_tavily_result_missing_optional_fields(self) -> None:
        """Test creating WebReference from Tavily result with missing optional fields."""
        minimal_result = {
            "url": "https://example.com",
            "title": "Minimal Result",
        }

        ref = WebReference.from_tavily_result(minimal_result)

        assert ref.url == "https://example.com"
        assert ref.title == "Minimal Result"
        assert ref.content is None
        assert ref.published_date is None
        assert ref.accessed_date == date.today()

    def test_web_reference_str_method(self) -> None:
        """Test __str__ method returns title and domain."""
        ref = WebReference(url="https://radiopaedia.org/articles/test", title="Test Article")

        assert str(ref) == "Test Article (radiopaedia.org)"

    def test_web_reference_repr_method(self) -> None:
        """Test __repr__ method returns constructor-style representation."""
        ref = WebReference(url="https://example.com", title="Test")

        assert repr(ref) == "WebReference(url='https://example.com', title='Test')"

    def test_web_reference_url_validation_min_length(self) -> None:
        """Test URL field validation requires non-empty string."""
        with pytest.raises(ValidationError) as exc_info:
            WebReference(url="", title="Test")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("url",) for error in errors)

    def test_web_reference_title_validation_min_length(self) -> None:
        """Test title field validation requires non-empty string."""
        with pytest.raises(ValidationError) as exc_info:
            WebReference(url="https://example.com", title="")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("title",) for error in errors)
