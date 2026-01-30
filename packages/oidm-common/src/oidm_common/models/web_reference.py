"""Web reference model compatible with Tavily search results."""

from datetime import date
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, computed_field


class WebReference(BaseModel):
    """Reference to a web resource, compatible with Tavily search results.

    Can be created manually with just url+title, or populated from
    Tavily search results with additional metadata like content and published_date.

    Parallels IndexCode:
    - IndexCode: system + code + display
    - WebReference: domain + url + title (with optional extras)

    Sources:
    - Tavily API Reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
    """

    # Core fields (always present)
    url: str = Field(description="The URL of the web resource", min_length=1)
    title: str = Field(description="The title of the page/resource", min_length=1)

    # Optional descriptive fields
    description: str | None = Field(
        default=None, description="Brief description or summary (manually added or from meta description)"
    )

    # Tavily search result fields (optional - only present if from search)
    content: str | None = Field(default=None, description="AI-extracted relevant content from the page (Tavily)")
    published_date: str | None = Field(default=None, description="Publication date if available (Tavily)")

    # Metadata
    accessed_date: date | None = Field(default=None, description="When this reference was accessed/verified")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def domain(self) -> str:
        """Extract domain from URL (e.g., 'radiopaedia.org')."""
        parsed = urlparse(self.url)
        # Remove 'www.' prefix if present
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    @classmethod
    def from_tavily_result(cls, result: dict[str, Any]) -> "WebReference":
        """Create WebReference from a Tavily search result dict.

        Args:
            result: A single result from Tavily's response["results"]

        Returns:
            WebReference populated with available fields
        """
        return cls(
            url=result["url"],
            title=result["title"],
            content=result.get("content"),
            published_date=result.get("published_date"),
            accessed_date=date.today(),
        )

    def __str__(self) -> str:
        return f"{self.title} ({self.domain})"

    def __repr__(self) -> str:
        return f"WebReference(url={self.url!r}, title={self.title!r})"
