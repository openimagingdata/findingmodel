"""Tests for the DuckDB-backed index implementation (read-only operations).

NOTE: Write operations (setup, add_or_update, update_from_directory, remove_entry)
are now tested in the oidm-maintenance package. This file only tests read operations
using a pre-built test database.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from findingmodel.finding_model import FindingModelBase, FindingModelFull
from findingmodel.index import DuckDBIndex, IndexEntry

# ============================================================================
# Fixtures Overview
# ============================================================================
#
# From conftest.py:
# - base_model: FindingModelBase (function-scoped)
# - full_model: FindingModelFull (function-scoped)
# - prebuilt_db_path: Path (session-scoped, path to pre-built test database)
#
# From this file:
# - index: Pre-built DuckDBIndex (function-scoped, read-only)
#   Use for: All tests that read from a populated index
#
# ============================================================================


@pytest_asyncio.fixture
async def index(prebuilt_db_path: Path) -> AsyncGenerator[DuckDBIndex, None]:
    """Load the pre-built test database (read-only)."""
    async with DuckDBIndex(prebuilt_db_path) as idx:
        yield idx


OIFM_IDS_IN_DEFS_DIR = [
    "OIFM_MSFT_573630",
    "OIFM_MSFT_356221",
    "OIFM_MSFT_156954",
    "OIFM_MSFT_367670",
    "OIFM_MSFT_932618",
    "OIFM_MSFT_134126",
]


# ============================================================================
# Basic Retrieval Tests
# ============================================================================


@pytest.mark.asyncio
async def test_count_method(index: DuckDBIndex) -> None:
    """Test count returns correct number of models."""
    count = await index.count()
    # Should be at least as many as the number of *.fm.json files in test/data/defs
    assert count >= len(OIFM_IDS_IN_DEFS_DIR)


@pytest.mark.asyncio
async def test_get_by_id(index: DuckDBIndex) -> None:
    """Test retrieving all models by ID from populated index."""
    for oifm_id in OIFM_IDS_IN_DEFS_DIR:
        entry = await index.get(oifm_id)
        assert entry is not None
        assert entry.oifm_id == oifm_id


@pytest.mark.asyncio
async def test_get_by_name(index: DuckDBIndex) -> None:
    """Test retrieving a model by name."""
    entry = await index.get("abdominal aortic aneurysm")
    assert entry is not None
    assert "abdominal aortic aneurysm" in entry.name.lower()


@pytest.mark.asyncio
async def test_get_full(index: DuckDBIndex) -> None:
    """Test retrieving full model data."""
    full = await index.get_full(OIFM_IDS_IN_DEFS_DIR[0])
    assert full is not None
    assert isinstance(full, FindingModelFull)
    assert full.oifm_id == OIFM_IDS_IN_DEFS_DIR[0]


@pytest.mark.asyncio
async def test_get_full_batch(index: DuckDBIndex) -> None:
    """Test batch retrieval of full models."""
    ids_to_fetch = OIFM_IDS_IN_DEFS_DIR[:3]
    models = await index.get_full_batch(ids_to_fetch)
    assert len(models) == 3
    assert all(isinstance(m, FindingModelFull) for m in models.values())
    assert set(models.keys()) == set(ids_to_fetch)


@pytest.mark.asyncio
async def test_contains_method(index: DuckDBIndex) -> None:
    """Test the contains method with ID and name lookups."""
    # Should exist by ID
    assert await index.contains(OIFM_IDS_IN_DEFS_DIR[0]) is True
    # Should exist by name
    assert await index.contains("abdominal aortic aneurysm") is True
    # Should not exist
    assert await index.contains("OIFM_NONEXISTENT_999999") is False
    assert await index.contains("nonexistent finding name") is False


# ============================================================================
# Pagination Tests
# ============================================================================


@pytest.mark.asyncio
async def test_all_pagination(index: DuckDBIndex) -> None:
    """Test paginated retrieval of all entries."""
    # Get first page
    page1, total1 = await index.all(limit=2, offset=0)
    assert len(page1) == 2
    assert total1 >= 6  # At least the 6 in OIFM_IDS_IN_DEFS_DIR

    # Get second page
    page2, total2 = await index.all(limit=2, offset=2)
    assert len(page2) == 2
    assert total2 == total1  # Total count should be same

    # Pages should be different
    page1_ids = {e.oifm_id for e in page1}
    page2_ids = {e.oifm_id for e in page2}
    assert page1_ids.isdisjoint(page2_ids)


@pytest.mark.asyncio
async def test_all_sorting_by_name(index: DuckDBIndex) -> None:
    """Test sorting all entries by name."""
    # Sort ascending (case-insensitive)
    entries_asc, _ = await index.all(order_by="name", order_dir="asc", limit=10)
    names_asc = [e.name for e in entries_asc]
    assert names_asc == sorted(names_asc, key=str.lower)

    # Sort descending (case-insensitive)
    entries_desc, _ = await index.all(order_by="name", order_dir="desc", limit=10)
    names_desc = [e.name for e in entries_desc]
    assert names_desc == sorted(names_desc, key=str.lower, reverse=True)


@pytest.mark.asyncio
async def test_all_sorting_by_id(index: DuckDBIndex) -> None:
    """Test sorting all entries by OIFM ID."""
    entries, _ = await index.all(order_by="oifm_id", order_dir="asc", limit=10)
    ids = [e.oifm_id for e in entries]
    assert ids == sorted(ids)


@pytest.mark.asyncio
async def test_all_case_insensitive_sorting(index: DuckDBIndex) -> None:
    """Test that sorting is case-insensitive."""
    entries, _ = await index.all(order_by="name", order_dir="asc")
    names = [e.name for e in entries]
    # Case-insensitive sorted order
    expected = sorted(names, key=str.lower)
    assert names == expected


@pytest.mark.asyncio
async def test_all_invalid_order_by(index: DuckDBIndex) -> None:
    """Test that invalid order_by raises ValueError."""
    with pytest.raises(ValueError, match="Invalid order_by"):
        await index.all(order_by="invalid_field")


@pytest.mark.asyncio
async def test_all_invalid_order_dir(index: DuckDBIndex) -> None:
    """Test that invalid order_dir raises ValueError."""
    with pytest.raises(ValueError, match="Invalid order_dir"):
        await index.all(order_dir="invalid")


@pytest.mark.asyncio
async def test_all_single_page(index: DuckDBIndex) -> None:
    """Test retrieving all entries in a single page."""
    count = await index.count()
    entries, total = await index.all(limit=count)
    assert len(entries) == count
    assert total == count


# ============================================================================
# Search Functionality Tests
# ============================================================================


@pytest.mark.asyncio
async def test_search_basic_functionality(index: DuckDBIndex) -> None:
    """Test basic search functionality with populated index."""
    # Search for "aneurysm" should find abdominal aortic aneurysm
    results = await index.search("aneurysm", limit=10)
    assert len(results) >= 1

    # Check that we get IndexEntry objects back
    assert all(isinstance(result, IndexEntry) for result in results)

    # Should find the abdominal aortic aneurysm model
    aneurysm_results = [r for r in results if "aneurysm" in r.name.lower()]
    assert len(aneurysm_results) >= 1


@pytest.mark.asyncio
async def test_search_by_name(index: DuckDBIndex) -> None:
    """Test search functionality by exact and partial name matches."""
    # Exact name search
    results = await index.search("abdominal aortic aneurysm")
    assert len(results) >= 1
    assert any("abdominal aortic aneurysm" in r.name.lower() for r in results)

    # Partial name search
    results = await index.search("aortic")
    assert len(results) >= 1
    assert any("aortic" in r.name.lower() for r in results)


@pytest.mark.asyncio
async def test_search_by_description(index: DuckDBIndex) -> None:
    """Test search functionality using description content."""
    # Search for terms that should appear in descriptions
    results = await index.search("dilation")
    assert isinstance(results, list)

    # Search for medical terms
    results = await index.search("diameter")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_by_synonyms(index: DuckDBIndex) -> None:
    """Test search functionality using synonyms."""
    # Search for "AAA" which should be a synonym for abdominal aortic aneurysm
    results = await index.search("AAA")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_limit_parameter(index: DuckDBIndex) -> None:
    """Test that search respects the limit parameter."""
    # Search with different limits
    results_limit_1 = await index.search("aneurysm", limit=1)
    results_limit_5 = await index.search("aneurysm", limit=5)

    assert len(results_limit_1) <= 1
    assert len(results_limit_5) <= 5

    # If there are results, limit should work correctly
    if results_limit_5:
        assert len(results_limit_1) <= len(results_limit_5)


@pytest.mark.asyncio
async def test_search_no_results(index: DuckDBIndex) -> None:
    """Test search with query that should return no results."""
    results = await index.search("zyxwvutsrqponmlkjihgfedcba")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_empty_query(index: DuckDBIndex) -> None:
    """Test search behavior with empty query."""
    results = await index.search("", limit=5)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_case_insensitive(index: DuckDBIndex) -> None:
    """Test that search is case insensitive."""
    results_lower = await index.search("aneurysm")
    results_upper = await index.search("ANEURYSM")
    results_mixed = await index.search("Aneurysm")

    # Should get same results regardless of case
    assert len(results_lower) == len(results_upper) == len(results_mixed)


@pytest.mark.asyncio
async def test_search_multiple_terms(index: DuckDBIndex) -> None:
    """Test search with multiple terms."""
    # Search for multiple terms
    results = await index.search("abdominal aortic")
    assert isinstance(results, list)

    # Should potentially find models containing either term
    if results:
        found_text = " ".join([r.name + " " + (r.description or "") for r in results]).lower()
        # At least one term should be found
        assert "abdominal" in found_text or "aortic" in found_text


@pytest.mark.asyncio
async def test_search_with_single_tag(index: DuckDBIndex) -> None:
    """Test search with tag filter."""
    # Get all entries to find a common tag
    all_entries, _ = await index.all(limit=100)
    if not all_entries:
        pytest.skip("No entries in index")

    # Find an entry with tags
    entry_with_tags = None
    for entry in all_entries:
        if entry.tags:
            entry_with_tags = entry
            break

    if not entry_with_tags:
        pytest.skip("No entries with tags found")

    # Search with one of its tags
    tag = entry_with_tags.tags[0]
    results = await index.search("", tags=[tag], limit=10)

    # All results should have the tag
    for result in results:
        assert tag in result.tags


@pytest.mark.asyncio
async def test_search_with_multiple_tags_and_logic(index: DuckDBIndex) -> None:
    """Test search with multiple tags and AND logic."""
    # Get all entries to find entries with multiple tags
    all_entries, _ = await index.all(limit=100)

    # Find an entry with at least 2 tags
    entry_with_multi_tags = None
    for entry in all_entries:
        if entry.tags and len(entry.tags) >= 2:
            entry_with_multi_tags = entry
            break

    if not entry_with_multi_tags:
        pytest.skip("No entries with multiple tags found")

    # Search with both tags using AND logic
    tags = entry_with_multi_tags.tags[:2]
    results = await index.search("", tags=tags, tag_logic="and", limit=10)

    # All results should have both tags
    for result in results:
        assert all(tag in result.tags for tag in tags)


@pytest.mark.asyncio
async def test_search_with_nonexistent_tags(index: DuckDBIndex) -> None:
    """Test search with tags that don't exist."""
    results = await index.search("", tags=["nonexistent_tag_12345"], limit=10)
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_batch_multiple_queries(index: DuckDBIndex) -> None:
    """Test batch search with multiple queries."""
    queries = ["aneurysm", "embolism", "fracture"]
    results = await index.search_batch(queries, limit=5)

    assert isinstance(results, dict)
    assert len(results) == 3
    assert all(isinstance(results[q], list) for q in queries)


@pytest.mark.asyncio
async def test_search_batch_empty_queries_list(index: DuckDBIndex) -> None:
    """Test batch search with empty list."""
    results = await index.search_batch([], limit=5)
    assert results == {}


# ============================================================================
# Slug Search Tests
# ============================================================================


@pytest.mark.asyncio
async def test_search_by_slug_exact_match(index: DuckDBIndex) -> None:
    """Test slug search with exact match."""
    results, _total = await index.search_by_slug("abdominal_aortic_aneurysm", limit=10)
    assert len(results) >= 1
    # First result should be exact match
    assert "abdominal_aortic_aneurysm" in results[0].slug_name


@pytest.mark.asyncio
async def test_search_by_slug_prefix_match(index: DuckDBIndex) -> None:
    """Test slug search with prefix match."""
    results, _total = await index.search_by_slug("abdominal_", limit=10)
    assert len(results) >= 1
    assert any(r.slug_name.startswith("abdominal_") for r in results)


@pytest.mark.asyncio
async def test_search_by_slug_contains_match(index: DuckDBIndex) -> None:
    """Test slug search with contains match."""
    results, _total = await index.search_by_slug("aortic", limit=10)
    assert len(results) >= 1
    assert any("aortic" in r.slug_name for r in results)


@pytest.mark.asyncio
async def test_search_by_slug_pattern_normalization(index: DuckDBIndex) -> None:
    """Test slug search handles pattern normalization."""
    # Search with spaces should work the same as hyphens
    results_with_spaces, _ = await index.search_by_slug("abdominal aortic", limit=10)
    results_with_hyphens, _ = await index.search_by_slug("abdominal-aortic", limit=10)

    # Should get same or similar results
    assert len(results_with_spaces) == len(results_with_hyphens)


@pytest.mark.asyncio
async def test_search_by_slug_pagination(index: DuckDBIndex) -> None:
    """Test slug search pagination."""
    # Get first page - pattern must be at least 3 characters
    page1, _total1 = await index.search_by_slug("abdominal", limit=2, offset=0)
    # Get second page
    page2, _total2 = await index.search_by_slug("abdominal", limit=2, offset=2)

    if len(page1) == 2 and len(page2) == 2:
        # Pages should be different
        page1_ids = {e.oifm_id for e in page1}
        page2_ids = {e.oifm_id for e in page2}
        assert page1_ids.isdisjoint(page2_ids)


@pytest.mark.asyncio
async def test_search_by_slug_no_matches(index: DuckDBIndex) -> None:
    """Test slug search with no matches."""
    results, _total = await index.search_by_slug("zyxwvutsrqponmlkjihgfedcba", limit=10)
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_count_search_exact_match(index: DuckDBIndex) -> None:
    """Test count_search with exact name match."""
    count = await index.count_search(pattern="abdominal-aortic-aneurysm", match_type="exact")
    assert count >= 1


@pytest.mark.asyncio
async def test_count_search_prefix_match(index: DuckDBIndex) -> None:
    """Test count_search with prefix match."""
    count = await index.count_search(pattern="abdominal", match_type="prefix")
    assert count >= 1


@pytest.mark.asyncio
async def test_count_search_contains_match(index: DuckDBIndex) -> None:
    """Test count_search with contains match."""
    count = await index.count_search(pattern="aortic", match_type="contains")
    assert count >= 1


# ============================================================================
# Contributor Management Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_people(index: DuckDBIndex) -> None:
    """Test retrieving all people from the index."""
    people = await index.get_people()
    # Pre-built test database has at least 1 person
    assert len(people) >= 1
    # Should be sorted by name
    names = [p.name for p in people]
    assert names == sorted(names)


@pytest.mark.asyncio
async def test_get_organizations(index: DuckDBIndex) -> None:
    """Test retrieving all organizations from the index."""
    orgs = await index.get_organizations()
    # Pre-built test database has at least 1 organization
    assert len(orgs) >= 1
    # Should be sorted by name
    names = [o.name for o in orgs]
    assert names == sorted(names)


@pytest.mark.asyncio
async def test_count_people(index: DuckDBIndex) -> None:
    """Test counting people in the index."""
    count = await index.count_people()
    # Pre-built test database has at least 1 person
    assert count >= 1


@pytest.mark.asyncio
async def test_count_organizations(index: DuckDBIndex) -> None:
    """Test counting organizations in the index."""
    count = await index.count_organizations()
    # Pre-built test database has at least 1 organization
    assert count >= 1


# ============================================================================
# ID Generation Tests
# ============================================================================


def test_generate_model_id_format(index: DuckDBIndex) -> None:
    """Test that generated model IDs follow the correct format."""
    model_id = index.generate_model_id(source="TEST")
    assert model_id.startswith("OIFM_TEST_")
    assert len(model_id) == 16  # OIFM_ + 4 chars + _ + 6 digits


def test_generate_model_id_uniqueness(index: DuckDBIndex) -> None:
    """Test that generated IDs are unique."""
    id1 = index.generate_model_id(source="TEST")
    id2 = index.generate_model_id(source="TEST")
    assert id1 != id2


def test_generate_model_id_different_sources_independent(index: DuckDBIndex) -> None:
    """Test that IDs from different sources are independent."""
    id_msft = index.generate_model_id(source="MSFT")
    id_test = index.generate_model_id(source="TEST")

    assert "MSFT" in id_msft
    assert "TEST" in id_test


def test_generate_model_id_cache_prevents_self_collision(index: DuckDBIndex) -> None:
    """Test that the ID cache prevents self-collision within a session."""
    # Generate many IDs in quick succession
    ids = {index.generate_model_id(source="TEST") for _ in range(100)}

    # All should be unique
    assert len(ids) == 100


def test_generate_model_id_invalid_source_too_short(index: DuckDBIndex) -> None:
    """Test that source must be at least 3 characters."""
    with pytest.raises(ValueError, match="Source must be 3-4 uppercase letters"):
        index.generate_model_id(source="AB")


def test_generate_model_id_invalid_source_too_long(index: DuckDBIndex) -> None:
    """Test that source must be at most 4 characters."""
    with pytest.raises(ValueError, match="Source must be 3-4 uppercase letters"):
        index.generate_model_id(source="TOOLONG")


def test_generate_model_id_invalid_source_contains_digits(index: DuckDBIndex) -> None:
    """Test that source cannot contain digits."""
    with pytest.raises(ValueError, match="Source must be 3-4 uppercase letters"):
        index.generate_model_id(source="TE5T")


def test_generate_model_id_source_normalization(index: DuckDBIndex) -> None:
    """Test that source is normalized to uppercase."""
    # Lowercase should be normalized
    id_lower = index.generate_model_id(source="test")
    assert "TEST" in id_lower

    # Mixed case should be normalized
    id_mixed = index.generate_model_id(source="TeSt")
    assert "TEST" in id_mixed


def test_generate_attribute_id_format(index: DuckDBIndex) -> None:
    """Test that generated attribute IDs follow the correct format."""
    attr_id = index.generate_attribute_id(source="TEST")
    assert attr_id.startswith("OIFMA_TEST_")
    assert len(attr_id) == 17  # OIFMA_ + 4 chars + _ + 6 digits


def test_generate_attribute_id_uniqueness(index: DuckDBIndex) -> None:
    """Test that generated attribute IDs are unique."""
    id1 = index.generate_attribute_id(source="TEST")
    id2 = index.generate_attribute_id(source="TEST")
    assert id1 != id2


def test_generate_attribute_id_independent_from_oifm_ids(index: DuckDBIndex) -> None:
    """Test that attribute IDs are independent from model IDs."""
    # Generate some model IDs
    model_id = index.generate_model_id(source="TEST")

    # Generate attribute ID - should not conflict
    attr_id = index.generate_attribute_id(source="TEST")

    # IDs should be different
    assert model_id != attr_id
    assert model_id.startswith("OIFM_")
    assert attr_id.startswith("OIFMA_")


def test_generate_attribute_id_infer_source_from_model_id(index: DuckDBIndex) -> None:
    """Test that source can be inferred from model ID."""
    attr_id = index.generate_attribute_id(model_oifm_id="OIFM_TEST_123456")
    assert "TEST" in attr_id


def test_generate_attribute_id_explicit_source_overrides_inference(index: DuckDBIndex) -> None:
    """Test that explicit source overrides inferred source."""
    attr_id = index.generate_attribute_id(model_oifm_id="OIFM_MSFT_123456", source="TEST")
    assert "TEST" in attr_id
    assert "MSFT" not in attr_id


def test_generate_attribute_id_default_source(index: DuckDBIndex) -> None:
    """Test that default source is used when no source is provided."""
    attr_id = index.generate_attribute_id()
    # Should use default source from settings
    assert attr_id.startswith("OIFMA_")


def test_generate_attribute_id_invalid_model_id_format(index: DuckDBIndex) -> None:
    """Test that invalid model ID format raises error."""
    with pytest.raises(ValueError, match="Cannot infer source from invalid model ID"):
        index.generate_attribute_id(model_oifm_id="INVALID_ID")


def test_generate_attribute_id_invalid_source(index: DuckDBIndex) -> None:
    """Test that invalid source raises error."""
    with pytest.raises(ValueError, match="Source must be 3-4 uppercase letters"):
        index.generate_attribute_id(source="AB")


def test_generate_attribute_id_cache_prevents_self_collision(index: DuckDBIndex) -> None:
    """Test that the ID cache prevents self-collision within a session."""
    # Generate many IDs in quick succession
    ids = {index.generate_attribute_id(source="TEST") for _ in range(100)}

    # All should be unique
    assert len(ids) == 100


# ============================================================================
# Model ID Processing Tests
# ============================================================================


def test_add_ids_to_model_complete_new_model(index: DuckDBIndex, base_model: FindingModelBase) -> None:
    """Test add_ids_to_model with a complete new model (no IDs)."""
    # Model should have no IDs initially
    assert not hasattr(base_model, "oifm_id")

    # Add IDs
    full_model = index.add_ids_to_model(base_model, source="TEST")

    # Should have OIFM ID
    assert full_model.oifm_id.startswith("OIFM_TEST_")

    # All attributes should have IDs
    for attr in full_model.attributes:
        assert attr.oifma_id.startswith("OIFMA_TEST_")


def test_add_ids_to_model_existing_oifm_id(index: DuckDBIndex, full_model: FindingModelFull) -> None:
    """Test add_ids_to_model preserves existing OIFM ID."""
    original_id = full_model.oifm_id

    # Process model
    processed = index.add_ids_to_model(full_model, source="TEST")

    # Should preserve original OIFM ID
    assert processed.oifm_id == original_id


def test_add_ids_to_model_multiple_attributes(index: DuckDBIndex) -> None:
    """Test add_ids_to_model assigns unique IDs to multiple attributes."""
    from findingmodel.finding_model import NumericAttribute

    model = FindingModelBase(
        name="Test Model",
        description="Test description",
        attributes=[
            NumericAttribute(
                name="Attr One",
                description="First attribute",
            ),
            NumericAttribute(
                name="Attr Two",
                description="Second attribute",
            ),
        ],
    )

    full_model = index.add_ids_to_model(model, source="TEST")

    # Both attributes should get unique IDs
    attr_ids = [attr.oifma_id for attr in full_model.attributes]
    assert len(attr_ids) == 2
    assert all(id.startswith("OIFMA_TEST_") for id in attr_ids)
    assert attr_ids[0] != attr_ids[1]  # IDs should be unique


def test_add_ids_to_model_all_ids_present(index: DuckDBIndex, full_model: FindingModelFull) -> None:
    """Test add_ids_to_model when all IDs are already present."""
    original_oifm_id = full_model.oifm_id
    original_attr_ids = [attr.oifma_id for attr in full_model.attributes]

    # Process model
    processed = index.add_ids_to_model(full_model, source="TEST")

    # All IDs should be preserved
    assert processed.oifm_id == original_oifm_id
    assert [attr.oifma_id for attr in processed.attributes] == original_attr_ids


def test_add_ids_to_model_source_used(index: DuckDBIndex, base_model: FindingModelBase) -> None:
    """Test that the specified source is used for generated IDs."""
    full_model = index.add_ids_to_model(base_model, source="ABCD")

    assert "ABCD" in full_model.oifm_id
    for attr in full_model.attributes:
        assert "ABCD" in attr.oifma_id


def test_add_ids_to_model_invalid_source_too_short(index: DuckDBIndex, base_model: FindingModelBase) -> None:
    """Test that invalid source raises error."""
    with pytest.raises(ValueError, match="Source must be 3-4 uppercase letters"):
        index.add_ids_to_model(base_model, source="AB")


def test_add_ids_to_model_invalid_source_too_long(index: DuckDBIndex, base_model: FindingModelBase) -> None:
    """Test that invalid source raises error."""
    with pytest.raises(ValueError, match="Source must be 3-4 uppercase letters"):
        index.add_ids_to_model(base_model, source="TOOLONG")


def test_add_ids_to_model_invalid_source_contains_digits(index: DuckDBIndex, base_model: FindingModelBase) -> None:
    """Test that invalid source raises error."""
    with pytest.raises(ValueError, match="Source must be 3-4 uppercase letters"):
        index.add_ids_to_model(base_model, source="TE5T")


def test_finalize_placeholder_single_placeholder(index: DuckDBIndex) -> None:
    """Test finalize_placeholder_attribute_ids with single placeholder."""
    from findingmodel.finding_model import NumericAttributeIded
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    model = FindingModelFull(
        oifm_id="OIFM_TEST_123456",
        name="Test Model",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Placeholder Attr",
            ),
        ],
    )

    finalized = index.finalize_placeholder_attribute_ids(model)

    # Placeholder should be replaced
    assert finalized.attributes[0].oifma_id != PLACEHOLDER_ATTRIBUTE_ID
    assert finalized.attributes[0].oifma_id.startswith("OIFMA_TEST_")


def test_finalize_placeholder_multiple_placeholders(index: DuckDBIndex) -> None:
    """Test finalize_placeholder_attribute_ids with multiple placeholders."""
    from findingmodel.finding_model import NumericAttributeIded
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    model = FindingModelFull(
        oifm_id="OIFM_TEST_123456",
        name="Test Model",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Placeholder 1",
            ),
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Placeholder 2",
            ),
        ],
    )

    finalized = index.finalize_placeholder_attribute_ids(model)

    # Both placeholders should be replaced
    assert finalized.attributes[0].oifma_id != PLACEHOLDER_ATTRIBUTE_ID
    assert finalized.attributes[1].oifma_id != PLACEHOLDER_ATTRIBUTE_ID

    # IDs should be unique
    assert finalized.attributes[0].oifma_id != finalized.attributes[1].oifma_id


def test_finalize_placeholder_no_placeholders(index: DuckDBIndex, full_model: FindingModelFull) -> None:
    """Test finalize_placeholder_attribute_ids when no placeholders exist."""
    original_attr_ids = [attr.oifma_id for attr in full_model.attributes]

    finalized = index.finalize_placeholder_attribute_ids(full_model)

    # IDs should remain unchanged
    assert [attr.oifma_id for attr in finalized.attributes] == original_attr_ids


def test_finalize_placeholder_source_inference(index: DuckDBIndex) -> None:
    """Test that source is inferred from model ID."""
    from findingmodel.finding_model import NumericAttributeIded
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    model = FindingModelFull(
        oifm_id="OIFM_ABCD_123456",
        name="Test Model",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Placeholder",
            ),
        ],
    )

    finalized = index.finalize_placeholder_attribute_ids(model)

    # Should use ABCD as source
    assert "ABCD" in finalized.attributes[0].oifma_id


def test_finalize_placeholder_explicit_source_override(index: DuckDBIndex) -> None:
    """Test that explicit source overrides inferred source."""
    from findingmodel.finding_model import NumericAttributeIded
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    model = FindingModelFull(
        oifm_id="OIFM_MSFT_123456",
        name="Test Model",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Placeholder",
            ),
        ],
    )

    finalized = index.finalize_placeholder_attribute_ids(model, source="TEST")

    # Should use TEST as source, not MSFT
    assert "TEST" in finalized.attributes[0].oifma_id
    assert "MSFT" not in finalized.attributes[0].oifma_id


def test_finalize_placeholder_choice_value_codes(index: DuckDBIndex) -> None:
    """Test that choice value codes are updated when placeholder is replaced."""
    from findingmodel.finding_model import ChoiceAttributeIded, ChoiceValueIded
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    model = FindingModelFull(
        oifm_id="OIFM_TEST_123456",
        name="Test Model",
        description="Test description",
        attributes=[
            ChoiceAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Placeholder",
                values=[
                    ChoiceValueIded(value_code=f"{PLACEHOLDER_ATTRIBUTE_ID}.0", name="Value 1"),
                    ChoiceValueIded(value_code=f"{PLACEHOLDER_ATTRIBUTE_ID}.1", name="Value 2"),
                ],
            ),
        ],
    )

    finalized = index.finalize_placeholder_attribute_ids(model)

    new_attr_id = finalized.attributes[0].oifma_id

    # Value codes should use the new attribute ID
    assert finalized.attributes[0].values[0].value_code == f"{new_attr_id}.0"
    assert finalized.attributes[0].values[1].value_code == f"{new_attr_id}.1"


def test_finalize_placeholder_invalid_source(index: DuckDBIndex) -> None:
    """Test that invalid source raises error."""
    from findingmodel.finding_model import NumericAttributeIded
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    model = FindingModelFull(
        oifm_id="OIFM_TEST_123456",
        name="Test Model",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Placeholder",
            ),
        ],
    )

    with pytest.raises(ValueError, match="Source must be 3-4 uppercase letters"):
        index.finalize_placeholder_attribute_ids(model, source="AB")


def test_finalize_placeholder_invalid_model_id_inference(index: DuckDBIndex) -> None:
    """Test that FindingModelFull validates oifm_id format at creation.

    Note: FindingModelFull validates oifm_id against pattern '^OIFM_[A-Z]{3,4}_[0-9]{6}$'
    at model creation time, so we can't test finalize_placeholder_attribute_ids
    with an invalid model ID - Pydantic prevents it.
    """
    from pydantic import ValidationError

    # Verify that creating FindingModelFull with invalid ID raises ValidationError
    with pytest.raises(ValidationError, match="oifm_id"):
        FindingModelFull(
            oifm_id="INVALID_ID",
            name="Test Model",
            description="Test description",
            attributes=[],
        )
