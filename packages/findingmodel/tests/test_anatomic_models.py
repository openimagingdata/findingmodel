"""Tests for anatomic location models (enums, AnatomicRef, AnatomicLocation)."""

from unittest.mock import MagicMock

import pytest
from findingmodel.anatomic_location import (
    AnatomicLocation,
    AnatomicRef,
    AnatomicRegion,
    BodySystem,
    Laterality,
    LocationType,
    StructureType,
)
from findingmodel.index_code import IndexCode
from findingmodel.web_reference import WebReference

# =============================================================================
# AnatomicRef Tests
# =============================================================================


class TestAnatomicRef:
    """Tests for AnatomicRef model."""

    def test_creation(self) -> None:
        """Test creating AnatomicRef."""
        ref = AnatomicRef(id="RID1234", display="test location")

        assert ref.id == "RID1234"
        assert ref.display == "test location"

    def test_resolve(self) -> None:
        """Test resolve method returns location from index."""
        # Create mock index and location
        mock_index = MagicMock()
        mock_location = MagicMock(spec=AnatomicLocation)
        mock_index.get.return_value = mock_location

        ref = AnatomicRef(id="RID1234", display="test")

        result = ref.resolve(mock_index)

        assert result == mock_location
        mock_index.get.assert_called_once_with("RID1234")


# =============================================================================
# AnatomicLocation Tests
# =============================================================================


class TestAnatomicLocation:
    """Tests for AnatomicLocation model."""

    def test_creation_minimal(self) -> None:
        """Test creating AnatomicLocation with minimal fields."""
        location = AnatomicLocation(id="RID1234", description="test location")

        assert location.id == "RID1234"
        assert location.description == "test location"
        assert location.region is None
        assert location.location_type == LocationType.STRUCTURE
        assert location.body_system is None
        assert location.structure_type is None
        assert location.laterality == Laterality.NONLATERAL
        assert location.definition is None
        assert location.sex_specific is None
        assert location.synonyms == []
        assert location.codes == []
        assert location.references == []

    def test_creation_with_all_fields(self) -> None:
        """Test creating AnatomicLocation with all fields."""
        codes = [IndexCode(system="SNOMED", code="123456", display="Test")]
        refs = [WebReference(url="https://example.com", title="Test")]
        parent_ref = AnatomicRef(id="RID1000", display="parent")
        child_ref = AnatomicRef(id="RID2000", display="child")

        location = AnatomicLocation(
            id="RID1234",
            description="left lung",
            region=AnatomicRegion.THORAX,
            location_type=LocationType.STRUCTURE,
            body_system=BodySystem.RESPIRATORY,
            structure_type=StructureType.SOLID_ORGAN,
            laterality=Laterality.LEFT,
            definition="The left lung",
            sex_specific=None,
            synonyms=["left pulmonary organ"],
            codes=codes,
            references=refs,
            containment_path="/RID1/RID1000/RID1234/",
            containment_parent=parent_ref,
            containment_depth=2,
            containment_children=[child_ref],
            partof_path="/RID1/RID1234/",
            partof_parent=None,
            partof_depth=1,
            partof_children=[],
            left_variant=None,
            right_variant=AnatomicRef(id="RID1235", display="right lung"),
            generic_variant=AnatomicRef(id="RID1233", display="lung"),
        )

        assert location.id == "RID1234"
        assert location.description == "left lung"
        assert location.region == AnatomicRegion.THORAX
        assert location.location_type == LocationType.STRUCTURE
        assert location.body_system == BodySystem.RESPIRATORY
        assert location.structure_type == StructureType.SOLID_ORGAN
        assert location.laterality == Laterality.LEFT
        assert location.definition == "The left lung"
        assert location.synonyms == ["left pulmonary organ"]
        assert len(location.codes) == 1
        assert len(location.references) == 1
        assert location.containment_path == "/RID1/RID1000/RID1234/"
        assert location.containment_parent == parent_ref
        assert location.containment_depth == 2
        assert len(location.containment_children) == 1

    def test_computed_field_is_bilateral(self) -> None:
        """Test is_bilateral computed property."""
        generic_location = AnatomicLocation(id="RID1234", description="lung", laterality=Laterality.GENERIC)

        assert generic_location.is_bilateral is True

        left_location = AnatomicLocation(id="RID1235", description="left lung", laterality=Laterality.LEFT)

        assert left_location.is_bilateral is False

    def test_computed_field_is_lateralized(self) -> None:
        """Test is_lateralized computed property."""
        left_location = AnatomicLocation(id="RID1235", description="left lung", laterality=Laterality.LEFT)

        assert left_location.is_lateralized is True

        right_location = AnatomicLocation(id="RID1236", description="right lung", laterality=Laterality.RIGHT)

        assert right_location.is_lateralized is True

        generic_location = AnatomicLocation(id="RID1234", description="lung", laterality=Laterality.GENERIC)

        assert generic_location.is_lateralized is False

        nonlateral_location = AnatomicLocation(id="RID1237", description="heart", laterality=Laterality.NONLATERAL)

        assert nonlateral_location.is_lateralized is False

    def test_as_index_code(self) -> None:
        """Test as_index_code method."""
        location = AnatomicLocation(id="RID1234", description="lung")

        index_code = location.as_index_code()

        assert index_code.system == "anatomic_locations"
        assert index_code.code == "RID1234"
        assert index_code.display == "lung"

    def test_get_code_found(self) -> None:
        """Test get_code when code exists."""
        codes = [
            IndexCode(system="SNOMED", code="123456", display="Test"),
            IndexCode(system="FMA", code="7890", display="Test FMA"),
        ]
        location = AnatomicLocation(id="RID1234", description="test", codes=codes)

        snomed_code = location.get_code("SNOMED")

        assert snomed_code is not None
        assert snomed_code.system == "SNOMED"
        assert snomed_code.code == "123456"

    def test_get_code_case_insensitive(self) -> None:
        """Test get_code is case-insensitive."""
        codes = [IndexCode(system="SNOMED", code="123456", display="Test")]
        location = AnatomicLocation(id="RID1234", description="test", codes=codes)

        code = location.get_code("snomed")

        assert code is not None
        assert code.system == "SNOMED"

    def test_get_code_not_found(self) -> None:
        """Test get_code when code doesn't exist."""
        location = AnatomicLocation(id="RID1234", description="test", codes=[])

        code = location.get_code("SNOMED")

        assert code is None

    def test_is_contained_in_true(self) -> None:
        """Test is_contained_in returns True when ancestor in path."""
        location = AnatomicLocation(
            id="RID1234",
            description="test",
            containment_path="/RID1/RID100/RID1234/",
        )

        assert location.is_contained_in("RID1") is True
        assert location.is_contained_in("RID100") is True

    def test_is_contained_in_false(self) -> None:
        """Test is_contained_in returns False when ancestor not in path."""
        location = AnatomicLocation(
            id="RID1234",
            description="test",
            containment_path="/RID1/RID100/RID1234/",
        )

        assert location.is_contained_in("RID999") is False

    def test_is_contained_in_no_path(self) -> None:
        """Test is_contained_in returns False when no containment path."""
        location = AnatomicLocation(id="RID1234", description="test")

        assert location.is_contained_in("RID1") is False

    def test_is_part_of_true(self) -> None:
        """Test is_part_of returns True when ancestor in path."""
        location = AnatomicLocation(
            id="RID1234",
            description="test",
            partof_path="/RID1/RID100/RID1234/",
        )

        assert location.is_part_of("RID1") is True
        assert location.is_part_of("RID100") is True

    def test_is_part_of_false(self) -> None:
        """Test is_part_of returns False when ancestor not in path."""
        location = AnatomicLocation(
            id="RID1234",
            description="test",
            partof_path="/RID1/RID100/RID1234/",
        )

        assert location.is_part_of("RID999") is False

    def test_is_part_of_no_path(self) -> None:
        """Test is_part_of returns False when no part-of path."""
        location = AnatomicLocation(id="RID1234", description="test")

        assert location.is_part_of("RID1") is False

    def test_string_representation(self) -> None:
        """Test __str__ method."""
        location = AnatomicLocation(id="RID1234", description="lung")

        assert str(location) == "RID1234: lung"

    def test_repr_representation(self) -> None:
        """Test __repr__ method."""
        location = AnatomicLocation(id="RID1234", description="lung")

        assert repr(location) == "AnatomicLocation(id='RID1234', description='lung')"

    def test_bind_method(self) -> None:
        """Test bind method sets weakref to index."""
        mock_index = MagicMock()
        location = AnatomicLocation(id="RID1234", description="test")

        result = location.bind(mock_index)

        # Should return self for chaining
        assert result is location
        # Should have weakref set (we can't easily test weakref value)
        assert location._index is not None

    def test_get_index_from_parameter(self) -> None:
        """Test _get_index returns parameter when provided."""
        mock_index = MagicMock()
        location = AnatomicLocation(id="RID1234", description="test")

        result = location._get_index(mock_index)

        assert result == mock_index

    def test_get_index_from_bound(self) -> None:
        """Test _get_index returns bound index when parameter is None."""
        mock_index = MagicMock()
        location = AnatomicLocation(id="RID1234", description="test")
        location.bind(mock_index)

        result = location._get_index(None)

        assert result == mock_index

    def test_get_index_error_when_not_bound(self) -> None:
        """Test _get_index raises ValueError when no index available."""
        location = AnatomicLocation(id="RID1234", description="test")

        with pytest.raises(ValueError, match="Index no longer available"):
            location._get_index(None)

    def test_serialization_excludes_private_attr(self) -> None:
        """Test that _index weakref is excluded from serialization."""
        mock_index = MagicMock()
        location = AnatomicLocation(id="RID1234", description="test")
        location.bind(mock_index)

        dumped = location.model_dump()

        # _index should not appear in dumped dict (it's a PrivateAttr)
        assert "_index" not in dumped
