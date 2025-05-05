import json
import shutil
from pathlib import Path

import pytest

from findingmodel.common import model_file_name
from findingmodel.finding_model import FindingModelFull
from findingmodel.index import Index, IndexEntry

# --- Fixtures ---


@pytest.fixture
def data_dir() -> Path:
    """Path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def defs_dir(data_dir: Path) -> Path:
    """Path to the definition files directory."""
    return data_dir / "defs"


@pytest.fixture
def index_jsonl(data_dir: Path) -> Path:
    """Path to the index JSONL file."""
    return data_dir / "index.jsonl"


@pytest.fixture
def temp_index_dir(tmp_path: Path, defs_dir: Path, index_jsonl: Path) -> Path:
    """Creates a temporary directory structure mimicking the repo."""
    # Copy defs
    temp_defs = tmp_path / "defs"
    shutil.copytree(defs_dir, temp_defs)
    # Copy index.jsonl
    shutil.copy(index_jsonl, tmp_path / "index.jsonl")
    return tmp_path


@pytest.fixture
def index_from_defs(tmp_path: Path, defs_dir: Path) -> Index:
    """Index initialized only from the defs directory."""
    temp_defs = tmp_path / "defs"
    shutil.copytree(defs_dir, temp_defs)
    # Ensure no index.jsonl exists for this test
    if (tmp_path / "index.jsonl").exists():
        (tmp_path / "index.jsonl").unlink()
    return Index(base_directory=tmp_path)


@pytest.fixture
def index_from_jsonl(tmp_path: Path, index_jsonl: Path) -> Index:
    """Index initialized only from the index.jsonl file."""
    # Ensure no defs dir exists for this test
    temp_defs = tmp_path / "defs"
    if temp_defs.exists():
        shutil.rmtree(temp_defs)
    shutil.copy(index_jsonl, tmp_path / "index.jsonl")
    return Index(base_directory=tmp_path)


# --- Test Cases ---


def test_init_from_defs(index_from_defs: Index) -> None:
    """Test index initialization from the defs directory."""
    assert len(index_from_defs) > 0
    assert all(isinstance(entry, IndexEntry) for entry in index_from_defs.entries)
    # Check if a known model is present
    assert "Pulmonary Embolism" in index_from_defs


def test_init_from_jsonl(index_from_jsonl: Index) -> None:
    """Test index initialization from the index.jsonl file."""
    assert len(index_from_jsonl) > 0
    assert all(isinstance(entry, IndexEntry) for entry in index_from_jsonl.entries)
    # Check if a known model is present by ID
    assert "OIFM_MSFT_156954" in index_from_jsonl  # Assuming this ID is in index.jsonl
    mammographic_entry = index_from_jsonl["OIFM_MSFT_156954"]
    assert isinstance(mammographic_entry, IndexEntry)
    assert mammographic_entry.name == "Mammographic malignancy assessment"
    assert "OIFM_MSFT_367677" not in index_from_jsonl  # Assuming this ID is not in index.jsonl


def test_init_prefers_jsonl(temp_index_dir: Path) -> None:
    """Test that index.jsonl is preferred over defs directory if both exist."""
    index = Index(base_directory=temp_index_dir)
    # Check length based on index.jsonl, not by scanning defs again
    with open(temp_index_dir / "index.jsonl", "r") as f:
        expected_len = sum(1 for _ in f)
    assert len(index) == expected_len


def test_init_no_source(tmp_path: Path) -> None:
    """Test initialization raises error if neither source exists."""
    with pytest.raises(FileNotFoundError):
        Index(base_directory=tmp_path)


def test_len(index_from_defs: Index) -> None:
    """Test the __len__ method."""
    # Count files in the original defs dir to know expected length
    expected_len = len(list((Path(__file__).parent / "data" / "defs").glob("*.fm.json")))
    assert len(index_from_defs) == expected_len


def test_contains(index_from_defs: Index) -> None:
    """Test the __contains__ method."""
    # Test with name (case-insensitive)
    assert "Pulmonary Embolism" in index_from_defs
    assert "pulmonary embolism" in index_from_defs
    # Test with ID
    pe_entry = index_from_defs["Pulmonary Embolism"]
    assert pe_entry is not None
    assert pe_entry.oifm_id in index_from_defs
    # Test with synonym (case-insensitive)
    assert "PE" in index_from_defs  # Assuming 'PE' is a synonym for Pulmonary Embolism
    assert "pe" in index_from_defs
    # Test non-existent
    assert "NonExistentModel" not in index_from_defs


def test_getitem(index_from_defs: Index) -> None:
    """Test the __getitem__ method."""
    # Test with name
    entry_by_name = index_from_defs["Pulmonary Embolism"]
    assert isinstance(entry_by_name, IndexEntry)
    assert entry_by_name.name == "pulmonary embolism"
    # Test with ID
    entry_by_id = index_from_defs[entry_by_name.oifm_id]
    assert entry_by_id == entry_by_name
    # Test with synonym
    entry_by_syn = index_from_defs["PE"]
    assert entry_by_syn == entry_by_name
    # Test non-existent
    assert index_from_defs["NonExistentModel"] is None


def test_write_model_to_file(index_from_defs: Index, full_model: FindingModelFull) -> None:
    """Test writing a model to a file."""
    expected_filename = model_file_name(full_model.name)

    assert (index_from_defs.defs_directory / expected_filename).exists() is False

    filename = index_from_defs.write_model_to_file(full_model)
    assert str(filename) == expected_filename

    # Check if the file was created
    assert (index_from_defs.defs_directory / filename).exists()

    # Check if the content is correct
    assert (index_from_defs.defs_directory / filename).read_text() == full_model.model_dump_json(
        indent=2, exclude_none=True
    )


def test_load_model(index_from_defs: Index) -> None:
    """Test loading a model from the index."""
    # Load by name
    model = index_from_defs.load_model("Pulmonary Embolism")
    assert isinstance(model, FindingModelFull)
    assert model.name == "pulmonary embolism"

    # Load by ID
    model = index_from_defs.load_model("OIFM_MSFT_156954")
    assert isinstance(model, FindingModelFull)
    assert model.name == "Mammographic malignancy assessment"

    # Load by synonym
    model = index_from_defs.load_model("PE")
    assert isinstance(model, FindingModelFull)
    assert model.name == "pulmonary embolism"

    # Non-existent
    with pytest.raises(KeyError):
        index_from_defs.load_model("NonExistentModel")

    # Delete a file and try to load it, should raise FileNotFoundError
    entry = index_from_defs["Pulmonary Embolism"]
    assert entry is not None
    filename = index_from_defs.defs_directory / entry.filename
    filename.unlink()  # Delete the file
    with pytest.raises(FileNotFoundError):
        index_from_defs.load_model("Pulmonary Embolism")


def test_add_entry(index_from_defs: Index, full_model: FindingModelFull) -> None:
    """Test adding a new entry."""
    initial_len = len(index_from_defs)
    filename = index_from_defs.write_model_to_file(full_model)

    index_from_defs.add_entry(full_model, filename=filename)

    assert len(index_from_defs) == initial_len + 1
    added_entry = index_from_defs[full_model.name]
    assert added_entry is not None
    assert added_entry.name == "Test Model"
    assert str(added_entry.filename) == str(filename)


def test_add_entry_existing_name(index_from_defs: Index, full_model: FindingModelFull) -> None:
    """Test adding an entry with an existing name."""

    filename = index_from_defs.defs_directory / model_file_name(full_model.name)
    full_model.name = "pulmonary embolism"  # Change the name to an existing one
    # Add the entry
    with pytest.raises(ValueError, match=r"already exist"):
        index_from_defs.add_entry(full_model, filename=filename)


def test_add_entry_existing_attribute_id(index_from_defs: Index, full_model: FindingModelFull) -> None:
    """Test adding an entry with an existing attribute ID."""
    full_model.attributes[0].oifma_id = "OIFMA_MSFT_196263"  # Assuming this ID exists
    filename = index_from_defs.write_model_to_file(full_model)

    # Change the attribute ID to an existing one
    with pytest.raises(ValueError, match=r"already exist"):
        index_from_defs.add_entry(full_model, filename=filename)


def test_add_entry_invalid_filename(index_from_defs: Index, full_model: FindingModelFull) -> None:
    """Test add_entry raises ValueError for invalid filename."""
    with pytest.raises(ValueError, match=r"Expect filename to end with '.fm.json'"):
        index_from_defs.add_entry(full_model, filename="invalid_name.json")


def test_remove_entry(index_from_defs: Index) -> None:
    """Test removing an entry."""
    initial_len = len(index_from_defs)
    entry = index_from_defs["Pulmonary Embolism"]
    assert entry is not None
    index_from_defs.remove_entry(entry.name)
    assert len(index_from_defs) == initial_len - 1
    assert entry.name not in index_from_defs


def test_add_entry_existing_synonym(index_from_defs: Index, full_model: FindingModelFull) -> None:
    """Test adding an entry with an existing synonym."""
    # Assuming "PE" is a synonym for "Pulmonary Embolism"
    assert isinstance(full_model.synonyms, list)
    full_model.synonyms.append("PE")
    filename = index_from_defs.write_model_to_file(full_model)

    # Add the entry
    with pytest.raises(ValueError, match=r"already exist"):
        index_from_defs.add_entry(full_model, filename=filename)


def test_update_entry(index_from_defs: Index, full_model: FindingModelFull) -> None:
    """Test updating an existing entry."""

    model_to_update = index_from_defs.load_model("pulmonary embolism")
    assert model_to_update is not None
    attributes = list(model_to_update.attributes)
    assert (old_length := len(attributes)) > 0
    attributes = attributes[1:]  # Remove the first attribute
    model_to_update.attributes = attributes
    index_from_defs.update_entry(
        model_to_update, model_file_name(model_to_update.name)
    )  # Update the entry in the index
    updated_entry = index_from_defs["pulmonary embolism"]
    assert updated_entry is not None
    assert len(updated_entry.attributes) == old_length - 1


def test_export_to_jsonl(index_from_defs: Index, tmp_path: Path) -> None:
    """Test exporting the index to a JSONL file."""
    output_file = tmp_path / "exported_index.jsonl"
    index_from_defs.export_to_jsonl(output_file)

    assert output_file.exists()
    with open(output_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == len(index_from_defs)
    # Check if the first line is valid JSON and looks like an IndexEntry
    first_entry_data = json.loads(lines[0])
    assert "oifm_id" in first_entry_data
    assert "filename" in first_entry_data
    assert "name" in first_entry_data


def test_id_exists(index_from_defs: Index) -> None:
    """Test the id_exists method."""
    pe_entry = index_from_defs["Pulmonary Embolism"]
    assert pe_entry is not None
    assert index_from_defs.id_exists(pe_entry.oifm_id) is True
    assert index_from_defs.id_exists("OIFM_NONEXISTENT_000000") is False


def test_attribute_id_exists(index_from_defs: Index) -> None:
    """Test the attribute_id_exists method."""
    pe_entry = index_from_defs["Pulmonary Embolism"]
    assert pe_entry is not None
    # Assuming PE has attributes and we know one ID
    if pe_entry.attributes:
        known_attr_id = pe_entry.attributes[0].attribute_id
        assert index_from_defs.attribute_id_exists(known_attr_id) is True
    assert index_from_defs.attribute_id_exists("OIFMA_NONEXISTENT_000000") is False


def test_find_similar_names(index_from_defs: Index) -> None:
    """Test the find_similar_names method."""
    # Exact match
    results_exact = index_from_defs.find_similar_names("Pulmonary Embolism", threshold=95)
    assert len(results_exact) >= 1
    assert results_exact[0][0] == "pulmonary embolism"
    assert results_exact[0][1] >= 95

    # Close match (synonym)
    results_syn = index_from_defs.find_similar_names("PE", threshold=80)
    assert len(results_syn) >= 1
    # The exact match might vary depending on scorer, check if PE or Pulmonary Embolism is top
    assert results_syn[0][0] in ["PE", "pulmonary embolism"]
    assert results_syn[0][1] >= 80

    # Partial/Misspelled match
    results_partial = index_from_defs.find_similar_names("Pulm Embol", threshold=70)
    assert len(results_partial) >= 1
    assert results_partial[0][0] == "pulmonary embolism"
    assert results_partial[0][1] >= 70

    results = index_from_defs.find_similar_names("abdominal")
    assert len(results) > 0 and results[0][0] == "abdominal aortic aneurysm"
    results = index_from_defs.find_similar_names("abdomen")
    assert len(results) > 0 and results[0][0] == "abdominal aortic aneurysm"

    results = index_from_defs.find_similar_names("mammogram")
    assert len(results) >= 2
    assert results[0][0] == "Mammographic malignancy assessment"
    assert results[1][0] == "Mammographic density"

    # Limit
    results_limit = index_from_defs.find_similar_names("embolism", threshold=50, limit=1)
    assert len(results_limit) == 1

    # No match above threshold
    results_none = index_from_defs.find_similar_names("CompletelyUnrelatedTerm", threshold=90)
    assert len(results_none) == 0
