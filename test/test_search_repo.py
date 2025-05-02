import shutil
from pathlib import Path

import pytest

from findingmodel import FindingModelBase, FindingModelFull

try:
    from findingmodel.search_repository import LANCEDB_FILE_NAME, SearchRepository
except ImportError:
    pytest.skip("Skipping repository tests: findingmodel.search_repository not available", allow_module_level=True)


@pytest.fixture
def repo_path_with_index(tmp_path: Path) -> Path:
    # Copy the defs directory and index.lancedb to a temp location
    data_dir = Path(__file__).parent / "data"
    defs_src = data_dir / "defs"
    defs_dst = tmp_path / "defs"
    shutil.copytree(defs_src, defs_dst)
    index_src = data_dir / "index.tar.gz"
    shutil.unpack_archive(index_src, tmp_path, "gztar")
    index_dir_path = tmp_path / LANCEDB_FILE_NAME
    assert index_dir_path.exists() and index_dir_path.is_dir()
    return tmp_path


@pytest.fixture
def repo_path_no_index(tmp_path: Path) -> Path:
    # Copy the defs directory to a temp location
    data_dir = Path(__file__).parent / "data"
    defs_src = data_dir / "defs"
    defs_dst = tmp_path / "defs"
    shutil.copytree(defs_src, defs_dst)
    return tmp_path


EXPECTED_NAMES = [
    "abdominal aortic aneurysm",
    "aortic dissection",
    "Breast density",
    "Mammographic malignancy assessment",
    "pulmonary embolism",
    "Ventricular diameters",
]

EXPECTED_IDS = [
    "OIFM_MSFT_134126",
    "OIFM_MSFT_156954",
    "OIFM_MSFT_356221",
    "OIFM_MSFT_367670",
    "OIFM_MSFT_573630",
    "OIFM_MSFT_932618",
]


def test_finding_models_sorted(repo_path_with_index: Path) -> None:
    repo = SearchRepository(repo_path_with_index)
    assert len(repo) == len(EXPECTED_NAMES)
    assert repo.model_names == EXPECTED_NAMES
    assert repo.model_ids == EXPECTED_IDS
    for model in repo.list_models():
        assert isinstance(model, FindingModelFull)
        assert model.name in EXPECTED_NAMES
        assert model.oifm_id in EXPECTED_IDS


def test_get_model_by_id(repo_path_with_index: Path) -> None:
    # Get all IDs from the index
    repo = SearchRepository(repo_path_with_index)
    for model_id in EXPECTED_IDS:
        model = repo.get_model(model_id)
        assert isinstance(model, FindingModelFull)
        assert model.oifm_id == model_id


@pytest.mark.callout
def test_index_building(repo_path_no_index: Path) -> None:
    # Copy only defs, not index, to test index building
    repo = SearchRepository(repo_path_no_index)
    assert len(repo) == len(EXPECTED_NAMES)


def test_repo_contains(repo_path_with_index: Path) -> None:
    # Test the contains method
    repo = SearchRepository(repo_path_with_index)
    for model_id in EXPECTED_IDS:
        assert model_id in repo
    assert "non_existent_id" not in repo
    for model_name in EXPECTED_NAMES:
        assert model_name in repo


@pytest.mark.callout
def test_save_base_model(repo_path_with_index: Path, base_model: FindingModelBase) -> None:
    # Test saving a model to the repository
    repo = SearchRepository(repo_path_with_index)
    initial_len = len(repo)
    full_model = repo.save_model(base_model, source="TEST")

    # Check we got back a new full model with an ID
    assert isinstance(full_model, FindingModelFull)
    assert full_model.oifm_id is not None
    assert full_model.name == base_model.name

    # Check if repo knows about the new model
    assert len(repo) == initial_len + 1
    assert full_model.oifm_id in repo
    assert full_model.name in repo

    # Check if the definition file exists
    expected_def_path = repo_path_with_index / "defs" / "test_model.fm.json"
    assert expected_def_path.exists()

    # Check if the index contains the new entry
    count = repo._table.count_rows(f"id = '{full_model.oifm_id}'")
    assert count > 0, f"Model {full_model.oifm_id} not found in index search"

    # Test retrieving the saved model
    retrieved_model = repo.get_model(base_model.name)
    assert isinstance(retrieved_model, FindingModelFull)
    assert retrieved_model.oifm_id == full_model.oifm_id
    assert retrieved_model.name == base_model.name


@pytest.mark.callout
def test_save_full_model(repo_path_with_index: Path, full_model: FindingModelFull) -> None:
    # Test saving a model to the repository
    repo = SearchRepository(repo_path_with_index)
    initial_len = len(repo)
    initial_id = full_model.oifm_id
    full_model = repo.save_model(full_model, source="TEST")

    # Check we got back a new full model with an ID
    assert isinstance(full_model, FindingModelFull)
    assert full_model.oifm_id == initial_id

    # Check if repo knows about the new model
    assert len(repo) == initial_len + 1
    assert full_model.oifm_id in repo
    assert full_model.name in repo

    # Check if the definition file exists
    expected_def_path = repo_path_with_index / "defs" / "test_model.fm.json"
    assert expected_def_path.exists()

    # Check if the index contains the new entry
    count = repo._table.count_rows(f"id = '{full_model.oifm_id}'")
    assert count > 0, f"Model {full_model.oifm_id} not found in index search"

    # Test retrieving the saved model
    retrieved_model = repo.get_model(full_model.name)
    assert isinstance(retrieved_model, FindingModelFull)
    assert retrieved_model.oifm_id == initial_id
    assert retrieved_model.name == full_model.name


def test_duplicate_id_check(repo_path_with_index: Path) -> None:
    # Test saving a model with an existing ID
    repo = SearchRepository(repo_path_with_index)
    EXISTING_MODEL_ID = "OIFM_MSFT_134126"  # Set to an existing ID
    error = repo.check_existing_id(EXISTING_MODEL_ID)
    assert error is not None and len(error) == 1
    assert error[0].name == "abdominal aortic aneurysm"
    EXISTING_ATTRIBUTE_ID = "OIFMA_MSFT_898601"  # Set to an existing attribute ID
    error = repo.check_existing_id(EXISTING_ATTRIBUTE_ID)
    assert error is not None and len(error) == 1
    assert error[0].name == "abdominal aortic aneurysm"


def test_duplicate_model_id_check(repo_path_with_index: Path, full_model: FindingModelFull) -> None:
    # Test saving a model with an existing ID
    repo = SearchRepository(repo_path_with_index)
    EXISTING_ID = "OIFM_MSFT_134126"  # Set to an existing ID
    full_model.oifm_id = EXISTING_ID
    errors = repo.check_model_for_duplicate_ids(full_model)
    assert len(errors) == 1
    assert EXISTING_ID in errors
    assert errors[EXISTING_ID].name == "abdominal aortic aneurysm"


def test_duplicate_attribute_id_check(repo_path_with_index: Path, full_model: FindingModelFull) -> None:
    # Test saving a model with an existing attribute ID
    repo = SearchRepository(repo_path_with_index)
    EXISTING_ID = "OIFMA_MSFT_898601"  # Set to an existing ID
    full_model.attributes[0].oifma_id = EXISTING_ID
    errors = repo.check_model_for_duplicate_ids(full_model)
    assert len(errors) == 1
    assert EXISTING_ID in errors
    assert errors[EXISTING_ID].name == "abdominal aortic aneurysm"


def test_save_model_with_duplicate_id(repo_path_with_index: Path, full_model: FindingModelFull) -> None:
    # Test saving a model with an existing ID
    repo = SearchRepository(repo_path_with_index)
    EXISTING_ID = "OIFM_MSFT_134126"  # Set to an existing ID
    full_model.oifm_id = EXISTING_ID
    with pytest.raises(ValueError) as excinfo:
        repo.save_model(full_model, source="TEST")
    assert str(excinfo.value) == f"Model {EXISTING_ID} has duplicate IDs: {EXISTING_ID}."


def test_remove_model(repo_path_with_index: Path) -> None:
    # Test removing a model from the repository
    repo = SearchRepository(repo_path_with_index)
    initial_len = len(repo)
    model_id = "OIFM_MSFT_134126"  # Set to an existing ID
    index_entry = repo._get_index_entry(model_id)
    assert index_entry is not None
    model_path = repo_path_with_index / "defs" / index_entry.file
    assert model_path.exists()
    repo.remove_model(model_id)
    assert len(repo) == initial_len - 1
    assert model_id not in repo
    assert not model_path.exists()


@pytest.mark.callout
def test_search(repo_path_with_index: Path) -> None:
    # Test searching for a model by name
    repo = SearchRepository(repo_path_with_index)
    EXPECTED_NAME = "ventricular diameters"
    search_results = list(repo.search_models("heart", limit=1))
    assert len(search_results) == 1
    top_result, score = search_results[0]
    assert isinstance(top_result, FindingModelFull)
    assert top_result.name.lower() == EXPECTED_NAME
    assert score > 0
    search_results = list(repo.search_models("breast"))
    assert len(search_results) > 1
    assert all(isinstance(result[0], FindingModelFull) for result in search_results)
    EXPECTED_NAME = "breast density"
    top_result, score = search_results[0]
    assert top_result.name.lower() == EXPECTED_NAME
    assert score > 0
