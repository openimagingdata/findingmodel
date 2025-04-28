import shutil
from pathlib import Path

import pytest

from findingmodel.finding_model import FindingModelBase, FindingModelFull
from findingmodel.repo import FindingModelRepository


@pytest.fixture
def repo_path_with_index(tmp_path: Path) -> Path:
    # Copy the defs directory and index.jsonl to a temp location
    data_dir = Path(__file__).parent / "data"
    defs_src = data_dir / "defs"
    defs_dst = tmp_path / "defs"
    shutil.copytree(defs_src, defs_dst)
    index_src = data_dir / "index.jsonl"
    index_dst = tmp_path / "index.jsonl"
    shutil.copy(index_src, index_dst)
    return tmp_path


@pytest.fixture
def repo_path_no_index(tmp_path: Path) -> Path:
    # Copy the defs directory and index.jsonl to a temp location
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
    repo = FindingModelRepository(repo_path_with_index)
    assert len(repo) == len(EXPECTED_NAMES)
    assert repo.model_names == EXPECTED_NAMES
    assert repo.model_ids == EXPECTED_IDS
    for model in repo.list_models():
        assert isinstance(model, FindingModelFull)
        assert model.name in EXPECTED_NAMES
        assert model.oifm_id in EXPECTED_IDS


def test_get_model_by_id(repo_path_with_index: Path) -> None:
    # Get all IDs from the index
    repo = FindingModelRepository(repo_path_with_index)
    for model_id in EXPECTED_IDS:
        model = repo.get_model(model_id)
        assert isinstance(model, FindingModelFull)
        assert model.oifm_id == model_id


def test_index_building(repo_path_no_index: Path) -> None:
    # Copy only defs, not index, to test index building
    repo = FindingModelRepository(repo_path_no_index)
    assert len(repo) == len(EXPECTED_NAMES)
    assert repo._index_file == repo_path_no_index / "index.jsonl"
    assert repo._index_file.exists()


def test_repo_contains(repo_path_with_index: Path) -> None:
    # Test the contains method
    repo = FindingModelRepository(repo_path_with_index)
    for model_id in EXPECTED_IDS:
        assert model_id in repo
    assert "non_existent_id" not in repo
    for model_name in EXPECTED_NAMES:
        assert model_name in repo


def test_save_base_model(repo_path_with_index: Path, base_model: FindingModelBase) -> None:
    # Test saving a model to the repository
    repo = FindingModelRepository(repo_path_with_index)
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

    # Check if the index file on disk contains the new entry
    found_in_index_file = False
    with open(repo._index_file, "r") as f:
        for line in f:
            if full_model.oifm_id in line and base_model.name in line:
                found_in_index_file = True
                break
    assert found_in_index_file, f"Model {full_model.oifm_id} not found in index file {repo._index_file}"

    # Test retrieving the saved model
    retrieved_model = repo.get_model(base_model.name)
    assert isinstance(retrieved_model, FindingModelFull)
    assert retrieved_model.oifm_id == full_model.oifm_id
    assert retrieved_model.name == base_model.name


def test_save_full_model(repo_path_with_index: Path, full_model: FindingModelFull) -> None:
    # Test saving a model to the repository
    repo = FindingModelRepository(repo_path_with_index)
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

    # Check if the index file on disk contains the new entry
    found_in_index_file = False
    with open(repo._index_file, "r") as f:
        for line in f:
            if full_model.oifm_id in line and full_model.name in line:
                found_in_index_file = True
                break
    assert found_in_index_file, f"Model {full_model.oifm_id} not found in index file {repo._index_file}"

    # Test retrieving the saved model
    retrieved_model = repo.get_model(full_model.name)
    assert isinstance(retrieved_model, FindingModelFull)
    assert retrieved_model.oifm_id == initial_id
    assert retrieved_model.name == full_model.name


def test_duplicate_id_check(repo_path_with_index: Path) -> None:
    # Test saving a model with an existing ID
    repo = FindingModelRepository(repo_path_with_index)
    EXISTING_MODEL_ID = "OIFM_MSFT_134126"  # Set to an existing ID
    error = repo.check_existing_id(EXISTING_MODEL_ID)
    assert error is not None
    assert error.name == "abdominal aortic aneurysm"
    EXISTING_ATTRIBUTE_ID = "OIFMA_MSFT_898601"  # Set to an existing attribute ID
    error = repo.check_existing_id(EXISTING_ATTRIBUTE_ID)
    assert error is not None
    assert error.name == "abdominal aortic aneurysm"


def test_duplicate_model_id_check(repo_path_with_index: Path, full_model: FindingModelFull) -> None:
    # Test saving a model with an existing ID
    repo = FindingModelRepository(repo_path_with_index)
    EXISTING_ID = "OIFM_MSFT_134126"  # Set to an existing ID
    full_model.oifm_id = EXISTING_ID
    errors = repo.check_model_for_duplicate_ids(full_model)
    assert len(errors) == 1
    assert EXISTING_ID in errors
    assert errors[EXISTING_ID].name == "abdominal aortic aneurysm"


def test_duplicate_attribute_id_check(repo_path_with_index: Path, full_model: FindingModelFull) -> None:
    # Test saving a model with an existing attribute ID
    repo = FindingModelRepository(repo_path_with_index)
    EXISTING_ID = "OIFMA_MSFT_898601"  # Set to an existing ID
    full_model.attributes[0].oifma_id = EXISTING_ID
    errors = repo.check_model_for_duplicate_ids(full_model)
    assert len(errors) == 1
    assert EXISTING_ID in errors
    assert errors[EXISTING_ID].name == "abdominal aortic aneurysm"


def test_save_model_with_duplicate_id(repo_path_with_index: Path, full_model: FindingModelFull) -> None:
    # Test saving a model with an existing ID
    repo = FindingModelRepository(repo_path_with_index)
    EXISTING_ID = "OIFM_MSFT_134126"  # Set to an existing ID
    full_model.oifm_id = EXISTING_ID
    with pytest.raises(ValueError) as excinfo:
        repo.save_model(full_model, source="TEST")
    assert str(excinfo.value) == f"Model {EXISTING_ID} has duplicate IDs: {EXISTING_ID}."


def test_remove_model(repo_path_with_index: Path) -> None:
    # Test removing a model from the repository
    repo = FindingModelRepository(repo_path_with_index)
    initial_len = len(repo)
    model_id = "OIFM_MSFT_134126"  # Set to an existing ID
    index_entry = repo._id_index[model_id]
    assert index_entry is not None
    model_path = repo_path_with_index / "defs" / index_entry.file
    assert model_path.exists()
    repo.remove_model(model_id)
    assert len(repo) == initial_len - 1
    assert model_id not in repo
    assert not model_path.exists()
