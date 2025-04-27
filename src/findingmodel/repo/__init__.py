"""Finding model repository module"""

import re
from pathlib import Path

from pydantic import BaseModel, Field

import findingmodel as fm


class IndexEntry(BaseModel):
    """An entry in the index file."""

    file: str = Field(..., description="File name of the finding model")
    id: str = Field(..., description="ID of the finding model")
    name: str = Field(..., description="Name of the finding model")
    description: str | None = Field(default=None, description="Description of the finding model")
    attributes: list[tuple[str, str]] = Field(..., description="Attributes of the finding model")

    @classmethod
    def from_filename_finding_model(cls, filename: str | Path, finding_model: fm.FindingModelFull) -> "IndexEntry":
        """Create an IndexEntry from a dictionary."""
        attributes = [(attr.oifma_id, attr.name) for attr in finding_model.attributes]
        return cls(
            file=filename if isinstance(filename, str) else filename.name,
            id=finding_model.oifm_id,
            name=finding_model.name,
            description=finding_model.description,
            attributes=attributes,
        )


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Model name must be a string, not {type(name)}")
    if len(name) < 3:
        raise ValueError(f"Model name must be at least 3 characters long, not {name}")
    return re.sub(r"[^a-zA-Z0-9]", "_", name).lower()


def _file_name_from_model_name(name: str) -> str:
    """Convert a finding model name to a file name."""
    return _normalize_name(name) + ".fm.json"


class FindingModelRepository:
    """A repository for finding models."""

    def __init__(self, in_path: Path | str) -> None:
        self._repo_root = Path(in_path)

        if not self._repo_root.exists():
            raise FileNotFoundError(f"Repository root {self._repo_root} does not exist.")
        self._index_file = self._repo_root / "index.jsonl"
        self._loaded_models: dict[str, fm.FindingModelFull] = {}
        self._models_path = self._repo_root / "defs"
        self._name_index: dict[str, IndexEntry] = {}
        self._id_index: dict[str, IndexEntry] = {}

    def _build_index(self) -> None:
        """Build the index of finding models."""
        if not self._models_path.exists():
            raise FileNotFoundError(f"Models path {self._models_path} does not exist.")
        if not self._models_path.is_dir():
            raise NotADirectoryError(f"Models path {self._models_path} is not a directory.")

        for model_file in self._models_path.glob("**/*.json"):
            if not model_file.is_file():
                continue
            json_data = model_file.read_text()
            finding_model = fm.FindingModelFull.model_validate_json(json_data)
            entry = IndexEntry.from_filename_finding_model(model_file, finding_model)
            self._name_index[_normalize_name(entry.name)] = entry
            self._id_index[entry.id] = entry

        self._write_index()

    def _load_index(self) -> None:
        """Load the index of finding models."""
        if not self._index_file.exists() or not self._index_file.is_file():
            raise FileNotFoundError(f"Index file {self._index_file} does not exist.")

        with self._index_file.open("r") as index_file:
            for line in index_file:
                entry = IndexEntry.model_validate_json(line)
                self._name_index[_normalize_name(entry.name)] = entry
                self._id_index[entry.id] = entry

    def _write_index(self) -> None:
        """Write the index of finding models."""
        if not self._models_path.exists():
            raise FileNotFoundError(f"Models path {self._models_path} does not exist.")
        if not self._models_path.is_dir():
            raise NotADirectoryError(f"Models path {self._models_path} is not a directory.")

        with self._index_file.open("w") as index_file:
            for entry in self._name_index.values():
                index_file.write(entry.model_dump_json(exclude_none=True) + "\n")

    def save_model(self, model: fm.FindingModelBase | fm.FindingModelFull, /, source: str | None = None) -> None:
        """Add a finding model to the repository."""
        if not isinstance(model, (fm.FindingModelBase, fm.FindingModelFull)):
            raise TypeError(f"Model must be of type FindingModelBase or FindingModelFull, not {type(model)}")
        if isinstance(model, fm.FindingModelBase):
            if not isinstance(source, str) or len(source) not in (3, 4):
                raise ValueError("Source must be a string with length 3 or 4.")
            model = fm.tools.add_ids_to_finding_model(model, source.upper())

        file_name = _file_name_from_model_name(model.name)
        model_file = self._models_path / file_name
        model_file.write_text(model.model_dump_json(exclude_none=True, indent=2))
        entry = IndexEntry.from_filename_finding_model(file_name, model)
        self._name_index[_normalize_name(model.name)] = entry
        self._id_index[model.oifm_id] = entry
        self._loaded_models[model.oifm_id] = model
        self._write_index()

    def get_model(self, name_or_id: str) -> fm.FindingModelFull | None:
        """Get a finding model from the repository."""
        if name_or_id in self._loaded_models:
            return self._loaded_models[name_or_id]
        if (name := _normalize_name(name_or_id)) in self._name_index:
            entry = self._name_index[name]
        elif name_or_id in self._id_index:
            entry = self._id_index[name_or_id]
        else:
            return None

        file_path = self._models_path / entry.file
        if not file_path.exists():
            return None
        json_data = file_path.read_text()
        finding_model = fm.FindingModelFull.model_validate_json(json_data)
        self._loaded_models[entry.id] = finding_model
        return finding_model

    def remove_model(self, model: str | fm.FindingModelFull) -> None:
        """Remove a finding model from the repository."""
        match model:
            case str() if model.startswith("OIFM_"):
                id = model
                entry = self._id_index[id]
                name = _normalize_name(entry.name)
            case str():
                name = _normalize_name(model)
                entry = self._name_index[name]
                id = entry.id
            case fm.FindingModelFull():
                name = _normalize_name(model.name)
                entry = self._name_index[name]
                id = model.oifm_id
            case _:
                raise TypeError(f"Model must be a string or FindingModelFull, not {type(model)}")
        file_path = self._models_path / entry.file
        if file_path.exists():
            file_path.unlink()
        del self._name_index[name]
        del self._id_index[entry.id]
        self._write_index()
        self._loaded_models.pop(entry.id, None)
