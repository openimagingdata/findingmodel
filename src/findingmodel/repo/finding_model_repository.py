"""Finding model repository module"""

import re
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

import findingmodel as fm
import findingmodel.tools as tools


class IndexEntry(BaseModel):
    """An entry in the index file."""

    file: str = Field(..., description="File name of the finding model")
    id: str = Field(..., description="ID of the finding model")
    name: str = Field(..., description="Name of the finding model")
    description: str | None = Field(default=None, description="Description of the finding model")
    synonyms: list[str] | None = Field(default=None, description="Synonyms of the finding model")
    tags: list[str] | None = Field(default=None, description="Tags of the finding model")
    attribute_names: list[str] = Field(..., description="List of attribute names")
    attribute_ids: list[str] = Field(..., description="List of attribute IDs")

    @classmethod
    def from_filename_finding_model(cls, filename: str | Path, finding_model: fm.FindingModelFull) -> "IndexEntry":
        """Create an IndexEntry from a dictionary."""
        attributes_ids = [attr.oifma_id for attr in finding_model.attributes]
        attributes_names = [attr.name for attr in finding_model.attributes]
        filename = Path(filename)
        assert filename.suffix == ".json", f"File name must end with .json, not {filename.suffix}"
        return cls(
            file=filename.name,
            id=finding_model.oifm_id,
            name=finding_model.name,
            description=finding_model.description,
            attribute_names=attributes_names,
            attribute_ids=attributes_ids,
        )

    def index_text(self) -> str:
        """Return a string representation of the index entry."""
        out = [self.name]
        if self.description:
            out.append(self.description)
        if self.synonyms:
            out.append("; ".join(self.synonyms))
        if self.tags:
            out.append("; ".join(self.tags))
        out.append("; ".join(self.attribute_names))
        return "\n".join(out)


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
        self._file_name_index: dict[str, IndexEntry] = {}
        self._attribute_ids: set[str] = set()
        if self._index_file.exists():
            self._load_index()
        else:
            self._build_index()

    def __len__(self) -> int:
        """Return the number of finding models in the repository."""
        return len(self._name_index)

    def __contains__(self, name_or_id: str) -> bool:
        """Check if a finding model is in the repository."""
        return (_normalize_name(name_or_id)) in self._name_index or name_or_id in self._id_index

    def _add_entry_to_indices(self, entry: IndexEntry) -> None:
        """Add an entry to the indices."""
        self._name_index[_normalize_name(entry.name)] = entry
        self._id_index[entry.id] = entry
        self._file_name_index[entry.file] = entry
        self._attribute_ids.update(entry.attribute_ids)

    def _add_model_to_indices(self, file_name: str | Path, finding_model: fm.FindingModelFull) -> None:
        """Add a finding model to the indices."""
        entry = IndexEntry.from_filename_finding_model(file_name, finding_model)
        self._add_entry_to_indices(entry)

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
            self._add_model_to_indices(model_file, finding_model)

        self._write_index()

    def rebuild_index(self) -> None:
        """Rebuild the index of finding models."""
        self._name_index.clear()
        self._id_index.clear()
        self._loaded_models.clear()
        self._file_name_index.clear()
        self._attribute_ids.clear()
        self._build_index()

    def get_new_model_files(self) -> int:
        """Check for new models in the repository."""
        if not self._models_path.exists():
            raise FileNotFoundError(f"Models path {self._models_path} does not exist.")
        if not self._models_path.is_dir():
            raise NotADirectoryError(f"Models path {self._models_path} is not a directory.")

        new_model_files = 0
        for model_file in self._models_path.glob("**/*.json"):
            if not model_file.is_file():
                continue
            if model_file.name in self._file_name_index:
                continue
            json_data = model_file.read_text()
            finding_model = fm.FindingModelFull.model_validate_json(json_data)
            self._add_model_to_indices(model_file, finding_model)
            new_model_files += 1
        if new_model_files > 0:
            self._write_index()
        return new_model_files

    def _load_index(self) -> None:
        """Load the index of finding models."""
        if not self._index_file.exists() or not self._index_file.is_file():
            raise FileNotFoundError(f"Index file {self._index_file} does not exist.")

        with self._index_file.open("r") as index_file:
            for line in index_file:
                entry = IndexEntry.model_validate_json(line)
                self._add_entry_to_indices(entry)

    def _write_index(self) -> None:
        """Write the index of finding models."""
        if not self._models_path.exists():
            raise FileNotFoundError(f"Models path {self._models_path} does not exist.")
        if not self._models_path.is_dir():
            raise NotADirectoryError(f"Models path {self._models_path} is not a directory.")

        with self._index_file.open("w") as index_file:
            for entry in self._name_index.values():
                index_file.write(entry.model_dump_json(exclude_none=True) + "\n")

    @property
    def model_names(self) -> list[str]:
        """List all finding model names in the repository (alphabetical order)."""
        return [e.name for e in sorted(self._name_index.values(), key=lambda e: _normalize_name(e.name))]

    @property
    def model_ids(self) -> list[str]:
        """List all finding model names in the repository (alphabetical order)."""
        return sorted(self._id_index.keys())

    def list_models(self) -> Iterator[fm.FindingModelFull]:
        """List all finding models in the repository (alphabetical order)."""
        for name in self.model_names:
            finding_model = self.get_model(name)
            if finding_model is None:
                continue
            yield finding_model

    def check_existing_id(self, id: str) -> IndexEntry | None:
        """Check if an ID is already used in the repository."""
        if id in self._id_index:
            return self._id_index[id]
        if id in self._attribute_ids:
            for entry in self._id_index.values():
                if id in entry.attribute_ids:
                    return entry
        return None

    def check_model_for_duplicate_ids(self, model: fm.FindingModelFull) -> dict[str, IndexEntry]:
        """Check for already-used IDs in a finding model."""
        if not isinstance(model, fm.FindingModelFull):
            raise TypeError(f"Model must be of type FindingModelFull, not {type(model)}")
        duplicate_ids: dict[str, IndexEntry] = {}
        ids_to_check = [model.oifm_id] + [attr.oifma_id for attr in model.attributes]
        for id in ids_to_check:
            if (entry := self.check_existing_id(id)) is not None:
                duplicate_ids[id] = entry
        return duplicate_ids

    def save_model(
        self, model: fm.FindingModelBase | fm.FindingModelFull, /, source: str | None = None
    ) -> fm.FindingModelFull:
        """Add a finding model to the repository."""
        if not isinstance(model, (fm.FindingModelBase, fm.FindingModelFull)):
            raise TypeError(f"Model must be of type FindingModelBase or FindingModelFull, not {type(model)}")

        if isinstance(model, fm.FindingModelBase):
            if not isinstance(source, str) or len(source) not in (3, 4):
                raise ValueError("Source must be a string with length 3 or 4.")
            model = tools.add_ids_to_finding_model(model, source.upper())

        if errors := self.check_model_for_duplicate_ids(model):
            raise ValueError(f"Model {model.oifm_id} has duplicate IDs: {', '.join(errors.keys())}.")

        file_name = _file_name_from_model_name(model.name)
        model_file = self._models_path / file_name
        model_file.write_text(model.model_dump_json(exclude_none=True, indent=2))
        self._add_model_to_indices(model_file, model)
        self._write_index()
        return model

    def get_model(self, name_or_id: str) -> fm.FindingModelFull | None:
        """Get a finding model from the repository."""
        if name_or_id in self._loaded_models:
            return self._loaded_models[name_or_id]
        if (name := _normalize_name(name_or_id)) in self._name_index:
            entry = self._name_index[name]
        elif name_or_id in self._id_index:
            entry = self._id_index[name_or_id]
        elif (model_path := self._models_path / _file_name_from_model_name(name_or_id)).exists():
            json_data = model_path.read_text()
            finding_model = fm.FindingModelFull.model_validate_json(json_data)
            self._add_model_to_indices(model_path, finding_model)
            self._loaded_models[name_or_id] = finding_model
            return finding_model
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
        del self._file_name_index[entry.file]
        self._attribute_ids.difference_update(entry.attribute_ids)
        self._write_index()
        self._loaded_models.pop(entry.id, None)
