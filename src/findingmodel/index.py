from pathlib import Path

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process, utils

from findingmodel.common import model_file_name
from findingmodel.finding_model import FindingModelFull


class AttributeInfo(BaseModel):
    """Represents basic information about an attribute in a FindingModelFull."""

    attribute_id: str
    name: str
    type: str


class IndexEntry(BaseModel):
    """Represents an entry in the Index with basic information about a FindingModelFull."""

    oifm_id: str
    filename: str
    name: str
    description: str | None = None
    synonyms: list[str] | None = None
    tags: list[str] | None = None
    attributes: list[AttributeInfo] = Field(default_factory=list)

    def match(self, name_or_id_or_synonym: str) -> bool:
        """
        Checks if the given name, ID, or synonym matches this entry.
        - If the entry's ID matches, return True.
        - If the entry's name matches (case-insensitive), return True.
        - If any of the entry's synonyms match (case-insensitive), return True.
        """
        if self.oifm_id == name_or_id_or_synonym:
            return True
        if self.name.casefold() == name_or_id_or_synonym.casefold():
            return True
        return bool(self.synonyms and any(syn.casefold() == name_or_id_or_synonym.casefold() for syn in self.synonyms))


class Index:
    """An Index for managing and querying FindingModelFull objects."""

    def __init__(self, base_directory: Path | str) -> None:
        """
        Initializes the Index.
        - If a JSON-L file is present in the base directory, loads the index from it.
        - Otherwise, scans the `defs` directory for definition files and populates the index.
        """
        self.base_directory = base_directory if isinstance(base_directory, Path) else Path(base_directory)
        self.defs_directory = self.base_directory / "defs"
        self.jsonl_file = self.base_directory / "index.jsonl"
        self.entries: list[IndexEntry] = []

        if self.jsonl_file.exists():
            self.load_from_jsonl(self.jsonl_file)
        elif self.defs_directory.exists():
            self.populate_from_directory(self.defs_directory)
        else:
            raise FileNotFoundError(
                f"Neither {self.jsonl_file} nor {self.defs_directory} were found in the base directory."
            )

    def __len__(self) -> int:
        """Returns the number of entries in the index."""
        return len(self.entries)

    def __contains__(self, id_or_name_or_syn: str) -> bool:
        """Checks if an ID or name exists in the index."""
        return any(entry.match(id_or_name_or_syn) for entry in self.entries)

    def __getitem__(self, item: str) -> IndexEntry | None:
        """Returns the IndexEntry for a given ID or name."""
        for entry in self.entries:
            if entry.match(item):
                return entry
        return None

    def _entry_from_model_filename(self, model: FindingModelFull, filename: str | Path) -> IndexEntry:
        """Creates an IndexEntry from a FindingModelFull object and a filename."""
        attributes = [
            AttributeInfo(
                attribute_id=attr.oifma_id,
                name=attr.name,
                type=attr.type,
            )
            for attr in model.attributes
        ]
        filename = filename.name if isinstance(filename, Path) else Path(filename).name
        if not filename.endswith(".fm.json"):
            raise ValueError("Expect filename to end with '.fm.json'")
        entry = IndexEntry(
            oifm_id=model.oifm_id,
            name=model.name,
            description=model.description,
            filename=filename,
            synonyms=(list(model.synonyms) if model.synonyms else None),
            tags=(list(model.tags) if model.tags else None),
            attributes=attributes,
        )
        return entry

    def write_model_to_file(
        self, model: FindingModelFull, /, filename: str | Path | None = None, overwrite: bool = False
    ) -> Path:
        match filename:
            case None:
                filename = Path(model_file_name(model.name))
            case str():
                filename = Path(filename)
            case Path():
                pass
            case _:
                raise ValueError("Filename must be a string or Path object.")
        if not filename.name.endswith(".fm.json"):
            raise ValueError("Expect filename to end with '.fm.json'")

        full_file_path = self.defs_directory / filename
        if full_file_path.exists() and not overwrite:
            raise FileExistsError(
                f"File {filename} already exists in {self.defs_directory}. Use overwrite=True to overwrite."
            )

        full_file_path.write_text(model.model_dump_json(indent=2, exclude_none=True))

        return filename

    def load_model(self, name_or_id_or_syn: str) -> FindingModelFull:
        """Loads a FindingModelFull object from the index using its name, ID, or synonym."""
        if (entry := self[name_or_id_or_syn]) is None:
            raise KeyError(f"Model with name, ID, or synonym '{name_or_id_or_syn}' not found in the index.")
        full_file_path = self.defs_directory / entry.filename
        if not full_file_path.exists():
            raise FileNotFoundError(f"File {entry.filename} not found in {self.defs_directory}.")
        return FindingModelFull.model_validate_json(full_file_path.read_text())

    def add_entry(self, model: FindingModelFull, filename: str | Path, allow_duplicate_synonyms: bool = False) -> None:
        """Adds a FindingModelFull object to the index."""
        filename = filename.name if isinstance(filename, Path) else Path(filename).name
        if not filename.endswith(".fm.json"):
            raise ValueError("Expect filename to end with '.fm.json'")
        if self.id_exists(model.oifm_id):
            raise ValueError(f"Model ID {model.oifm_id} already exists in the index.")
        if model.name in self:
            raise ValueError(f"Model name {model.name} already exists in the index.")
        if any(self.attribute_id_exists(attr.oifma_id) for attr in model.attributes):
            raise ValueError("One or more attribute IDs already exist in the index.")
        if not allow_duplicate_synonyms and model.synonyms:
            for synonym in model.synonyms:
                if synonym in self:
                    raise ValueError(f"Model synonym {synonym} already exists in the index.")
        entry = self._entry_from_model_filename(model, filename)
        self.entries.append(entry)

    def update_entry(self, model: FindingModelFull, filename: str | Path) -> None:
        """Updates an existing entry in the index with a FindingModelFull object."""
        filename = filename.name if isinstance(filename, Path) else Path(filename).name
        if not filename.endswith(".fm.json"):
            raise ValueError("Expect filename to end with '.fm.json'")
        # Get the index of the entry to update
        for index, entry in enumerate(self.entries):
            if entry.oifm_id == model.oifm_id:
                # Update the entry
                self.entries[index] = self._entry_from_model_filename(model, filename)
                return
        raise ValueError(f"Model ID {model.oifm_id} not found in the index.")

    def remove_entry(self, id_or_name: str) -> None:
        """Removes an entry from the index using its ID or name."""
        for index, entry in enumerate(self.entries):
            if entry.match(id_or_name):
                del self.entries[index]
                return
        raise KeyError(f"Model with ID or name '{id_or_name}' not found in the index.")

    def populate_from_directory(self, directory: Path) -> None:
        """Populates the index with FindingModelFull objects from JSON files in a directory."""
        for file in directory.glob("*.fm.json"):
            with open(file, "r") as f:
                model = FindingModelFull.model_validate_json(f.read())
                self.add_entry(model, filename=file)

    def export_to_jsonl(self, file_path: Path | None = None) -> None:
        """Exports the index to a JSON-L file."""
        file_path = file_path or self.jsonl_file
        with open(file_path, "w") as f:
            for entry in self.entries:
                f.write(entry.model_dump_json() + "\n")

    def load_from_jsonl(self, file_path: Path) -> None:
        """Loads the index from a JSON-L file."""
        with open(file_path, "r") as f:
            for line in f:
                entry = IndexEntry.model_validate_json(line.strip())
                self.entries.append(entry)

    def id_exists(self, oifm_id: str) -> bool:
        """Checks if an ID already exists in the index."""
        return any(entry.oifm_id == oifm_id for entry in self.entries)

    def attribute_id_exists(self, attribute_id: str) -> bool:
        """Checks if an attribute ID already exists in the index."""
        return any(attribute_id == attr.attribute_id for entry in self.entries for attr in entry.attributes or [])

    DEFAULT_THRESHOLD = 30

    def find_similar_names(
        self, name: str, *, threshold: float = DEFAULT_THRESHOLD, limit: int = 3
    ) -> list[tuple[str, float]]:
        """Finds similar names in the index using RapidFuzz."""
        names_and_synonyms = [entry.name for entry in self.entries if entry.name and entry.name.strip()] + [
            syn for entry in self.entries if entry.synonyms for syn in entry.synonyms if syn and syn.strip()
        ]
        results = process.extract(
            name, names_and_synonyms, processor=utils.default_process, scorer=fuzz.WRatio, limit=limit
        )
        return [(match, score) for match, score, _ in results if score >= threshold]
