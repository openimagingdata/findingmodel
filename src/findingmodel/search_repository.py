"""Finding model repository module"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field, model_validator

import findingmodel as fm
import findingmodel.tools as tools
from findingmodel.common import model_file_name, normalize_name

embeddings_model = get_registry().get("openai").create(name="text-embedding-3-large")


class SearchIndexEntry(LanceModel):  # type: ignore
    """An entry in the index file."""

    file: str = Field(..., description="File name of the finding model")
    id: str = Field(..., description="ID of the finding model")
    name: str = Field(..., description="Name of the finding model")
    slug_name: str = Field(..., description="Slug name of the finding model")
    description: str | None = Field(default=None, description="Description of the finding model")
    synonyms: str | None = Field(default=None, description="Synonyms of the finding model")
    tags: list[str] | None = Field(default=None, description="Tags of the finding model")
    index_text: str = embeddings_model.SourceField()
    attribute_names: list[str] = Field(..., description="List of attribute names")
    attribute_ids: list[str] = Field(..., description="List of attribute IDs")
    vector: Vector(embeddings_model.ndims()) = embeddings_model.VectorField(  # type: ignore
        default=None, description="Vector representation of the finding model"
    )

    @classmethod
    def from_filename_finding_model(
        cls, filename: str | Path, finding_model: fm.FindingModelFull
    ) -> "SearchIndexEntry":
        """Create an IndexEntry from a FindingModel object."""
        attributes_ids = [attr.oifma_id for attr in finding_model.attributes]
        attributes_names = [attr.name for attr in finding_model.attributes]
        filename = Path(filename)
        assert filename.suffix == ".json", f"File name must end with .json, not {filename.suffix}"
        return cls(
            file=filename.name,
            id=str(finding_model.oifm_id),
            name=finding_model.name,
            slug_name=normalize_name(finding_model.name),
            description=finding_model.description,
            attribute_names=attributes_names,
            attribute_ids=attributes_ids,
        )

    @model_validator(mode="before")
    @classmethod
    def _set_index_text(cls, values: Any) -> Any:  # noqa: ANN401
        """Return a string representation of the index entry."""
        if not isinstance(values, dict) or "name" not in values or "attribute_names" not in values:
            raise ValueError("Input values must have 'name' and 'attribute_names' keys.")
        index_text = [str(values["name"])]
        if "description" in values:
            index_text.append(values["description"])
        if "synonyms" in values and isinstance(values["synonyms"], list):
            index_text.append("Synonyms: " + "; ".join(values["synonyms"]))
        index_text.append("Attributes: " + "; ".join(values["attribute_names"]))
        values["index_text"] = "\n".join(index_text)
        return values


@dataclass
class SearchResult:
    id: str
    name: str
    file: str
    score: float
    attribute_names: list[str]
    description: str | None = None
    synonyms: str | None = None
    tags: list[str] | None = None


FileNameModelPair = tuple[str | Path, fm.FindingModelFull]

LANCEDB_FILE_NAME = "index.lancedb"


class SearchRepository:
    """A repository for finding models."""

    def __init__(self, in_path: Path | str) -> None:
        self._repo_root = Path(in_path)

        if not self._repo_root.exists():
            raise FileNotFoundError(f"Repository root {self._repo_root} does not exist.")
        self._models_path = self._repo_root / "defs"
        if not self._models_path.exists():
            raise FileNotFoundError(f"Models path {self._models_path} does not exist.")
        if not self._models_path.is_dir():
            raise NotADirectoryError(f"Models path {self._models_path} is not a directory.")
        lancedb_path = self._repo_root / LANCEDB_FILE_NAME
        # existing_or_new = "existing" if lancedb_path.exists() else "new"
        # print(f"Opening {existing_or_new} LanceDB index at {self._repo_root / LANCEDB_FILE_NAME}...", end=" ")
        self._db = lancedb.connect(lancedb_path)
        # print("done.")
        self._table = self._setup_table()
        if self._table.count_rows() == 0:
            self._build_index()

    def _setup_table(self, drop_first: bool = False) -> lancedb.table.Table:
        """Set up the LanceDB table."""
        if "finding_models" in self._db.table_names():
            if drop_first:
                self._db.drop_table("finding_models")
            else:
                return self._db.open_table("finding_models")
        # print("Creating LanceDB table for finding models...", end=" ")
        table = self._db.create_table(name="finding_models", schema=SearchIndexEntry)
        table.create_scalar_index("id")
        table.create_scalar_index("name")
        table.create_scalar_index("file")
        table.create_scalar_index("attribute_ids", index_type="LABEL_LIST")
        table.create_fts_index(
            ["name", "description", "synonyms"], stem=True, remove_stop_words=True, tokenizer_name="en_stem"
        )
        # print("done.")
        return table

    def __len__(self) -> int:
        """Return the number of finding models in the repository."""
        return int(self._table.count_rows())

    def __contains__(self, name_or_id: str) -> bool:
        """Check if a finding model is in the repository."""
        query = f"id == '{name_or_id}' or name == '{name_or_id}'"
        rows = int(self._table.count_rows(query))
        return rows > 0

    def _contains_file(self, file_name: str | Path) -> bool:
        """Check if a file name is in the repository."""
        rows = int(self._table.count_rows(f"file == '{file_name}'"))
        return rows > 0

    def _do_upsert(self, entries: list[SearchIndexEntry]) -> None:
        """Upsert a list of finding models."""
        # print(f"upserting {len(entries)} entries...", end=" ", flush=True)
        self._table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(entries)

    def _upsert_models_to_indices(self, file_name_model_pairs: list[FileNameModelPair]) -> None:
        """Add finding models to the indices."""
        entries = [
            SearchIndexEntry.from_filename_finding_model(file_name, finding_model)
            for file_name, finding_model in file_name_model_pairs
        ]
        self._do_upsert(entries)

    def _add_model_to_indices(self, file_name: str | Path, finding_model: fm.FindingModelFull) -> None:
        """Add a finding model to the indices."""
        entry = SearchIndexEntry.from_filename_finding_model(file_name, finding_model)
        self._do_upsert([entry])

    def _files_in_models_dir(self) -> list[Path]:
        """List all files in the models directory."""
        return [f for f in self._models_path.glob("**/*.json") if f.is_file()]

    def _build_index(self, reset_table: bool = False) -> list[str]:
        """Build the index of finding models."""
        if reset_table:
            self._table = self._setup_table(drop_first=True)

        # print("Building index...", end=" ", flush=True)
        file_names: list[str] = []
        file_name_model_pairs: list[FileNameModelPair] = []
        for model_file in self._files_in_models_dir():
            json_data = model_file.read_text()
            finding_model = fm.FindingModelFull.model_validate_json(json_data)
            file_name_model_pairs.append((model_file, finding_model))
            file_names.append(model_file.name)

        if file_name_model_pairs:
            self._upsert_models_to_indices(file_name_model_pairs)
        # print(f"done ({len(file_name_model_pairs)} found).", flush=True)
        return file_names

    # TODO: Command to refresh or rebuild the index
    # TODO: - need to make sure to delete entries for files that no longer exist

    @property
    def model_names(self) -> list[str]:
        """List all finding model names in the repository (alphabetical order)."""
        results = self._table.search().select(["name"]).to_list()
        return sorted([r["name"] for r in results], key=lambda e: e.lower())

    @property
    def model_ids(self) -> list[str]:
        """List all finding model names in the repository (alphabetical order)."""
        results = self._table.search().select(["id"]).to_list()
        return sorted([r["id"] for r in results])

    def list_models(self) -> Iterator[fm.FindingModelFull]:
        """List all finding models in the repository (alphabetical order)."""
        results = self._table.search().select(["name", "file"]).to_list()
        for file in [r["file"] for r in sorted(results, key=lambda e: e["name"].lower())]:
            json_data = (self._models_path / file).read_text()
            finding_model = fm.FindingModelFull.model_validate_json(json_data)
            yield finding_model

    def check_existing_id(self, id_to_check: str) -> list[SearchIndexEntry]:
        """Check if an ID is already used in the repository."""
        query = f"id == '{id_to_check}' or array_has_any(attribute_ids, ['{id_to_check}'])"
        results = self._table.search().where(query).to_pydantic(SearchIndexEntry)
        return results  # type: ignore

    def check_model_for_duplicate_ids(self, model: fm.FindingModelFull) -> dict[str, SearchIndexEntry]:
        """Check for already-used IDs in a finding model."""
        if not isinstance(model, fm.FindingModelFull):
            raise TypeError(f"Model must be of type FindingModelFull, not {type(model)}")
        duplicate_ids: dict[str, SearchIndexEntry] = {}
        ids_to_check = [model.oifm_id] + [attr.oifma_id for attr in model.attributes]
        for id in ids_to_check:
            if (results := self.check_existing_id(id)) is not None and len(results) > 0:
                duplicate_ids[id] = results[0]
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

        file_name = model_file_name(model.name)
        model_file = self._models_path / file_name
        model_file.write_text(model.model_dump_json(exclude_none=True, indent=2))
        self._add_model_to_indices(model_file, model)
        return model

    def _get_index_entry(self, name_or_id: str) -> SearchIndexEntry | None:
        """Get an index entry from the repository."""
        query = f"id == '{name_or_id}' or slug_name == '{normalize_name(name_or_id)}'"
        results = self._table.search().where(query).limit(1).to_pydantic(SearchIndexEntry)
        if len(results) == 0:
            return None
        return results[0]  # type: ignore

    def _load_model(self, file_name: str) -> fm.FindingModelFull | None:
        """Load a finding model from a file."""
        file_path = self._models_path / file_name
        if not file_path.exists():
            # TODO: Log a warning or error? The index points to a non-existent file.
            return None
        json_data = file_path.read_text()
        finding_model = fm.FindingModelFull.model_validate_json(json_data)
        return finding_model

    def get_model(self, name_or_id: str) -> fm.FindingModelFull | None:
        """Get a finding model from the repository."""
        query = f"id == '{name_or_id}' or slug_name == '{normalize_name(name_or_id)}'"
        results = self._table.search().where(query).select(["file", "id"]).limit(1).to_list()
        if len(results) == 0:
            return None

        entry = results[0]
        return self._load_model(entry["file"])

    def remove_model(self, model: str | fm.FindingModelFull) -> None:
        """Remove a finding model from the repository."""
        match model:
            case str() if model.startswith("OIFM_"):
                query = f"id == '{model}'"
            case str():
                query = f"slug_name == '{normalize_name(model)}'"
            case fm.FindingModelFull():
                query = f"id == '{model.oifm_id}'"
            case _:
                raise TypeError(f"Model must be a string or FindingModelFull, not {type(model)}")
        results = self._table.search().where(query).select(["file", "id"]).limit(1).to_list()
        if len(results) == 0:
            raise ValueError(f"Model {model} not found in the repository.")
        entry = results[0]
        file_path = self._models_path / entry["file"]
        if file_path.exists():
            file_path.unlink()
        query = f"id == '{entry['id']}'"
        self._table.delete(query)

    def search_summary(self, query: str, limit: int = 3) -> list[SearchResult]:
        """Search for finding models in the repository."""
        fields = ["id", "name", "file", "description", "synonyms", "tags", "attribute_names"]
        results = self._table.search(query, query_type="hybrid").select(fields).limit(limit).to_list()
        out = []
        for r in results:
            r["score"] = r["_relevance_score"]
            del r["_relevance_score"]
            out.append(SearchResult(**r))
        return out

    # Make this a generator that only loads the models when needed
    def search_models(self, query: str, limit: int = 3) -> Iterator[tuple[fm.FindingModelFull, float]]:
        """Search for finding models in the repository."""
        results = self.search_summary(query, limit)
        # If results is empty, the loop below won't execute, and the generator will correctly yield nothing.
        for summary in results:
            model = self._load_model(summary.file)
            assert model is not None, f"Model {summary.file} from index not found in the repository."
            yield (model, summary.score)
