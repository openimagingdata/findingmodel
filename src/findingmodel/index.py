import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

from findingmodel.common import normalize_name
from findingmodel.config import settings
from findingmodel.contributor import Person
from findingmodel.finding_model import FindingModelFull


class AttributeInfo(BaseModel):
    """Represents basic information about an attribute in a FindingModelFull."""

    attribute_id: str
    name: str
    type: str


class IndexEntry(BaseModel):
    """Represents an entry in the Index with basic information about a FindingModelFull."""

    oifm_id: str
    name: str
    slug_name: str
    filename: str
    file_hash_sha256: str
    description: str | None = None
    synonyms: list[str] | None = None
    tags: list[str] | None = None
    contributors: list[str] | None = None
    attributes: list[AttributeInfo] = Field(default_factory=list, min_length=1)

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


class IndexReturnType(StrEnum):
    ADDED = "added"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class Index:
    """An Index for managing and querying FindingModelFull objects."""

    def __init__(self) -> None:
        """
        Initializes the Index.
        - If a JSON-L file is present in the base directory, loads the index from it.
        - Otherwise, scans the `defs` directory for definition files and populates the index.
        """
        self.client: AsyncIOMotorClient[Any] = AsyncIOMotorClient(settings.mongodb_uri.get_secret_value())
        self.db = self.client.get_database(settings.mongodb_db)
        self.collection = self.db.get_collection(settings.mongodb_collection)
        # self.use_atlas_search = settings.mongodb_use_atlas_search

    async def setup_indexes(self) -> None:
        await self.collection.create_index([("oifm_id", 1)], unique=True)
        await self.collection.create_index([("slug_name", 1)], unique=True)
        await self.collection.create_index([("name", 1)], unique=True)
        await self.collection.create_index([("filename", 1)], unique=True)
        await self.collection.create_index([("synonyms", 1)])
        await self.collection.create_index([("tags", 1)])
        await self.collection.create_index([("attributes.attribute_id", 1)], unique=True)
        # For local/dev: basic TEXT index (not used by Atlas Search)
        # if not self.use_atlas_search:
        if not False:
            await self.collection.create_index(
                [
                    ("name", "text"),
                    ("description", "text"),
                    ("synonyms", "text"),
                    ("tags", "text"),
                    ("attributes.name", "text"),
                ],
                name="fts_allfields",
            )

    async def count(self) -> int:
        """Returns the number of entries in the index."""
        return await self.collection.count_documents({})

    def _id_or_name_or_syn_query(self, id_or_name_or_syn: str) -> dict[str, Any]:
        """Helper method to create a query for ID, name, or synonym."""
        return {
            "$or": [
                {"oifm_id": id_or_name_or_syn},
                {"name": {"$regex": f"^{id_or_name_or_syn}$", "$options": "i"}},
                {"synonyms": {"$regex": f"^{id_or_name_or_syn}$", "$options": "i"}},
            ]
        }

    async def contains(self, id_or_name_or_syn: str) -> bool:
        """Checks if an ID or name exists in the index."""
        # Search for a matching ID, name, or a synonym in the database
        query = self._id_or_name_or_syn_query(id_or_name_or_syn)
        return bool(await self.collection.find_one(query))

    async def get(self, id_or_name_or_syn: str) -> IndexEntry | None:
        """Retrieves an IndexEntry by its ID, name, or synonym."""
        query = self._id_or_name_or_syn_query(id_or_name_or_syn)
        entry_data = await self.collection.find_one(query)
        if entry_data:
            return IndexEntry.model_validate(entry_data)
        return None

    def _calculate_file_hash(self, filename: str | Path) -> str:
        """Calculates the SHA-256 hash of a file."""
        filepath = filename if isinstance(filename, Path) else Path(filename)
        if not filepath.exists() or not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} not found.")
        try:
            file_bytes = filepath.read_bytes()
            return hashlib.sha256(file_bytes).hexdigest()
        except IOError as e:
            raise IOError(f"Error reading file {filepath}: {e}") from e

    def _entry_from_model_file(
        self, model: FindingModelFull, filepath: str | Path, file_hash: str | None = None
    ) -> IndexEntry:
        """Creates an IndexEntry from a FindingModelFull object and a filename."""
        filepath = filepath if isinstance(filepath, Path) else Path(filepath)
        attributes = [
            AttributeInfo(
                attribute_id=attr.oifma_id,
                name=attr.name,
                type=attr.type,
            )
            for attr in model.attributes
        ]
        contributors: list[str] | None = None
        if model.contributors:
            contributors = [
                contributor.github_username if isinstance(contributor, Person) else contributor.code
                for contributor in model.contributors
            ]
        if not filepath.name.endswith(".fm.json"):
            raise ValueError("Expect filename to end with '.fm.json'")
        file_hash = file_hash or self._calculate_file_hash(filepath)
        entry = IndexEntry(
            oifm_id=model.oifm_id,
            name=model.name,
            slug_name=normalize_name(model.name),
            filename=filepath.name,
            file_hash_sha256=file_hash,
            description=model.description,
            synonyms=(list(model.synonyms) if model.synonyms else None),
            tags=(list(model.tags) if model.tags else None),
            contributors=contributors,
            attributes=attributes,
        )
        return entry

    async def validate_model(self, model: FindingModelFull, allow_duplicate_synonyms: bool = False) -> list[str]:
        """Validates that the FindingModelFull object is ready to be indexed."""
        errors = []

        # Check if the model has a valid ID and name
        duplicate_name_id_query = {
            "$or": [
                {"oifm_id": model.oifm_id},
                {"name": {"$regex": f"^{model.name}$", "$options": "i"}},
            ]
        }
        duplicate_matches = await self.collection.find(duplicate_name_id_query).to_list(length=None)
        if duplicate_matches:
            errors.append(f"Duplicate ID or name found: {model.oifm_id} or {model.name}. ")

        attribute_ids = {attr.oifma_id for attr in model.attributes}
        attribute_id_query = {"attributes.attribute_id": {"$in": list(attribute_ids)}}
        duplicate_attributes = await self.collection.find(attribute_id_query).to_list(length=None)
        if duplicate_attributes:
            errors.append(f"Duplicate attribute IDs found: {duplicate_attributes!s}. ")

        return errors

    async def add_or_update_entry_from_file(
        self, filename: str | Path, model: FindingModelFull | None = None, allow_duplicate_synonyms: bool = False
    ) -> IndexReturnType:
        """Adds a FindingModelFull object to the index."""
        filename = filename if isinstance(filename, Path) else Path(filename)
        if not filename.name.endswith(".fm.json"):
            raise ValueError("Expect filename to end with '.fm.json'")
        existing_entry = await self.collection.find_one(
            {"filename": filename.name}, {"filename": 1, "oifm_id": 1, "file_hash_sha256": 1}
        )
        current_hash: str | None = None
        if existing_entry:
            # If the entry already exists, check if the file hash matches
            existing_hash = existing_entry.get("file_hash_sha256")
            current_hash = self._calculate_file_hash(filename)
            if existing_hash == current_hash:
                # logger.info(f"Entry for {filename.name} already exists with matching hash. Skipping addition.")
                return IndexReturnType.UNCHANGED
            # logger.info(f"Deleting existing out-of-date entry for {filename.name} (hash changed).")
            await self.collection.delete_one({"oifm_id": existing_entry["oifm_id"]})

        model = model or FindingModelFull.model_validate_json(filename.read_text())
        errors = await self.validate_model(model, allow_duplicate_synonyms=allow_duplicate_synonyms)
        if errors:
            # logger.error(f"Model validation failed for {filename.name}: {errors}")
            raise ValueError(f"Model validation failed: {'; '.join(errors)}")
        new_entry = self._entry_from_model_file(model, filename, current_hash)
        # logger.info(f"Adding new entry for {filename.name} with ID {new_entry.oifm_id}.")
        await self.collection.insert_one(new_entry.model_dump())
        return IndexReturnType.UPDATED if existing_entry else IndexReturnType.ADDED

    async def remove_unused_entries(self, active_filenames: Iterable[str]) -> Iterable[str]:
        """
        Asynchronously removes entries from the MongoDB collection whose filenames are not in the provided list of used filenames.
        This method interacts directly with the MongoDB collection and may have side effects if used concurrently.
        """
        active_filenames = set(active_filenames)
        assert isinstance(active_filenames, set), "active_filenames must be a set for efficient lookup"
        current_filenames = await self.collection.distinct("filename")
        unused_filenames = set(current_filenames) - active_filenames
        if not unused_filenames:
            return []
        # logger.info(f"Removing {len(unused_filenames)} unused entries from the index.")
        result = await self.collection.delete_many({"filename": {"$in": list(unused_filenames)}})
        if result.deleted_count == len(unused_filenames):
            # logger.info(f"Successfully removed {result.deleted_count} unused entries.")
            pass
        else:
            # logger.warning(f"Expected to remove {len(unused_filenames)} entries, but only removed {result.deleted_count}.")
            pass
        return unused_filenames

    async def to_markdown(self) -> str:
        """Converts the index to a Markdown table."""
        length = await self.count()
        header = f"""

# Finding Model Index

{length} entries

| ID | Name | Synonyms | Tags | Contributors | Attributes |\n"""
        separator = "|----|------|----------|------|--------------|------------|\n"
        rows = []
        all_entries_sorted = self.collection.find({})
        async for entry_data in all_entries_sorted:
            entry = IndexEntry.model_validate(entry_data)
            md_filename = entry.filename.replace(".fm.json", ".md")
            entry_name_with_links = f"[{entry.name}](text/{md_filename}) [JSON](defs/{entry.filename})"
            row = f"| {entry.oifm_id} | {entry_name_with_links} | {', '.join(entry.synonyms or [])} | {', '.join(entry.tags or [])} | {', '.join(entry.contributors or [])} | {', '.join(attr.name for attr in entry.attributes)} |\n"
            rows.append(row)
        return header + separator + "".join(rows)
