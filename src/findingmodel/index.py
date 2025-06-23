import hashlib
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from pymongo import UpdateOne

from findingmodel import logger
from findingmodel.common import normalize_name
from findingmodel.config import settings
from findingmodel.contributor import Organization, Person
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

    def __init__(
        self,
        *,
        mongodb_uri: str | None = None,
        db_name: str | None = None,
        client: AsyncIOMotorClient[Any] | None = None,
        branch: str = "main",
    ) -> None:
        """
        Initializes the Index.
        - Can be initialized with a mongodb_uri or an existing AsyncIOMotorClient.
        - If a client is not provided, a new one will be created using the mongodb_uri.
        - If mongodb_uri is not provided, it will be taken from settings.
        """
        if client:
            self.client: AsyncIOMotorClient[Any] = client
        else:
            mongodb_uri = mongodb_uri or settings.mongodb_uri.get_secret_value()
            self.client = AsyncIOMotorClient(mongodb_uri)

        db_name = db_name or settings.mongodb_db
        self.db = self.client.get_database(db_name)
        self.index_collection = self.db.get_collection(settings.mongodb_index_collection_base + f"_{branch}")
        self.people_collection = self.db.get_collection(settings.mongodb_people_collection_base + f"_{branch}")
        self.organizations_collection = self.db.get_collection(
            settings.mongodb_organizations_collection_base + f"_{branch}"
        )
        # self.use_atlas_search = settings.mongodb_use_atlas_search

    async def setup_indexes(self) -> None:
        # Indexes for the main index collection
        existing_indexes = await self.index_collection.index_information()
        if "oifm_id_1" in existing_indexes and "slug_name_1" in existing_indexes and "name_1" in existing_indexes:
            logger.info("Indexes already set up, skipping index creation.")
        else:
            await self.index_collection.create_index([("oifm_id", 1)], unique=True)
            await self.index_collection.create_index([("slug_name", 1)], unique=True)
            await self.index_collection.create_index([("name", 1)], unique=True)
            await self.index_collection.create_index([("filename", 1)], unique=True)
            await self.index_collection.create_index([("synonyms", 1)])
            await self.index_collection.create_index([("tags", 1)])
            await self.index_collection.create_index([("attributes.attribute_id", 1)], unique=True)
            await self.index_collection.create_index(
                [
                    ("name", "text"),
                    ("description", "text"),
                    ("synonyms", "text"),
                    ("tags", "text"),
                    ("attributes.name", "text"),
                ],
                weights={
                    "name": 10,
                    "synonyms": 8,
                    "description": 3,
                    "tags": 1,
                },
                name="fts_allfields",
            )
            logger.info(
                "Created indices for ID, slug_name, name, filename, synonyms, tags, attribute IDs, and text search."
            )

        # Indexes for the people collection
        people_indexes = await self.people_collection.index_information()
        if "github_username_1" not in people_indexes or "name_1" not in people_indexes:
            await self.people_collection.create_index([("github_username", 1)], unique=True)
            await self.people_collection.create_index([("name", 1)])
            logger.info("Created unique index for github_username and index for name in people collection.")
        else:
            logger.info("People collection already has required indices (github_username, name). Skipping creation.")

        # Indexes for the organizations collection
        org_indexes = await self.organizations_collection.index_information()
        if "code_1" not in org_indexes or "name_1" not in org_indexes:
            await self.organizations_collection.create_index([("code", 1)], unique=True)
            await self.organizations_collection.create_index([("name", 1)])
            logger.info("Created unique index for code and index for name in organizations collection.")
        else:
            logger.info("Organizations collection already has required indices (code, name). Skipping creation.")

    async def count(self) -> int:
        """Returns the number of entries in the index."""
        return await self.index_collection.count_documents({})

    async def count_people(self) -> int:
        """Returns the number of people in the people collection."""
        return await self.people_collection.count_documents({})

    async def count_organizations(self) -> int:
        """Returns the number of organizations in the organizations collection."""
        return await self.organizations_collection.count_documents({})

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
        return bool(await self.index_collection.find_one(query))

    async def get(self, id_or_name_or_syn: str) -> IndexEntry | None:
        """Retrieves an IndexEntry by its ID, name, or synonym."""
        query = self._id_or_name_or_syn_query(id_or_name_or_syn)
        entry_data = await self.index_collection.find_one(query)
        if entry_data:
            return IndexEntry.model_validate(entry_data)
        return None

    async def get_person(self, github_username: str) -> Person | None:
        """Retrieve a Person by github_username."""
        doc = await self.people_collection.find_one({"github_username": github_username})
        if doc:
            return Person.model_validate(doc)
        return None

    async def get_organization(self, code: str) -> Organization | None:
        """Retrieve an Organization by code."""
        doc = await self.organizations_collection.find_one({"code": code})
        if doc:
            return Organization.model_validate(doc)
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

    async def _get_validation_data(self) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """Get dicts for validation: name->oifm_id, attr_id->oifm_id, oifm_id->filename."""
        cursor = self.index_collection.find({}, {"oifm_id": 1, "name": 1, "filename": 1, "attributes.attribute_id": 1})
        name_to_oifm = {}
        attrid_to_oifm = {}
        oifm_to_filename = {}
        async for entry in cursor:
            oifm_id = entry["oifm_id"]
            name_to_oifm[entry["name"].casefold()] = oifm_id
            oifm_to_filename[oifm_id] = entry.get("filename", "?")
            if "attributes" in entry:
                for attr in entry["attributes"]:
                    attrid_to_oifm[attr["attribute_id"]] = oifm_id
        return name_to_oifm, attrid_to_oifm, oifm_to_filename

    def _check_id_conflict(
        self, oifm_id: str, name_fold: str, oifm_to_name: dict[str, str], exclude_oifm_id: str | None
    ) -> str | None:
        """Check for ID conflicts."""
        if oifm_id in oifm_to_name:
            other_name = oifm_to_name[oifm_id]
            if other_name != name_fold:
                return f"Duplicate ID '{oifm_id}' (also in {other_name})"
        return None

    def _check_name_conflict(
        self, name_fold: str, name_to_oifm: dict[str, str], batch_names: dict[str, str], exclude_oifm_id: str | None
    ) -> list[str]:
        """Check for name conflicts."""
        errors = []
        if name_fold in name_to_oifm:
            other = name_to_oifm[name_fold]
            if not exclude_oifm_id or other != exclude_oifm_id:
                errors.append(f"Duplicate name '{name_fold}' (also in {other})")
        for n, other_id in batch_names.items():
            if n == name_fold and other_id != exclude_oifm_id:
                errors.append(f"Duplicate name '{name_fold}' in batch (also in {other_id})")
        return errors

    def _check_attribute_id_conflict(
        self,
        aid: str,
        oifm_id: str,
        attrid_to_oifm: dict[str, str],
        batch_attrids: dict[str, str],
        oifm_to_filename: dict[str, str],
        exclude_oifm_id: str | None,
    ) -> list[str]:
        """Check for attribute ID conflicts."""
        errors = []
        if aid in attrid_to_oifm:
            other = attrid_to_oifm[aid]
            if other != oifm_id and (not exclude_oifm_id or other != exclude_oifm_id):
                errors.append(f"Attribute ID conflict: '{aid}' also in {other} ({oifm_to_filename.get(other, '?')})")
        # Attribute ID conflict in batch
        for bid, other_id in batch_attrids.items():
            if bid == aid and other_id != oifm_id:
                errors.append(f"Attribute ID conflict: '{aid}' in batch (also in {other_id})")

        return errors

    async def validate_models_batch(
        self, models: list[tuple[FindingModelFull, str | None]], allow_duplicate_synonyms: bool = False
    ) -> dict[str, list[str]]:
        """Validate multiple models efficiently with detailed conflict info."""
        name_to_oifm, attrid_to_oifm, oifm_to_filename = await self._get_validation_data()
        oifm_to_name = {v: k for k, v in name_to_oifm.items()}
        validation_results = {}
        # Also check for conflicts within the batch
        batch_names = {}
        batch_attrids = {}
        for model, _ in models:
            batch_names[model.name.casefold()] = model.oifm_id
            for attr in model.attributes:
                batch_attrids[attr.oifma_id] = model.oifm_id
        for model, exclude_oifm_id in models:
            errors = []
            oifm_id = model.oifm_id
            name_fold = model.name.casefold()
            # ID conflict
            id_conflict = self._check_id_conflict(oifm_id, name_fold, oifm_to_name, exclude_oifm_id)
            if id_conflict:
                errors.append(id_conflict)
            # Name conflict
            name_conflicts = self._check_name_conflict(name_fold, name_to_oifm, batch_names, exclude_oifm_id=oifm_id)
            if name_conflicts:
                errors.extend(name_conflicts)
            # Attribute ID conflict
            for attr in model.attributes:
                aid = attr.oifma_id
                attr_conflicts = self._check_attribute_id_conflict(
                    aid, oifm_id, attrid_to_oifm, batch_attrids, oifm_to_filename, exclude_oifm_id
                )
                if attr_conflicts:
                    errors.extend(attr_conflicts)
            validation_results[oifm_id] = errors
        return validation_results

    async def validate_model(
        self, model: FindingModelFull, allow_duplicate_synonyms: bool = False, exclude_oifm_id: str | None = None
    ) -> list[str]:
        """Validates a FindingModelFull object using the new batch validation logic."""
        return (await self.validate_models_batch([(model, exclude_oifm_id)], allow_duplicate_synonyms))[model.oifm_id]

    async def add_or_update_contributors(self, contributors: list[Person | Organization]) -> list[str] | None:  # noqa: C901
        """
        Insert or update unique Person/Organization contributors.
        Returns list of conflicts conflicting data for the same github_username/code is found.
        """

        # Separate by type and key
        people_by_username: dict[str, Person] = {}
        orgs_by_code: dict[str, Organization] = {}
        person_conflicts = defaultdict(list)
        org_conflicts = defaultdict(list)

        for c in contributors:
            if isinstance(c, Person):
                key = c.github_username
                if key in people_by_username:
                    if c.model_dump() != people_by_username[key].model_dump():
                        person_conflicts[key].append(c)
                else:
                    people_by_username[key] = c
            elif isinstance(c, Organization):
                key = c.code
                if key in orgs_by_code:
                    if c.model_dump() != orgs_by_code[key].model_dump():
                        org_conflicts[key].append(c)
                else:
                    orgs_by_code[key] = c

        # Raise error if any conflicts
        if person_conflicts or org_conflicts:
            errors: list[str] = []
            if person_conflicts:
                errors.append(f"Person conflicts: {list(person_conflicts.keys())}")
            if org_conflicts:
                errors.append(f"Organization conflicts: {list(org_conflicts.keys())}")
            return errors

        logger.info(f"Upserting {len(people_by_username)} people and {len(orgs_by_code)} organizations.")

        # Upsert people
        people_ops: list[UpdateOne] = []
        for person in people_by_username.values():
            person_dict = person.model_dump()
            if "url" in person_dict:
                person_dict["url"] = str(person_dict["url"])
            people_ops.append(
                UpdateOne(
                    {"github_username": person.github_username},
                    {"$set": person_dict},
                    upsert=True,
                )
            )
        if people_ops:
            result = await self.people_collection.bulk_write(people_ops)
            logger.info(f"Upserted {result.upserted_count} people and modified {result.modified_count} people.")

        # Upsert organizations
        org_ops: list[UpdateOne] = []
        for org in orgs_by_code.values():
            org_dict = org.model_dump()
            if "url" in org_dict:
                org_dict["url"] = str(org_dict["url"])
            org_ops.append(
                UpdateOne(
                    {"code": org.code},
                    {"$set": org_dict},
                    upsert=True,
                )
            )
        if org_ops:
            result = await self.organizations_collection.bulk_write(org_ops)
            logger.info(
                f"Upserted {result.upserted_count} organizations and modified {result.modified_count} organizations."
            )

        return None

    async def add_or_update_entry_from_file(
        self, filename: str | Path, model: FindingModelFull | None = None, allow_duplicate_synonyms: bool = False
    ) -> IndexReturnType:
        """Adds a FindingModelFull object to the index."""
        filename = filename if isinstance(filename, Path) else Path(filename)
        if not filename.name.endswith(".fm.json"):
            raise ValueError("Expect filename to end with '.fm.json'")
        existing_entry = await self.index_collection.find_one(
            {"filename": filename.name}, {"filename": 1, "oifm_id": 1, "file_hash_sha256": 1}
        )
        current_hash: str | None = None
        if existing_entry:
            # If the entry already exists, check if the file hash matches
            existing_hash = existing_entry.get("file_hash_sha256")
            current_hash = self._calculate_file_hash(filename)
            if existing_hash == current_hash:
                logger.info(f"Entry for {filename.name} already exists with matching hash. Skipping addition.")
                return IndexReturnType.UNCHANGED
            logger.info(f"Deleting existing out-of-date entry for {filename.name} (hash changed).")
            await self.index_collection.delete_one({"oifm_id": existing_entry["oifm_id"]})

        model = model or FindingModelFull.model_validate_json(filename.read_text())
        errors: list[str] | None = await self.validate_model(model, allow_duplicate_synonyms=allow_duplicate_synonyms)
        if errors:
            logger.error(f"Model validation failed for {filename.name}: {errors}")
            raise ValueError(f"Model validation failed: {'; '.join(errors)}")
        if model.contributors:
            errors = await self.add_or_update_contributors(model.contributors)
            if errors:
                logger.error(f"Contributor validation failed for {filename.name}: {errors}")
                raise ValueError(f"Contributor validation failed: {'; '.join(errors)}")
        new_entry = self._entry_from_model_file(model, filename, current_hash)
        logger.info(f"Adding new entry for {filename.name} with ID {new_entry.oifm_id}.")
        await self.index_collection.insert_one(new_entry.model_dump())
        return IndexReturnType.UPDATED if existing_entry else IndexReturnType.ADDED

    async def remove_entry(self, id_or_name: str) -> bool:
        """Removes an entry from the index by its ID or name."""
        result = await self.index_collection.delete_one({"$or": [{"oifm_id": id_or_name}, {"name": id_or_name}]})
        logger.info(f"Removed entry for {id_or_name}. Deleted count: {result.deleted_count}")
        return result.deleted_count > 0

    async def remove_unused_entries(self, active_filenames: Iterable[str]) -> Iterable[str]:
        """
        Asynchronously removes entries from the MongoDB collection whose filenames are not in the provided list of used filenames.
        This method interacts directly with the MongoDB collection and may have side effects if used concurrently.
        """
        active_filenames = set(active_filenames)
        assert isinstance(active_filenames, set), "active_filenames must be a set for efficient lookup"
        current_filenames = await self.index_collection.distinct("filename")
        unused_filenames = set(current_filenames) - active_filenames
        if not unused_filenames:
            return []
        logger.info(f"Removing {len(unused_filenames)} unused entries from the index.")
        result = await self.index_collection.delete_many({"filename": {"$in": list(unused_filenames)}})
        if result.deleted_count == len(unused_filenames):
            logger.info(f"Successfully removed {result.deleted_count} unused entries.")
            pass
        else:
            logger.warning(
                f"Expected to remove {len(unused_filenames)} entries, but only removed {result.deleted_count}."
            )
            pass
        return unused_filenames

    async def search(self, query: str, limit: int = 10) -> list[IndexEntry]:
        """
        Searches the index for entries matching the given query.
        Uses MongoDB's text search capabilities.
        """
        search_query = {
            "$text": {"$search": query},
        }
        cursor = self.index_collection.find(search_query).limit(limit)
        results = []
        async for entry_data in cursor:
            entry = IndexEntry.model_validate(entry_data)
            results.append(entry)
        logger.info(f"Search completed. Found {len(results)} entries (limit {limit}) matching query '{query}'.")
        return results

    async def _get_existing_file_info(self) -> dict[str, dict[str, str]]:
        """Get all existing filename/hash/oifm_id pairs from the database."""
        existing_entries = {}
        cursor = self.index_collection.find({}, {"filename": 1, "file_hash_sha256": 1, "oifm_id": 1})
        async for entry in cursor:
            existing_entries[entry["filename"]] = {"hash": entry["file_hash_sha256"], "oifm_id": entry["oifm_id"]}
        return existing_entries

    def _get_local_file_info(self, file_paths: list[Path]) -> dict[str, dict[str, Any]]:
        """Get all filename/hash pairs from the local directory."""
        local_files = {}
        for file_path in file_paths:
            filename = file_path.name
            file_hash = self._calculate_file_hash(file_path)
            local_files[filename] = {"path": file_path, "hash": file_hash}
        return local_files

    def _determine_operations(
        self, local_files: dict[str, dict[str, Any]], existing_entries: dict[str, dict[str, str]]
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any], str]], list[tuple[str, str]], int]:
        """Determine what operations need to be performed."""
        to_insert = []  # New files
        to_update = []  # Updated files (hash changed)
        to_remove = []  # Files no longer present
        unchanged = 0

        # Check for new and updated files
        for filename, file_info in local_files.items():
            if filename not in existing_entries:
                # New file
                to_insert.append((filename, file_info))
            elif existing_entries[filename]["hash"] != file_info["hash"]:
                # Updated file (hash changed)
                to_update.append((filename, file_info, existing_entries[filename]["oifm_id"]))
            else:
                # Unchanged file
                unchanged += 1

        # Check for removed files
        active_filenames = set(local_files.keys())
        removed_filenames = set(existing_entries.keys()) - active_filenames
        for filename in removed_filenames:
            to_remove.append((filename, existing_entries[filename]["oifm_id"]))

        return to_insert, to_update, to_remove, unchanged

    async def _prepare_entries_for_batch(
        self,
        to_insert: list[tuple[str, dict[str, Any]]],
        to_update: list[tuple[str, dict[str, Any], str]],
        allow_duplicate_synonyms: bool,
    ) -> tuple[list[dict[str, Any]], list[tuple[str | None, dict[str, Any]]]]:
        """Prepare IndexEntry objects for batch operations."""
        insert_entries: list[dict[str, Any]] = []
        update_entries: list[tuple[str | None, dict[str, Any]]] = []

        # First load all models and prepare for batch validation
        models_to_validate: list[tuple[FindingModelFull, str | None]] = []
        models_by_oifm_id: dict[str, tuple[FindingModelFull, dict[str, Any], str | None]] = {}

        # Process files to insert
        for _filename, file_info in to_insert:
            model = FindingModelFull.model_validate_json(file_info["path"].read_text())
            models_to_validate.append((model, None))
            models_by_oifm_id[model.oifm_id] = (model, file_info, None)

        # Process files to update
        for _filename, file_info, old_oifm_id in to_update:
            model = FindingModelFull.model_validate_json(file_info["path"].read_text())
            models_to_validate.append((model, old_oifm_id))
            models_by_oifm_id[model.oifm_id] = (model, file_info, old_oifm_id)

        # Validate all models in batch
        validation_results = await self.validate_models_batch(models_to_validate, allow_duplicate_synonyms)

        # Check for any validation errors
        validation_errors = []
        for oifm_id, errors in validation_results.items():
            if errors:
                model, file_info, _ = models_by_oifm_id[oifm_id]
                validation_errors.append(f"Model validation failed for {file_info['path'].name}: {errors}")

        if validation_errors:
            error_msg = "; ".join(validation_errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Create entries for valid models
        contributors: list[Person | Organization] = []
        for _oifm_id, (model, file_info, old_oifm_id) in models_by_oifm_id.items():  # type: ignore
            if model.contributors:
                contributors.extend(model.contributors)
            entry = self._entry_from_model_file(model, file_info["path"], file_info["hash"])
            if old_oifm_id is not None:
                update_entries.append((old_oifm_id, entry.model_dump()))
                logger.info(f"Prepared {file_info['path'].name} for update with ID {entry.oifm_id}")
            else:
                insert_entries.append(entry.model_dump())
                logger.info(f"Prepared {file_info['path'].name} for insertion with ID {entry.oifm_id}")

        await self.add_or_update_contributors(contributors)

        return insert_entries, update_entries

    async def _execute_batch_operations(
        self,
        to_remove: list[tuple[str, str]],
        update_entries: list[tuple[str | None, dict[str, Any]]],
        insert_entries: list[dict[str, Any]],
    ) -> None:
        """Execute the batch database operations."""
        # Remove old entries first (to avoid constraint conflicts)
        if to_remove:
            remove_oifm_ids = [oifm_id for _, oifm_id in to_remove]
            remove_result = await self.index_collection.delete_many({"oifm_id": {"$in": remove_oifm_ids}})
            logger.info(f"Removed {remove_result.deleted_count} entries")

        # Update existing entries
        if update_entries:
            # Delete old entries and insert new ones in batch
            old_oifm_ids = [old_oifm_id for old_oifm_id, _ in update_entries]
            new_entry_data = [new_entry_data for _, new_entry_data in update_entries]

            delete_result = await self.index_collection.delete_many({"oifm_id": {"$in": old_oifm_ids}})
            insert_result = await self.index_collection.insert_many(new_entry_data)
            logger.info(
                f"Updated {len(update_entries)} entries (deleted: {delete_result.deleted_count}, inserted: {len(insert_result.inserted_ids)})"
            )

        # Insert new entries
        if insert_entries:
            insert_result = await self.index_collection.insert_many(insert_entries)
            logger.info(f"Inserted {len(insert_result.inserted_ids)} new entries")

    async def update_from_directory(
        self, directory: str | Path, allow_duplicate_synonyms: bool = False
    ) -> tuple[int, int, int]:
        """
        Updates the index from a directory containing FindingModelFull JSON files.
        - Scans the directory for files ending with '.fm.json'.
        - Adds or updates entries in the index based on the contents of these files.
        - Removes entries that are (no longer) present in the directory.
        Uses batch operations for better performance.
        """
        directory = Path(directory) if isinstance(directory, str) else directory
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a valid directory.")

        file_paths = list(directory.glob("*.fm.json"))  # Convert to list to avoid generator exhaustion
        logger.info(f"Updating index from directory {directory}. Found {len(file_paths)} files.")

        # Get existing and local file information
        existing_entries = await self._get_existing_file_info()
        local_files = self._get_local_file_info(file_paths)

        # Determine what operations need to be performed
        to_insert, to_update, to_remove, unchanged = self._determine_operations(local_files, existing_entries)

        logger.info(
            f"Batch operations: {len(to_insert)} to insert, {len(to_update)} to update, {len(to_remove)} to remove, {unchanged} unchanged"
        )

        # Prepare entries for batch operations
        try:
            insert_entries, update_entries = await self._prepare_entries_for_batch(
                to_insert, to_update, allow_duplicate_synonyms
            )
        except Exception as e:
            logger.error(f"Failed to prepare entries for batch operations: {e}")
            raise

        # Execute batch operations
        await self._execute_batch_operations(to_remove, update_entries, insert_entries)

        added = len(to_insert)
        updated = len(to_update)
        removed = len(to_remove)

        logger.info(
            f"Index update complete: {added} added, {updated} updated, {removed} removed, {unchanged} unchanged."
        )
        return (added, updated, removed)

    async def to_markdown(self) -> str:
        """Converts the index to a Markdown table."""
        length = await self.count()
        header = f"# Finding Model Index\n\n{length} entries\n\n"
        header += "| ID | Name | Synonyms | Tags | Contributors | Attributes |\n"
        separator = "|----|------|----------|------|--------------|------------|\n"
        rows = []
        all_entries_sorted = self.index_collection.find({}, {"_id": 0}).sort("name", 1)
        async for entry_data in all_entries_sorted:
            entry = IndexEntry.model_validate(entry_data)
            md_filename = entry.filename.replace(".fm.json", ".md")
            entry_name_with_links = f"[{entry.name}](text/{md_filename}) [JSON](defs/{entry.filename})"
            row = (
                f"| {entry.oifm_id} | {entry_name_with_links} | {', '.join(entry.synonyms or [])} | "
                + f"{', '.join(entry.tags or [])} | {', '.join(entry.contributors or [])} | "
                + f"{', '.join(attr.name for attr in entry.attributes)} |\n"
            )
            rows.append(row)
        return header + separator + "".join(rows)
