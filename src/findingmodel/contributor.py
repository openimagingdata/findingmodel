"""
User and organization classes to indicate contributors to a finding model.
"""

from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import BaseModel, EmailStr, Field, HttpUrl, model_validator
from typing_extensions import Self

OrganizationCodeField = Annotated[
    str, Field(description="Short (3-4 letter) code for the organization", pattern=r"^[A-Z]{3,4}$")
]


class Organization(BaseModel):
    _org_registry: ClassVar[dict[str, Self]] = {}

    name: str = Field(..., description="Full name of the organization", min_length=5)
    code: OrganizationCodeField
    url: HttpUrl | None = Field(default=None, description="URL for more information about the organization")

    @model_validator(mode="after")
    def _register_org(self) -> Self:
        """
        Register the organization in the organization registry.
        """
        self._org_registry[self.code] = self
        return self

    @classmethod
    def get(cls, code: str) -> Self | None:
        """
        Retrieve an organization from the registry by its code.
        """
        return cls._org_registry.get(code)

    @classmethod
    def organizations(cls) -> list[Self]:
        """
        Retrieve all registered organizations.
        """
        return list(cls._org_registry.values())

    @classmethod
    def load_jsonl(cls, jsonl_file: str | Path) -> None:
        """
        Load organizations from a JSONL file.
        """
        jsonl_file = Path(jsonl_file) if isinstance(jsonl_file, str) else jsonl_file
        if not jsonl_file.exists() or not jsonl_file.is_file():
            raise FileNotFoundError(f"File {jsonl_file} not found.")
        with jsonl_file.open("r") as file:
            for line in file:
                cls.model_validate_json(line)

    @classmethod
    def save_jsonl(cls, jsonl_file: str | Path) -> None:
        """
        Save organizations to a JSONL file.
        """
        jsonl_file = Path(jsonl_file) if isinstance(jsonl_file, str) else jsonl_file
        with jsonl_file.open("w") as file:
            for org in cls._org_registry.values():
                file.write(org.model_dump_json(exclude_none=True) + "\n")


class Person(BaseModel):
    _person_registry: ClassVar[dict[str, Self]] = {}

    github_username: str = Field(..., description="GitHub username of the person", min_length=3)
    email: EmailStr = Field(..., description="Email address of the person")
    name: str = Field(..., description="Full name of the person", min_length=3)
    organization_code: OrganizationCodeField
    url: HttpUrl | None = Field(default=None, description="URL for more information about the person")

    @property
    def organization(self) -> Organization:
        """
        Retrieve the organization object from the registry.
        """
        if (org := Organization.get(self.organization_code)) is None:
            raise ValueError(f"Organization {self.organization_code} not found in registry.")
        return org

    @model_validator(mode="after")
    def _register_person(self) -> Self:
        """
        Register the person in the person registry.
        """
        self._person_registry[self.github_username] = self
        return self

    @classmethod
    def get(cls, username: str) -> Self | None:
        """
        Retrieve a person from the registry by their GitHub username.
        """
        return cls._person_registry.get(username)

    @classmethod
    def people(cls) -> list[Self]:
        """
        Retrieve all registered users.
        """
        return list(cls._person_registry.values())

    @classmethod
    def load_jsonl(cls, jsonl_file: str | Path) -> None:
        """
        Load users from a JSONL file.
        """
        jsonl_file = Path(jsonl_file) if isinstance(jsonl_file, str) else jsonl_file
        if not jsonl_file.exists() or not jsonl_file.is_file():
            raise FileNotFoundError(f"File {jsonl_file} not found.")
        with jsonl_file.open("r") as file:
            for line in file:
                cls.model_validate_json(line)

    @classmethod
    def save_jsonl(cls, jsonl_file: str | Path) -> None:
        """
        Save people to a JSONL file.
        """
        jsonl_file = Path(jsonl_file) if isinstance(jsonl_file, str) else jsonl_file
        with jsonl_file.open("w") as file:
            for person in cls._person_registry.values():
                file.write(person.model_dump_json(exclude_none=True) + "\n")
