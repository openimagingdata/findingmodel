import time

import httpx

from findingmodel import logger
from findingmodel.finding_model import (
    FindingModelBase,
    FindingModelFull,
    generate_oifm_id,
    generate_oifma_id,
)

GITHUB_IDS_URL = "https://raw.githubusercontent.com/openimagingdata/findingmodels/refs/heads/main/ids.json"


class IdManager:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or GITHUB_IDS_URL
        self.oifm_ids: dict[str, str] = {}
        self.attribute_ids: dict[str, tuple[str, str]] = {}

    def load_used_ids_from_github(self, refresh_cache: bool = False) -> None:
        """
        Load the OIFM and OIFMA IDs from the GitHub repository.
        :param refresh_cache: If True, refresh the cache by reloading the IDs from GitHub.
        :return: None
        """
        if self.oifm_ids and self.attribute_ids and not refresh_cache:
            logger.info("OIFM IDs and attribute IDs already loaded, skipping refresh.")
            return

        start_time = time.time()
        logger.info(f"Loading OIFM IDs and attribute IDs from: {self.url}")
        try:
            with httpx.Client() as client:
                response = client.get(self.url, timeout=5.0)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            logger.error("Timeout while loading GitHub IDs.")
            return
        except httpx.HTTPError as e:
            logger.error(f"Error loading GitHub IDs: {e}")
            return

        assert isinstance(data, dict), "Expected a dictionary from the GitHub IDs JSON"
        assert "oifm_ids" in data, "Expected 'oifm_ids' key in the GitHub IDs JSON"
        assert "attribute_ids" in data, "Expected 'attribute_ids' key in the GitHub IDs JSON"
        assert isinstance(data["oifm_ids"], dict), "Expected 'oifm_ids' to be a dictionary"
        assert isinstance(data["attribute_ids"], dict), "Expected 'attribute_ids' to be a dictionary"

        self.oifm_ids.clear()
        self.oifm_ids.update(data["oifm_ids"])
        self.attribute_ids.clear()
        self.attribute_ids.update({k: tuple(v) for k, v in data["attribute_ids"].items()})
        elapsed_time = time.time() - start_time
        logger.info(f"Loaded OIFM IDs and attribute IDs from GitHub in {elapsed_time:.2f} seconds.")
        logger.info(f"OIFM IDs {len(self.oifm_ids)} and attribute IDs {len(self.attribute_ids)} loaded from GitHub.")

    def add_ids_to_model(
        self,
        finding_model: FindingModelBase,
        source: str,
    ) -> FindingModelFull:
        """
        Generate and add OIFM IDs to the ID-less finding models with a source code.
        :param finding_model: The finding model to add IDs to.
        :param source: 3-4 letter code for the originating organization.
        :return: The finding model with IDs added.
        """
        self.load_used_ids_from_github()
        finding_model_dict = finding_model.model_dump()
        if "oifm_id" not in finding_model_dict:
            new_id = None
            while new_id is None or new_id in self.oifm_ids:
                new_id = generate_oifm_id(source)
            self.oifm_ids[new_id] = finding_model_dict["name"]
            finding_model_dict["oifm_id"] = new_id
        for attribute in finding_model_dict["attributes"]:
            if "oifma_id" not in attribute:
                new_id = None
                while new_id is None or new_id in self.attribute_ids:
                    new_id = generate_oifma_id(source)
                self.attribute_ids[new_id] = (finding_model_dict["oifm_id"], attribute["name"])
                attribute["oifma_id"] = new_id
        logger.debug(f"Adding IDs to finding model {finding_model.name} from source {source}")
        return FindingModelFull.model_validate(finding_model_dict)

    # Deprecated alias for backward compatibility
    def add_ids_to_finding_model(
        self,
        finding_model: FindingModelBase,
        source: str,
    ) -> FindingModelFull:
        """
        DEPRECATED: Use add_ids_to_model instead.
        Generate and add OIFM IDs to the ID-less finding models with a source code.
        """
        import warnings

        warnings.warn(
            "add_ids_to_finding_model is deprecated, use add_ids_to_model instead", DeprecationWarning, stacklevel=2
        )
        return self.add_ids_to_model(finding_model, source)


# Create a singleton instance of the IdManager
id_manager = IdManager()
