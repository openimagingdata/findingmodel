"""Convert Hood-style JSON definitions into :class:`findingmodel.finding_model.FindingModelFull`.

Vended so ``findingmodel-ai`` does not depend on the separate ``findingmodels`` content package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from findingmodel import FindingModelBase, FindingModelFull
from findingmodel.common import model_file_name
from findingmodel.tools import add_ids_to_model
from loguru import logger

from findingmodel_ai.authoring import create_info_from_name


class HoodJsonAdapter:
    """Structural adapter from Hood JSON dicts to OIFM finding models."""

    @staticmethod
    def _truncate_description(description: str, max_length: int = 500) -> str:
        if not description or len(description) <= max_length:
            return description
        return description[: max_length - 4] + "..."

    @staticmethod
    def _expand_short_name(name: str) -> str:
        expansions: dict[str, str] = {
            "T0": "T0 stage",
            "T1": "T1 stage",
            "T2": "T2 stage",
            "T3": "T3 stage",
            "T4": "T4 stage",
            "T5": "T5 stage",
            "T6": "T6 stage",
            "T7": "T7 stage",
            "T8": "T8 stage",
            "T9": "T9 stage",
            "C1": "C1 vertebra",
            "C2": "C2 vertebra",
            "C3": "C3 vertebra",
            "C4": "C4 vertebra",
            "C5": "C5 vertebra",
            "C6": "C6 vertebra",
            "C7": "C7 vertebra",
            "L1": "L1 vertebra",
            "L2": "L2 vertebra",
            "L3": "L3 vertebra",
            "L4": "L4 vertebra",
            "L5": "L5 vertebra",
            "S1": "S1 vertebra",
            "S2": "S2 vertebra",
            "S3": "S3 vertebra",
            "S4": "S4 vertebra",
            "S5": "S5 vertebra",
            "A0": "A0 stage",
            "A1": "A1 stage",
            "A2": "A2 stage",
            "A3": "A3 stage",
            "A4": "A4 stage",
            "B1": "B1 stage",
            "B2": "B2 stage",
            "B3": "B3 stage",
            "M1": "M1 stage",
            "M2": "M2 stage",
            "M3": "M3 stage",
            "M4": "M4 stage",
            "F1": "F1 stage",
            "F2": "F2 stage",
            "F3": "F3 stage",
            "F4": "F4 stage",
            "Cabg": "CABG surgery",
            "Ipmn": "IPMN lesion",
            "Picc": "PICC line",
            "CVC": "central venous catheter",
            "ECMO": "ECMO cannula",
            "LVAD": "LVAD device",
            "PFO": "PFO closure",
            "UIP": "UIP pattern",
        }
        if name in expansions:
            return expansions[name]
        if len(name) < 3:
            return f"{name} Value"
        if len(name) < 5:
            return f"{name} Finding"
        return name

    @staticmethod
    def _create_oidm_organization() -> dict[str, str]:
        return {
            "name": "Open Imaging Data Model",
            "code": "OIDM",
            "url": "https://openimagingdata.org",
        }

    @staticmethod
    def _create_mgb_organization() -> dict[str, str]:
        return {"name": "Massachusetts General Brigham", "code": "MGB"}

    @staticmethod
    def _create_person() -> dict[str, str]:
        return {
            "github_username": "hoodcm",
            "email": "chood@mgh.harvard.edu",
            "name": "C. Michael Hood, MD",
            "organization_code": "MGB",
        }

    @staticmethod
    def _create_default_contributors() -> list[dict[str, str]]:
        return [
            HoodJsonAdapter._create_oidm_organization(),
            HoodJsonAdapter._create_mgb_organization(),
            HoodJsonAdapter._create_person(),
        ]

    @staticmethod
    async def adapt_hood_json(hood_data: dict[str, Any], filename: str) -> FindingModelFull:  # noqa: C901
        finding_name = hood_data["finding_name"]

        expanded_finding_name = HoodJsonAdapter._expand_short_name(finding_name)
        if expanded_finding_name != finding_name:
            logger.info("Fixed short name {!r} -> {!r}", finding_name, expanded_finding_name)

        description = hood_data.get("description", "")
        if not description or len(description) < 5:
            try:
                finding_info = await create_info_from_name(expanded_finding_name)
                description = finding_info.description
                logger.info(
                    "Generated description for {!r} (first 100 chars): {:.100}s...",
                    expanded_finding_name,
                    description,
                )
            except Exception as e:
                logger.warning("Could not generate description for {!r}: {}", expanded_finding_name, e)
                description = f"Description for {expanded_finding_name}"
        else:
            description = HoodJsonAdapter._truncate_description(description)

        finding_model_dict: dict[str, Any] = {
            "name": expanded_finding_name.replace("_", " ").title(),
            "description": description,
            "attributes": [],
            "contributors": HoodJsonAdapter._create_default_contributors(),
        }

        for attribute in hood_data.get("attributes", []):
            attr_name = HoodJsonAdapter._expand_short_name(attribute["name"])
            if attr_name != attribute["name"]:
                logger.info("Fixed short attribute name {!r} -> {!r}", attribute["name"], attr_name)

            attr_description = attribute.get("description", "")
            if not attr_description or len(attr_description) < 5:
                attr_description = None
                logger.info("Short attribute description -> None for {!r} (schema default)", attr_name)
            else:
                attr_description = HoodJsonAdapter._truncate_description(attr_description)

            adapted_attr: dict[str, Any] = {
                "name": attr_name,
                "description": attr_description,
                "type": attribute["type"],
                "required": attribute.get("required", False),
            }

            if attribute["type"] == "choice":
                adapted_attr["max_selected"] = 1
                processed_values: list[dict[str, str]] = []
                for value in attribute["values"]:
                    processed_value: dict[str, str] = {"name": value["name"]}
                    if "description" in value and value["description"] != value["name"]:
                        value_description = value["description"]
                        if value_description and len(value_description) >= 5:
                            processed_value["description"] = value_description
                    processed_values.append(processed_value)
                adapted_attr["values"] = processed_values

            elif attribute["type"] == "numeric":
                adapted_attr["minimum"] = attribute.get("minimum", 0)
                adapted_attr["maximum"] = attribute.get("maximum", 100)
                adapted_attr["unit"] = attribute.get("unit", "unit")

            finding_model_dict["attributes"].append(adapted_attr)

        if not finding_model_dict["attributes"]:
            logger.warning("No attributes found in {}", filename)

        base_model = FindingModelBase(**finding_model_dict)
        return add_ids_to_model(base_model, source="MGB")

    @staticmethod
    async def process_file(input_file: str, output_dir: str) -> bool:
        try:
            try:
                with open(input_file, encoding="utf-8") as f:
                    hood_data = json.load(f)
            except UnicodeDecodeError:
                try:
                    with open(input_file, encoding="latin-1") as f:
                        hood_data = json.load(f)
                except UnicodeDecodeError:
                    with open(input_file, encoding="cp1252") as f:
                        hood_data = json.load(f)

            filename = Path(input_file).name

            fm = await HoodJsonAdapter.adapt_hood_json(hood_data, filename)
            output_file = Path(output_dir) / model_file_name(fm.name)

            output_file.parent.mkdir(parents=True, exist_ok=True)

            output_file.write_text(fm.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")

            logger.info("Adapted {} -> {}", filename, output_file.name)
            return True

        except Exception as e:
            logger.exception("Error processing {}: {}", input_file, e)
            return False

    @staticmethod
    async def process_directory(input_dir: str, output_dir: str) -> None:
        """Process all ``.json`` files in a directory (excludes ``*.cde.json``)."""
        os.makedirs(output_dir, exist_ok=True)

        successful_count = 0
        failed_count = 0
        total_files = 0

        logger.info("Starting Hood JSON adaptation from {}", input_dir)
        for filename in os.listdir(input_dir):
            if filename.endswith(".json") and not filename.endswith(".cde.json"):
                total_files += 1
                input_path = os.path.join(input_dir, filename)
                if await HoodJsonAdapter.process_file(input_path, output_dir):
                    successful_count += 1
                else:
                    failed_count += 1
                    logger.error("Failed to process {}", filename)

        unique_output_files = len([f for f in os.listdir(output_dir) if f.endswith(".fm.json")])

        logger.info(
            "Hood batch done: files={} ok={} failed={} outputs={}",
            total_files,
            successful_count,
            failed_count,
            unique_output_files,
        )
