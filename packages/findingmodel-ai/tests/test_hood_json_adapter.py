"""Tests for Hood JSON structural adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from findingmodel.finding_info import FindingInfo
from findingmodel.finding_model import FindingModelFull
from findingmodel_ai.hood_json_adapter import HoodJsonAdapter


@pytest.mark.asyncio
async def test_adapt_hood_json_minimal_with_mocked_ids() -> None:
    hood_data = {
        "finding_name": "TestNoduleFinding",
        "description": "A sufficiently long description for the finding name here.",
        "attributes": [
            {
                "name": "presence",
                "type": "choice",
                "required": True,
                "values": [{"name": "absent"}, {"name": "present"}],
            }
        ],
    }

    def fake_add_ids(base: object, source: str) -> FindingModelFull:
        from findingmodel.finding_model import FindingModelBase

        assert source == "MGB"
        assert isinstance(base, FindingModelBase)
        d = base.model_dump(exclude_none=False)
        d["oifm_id"] = "OIFM_MGB_123456"
        for i, attr in enumerate(d.get("attributes", [])):
            if attr.get("type") == "choice":
                values = []
                for j, v in enumerate(attr.get("values", [])):
                    values.append({
                        "value_code": f"OIFMA_MGB_00000{i}.{j}",
                        "name": v["name"],
                        "description": v.get("description"),
                    })
                attr["values"] = values
                attr["oifma_id"] = f"OIFMA_MGB_00000{i}"
        return FindingModelFull.model_validate(d)

    with patch("findingmodel_ai.hood_json_adapter.add_ids_to_model", side_effect=fake_add_ids):
        fm = await HoodJsonAdapter.adapt_hood_json(hood_data, "test.json")

    assert fm.oifm_id == "OIFM_MGB_123456"
    assert "Test" in fm.name or "test" in fm.name.lower()
    assert len(fm.attributes) == 1
    assert fm.attributes[0].name == "presence"


@pytest.mark.asyncio
async def test_adapt_hood_json_short_description_uses_create_info() -> None:
    hood_data = {
        "finding_name": "LongNameForTest",
        "description": "",
        "attributes": [
            {
                "name": "presence",
                "type": "choice",
                "required": True,
                "values": [{"name": "absent"}, {"name": "present"}],
            }
        ],
    }

    def fake_add_ids(base: object, source: str) -> FindingModelFull:
        from findingmodel.finding_model import FindingModelBase

        assert isinstance(base, FindingModelBase)
        d = base.model_dump(exclude_none=False)
        d["oifm_id"] = "OIFM_MGB_999999"
        for i, attr in enumerate(d.get("attributes", [])):
            if attr.get("type") == "choice":
                values = []
                for j, v in enumerate(attr.get("values", [])):
                    values.append({
                        "value_code": f"OIFMA_MGB_90000{i}.{j}",
                        "name": v["name"],
                        "description": v.get("description"),
                    })
                attr["values"] = values
                attr["oifma_id"] = f"OIFMA_MGB_90000{i}"
        return FindingModelFull.model_validate(d)

    mock_info = AsyncMock(
        return_value=FindingInfo(
            name="LongNameForTest",
            description="Generated description that is long enough for validation rules.",
        )
    )
    with (
        patch("findingmodel_ai.hood_json_adapter.create_info_from_name", mock_info),
        patch("findingmodel_ai.hood_json_adapter.add_ids_to_model", side_effect=fake_add_ids),
    ):
        fm = await HoodJsonAdapter.adapt_hood_json(hood_data, "x.json")

    assert "Generated description" in fm.description
    assert fm.oifm_id == "OIFM_MGB_999999"


def test_process_file_writes_output(tmp_path: Path) -> None:
    """Integration-style test with mocked ID assignment."""
    import asyncio

    inp = tmp_path / "n.json"
    inp.write_text(
        json.dumps({
            "finding_name": "BatchTest",
            "description": "Long enough description for batch test case here.",
            "attributes": [
                {
                    "name": "presence",
                    "type": "choice",
                    "required": True,
                    "values": [{"name": "absent"}, {"name": "present"}],
                }
            ],
        }),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_add_ids(base: object, source: str) -> FindingModelFull:

        d = base.model_dump(exclude_none=False)
        d["oifm_id"] = "OIFM_MGB_111111"
        for i, attr in enumerate(d.get("attributes", [])):
            if attr.get("type") == "choice":
                values = []
                for j, v in enumerate(attr.get("values", [])):
                    values.append({"value_code": f"OIFMA_MGB_10000{i}.{j}", "name": v["name"]})
                attr["values"] = values
                attr["oifma_id"] = f"OIFMA_MGB_10000{i}"
        return FindingModelFull.model_validate(d)

    async def _run() -> bool:
        with patch("findingmodel_ai.hood_json_adapter.add_ids_to_model", side_effect=fake_add_ids):
            return await HoodJsonAdapter.process_file(str(inp), str(out_dir))

    ok = asyncio.run(_run())

    assert ok
    fm_files = list(out_dir.glob("*.fm.json"))
    assert len(fm_files) == 1
    loaded = json.loads(fm_files[0].read_text(encoding="utf-8"))
    assert loaded["oifm_id"] == "OIFM_MGB_111111"
