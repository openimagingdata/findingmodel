"""Tests for hood_pipeline loaders and normalize_output (no live API)."""

from pathlib import Path

import pytest
from findingmodel_ai.hood_pipeline import load_definition, should_process_file
from findingmodel_ai.hood_pipeline.normalize_output import normalize_for_validation, strip_sub_finding_attributes


def test_should_process_json_priority_over_md(tmp_path: Path) -> None:
    md = tmp_path / "foo.md"
    js = tmp_path / "foo.json"
    md.write_text("# x", encoding="utf-8")
    js.write_text("{}", encoding="utf-8")
    all_files = [md, js]
    assert should_process_file(js, all_files) is True
    assert should_process_file(md, all_files) is False


def test_should_process_md_when_no_json_sibling(tmp_path: Path) -> None:
    md = tmp_path / "bar.md"
    md.write_text("# y", encoding="utf-8")
    all_files = [md]
    assert should_process_file(md, all_files) is True


def test_should_skip_cde_json(tmp_path: Path) -> None:
    cde = tmp_path / "x.cde.json"
    cde.write_text("{}", encoding="utf-8")
    all_files = [cde]
    assert should_process_file(cde, all_files) is False


def test_should_ignore_non_md_json(tmp_path: Path) -> None:
    txt = tmp_path / "nope.txt"
    txt.write_text("a", encoding="utf-8")
    assert should_process_file(txt, [txt]) is False


@pytest.mark.asyncio
async def test_load_definition_md(tmp_path: Path) -> None:
    p = tmp_path / "d.md"
    p.write_text("# Hello", encoding="utf-8")
    data, md, ft = await load_definition(p)
    assert data is None
    assert md == "# Hello"
    assert ft == "md"


@pytest.mark.asyncio
async def test_load_definition_json(tmp_path: Path) -> None:
    p = tmp_path / "d.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    data, md, ft = await load_definition(p)
    assert data == {"a": 1}
    assert md is None
    assert ft == "json"


def test_normalize_anatomic_locations_invalid_cleared() -> None:
    d = {
        "name": "Test",
        "anatomic_locations": [{"name": "Chest"}],  # missing system/code shape
    }
    out = normalize_for_validation(d)
    assert out["anatomic_locations"] is None


def test_normalize_anatomic_locations_valid_kept() -> None:
    d = {
        "name": "Test",
        "anatomic_locations": [{"system": "RADLEX", "code": "RID123"}],
    }
    out = normalize_for_validation(d)
    assert out["anatomic_locations"] == [{"system": "RADLEX", "code": "RID123"}]


def test_normalize_contributors_mgb_short_name() -> None:
    d = {
        "name": "Test",
        "contributors": [{"name": "mgb", "code": "MGB"}],
    }
    out = normalize_for_validation(d)
    assert out["contributors"] == [{"name": "Massachusetts General Brigham", "code": "MGB"}]


def test_strip_sub_finding_attributes_removes_matching_names() -> None:
    d = {
        "name": "Parent",
        "attributes": [
            {"name": "Aneurysm"},
            {"name": "Aneurysm size"},
            {"name": "Other"},
        ],
    }
    out = strip_sub_finding_attributes(d, ["aneurysm"])
    names = {a["name"] for a in out["attributes"]}
    assert names == {"Other"}
