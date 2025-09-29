import asyncio
import random
from pathlib import Path

from findingmodel.finding_model import FindingModelFull
from findingmodel.tools.model_editor import EditResult, edit_model_markdown, export_model_for_editing


async def main() -> None:
    # Pick a random real definition from notebooks/data/defs
    data_dir = Path(__file__).parent / "data" / "defs"
    candidates = sorted(p for p in data_dir.glob("*.fm.json") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No definition files found in {data_dir}")
    selected = random.choice(candidates)
    model = FindingModelFull.model_validate_json(selected.read_text())
    md = export_model_for_editing(model)
    print("Original Markdown:\n", md)
    # Simulate user edit in the editable format: add a new attribute section
    appended = "\n".join([
        "### severity",
        "",
        "- mild",
        "- moderate",
        "- severe",
        "",
    ])
    edited_md = md + appended
    result: EditResult = await edit_model_markdown(model, edited_md)
    if result.rejections:
        print("\nRejections:")
        for r in result.rejections:
            print("-", r)
    if result.changes:
        print("\nChanges:")
        for change in result.changes:
            print("-", change)
    elif not result.rejections:
        print("\nNo changes recorded.")
    print("\nEdited Model JSON:\n", result.model.model_dump_json(indent=2, exclude_none=True))


# Demo: Load a model, export to markdown, simulate edit, and print result
if __name__ == "__main__":
    asyncio.run(main())
