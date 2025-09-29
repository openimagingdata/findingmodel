import random
from pathlib import Path

from findingmodel.finding_model import FindingModelFull
from findingmodel.tools.model_editor import EditResult, edit_model_natural_language

# Demo: Load a model, apply a natural language command, and print result
if __name__ == "__main__":
    # Pick a random real definition from notebooks/data/defs
    data_dir = Path(__file__).parent / "data" / "defs"
    candidates = sorted(p for p in data_dir.glob("*.fm.json") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No definition files found in {data_dir}")
    selected = random.choice(candidates)

    model = FindingModelFull.model_validate_json(selected.read_text())
    command = "Add attribute 'severity' with values mild, moderate, severe."
    import asyncio

    result: EditResult = asyncio.run(edit_model_natural_language(model, command))
    if result.rejections:
        print("Rejections:")
        for r in result.rejections:
            print("-", r)
    if result.changes:
        print("Changes:")
        for change in result.changes:
            print("-", change)
    elif not result.rejections:
        print("No changes recorded.")
    print(result.model.model_dump_json(indent=2, exclude_none=True))
