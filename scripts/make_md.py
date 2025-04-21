import sys
from pathlib import Path

from findingmodel.finding_model import FindingModelFull


def main(filename: str) -> str:
    # Parse model
    model = FindingModelFull.model_validate_json(Path(filename).read_text())

    # Generate markdown
    return model.as_markdown()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_md.py <path_to_json>")
        sys.exit(1)
    print(main(sys.argv[1]))
