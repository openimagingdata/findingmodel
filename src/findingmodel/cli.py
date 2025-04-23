import asyncio
from pathlib import Path

import click
from rich.console import Console

from .config import settings
from .finding_info import FindingInfo
from .tools import (
    create_finding_model_from_markdown,
    create_finding_model_stub_from_finding_info,
    describe_finding_name,
    get_detail_on_finding,
)


@click.group()
def cli() -> None:
    pass


@cli.command()
def config() -> None:
    """Show the currently active configuration."""
    console = Console()
    console.print("[yellow bold]Finding Model Forge configuration:")
    console.print_json(settings.model_dump_json())


def print_info_truncate_detail(console: Console, finding_info: FindingInfo) -> None:
    out = finding_info.model_dump()
    if out.get("detail") and len(out["detail"]) > 100:
        out["detail"] = out["detail"][0:100] + "..."
    console.print(out)


@cli.command()
@click.argument("finding_name", default="Pneumothorax")
@click.option("--detailed", "-d", is_flag=True, help="Get detailed information on the finding.")
def make_info(finding_name: str, detailed: bool) -> None:
    """Generate description/synonyms and more details/citations for a finding name."""

    console = Console()

    async def _do_make_info(finding_name: str, detailed: bool) -> None:
        with console.status("[bold green]Getting description and synonyms..."):
            described_finding = await describe_finding_name(finding_name)
        if not isinstance(described_finding, FindingInfo):
            raise ValueError("Finding info not returned.")
        if detailed:
            with console.status("Getting detailed information... "):
                detailed_response = await get_detail_on_finding(described_finding)
            if not isinstance(detailed_response, FindingInfo):
                raise ValueError("Detailed finding info not returned.")
            described_finding = detailed_response
        print_info_truncate_detail(console, described_finding)

    asyncio.run(_do_make_info(finding_name, detailed))


@cli.command()
@click.argument("finding_name", default="Pneumothorax")
@click.option("--tags", "-t", multiple=True, help="Tags to add to the model.")
def make_stub_model(finding_name: str, tags: list[str]) -> None:
    """Generate a simple finding model object (presence and change elements only) from a finding name."""

    console = Console()

    async def _do_make_stub_model(finding_name: str, tags: list[str]) -> None:
        console.print(f"[gray] Getting stub model for [yellow bold]{finding_name}")
        # Get it from the database if it's already there
        with console.status("[bold green]Getting description and synonyms..."):
            described_finding = await describe_finding_name(finding_name)
        assert isinstance(described_finding, FindingInfo)
        stub = create_finding_model_stub_from_finding_info(described_finding, tags)
        console.print("Saving to database...")
        console.print_json(stub.model_dump_json())

    asyncio.run(_do_make_stub_model(finding_name, tags))


@cli.command()
# Indicate that the argument should be a filename
@click.argument("finding_path", type=click.Path(exists=True, path_type=Path))
def markdown_to_fm(finding_path: Path) -> None:
    """Convert markdown file to finding model format."""

    console = Console()

    async def _do_markdown_to_fm(finding_path: Path) -> None:
        finding_name = finding_path.stem.replace("_", " ").replace("-", " ")
        with console.status("[bold green]Getting description..."):
            described_finding = await describe_finding_name(finding_name)
        print_info_truncate_detail(console, described_finding)
        assert isinstance(described_finding, FindingInfo), "Finding info not returned."

        with console.status("Creating model from Markdown description..."):
            model = await create_finding_model_from_markdown(described_finding, markdown_path=finding_path)
        console.print(model.model_dump())

    asyncio.run(_do_markdown_to_fm(finding_path))


if __name__ == "__main__":
    cli()
