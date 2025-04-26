import asyncio
from pathlib import Path

import click
from rich.console import Console

from .config import settings
from .finding_info import FindingInfo
from .finding_model import FindingModelBase, FindingModelFull
from .tools import (
    add_ids_to_finding_model,
    add_standard_codes_to_finding_model,
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
@click.option(
    "--output", "-o", type=click.Path(exists=False, dir_okay=True), help="Output file to save the finding info."
)
def make_info(finding_name: str, detailed: bool, output: Path | None) -> None:
    """Generate description/synonyms and more details/citations for a finding name."""

    console = Console()

    async def _do_make_info(finding_name: str, detailed: bool, output: Path | None) -> None:
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
        if output:
            with open(output, "w") as f:
                f.write(described_finding.model_dump_json(indent=2, exclude_none=True))
            console.print(f"[green]Saved finding info to [yellow]{output}")
        else:
            print_info_truncate_detail(console, described_finding)

    asyncio.run(_do_make_info(finding_name, detailed, output))


@cli.command()
@click.argument("finding_name", default="Pneumothorax")
@click.option("--tags", "-t", multiple=True, help="Tags to add to the model.")
@click.option("--with-codes", "-c", is_flag=True, help="Include standard index codes in the model.")
@click.option("--with-ids", "-i", is_flag=True, help="Include OIFM IDs in the model.")
@click.option("--source", "-s", help="Three/four letter code of originating organization (required for IDs).")
@click.option(
    "--output", "-o", type=click.Path(exists=False, dir_okay=True), help="Output file to save the finding model."
)
def make_stub_model(
    finding_name: str, tags: list[str], with_codes: bool, with_ids: bool, source: str | None, output: Path | None
) -> None:
    """Generate a simple finding model object (presence and change elements only) from a finding name."""

    console = Console()

    async def _do_make_stub_model(
        finding_name: str, tags: list[str], with_codes: bool, with_ids: bool, source: str | None, output: Path | None
    ) -> None:
        console.print(f"[gray] Getting stub model for [yellow bold]{finding_name}")
        # Get it from the database if it's already there
        with console.status("[bold green]Getting description and synonyms..."):
            described_finding = await describe_finding_name(finding_name)
        assert isinstance(described_finding, FindingInfo)
        stub = create_finding_model_stub_from_finding_info(described_finding, tags)
        if with_ids:
            if source and len(source) in [3, 4]:
                stub = add_ids_to_finding_model(stub, source.upper())  # type: ignore
            else:
                console.print("[red]Error: --source is required to generate IDs")
            if with_codes:
                add_standard_codes_to_finding_model(stub)  # type: ignore
        if with_codes and not with_ids:
            console.print("[red]Error: --with-codes requires --with-ids to be set")
        if output:
            with open(output, "w") as f:
                f.write(stub.model_dump_json(indent=2, exclude_none=True))
            console.print(f"[green]Saved finding model to [yellow]{output}")
        else:
            console.print_json(stub.model_dump_json(indent=2, exclude_none=True))

    asyncio.run(_do_make_stub_model(finding_name, tags, with_codes, with_ids, source, output))


@cli.command()
# Indicate that the argument should be a filename
@click.argument("finding_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o", type=click.Path(exists=False, dir_okay=True), help="Output file to save the finding info."
)
@click.option("--with-ids", "-i", is_flag=True, help="Include OIFM IDs in the model.")
@click.option("--source", "-s", help="Three/four letter code of originating organization (required for IDs).")
def markdown_to_fm(finding_path: Path, with_ids: bool, source: str | None, output: Path | None) -> None:
    """Convert markdown file to finding model format."""

    console = Console()

    async def _do_markdown_to_fm(finding_path: Path, with_ids: bool, source: str | None, output: Path | None) -> None:
        finding_name = finding_path.stem.replace("_", " ").replace("-", " ")
        with console.status("[bold green]Getting description..."):
            described_finding = await describe_finding_name(finding_name)
        print_info_truncate_detail(console, described_finding)
        assert isinstance(described_finding, FindingInfo), "Finding info not returned."

        with console.status("Creating model from Markdown description..."):
            model: FindingModelBase | FindingModelFull = await create_finding_model_from_markdown(
                described_finding, markdown_path=finding_path
            )
        if with_ids:
            if source and len(source) in [3, 4]:
                assert isinstance(model, FindingModelBase)
                model = add_ids_to_finding_model(model, source.upper())
            else:
                console.print("[red]Error: --source is required to generate IDs")
        if output:
            with open(output, "w") as f:
                f.write(model.model_dump_json(indent=2, exclude_none=True))
            console.print(f"[green]Saved finding model to [yellow]{output}")
        else:
            console.print_json(model.model_dump_json(indent=2, exclude_none=True))

    asyncio.run(_do_markdown_to_fm(finding_path, with_ids, source, output))


@cli.command()
@click.argument("finding_model_path", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option(
    "--output", "-o", type=click.Path(exists=False, dir_okay=True), help="Output file to save the finding model."
)
def fm_to_markdown(finding_model_path: Path, output: Path | None) -> None:
    """Convert finding model JSON file to Markdown format."""

    console = Console()
    console.print("[bold green]Loading finding model...")
    with open(finding_model_path, "r") as f:
        json = f.read()
        if "oifm_id" in json:
            fm_full = FindingModelFull.model_validate_json(json)
            markdown = fm_full.as_markdown()
        else:
            fm_base = FindingModelBase.model_validate_json(json)
            markdown = fm_base.as_markdown()
    if output:
        with open(output, "w") as f:
            f.write(markdown.strip() + "\n")
        console.print(f"[green]Saved Markdown to [yellow]{output}")
    else:
        from rich.markdown import Markdown

        console.print(Markdown(markdown))


if __name__ == "__main__":
    cli()
