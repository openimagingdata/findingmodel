import asyncio
from pathlib import Path

import click
from findingmodel.config import ConfigurationError
from findingmodel.finding_info import FindingInfo
from findingmodel.finding_model import FindingModelBase, FindingModelFull
from rich.console import Console

from findingmodel.tools import add_ids_to_model, add_standard_codes_to_model
from findingmodel_ai.authoring.description import add_details_to_info, create_info_from_name
from findingmodel_ai.authoring.markdown_in import create_model_from_markdown
from findingmodel_ai.config import settings
from findingmodel_ai.metadata import assign_metadata
from findingmodel_ai.observability import ensure_logfire_configured

console = Console()


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """findingmodel-ai: AI-powered tools for finding model authoring."""
    # Validate AI model API keys before running AI commands.
    # The ontology subcommand only needs BIOONTOLOGY_API_KEY, not AI model keys.
    if ctx.invoked_subcommand not in (None, "ontology"):
        try:
            settings.validate_default_model_keys()
        except ConfigurationError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None


def print_info_truncate_detail(finding_info: FindingInfo) -> None:
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

    async def _do_make_info(finding_name: str, detailed: bool, output: Path | None) -> None:
        with console.status("[bold green]Getting description and synonyms..."):
            described_finding = await create_info_from_name(finding_name)
        if not isinstance(described_finding, FindingInfo):
            raise ValueError("Finding info not returned.")
        if detailed:
            with console.status("Getting detailed information... "):
                detailed_response = await add_details_to_info(described_finding)
            if not isinstance(detailed_response, FindingInfo):
                raise ValueError("Detailed finding info not returned.")
            described_finding = detailed_response
        if output:
            with open(output, "w") as f:
                f.write(described_finding.model_dump_json(indent=2, exclude_none=True))
            console.print(f"[green]Saved finding info to [yellow]{output}")
        else:
            print_info_truncate_detail(described_finding)

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

    async def _do_make_stub_model(
        finding_name: str, tags: list[str], with_codes: bool, with_ids: bool, source: str | None, output: Path | None
    ) -> None:
        from findingmodel.create_stub import create_model_stub_from_info

        console.print(f"[gray] Getting stub model for [yellow bold]{finding_name}")
        with console.status("[bold green]Getting description and synonyms..."):
            described_finding = await create_info_from_name(finding_name)
        assert isinstance(described_finding, FindingInfo)
        stub: FindingModelBase | FindingModelFull = create_model_stub_from_info(described_finding, list(tags))
        if with_ids:
            if source and len(source) in [3, 4]:
                stub = add_ids_to_model(stub, source.upper())
            else:
                console.print("[red]Error: --source is required to generate IDs")
            if with_codes:
                assert isinstance(stub, FindingModelFull)
                add_standard_codes_to_model(stub)
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
@click.argument("finding_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o", type=click.Path(exists=False, dir_okay=True), help="Output file to save the finding info."
)
@click.option("--with-ids", "-i", is_flag=True, help="Include OIFM IDs in the model.")
@click.option("--source", "-s", help="Three/four letter code of originating organization (required for IDs).")
def markdown_to_fm(finding_path: Path, with_ids: bool, source: str | None, output: Path | None) -> None:
    """Convert markdown file to finding model format."""

    async def _do_markdown_to_fm(finding_path: Path, with_ids: bool, source: str | None, output: Path | None) -> None:
        finding_name = finding_path.stem.replace("_", " ").replace("-", " ")
        with console.status("[bold green]Getting description..."):
            described_finding = await create_info_from_name(finding_name)
        print_info_truncate_detail(described_finding)
        assert isinstance(described_finding, FindingInfo), "Finding info not returned."

        markdown_text = finding_path.read_text()

        with console.status("Creating model from Markdown description..."):
            model: FindingModelBase | FindingModelFull = await create_model_from_markdown(
                described_finding, markdown_text=markdown_text
            )
        if with_ids:
            if source and len(source) in [3, 4]:
                assert isinstance(model, FindingModelBase)
                model = add_ids_to_model(model, source.upper())
            else:
                console.print("[red]Error: --source is required to generate IDs")
        if output:
            with open(output, "w") as f:
                f.write(model.model_dump_json(indent=2, exclude_none=True))
            console.print(f"[green]Saved finding model to [yellow]{output}")
        else:
            console.print_json(model.model_dump_json(indent=2, exclude_none=True))

    asyncio.run(_do_markdown_to_fm(finding_path, with_ids, source, output))


@cli.command("assign-metadata")
@click.argument("finding_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Output path for the updated finding model JSON. Defaults to stdout.",
)
@click.option(
    "--review-output",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Optional output path for the metadata-assignment review JSON.",
)
@click.option(
    "--logfire",
    is_flag=True,
    help="Opt in to Logfire instrumentation for this run, including outbound HTTP calls.",
)
def assign_metadata_command(
    finding_path: Path,
    output: Path | None,
    review_output: Path | None,
    logfire: bool,
) -> None:
    """Assign canonical structured metadata to an existing .fm.json model."""

    async def _do_assign_metadata() -> None:
        if logfire:
            ensure_logfire_configured(console=False)

        finding_model = FindingModelFull.model_validate_json(finding_path.read_text())

        with console.status("[bold green]Assigning canonical metadata..."):
            result = await assign_metadata(finding_model)

        model_json = result.model.model_dump_json(indent=2, exclude_none=True)
        review_json = result.review.model_dump_json(indent=2, exclude_none=True)

        if output:
            output.write_text(model_json + "\n")
            console.print(f"[green]Saved updated finding model to [yellow]{output}[/yellow][/green]")
        else:
            console.print_json(model_json)

        if review_output:
            review_output.write_text(review_json + "\n")
            console.print(f"[green]Saved metadata review to [yellow]{review_output}[/yellow][/green]")

        if result.review.logfire_trace_id:
            click.echo(f"Logfire trace_id: {result.review.logfire_trace_id}", err=True)

    asyncio.run(_do_assign_metadata())


@cli.group()
def ontology() -> None:
    """Search medical ontologies via BioOntology.org."""


@ontology.command("search")
@click.argument("query")
@click.option(
    "--ontology",
    "-o",
    multiple=True,
    metavar="CODE",
    help="Ontology to search (repeatable; default: SNOMEDCT, RADLEX, LOINC).",
)
@click.option("--max-results", "-n", default=20, show_default=True, help="Maximum number of results.")
@click.option("--exact", is_flag=True, help="Require exact match only.")
@click.option(
    "--semantic-type",
    "-t",
    multiple=True,
    metavar="TYPE",
    help="UMLS semantic type to filter by (repeatable; e.g. T047 for diseases).",
)
def ontology_search(
    query: str, ontology: tuple[str, ...], max_results: int, exact: bool, semantic_type: tuple[str, ...]
) -> None:
    """Search medical ontologies for QUERY via BioOntology.org."""
    from rich.table import Table

    from findingmodel_ai.search.bioontology import BioOntologySearchClient, BioOntologySearchResult

    if not settings.bioontology_api_key:
        console.print("[red]Error:[/red] BIOONTOLOGY_API_KEY is not set. Add it to .env or the environment.")
        raise SystemExit(1)

    ontologies = list(ontology) or None  # None → client uses its defaults
    semantic_types = list(semantic_type) or None  # None → no semantic type filter

    async def _run() -> list[BioOntologySearchResult]:
        with console.status(f"[bold green]Searching BioOntology for {query!r}..."):
            async with BioOntologySearchClient() as client:
                return await client.search_all_pages(
                    query=query,
                    ontologies=ontologies,
                    max_results=max_results,
                    require_exact_match=exact,
                    semantic_types=semantic_types,
                )

    results = asyncio.run(_run())

    if not results:
        console.print(f"No results for [bold]{query!r}[/bold].")
        return

    table = Table(title=f"BioOntology: {query!r}", show_lines=True)
    table.add_column("Ontology", style="bold cyan", no_wrap=True)
    table.add_column("Code", no_wrap=True)
    table.add_column("Label", style="green")
    table.add_column("Definition")

    for r in results:
        code = r.concept_id.split("/")[-1]
        defn = r.definition or ""
        if len(defn) > 100:
            defn = defn[:97] + "..."
        table.add_row(r.ontology, code, r.pref_label, defn)

    console.print(table)
    console.print(f"[dim]{len(results)} result(s)[/dim]")


if __name__ == "__main__":
    cli()
