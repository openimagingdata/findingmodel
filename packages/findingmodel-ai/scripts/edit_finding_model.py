from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from findingmodel.finding_model import FindingModelFull
from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID
from findingmodel_ai.authoring.editor import (
    EditResult,
    assign_real_attribute_ids,
    edit_model_markdown,
    edit_model_natural_language,
    export_model_for_editing,
)
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import prompt as prompt_multiline
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
T = TypeVar("T")
CommandHandler = Callable[["DemoState", list[str]], None]


class DemoState:
    """Mutable state for the interactive demo session."""

    def __init__(self, model: FindingModelFull, save_path: Path) -> None:
        self.current_model = model
        self.save_path = save_path
        self._saved_model_json = model.model_dump_json(exclude_none=True)
        self.dirty = False

    @property
    def markdown(self) -> str:
        return export_model_for_editing(self.current_model)

    def mark_updated(self, result: EditResult) -> None:
        self.current_model = result.model
        current_json = result.model.model_dump_json(exclude_none=True)
        self.dirty = current_json != self._saved_model_json

    def reset(self) -> None:
        self.current_model = FindingModelFull.model_validate_json(self._saved_model_json)
        self.dirty = False

    def record_save(self) -> None:
        self._saved_model_json = self.current_model.model_dump_json(exclude_none=True)
        self.dirty = False


async def _await_with_timer(coro: Awaitable[T], message: str) -> T:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task_id = progress.add_task(message, start=True)
        try:
            result = await coro
        finally:
            progress.update(task_id, completed=1)
    return result


def run_with_timer(coro: Awaitable[T], message: str) -> T:
    async def _runner() -> T:
        return await _await_with_timer(coro, message)

    return asyncio.run(_runner())


def load_model(path: Path) -> FindingModelFull:
    data = json.loads(path.read_text())
    return FindingModelFull.model_validate(data)


def save_model(model: FindingModelFull, path: Path) -> None:
    path.write_text(model.model_dump_json(indent=2, exclude_none=True))


def render_rejections(rejections: list[str]) -> None:
    if not rejections:
        return
    table = Table(title="Rejections", show_edge=True)
    table.add_column("#", style="cyan", justify="right")
    for idx, reason in enumerate(rejections, start=1):
        table.add_row(str(idx), reason)
    console.print(table)


def render_changes(changes: list[str]) -> None:
    if not changes:
        return
    table = Table(title="Changes", show_edge=True)
    table.add_column("#", style="green", justify="right")
    table.add_column("Summary", style="green")
    for idx, text in enumerate(changes, start=1):
        table.add_row(str(idx), text)
    console.print(table)


def display_current(state: DemoState) -> None:
    subtitle = f"{state.current_model.name} ({state.current_model.oifm_id})"
    console.print(Panel(Markdown(state.markdown), title="Current Markdown", subtitle=subtitle))


def explain_commands() -> None:
    table = Table(title="Commands", show_edge=True)
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="green")
    table.add_row("/help", "Show this help")
    table.add_row(
        "/edit",
        "Open a multiline Markdown editor (Esc followed by Enter to submit, Ctrl+C to cancel)",
    )
    table.add_row(
        "/command [text]",
        "Send text through the natural-language editor (omit [text] to be prompted)",
    )
    table.add_row(
        "<free text>",
        "Any input without a leading slash routes to the natural-language editor",
    )
    table.add_row(
        "/save [path]",
        "Persist the model JSON to disk (defaults to the original path)",
    )
    table.add_row(
        "/reset",
        "Revert to the last saved model",
    )
    table.add_row(
        "/quit",
        "Exit the session",
    )
    console.print(table)


def prompt_markdown_edit(state: DemoState) -> str | None:
    console.print("[bold]Entering Markdown editor.[/] Press Esc then Enter to submit, or Ctrl+C to cancel.")
    try:
        edited = prompt_multiline(
            HTML("<skyblue>Markdown edit> </skyblue>"),
            default=state.markdown,
            multiline=True,
        )
    except KeyboardInterrupt:
        console.print("[yellow]Markdown edit cancelled.[/]")
        return None
    except EOFError:
        console.print("[yellow]Markdown edit aborted.[/]")
        return None
    return edited


def prompt_command_text(provided: list[str]) -> str:
    if provided:
        return " ".join(provided)
    console.print("Enter a natural-language command. Submit with Enter. Ctrl+C to cancel.")
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())
    try:
        return session.prompt("command> ")
    except KeyboardInterrupt:
        console.print("[yellow]Command cancelled.[/]")
        return ""
    except EOFError:
        console.print("[yellow]Command aborted.[/]")
        return ""


def apply_edit(state: DemoState, result: EditResult, context: str) -> None:
    state.mark_updated(result)
    if result.rejections:
        console.print(f"[bold yellow]{context} completed with rejections.[/]")
        render_rejections(result.rejections)
    elif state.dirty:
        console.print(f"[bold green]{context} applied successfully.[/]")
    else:
        console.print(f"[cyan]{context} made no changes.[/]")

    if result.changes:
        render_changes(result.changes)

    if state.dirty:
        display_current(state)
    elif not result.rejections:
        console.print("[dim]Model unchanged; JSON output left as-is.[/dim]")


def handle_natural_language(state: DemoState, text: str) -> None:
    if not text.strip():
        console.print("[yellow]No command provided.[/]")
        return
    try:
        result = run_with_timer(
            edit_model_natural_language(state.current_model, text),
            "Natural-language edit in progress...",
        )
    except Exception as exc:
        console.print(f"[red]Natural-language edit failed: {exc}[/]")
        return
    apply_edit(state, result, "Natural-language edit")


def handle_markdown_edit(state: DemoState) -> None:
    current_markdown = state.markdown
    edited = prompt_markdown_edit(state)
    if edited is None or edited.strip() == current_markdown.strip():
        if edited is not None:
            console.print("[yellow]No Markdown changes detected.[/]")
        return
    try:
        result = run_with_timer(
            edit_model_markdown(state.current_model, edited),
            "Markdown edit in progress...",
        )
    except Exception as exc:
        console.print(f"[red]Markdown edit failed: {exc}[/]")
        return
    apply_edit(state, result, "Markdown edit")


def handle_save(state: DemoState, maybe_path: list[str]) -> None:
    target = state.save_path if not maybe_path else Path(maybe_path[0]).expanduser()
    if state.dirty:
        placeholders_present = any(
            getattr(attr, "oifma_id", None) == PLACEHOLDER_ATTRIBUTE_ID
            for attr in getattr(state.current_model, "attributes", [])
        )
        if placeholders_present:
            state.current_model = assign_real_attribute_ids(state.current_model)
            console.print("[cyan]Assigned permanent attribute IDs before saving.[/]")
    save_model(state.current_model, target)
    state.save_path = target
    state.record_save()
    console.print(f"[green]Model saved to {target}[/]")


def _handle_command_input(state: DemoState, args: list[str]) -> None:
    text = prompt_command_text(args)
    if text:
        handle_natural_language(state, text)


def _handle_reset(state: DemoState, _: list[str]) -> None:
    state.reset()
    console.print("[cyan]Reverted to last saved model.[/]")
    display_current(state)


COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "/help": lambda state, _: explain_commands(),
    "/edit": lambda state, _: handle_markdown_edit(state),
    "/command": _handle_command_input,
    "/save": handle_save,
    "/reset": _handle_reset,
}


def handle_command(state: DemoState, command: str, args: list[str]) -> bool:
    if command == "/quit":
        console.print("[bold]Goodbye.[/]")
        return False
    handler = COMMAND_HANDLERS.get(command)
    if handler is None:
        console.print(f"[red]Unknown command: {command}[/]")
        return True
    handler(state, args)
    return True


def run_session(state: DemoState) -> None:
    history = InMemoryHistory()
    session: PromptSession[str] = PromptSession(history=history)
    completer = NestedCompleter.from_nested_dict({**dict.fromkeys(COMMAND_HANDLERS, None), "/quit": None})

    explain_commands()

    while True:
        prompt_label = "fm* > " if state.dirty else "fm > "
        try:
            user_input = session.prompt(prompt_label, completer=completer)
        except KeyboardInterrupt:
            console.print("[yellow]Use /quit to exit. Press Enter to continue.")
            continue
        except EOFError:
            console.print("\n[bold]Exiting.[/]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            parts = shlex.split(stripped)
            command = parts[0]
            args = parts[1:]
            if not handle_command(state, command, args):
                break
            continue

        handle_natural_language(state, stripped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive demo for editing a FindingModel using natural language or Markdown edits.",
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the .fm.json file to edit",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path to persist edits when using /save (defaults to the input file)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser()
    if not model_path.exists():
        console.print(f"[red]Model file not found: {model_path}[/]")
        return 1

    try:
        model = load_model(model_path)
    except Exception as exc:
        console.print(f"[red]Failed to load model: {exc}[/]")
        return 1

    state = DemoState(model, args.save_path.expanduser() if args.save_path else model_path)
    console.print(
        Panel(
            f"Loaded [bold]{state.current_model.name}[/] ({state.current_model.oifm_id})\n"
            "Type /help for available commands. Natural-language input without a leading slash is sent directly to the agent.",
            title="Finding Model Editor Demo",
        )
    )
    display_current(state)

    run_session(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
