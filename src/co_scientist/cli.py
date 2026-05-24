from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from co_scientist.config import load_config

app = typer.Typer(help="Co-Scientist reproduction CLI.")
console = Console()


def _load_config_or_exit(
    session_config: Path | None,
    max_ideas: int | None,
    max_matches_per_idea: int | None,
    worker_concurrency: int | None,
):
    overrides = {
        "runtime.max_ideas": max_ideas,
        "runtime.max_matches_per_idea": max_matches_per_idea,
        "runtime.worker_concurrency": worker_concurrency,
    }
    try:
        return load_config(session_path=session_config, cli_overrides=overrides)
    except Exception as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


@app.callback()
def main() -> None:
    """Run Co-Scientist commands."""


@app.command("config-show")
def config_show(
    session_config: Annotated[
        Path | None,
        typer.Option("--session-config", help="YAML config for this research session."),
    ] = None,
    max_ideas: Annotated[
        int | None,
        typer.Option("--max-ideas", help="Override runtime.max_ideas."),
    ] = None,
    max_matches_per_idea: Annotated[
        int | None,
        typer.Option("--max-matches-per-idea", help="Override runtime.max_matches_per_idea."),
    ] = None,
    worker_concurrency: Annotated[
        int | None,
        typer.Option("--worker-concurrency", help="Override runtime.worker_concurrency."),
    ] = None,
) -> None:
    """Print the resolved Phase 0 configuration."""
    config = _load_config_or_exit(
        session_config,
        max_ideas,
        max_matches_per_idea,
        worker_concurrency,
    )
    console.print(yaml.safe_dump(config.model_dump(), sort_keys=False))


@app.command()
def new(
    goal_file: Annotated[Path, typer.Argument(help="Research goal file.")],
    session_config: Annotated[
        Path | None,
        typer.Option("--session-config", help="YAML config for this research session."),
    ] = None,
) -> None:
    """Start a new research session. Implemented in a later phase."""
    _load_config_or_exit(session_config, None, None, None)
    console.print(
        f"Phase 0 scaffold is ready. Session execution is not implemented yet: {goal_file}"
    )


@app.command()
def resume(session_id: Annotated[str, typer.Argument(help="Session identifier.")]) -> None:
    """Resume a research session. Implemented in a later phase."""
    console.print(f"Resume is not implemented in Phase 0: {session_id}")


@app.command()
def status(session_id: Annotated[str, typer.Argument(help="Session identifier.")]) -> None:
    """Show session status. Implemented in a later phase."""
    console.print(f"Status is not implemented in Phase 0: {session_id}")


@app.command()
def export(session_id: Annotated[str, typer.Argument(help="Session identifier.")]) -> None:
    """Export session output. Implemented in a later phase."""
    console.print(f"Export is not implemented in Phase 0: {session_id}")
