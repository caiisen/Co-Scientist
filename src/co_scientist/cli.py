from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from co_scientist.config import load_config
from co_scientist.memory import SQLiteStore
from co_scientist.supervisor import (
    collect_session_stats,
    export_session_markdown,
    run_new_session,
    run_resume_session,
)

app = typer.Typer(help="Co-Scientist reproduction CLI.")
console = Console()
DEFAULT_DB_PATH = Path("runs") / "co_scientist.sqlite"


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
    db_path: Annotated[
        Path,
        typer.Option("--db-path", help="SQLite database path."),
    ] = DEFAULT_DB_PATH,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print detailed phase, task, input, and output logs."),
    ] = False,
) -> None:
    """Start a Phase 5 research session in the foreground."""
    config = _load_config_or_exit(session_config, None, None, None)
    goal = goal_file.read_text(encoding="utf-8").strip()
    if not goal:
        console.print("[red]Goal file is empty.[/red]")
        raise typer.Exit(code=2)
    try:
        session_id = _run_async(
            run_new_session(db_path=db_path, config=config, goal=goal, verbose=verbose)
        )
    except Exception as exc:
        console.print(f"[red]Session failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Session complete: {session_id}")


@app.command()
def resume(
    session_id: Annotated[str, typer.Argument(help="Session identifier.")],
    session_config: Annotated[
        Path | None,
        typer.Option("--session-config", help="YAML config for this research session."),
    ] = None,
    db_path: Annotated[
        Path,
        typer.Option("--db-path", help="SQLite database path."),
    ] = DEFAULT_DB_PATH,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print detailed phase, task, input, and output logs."),
    ] = False,
) -> None:
    """Resume pending Phase 5 work for a research session."""
    config = _load_config_or_exit(session_config, None, None, None)
    try:
        _run_async(
            run_resume_session(
                db_path=db_path,
                config=config,
                session_id=session_id,
                verbose=verbose,
            )
        )
    except Exception as exc:
        console.print(f"[red]Resume failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Session resumed: {session_id}")


@app.command()
def status(
    session_id: Annotated[str, typer.Argument(help="Session identifier.")],
    db_path: Annotated[
        Path,
        typer.Option("--db-path", help="SQLite database path."),
    ] = DEFAULT_DB_PATH,
) -> None:
    """Show Phase 5 session status."""
    try:
        stats, review_count = _run_async(_load_status(db_path, session_id))
    except Exception as exc:
        console.print(f"[red]Status failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Session: {session_id}")
    console.print(f"Hypotheses: {stats.hypothesis_count}")
    console.print(f"Reviews: {review_count}")
    console.print(f"Matches: {stats.match_count}")
    console.print(f"Matches per idea: {stats.matches_per_idea:.2f}")
    if stats.top_hypotheses:
        console.print("Top hypotheses:")
        for hypothesis in stats.top_hypotheses:
            console.print(f"  {hypothesis.elo}: {hypothesis.summary}")
    console.print("Tasks:")
    for task_status, count in sorted(stats.tasks_by_status.items(), key=lambda item: item[0].value):
        console.print(f"  {task_status.value}: {count}")


@app.command()
def export(
    session_id: Annotated[str, typer.Argument(help="Session identifier.")],
    session_config: Annotated[
        Path | None,
        typer.Option("--session-config", help="YAML config for this research session."),
    ] = None,
    db_path: Annotated[
        Path,
        typer.Option("--db-path", help="SQLite database path."),
    ] = DEFAULT_DB_PATH,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write markdown to this file."),
    ] = None,
) -> None:
    """Export Phase 4 output as markdown."""
    config = _load_config_or_exit(session_config, None, None, None)
    try:
        markdown = _run_async(
            export_session_markdown(db_path=db_path, config=config, session_id=session_id)
        )
    except Exception as exc:
        console.print(f"[red]Export failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if output is None:
        console.print(markdown)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    console.print(f"Wrote {output}")


async def _load_status(db_path: Path, session_id: str):
    async with SQLiteStore(db_path) as store:
        if await store.get_session(session_id) is None:
            raise ValueError(f"unknown session: {session_id}")
        stats = await collect_session_stats(store, session_id)
        review_count = await store.count_reviews(session_id)
        return stats, review_count


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)
