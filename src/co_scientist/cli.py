from __future__ import annotations

import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from co_scientist.agents.base import AgentContext
from co_scientist.config import load_config
from co_scientist.llm.client import LLMRouter
from co_scientist.memory import Hypothesis, Review, ReviewType, SQLiteStore, Task, TaskPriority
from co_scientist.supervisor import (
    collect_session_stats,
    export_session_markdown,
    run_new_session,
    run_resume_session,
)
from co_scientist.supervisor.planner import create_research_plan

app = typer.Typer(help="Co-Scientist reproduction CLI.")
console = Console()
DEFAULT_DB_PATH = Path("runs") / "co_scientist.sqlite"


class ExportFormat(StrEnum):
    MD = "md"
    NIH_AIMS = "nih-aims"


def _load_config_or_exit(
    session_config: Path | None,
    initial_ideas: int | None,
    max_ideas: int | None,
    max_matches_per_idea: int | None,
    worker_concurrency: int | None,
):
    overrides = {
        "runtime.initial_ideas": initial_ideas,
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
    initial_ideas: Annotated[
        int | None,
        typer.Option("--initial-ideas", help="Override runtime.initial_ideas."),
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
        initial_ideas,
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
    initial_ideas: Annotated[
        int | None,
        typer.Option("--initial-ideas", help="Override runtime.initial_ideas."),
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
    """Start a Phase 7 research session in the foreground."""
    config = _load_config_or_exit(
        session_config,
        initial_ideas,
        max_ideas,
        max_matches_per_idea,
        worker_concurrency,
    )
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
    initial_ideas: Annotated[
        int | None,
        typer.Option("--initial-ideas", help="Override runtime.initial_ideas."),
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
    """Resume pending Phase 7 work for a research session."""
    config = _load_config_or_exit(
        session_config,
        initial_ideas,
        max_ideas,
        max_matches_per_idea,
        worker_concurrency,
    )
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
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Number of top hypotheses to show."),
    ] = 5,
) -> None:
    """Show Phase 7 session status."""
    try:
        stats, review_count = _run_async(_load_status(db_path, session_id, top_k=top_k))
    except Exception as exc:
        console.print(f"[red]Status failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Session: {session_id}")
    summary = Table(title="Session Summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Hypotheses", str(stats.hypothesis_count))
    summary.add_row("Reviews", str(review_count))
    summary.add_row("Matches", str(stats.match_count))
    summary.add_row("Matches per idea", f"{stats.matches_per_idea:.2f}")
    console.print(summary)
    if stats.top_hypotheses:
        top = Table(title="Top Hypotheses")
        top.add_column("Elo", justify="right")
        top.add_column("Summary")
        for hypothesis in stats.top_hypotheses:
            top.add_row(str(hypothesis.elo), hypothesis.summary)
        console.print(top)
        console.print("Top-k Elo buckets: " + _elo_histogram(stats.top_hypotheses))
    tasks = Table(title="Tasks")
    tasks.add_column("Status")
    tasks.add_column("Count", justify="right")
    for task_status, count in sorted(stats.tasks_by_status.items(), key=lambda item: item[0].value):
        tasks.add_row(task_status.value, str(count))
    console.print(tasks)


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
    export_format: Annotated[
        ExportFormat,
        typer.Option("--format", case_sensitive=False, help="Export format."),
    ] = ExportFormat.MD,
) -> None:
    """Export Phase 7 output as markdown."""
    config = _load_config_or_exit(session_config, None, None, None, None)
    try:
        if export_format == ExportFormat.MD:
            markdown = _run_async(
                export_session_markdown(db_path=db_path, config=config, session_id=session_id)
            )
        else:
            markdown = _run_async(_export_nih_aims(db_path, session_id))
    except Exception as exc:
        console.print(f"[red]Export failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if output is None:
        console.print(markdown)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    console.print(f"Wrote {output}")


@app.command()
def review(
    hypothesis_id: Annotated[int, typer.Argument(help="Hypothesis identifier.")],
    score: Annotated[
        float,
        typer.Option("--score", min=0.0, max=10.0, help="Manual review score from 0 to 10."),
    ],
    comment: Annotated[str, typer.Option("--comment", help="Manual review comment.")],
    db_path: Annotated[
        Path,
        typer.Option("--db-path", help="SQLite database path."),
    ] = DEFAULT_DB_PATH,
) -> None:
    """Add a manual expert review for a hypothesis."""
    try:
        session_id = _run_async(_add_manual_review(db_path, hypothesis_id, score, comment))
    except Exception as exc:
        console.print(f"[red]Review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Manual review added for hypothesis {hypothesis_id} in session {session_id}")


@app.command()
def contribute(
    session_id: Annotated[str, typer.Argument(help="Session identifier.")],
    file: Annotated[Path, typer.Option("--file", "-f", help="Markdown hypothesis file.")],
    summary: Annotated[
        str | None,
        typer.Option("--summary", help="Short summary for the contributed hypothesis."),
    ] = None,
    db_path: Annotated[
        Path,
        typer.Option("--db-path", help="SQLite database path."),
    ] = DEFAULT_DB_PATH,
) -> None:
    """Inject a user-provided hypothesis into the standard review pipeline."""
    content = file.read_text(encoding="utf-8").strip()
    if not content:
        console.print("[red]Hypothesis file is empty.[/red]")
        raise typer.Exit(code=2)
    try:
        hypothesis_id = _run_async(_contribute_hypothesis(db_path, session_id, content, summary))
    except Exception as exc:
        console.print(f"[red]Contribute failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Contributed hypothesis {hypothesis_id} and queued full review")


@app.command("revise-goal")
def revise_goal(
    session_id: Annotated[str, typer.Argument(help="Session identifier.")],
    goal_file: Annotated[Path, typer.Argument(help="Updated research goal file.")],
    session_config: Annotated[
        Path | None,
        typer.Option("--session-config", help="YAML config for this research session."),
    ] = None,
    db_path: Annotated[
        Path,
        typer.Option("--db-path", help="SQLite database path."),
    ] = DEFAULT_DB_PATH,
    force: Annotated[
        bool,
        typer.Option("--force", help="Skip confirmation for destructive tournament reset."),
    ] = False,
    initial_ideas: Annotated[
        int | None,
        typer.Option("--initial-ideas", help="Override runtime.initial_ideas."),
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
    """Apply supplemental goal constraints and re-review all hypotheses."""
    goal = goal_file.read_text(encoding="utf-8").strip()
    if not goal:
        console.print("[red]Goal file is empty.[/red]")
        raise typer.Exit(code=2)
    confirm_message = (
        "This will reset Elo, matches, proximity, feedback, and overview for this session. "
        "Continue?"
    )
    if not force and not typer.confirm(confirm_message):
        raise typer.Exit(code=1)
    config = _load_config_or_exit(
        session_config,
        initial_ideas,
        max_ideas,
        max_matches_per_idea,
        worker_concurrency,
    )
    try:
        hypothesis_count = _run_async(_revise_goal(db_path, config, session_id, goal))
    except Exception as exc:
        console.print(f"[red]Goal revision failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Updated goal for {session_id}; reset tournament and queued "
        f"{hypothesis_count} full review task(s)"
    )


@app.command()
def tail(
    session_id: Annotated[str, typer.Argument(help="Session identifier.")],
    db_path: Annotated[
        Path,
        typer.Option("--db-path", help="SQLite database path."),
    ] = DEFAULT_DB_PATH,
    limit: Annotated[int, typer.Option("--limit", help="Number of recent events to show.")] = 20,
    follow: Annotated[
        bool,
        typer.Option("--follow/--no-follow", help="Poll for new events."),
    ] = False,
    interval: Annotated[
        float,
        typer.Option("--interval", help="Polling interval in seconds for --follow."),
    ] = 2.0,
) -> None:
    """Show recent hypotheses and tournament matches."""
    seen: set[tuple[str, int]] = set()
    while True:
        try:
            events = _run_async(_load_tail_events(db_path, session_id, limit=limit))
        except Exception as exc:
            console.print(f"[red]Tail failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        for event in events:
            key = (event["kind"], int(event["id"]))
            if key in seen:
                continue
            seen.add(key)
            console.print(event["line"])
        if not follow:
            return
        time.sleep(interval)


async def _load_status(db_path: Path, session_id: str, *, top_k: int = 5):
    async with SQLiteStore(db_path) as store:
        if await store.get_session(session_id) is None:
            raise ValueError(f"unknown session: {session_id}")
        stats = await collect_session_stats(store, session_id, top_k=top_k)
        review_count = await store.count_reviews(session_id)
        return stats, review_count


async def _add_manual_review(
    db_path: Path,
    hypothesis_id: int,
    score: float,
    comment: str,
) -> str:
    async with SQLiteStore(db_path) as store:
        hypothesis = await store.get_hypothesis(hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"unknown hypothesis: {hypothesis_id}")
        await store.add_review(
            Review(
                session_id=hypothesis.session_id,
                hypothesis_id=hypothesis_id,
                type=ReviewType.MANUAL,
                score=score,
                content=comment,
            )
        )
        if not await store.has_active_task(
            hypothesis.session_id,
            agent="proximity",
            action="update_proximity_graph",
        ):
            await store.add_task(
                Task(
                    session_id=hypothesis.session_id,
                    agent="proximity",
                    action="update_proximity_graph",
                    priority=int(TaskPriority.PROXIMITY),
                )
            )
        return hypothesis.session_id


async def _contribute_hypothesis(
    db_path: Path,
    session_id: str,
    content: str,
    summary: str | None,
) -> int:
    async with SQLiteStore(db_path) as store:
        if await store.get_session(session_id) is None:
            raise ValueError(f"unknown session: {session_id}")
        hypothesis = await store.add_hypothesis(
            Hypothesis(
                session_id=session_id,
                content=content,
                summary=summary or _derive_summary(content),
                source_strategy="user_contributed",
            )
        )
        if hypothesis.id is None:
            raise RuntimeError("failed to store contributed hypothesis")
        await store.add_task(
            Task(
                session_id=session_id,
                agent="reflection",
                action="full_review",
                target_id=hypothesis.id,
                priority=int(TaskPriority.REFLECTION),
            )
        )
        return hypothesis.id


async def _revise_goal(db_path: Path, config, session_id: str, goal: str) -> int:
    async with SQLiteStore(db_path) as store:
        session = await store.get_session(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        if await store.has_running_tasks(session_id):
            raise ValueError(
                "cannot revise goal while tasks are running; stop the supervisor first"
            )
        await store.update_session_goal(session_id, goal)
        await create_research_plan(
            goal,
            AgentContext(
                store=store,
                llm_router=LLMRouter(config.llm),
                config=config,
                session_id=session_id,
            ),
        )
        await store.reset_tournament_for_goal_revision(session_id)
        hypotheses = await store.list_session_hypotheses(session_id)
        for hypothesis in hypotheses:
            if hypothesis.id is None:
                continue
            await store.add_task(
                Task(
                    session_id=session_id,
                    agent="reflection",
                    action="full_review",
                    target_id=hypothesis.id,
                    priority=int(TaskPriority.REFLECTION),
                )
            )
        await store.add_task(
            Task(
                session_id=session_id,
                agent="metareview",
                action="generate_system_feedback",
                priority=int(TaskPriority.META_REVIEW),
            )
        )
        return len(hypotheses)


async def _export_nih_aims(db_path: Path, session_id: str) -> str:
    async with SQLiteStore(db_path) as store:
        session = await store.get_session(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        overview = await store.latest_overview(session_id)
        top = await store.top_k_by_elo(session_id, k=3)
        lines = [
            f"# NIH Specific Aims: {session_id}",
            "",
            "## Overall Goal",
            session.goal,
            "",
            "## Central Hypothesis",
            overview.content if overview is not None else _top_hypothesis_text(top),
            "",
            "## Specific Aims",
        ]
        if top:
            for index, hypothesis in enumerate(top, start=1):
                lines.append(f"{index}. {hypothesis.summary}")
        else:
            lines.append("1. No ranked hypotheses are available yet.")
        lines.extend(
            [
                "",
                "## Initial Validation",
                (
                    "Use the top-ranked hypotheses and their latest reviews to prioritize "
                    "feasible experiments."
                ),
            ]
        )
        return "\n".join(lines).rstrip() + "\n"


async def _load_tail_events(
    db_path: Path,
    session_id: str,
    *,
    limit: int,
) -> list[dict[str, str | int]]:
    async with SQLiteStore(db_path) as store:
        if await store.get_session(session_id) is None:
            raise ValueError(f"unknown session: {session_id}")
        async with store.db.execute(
            """
            SELECT 'hypothesis' AS kind, id, created_at,
                   printf('hypothesis %d elo=%d %s', id, elo, summary) AS line
            FROM hypotheses
            WHERE session_id = ?
            UNION ALL
            SELECT 'match' AS kind, id, created_at,
                   printf('match %d winner=%d pair=%d/%d', id, winner_id, hypo_a_id, hypo_b_id)
                   AS line
            FROM matches
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (session_id, session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in reversed(rows)]


def _derive_summary(content: str) -> str:
    first_line = next(
        (line.strip("# ").strip() for line in content.splitlines() if line.strip()),
        "",
    )
    if not first_line:
        return "User contributed hypothesis"
    return first_line if len(first_line) <= 120 else first_line[:117].rstrip() + "..."


def _top_hypothesis_text(hypotheses: list[Hypothesis]) -> str:
    if not hypotheses:
        return "No final overview is available yet."
    return hypotheses[0].content


def _elo_histogram(hypotheses: list[Hypothesis]) -> str:
    buckets: dict[str, int] = {}
    for hypothesis in hypotheses:
        lower = (hypothesis.elo // 100) * 100
        label = f"{lower}-{lower + 99}"
        buckets[label] = buckets.get(label, 0) + 1
    return " ".join(f"{label}:{'#' * count}" for label, count in sorted(buckets.items()))


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)
