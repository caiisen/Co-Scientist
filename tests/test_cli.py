from __future__ import annotations

import asyncio
from pathlib import Path

from typer.testing import CliRunner

from co_scientist.cli import app
from co_scientist.memory import (
    Hypothesis,
    Match,
    ResearchOverview,
    ResearchPlan,
    Review,
    SQLiteStore,
    SystemFeedback,
    Task,
    TaskPriority,
    TaskStatus,
)


def test_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Co-Scientist reproduction CLI" in result.output


def test_config_show_cli_override() -> None:
    result = CliRunner().invoke(
        app,
        ["config-show", "--initial-ideas", "4", "--max-ideas", "9"],
    )

    assert result.exit_code == 0
    assert "initial_ideas: 4" in result.output
    assert "max_ideas: 9" in result.output


def test_new_and_resume_expose_verbose_option() -> None:
    runner = CliRunner()

    new_help = runner.invoke(app, ["new", "--help"])
    resume_help = runner.invoke(app, ["resume", "--help"])

    assert new_help.exit_code == 0
    assert resume_help.exit_code == 0
    assert "--verbose" in new_help.output
    assert "--verbose" in resume_help.output
    assert "--initial-ideas" in new_help.output
    assert "--max-ideas" in new_help.output
    assert "--max-matches-per-idea" in resume_help.output


def test_review_adds_manual_review_and_queues_proximity(tmp_path: Path) -> None:
    db_path = tmp_path / "review.sqlite"
    hypothesis_id, session_id = asyncio.run(_seed_hypothesis(db_path))

    result = CliRunner().invoke(
        app,
        [
            "review",
            str(hypothesis_id),
            "--score",
            "8.5",
            "--comment",
            "Expert critique.",
            "--db-path",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    assert "Manual review added" in result.output
    review_type, task_action = asyncio.run(_manual_review_state(db_path, hypothesis_id, session_id))
    assert review_type == "manual"
    assert task_action == "update_proximity_graph"


def test_contribute_adds_hypothesis_and_queues_review(tmp_path: Path) -> None:
    db_path = tmp_path / "contribute.sqlite"
    session_id = asyncio.run(_seed_session(db_path))
    hypothesis_file = tmp_path / "hypothesis.md"
    hypothesis_file.write_text("# User idea\nDetailed mechanism.", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "contribute",
            session_id,
            "--file",
            str(hypothesis_file),
            "--db-path",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    assert "Contributed hypothesis" in result.output
    source_strategy, task_action = asyncio.run(_contribute_state(db_path, session_id))
    assert source_strategy == "user_contributed"
    assert task_action == "full_review"


def test_revise_goal_resets_tournament_replans_and_requeues_reviews(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "revise.sqlite"
    session_id = asyncio.run(_seed_ranked_session(db_path))
    goal_file = tmp_path / "goal.txt"
    goal_file.write_text("Updated goal with stricter assay constraints.", encoding="utf-8")

    async def fake_create_research_plan(goal, ctx):
        return await ctx.store.save_research_plan(
            ResearchPlan(
                session_id=ctx.session_id,
                goal=goal,
                preferences=["new preference"],
                attributes=["new attribute"],
                constraints=["new constraint"],
                idea_attributes=["new idea attribute"],
            )
        )

    monkeypatch.setattr("co_scientist.cli.create_research_plan", fake_create_research_plan)

    result = CliRunner().invoke(
        app,
        ["revise-goal", session_id, str(goal_file), "--force", "--db-path", str(db_path)],
    )

    assert result.exit_code == 0
    assert "reset tournament" in result.output
    state = asyncio.run(_revised_goal_state(db_path, session_id))
    assert state["goal"] == "Updated goal with stricter assay constraints."
    assert state["plan_goal"] == "Updated goal with stricter assay constraints."
    assert state["preferences"] == ["new preference"]
    assert state["elos"] == [1200, 1200]
    assert state["meta_review_rounds"] == [None, None]
    assert state["matches"] == 0
    assert state["feedback"] == 0
    assert state["overview"] == 0
    assert state["pending_full_reviews"] == 2
    assert state["pending_metareviews"] == 1
    assert state["cancelled_tasks"] == 1


def test_tail_no_follow_and_nih_export_show_session_outputs(tmp_path: Path) -> None:
    db_path = tmp_path / "display.sqlite"
    session_id = asyncio.run(_seed_ranked_session(db_path))
    runner = CliRunner()

    tail_result = runner.invoke(
        app,
        ["tail", session_id, "--no-follow", "--db-path", str(db_path)],
    )
    export_result = runner.invoke(
        app,
        ["export", session_id, "--format", "nih-aims", "--db-path", str(db_path)],
    )

    assert tail_result.exit_code == 0
    assert "hypothesis" in tail_result.output
    assert "match" in tail_result.output
    assert export_result.exit_code == 0
    assert "# NIH Specific Aims" in export_result.output
    assert "Final overview." in export_result.output


async def _seed_session(db_path: Path) -> str:
    async with SQLiteStore(db_path) as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        return session.id


async def _seed_hypothesis(db_path: Path) -> tuple[int, str]:
    async with SQLiteStore(db_path) as store:
        session = await store.create_session("goal")
        hypothesis = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="Content", summary="Summary")
        )
        assert hypothesis.id is not None
        return hypothesis.id, session.id


async def _seed_ranked_session(db_path: Path) -> str:
    async with SQLiteStore(db_path) as store:
        session = await store.create_session("goal")
        await store.save_research_plan(ResearchPlan(session_id=session.id, goal=session.goal))
        first = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="A content", summary="A", elo=1300)
        )
        second = await store.add_hypothesis(
            Hypothesis(session_id=session.id, content="B content", summary="B", elo=1100)
        )
        assert first.id is not None and second.id is not None
        await store.add_review(
            Review(session_id=session.id, hypothesis_id=first.id, type="full", content="Review")
        )
        await store.add_match_and_update_elo(
            Match(
                session_id=session.id,
                hypo_a_id=first.id,
                hypo_b_id=second.id,
                winner_id=first.id,
                transcript="A wins.",
            )
        )
        await store.add_feedback(SystemFeedback(session_id=session.id, round=1, content="Feedback"))
        await store.add_overview(
            ResearchOverview(
                session_id=session.id,
                round=1,
                content="Final overview.",
                top_hypothesis_ids=[first.id],
            )
        )
        await store.add_task(
            Task(
                session_id=session.id,
                agent="ranking",
                action="run_tournament_match",
                priority=int(TaskPriority.RANKING),
            )
        )
    return session.id


async def _manual_review_state(
    db_path: Path,
    hypothesis_id: int,
    session_id: str,
) -> tuple[str, str]:
    async with SQLiteStore(db_path) as store:
        reviews = await store.reviews_for_hypothesis(hypothesis_id)
        pending = await store.pending_tasks(session_id)
        return reviews[-1].type, pending[0].action


async def _contribute_state(db_path: Path, session_id: str) -> tuple[str | None, str]:
    async with SQLiteStore(db_path) as store:
        hypotheses = await store.list_session_hypotheses(session_id)
        pending = await store.pending_tasks(session_id)
        return hypotheses[-1].source_strategy, pending[0].action


async def _revised_goal_state(db_path: Path, session_id: str) -> dict:
    async with SQLiteStore(db_path) as store:
        session = await store.get_session(session_id)
        plan = await store.get_research_plan(session_id)
        hypotheses = await store.list_session_hypotheses(session_id)
        tasks = await store.tasks_by_status(session_id)
        assert session is not None
        assert plan is not None
        return {
            "goal": session.goal,
            "plan_goal": plan.goal,
            "preferences": plan.preferences,
            "elos": [hypothesis.elo for hypothesis in hypotheses],
            "meta_review_rounds": [hypothesis.meta_review_round for hypothesis in hypotheses],
            "matches": await store.count_matches(session_id),
            "feedback": await store.count_feedback(session_id),
            "overview": await store._count("overview", session_id),
            "pending_full_reviews": len(
                [
                    task
                    for task in await store.pending_tasks(session_id)
                    if task.agent == "reflection" and task.action == "full_review"
                ]
            ),
            "pending_metareviews": len(
                [
                    task
                    for task in await store.pending_tasks(session_id)
                    if task.agent == "metareview"
                    and task.action == "generate_system_feedback"
                ]
            ),
            "cancelled_tasks": tasks[TaskStatus.CANCELLED],
        }
