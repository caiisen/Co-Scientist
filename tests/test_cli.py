from __future__ import annotations

from typer.testing import CliRunner

from co_scientist.cli import app


def test_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Co-Scientist reproduction CLI" in result.output


def test_config_show_cli_override() -> None:
    result = CliRunner().invoke(app, ["config-show", "--max-ideas", "9"])

    assert result.exit_code == 0
    assert "max_ideas: 9" in result.output


def test_new_and_resume_expose_verbose_option() -> None:
    runner = CliRunner()

    new_help = runner.invoke(app, ["new", "--help"])
    resume_help = runner.invoke(app, ["resume", "--help"])

    assert new_help.exit_code == 0
    assert resume_help.exit_code == 0
    assert "--verbose" in new_help.output
    assert "--verbose" in resume_help.output
