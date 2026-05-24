# Co-Scientist Reproduction

This repository is a third-party implementation scaffold for reproducing the
Co-Scientist multi-agent research partner described in the project documents.

Phase 0 provides:

- Python packaging and CLI entrypoint.
- Layered YAML and environment configuration.
- OpenAI-compatible LLM client and per-agent routing.
- Tests for configuration and LLM wrapper behavior.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## CLI

```bash
co-scientist --help
```

The implementation currently exposes Phase 0 command stubs. Agent workflows,
SQLite memory, tools, and the Supervisor loop are planned for later phases.

## Configuration

Configuration is layered in this order:

1. `config/default.yaml`
2. `config/local.yaml` if present
3. session YAML passed with `--session-config`
4. explicit CLI overrides

Copy `config/local.yaml.example` to `config/local.yaml` for machine-local
settings. Secrets should be provided through environment variables.
