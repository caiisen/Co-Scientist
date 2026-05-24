# Co-Scientist Reproduction

This repository is a third-party implementation scaffold for reproducing the
Co-Scientist multi-agent research partner described in the project documents.

Implemented foundations provide:

- Python packaging and CLI entrypoint.
- Layered YAML and environment configuration.
- OpenAI-compatible LLM client and per-agent routing.
- SQLite context memory, Elo updates, and persistent task queue primitives.
- Phase 2 literature tools for PubMed, Semantic Scholar, arXiv, and optional
  Tavily web search.
- Tests for configuration, LLM wrapper behavior, memory, queueing, and tools.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## CLI

```bash
co-scientist --help
```

The implementation currently exposes command stubs. Agent workflows and the
Supervisor loop are planned for later phases.

## Configuration

Configuration is layered in this order:

1. `config/default.yaml`
2. `config/local.yaml` if present
3. session YAML passed with `--session-config`
4. explicit CLI overrides

Copy `config/local.yaml.example` to `config/local.yaml` for machine-local
settings. Secrets should be provided through environment variables.

Tool-related environment variables:

- `TAVILY_API_KEY` enables Tavily web search. Tavily is optional; without a key
  the Tavily tool returns a degraded/failed source result while other literature
  sources can still run.
- `NCBI_EMAIL` and `NCBI_API_KEY` configure PubMed E-utilities. PubMed can run
  without an API key at lower rate limits, but NCBI recommends providing an
  email address.
- `SEMANTIC_SCHOLAR_API_KEY` enables higher Semantic Scholar Graph API limits.

## Literature Tools

The Phase 2 tool facade is:

```python
from co_scientist.tools import search_literature

result = await search_literature("AML drug repurposing", domain="biomed")
print(result.status)
print(result.format_evidence_pack())
```

The default test suite uses fixtures and mocks only; it does not require network
access or API keys. A manual live smoke can be run from an installed environment
with API keys configured by calling `search_literature()` for a small query such
as `"AML drug repurposing"`.
