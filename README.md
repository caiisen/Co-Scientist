# Co-Scientist Reproduction

[中文 README](doc/README.zh.md)

This repository is a third-party reproduction of the Co-Scientist multi-agent
research partner described in *Accelerating scientific discovery with
Co-Scientist* and its Supplementary Notes.

The project now implements the main functional loop: given a research goal, it
generates candidate hypotheses, grounds them with public and private literature,
reviews them, ranks them through an Elo tournament, evolves stronger hypotheses
when useful, and exports a final research overview.

## Background

Co-Scientist is not a single-prompt question answering system. It decomposes
scientific reasoning into specialized agents:

- `Generation` proposes candidate scientific hypotheses.
- `Reflection` reviews hypotheses against literature evidence.
- `Proximity` tracks similarity between hypotheses.
- `Ranking` compares hypotheses through an Elo tournament.
- `Evolution` improves top hypotheses without replacing the originals.
- `Meta-review` summarizes system-level feedback and produces the final overview.

The `Supervisor` schedules these agents through a persistent task queue. All
intermediate state is stored in SQLite, so runs can be resumed, inspected, and
exported.

## Features

- OpenAI-compatible LLM client for OpenAI, DeepSeek, Qwen, Kimi, and similar
  providers.
- Layered YAML configuration: default config, local config, session config, and
  CLI overrides.
- SQLite context memory for sessions, research plans, hypotheses, reviews,
  matches, tasks, citations, feedback, overview, and private-corpus chunks.
- Literature tools for PubMed, Semantic Scholar, arXiv, and Tavily web search.
- Private literature corpus support for local Markdown/txt directories, with
  chunking, caching, embedding retrieval, and lexical fallback.
- Elo tournament ranking with automatic pair selection.
- Self-improvement loop: meta-review feedback is injected into later agent
  prompts, and evolution-generated hypotheses re-enter review/ranking.
- Scientist-in-the-loop CLI: manual reviews, user-contributed hypotheses, goal
  revision, resume, status, tail, and export.
- Observability: each session writes `runs/<session_id>/metrics.jsonl` with
  task, LLM, error, and Elo/checkpoint events.
- Offline test suite using mocks/stubs only; no API keys or network are needed
  for tests.

## Repository Layout

```text
config/
  default.yaml              # Runtime, search, LLM, and observability defaults
  local.yaml.example        # Machine-local override example
  prompts/                  # Agent prompt templates
doc/
  README.zh.md              # Chinese README
src/co_scientist/
  agents/                   # Generation, Reflection, Ranking, etc.
  llm/                      # OpenAI-compatible client/router
  memory/                   # SQLite schema/store/models/Elo
  supervisor/               # Main loop, task queue, stats, metrics
  tools/                    # Literature search, private corpus, tool models
tests/                      # Unit tests and end-to-end smoke test
```

## Installation

Use Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

Check the CLI:

```bash
co-scientist --help
```

Run tests:

```bash
python -m pytest
python -m ruff check src tests
```

## Configuration

Configuration is loaded in this order, from lowest to highest precedence:

1. `config/default.yaml`
2. `config/local.yaml`, for machine-local settings and ignored by git
3. `--session-config <yaml>`, for one research session
4. CLI overrides such as `--max-ideas`

Create a local config:

```bash
cp config/local.yaml.example config/local.yaml
```

Example:

```yaml
llm:
  default_provider: deepseek
  providers:
    deepseek:
      api_key_env: DEEPSEEK_API_KEY
      base_url: https://api.deepseek.com
      chat_model: deepseek-chat
      temperature: 0.3

search:
  tavily_enabled: true
  pubmed_enabled: true
  semantic_scholar_enabled: true
  arxiv_enabled: true
  private_corpus_enabled: true
  private_corpus_paths:
    - /absolute/path/to/my/literature-markdown
```

Common environment variables:

- `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`
- `QWEN_API_KEY` / `QWEN_BASE_URL`
- `KIMI_API_KEY` / `KIMI_BASE_URL`
- `TAVILY_API_KEY`
- `NCBI_EMAIL` / `NCBI_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY`

The default test suite does not require external API keys. Real runs will have
better grounding when literature-search credentials are configured.

## Runtime Flow

A session follows this high-level flow:

1. `Planner` parses the natural-language goal into a `ResearchPlan`.
2. `Generation` creates up to `initial_ideas` initial hypotheses with multiple strategies.
3. Each hypothesis enters `Reflection.full_review`, which searches public and
   private literature and assigns a review score.
4. `Proximity` computes hypothesis similarity.
5. `Ranking` selects hypothesis pairs, asks the LLM to compare them, and updates
   Elo ratings.
6. When Elo stagnates or the match target is reached, `Meta-review` creates
   system feedback.
7. If `max_ideas` has not been reached, `Evolution` creates new hypotheses from
   top-ranked candidates.
8. Once idea and match targets are reached, `Meta-review` generates the final
   research overview.
9. The CLI exports the report as Markdown or NIH Specific Aims style text.

All state is persisted in SQLite. The default database path is:

```text
runs/co_scientist.sqlite
```

## Usage

Create a goal file:

```bash
cat > goal.txt <<'EOF'
What impact did the selection pressure that caused the differentiation between
cannabis in high latitudes of China and cannabis in low latitudes play in the
differentiation of cannabis in high latitudes to European fiber and the
differentiation of cannabis in low latitudes to medicinal use in South Asia?
EOF
```

Start a new session:

```bash
co-scientist new goal.txt --initial-ideas 5 --max-ideas 8 --max-matches-per-idea 2 --verbose
```

Show status:

```bash
co-scientist status <session-id>
```

Tail new hypotheses and matches:

```bash
co-scientist tail <session-id> --follow
```

Resume pending work:

```bash
co-scientist resume <session-id>
```

Export reports:

```bash
co-scientist export <session-id> -o runs/report.md
co-scientist export <session-id> --format nih-aims -o runs/aims.md
```

Add a manual expert review:

```bash
co-scientist review <hypothesis-id> --score 8.5 --comment "Strong mechanism, needs better controls."
```

Contribute a user hypothesis:

```bash
co-scientist contribute <session-id> --file my_hypothesis.md
co-scientist resume <session-id>
```

Revise the goal and re-review existing hypotheses:

```bash
co-scientist revise-goal <session-id> updated_goal.txt --force
co-scientist resume <session-id>
```

## Private Literature Corpus

The first private-corpus implementation supports Markdown and txt files. It is
intended for literature exported from Zotero, MinerU, or a manually curated text
directory.

Configuration:

```yaml
search:
  private_corpus_enabled: true
  private_corpus_paths:
    - /absolute/path/to/doc/LLM-for-Zotero-MinerU-supplementary
  private_corpus_max_results: 3
  private_corpus_chunk_chars: 1600
  private_corpus_chunk_overlap: 200
```

Runtime behavior:

- Scans `.md` and `.txt` files.
- Splits files into fixed-size overlapping chunks.
- Stores chunk text, mtime, file size, and hash in SQLite.
- Skips re-indexing unchanged files.
- Uses embedding retrieval when an embedding model is available.
- Falls back to lexical search when embedding fails, returning a `DEGRADED`
  tool status.

Private corpus results are merged into the same evidence pack as PubMed,
Semantic Scholar, arXiv, and Tavily results for `Reflection.full_review`.

## Output and Observability

Each session may produce:

- SQLite state: `runs/co_scientist.sqlite`
- JSONL metrics: `runs/<session_id>/metrics.jsonl`
- Optional exported reports from `co-scientist export`

Common `metrics.jsonl` events:

- `session.start`
- `session.resume`
- `session.done`
- `task.start`
- `task.done`
- `task.failed`
- `llm.chat`

These events help debug failed tasks, LLM token/latency behavior, ranking
progress, and final overview generation.

## Reproduction Tips

- For the first real run, start small: `--max-ideas 5 --max-matches-per-idea 1`.
- If public search is unstable, disable Tavily or Semantic Scholar and use
  PubMed/private corpus first.
- Chinese goals are passed through as-is into English prompts; most compatible
  models handle this, but English goals tend to be more stable.
- Keep `metrics.jsonl` and exported reports for long sessions so different
  configurations can be compared later.

## Completed and Deferred Work

Completed:

- Main MVP functionality for Phases 0-7.
- Phase 8.3 end-to-end smoke test.
- Phase 9.2 private literature corpus.
- Phase 9.3 observability.

Deferred or lower priority:

- Phase 8.1 Elo time-bucket plotting.
- Phase 8.2 LLM-as-judge preference evaluation.
- Phase 9.1 hard safety filtering. Safety is currently evaluated in prompts and
  reviews, but not used as a hard gate.
- Phase 9.4 dedicated performance optimization. The project already has caching
  and some batching, but no full profiling pass yet.
- PDF extraction, Zotero API integration, OpenTelemetry, Web UI, and multi-user
  access control.

## Documentation

- [Chinese README](doc/README.zh.md)
- [Implementation plan](doc/01-实现方案.md)
- [Task list and roadmap](doc/02-任务清单与开发路线图.md)
- [Design decisions and open questions](doc/03-待讨论的实现细节问题.md)
