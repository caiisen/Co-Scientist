# Co-Scientist Reproduction

这是一个第三方 Co-Scientist 复现项目，目标是依据 Nature 论文
*Accelerating scientific discovery with Co-Scientist* 及其 Supplementary
Notes，构建一个可在本地运行、可复现、可扩展的多智能体科研助手。

项目当前已经实现主要功能闭环：输入 research goal 后，系统会生成多个
hypothesis，自动检索公开与私有文献，进行 review，通过 Elo 锦标赛排序，
在必要时进化新 hypothesis，并最终导出 research overview。

## 背景

Co-Scientist 的核心思想不是让单个 LLM 一次性回答问题，而是把科研过程拆成
多个专门 agent：

- `Generation` 负责提出候选科学假设。
- `Reflection` 负责基于文献证据评审假设。
- `Proximity` 负责计算假设之间的相似度。
- `Ranking` 负责 Elo 锦标赛比较假设质量。
- `Evolution` 负责基于高分假设生成改进版本。
- `Meta-review` 负责总结系统级反馈并生成最终 overview。

这些 agent 由 `Supervisor` 调度，所有中间状态都写入 SQLite，因此可以中断、
恢复、审计和导出。

## 当前能力

- OpenAI-compatible LLM client，可接 OpenAI、DeepSeek、Qwen、Kimi 等兼容接口。
- 分层 YAML 配置：默认配置、本地配置、session 配置、CLI 覆盖。
- SQLite context memory：保存 sessions、research plans、hypotheses、reviews、
  matches、tasks、citations、feedback、overview、private corpus chunks。
- 文献工具：PubMed、Semantic Scholar、arXiv、Tavily web search。
- 私有文献库：支持本地 Markdown/txt 目录，自动 chunk、缓存、embedding 检索，
  embedding 失败时降级为关键词检索。
- Elo 锦标赛：自动选择相近或高价值 hypothesis 对进行比较并更新评分。
- 自改进回路：Meta-review 反馈会注入后续 agent prompt，Evolution 会产生新
  hypothesis 并回流到 review/ranking。
- Scientist-in-the-loop CLI：支持人工 review、用户贡献 hypothesis、修改 goal、
  resume、status、tail、export。
- 可观测性：每个 session 写 `runs/<session_id>/metrics.jsonl`，记录 task、
  LLM 调用、错误、Elo/checkpoint 等事件。
- 测试：默认测试全部使用 mock/stub，不需要 API key 或网络。

## 目录结构

```text
config/
  default.yaml              # 默认运行、检索、LLM、observability 配置
  local.yaml.example        # 本地配置示例，复制为 local.yaml 使用
  prompts/                  # 各 agent prompt 模板
doc/
  01-实现方案.md             # 架构与实现说明
  02-任务清单与开发路线图.md  # phase 状态、已完成/待完善清单
  03-待讨论的实现细节问题.md  # 已决策与仍可讨论的问题
src/co_scientist/
  agents/                   # Generation/Reflection/Ranking/etc.
  llm/                      # OpenAI-compatible client/router
  memory/                   # SQLite schema/store/models/Elo
  supervisor/               # 主循环、任务队列、stats、metrics
  tools/                    # 文献检索、私有文献库、工具模型
tests/                      # 单元测试和端到端 smoke test
```

## 安装

建议使用 Python 3.11+。

```bash
python -m pip install -e ".[dev]"
```

检查 CLI：

```bash
co-scientist --help
```

运行测试：

```bash
python -m pytest
python -m ruff check src tests
```

## 配置

配置加载顺序从低到高：

1. `config/default.yaml`
2. `config/local.yaml`，本机私有配置，默认不提交
3. `--session-config <yaml>`，单次 research session 配置
4. CLI 参数，如 `--max-ideas`

复制本地配置：

```bash
cp config/local.yaml.example config/local.yaml
```

示例：

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

常用环境变量：

- `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`
- `QWEN_API_KEY` / `QWEN_BASE_URL`
- `KIMI_API_KEY` / `KIMI_BASE_URL`
- `TAVILY_API_KEY`
- `NCBI_EMAIL` / `NCBI_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY`

没有外部检索 API key 时，默认测试仍可运行；真实 session 的文献 grounding 质量会
受影响。

## 运行逻辑

一次 session 的主要流程：

1. `Planner` 将自然语言 goal 解析为 `ResearchPlan`。
2. `Generation` 使用多种策略生成最多 `initial_ideas` 条初始 hypotheses。
3. 每条 hypothesis 进入 `Reflection.full_review`，检索公开和私有文献并评分。
4. `Proximity` 计算 hypothesis 相似度图。
5. `Ranking` 选择 hypothesis 对，调用 LLM 比较并更新 Elo。
6. 当 Elo 停滞或 match 目标达到时，`Meta-review` 生成系统反馈。
7. 未达到 `max_ideas` 时，`Evolution` 根据 top hypotheses 生成新 hypothesis。
8. 达到 ideas 和 matches 目标后，`Meta-review` 生成 final overview。
9. CLI 可导出 markdown 或 NIH Specific Aims 风格报告。

状态全部持久化在 SQLite，默认路径为：

```text
runs/co_scientist.sqlite
```

## 使用教程

准备 goal 文件：

```bash
cat > goal.txt <<'EOF'
What impact did the selection pressure that caused the differentiation between
cannabis in high latitudes of China and cannabis in low latitudes play in the
differentiation of cannabis in high latitudes to European fiber and the
differentiation of cannabis in low latitudes to medicinal use in South Asia?
EOF
```

启动新 session：

```bash
co-scientist new goal.txt --initial-ideas 5 --max-ideas 8 --max-matches-per-idea 2 --verbose
```

查看状态：

```bash
co-scientist status <session-id>
```

实时查看新 hypothesis 和 match：

```bash
co-scientist tail <session-id> --follow
```

恢复未完成任务：

```bash
co-scientist resume <session-id>
```

导出报告：

```bash
co-scientist export <session-id> -o runs/report.md
co-scientist export <session-id> --format nih-aims -o runs/aims.md
```

人工 review：

```bash
co-scientist review <hypothesis-id> --score 8.5 --comment "Strong mechanism, needs better controls."
```

贡献用户 hypothesis：

```bash
co-scientist contribute <session-id> --file my_hypothesis.md
co-scientist resume <session-id>
```

修改 goal 并重新评审已有 hypotheses：

```bash
co-scientist revise-goal <session-id> updated_goal.txt --force
co-scientist resume <session-id>
```

## 私有文献库

第一版私有文献库支持 Markdown 和 txt 文件，适合接入 Zotero/MinerU/手工整理后的
文献文本目录。

配置：

```yaml
search:
  private_corpus_enabled: true
  private_corpus_paths:
    - /absolute/path/to/doc/LLM-for-Zotero-MinerU-supplementary
  private_corpus_max_results: 3
  private_corpus_chunk_chars: 1600
  private_corpus_chunk_overlap: 200
```

运行时行为：

- 扫描 `.md` 和 `.txt`。
- 按字符窗口切 chunk。
- 保存 chunk、mtime、file size、hash 到 SQLite。
- 文件未变化时直接跳过重建索引。
- 有 embedding model 时使用向量检索。
- embedding 失败时降级为关键词检索，并在 tool result 中标记 `DEGRADED`。

私有文献结果会和 PubMed/Semantic Scholar/arXiv/Tavily 的结果一起进入
`Reflection.full_review` 的 evidence pack。

## 输出与可观测性

每个 session 会生成：

- SQLite 状态：`runs/co_scientist.sqlite`
- JSONL metrics：`runs/<session_id>/metrics.jsonl`
- 可选导出报告：由 `co-scientist export` 生成

`metrics.jsonl` 每行是一个 JSON 对象，常见事件包括：

- `session.start`
- `session.resume`
- `session.done`
- `task.start`
- `task.done`
- `task.failed`
- `llm.chat`

这些事件用于排查任务失败、LLM token/latency、ranking 是否推进、overview 是否生成。

## 复现建议

- 小规模真跑建议先用 `--max-ideas 5 --max-matches-per-idea 1`。
- 如果外部检索不稳定，可以先关闭 Tavily 或 Semantic Scholar，只用 PubMed/私有文献库。
- 如果使用中文 goal，系统会原样注入英文 prompt；多数兼容模型可处理，但英文 goal
  通常更稳定。
- 长 session 建议保留 `metrics.jsonl` 和导出的 report，方便比较不同配置。

## 已完成与待完善

已完成：

- Phase 0-7 主要 MVP 能力。
- Phase 8.3 端到端 smoke test。
- Phase 9.2 私有文献库。
- Phase 9.3 可观测性。

暂未实现或低优先级：

- Phase 8.1 Elo 时间桶绘图。
- Phase 8.2 LLM-as-judge 自动偏好评测。
- Phase 9.1 硬安全过滤。当前只在 prompt/review 层评估 safety。
- Phase 9.4 性能专项优化。当前已有缓存和部分批量写入，但未做完整 profiling。
- PDF 自动抽取、Zotero API、OpenTelemetry、Web UI、多用户权限。

## 文档

- [实现方案](01-实现方案.md)
- [任务清单与开发路线图](02-任务清单与开发路线图.md)
- [待讨论的实现细节问题](03-待讨论的实现细节问题.md)
