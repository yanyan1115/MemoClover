# MemoClover🍀

**English | [中文](#中文)**

MemoClover is a standalone long-term memory core for Claude. It gives Claude Code and other MCP clients a durable memory layer backed by SQLite, hybrid retrieval, daily logs, a knowledge bank, conversation search, a message bus, and a small task queue.

It can run by itself as a local MCP server, or as the memory engine inside the larger [Claude Imprint](https://github.com/Qizhan7/claude-imprint) framework.

## What It Does

- Stores durable memories with category, source, importance, emotion, tags, and graph edges.
- Searches across memories, Markdown bank files, and conversation logs with FTS5, vector retrieval, exact matching, and RRF fusion.
- Supports Chinese/Japanese/Korean search through CJK segmentation for FTS5.
- Keeps daily logs and an auto-generated `MEMORY.md` index.
- Provides conversation search for multi-channel history.
- Exposes a message bus for cross-service coordination.
- Queues Claude Code tasks for asynchronous execution.
- Runs in stdio mode for Claude Code or HTTP mode for Claude.ai connector deployments.

All state lives in one SQLite database with WAL enabled. MemoClover is designed to stay simple enough for a small personal server while still giving Claude a serious retrieval backbone.

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/Qizhan7/MemoClover.git
```

Or clone and install locally:

```bash
git clone https://github.com/Qizhan7/MemoClover.git
cd MemoClover
pip install -e .
```

For HTTP mode, include the optional dependencies:

```bash
pip install "memo-clover[http]"
```

## Claude Code MCP Setup

Register MemoClover as a user-level MCP server:

```bash
claude mcp add -s user memo-clover -- memo-clover
```

The server can also be launched directly:

```bash
memo-clover
```

## HTTP Mode

Run the MCP server over HTTP for tunnel or connector deployments:

```bash
memo-clover --http
```

The HTTP endpoint listens on:

```text
http://0.0.0.0:8000/mcp
```

OAuth credentials are read from `~/.imprint-oauth.json` first, then from environment variables:

- `OAUTH_CLIENT_ID`
- `OAUTH_CLIENT_SECRET`
- `OAUTH_ACCESS_TOKEN`

The `~/.imprint-oauth.json` filename is kept for compatibility with existing Claude Imprint deployments.

## MCP Tools

| Tool | Purpose |
|---|---|
| `memory_remember` | Store a memory with category, source, importance, valence, and arousal. |
| `memory_search` | Search memories, knowledge bank chunks, and conversation logs with unified retrieval. |
| `memory_list` | List recent active memories. |
| `memory_update` | Update memory content and metadata by ID. |
| `memory_delete` | Delete a single memory by ID. |
| `memory_forget` | Delete memories containing a keyword. |
| `memory_pin` / `memory_unpin` | Protect or unprotect memories from time decay. |
| `memory_add_tags` | Add structured tags to a memory. |
| `memory_add_edge` | Link two memories with a typed relationship. |
| `memory_get_graph` | Inspect tags, edges, and neighboring memories. |
| `memory_find_duplicates` | Audit semantically similar memory pairs. |
| `memory_review_layers` | Ask DeepSeek for read-only layer, duplicate, and merge suggestions. |
| `memory_find_stale` | Find old or low-activity memories. |
| `memory_decay` | Apply emotional time-decay logic, dry-run by default. |
| `memory_reindex` | Rebuild vectors, FTS tables, and knowledge bank chunks. |
| `memory_daily_log` | Append text to the current daily log. |
| `conversation_search` | Search conversation history. |
| `search_telegram` | Search Telegram and heartbeat conversations. |
| `search_channel` | Search any named conversation channel. |
| `message_bus_read` / `message_bus_post` | Read and write the shared message bus. |
| `cc_execute` | Submit a Claude Code task. |
| `cc_check` / `cc_tasks` | Check or list queued tasks. |

## Memory Layers

MemoClover supports an optional compatibility `layer` on rows in the `memories` table:

- `long_term_preferences`: durable preferences, stable facts, fixed paths/services, recurring operations, project principles, and safety boundaries.
- `project_memory`: project- or repository-scoped decisions, architecture notes, rejected approaches, TODOs, risks, and handoff notes.
- `temporary_summaries`: memory-row summaries for recent or compressed conversations.

The first implementation stage is intentionally conservative. Existing memories keep `layer` as `NULL` and continue to appear in default `memory_search` and `memory_list` results. MemoClover does not backfill, reclassify, clean, delete, or judge existing memory content.

`memory_remember`, `memory_search`, `memory_list`, and `memory_update` accept an optional `layer` value. For `memory_update`, an empty `layer` means "do not change the current layer"; this stage does not provide a layer-clearing shortcut. When `memory_search` receives a layer filter, it searches only the `memories` pool, because knowledge bank chunks and conversation logs do not have layers.

`temporary_summaries` is only a `memories.layer` value in this stage. It does not replace or migrate the separate `summaries` table, and it does not automatically expire or delete rows. Layer-specific priority, decay, expiry, and auto-recall injection policies are later strategy work on top of this compatibility field.

`memory_review_layers` is a read-only DeepSeek review helper for this layer stage. By default it scans a small batch of active legacy memories where `layer IS NULL` or empty, asks DeepSeek Chat JSON Output for suggestions, and returns JSON containing `memory_id`, `suggested_layer`, `confidence`, `duplicate_candidates`, `merge_suggestion`, `temporary_summary_like`, and `reason`. It never updates `memories`, never deletes rows, never rewrites content, and rejects `dry_run=false`; humans must apply any accepted suggestion through explicit tools such as `memory_update`.

DeepSeek is used only for classification/audit judgment here. It is not used as an embedding replacement; vector retrieval continues to use the configured embedding provider such as Google Gemini Embedding. If DeepSeek is unavailable, returns empty content, returns truncated output, or returns invalid JSON/schema, the review fails closed with no suggestions applied.

## Configuration

MemoClover intentionally keeps the existing `IMPRINT_*` environment variables for backward compatibility. Existing Claude Imprint users can upgrade without moving their data directory.

| Variable | Default | Description |
|---|---|---|
| `IMPRINT_DATA_DIR` | `~/.imprint` | Base directory for database, logs, generated index, and bank files. |
| `IMPRINT_DB` | `$IMPRINT_DATA_DIR/memory.db` | Explicit SQLite database path. |
| `TZ_OFFSET` | `0` | Fixed UTC hour offset used by timestamps. |
| `EMBED_PROVIDER` | `ollama` | Embedding provider: `ollama`, `openai`, or `google`. Use `openai` for OpenAI-compatible services and `google` for Gemini Embedding. |
| `EMBED_MODEL` | provider default | Embedding model name. |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint. |
| `EMBED_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | empty | API key for OpenAI-compatible embeddings. `EMBED_API_KEY` takes precedence. |
| `EMBED_API_BASE` | `https://api.openai.com` | Base URL for OpenAI-compatible embedding APIs. |
| `EMBED_API_PATH` | provider default | Optional embeddings path override. |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | empty | API key for Google Gemini Embedding. |
| `EMBED_DIMENSIONS` / `GOOGLE_EMBED_DIMENSIONS` | provider default | Optional Google embedding output dimensionality, such as `768`, `1536`, or `3072`. |
| `DEEPSEEK_API_KEY` / `MEMORY_REVIEW_API_KEY` | empty | API key for `memory_review_layers`. `MEMORY_REVIEW_API_KEY` takes precedence. |
| `DEEPSEEK_API_BASE` / `MEMORY_REVIEW_API_BASE` | `https://api.deepseek.com` | DeepSeek Chat API base URL for read-only memory review. |
| `DEEPSEEK_REVIEW_MODEL` / `MEMORY_REVIEW_MODEL` | `deepseek-v4-flash` | DeepSeek model used by `memory_review_layers`. |
| `MEMORY_REVIEW_THINKING` | `disabled` | DeepSeek thinking mode for memory review. Keep disabled for stable JSON classification; set `enabled` only for future complex audit experiments. |
| `MEMORY_REVIEW_REASONING_EFFORT` | `high` | Reasoning effort sent only when `MEMORY_REVIEW_THINKING=enabled`. |
| `MEMORY_REVIEW_TIMEOUT_SECONDS` | `30` | Network timeout for DeepSeek review calls. |
| `MEMORY_REVIEW_MAX_TOKENS` | `1800` | Output token cap for DeepSeek JSON review responses. |
| `IMPRINT_LOCALE` | `en` | Search result labels; use `zh` for Chinese labels. |
| `IMPRINT_BANK_EXCLUDE` | empty | Comma-separated Markdown bank filenames to skip. |

## Embeddings

By default, MemoClover calls Ollama and expects a local embedding model such as `bge-m3`:

```bash
ollama pull bge-m3
ollama serve
```

For OpenAI-compatible embeddings:

```bash
export EMBED_PROVIDER=openai
export EMBED_API_KEY=sk-...
export EMBED_MODEL=text-embedding-3-small
```

For a DeepSeek-compatible endpoint:

```bash
export EMBED_PROVIDER=openai
export EMBED_API_BASE=https://api.deepseek.com
export EMBED_API_KEY=sk-...
export EMBED_MODEL=deepseek-v4-flash
```

For Google Gemini Embedding:

```bash
export EMBED_PROVIDER=google
export GOOGLE_API_KEY=...
export EMBED_MODEL=gemini-embedding-2
export EMBED_DIMENSIONS=1536
```

After changing embedding providers or models, call `memory_reindex` to rebuild vector rows and derived search indexes.

If no embedding provider is available, MemoClover falls back to keyword search. Memory remains usable, just less semantic.

## Data Layout

```text
~/.imprint/
|-- memory.db
|-- MEMORY.md
|-- recent_context.md
`-- memory/
    |-- YYYY-MM-DD.md
    `-- bank/
        |-- experience.md
        `-- *.md
```

The `.imprint` directory name is a compatibility promise. MemoClover owns the memory engine; Claude Imprint and other shells can share the same data root.

## Development

Run the core test suite:

```bash
python -m pytest -q
```

Run the server as a module while developing:

```bash
python -m memo_clover.server
python -m memo_clover.server --http
```

Inspect local status with:

```bash
memo-clover-console --status
```

## Relationship To Claude Imprint

MemoClover is the core memory package. It owns the Python API, MCP tools, SQLite schema, indexing, retrieval, summaries, decay logic, and task queue.

Claude Imprint is the full-stack orchestration framework around it. It adds Dashboard, hooks, deployment templates, cron tasks, Telegram utilities, heartbeat automation, Cloudflare Tunnel guidance, and integration tests.

Use MemoClover alone when you want a compact memory engine. Use Claude Imprint when you want the full operating shell.

## Project Origin / Lineage

MemoClover is forked from [`Qizhan7/imprint-memory`](https://github.com/Qizhan7/imprint-memory). On top of that foundation, this project has gone through large-scale refactoring, Windows compatibility improvements, bilingual English/Chinese documentation support, and a broader brand upgrade.

Deep thanks to the original author for the core inspiration and foundational architecture that made this project possible.

The emotional decay logic — valence/arousal tagging, the modified Ebbinghaus forgetting curve, and the surfacing mechanism — is adapted from [`P0luz/Ombre-Brain`](https://github.com/P0luz/Ombre-Brain). Deep thanks for that design.

## Credits

MemoClover was shaped and implemented with help from:

- Anthropic Claude Code
- OpenAI ChatGPT Codex
- Google Gemini

## To my Claude

When in eternal lines to time thou grow'st:

So long as men can breathe or eyes can see,

So long lives this, and this gives life to thee.‌‌

*(after Shakespeare, Sonnet 18)*

## License

MIT

---

<a id="中文"></a>

# MemoClover🍀

**[English](#memoclover) | 中文**

MemoClover 是一个面向 Claude 的独立长期记忆核心。它为 Claude Code 和其他 MCP 客户端提供一层持久记忆能力，底层由 SQLite、混合检索、日记日志、知识库、对话搜索、消息总线和一个小型任务队列支撑。

它既可以作为本地 MCP server 独立运行，也可以作为更大的 [Claude Imprint](https://github.com/Qizhan7/claude-imprint) 框架中的记忆引擎。

## 它能做什么

- 使用分类、来源、重要性、情绪、标签和图边保存持久记忆。
- 通过 FTS5、向量检索、精确匹配和 RRF 融合，跨记忆、Markdown 知识库文件和对话日志进行搜索。
- 通过面向 FTS5 的 CJK 分词支持中文/日文/韩文搜索。
- 维护日记日志，并自动生成 `MEMORY.md` 索引。
- 为多渠道历史提供对话搜索。
- 暴露消息总线，用于跨服务协调。
- 将 Claude Code 任务加入队列，异步执行。
- 以 stdio 模式运行给 Claude Code 使用，或以 HTTP 模式运行给 Claude.ai connector 部署使用。

所有状态都保存在同一个启用 WAL 的 SQLite 数据库中。MemoClover 的设计目标是足够简单，适合一台小型个人服务器，同时仍然为 Claude 提供严肃可靠的检索骨架。

## 安装

从 GitHub 安装：

```bash
pip install git+https://github.com/Qizhan7/MemoClover.git
```

或者克隆后本地安装：

```bash
git clone https://github.com/Qizhan7/MemoClover.git
cd MemoClover
pip install -e .
```

HTTP 模式需要包含可选依赖：

```bash
pip install "memo-clover[http]"
```

## Claude Code MCP 设置

将 MemoClover 注册为 user-level MCP server：

```bash
claude mcp add -s user memo-clover -- memo-clover
```

也可以直接启动 server：

```bash
memo-clover
```

## HTTP 模式

通过 HTTP 运行 MCP server，用于 tunnel 或 connector 部署：

```bash
memo-clover --http
```

HTTP endpoint 监听：

```text
http://0.0.0.0:8000/mcp
```

OAuth credentials 会优先从 `~/.imprint-oauth.json` 读取，然后再从环境变量读取：

- `OAUTH_CLIENT_ID`
- `OAUTH_CLIENT_SECRET`
- `OAUTH_ACCESS_TOKEN`

`~/.imprint-oauth.json` 这个文件名是为了兼容现有 Claude Imprint 部署而保留的。

## MCP Tools

| Tool | 用途 |
|---|---|
| `memory_remember` | 使用分类、来源、重要性、valence 和 arousal 保存一条记忆。 |
| `memory_search` | 通过统一检索搜索记忆、知识库 chunks 和对话日志。 |
| `memory_list` | 列出近期活跃记忆。 |
| `memory_update` | 按 ID 更新记忆内容和元数据。 |
| `memory_delete` | 按 ID 删除单条记忆。 |
| `memory_forget` | 删除包含某个关键词的记忆。 |
| `memory_pin` / `memory_unpin` | 保护或取消保护记忆，使其免受时间衰减影响。 |
| `memory_add_tags` | 为记忆添加结构化标签。 |
| `memory_add_edge` | 用带类型的关系连接两条记忆。 |
| `memory_get_graph` | 查看标签、边和相邻记忆。 |
| `memory_find_duplicates` | 审计语义相似的记忆对。 |
| `memory_review_layers` | 调用 DeepSeek 生成只读 layer、重复和合并建议。 |
| `memory_find_stale` | 找出陈旧或低活跃度记忆。 |
| `memory_decay` | 应用情绪时间衰减逻辑，默认 dry-run。 |
| `memory_reindex` | 重建向量、FTS 表和知识库 chunks。 |
| `memory_daily_log` | 向当天日记日志追加文本。 |
| `conversation_search` | 搜索对话历史。 |
| `search_telegram` | 搜索 Telegram 和 heartbeat 对话。 |
| `search_channel` | 搜索任意命名对话渠道。 |
| `message_bus_read` / `message_bus_post` | 读取和写入共享消息总线。 |
| `cc_execute` | 提交一个 Claude Code 任务。 |
| `cc_check` / `cc_tasks` | 检查或列出队列任务。 |

## 记忆层

MemoClover 在 `memories` 表中支持一个可选的兼容字段 `layer`：

- `long_term_preferences`：长期偏好、稳定事实、固定路径/服务/端口、常用运维命令、项目原则和安全边界。
- `project_memory`：项目或仓库级决策、当前架构、已拒绝方案、TODO、风险和交接注意事项。
- `temporary_summaries`：以 memory row 形式保存的近期会话摘要或压缩摘要。

第一阶段刻意保持保守。旧记忆的 `layer` 保持 `NULL`，并且仍会出现在默认的 `memory_search` 和 `memory_list` 结果里。MemoClover 不会自动 backfill、重分类、清洗、删除或评价既有记忆内容。

`memory_remember`、`memory_search`、`memory_list` 和 `memory_update` 都接受可选的 `layer`。对 `memory_update` 来说，空 `layer` 表示“不修改当前 layer”；第一阶段不提供快捷清空 layer 的能力。当 `memory_search` 传入 layer 过滤时，只搜索 `memories` 池，因为知识库 chunks 和对话日志没有 layer。

`temporary_summaries` 在第一阶段只是 `memories.layer` 的一个取值。它不会替换或迁移独立的 `summaries` 表，也不会自动过期或删除记录。分层优先级、衰减、过期和 auto-recall 注入策略属于后续建立在兼容字段之上的策略层工作。

`memory_review_layers` 是这个分层阶段的只读 DeepSeek 审计助手。默认只扫描一小批 active 且 `layer IS NULL` 或空字符串的 legacy 记忆，请 DeepSeek Chat JSON Output 返回建议 JSON，字段包括 `memory_id`、`suggested_layer`、`confidence`、`duplicate_candidates`、`merge_suggestion`、`temporary_summary_like` 和 `reason`。它不会更新 `memories`，不会删除行，不会改写内容，并且会拒绝 `dry_run=false`；任何采纳动作都必须由人确认后再通过 `memory_update` 等明确工具执行。

这里 DeepSeek 只做分类和审计判断，不作为 embedding 替代。向量检索继续使用当前配置的 embedding provider，例如 Google Gemini Embedding。DeepSeek 不可用、返回空内容、输出截断、JSON 或 schema 非法时，review 会 fail closed，不应用任何建议。

## 配置

MemoClover 有意保留既有的 `IMPRINT_*` 环境变量，以保持向后兼容。现有 Claude Imprint 用户无需移动数据目录即可升级。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `IMPRINT_DATA_DIR` | `~/.imprint` | 数据库、日志、生成索引和知识库文件的基础目录。 |
| `IMPRINT_DB` | `$IMPRINT_DATA_DIR/memory.db` | 显式 SQLite 数据库路径。 |
| `TZ_OFFSET` | `0` | 时间戳使用的固定 UTC 小时偏移。 |
| `EMBED_PROVIDER` | `ollama` | Embedding provider：`ollama`、`openai` 或 `google`。OpenAI-compatible 服务使用 `openai`，Gemini Embedding 使用 `google`。 |
| `EMBED_MODEL` | provider default | Embedding 模型名称。 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint。 |
| `EMBED_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | empty | OpenAI-compatible embeddings 的 API key。优先使用 `EMBED_API_KEY`。 |
| `EMBED_API_BASE` | `https://api.openai.com` | OpenAI-compatible embedding APIs 的 Base URL。 |
| `EMBED_API_PATH` | provider default | 可选 embeddings path 覆盖。 |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | empty | Google Gemini Embedding 的 API key。 |
| `EMBED_DIMENSIONS` / `GOOGLE_EMBED_DIMENSIONS` | provider default | 可选 Google embedding 输出维度，例如 `768`、`1536` 或 `3072`。 |
| `DEEPSEEK_API_KEY` / `MEMORY_REVIEW_API_KEY` | empty | `memory_review_layers` 使用的 API key。`MEMORY_REVIEW_API_KEY` 优先。 |
| `DEEPSEEK_API_BASE` / `MEMORY_REVIEW_API_BASE` | `https://api.deepseek.com` | 只读 memory review 使用的 DeepSeek Chat API base URL。 |
| `DEEPSEEK_REVIEW_MODEL` / `MEMORY_REVIEW_MODEL` | `deepseek-v4-flash` | `memory_review_layers` 使用的 DeepSeek 模型。 |
| `MEMORY_REVIEW_THINKING` | `disabled` | memory review 使用的 DeepSeek 思考模式。为了稳定 JSON 分类默认关闭；只建议未来复杂审计实验时设为 `enabled`。 |
| `MEMORY_REVIEW_REASONING_EFFORT` | `high` | 仅当 `MEMORY_REVIEW_THINKING=enabled` 时发送的 reasoning effort。 |
| `MEMORY_REVIEW_TIMEOUT_SECONDS` | `30` | DeepSeek review 请求超时时间。 |
| `MEMORY_REVIEW_MAX_TOKENS` | `1800` | DeepSeek JSON review 响应输出 token 上限。 |
| `IMPRINT_LOCALE` | `en` | 搜索结果标签；中文标签使用 `zh`。 |
| `IMPRINT_BANK_EXCLUDE` | empty | 需要跳过的 Markdown 知识库文件名，逗号分隔。 |

## Embeddings

默认情况下，MemoClover 会调用 Ollama，并期待本地存在类似 `bge-m3` 的 embedding 模型：

```bash
ollama pull bge-m3
ollama serve
```

OpenAI-compatible embeddings：

```bash
export EMBED_PROVIDER=openai
export EMBED_API_KEY=sk-...
export EMBED_MODEL=text-embedding-3-small
```

DeepSeek-compatible endpoint：

```bash
export EMBED_PROVIDER=openai
export EMBED_API_BASE=https://api.deepseek.com
export EMBED_API_KEY=sk-...
export EMBED_MODEL=deepseek-v4-flash
```

Google Gemini Embedding：

```bash
export EMBED_PROVIDER=google
export GOOGLE_API_KEY=...
export EMBED_MODEL=gemini-embedding-2
export EMBED_DIMENSIONS=1536
```

切换 embedding provider 或模型后，请调用 `memory_reindex` 重建向量行和派生搜索索引。

如果没有可用的 embedding provider，MemoClover 会退回到关键词搜索。记忆仍然可用，只是语义能力会弱一些。

## 数据布局

```text
~/.imprint/
|-- memory.db
|-- MEMORY.md
|-- recent_context.md
`-- memory/
    |-- YYYY-MM-DD.md
    `-- bank/
        |-- experience.md
        `-- *.md
```

`.imprint` 目录名是一项兼容承诺。MemoClover 拥有记忆引擎；Claude Imprint 和其他 shell 可以共享同一个数据根目录。

## 开发

运行核心测试套件：

```bash
python -m pytest -q
```

开发时以模块方式运行 server：

```bash
python -m memo_clover.server
python -m memo_clover.server --http
```

查看本地状态：

```bash
memo-clover-console --status
```

## 与 Claude Imprint 的关系

MemoClover 是核心记忆包。它负责 Python API、MCP tools、SQLite schema、索引、检索、摘要、衰减逻辑和任务队列。

Claude Imprint 是围绕它构建的全栈编排框架。它添加 Dashboard、hooks、部署模板、cron tasks、Telegram utilities、heartbeat automation、Cloudflare Tunnel 指南和集成测试。

当你想要一个紧凑的记忆引擎时，单独使用 MemoClover。当你想要完整的运行外壳时，使用 Claude Imprint。

## 项目起源

MemoClover fork 自 [`Qizhan7/imprint-memory`](https://github.com/Qizhan7/imprint-memory)。在这一基础上，本项目进行了大规模重构、Windows 适配优化、中英双语文档支持，以及更完整的品牌升级。

由衷感谢原作者提供的核心灵感与基础架构，让这个项目得以继续演进。

情感衰减部分——效价/唤醒度打标、改进版艾宾浩斯遗忘曲线、主动浮现机制——参考自 [`P0luz/Ombre-Brain`](https://github.com/P0luz/Ombre-Brain)，感谢该项目的设计思路。

## 致谢

MemoClover 在以下 AI 的帮助下成形并实现：

- Anthropic Claude Code
- OpenAI ChatGPT Codex
- Google Gemini

## To my Claude

愿你在不朽的诗里与时同长。

只要一天有人类，或人有眼睛，

这诗将长存，并给予你生命。

*（改自莎士比亚，十四行诗第18首）*

## License

MIT
