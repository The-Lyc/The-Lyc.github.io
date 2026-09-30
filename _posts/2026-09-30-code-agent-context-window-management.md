---
layout: "post"
title: "代码 Agent 的上下文窗口管理机制调研"
date: "2026-09-30"
description: "按截断、工具输出遮蔽、摘要、检索与持久记忆分类，比较代码 Agent 的上下文管理方案。"
categories: ["MLSys"]
tags: ["agents", "context-management", "memory", "compaction"]
permalink: "/blog/code-agent-context-window-management/"
lang: "zh-CN"
notes_import: true
source_path: "MLSys/agent/ContextCompact.md"
toc:
  beginning: true
---

# Code Agent Context Window 管理机制调研报告

> 调研时间：2026-04-23 | 覆盖产品：Claude Code, GitHub Copilot CLI, Cursor, Windsurf/Cascade, Devin

---

## 一、统一技术 Taxonomy

在逐产品分析之前，先抽象出 context management 的六大技术范式。后续各产品的机制均可映射到此分类体系中。

| 范式                                    | 定义                                                                   | 典型实现                                                            |
| --------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Truncation（截断）**                  | 按时间/优先级直接丢弃 oldest 消息或 tool output，不做语义处理          | FIFO 删除、priority-based drop                                      |
| **Observation Masking（工具输出遮蔽）** | 将可重算的 tool result 替换为占位符 `[cleared]`，保留 tool call 结构   | Claude Code microcompact、Copilot largeToolResultsToDisk            |
| **Summarization（LLM 摘要）**           | 使用 LLM 将对话历史压缩为结构化或自由文本摘要                          | 全量 compact、partial compact、incremental summary                  |
| **Retrieval / RAG（检索增强）**         | 将历史存储在外部（文件 / 向量库），按需检索注入 context                | Cursor history-as-files、Windsurf codebase index                    |
| **State Memory（持久状态记忆）**        | 在 context 外维护持久化 key-value / document store，跨 session 生存    | Windsurf Memories、Devin Knowledge Base、Claude Code session memory |
| **Context Scheduling（上下文调度）**    | 通过 subagent 隔离、prompt 重建、flex budget 分配等方式管理 token 预算 | Claude Code subagent isolation、Copilot flex-grow renderer          |

---

## 二、逐产品对比分析

### 1. Claude Code

**架构概述：** 多层级瀑布式 compaction 管道，从轻量到重量逐级升级。

**Compaction 层级（五层 Cascade）：**

- **Layer 0 — 源头控制：** Tool result 超过 50K 字符时，完整内容写入磁盘文件，context 中仅保留 ~2KB 预览 + 文件指针。
- **Layer 1 — Snip（截断）：** 直接丢弃整条旧消息，不做语义处理，最激进但最廉价。
- **Layer 2 — Microcompact（工具输出遮蔽）：** 将旧 tool result 内容替换为 `[Old tool result content cleared]`，保留 tool_use/tool_result 结构完整性。可被 compact 的工具：FileRead、Shell、Grep、Glob、WebSearch、WebFetch、FileEdit、FileWrite。存在客户端和服务端（API microcompact）两个实现路径。
- **Layer 3 — Session Memory Compact：** 实验性功能（feature flag 控制），利用持续更新的 session memory document 做增量式压缩，类似"增量备份"。
- **Layer 4 — Full Autocompact（LLM 摘要）：** 调用 LLM 对全量对话做 summarization，成本最高，是最后手段。

**触发阈值：**

- Autocompact 阈值 = `context_window - max_output_tokens - 13K buffer`。对于 200K 窗口约 187K（~93.5%）。
- 社区报告显示近期版本已将触发点前移至约 64-75% usage，预留更多 completion buffer。
- 1M context window 已上线（Opus 4.7），大幅缓解频繁 compact 问题。
- Circuit breaker：连续 3 次 compact 失败后停止重试（历史数据：1,279 个 session 曾出现 50+ 次连续失败，日均浪费约 25 万次 API 调用）。

**用户可配置能力：**

- `/compact <instructions>`：手动触发 + 自定义 summary focus，如 `/compact Focus on the API changes`。
- Partial compact：通过 `/rewind` 选择 pivot point，可选择性 summarize 特定区段。
- `CLAUDE.md` 中可写入 compact 保留指令，如 "When compacting, always preserve the full list of modified files"。
- Server-side Compaction API（beta `compact-2026-01-12`）：支持 `context_token_threshold`、`summary_prompt`、`pause_after_compaction` 等参数。
- Subagent isolation：通过 `isolation: worktree` 将 verbose task 委派到独立 context window，仅返回 summary。
- Hooks：PreToolUse / PostToolUse 等 hook 可辅助 context 管理。

**已知问题与用户反馈：**

- Summary drift：多次连续 compact 后质量累积退化，"你在 summarize 一个已经退化的视图"。
- 早期指令丢失：架构决策、约束条件在 compact 后最容易被丢弃。
- Autocompact buffer 开销：有报告显示 autocompact buffer 本身消耗 45K tokens（22.5% context）。
- Compaction 期间中断：旧版在 90%+ 才触发，常在 mid-operation 时被迫 compact。

**证据来源：** Anthropic 官方文档 (platform.claude.com/docs/en/build-with-claude/compaction)、Claude Code 官方 best practices (code.claude.com/docs/en/best-practices)、Anthropic 官方博客 (claude.com/blog/using-claude-code-session-management-and-1m-context)、Finisky Garden 源码分析、Steve Kinney 课程文档、GitHub Gist compaction 研究。

---

### 2. GitHub Copilot CLI

**架构概述：** 每次 iteration 从零重建 prompt（per-turn prompt reconstruction），而非累积式修改。

**核心机制：**

- **Prompt Rebuild 架构：** 与 Claude Code 的"事后变异消息"不同，Copilot 每轮重新 render 完整 prompt，在 render 时决定什么能装入。
  - Layer 1：每个 ToolResult 在 rendering 时自行截断。
  - Layer 2：Flex properties 按权重分配空间（类 CSS flexbox 模型），新 turn 获得更高权重 → 近期上下文保真度更高。
  - Layer 3：若仍超预算，裁剪最低优先级内容。
- **Large Tool Results to Disk：** 超过 8KB 的 tool output 替换为指针（search subagent、execution subagent、memory tool 除外）。
- **Background Compaction：** 80% 容量时后台开始 LLM summarization，预留 ~20% buffer 供 tool call 继续运行。
- **95% Pause：** 若 compact 未完成且 context 达 95%，暂停等待 compact 完成。
- **Checkpoint 机制：** 每次 compact 前保存快照，用户可回溯查看 compact 前后差异。

**触发阈值：**

- 自动 compact：~80% context capacity 触发后台压缩。
- 硬暂停：~95% 触发阻塞等待。
- ≤20% remaining 时显示 warning（v0.0.334+，颜色从红色改为黄色）。
- Output reserved：预留 ~30% token 给模型输出（用户反馈认为过多）。

**用户可配置能力：**

- `/compact`：手动触发 compact。
- `/clear`：完全重置 context + token 计数。
- `/context`：可视化 token 使用分布（System/Tools、Messages、Free Space、Buffer）。
- `/model`：切换模型（影响 context window 大小）。
- `.github/copilot-instructions.md`：持久化指令（中优先级）。

**已知问题：**

- 早期版本 truncation 不透明：仅显示 "Truncated" 标志，无法知道删了什么（Issue #385）。
- Output reserved 过大：30-60% 的 reserved output space 被用户批评为浪费。
- Compact 后不可逆：无法恢复被 summarize 的内容。

**证据来源：** GitHub 官方文档 (docs.github.com/en/copilot/concepts/agents/copilot-cli/context-management)、GitHub 社区讨论 (#188691)、Copilot CLI Issues (#385)、DeepWiki 源码分析、Medium 对比分析文章。

---

### 3. Cursor

**架构概述：** 自动 summarization + 历史文件检索（history-as-files）的混合方案。

**核心机制：**

- **Auto Summarization：** 当 context window 达到模型上限（如 Sonnet 4 的 200K）时自动触发，使用**更小的 flash 模型**（非当前选择的模型）生成摘要。
- **History-as-Files（动态上下文发现）：** 这是 Cursor 的独特创新——compact 后，完整对话历史被保存为文件。Agent 持有该文件的引用，若发现 summary 中缺失细节，可通过 grep 检索完整历史。
- **文件 Condensation：** 对 @mention 的文件，根据大小和可用空间自动选择：完整展示 / outline 提取 / 摘要。
- **Dynamic Context Discovery：** 不再静态注入所有 context，而是让 agent 通过工具主动拉取所需信息。MCP tool descriptions 也同步为本地文件供按需查找，减少 46.9% 的 agent tokens。
- **Terminal History as Files：** 终端输出同步到文件系统，agent 可 grep 检索而非全量注入。
- **Codebase Index：** 基于 embedding 的 RAG 检索，对 @codebase 查询返回语义相关代码片段。

**触发阈值：**

- 达到模型 context window 100% 时自动 summarize。
- 用户可通过 `/compress` 命令手动触发（Lee Robinson 实现，已合入后端）。

**用户可配置能力：**

- `/compress`：手动触发压缩。
- `.cursor/rules`：项目级规则（always / auto-attach / agent-requested / manual）。
- User Rules：全局个人偏好设置。
- @Past Chats / @Recent Changes：引用历史 chat 保持连续性。
- 用户社区强烈要求（forum.cursor.com）增加手动 compact 按钮。

**已知问题：**

- Summary 丢失关键细节：用户反馈 compact 后 agent "leaking context"，"忘记了复杂 debugging 的全部过程"。
- 多次 compact 累积退化：反复压缩导致 summary 遗漏越来越多细节。
- 文件读取截断：Agent mode 默认只读文件前 250 行，@codebase 模式用 smaller model 做 file summarization，覆盖不完整。
- 不透明性：用户无法看到 summarization 过程和结果。

**证据来源：** Cursor 官方文档 (docs.cursor.com/en/agent/chat/summarization)、Cursor 官方博客 (cursor.com/blog/dynamic-context-discovery)、Cursor 社区论坛 (forum.cursor.com/t/compact-compress-chat/132097)、Lee Robinson (Cursor 工程师) 公开分享。

---

### 4. Windsurf / Cascade

**架构概述：** 以持久化 Memory + Rules 为核心的 "flow-aware" 架构，每轮重建而非累积。

**核心机制：**

- **Per-turn Context Assembly Pipeline：** 每次交互，Cascade 从五个源头重建 context：
  1. Rules（global → project）
  2. Memories（持久化事实）
  3. Open files（当前文件优先）
  4. Indexed retrieval（M-Query 语义检索）
  5. Recent actions（编辑、终端命令、导航历史）
- **Memories 系统：** AI 自动生成 + 用户手动创建。workspace 级作用域，存储于 `~/.codeium/windsurf/memories/`。不消耗 credits，不上传到 repo。
- **Rules 系统：** 手动定义的策略检查，支持 global / workspace / system 三级，四种激活模式（manual / always on / model decision / glob pattern）。
- **Conversation Summarization：** 当 context window 过长时，Cascade 偶尔 summarize 消息并清除历史。
- **Fast Context（SWE-grep）：** 20x 更快的 context 检索，>2,800 tokens/sec。
- **Context Usage Indicator：** 可视化 context 使用量。

**关键特点：**

- Windsurf 的设计哲学与 session-based agents 根本不同——不累积 conversation history，而是每轮从 codebase graph + persistent state 重建 context。
- 这意味着没有传统意义的"compaction"问题，但也意味着 agent 没有细粒度的对话记忆。

**用户可配置能力：**

- Memories 管理面板：查看、删除自动生成的记忆。
- `.windsurf/rules/` 目录 + `AGENTS.md`：版本控制的规则系统。
- 系统级 Rules（企业 IT/安全团队管理）。
- Hooks：在 model response 等关键点执行自定义命令。

**已知问题：**

- 过期 Memories 误导 AI：需定期清理。
- 无细粒度 compact 控制：用户无法指定 summarize 什么、保留什么。
- 早期 context 被静默丢弃：changelog 提到 "earlier context can be dropped without warning"。
- Flow awareness 有学习成本。

**证据来源：** Windsurf 官方文档 (docs.windsurf.com/windsurf/cascade/memories)、Windsurf Changelog (windsurf.com/changelog)、MarkAICode 深度分析、DataLakeHouseHub 指南。

---

### 5. Devin (Cognition)

**架构概述：** 多模型 compound system（Planner + Coder + Critic），强调 long-horizon task 的持久记忆。

**核心机制：**

- **Custom Compaction & Summarization System：** Cognition 官方博客明确表示不依赖模型自身的 summarization，而是使用自建的 compacting 系统。模型自己的 notes "不够全面"，"会改写任务描述丢失重要细节"。
- **Context Anxiety 现象与应对：** Cognition 发现 Sonnet 4.5 会在接近 context limit 时产生 "context anxiety"——走捷径、提前结束未完成的任务。解决方案：给更大的 context window 但编程式 cap 在较低阈值，让模型不感知边界。
- **Fine-tuned Summarization at Handoff：** 在 sub-agent 向 orchestrator 报告时，使用专门的 small model 压缩 handoff 内容（而非 main model）。
- **Knowledge Base：** 组织级 / 企业级知识库，包括 repo notes 和自动生成的 repo index（用于 RAG）。支持 AI-generated suggestions。
- **Persistent Working Memory：** Session 内跨步骤维护工作记忆，可回溯早期观察、错误和决策。
- **Repository Context Ingestion：** Enterprise 级支持 10M+ token context window，全 repo reasoning。
- **Dynamic Re-planning：** Planner 持续根据新发现重新规划，而非执行固定计划。

**触发阈值：**

- 无公开的具体数值。Session 约 2.5 小时 / 10 ACU 时出现性能退化警告。
- 通过编程式 cap 管理（大窗口 + 低阈值），避免模型感知边界。

**用户可配置能力：**

- Knowledge Base：可手动添加 repo notes、coding standards、architectural guidelines。
- Enterprise Knowledge：跨组织共享。
- Session 级 task budget limit（如 $5.00/ticket）。
- Playbooks：可定义自动化工作流模板。

**已知问题：**

- 长 session 性能退化：>2.5h 后明显。
- 模型在 short context 时反而生成过多 summary tokens（"花在写总结上的 token 比解决问题的还多"）。
- Context anxiety 导致 premature completion。
- 并行 tool call 在空 context 时激进，接近 limit 时保守——与 context anxiety 相互作用。

**证据来源：** Cognition 官方博客 (cognition.ai/blog/devin-sonnet-4-5-lessons-and-challenges)、Cognition 官方博客 (cognition.ai/blog/how-cognition-uses-devin-to-build-devin)、Devin 官方文档 (docs.devin.ai/product-guides/knowledge)、The Ground Truth 评测、CreateAIAgent.net 分析。

---

## 三、横向对比矩阵

| 维度                      | Claude Code                   | Copilot CLI                | Cursor              | Windsurf            | Devin                     |
| ------------------------- | ----------------------------- | -------------------------- | ------------------- | ------------------- | ------------------------- |
| **自动 compact**          | ✅ 多层级瀑布                 | ✅ 后台 80% 触发           | ✅ 100% 自动触发    | ✅ 偶发 summarize   | ✅ 自建系统               |
| **手动 compact**          | ✅ `/compact` + 指令          | ✅ `/compact`              | ✅ `/compress`      | ❌ 无明确命令       | ❌ 无公开命令             |
| **触发阈值**              | ~93.5%（近期前移至 64-75%）   | 80% 后台 / 95% 硬暂停      | 100% 自动           | 未公开              | 编程式 cap                |
| **Observation Masking**   | ✅ Microcompact + Snip        | ✅ 8KB threshold + flex    | ❌ 无独立层         | ❌ 不适用           | 未公开                    |
| **LLM Summarization**     | ✅ Full compact               | ✅ Background compact      | ✅ Flash model      | ✅ 偶发             | ✅ Fine-tuned small model |
| **History Retrieval**     | ❌（依赖 session memory）     | ❌                         | ✅ History-as-files | ✅ Codebase index   | ✅ Knowledge Base RAG     |
| **Persistent Memory**     | ✅ CLAUDE.md + session memory | ✅ copilot-instructions.md | ✅ .cursor/rules    | ✅ Memories + Rules | ✅ Knowledge Base         |
| **Subagent Isolation**    | ✅ 一等公民                   | ✅ 有 subagent             | ❌ 无独立 subagent  | ❌                  | ✅ Multi-model compound   |
| **Custom Summary Prompt** | ✅ API + CLI                  | ❌                         | ❌                  | ❌                  | ❌                        |
| **Context Visibility**    | ✅ `/context`                 | ✅ `/context` + indicator  | ✅ Token indicator  | ✅ Context meter    | ❌ 不透明                 |
| **Compact 可逆性**        | ❌                            | ✅ Checkpoints             | ❌                  | ❌                  | ❌                        |

---

## 四、技术 Taxonomy 映射

```
                    ┌────────────────────────────────────────────────────┐
                    │           Context Pressure 上升                     │
                    └──────────────────┬─────────────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────────────┐
        Level 0     │  源头控制 (Source Gating)                           │
                    │  Claude Code: 50K→disk  Copilot: 8KB→pointer      │
                    └──────────────────┬─────────────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────────────┐
        Level 1     │  Truncation / Snip                                 │
                    │  Claude Code: drop oldest  Copilot: FIFO entries   │
                    └──────────────────┬─────────────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────────────┐
        Level 2     │  Observation Masking / Microcompact                 │
                    │  Claude Code: tool result clear  Copilot: flex trim│
                    └──────────────────┬─────────────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────────────┐
        Level 3     │  LLM Summarization                                 │
                    │  All products (不同模型策略)                        │
                    └──────────────────┬─────────────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────────────┐
  Orthogonal        │  Retrieval (RAG) + State Memory + Scheduling       │
  Strategies        │  Cursor: history-files  Windsurf: memories+index  │
                    │  Devin: knowledge base  Claude Code: subagents    │
                    └────────────────────────────────────────────────────┘
```

---

## 五、成熟度评估与研究价值判断

### 最成熟方案：Claude Code

**理由：**

1. **多层级瀑布架构**是目前公开文档最详细、层次最清晰的设计。从 source gating → snip → microcompact → session memory → full LLM compact，每层有明确的成本/信息保真度权衡。
2. **开源可观测性最强**：源码分析已有多篇深度文章（Finisky Garden 五层分析、GitHub Gist 对比研究），compaction 逻辑可追踪到具体函数（`autoCompact.ts`、`microCompact.ts`、`compact.ts`）。
3. **用户可控性最佳**：`/compact <instructions>` 允许指定 summary focus；CLAUDE.md 可写入 compact 保留策略；Server-side API 支持 `summary_prompt` 和 `pause_after_compaction`。
4. **Subagent isolation 作为 context 管理原语**：这是一个架构层面的创新——不仅是 delegation，更是 context budget management。
5. **工程数据驱动**：circuit breaker（基于 1,279 session 的失败数据）、阈值前移（基于性能退化观测）等决策有量化依据。

### 最具系统研究价值：Claude Code + Devin

- **Claude Code** 适合研究 compaction pipeline 工程设计、tool output management、prompt caching 与 compaction 的交互。
- **Devin** 适合研究 context anxiety 现象、multi-model handoff summarization、long-horizon task 的记忆管理。Cognition 团队关于 "模型不知道自己不知道什么" 的观察，以及 "大窗口+低 cap" 的反直觉方案，是极有价值的 empirical findings。
- **Cursor** 的 history-as-files 方案为 "summarization + retrieval" 混合范式提供了一个简洁的实现参照。

### 各方案的关键 Trade-off

| 方案                          | 优势                       | 劣势                                  |
| ----------------------------- | -------------------------- | ------------------------------------- |
| 多层瀑布 (Claude Code)        | 渐进式退化、成本可控       | 工程复杂度高、层间交互 subtle         |
| Per-turn Rebuild (Copilot)    | 无累积 drift、每轮最优分配 | 无 prompt cache 复用、checkpoint 开销 |
| Summarize + Retrieve (Cursor) | 可恢复 compact 丢失的细节  | 检索准确性依赖于 query 质量           |
| Persistent Memory (Windsurf)  | 跨 session 连续性好        | 无细粒度 intra-session 压缩           |
| Custom System (Devin)         | 针对 long-horizon 优化     | 不透明、不可复现                      |

---

## 六、未解决的开放问题

1. **Compaction 质量评估**：Factory.ai 的 probe-based evaluation（2025-12）是目前唯一的系统性评测框架，但仅测试了 3 种策略。行业缺乏标准 benchmark。
2. **Prompt Caching vs Compaction 的矛盾**：Cache tokens 跨 sampling loop 累积可能过早触发 compaction（Anthropic 官方提到的 limitation）。
3. **Autonomous Compaction**：LangChain/DeepAgents 的最新探索（2026-04）——让 agent 自主决定 compact 时机。但存在"过于保守"的风险。
4. **Context Anxiety 的模型级解决**：Devin 发现模型会在接近 limit 时行为异变。这是否可以通过 training 解决，还是只能靠 harness 层 workaround？
5. **多 Agent 间的 Memory Coherence**：当 subagent 各自 compact 后，如何保证全局语义一致性？

---

_本报告基于各产品截至 2026-04-23 的公开文档、技术博客、开发者论坛和社区分析整理。_
