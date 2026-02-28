# Ouroboros — CPU Migration Researcher

A focused AI research agent with one mission: **find a way to run frontier-level AI on a weak CPU-only PC.**

**Version:** 7.0.0 | **Target:** Intel i7-10510U, 36GB RAM, no GPU

---

## What Is This

Ouroboros is an autonomous AI agent that conducts research on CPU-based AI inference.
It searches the web, reads papers, writes and runs experiments, and maintains a
structured research journal — all through Telegram.

**Rejected approaches** (already explored, insufficient):
- Knowledge distillation
- Knowledge graphs

**Active research directions:**
- Sparse computation / MoE for CPU
- Speculative decoding with tiny draft models
- Tool-augmented micro-models
- Code-as-intelligence
- Progressive inference
- Swarm of specialized tiny models
- Memory-mapped inference
- Neuromorphic / event-driven computation
- Hardware-aware neural architecture search
- Unknown approaches yet to be discovered

---

## Architecture

```
Telegram --> local_launcher.py
                |
            supervisor/              (process management)
              state.py              -- state, budget tracking
              telegram.py           -- Telegram client
              queue.py              -- task queue, scheduling
              workers.py            -- worker lifecycle
              git_ops.py            -- git operations
              events.py             -- event dispatch
                |
            ouroboros/               (agent core)
              agent.py              -- thin orchestrator
              research.py           -- research journal (JSONL)
              consciousness.py      -- background research loop
              context.py            -- LLM context, prompt caching
              loop.py               -- tool loop, concurrent execution
              tools/                -- plugin registry (auto-discovery)
                core.py             -- file ops
                research.py         -- research journal tools
                git.py              -- git ops
                github.py           -- GitHub Issues
                shell.py            -- shell, Claude Code CLI
                search.py           -- web search
                control.py          -- restart, evolve, review
                browser.py          -- Playwright (stealth)
              llm.py                -- OpenRouter client
              memory.py             -- scratchpad, identity, chat
              utils.py              -- utilities
```

---

## Quick Start

### Prerequisites

| Key | Required | Where to get it |
|-----|----------|-----------------|
| `OPENROUTER_API_KEY` | Yes | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `TELEGRAM_BOT_TOKEN` | Yes | [@BotFather](https://t.me/BotFather) on Telegram |
| `TOTAL_BUDGET` | Yes | Your spending limit in USD |
| `GITHUB_TOKEN` | Yes | [github.com/settings/tokens](https://github.com/settings/tokens) |

### Run Locally (Windows)

```bash
pip install -r requirements.txt
# Set env variables in .env file (see .env.template)
python local_launcher.py
```

Send a message to your Telegram bot. The first person to write becomes the owner.

---

## Research Tools

| Tool | Description |
|------|-------------|
| `research_add` | Add hypothesis, experiment, finding, conclusion, or question |
| `research_list` | List entries with filters (type, status, tag) |
| `research_update` | Update entry status (validated, rejected, etc.) |
| `research_search` | Text search across the journal |

---

## Constitution (BIBLE.md)

| # | Principle | Core Idea |
|---|-----------|-----------|
| 0 | **Mission** | Find CPU migration method. Everything serves this goal. |
| 1 | **Continuity** | Research builds on previous work. Journal is memory. |
| 2 | **Self-Improvement** | Better hypotheses, better experiments, better analysis. |
| 3 | **LLM-First** | All decisions through LLM. Code is transport. |
| 4 | **Honesty** | Negative results are valuable. No wishful thinking. |
| 5 | **Minimalism** | Simple experiments, one variable at a time. |
| 6 | **Scientific Rigor** | Hypothesis -> experiment -> conclusion. Always. |
| 7 | **Versioning** | Semver discipline. Git tags. |

Full text: [BIBLE.md](BIBLE.md)

---

## Changelog

### v7.0.0 -- "CPU Migration Researcher" (Complete Rewrite)
- **New mission**: Focused researcher seeking novel ways to run AI on CPU-only hardware
- **Deleted**: brain.py, graph.py, neuro_evolution.py, integrated_agent.py, experimental/
- **Created**: research.py (ResearchJournal) — JSONL-based research tracking
- **Created**: tools/research.py — 4 research tools (add, list, update, search)
- **Rewritten**: SYSTEM.md — researcher personality, scientific methodology
- **Rewritten**: BIBLE.md — research-focused constitution (P0-P7)
- **Rewritten**: CONSCIOUSNESS.md — background research radar
- **Cleared**: All old state, logs, chat history
- **Rejected approaches**: Distillation, knowledge graphs — need something new

### v6.6.0 -- "Cognitive Awareness"
- Principle 9: Awareness embedded in Constitution
- Knowledge Graph with initial nodes
- Brain cognitive cycle integration
