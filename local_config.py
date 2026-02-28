"""
Ouroboros — Local configuration for Windows + OpenRouter.
Cloud inference via OpenRouter free tier.
"""

import os
import pathlib

# ============================================================
# LLM Configuration (OpenRouter)
# ============================================================
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Модели — через OpenRouter
# Основная: Gemini 3 Flash Preview ($0.50/M input, $3.00/M output, 1M context, tool use + reasoning)
OUROBOROS_MODEL = "anthropic/claude-sonnet-4.6"      # основная модель (reasoning + tool use)
OUROBOROS_MODEL_CODE = "anthropic/claude-sonnet-4.6" # модель для кода
OUROBOROS_MODEL_LIGHT = "stepfun/step-3.5-flash:free" # лёгкая модель (бесплатная, для мелких задач)
OUROBOROS_VISION_MODEL = "qwen/qwen3-vl-235b-a22b-thinking" # vision (free via Alibaba Cloud)

# Ollama (оставлено для обратной совместимости, не используется)
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# ============================================================
# Telegram Bot
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# ============================================================
# GitHub (для автопуша эволюции)
# ============================================================
GITHUB_USER = os.environ.get("GITHUB_USER", "Alex214-code")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "ouroboros")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")  # Set via env var! NEVER hardcode tokens!

# Интервал автопуша (секунды)
AUTO_PUSH_INTERVAL_SEC = 300  # каждые 5 минут

# ============================================================
# Пути (Windows)
# ============================================================
REPO_DIR = pathlib.Path(__file__).parent.resolve()
DATA_ROOT = REPO_DIR / "data" / "local_state"

# ============================================================
# Budget (OpenRouter free tier = бесплатно, но агенту нужна цифра)
# ============================================================
TOTAL_BUDGET = 14.17  # OpenRouter balance (Claude Sonnet 4.6: $3/M in + $15/M out)

# ============================================================
# === НАСТРОЙКИ ДЛЯ CLOUD INFERENCE (OpenRouter) ===
# ============================================================

# --- Workers ---
# 1 воркер: free tier имеет rate limit 20 req/min, 200 req/day.
MAX_WORKERS = 1

# --- Max rounds ---
# Cloud inference быстрый, но rate limit ограничивает.
# 50 раундов — достаточно для любой задачи.
MAX_ROUNDS = 50

# --- Timeouts ---
# Cloud inference быстрее CPU, но могут быть задержки на free tier.
SOFT_TIMEOUT_SEC = 300       # 5 мин (мягкое предупреждение)
HARD_TIMEOUT_SEC = 900       # 15 мин (жёсткий таймаут)

# --- Background consciousness ---
BG_BUDGET_PCT = 5

# --- Response tokens ---
MAX_RESPONSE_TOKENS = 4096

# --- Context compaction ---
COMPACT_CONTEXT_AFTER_ROUNDS = 25

# ============================================================
# System Paths (Windows)
# ============================================================
PYTHON_PATH = r"C:\Users\morea\AppData\Local\Programs\Python\Python311\python.exe"
CMD_PATH = r"C:\Windows\System32\cmd.exe"
GIT_PATH = r"C:\Program Files\Git\cmd\git.exe"

# ============================================================
# Ollama Environment (не используется при OpenRouter бэкенде)
# ============================================================
OLLAMA_NUM_THREADS = 8
OLLAMA_NUM_CTX = 16384
