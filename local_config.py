"""
Ouroboros — Local configuration for Windows + Ollama.
Optimized for: i7-10510U (4C/8T) + 36GB RAM + no GPU.
"""

import os
import pathlib

# ============================================================
# LLM Configuration (Ollama)
# ============================================================
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Модели — все через Ollama
OUROBOROS_MODEL = "qwen3:14b"             # основная модель (текст)
OUROBOROS_MODEL_CODE = "qwen3:14b"        # модель для кода
OUROBOROS_MODEL_LIGHT = "qwen3:14b"       # лёгкая модель
OUROBOROS_VISION_MODEL = "minicpm-v:8b"   # vision-модель (подгружается по требованию, ~5GB RAM)

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
# Budget (Ollama = бесплатно, но агенту нужна цифра)
# ============================================================
TOTAL_BUDGET = 999999.0

# ============================================================
# === ОПТИМИЗАЦИИ ДЛЯ CPU (i7-10510U, 36GB RAM) ===
# ============================================================

# --- Workers ---
# 1 воркер, не 2 и не 5. Ollama обрабатывает запросы последовательно,
# несколько воркеров только создают очередь и тратят RAM на процессы.
MAX_WORKERS = 1

# --- Max rounds ---
# Лимит раундов LLM на задачу. На CPU каждый раунд = 30-120 сек,
# 50 раундов = максимум ~1.5 часа на задачу. Достаточно для любой работы.
MAX_ROUNDS = 50

# --- Timeouts ---
# Увеличены: на CPU всё медленнее, не нужно убивать задачу раньше времени.
SOFT_TIMEOUT_SEC = 900       # 15 мин (мягкое предупреждение)
HARD_TIMEOUT_SEC = 3600      # 60 мин (жёсткий таймаут)

# --- Background consciousness ---
# На CPU фоновое сознание жрёт те же ресурсы, что основная задача.
# 5% бюджета + увеличенный интервал = не мешает основной работе.
BG_BUDGET_PCT = 5

# --- Response tokens ---
# Чем меньше токенов ответа, тем быстрее генерация.
# 4096 достаточно для большинства задач; агент может вызвать compact_context.
MAX_RESPONSE_TOKENS = 4096

# --- Context compaction ---
# Сжимать контекст после N раундов. Сжимаются ТОЛЬКО старые результаты
# тулзов (git diff, web_search и т.п.), НЕ личность и НЕ память.
# identity.md, scratchpad.md, BIBLE.md, knowledge base — всегда полные.
COMPACT_CONTEXT_AFTER_ROUNDS = 25

# ============================================================
# System Paths (Windows)
# ============================================================
PYTHON_PATH = r"C:\Users\morea\AppData\Local\Programs\Python\Python311\python.exe"
CMD_PATH = r"C:\Windows\System32\cmd.exe"
GIT_PATH = r"C:\Program Files\Git\cmd\git.exe"

# ============================================================
# Ollama Environment (будут установлены при запуске)
# ============================================================
# Количество потоков для инференса (i7-10510U = 4 ядра, 8 потоков)
OLLAMA_NUM_THREADS = 8

# Контекстное окно. gpt-oss:20b поддерживает 128K, но на CPU это убийственно.
# 16384 — хороший баланс: достаточно для кода и промптов, не тормозит.
OLLAMA_NUM_CTX = 16384
