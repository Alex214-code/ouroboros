"""
Ouroboros -- Local launcher for Windows + Ollama.
Replaces colab_launcher.py for local execution.

Usage:
    1. Edit local_config.py (Telegram token, GitHub credentials)
    2. Start Ollama: ollama serve
    3. Pull model: ollama pull gpt-oss:20b
    4. Run: python local_launcher.py
"""

import logging
import os
import sys
import json
import time
import uuid
import pathlib
import subprocess
import datetime
import threading
import queue as _queue_mod
from typing import Any, Dict, List, Optional, Set, Tuple

# ----------------------------
# 0) Setup logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ----------------------------
# 0.1) Load .env file (if exists)
# ----------------------------
_env_path = pathlib.Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
    log.info("Loaded .env file")

# ----------------------------
# 0.1b) Load local config
# ----------------------------
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import local_config as cfg

# ----------------------------
# 0.2) Set environment variables BEFORE any imports
# ----------------------------
os.environ["OUROBOROS_LLM_BACKEND"] = "ollama"
os.environ["OLLAMA_BASE_URL"] = cfg.OLLAMA_BASE_URL
os.environ["OUROBOROS_MODEL"] = cfg.OUROBOROS_MODEL
os.environ["OUROBOROS_MODEL_CODE"] = cfg.OUROBOROS_MODEL_CODE
os.environ["OUROBOROS_MODEL_LIGHT"] = cfg.OUROBOROS_MODEL_LIGHT
os.environ["TELEGRAM_BOT_TOKEN"] = cfg.TELEGRAM_BOT_TOKEN
os.environ["GITHUB_USER"] = cfg.GITHUB_USER
os.environ["GITHUB_REPO"] = cfg.GITHUB_REPO
os.environ["GITHUB_TOKEN"] = cfg.GITHUB_TOKEN
os.environ["TOTAL_BUDGET"] = str(cfg.TOTAL_BUDGET)
os.environ["OUROBOROS_MAX_ROUNDS"] = str(cfg.MAX_ROUNDS)
os.environ["OUROBOROS_BG_BUDGET_PCT"] = str(cfg.BG_BUDGET_PCT)
# Dummy keys so assertions don't fail
os.environ.setdefault("OPENROUTER_API_KEY", "local-ollama-no-key-needed")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("ANTHROPIC_API_KEY", "")

# Ollama CPU optimizations
os.environ.setdefault("OLLAMA_NUM_THREADS", str(cfg.OLLAMA_NUM_THREADS))
os.environ.setdefault("OLLAMA_NUM_CTX", str(cfg.OLLAMA_NUM_CTX))
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")      # 1 request at a time (CPU)
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "2")  # gpt-oss + vision (~21GB, ~15GB free)

# Vision model for desktop analysis (loads on demand)
os.environ["OUROBOROS_VISION_MODEL"] = getattr(cfg, "OUROBOROS_VISION_MODEL", "minicpm-v:8b")

# Disable pre-push tests for faster local iteration
os.environ.setdefault("OUROBOROS_PRE_PUSH_TESTS", "0")

DIAG_HEARTBEAT_SEC = 30
DIAG_SLOW_CYCLE_SEC = 20
os.environ["OUROBOROS_DIAG_HEARTBEAT_SEC"] = str(DIAG_HEARTBEAT_SEC)
os.environ["OUROBOROS_DIAG_SLOW_CYCLE_SEC"] = str(DIAG_SLOW_CYCLE_SEC)


# ----------------------------
# Helper functions (defined at module level, safe for child processes)
# ----------------------------

def check_ollama():
    """Check that Ollama is running and the model is available."""
    try:
        import requests
        resp = requests.get(cfg.OLLAMA_BASE_URL.replace("/v1", "/api/tags"), timeout=5)
        if resp.status_code != 200:
            log.error("Ollama is not responding. Start it with: ollama serve")
            sys.exit(1)
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        model_base = cfg.OUROBOROS_MODEL.split(":")[0] if ":" in cfg.OUROBOROS_MODEL else cfg.OUROBOROS_MODEL
        found = any(model_base in m for m in models)
        if not found:
            log.warning(f"Model '{cfg.OUROBOROS_MODEL}' not found in Ollama. Available: {models}")
            log.warning(f"Pulling model: ollama pull {cfg.OUROBOROS_MODEL}")
            subprocess.run(["ollama", "pull", cfg.OUROBOROS_MODEL], check=False)
        else:
            log.info(f"Ollama OK. Model '{cfg.OUROBOROS_MODEL}' available.")
    except Exception as e:
        log.error(f"Cannot connect to Ollama at {cfg.OLLAMA_BASE_URL}: {e}")
        log.error("Make sure Ollama is running: ollama serve")
        sys.exit(1)


def install_deps():
    """Install Python dependencies."""
    req_path = cfg.REPO_DIR / "requirements.txt"
    if req_path.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)],
            check=False,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "openai>=1.0.0", "requests"],
            check=False,
        )


# ============================================================
# MAIN — all operational code lives here.
# On Windows, multiprocessing "spawn" re-imports this module
# in child processes.  Without this guard spawn_workers()
# recurses and crashes with RuntimeError.
# ============================================================

def main():
    from ouroboros.apply_patch import install as install_apply_patch

    check_ollama()
    install_deps()
    install_apply_patch()

    # ----------------------------
    # 1) Paths
    # ----------------------------
    REPO_DIR = cfg.REPO_DIR
    DATA_ROOT = cfg.DATA_ROOT

    for sub in ["state", "logs", "memory", "index", "locks", "archive"]:
        (DATA_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # Clear stale owner mailbox files from previous session
    try:
        from ouroboros.owner_inject import get_pending_path
        _stale_inject = get_pending_path(DATA_ROOT)
        if _stale_inject.exists():
            _stale_inject.unlink(missing_ok=True)
        _mailbox_dir = DATA_ROOT / "memory" / "owner_mailbox"
        if _mailbox_dir.exists():
            for _f in _mailbox_dir.iterdir():
                _f.unlink(missing_ok=True)
    except Exception:
        pass

    CHAT_LOG_PATH = DATA_ROOT / "logs" / "chat.jsonl"
    if not CHAT_LOG_PATH.exists():
        CHAT_LOG_PATH.write_text("", encoding="utf-8")

    # ----------------------------
    # 2) Git constants
    # ----------------------------
    BRANCH_DEV = "ouroboros"
    BRANCH_STABLE = "ouroboros-stable"
    if cfg.GITHUB_TOKEN and cfg.GITHUB_USER:
        REMOTE_URL = f"https://{cfg.GITHUB_TOKEN}:x-oauth-basic@github.com/{cfg.GITHUB_USER}/{cfg.GITHUB_REPO}.git"
    else:
        REMOTE_URL = ""
        log.warning("No GITHUB_TOKEN/GITHUB_USER set. Git push disabled.")

    TOTAL_BUDGET_LIMIT = cfg.TOTAL_BUDGET

    # ----------------------------
    # 3) Initialize supervisor modules
    # ----------------------------
    from supervisor.state import (
        init as state_init, load_state, save_state, append_jsonl,
        update_budget_from_usage, status_text, rotate_chat_log_if_needed,
        init_state,
    )
    state_init(DATA_ROOT, TOTAL_BUDGET_LIMIT)
    init_state()

    from supervisor.telegram import (
        init as telegram_init, TelegramClient, send_with_budget, log_chat,
    )
    TG = TelegramClient(str(cfg.TELEGRAM_BOT_TOKEN))
    telegram_init(
        drive_root=DATA_ROOT,
        total_budget_limit=TOTAL_BUDGET_LIMIT,
        budget_report_every=10,
        tg_client=TG,
    )

    from supervisor.git_ops import (
        init as git_ops_init, ensure_repo_present, checkout_and_reset,
        sync_runtime_dependencies, import_test, safe_restart,
    )
    git_ops_init(
        repo_dir=REPO_DIR, drive_root=DATA_ROOT, remote_url=REMOTE_URL,
        branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
    )

    from supervisor.queue import (
        enqueue_task, enforce_task_timeouts, enqueue_evolution_task_if_needed,
        persist_queue_snapshot, restore_pending_from_snapshot,
        cancel_task_by_id, queue_review_task, sort_pending,
    )

    from supervisor.workers import (
        init as workers_init, get_event_q, WORKERS, PENDING, RUNNING,
        spawn_workers, kill_workers, assign_tasks, ensure_workers_healthy,
        handle_chat_direct, _get_chat_agent, auto_resume_after_restart,
    )
    workers_init(
        repo_dir=REPO_DIR, drive_root=DATA_ROOT, max_workers=cfg.MAX_WORKERS,
        soft_timeout=cfg.SOFT_TIMEOUT_SEC, hard_timeout=cfg.HARD_TIMEOUT_SEC,
        total_budget_limit=TOTAL_BUDGET_LIMIT,
        branch_dev=BRANCH_DEV, branch_stable=BRANCH_STABLE,
    )

    from supervisor.events import dispatch_event

    # ----------------------------
    # 4) Bootstrap repo
    # ----------------------------
    subprocess.run(["git", "config", "user.name", "Ouroboros"],
                   cwd=str(REPO_DIR), check=False)
    subprocess.run(["git", "config", "user.email", "ouroboros@users.noreply.github.com"],
                   cwd=str(REPO_DIR), check=False)

    if REMOTE_URL:
        subprocess.run(["git", "remote", "set-url", "origin", REMOTE_URL],
                       cwd=str(REPO_DIR), check=False)

    rc = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=str(REPO_DIR), capture_output=True, text=True)
    current_branch = rc.stdout.strip() if rc.returncode == 0 else ""
    if current_branch != BRANCH_DEV:
        subprocess.run(["git", "checkout", BRANCH_DEV], cwd=str(REPO_DIR), check=False)

    st = load_state()
    rc_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR),
                            capture_output=True, text=True)
    st["current_branch"] = BRANCH_DEV
    st["current_sha"] = rc_sha.stdout.strip() if rc_sha.returncode == 0 else "unknown"
    save_state(st)

    log.info(f"Repo ready: branch={BRANCH_DEV}, sha={st.get('current_sha', 'unknown')[:8]}")

    # ----------------------------
    # 5) Auto-push daemon thread
    # ----------------------------
    _last_push_ts = time.time()
    _auto_push_lock = threading.Lock()

    def _auto_push_loop():
        nonlocal _last_push_ts
        interval = cfg.AUTO_PUSH_INTERVAL_SEC
        if interval <= 0 or not REMOTE_URL:
            log.info("Auto-push disabled (interval=0 or no remote)")
            return

        log.info(f"Auto-push daemon started (interval={interval}s)")
        while True:
            time.sleep(interval)
            with _auto_push_lock:
                try:
                    rc = subprocess.run(
                        ["git", "log", "--oneline", f"origin/{BRANCH_DEV}..HEAD"],
                        cwd=str(REPO_DIR), capture_output=True, text=True,
                    )
                    unpushed = rc.stdout.strip() if rc.returncode == 0 else ""
                    if not unpushed:
                        continue

                    count = len(unpushed.splitlines())
                    log.info(f"Auto-push: {count} unpushed commit(s), pushing...")

                    subprocess.run(["git", "fetch", "origin"],
                                   cwd=str(REPO_DIR), capture_output=True, check=False)
                    subprocess.run(["git", "pull", "--rebase", "origin", BRANCH_DEV],
                                   cwd=str(REPO_DIR), capture_output=True, check=False)
                    result = subprocess.run(
                        ["git", "push", "origin", BRANCH_DEV],
                        cwd=str(REPO_DIR), capture_output=True, text=True,
                    )
                    if result.returncode == 0:
                        _last_push_ts = time.time()
                        log.info(f"Auto-push: successfully pushed {count} commit(s)")
                        append_jsonl(DATA_ROOT / "logs" / "supervisor.jsonl", {
                            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "type": "auto_push_ok",
                            "commits_pushed": count,
                        })
                    else:
                        log.warning(f"Auto-push failed: {result.stderr}")
                        append_jsonl(DATA_ROOT / "logs" / "supervisor.jsonl", {
                            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "type": "auto_push_failed",
                            "error": result.stderr[:500],
                        })
                except Exception as e:
                    log.warning(f"Auto-push error: {e}", exc_info=True)

    _push_thread = threading.Thread(target=_auto_push_loop, daemon=True)
    _push_thread.start()

    # ----------------------------
    # 6) Start workers
    # ----------------------------
    kill_workers()
    spawn_workers(cfg.MAX_WORKERS)
    restored_pending = restore_pending_from_snapshot()
    persist_queue_snapshot(reason="startup")
    if restored_pending > 0:
        st_boot = load_state()
        if st_boot.get("owner_chat_id"):
            send_with_budget(int(st_boot["owner_chat_id"]),
                             f"Restored pending queue from snapshot: {restored_pending} tasks.")

    append_jsonl(DATA_ROOT / "logs" / "supervisor.jsonl", {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "type": "launcher_start",
        "mode": "local_ollama",
        "branch": st.get("current_branch"),
        "sha": st.get("current_sha"),
        "max_workers": cfg.MAX_WORKERS,
        "model_default": cfg.OUROBOROS_MODEL,
        "model_code": cfg.OUROBOROS_MODEL_CODE,
        "model_light": cfg.OUROBOROS_MODEL_LIGHT,
        "ollama_url": cfg.OLLAMA_BASE_URL,
        "auto_push_interval": cfg.AUTO_PUSH_INTERVAL_SEC,
    })

    # ----------------------------
    # 6.1) Auto-resume after restart
    # ----------------------------
    auto_resume_after_restart()

    # ----------------------------
    # 6.2) Direct-mode watchdog
    # ----------------------------
    def _chat_watchdog_loop():
        soft_warned = False
        while True:
            time.sleep(30)
            try:
                agent = _get_chat_agent()
                if not agent._busy:
                    soft_warned = False
                    continue

                now = time.time()
                idle_sec = now - agent._last_progress_ts
                total_sec = now - agent._task_started_ts

                if idle_sec >= cfg.HARD_TIMEOUT_SEC:
                    _st = load_state()
                    if _st.get("owner_chat_id"):
                        send_with_budget(
                            int(_st["owner_chat_id"]),
                            f"Task stuck ({int(total_sec)}s without progress). Restarting agent.",
                        )
                    reset_chat_agent()
                    soft_warned = False
                    continue

                if idle_sec >= cfg.SOFT_TIMEOUT_SEC and not soft_warned:
                    soft_warned = True
                    _st = load_state()
                    if _st.get("owner_chat_id"):
                        send_with_budget(
                            int(_st["owner_chat_id"]),
                            f"Task running for {int(total_sec)}s, last progress {int(idle_sec)}s ago.",
                        )
            except Exception:
                pass

    _watchdog_thread = threading.Thread(target=_chat_watchdog_loop, daemon=True)
    _watchdog_thread.start()

    # ----------------------------
    # 6.3) Background consciousness
    # ----------------------------
    from ouroboros.consciousness import BackgroundConsciousness

    def _get_owner_chat_id() -> Optional[int]:
        try:
            _st = load_state()
            cid = _st.get("owner_chat_id")
            return int(cid) if cid else None
        except Exception:
            return None

    _consciousness = BackgroundConsciousness(
        drive_root=DATA_ROOT,
        repo_dir=REPO_DIR,
        event_queue=get_event_q(),
        owner_chat_id_fn=_get_owner_chat_id,
    )

    def reset_chat_agent():
        import supervisor.workers as _w
        _w._chat_agent = None

    # ----------------------------
    # 7) Build event context
    # ----------------------------
    import types
    _event_ctx = types.SimpleNamespace(
        DRIVE_ROOT=DATA_ROOT,
        REPO_DIR=REPO_DIR,
        BRANCH_DEV=BRANCH_DEV,
        BRANCH_STABLE=BRANCH_STABLE,
        TG=TG,
        WORKERS=WORKERS,
        PENDING=PENDING,
        RUNNING=RUNNING,
        MAX_WORKERS=cfg.MAX_WORKERS,
        send_with_budget=send_with_budget,
        load_state=load_state,
        save_state=save_state,
        update_budget_from_usage=update_budget_from_usage,
        append_jsonl=append_jsonl,
        enqueue_task=enqueue_task,
        cancel_task_by_id=cancel_task_by_id,
        queue_review_task=queue_review_task,
        persist_queue_snapshot=persist_queue_snapshot,
        safe_restart=safe_restart,
        kill_workers=kill_workers,
        spawn_workers=spawn_workers,
        sort_pending=sort_pending,
        consciousness=_consciousness,
    )

    def _handle_supervisor_command(text: str, chat_id: int, tg_offset: int = 0):
        lowered = text.strip().lower()

        if lowered.startswith("/panic"):
            send_with_budget(chat_id, "PANIC: stopping everything now.")
            kill_workers()
            st2 = load_state()
            st2["tg_offset"] = tg_offset
            save_state(st2)
            raise SystemExit("PANIC")

        if lowered.startswith("/restart"):
            st2 = load_state()
            st2["session_id"] = uuid.uuid4().hex
            st2["tg_offset"] = tg_offset
            save_state(st2)
            send_with_budget(chat_id, "Restarting (soft).")
            kill_workers()
            os.execv(sys.executable, [sys.executable, __file__])

        if lowered.startswith("/status"):
            status = status_text(WORKERS, PENDING, RUNNING, cfg.SOFT_TIMEOUT_SEC, cfg.HARD_TIMEOUT_SEC)
            send_with_budget(chat_id, status, force_budget=True)
            return "[Supervisor handled /status]\n"

        if lowered.startswith("/review"):
            queue_review_task(reason="owner:/review", force=True)
            return "[Supervisor handled /review -- review task queued]\n"

        if lowered.startswith("/evolve"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "on"
            turn_on = action not in ("off", "stop", "0")
            st2 = load_state()
            st2["evolution_mode_enabled"] = bool(turn_on)
            save_state(st2)
            if not turn_on:
                PENDING[:] = [t for t in PENDING if str(t.get("type")) != "evolution"]
                sort_pending()
                persist_queue_snapshot(reason="evolve_off")
            state_str = "ON" if turn_on else "OFF"
            send_with_budget(chat_id, f"Evolution: {state_str}")
            return f"[Supervisor handled /evolve -- evolution toggled {state_str}]\n"

        if lowered.startswith("/bg"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "status"
            if action in ("start", "on", "1"):
                result = _consciousness.start()
                send_with_budget(chat_id, f"BG: {result}")
            elif action in ("stop", "off", "0"):
                result = _consciousness.stop()
                send_with_budget(chat_id, f"BG: {result}")
            else:
                bg_status = "running" if _consciousness.is_running else "stopped"
                send_with_budget(chat_id, f"Background consciousness: {bg_status}")
            return f"[Supervisor handled /bg {action}]\n"

        if lowered.startswith("/push"):
            try:
                rc = subprocess.run(
                    ["git", "push", "origin", BRANCH_DEV],
                    cwd=str(REPO_DIR), capture_output=True, text=True,
                )
                if rc.returncode == 0:
                    send_with_budget(chat_id, "Push OK.")
                else:
                    send_with_budget(chat_id, f"Push failed: {rc.stderr[:200]}")
            except Exception as e:
                send_with_budget(chat_id, f"Push error: {e}")
            return True

        return ""

    # ----------------------------
    # 8) Main polling loop
    # ----------------------------
    offset = int(load_state().get("tg_offset") or 0)
    _last_diag_heartbeat_ts = 0.0
    _last_message_ts: float = time.time()
    _ACTIVE_MODE_SEC: int = 300

    try:
        _consciousness.start()
        log.info("Background consciousness auto-started")
    except Exception as e:
        log.warning("Consciousness auto-start failed: %s", e)

    log.info("=" * 50)
    log.info("Ouroboros LOCAL mode started")
    log.info(f"  Model: {cfg.OUROBOROS_MODEL}")
    log.info(f"  Ollama: {cfg.OLLAMA_BASE_URL}")
    log.info(f"  Workers: {cfg.MAX_WORKERS}")
    log.info(f"  Auto-push: every {cfg.AUTO_PUSH_INTERVAL_SEC}s")
    log.info("  Send any message to your Telegram bot to begin.")
    log.info("=" * 50)

    while True:
        loop_started_ts = time.time()
        rotate_chat_log_if_needed(DATA_ROOT)
        ensure_workers_healthy()

        # Drain worker events
        event_q = get_event_q()
        while True:
            try:
                evt = event_q.get_nowait()
            except _queue_mod.Empty:
                break
            dispatch_event(evt, _event_ctx)

        enforce_task_timeouts()
        enqueue_evolution_task_if_needed()
        assign_tasks()
        persist_queue_snapshot(reason="main_loop")

        _now = time.time()
        _active = (_now - _last_message_ts) < _ACTIVE_MODE_SEC
        _poll_timeout = 0 if _active else 10
        try:
            updates = TG.get_updates(offset=offset, timeout=_poll_timeout)
        except Exception as e:
            append_jsonl(
                DATA_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "telegram_poll_error", "offset": offset, "error": repr(e),
                },
            )
            time.sleep(1.5)
            continue

        for upd in updates:
            offset = int(upd["update_id"]) + 1
            msg = upd.get("message") or upd.get("edited_message") or {}
            if not msg:
                continue

            chat_id = int(msg["chat"]["id"])
            from_user = msg.get("from") or {}
            user_id = int(from_user.get("id") or 0)
            text = str(msg.get("text") or "")
            caption = str(msg.get("caption") or "")
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # Extract image if present
            image_data = None
            if msg.get("photo"):
                best_photo = msg["photo"][-1]
                file_id = best_photo.get("file_id")
                if file_id:
                    b64, mime = TG.download_file_base64(file_id)
                    if b64:
                        image_data = (b64, mime, caption)
            elif msg.get("document"):
                doc = msg["document"]
                mime_type = str(doc.get("mime_type") or "")
                if mime_type.startswith("image/"):
                    file_id = doc.get("file_id")
                    if file_id:
                        b64, mime = TG.download_file_base64(file_id)
                        if b64:
                            image_data = (b64, mime, caption)

            st = load_state()
            if st.get("owner_id") is None:
                st["owner_id"] = user_id
                st["owner_chat_id"] = chat_id
                st["last_owner_message_at"] = now_iso
                save_state(st)
                log_chat("in", chat_id, user_id, text)
                send_with_budget(chat_id, "Owner registered. Ouroboros online (local mode).")
                continue

            if user_id != int(st.get("owner_id")):
                continue

            log_chat("in", chat_id, user_id, text)
            st["last_owner_message_at"] = now_iso
            _last_message_ts = time.time()
            save_state(st)

            # Supervisor commands
            if text.strip().lower().startswith("/"):
                try:
                    result = _handle_supervisor_command(text, chat_id, tg_offset=offset)
                    if result is True:
                        continue
                    elif result:
                        text = result + text
                except SystemExit:
                    raise
                except Exception:
                    log.warning("Supervisor command handler error", exc_info=True)

            if not text and not image_data:
                continue

            _consciousness.inject_observation(f"Owner message: {text[:100]}")

            agent = _get_chat_agent()

            if agent._busy:
                if image_data:
                    if text:
                        agent.inject_message(text)
                    send_with_budget(chat_id, "Photo received, but a task is in progress.")
                elif text:
                    agent.inject_message(text)
            else:
                # Batch-collect burst messages
                _BATCH_WINDOW_SEC = 1.5
                _EARLY_EXIT_SEC = 0.15
                _batch_start = time.time()
                _batch_deadline = _batch_start + _BATCH_WINDOW_SEC
                _batched_texts = [text] if text else []
                _batched_image = image_data

                _batch_state = load_state()
                _batch_state_dirty = False
                while time.time() < _batch_deadline:
                    time.sleep(0.1)
                    try:
                        _extra_updates = TG.get_updates(offset=offset, timeout=0) or []
                    except Exception:
                        _extra_updates = []
                    if not _extra_updates and (time.time() - _batch_start) < _EARLY_EXIT_SEC:
                        break
                    for _upd in _extra_updates:
                        offset = max(offset, int(_upd.get("update_id", offset - 1)) + 1)
                        _msg2 = _upd.get("message") or _upd.get("edited_message") or {}
                        _uid2 = (_msg2.get("from") or {}).get("id")
                        _cid2 = (_msg2.get("chat") or {}).get("id")
                        _txt2 = _msg2.get("text") or _msg2.get("caption") or ""
                        if _uid2 and _batch_state.get("owner_id") and _uid2 == int(_batch_state["owner_id"]):
                            log_chat("in", _cid2, _uid2, _txt2)
                            _batch_state["last_owner_message_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                            _batch_state_dirty = True
                            if _txt2.strip().lower().startswith("/"):
                                try:
                                    _cmd_result = _handle_supervisor_command(_txt2, _cid2, tg_offset=offset)
                                    if _cmd_result is True:
                                        continue
                                    elif _cmd_result:
                                        _txt2 = _cmd_result + _txt2
                                except SystemExit:
                                    raise
                                except Exception:
                                    pass
                            if _txt2:
                                _batched_texts.append(_txt2)
                                _batch_deadline = max(_batch_deadline, time.time() + 0.3)

                if _batch_state_dirty:
                    save_state(_batch_state)

                if len(_batched_texts) > 1:
                    final_text = "\n\n".join(_batched_texts)
                elif _batched_texts:
                    final_text = _batched_texts[0]
                else:
                    final_text = text

                if agent._busy:
                    if final_text:
                        agent.inject_message(final_text)
                else:
                    _consciousness.pause()
                    def _run_task_and_resume(cid, txt, img):
                        try:
                            handle_chat_direct(cid, txt, img)
                        finally:
                            _consciousness.resume()
                    _t = threading.Thread(
                        target=_run_task_and_resume,
                        args=(chat_id, final_text, _batched_image),
                        daemon=True,
                    )
                    try:
                        _t.start()
                    except Exception as _te:
                        log.error("Failed to start chat thread: %s", _te)
                        _consciousness.resume()

        st = load_state()
        st["tg_offset"] = offset
        save_state(st)

        now_epoch = time.time()
        loop_duration_sec = now_epoch - loop_started_ts

        if DIAG_SLOW_CYCLE_SEC > 0 and loop_duration_sec >= float(DIAG_SLOW_CYCLE_SEC):
            append_jsonl(
                DATA_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "main_loop_slow_cycle",
                    "duration_sec": round(loop_duration_sec, 3),
                },
            )

        if DIAG_HEARTBEAT_SEC > 0 and (now_epoch - _last_diag_heartbeat_ts) >= float(DIAG_HEARTBEAT_SEC):
            append_jsonl(
                DATA_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "main_loop_heartbeat",
                    "offset": offset,
                    "workers_total": len(WORKERS),
                    "workers_alive": sum(1 for w in WORKERS.values() if w.proc.is_alive()),
                    "pending_count": len(PENDING),
                    "running_count": len(RUNNING),
                },
            )
            _last_diag_heartbeat_ts = now_epoch

        _loop_sleep = 0.1 if (_now - _last_message_ts) < _ACTIVE_MODE_SEC else 0.5
        time.sleep(_loop_sleep)


# ============================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
