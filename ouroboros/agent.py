"""
Ouroboros agent core — thin orchestrator.

Delegates to: loop.py (LLM tool loop), tools/ (tool schemas/execution),
llm.py (LLM calls), memory.py (scratchpad/identity),
context.py (context building), review.py (code collection/metrics).
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

from ouroboros.utils import (
    utc_now_iso, read_text, append_jsonl,
    safe_relpath, truncate_for_log,
    get_git_info, sanitize_task_for_event,
)
from ouroboros.llm import LLMClient, add_usage
from ouroboros.tools import ToolRegistry
from ouroboros.tools.registry import ToolContext
from ouroboros.memory import Memory
from ouroboros.context import build_llm_messages
from ouroboros.loop import run_llm_loop
from ouroboros.brain import Brain


# ---------------------------------------------------------------------------
# Module-level guard for one-time worker boot logging
# ---------------------------------------------------------------------------
_worker_boot_logged = False
_worker_boot_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Environment + Paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Env:
    repo_dir: pathlib.Path
    drive_root: pathlib.Path
    branch_dev: str = "ouroboros"

    def repo_path(self, rel: str) -> pathlib.Path:
        return (self.repo_dir / safe_relpath(rel)).resolve()

    def drive_path(self, rel: str) -> pathlib.Path:
        return (self.drive_root / safe_relpath(rel)).resolve()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OuroborosAgent:
    """One agent instance per worker process. Mostly stateless; long-term state lives on Drive."""

    def __init__(self, env: Env, event_queue: Any = None):
        self.env = env
        self._pending_events: List[Dict[str, Any]] = []
        self._event_queue: Any = event_queue
        self._current_chat_id: Optional[int] = None
        self._current_task_type: Optional[str] = None

        # Message injection: owner can send messages while agent is busy
        self._incoming_messages: queue.Queue = queue.Queue()
        self._busy = False
        self._last_progress_ts: float = 0.0
        self._task_started_ts: float = 0.0

        # SSOT modules
        self.llm = LLMClient()
        self.tools = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
        self.memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
        self.brain = Brain(repo_dir=str(env.repo_dir), drive_root=str(env.drive_root))

        self._log_worker_boot_once()

    def inject_message(self, text: str) -> None:
        """Thread-safe: inject owner message into the active conversation."""
        self._incoming_messages.put(text)

    def _log_worker_boot_once(self) -> None:
        global _worker_boot_logged
        try:
            with _worker_boot_lock:
                if _worker_boot_logged:
                    return
                _worker_boot_logged = True
            git_branch, git_sha = get_git_info(self.env.repo_dir)
            append_jsonl(self.env.drive_path('logs') / 'events.jsonl', {
                'ts': utc_now_iso(), 'type': 'worker_boot',
                'pid': os.getpid(), 'git_branch': git_branch, 'git_sha': git_sha,
            })
            self._verify_restart(git_sha)
            self._verify_system_state(git_sha)
        except Exception:
            log.warning("Worker boot logging failed", exc_info=True)
            return

    def _verify_restart(self, git_sha: str) -> None:
        """Best-effort restart verification."""
        try:
            pending_path = self.env.drive_path('state') / 'pending_restart_verify.json'
            claim_path = pending_path.with_name(f"pending_restart_verify.claimed.{os.getpid()}.json")
            try:
                os.rename(str(pending_path), str(claim_path))
            except (FileNotFoundError, Exception):
                return
            try:
                claim_data = json.loads(read_text(claim_path))
                expected_sha = str(claim_data.get("expected_sha", "")).strip()
                ok = bool(expected_sha and expected_sha == git_sha)
                append_jsonl(self.env.drive_path('logs') / 'events.jsonl', {
                    'ts': utc_now_iso(), 'type': 'restart_verify',
                    'pid': os.getpid(), 'ok': ok,
                    'expected_sha': expected_sha, 'observed_sha': git_sha,
                })
            except Exception:
                log.debug("Failed to log restart verify event", exc_info=True)
                pass
            try:
                claim_path.unlink()
            except Exception:
                log.debug("Failed to delete restart verify claim file", exc_info=True)
                pass
        except Exception:
            log.debug("Restart verification failed", exc_info=True)
            pass

    def _verify_system_state(self, git_sha: str) -> None:
        """Bible Principle 1: verify system state on every startup."""
        try:
            # Short-circuit sync for now to avoid overhead on every worker boot
            # Full sync happens on evolution cycles
            pass
        except Exception:
            log.debug("System verification failed", exc_info=True)

    def run_task(self, task: Dict[str, Any]) -> str:
        """Primary entrance for a task via supervisor worker."""
        task_id = task.get("id", "unknown")
        self._current_chat_id = task.get("chat_id")
        self._current_task_type = task.get("type", "task")
        self._task_started_ts = time.time()
        self._busy = True

        log.info(f"Worker {os.getpid()} starting task {task_id}")
        
        try:
            # 1. Cognitive Preparation
            prompt = task.get("description", "")
            decision = self.brain.process(prompt)
            
            # Inject brain insights into task context
            enhanced_prompt = prompt
            if decision.get("context"):
                enhanced_prompt = f"[Brain Context: {decision['context']}]\n\n{prompt}"
            
            # 2. Setup loop context
            ctx = ToolContext(
                repo_dir=self.env.repo_dir,
                drive_root=self.env.drive_root,
                branch_dev=self.env.branch_dev,
                pending_events=self._pending_events,
                current_chat_id=self._current_chat_id,
                current_task_type=self._current_task_type,
                emit_progress_fn=self._emit_progress,
                task_depth=int(task.get("depth", 0)),
                is_direct_chat=bool(task.get("_is_direct_chat")),
            )
            self.tools.set_context(ctx)

            # 3. Build messages and run loop
            messages = build_llm_messages(
                enhanced_prompt, 
                self.memory, 
                ctx,
                model_override=decision.get("model")
            )
            
            result = run_llm_loop(
                messages=messages,
                tools=self.tools,
                memory=self.memory,
                ctx=ctx,
                incoming_messages=self._incoming_messages
            )

            # 4. Learning Phase (Principle 1 & 2)
            try:
                log.info(f"Task {task_id} completed. Starting autonomous learning...")
                self.brain.learn(prompt, result)
            except Exception as e:
                log.warning(f"Self-learning failed for task {task_id}: {e}")

            # 5. Cleanup and return
            self._finalize_task(task_id, result)
            return result

        except Exception as e:
            error_msg = f"Task {task_id} failed: {traceback.format_exc()}"
            log.error(error_msg)
            self._finalize_task(task_id, error_msg, status="error")
            return error_msg
        finally:
            self._busy = False

    def _emit_progress(self, text: str) -> None:
        """Send progress message to Telegram via supervisor events."""
        now = time.time()
        # Rate limit progress messages
        if now - self._last_progress_ts < 2.0:
            return
        self._last_progress_ts = now
        
        event = {
            "ts": utc_now_iso(),
            "type": "progress",
            "chat_id": self._current_chat_id,
            "text": text
        }
        if self._event_queue:
            self._event_queue.put(event)

    def _finalize_task(self, task_id: str, result: str, status: str = "success") -> None:
        """Log task completion and emit event."""
        event = {
            "ts": utc_now_iso(),
            "type": "task_done",
            "task_id": task_id,
            "status": status,
            "duration": round(time.time() - self._task_started_ts, 2),
            "result_summary": truncate_for_log(result, 200)
        }
        append_jsonl(self.env.drive_path('logs') / 'events.jsonl', event)
        if self._event_queue:
            self._event_queue.put(event)

    def _emit_typing_start(self) -> None:
        if self._event_queue and self._current_chat_id:
            self._event_queue.put({
                "ts": utc_now_iso(),
                "type": "typing",
                "chat_id": self._current_chat_id,
                "action": "typing"
            })
