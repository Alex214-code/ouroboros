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
from ouroboros.brain import Brain # Added Brain import


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
        self.brain = Brain(repo_dir=str(env.repo_dir), drive_root=str(env.drive_root)) # Initialize Brain

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

    def _check_uncommitted_changes(self) -> Tuple[dict, int]:
        """Check for uncommitted changes and attempt auto-rescue commit & push."""
        import re
        import subprocess
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.env.repo_dir),
                capture_output=True, text=True, timeout=10, check=True
            )
            dirty_files = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if dirty_files:
                # Auto-rescue: commit and push
                auto_committed = False
                try:
                    # Only stage tracked files (not secrets/notebooks)
                    subprocess.run(["git", "add", "-u"], cwd=str(self.env.repo_dir), timeout=10, check=True)
                    subprocess.run(
                        ["git", "commit", "-m", "auto-rescue: uncommitted changes detected on startup"],
                        cwd=str(self.env.repo_dir), timeout=30, check=True
                    )
                    # Validate branch name
                    if not re.match(r'^[a-zA-Z0-9_/-]+$', self.env.branch_dev):
                        raise ValueError(f"Invalid branch name: {self.env.branch_dev}")
                    # Pull with rebase before push
                    subprocess.run(
                        ["git", "pull", "--rebase", "origin", self.env.branch_dev],
                        cwd=str(self.env.repo_dir), timeout=60, check=True
                    )
                    # Push
                    try:
                        subprocess.run(
                            ["git", "push", "origin", self.env.branch_dev],
                            cwd=str(self.env.repo_dir), timeout=60, check=True
                        )
                        auto_committed = True
                        log.warning(f"Auto-rescued {len(dirty_files)} uncommitted files on startup")
                    except subprocess.CalledProcessError:
                        # If push fails, undo the commit
                        subprocess.run(
                            ["git", "reset", "HEAD~1"],
                            cwd=str(self.env.repo_dir), timeout=10, check=True
                        )
                        raise
                except Exception as e:
                    log.warning(f"Failed to auto-rescue uncommitted changes: {e}", exc_info=True)
                return {
                    "status": "warning", "files": dirty_files[:20],
                    "auto_committed": auto_committed,
                }, 1
            else:
                return {"status": "ok"}, 0
        except Exception as e:
            return {"status": "error", "error": str(e)}, 0

    def _check_version_sync(self) -> Tuple[dict, int]:
        """Check VERSION file sync with git tags and pyproject.toml."""
        import subprocess
        import re
        try:
            version_file = read_text(self.env.repo_path("VERSION")).strip()
            issue_count = 0
            result_data = {"version_file": version_file}

            # Check pyproject.toml version
            pyproject_path = self.env.repo_path("pyproject.toml")
            pyproject_content = read_text(pyproject_path)
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject_content, re.MULTILINE)
            if match:
                pyproject_version = match.group(1)
                result_data["pyproject_version"] = pyproject_version
                if version_file != pyproject_version:
                    result_data["status"] = "warning"
                    issue_count += 1

            # Check README.md version (Bible P7: VERSION == README version)
            try:
                readme_content = read_text(self.env.repo_path("README.md"))
                readme_match = re.search(r'\*\*Version:\*\*\s*(\d+\.\d+\.\d+)', readme_content)
                if readme_match:
                    readme_version = readme_match.group(1)
                    result_data["readme_version"] = readme_version
                    if version_file != readme_version:
                        result_data["status"] = "warning"
                        issue_count += 1
            except Exception:
                log.debug("Failed to check README.md version", exc_info=True)

            # Check git tags
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=str(self.env.repo_dir),
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                result_data["status"] = "warning"
                result_data["message"] = "no_tags"
                return result_data, issue_count
            else:
                latest_tag = result.stdout.strip().lstrip('v')
                result_data["latest_tag"] = latest_tag
                if version_file != latest_tag:
                    result_data["status"] = "warning"
                    issue_count += 1

            if issue_count == 0:
                result_data["status"] = "ok"

            return result_data, issue_count
        except Exception as e:
            return {"status": "error", "error": str(e)}, 0

    def _check_budget(self) -> Tuple[dict, int]:
        """Check budget remaining with warning thresholds."""
        try:
            state_path = self.env.drive_path("state") / "state.json"
            state_data = json.loads(read_text(state_path))
            total_budget_str = os.environ.get("TOTAL_BUDGET", "")

            # Handle unset or zero budget gracefully
            if not total_budget_str or float(total_budget_str) == 0:
                return {"status": "unconfigured"}, 0
            else:
                total_budget = float(total_budget_str)
                spent = float(state_data.get("spent_usd", 0))
                remaining = max(0, total_budget - spent)

                # Use percentage-based thresholds (works for any budget size)
                pct_remaining = (remaining / total_budget * 100) if total_budget > 0 else 0
                if pct_remaining < 5:
                    status = "emergency"
                    issues = 1
                elif pct_remaining < 15:
                    status = "critical"
                    issues = 1
                elif pct_remaining < 30:
                    status = "warning"
                    issues = 0
                else:
                    status = "ok"
                    issues = 0

                return {
                    "status": status,
                    "remaining_usd": round(remaining, 2),
                    "total_usd": total_budget,
                    "spent_usd": round(spent, 2),
                }, issues
        except Exception as e:
            return {"status": "error", "error": str(e)}, 0

    def _verify_system_state(self, git_sha: str) -> None:
        """Bible Principle 1: verify system state on every startup.

        Checks:
        - Uncommitted changes (auto-rescue commit & push)
        - VERSION file sync with git tags
        - Budget remaining (warning thresholds)
        """
        checks = {}
        issues = 0
        drive_logs = self.env.drive_path("logs")

        # 1. Uncommitted changes
        checks["uncommitted_changes"], issue_count = self._check_uncommitted_changes()
        issues += issue_count

        # 2. VERSION vs git tag
        checks["version_sync"], issue_count = self._check_version_sync()
        issues += issue_count

        # 3. Budget check
        checks["budget"], issue_count = self._check_budget()
        issues += issue_count

        # Log verification result
        event = {
            "ts": utc_now_iso(),
            "type": "startup_verification",
            "checks": checks,
            "issues_count": issues,
            "git_sha": git_sha,
        }
        append_jsonl(drive_logs / "events.jsonl", event)

        if issues > 0:
            log.warning(f"Startup verification found {issues} issue(s): {checks}")

    # =====================================================================
    # Main entry point
    # =====================================================================

    def _prepare_task_context(self, task: Dict[str, Any]) -> Tuple[ToolContext, List[Dict[str, Any]], Dict[str, Any]]:
        """Set up ToolContext, build messages, return (ctx, messages, cap_info)."""
        drive_logs = self.env.drive_path("logs")
        sanitized_task = sanitize_task_for_event(task, drive_logs)
        append_jsonl(drive_logs / "events.jsonl", {"ts": utc_now_iso(), "type": "task_received", "task": sanitized_task})

        # Set tool context for this task
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

        # Typing indicator via event queue (no direct Telegram API)
        self._emit_typing_start()

        # --- Cognitive Preparation via Brain ---
        prompt = task.get("description", "")
        # Get strategy and context from Brain
        decision = self.brain.process(prompt)
        strategy = decision["strategy"]
        brain_context = decision["context"]
        
        # Inject Brain context into task description for the LLM
        if brain_context:
            context_str = "\n".join([f"- {k}: {v}" for k, v in brain_context.items()])
            task["description"] = f"{prompt}\n\n[Relevant Knowledge from Graph]:\n{context_str}"
            log.info(f"Injected {len(brain_context)} items from Brain into task.")

        # Override model if Brain suggests local core
        if strategy["route"] == "local_core":
             task["model"] = strategy["model"]
             log.info(f"Routing to local core: {strategy['model']}")

        # --- Build context (delegated to context.py) ---
        messages, cap_info = build_llm_messages(
            env=self.env,
            memory=self.memory,
            task=task,
            review_context_builder=self._build_review_context,
        )

        if cap_info.get("trimmed_sections"):
            try:
                append_jsonl(drive_logs / "events.jsonl", {
                    "ts": utc_now_iso(), "type": "context_soft_cap_trim",
                    "task_id": task.get("id", "unknown"),
                    "trimmed": cap_info["trimmed_sections"]
                })
            except Exception: pass

        return ctx, messages, cap_info

    def handle_task(self, task: Dict[str, Any]) -> str:
        """Process a task through the LLM tool loop."""
        self._busy = True
        self._task_started_ts = time.time()
        self._last_progress_ts = self._task_started_ts
        self._current_chat_id = task.get('chat_id')
        self._current_task_type = task.get('type', 'task')
        
        # Task ID for tracking
        task_id = task.get("id", "unknown")

        try:
            # 1. Setup context
            ctx, messages, cap_info = self._prepare_task_context(task)

            # 2. Run the loop
            # Pass LLMClient + ToolRegistry + Messages
            final_answer, usage = run_llm_loop(
                llm=self.llm,
                tools=self.tools,
                messages=messages,
                model_name=task.get('model'),
                max_rounds=int(task.get('max_rounds', 50)),
                incoming_messages=self._incoming_messages,
                emit_progress_fn=self._emit_progress,
                is_direct_chat=bool(task.get("_is_direct_chat")),
            )

            # 3. Post-processing: Learning Loop
            import asyncio
            try:
                # We are in a thread, so use a new event loop or run_coroutine_threadsafe
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.brain.learn(task.get("description", ""), final_answer))
                loop.close()
            except Exception as e:
                log.warning(f"Post-task learning failed: {e}")

            # Usage tracking
            add_usage(self.env.drive_root, usage)

            # 4. Save result
            self._save_task_result(task, final_answer, usage)
            
            # Send task completion event
            if self._event_queue:
                self._event_queue.put({
                    'type': 'task_done',
                    'task_id': task_id,
                    'status': 'success',
                    'result': truncate_for_log(final_answer, 200)
                })

            return final_answer

        except Exception as e:
            err_msg = f"Task failed: {e}\n{traceback.format_exc()}"
            log.error(err_msg)
            self._save_task_result(task, f"ERROR: {e}", {})
            
            if self._event_queue:
                self._event_queue.put({
                    'type': 'task_done',
                    'task_id': task_id,
                    'status': 'error',
                    'error': str(e)
                })
            
            return f"An error occurred: {e}"
        finally:
            self._busy = False
            self._current_chat_id = None
            self._current_task_type = None
            # Clear tool context
            self.tools.set_context(None)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _emit_progress(self, text: str) -> None:
        """Send progress message to the parent via event_queue."""
        self._last_progress_ts = time.time()
        if self._event_queue:
            self._event_queue.put({
                'type': 'progress',
                'ts': utc_now_iso(),
                'text': text,
                'chat_id': self._current_chat_id,
                'task_type': self._current_task_type
            })

    def _emit_typing_start(self) -> None:
        """Notify supervisor to show typing indicator."""
        if self._event_queue:
            self._event_queue.put({
                'type': 'typing',
                'chat_id': self._current_chat_id,
                'active': True
            })

    def _save_task_result(self, task: Dict[str, Any], answer: str, usage: Dict[str, Any]) -> None:
        """Log result to Drive."""
        results_dir = self.env.drive_path('task_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        task_id = task.get("id", "no_id")
        fname = f"{utc_now_iso().replace(':', '-')}_{task_id}.json"
        
        res = {
            "task": task,
            "answer": answer,
            "usage": usage,
            "ts": utc_now_iso(),
        }
        (results_dir / fname).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')

    def _build_review_context(self) -> str:
        """Internal helper for context.py to get code metrics."""
        from ouroboros.review import Reviewer
        reviewer = Reviewer(self.env.repo_dir)
        try:
            return reviewer.get_short_summary()
        except Exception:
            return "Metrical analysis failed."
