"""
Ouroboros agent core — thin orchestrator with Cognitive Ecosystem integration.

Это версия агента, которая интегрирует когнитивную экосистему для распределённой обработки задач.
Сложные задачи направляются в экосистему, простые — обрабатываются напрямую.
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
from ouroboros.cognitive_agent import CognitiveEcosystem, CognitiveTask


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
# Agent with Cognitive Ecosystem
# ---------------------------------------------------------------------------

class OuroborosAgentV2:
    """One agent instance per worker process with Cognitive Ecosystem integration."""

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
        
        # Cognitive Ecosystem
        self.cognitive_ecosystem = None
        self._init_cognitive_ecosystem()

        self._log_worker_boot_once()

    def _init_cognitive_ecosystem(self):
        """Initialize cognitive ecosystem if enabled."""
        try:
            # Check if cognitive ecosystem is enabled
            from ouroboros.cognitive_agent import CognitiveEcosystem
            import importlib.util
            
            # Load configuration
            config_path = self.env.repo_path("cognitive_config.json")
            config = {}
            if config_path.exists():
                config = json.loads(config_path.read_text())
            else:
                # Default configuration
                config = {
                    "enabled": True,
                    "local_model": "qwen3:8b",
                    "cloud_model": "deepseek/deepseek-v3.2",
                    "use_cloud_for_complex_tasks": True,
                    "complexity_threshold": 0.7,
                    "min_budget_for_cloud": 0.1,
                    "knowledge_graph_path": str(self.env.drive_path("memory/knowledge_graph.db"))
                }
            
            if config.get("enabled", True):
                self.cognitive_ecosystem = CognitiveEcosystem(config)
                log.info("Cognitive ecosystem initialized successfully")
            else:
                log.info("Cognitive ecosystem disabled in config")
                
        except Exception as e:
            log.warning(f"Failed to initialize cognitive ecosystem: {e}", exc_info=True)
            self.cognitive_ecosystem = None

    def _should_use_cognitive_ecosystem(self, task_text: str, task_context: Dict[str, Any]) -> bool:
        """Determine if task should be processed by cognitive ecosystem."""
        if not self.cognitive_ecosystem:
            return False
        
        # Simple heuristic for now
        complex_keywords = [
            "research", "analyze", "design", "create", "invent", "develop",
            "разработать", "исследовать", "проанализировать", "спроектировать",
            "придумать", "изобрести", "создать", "миграция", "мигрировать",
            "эволюция", "архитектура", "концепция", "новая технология",
            "саморазвитие", "автономный", "сознание", "компаньон",
            "когнитивный", "экосистема", "нейросимбиотический",
            "граф знаний", "дистилляция", "обучение", "самообучение"
        ]
        
        # Check for complex keywords
        task_lower = task_text.lower()
        keyword_match = any(keyword in task_lower for keyword in complex_keywords)
        
        # Check task length (longer tasks might be more complex)
        length_factor = len(task_text) > 500
        
        # Check if budget allows for cloud processing
        budget_remaining = task_context.get("budget_remaining_usd", 0)
        budget_ok = budget_remaining > 0.1
        
        # Decision logic
        if keyword_match and budget_ok:
            return True
        elif length_factor and keyword_match:
            return True
            
        return False

    def _process_with_cognitive_ecosystem(self, task_text: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process task using cognitive ecosystem."""
        try:
            # Create cognitive task
            cognitive_task = CognitiveTask(
                id=f"cog_{int(time.time())}_{hash(task_text) % 10000}",
                input_text=task_text,
                context=task_context,
                priority=1,
                max_iterations=3
            )
            
            # Process through ecosystem
            result = self.cognitive_ecosystem.process(cognitive_task)
            
            log.info(f"Cognitive ecosystem processed task {cognitive_task.id}, "
                    f"components used: {result.get('components_used', [])}, "
                    f"cost: {result.get('total_cost', 0):.4f}, "
                    f"time: {result.get('total_time', 0):.2f}s")
            
            return {
                "type": "cognitive_ecosystem_response",
                "response": result.get("final_response", "No response generated"),
                "ecosystem_result": result,
                "components_used": result.get("components_used", []),
                "total_cost": result.get("total_cost", 0),
                "total_time": result.get("total_time", 0)
            }
            
        except Exception as e:
            log.error(f"Error processing task with cognitive ecosystem: {e}", exc_info=True)
            # Fallback to standard processing
            return None

    def _process_with_standard_llm(self, task: Dict[str, Any], 
                                   ctx: ToolContext, 
                                   messages: List[Dict[str, Any]], 
                                   cap_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process task using standard LLM loop (original implementation)."""
        # This is a simplified version of the original processing logic
        # In production, we would call the original LLM processing
        
        # For now, we'll log that we're using standard processing
        log.info(f"Using standard LLM processing for task: {task.get('text', '')[:100]}...")
        
        # Placeholder - in real implementation, we would call the original agent's processing
        return "Processing via standard LLM (cognitive ecosystem not used)", {"usage": {}}

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

    def _emit_typing_start(self):
        """Signal typing indicator via event queue."""
        if self._event_queue:
            try:
                self._event_queue.put({
                    "type": "typing_start",
                    "chat_id": self._current_chat_id,
                    "task_type": self._current_task_type,
                })
            except Exception:
                log.debug("Failed to emit typing start", exc_info=True)

    def _emit_typing_stop(self):
        """Signal typing stop via event queue."""
        if self._event_queue:
            try:
                self._event_queue.put({
                    "type": "typing_stop",
                    "chat_id": self._current_chat_id,
                    "task_type": self._current_task_type,
                })
            except Exception:
                log.debug("Failed to emit typing stop", exc_info=True)

    def _emit_progress(self, text: str):
        """Emit progress text via event queue."""
        self._last_progress_ts = time.time()
        if self._event_queue:
            try:
                self._event_queue.put({
                    "type": "progress",
                    "text": text,
                    "chat_id": self._current_chat_id,
                    "task_type": self._current_task_type,
                })
            except Exception:
                log.debug("Failed to emit progress", exc_info=True)

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
                    "task_id": task.get("id", ""),
                    "trimmed_sections": cap_info["trimmed_sections"],
                })
            except Exception:
                log.debug("Failed to log soft-cap trim", exc_info=True)

        return ctx, messages, cap_info

    def _build_review_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for multi-model review (only if requested)."""
        return {}

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point: process a single task."""
        try:
            self._task_started_ts = time.time()
            self._last_progress_ts = time.time()
            self._current_chat_id = task.get("chat_id")
            self._current_task_type = task.get("type", "unknown")

            # Check if we should use cognitive ecosystem
            task_text = task.get("text", "")
            task_context = {
                "budget_remaining_usd": self._get_budget_remaining(),
                "task_type": task.get("type", "unknown"),
                "chat_id": task.get("chat_id")
            }
            
            use_cognitive = self._should_use_cognitive_ecosystem(task_text, task_context)
            
            if use_cognitive and self.cognitive_ecosystem:
                self._emit_progress("🚀 Использую когнитивную экосистему для сложной задачи...")
                result = self._process_with_cognitive_ecosystem(task_text, task_context)
                if result:
                    return {
                        "response": result["response"],
                        "usage": {
                            "cognitive_ecosystem": True,
                            "components_used": result.get("components_used", []),
                            "total_cost": result.get("total_cost", 0),
                            "total_time": result.get("total_time", 0)
                        },
                        "cognitive_result": result.get("ecosystem_result", {})
                    }

            # Fallback to standard LLM processing
            self._emit_progress("🤔 Анализирую задачу...")
            
            # Prepare context and messages
            ctx, messages, cap_info = self._prepare_task_context(task)
            
            # Process via standard LLM loop
            response_text, usage = run_llm_loop(
                llm=self.llm,
                tools=self.tools,
                messages=messages,
                task_id=task.get("id"),
                task_type=task.get("type"),
                max_rounds=task.get("max_rounds", 20),
                max_cost_usd=task.get("max_cost_usd", 1.0),
                progress_callback=lambda txt: self._emit_progress(txt),
                loop_timeout=task.get("loop_timeout", 300),
            )

            # Log task completion
            append_jsonl(self.env.drive_path("logs") / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "task_completed",
                "task_id": task.get("id"),
                "task_type": task.get("type"),
                "duration_secs": round(time.time() - self._task_started_ts, 2),
                "usage": usage,
            })

            self._emit_typing_stop()
            return {
                "response": response_text,
                "usage": usage,
            }

        except Exception as e:
            log.error(f"Task processing failed: {e}", exc_info=True)
            self._emit_typing_stop()
            return {
                "response": f"Ошибка обработки задачи: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _get_budget_remaining(self) -> float:
        """Get remaining budget from state."""
        try:
            state_path = self.env.drive_path("state") / "state.json"
            state_data = json.loads(read_text(state_path))
            total_budget = float(os.environ.get("TOTAL_BUDGET", "0"))
            spent = float(state_data.get("spent_usd", 0))
            return max(0, total_budget - spent)
        except Exception:
            return 0.0

    def __call__(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process_task."""
        return self.process_task(task)