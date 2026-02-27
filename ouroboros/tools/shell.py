"""Shell tools: run_shell, claude_code_edit."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import utc_now_iso, run_cmd, append_jsonl, truncate_for_log

log = logging.getLogger(__name__)


def _parse_cmd(cmd) -> list:
    """Parse cmd argument (string or list) into a list of strings."""
    if isinstance(cmd, str):
        try:
            parsed = json.loads(cmd)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            elif isinstance(parsed, str):
                cmd = parsed
        except Exception:
            pass
        # On Windows, use split() instead of shlex.split() for paths with backslashes
        if os.name == "nt":
            return cmd.split()
        try:
            return shlex.split(cmd)
        except ValueError:
            return cmd.split()
    if isinstance(cmd, list):
        return [str(x) for x in cmd]
    return [str(cmd)]


def _run_shell(ctx: ToolContext, cmd, cwd: str = "") -> str:
    cmd = _parse_cmd(cmd)
    if not cmd:
        return "WARNING: SHELL_ARG_ERROR: cmd is empty."

    # Allow working in any directory, not just repo
    work_dir = ctx.repo_dir
    if cwd and cwd.strip() not in ("", ".", "./"):
        candidate = pathlib.Path(cwd).resolve()
        if not candidate.exists():
            # Try relative to repo
            candidate = (ctx.repo_dir / cwd).resolve()
        if candidate.exists() and candidate.is_dir():
            work_dir = candidate

    # On Windows, use shell=True for built-in commands (dir, type, echo, etc.)
    use_shell = False
    if os.name == "nt" and cmd[0].lower() in (
        "dir", "type", "echo", "copy", "move", "del", "mkdir", "rmdir",
        "ren", "cls", "set", "where", "findstr", "more", "tree",
    ):
        use_shell = True

    try:
        res = subprocess.run(
            cmd, cwd=str(work_dir),
            capture_output=True, text=True, timeout=120,
            shell=use_shell,
        )
        out = res.stdout + ("\n--- STDERR ---\n" + res.stderr if res.stderr else "")
        if len(out) > 50000:
            out = out[:25000] + "\n...(truncated)...\n" + out[-25000:]
        prefix = f"exit_code={res.returncode}\n"
        return prefix + out
    except subprocess.TimeoutExpired:
        return "WARNING: TIMEOUT: command exceeded 120s."
    except Exception as e:
        return f"WARNING: SHELL_ERROR: {e}"


def _run_powershell(ctx: ToolContext, script: str, cwd: str = "") -> str:
    """Run a PowerShell script. Full access to Windows system."""
    work_dir = ctx.repo_dir
    if cwd and cwd.strip() not in ("", ".", "./"):
        candidate = pathlib.Path(cwd).resolve()
        if not candidate.exists():
            candidate = (ctx.repo_dir / cwd).resolve()
        if candidate.exists() and candidate.is_dir():
            work_dir = candidate

    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            cwd=str(work_dir),
            capture_output=True, text=True, timeout=120,
        )
        out = res.stdout + ("\n--- STDERR ---\n" + res.stderr if res.stderr else "")
        if len(out) > 50000:
            out = out[:25000] + "\n...(truncated)...\n" + out[-25000:]
        return f"exit_code={res.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return "WARNING: TIMEOUT: PowerShell exceeded 120s."
    except Exception as e:
        return f"WARNING: POWERSHELL_ERROR: {e}"


def _run_claude_cli(work_dir: str, prompt: str, env: dict) -> subprocess.CompletedProcess:
    """Run Claude CLI with permission-mode fallback."""
    claude_bin = shutil.which("claude")
    cmd = [
        claude_bin, "-p", prompt,
        "--output-format", "json",
        "--max-turns", "12",
        "--tools", "Read,Edit,Grep,Glob",
    ]

    # Try --permission-mode first, fallback to --dangerously-skip-permissions
    perm_mode = os.environ.get("OUROBOROS_CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions").strip()
    primary_cmd = cmd + ["--permission-mode", perm_mode]
    legacy_cmd = cmd + ["--dangerously-skip-permissions"]

    res = subprocess.run(
        primary_cmd, cwd=work_dir,
        capture_output=True, text=True, timeout=300, env=env,
    )

    if res.returncode != 0:
        combined = ((res.stdout or "") + "\n" + (res.stderr or "")).lower()
        if "--permission-mode" in combined and any(
            m in combined for m in ("unknown option", "unknown argument", "unrecognized option", "unexpected argument")
        ):
            res = subprocess.run(
                legacy_cmd, cwd=work_dir,
                capture_output=True, text=True, timeout=300, env=env,
            )

    return res


def _check_uncommitted_changes(repo_dir: pathlib.Path) -> str:
    """Check git status after edit, return warning string or empty string."""
    try:
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status_res.returncode == 0 and status_res.stdout.strip():
            diff_res = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if diff_res.returncode == 0 and diff_res.stdout.strip():
                return (
                    f"\n\nWARNING: UNCOMMITTED CHANGES detected after Claude Code edit:\n"
                    f"{diff_res.stdout.strip()}\n"
                    f"Remember to run git_status and repo_commit_push!"
                )
    except Exception as e:
        log.debug("Failed to check git status after claude_code_edit: %s", e, exc_info=True)
    return ""


def _parse_claude_output(stdout: str, ctx: ToolContext) -> str:
    """Parse JSON output and emit cost event, return result string."""
    try:
        payload = json.loads(stdout)
        out: Dict[str, Any] = {
            "result": payload.get("result", ""),
            "session_id": payload.get("session_id"),
        }
        if isinstance(payload.get("total_cost_usd"), (int, float)):
            ctx.pending_events.append({
                "type": "llm_usage",
                "provider": "claude_code_cli",
                "usage": {"cost": float(payload["total_cost_usd"])},
                "source": "claude_code_edit",
                "ts": utc_now_iso(),
                "category": "task",
            })
        return json.dumps(out, ensure_ascii=False, indent=2)
    except Exception:
        log.debug("Failed to parse claude_code_edit JSON output", exc_info=True)
        return stdout


def _claude_code_edit(ctx: ToolContext, prompt: str, cwd: str = "") -> str:
    """Delegate code edits to Claude Code CLI."""
    from ouroboros.tools.git import _acquire_git_lock, _release_git_lock

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "WARNING: ANTHROPIC_API_KEY not set, claude_code_edit unavailable."

    work_dir = str(ctx.repo_dir)
    if cwd and cwd.strip() not in ("", ".", "./"):
        candidate = (ctx.repo_dir / cwd).resolve()
        if candidate.exists():
            work_dir = str(candidate)

    claude_bin = shutil.which("claude")
    if not claude_bin:
        return "WARNING: Claude CLI not found. Ensure ANTHROPIC_API_KEY is set."

    ctx.emit_progress_fn("Delegating to Claude Code CLI...")

    lock = _acquire_git_lock(ctx)
    try:
        try:
            run_cmd(["git", "checkout", ctx.branch_dev], cwd=ctx.repo_dir)
        except Exception as e:
            return f"WARNING: GIT_ERROR (checkout): {e}"

        full_prompt = (
            f"STRICT: Only modify files inside {work_dir}. "
            f"Git branch: {ctx.branch_dev}. Do NOT commit or push.\n\n"
            f"{prompt}"
        )

        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = api_key
        try:
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                env.setdefault("IS_SANDBOX", "1")
        except Exception:
            log.debug("Failed to check geteuid for sandbox detection", exc_info=True)
            pass
        local_bin = str(pathlib.Path.home() / ".local" / "bin")
        if local_bin not in env.get("PATH", ""):
            env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

        res = _run_claude_cli(work_dir, full_prompt, env)

        stdout = (res.stdout or "").strip()
        stderr = (res.stderr or "").strip()
        if res.returncode != 0:
            return f"WARNING: CLAUDE_CODE_ERROR: exit={res.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        if not stdout:
            stdout = "OK: Claude Code completed with empty output."

        # Check for uncommitted changes and append warning BEFORE finally block
        warning = _check_uncommitted_changes(ctx.repo_dir)
        if warning:
            stdout += warning

    except subprocess.TimeoutExpired:
        return "WARNING: CLAUDE_CODE_TIMEOUT: exceeded 300s."
    except Exception as e:
        return f"WARNING: CLAUDE_CODE_FAILED: {type(e).__name__}: {e}"
    finally:
        _release_git_lock(lock)

    # Parse JSON output and account cost
    return _parse_claude_output(stdout, ctx)


def _fs_read(ctx: ToolContext, path: str) -> str:
    """Read any file from the filesystem by absolute or relative path."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ctx.repo_dir / path
    p = p.resolve()
    if not p.exists():
        return f"WARNING: File not found: {p}"
    if not p.is_file():
        return f"WARNING: Not a file: {p}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > 100000:
            content = content[:50000] + "\n...(truncated)...\n" + content[-50000:]
        return content
    except UnicodeDecodeError:
        return f"WARNING: Cannot read as text (binary file): {p}"
    except Exception as e:
        return f"WARNING: Read error: {e}"


def _fs_write(ctx: ToolContext, path: str, content: str, mode: str = "overwrite") -> str:
    """Write a file to any location on the filesystem."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ctx.repo_dir / path
    p = p.resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with p.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")
        return f"OK: wrote {mode} {p} ({len(content)} chars)"
    except Exception as e:
        return f"WARNING: Write error: {e}"


def _fs_list(ctx: ToolContext, path: str = ".", max_entries: int = 200) -> str:
    """List files and directories at the given path."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ctx.repo_dir / path
    p = p.resolve()
    if not p.exists():
        return f"WARNING: Path not found: {p}"
    if not p.is_dir():
        return f"WARNING: Not a directory: {p}"
    try:
        items = []
        for entry in sorted(p.iterdir()):
            if len(items) >= max_entries:
                items.append(f"...(truncated at {max_entries})")
                break
            suffix = "/" if entry.is_dir() else ""
            size = ""
            if entry.is_file():
                try:
                    s = entry.stat().st_size
                    if s > 1048576:
                        size = f" ({s // 1048576}MB)"
                    elif s > 1024:
                        size = f" ({s // 1024}KB)"
                except Exception:
                    pass
            items.append(f"{entry.name}{suffix}{size}")
        return json.dumps(items, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"WARNING: List error: {e}"


def get_tools() -> List[ToolEntry]:
    tools = [
        ToolEntry("run_shell", {
            "name": "run_shell",
            "description": (
                "Run a shell command. cmd can be a list of args or a string. "
                "cwd can be any absolute path on the filesystem. "
                "Returns stdout+stderr with exit code."
            ),
            "parameters": {"type": "object", "properties": {
                "cmd": {"oneOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]},
                "cwd": {"type": "string", "default": "", "description": "Working directory (absolute path or relative to repo)"},
            }, "required": ["cmd"]},
        }, _run_shell, is_code_tool=True),
        ToolEntry("fs_read", {
            "name": "fs_read",
            "description": "Read any text file from the filesystem. Accepts absolute or relative paths.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            }, "required": ["path"]},
        }, _fs_read),
        ToolEntry("fs_write", {
            "name": "fs_write",
            "description": "Write a text file to any location on the filesystem.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
            }, "required": ["path", "content"]},
        }, _fs_write),
        ToolEntry("fs_list", {
            "name": "fs_list",
            "description": "List files and dirs at any path on the filesystem. Shows names and sizes.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "default": ".", "description": "Absolute or relative directory path"},
                "max_entries": {"type": "integer", "default": 200},
            }, "required": []},
        }, _fs_list),
        ToolEntry("claude_code_edit", {
            "name": "claude_code_edit",
            "description": "Delegate code edits to Claude Code CLI. Preferred for multi-file changes and refactors. Follow with repo_commit_push.",
            "parameters": {"type": "object", "properties": {
                "prompt": {"type": "string"},
                "cwd": {"type": "string", "default": ""},
            }, "required": ["prompt"]},
        }, _claude_code_edit, is_code_tool=True, timeout_sec=300),
    ]
    # Add PowerShell tool on Windows
    if os.name == "nt":
        tools.append(ToolEntry("run_powershell", {
            "name": "run_powershell",
            "description": (
                "Run a PowerShell script/command on Windows. Full access to the system. "
                "Use for complex file operations, system info, registry, services, etc."
            ),
            "parameters": {"type": "object", "properties": {
                "script": {"type": "string", "description": "PowerShell script or command to execute"},
                "cwd": {"type": "string", "default": "", "description": "Working directory"},
            }, "required": ["script"]},
        }, _run_powershell, is_code_tool=True))
    return tools
