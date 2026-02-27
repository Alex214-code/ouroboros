"""
Ouroboros Desktop — lightweight web dashboard for Windows.
Runs alongside local_launcher.py, provides a chat UI in the browser.

Usage:
    python desktop/server.py
    (opens http://localhost:8765 in your browser)
"""

import asyncio
import json
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Add parent dir to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

try:
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket
    import uvicorn
except ImportError:
    print("Installing web server dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "starlette", "uvicorn[standard]", "websockets"], check=True)
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket
    import uvicorn

log = logging.getLogger(__name__)

# Paths
ROOT = pathlib.Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "local_state"
STATE_PATH = DATA_ROOT / "state" / "state.json"
CHAT_LOG_PATH = DATA_ROOT / "logs" / "chat.jsonl"
EVENTS_LOG_PATH = DATA_ROOT / "logs" / "supervisor.jsonl"

# Connected WebSocket clients
WS_CLIENTS: List[WebSocket] = []

# Telegram bot proxy (sends messages via bot API)
BOT_TOKEN = ""
OWNER_CHAT_ID = ""


def _load_state() -> Dict[str, Any]:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_chat_log(last_n: int = 50) -> List[Dict[str, Any]]:
    """Load last N lines from chat.jsonl."""
    try:
        if not CHAT_LOG_PATH.exists():
            return []
        lines = CHAT_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        result = []
        for line in lines[-last_n:]:
            try:
                result.append(json.loads(line))
            except Exception:
                pass
        return result
    except Exception:
        return []


def _send_telegram_message(text: str) -> bool:
    """Send a message to the Telegram bot (as if from the owner)."""
    global BOT_TOKEN, OWNER_CHAT_ID
    if not BOT_TOKEN or not OWNER_CHAT_ID:
        st = _load_state()
        OWNER_CHAT_ID = str(st.get("owner_chat_id", ""))
        # Try to get token from env or config
        try:
            import local_config as cfg
            BOT_TOKEN = cfg.TELEGRAM_BOT_TOKEN
        except Exception:
            BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

    if not BOT_TOKEN:
        return False

    # We can't send as the owner through the bot API directly.
    # Instead, we'll write to a command file that the launcher reads.
    cmd_file = DATA_ROOT / "memory" / "desktop_commands.jsonl"
    cmd_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "source": "desktop_ui",
    }
    with open(cmd_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return True


# HTML for the dashboard
def _get_html() -> str:
    return """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ouroboros</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.header {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
}

.header .logo {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

.header h1 {
    font-size: 16px;
    font-weight: 600;
    color: #e6edf3;
}

.header .status-badge {
    margin-left: auto;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
}

.status-badge.online { background: #0d4429; color: #3fb950; }
.status-badge.offline { background: #4a1c1c; color: #f85149; }

.main {
    flex: 1;
    display: flex;
    overflow: hidden;
}

.sidebar {
    width: 240px;
    background: #161b22;
    border-right: 1px solid #30363d;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex-shrink: 0;
    overflow-y: auto;
}

.sidebar h3 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #8b949e;
    margin-bottom: 4px;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 4px 0;
}

.stat-row .label { color: #8b949e; }
.stat-row .value { color: #e6edf3; font-family: 'Consolas', monospace; }

.cmd-btn {
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-family: 'Consolas', monospace;
    text-align: left;
    transition: background 0.15s;
}

.cmd-btn:hover { background: #30363d; }
.cmd-btn.danger { border-color: #f8514950; color: #f85149; }
.cmd-btn.danger:hover { background: #4a1c1c; }

.chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.msg {
    max-width: 80%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
    white-space: pre-wrap;
}

.msg.user {
    align-self: flex-end;
    background: #1f6feb;
    color: #fff;
    border-bottom-right-radius: 4px;
}

.msg.bot {
    align-self: flex-start;
    background: #21262d;
    color: #e6edf3;
    border-bottom-left-radius: 4px;
    border: 1px solid #30363d;
}

.msg.system {
    align-self: center;
    background: transparent;
    color: #8b949e;
    font-size: 12px;
    font-style: italic;
}

.msg .time {
    font-size: 10px;
    color: #8b949e;
    margin-top: 4px;
}

.input-area {
    background: #161b22;
    border-top: 1px solid #30363d;
    padding: 12px 20px;
    display: flex;
    gap: 10px;
    flex-shrink: 0;
}

.input-area textarea {
    flex: 1;
    background: #0d1117;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14px;
    font-family: 'Segoe UI', system-ui, sans-serif;
    resize: none;
    outline: none;
    min-height: 42px;
    max-height: 120px;
}

.input-area textarea:focus {
    border-color: #1f6feb;
}

.input-area button {
    background: #238636;
    border: none;
    color: #fff;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    transition: background 0.15s;
    align-self: flex-end;
}

.input-area button:hover { background: #2ea043; }
.input-area button:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }

.setup-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
}

.setup-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 32px;
    width: 420px;
    max-width: 90vw;
}

.setup-card h2 { color: #e6edf3; margin-bottom: 16px; font-size: 20px; }
.setup-card p { color: #8b949e; margin-bottom: 20px; font-size: 14px; }

.setup-card label {
    display: block;
    color: #c9d1d9;
    font-size: 13px;
    margin-bottom: 4px;
    margin-top: 12px;
}

.setup-card input {
    width: 100%;
    background: #0d1117;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 14px;
    font-family: 'Consolas', monospace;
}

.setup-card input:focus { outline: none; border-color: #1f6feb; }

.setup-card .save-btn {
    margin-top: 20px;
    width: 100%;
    background: #238636;
    border: none;
    color: #fff;
    padding: 10px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
}

.setup-card .save-btn:hover { background: #2ea043; }

@media (max-width: 700px) {
    .sidebar { display: none; }
    .msg { max-width: 95%; }
}
</style>
</head>
<body>

<div class="header">
    <div class="logo">&#x1F40D;</div>
    <h1>Ouroboros</h1>
    <span id="statusBadge" class="status-badge offline">offline</span>
</div>

<div class="main">
    <div class="sidebar">
        <h3>Status</h3>
        <div class="stat-row"><span class="label">Branch</span><span class="value" id="sBranch">-</span></div>
        <div class="stat-row"><span class="label">SHA</span><span class="value" id="sSha">-</span></div>
        <div class="stat-row"><span class="label">Model</span><span class="value" id="sModel">-</span></div>
        <div class="stat-row"><span class="label">Spent</span><span class="value" id="sSpent">-</span></div>
        <div class="stat-row"><span class="label">Workers</span><span class="value" id="sWorkers">-</span></div>
        <div class="stat-row"><span class="label">BG</span><span class="value" id="sBg">-</span></div>

        <h3 style="margin-top:16px">Commands</h3>
        <button class="cmd-btn" onclick="sendCmd('/status')">/status</button>
        <button class="cmd-btn" onclick="sendCmd('/evolve')">/evolve</button>
        <button class="cmd-btn" onclick="sendCmd('/evolve stop')">/evolve stop</button>
        <button class="cmd-btn" onclick="sendCmd('/review')">/review</button>
        <button class="cmd-btn" onclick="sendCmd('/bg start')">/bg start</button>
        <button class="cmd-btn" onclick="sendCmd('/bg stop')">/bg stop</button>
        <button class="cmd-btn" onclick="sendCmd('/push')">/push</button>
        <button class="cmd-btn" onclick="sendCmd('/restart')">/restart</button>
        <button class="cmd-btn danger" onclick="if(confirm('Stop everything?')) sendCmd('/panic')">/panic</button>

        <h3 style="margin-top:16px">Info</h3>
        <div class="stat-row"><span class="label">Mode</span><span class="value">Local</span></div>
        <div class="stat-row"><span class="label">Backend</span><span class="value">Ollama</span></div>
    </div>

    <div class="chat-area">
        <div class="messages" id="messages"></div>
        <div class="input-area">
            <textarea id="msgInput" placeholder="Message Ouroboros..." rows="1"
                      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage()}"></textarea>
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
</div>

<!-- Setup overlay (shown if no Telegram token) -->
<div class="setup-overlay" id="setupOverlay" style="display:none">
    <div class="setup-card">
        <h2>Ouroboros Setup</h2>
        <p>Enter your credentials to get started. These will be saved locally.</p>

        <label>Telegram Bot Token</label>
        <input id="cfgTelegram" type="password" placeholder="7123456789:AAF...">

        <label>GitHub Token (optional)</label>
        <input id="cfgGithub" type="password" placeholder="ghp_...">

        <button class="save-btn" onclick="saveSetup()">Save & Start</button>
    </div>
</div>

<script>
const WS_URL = 'ws://' + location.host + '/ws';
let ws = null;
let reconnectTimer = null;

function connect() {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {
        document.getElementById('statusBadge').className = 'status-badge online';
        document.getElementById('statusBadge').textContent = 'online';
        ws.send(JSON.stringify({type: 'get_history'}));
        ws.send(JSON.stringify({type: 'get_state'}));
    };
    ws.onclose = () => {
        document.getElementById('statusBadge').className = 'status-badge offline';
        document.getElementById('statusBadge').textContent = 'offline';
        reconnectTimer = setTimeout(connect, 3000);
    };
    ws.onerror = () => {};
    ws.onmessage = (evt) => {
        try {
            const data = JSON.parse(evt.data);
            handleMessage(data);
        } catch(e) {}
    };
}

function handleMessage(data) {
    if (data.type === 'chat_history') {
        const el = document.getElementById('messages');
        el.innerHTML = '';
        (data.messages || []).forEach(m => addChatBubble(m));
        el.scrollTop = el.scrollHeight;
    }
    else if (data.type === 'chat_message') {
        addChatBubble(data);
        const el = document.getElementById('messages');
        el.scrollTop = el.scrollHeight;
    }
    else if (data.type === 'state') {
        const s = data.state || {};
        document.getElementById('sBranch').textContent = s.current_branch || '-';
        document.getElementById('sSha').textContent = (s.current_sha || '-').substring(0, 7);
        document.getElementById('sModel').textContent = (s.model || 'gpt-oss:20b').split('/').pop();
        document.getElementById('sSpent').textContent = '$' + (s.spent_usd || 0).toFixed(2);
        document.getElementById('sWorkers').textContent = s.workers || '-';
        document.getElementById('sBg').textContent = s.bg_status || '-';
    }
    else if (data.type === 'need_setup') {
        document.getElementById('setupOverlay').style.display = 'flex';
    }
}

function addChatBubble(m) {
    const el = document.getElementById('messages');
    const div = document.createElement('div');
    const dir = m.direction || m.dir || 'bot';
    div.className = 'msg ' + (dir === 'in' || dir === 'user' ? 'user' : 'bot');
    const text = m.text || m.content || '';
    const timeStr = m.ts ? new Date(m.ts).toLocaleTimeString('ru-RU', {hour:'2-digit',minute:'2-digit'}) : '';
    div.innerHTML = text.replace(/</g,'&lt;').replace(/>/g,'&gt;') +
        (timeStr ? '<div class="time">' + timeStr + '</div>' : '');
    el.appendChild(div);
}

function sendMessage() {
    const input = document.getElementById('msgInput');
    const text = input.value.trim();
    if (!text || !ws || ws.readyState !== 1) return;

    ws.send(JSON.stringify({type: 'send_message', text: text}));
    addChatBubble({direction: 'user', text: text, ts: new Date().toISOString()});
    input.value = '';
    const el = document.getElementById('messages');
    el.scrollTop = el.scrollHeight;
}

function sendCmd(cmd) {
    if (!ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify({type: 'send_message', text: cmd}));
    addChatBubble({direction: 'user', text: cmd, ts: new Date().toISOString()});
}

function saveSetup() {
    const tg = document.getElementById('cfgTelegram').value.trim();
    const gh = document.getElementById('cfgGithub').value.trim();
    if (!tg) { alert('Telegram token is required'); return; }
    if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({type: 'save_config', telegram_token: tg, github_token: gh}));
    }
    document.getElementById('setupOverlay').style.display = 'none';
}

// Auto-resize textarea
document.getElementById('msgInput').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

connect();

// Refresh state every 5s
setInterval(() => {
    if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({type: 'get_state'}));
    }
}, 5000);
</script>
</body>
</html>"""


# --- API Endpoints ---

async def homepage(request):
    return HTMLResponse(_get_html())


async def api_state(request):
    return JSONResponse(_load_state())


async def api_chat_log(request):
    return JSONResponse(_load_chat_log(100))


# --- WebSocket handler ---

async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    WS_CLIENTS.append(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                continue

            msg_type = data.get("type", "")

            if msg_type == "get_history":
                history = _load_chat_log(100)
                await websocket.send_json({"type": "chat_history", "messages": history})

            elif msg_type == "get_state":
                st = _load_state()
                model = os.environ.get("OUROBOROS_MODEL", "gpt-oss:20b")
                await websocket.send_json({"type": "state", "state": {
                    "current_branch": st.get("current_branch", "-"),
                    "current_sha": st.get("current_sha", "-"),
                    "model": model,
                    "spent_usd": st.get("spent_usd", 0),
                    "workers": st.get("active_workers", "-"),
                    "bg_status": "on" if st.get("bg_running") else "off",
                }})

            elif msg_type == "send_message":
                text = data.get("text", "").strip()
                if text:
                    _send_telegram_message(text)
                    # Also write to chat log for immediate display
                    entry = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "direction": "in",
                        "text": text,
                        "source": "desktop",
                    }
                    try:
                        with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

            elif msg_type == "save_config":
                tg_token = data.get("telegram_token", "")
                gh_token = data.get("github_token", "")
                _save_env_config(tg_token, gh_token)

    except Exception:
        pass
    finally:
        if websocket in WS_CLIENTS:
            WS_CLIENTS.remove(websocket)


def _save_env_config(telegram_token: str, github_token: str):
    """Save tokens to .env file."""
    env_path = ROOT / ".env"
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    # Update or add values
    def set_value(key, val):
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={val}"
                return
        lines.append(f"{key}={val}")

    if telegram_token:
        set_value("TELEGRAM_BOT_TOKEN", telegram_token)
    if github_token:
        set_value("GITHUB_TOKEN", github_token)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Chat log watcher (sends new messages to WebSocket clients) ---

_last_chat_log_size = 0

async def _watch_chat_log():
    """Watch chat.jsonl for new lines and push to WebSocket clients."""
    global _last_chat_log_size
    while True:
        await asyncio.sleep(1)
        try:
            if not CHAT_LOG_PATH.exists():
                continue
            size = CHAT_LOG_PATH.stat().st_size
            if size <= _last_chat_log_size:
                if size < _last_chat_log_size:
                    _last_chat_log_size = 0  # file was truncated/rotated
                continue

            with open(CHAT_LOG_PATH, "r", encoding="utf-8") as f:
                f.seek(_last_chat_log_size)
                new_lines = f.read()
            _last_chat_log_size = size

            for line in new_lines.strip().splitlines():
                try:
                    entry = json.loads(line)
                    # Only push bot responses (not user messages we already showed)
                    if entry.get("source") != "desktop":
                        for client in list(WS_CLIENTS):
                            try:
                                await client.send_json({
                                    "type": "chat_message",
                                    **entry,
                                })
                            except Exception:
                                if client in WS_CLIENTS:
                                    WS_CLIENTS.remove(client)
                except Exception:
                    pass
        except Exception:
            pass


# --- App ---

app = Starlette(
    routes=[
        Route("/", homepage),
        Route("/api/state", api_state),
        Route("/api/chat", api_chat_log),
        WebSocketRoute("/ws", ws_endpoint),
    ],
    on_startup=[lambda: asyncio.create_task(_watch_chat_log())],
)


def main():
    # Load .env if exists
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    port = int(os.environ.get("OUROBOROS_DASHBOARD_PORT", "8765"))
    print(f"\n  Ouroboros Dashboard: http://localhost:{port}\n")

    # Open browser after short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
