"""
Desktop automation tools for Ouroboros on Windows.

Provides full control over the Windows desktop:
- Screenshots (full screen or region)
- Mouse control (click, move, scroll, drag)
- Keyboard input (type, hotkeys)
- Window management (list, switch, resize, minimize/maximize)
- Text extraction from screen via OCR (no vision model needed!)
- Visual analysis via Ollama vision model (slow path, only when needed)

Architecture:
  Fast path: pyautogui + mss + OCR -> instant, no LLM cost
  Slow path: Ollama vision model -> loads on demand, ~20s swap

Dependencies: pyautogui, mss, Pillow, pytesseract (optional), pywin32
"""

from __future__ import annotations

import base64
import ctypes
import io
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports with auto-install
# ---------------------------------------------------------------------------

_pyautogui = None
_mss = None
_PIL = None
_pytesseract = None
_win32gui = None
_win32con = None
_win32process = None


def _ensure_deps():
    """Install and import dependencies on first use."""
    global _pyautogui, _mss, _PIL, _pytesseract, _win32gui, _win32con, _win32process

    if _pyautogui is not None:
        return

    missing = []
    try:
        import pyautogui
        _pyautogui = pyautogui
    except ImportError:
        missing.append("pyautogui")

    try:
        import mss
        _mss = mss
    except ImportError:
        missing.append("mss")

    try:
        from PIL import Image
        _PIL = Image
    except ImportError:
        missing.append("Pillow")

    try:
        import win32gui
        import win32con
        import win32process
        _win32gui = win32gui
        _win32con = win32con
        _win32process = win32process
    except ImportError:
        missing.append("pywin32")

    if missing:
        log.info("Installing desktop deps: %s", missing)
        pip_map = {"Pillow": "Pillow", "pyautogui": "pyautogui",
                   "mss": "mss", "pywin32": "pywin32"}
        for pkg in missing:
            pip_name = pip_map.get(pkg, pkg)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        # Re-import after install
        import pyautogui
        _pyautogui = pyautogui
        import mss
        _mss = mss
        from PIL import Image
        _PIL = Image
        try:
            import win32gui, win32con, win32process
            _win32gui = win32gui
            _win32con = win32con
            _win32process = win32process
        except ImportError:
            log.warning("pywin32 failed to import after install, window management limited")

    # Configure pyautogui for safe operation
    _pyautogui.FAILSAFE = True   # move mouse to top-left corner to abort
    _pyautogui.PAUSE = 0.05      # minimal pause between actions (50ms)

    # Try pytesseract (optional - OCR)
    try:
        import pytesseract
        # Check if tesseract binary exists
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for tp in tesseract_paths:
            if os.path.exists(tp):
                pytesseract.pytesseract.tesseract_cmd = tp
                break
        pytesseract.get_tesseract_version()
        _pytesseract = pytesseract
    except Exception:
        log.info("pytesseract/Tesseract not available, OCR will use fallback")


# ---------------------------------------------------------------------------
# Screenshot helpers
# ---------------------------------------------------------------------------

def _take_screenshot(region: Optional[Tuple[int, int, int, int]] = None,
                     scale: float = 1.0) -> Tuple[str, int, int]:
    """Take screenshot, return (base64_png, width, height).

    region: (left, top, width, height) or None for full screen.
    scale: downscale factor (0.5 = half size, saves RAM and tokens).
    """
    _ensure_deps()
    with _mss.mss() as sct:
        if region:
            monitor = {"left": region[0], "top": region[1],
                       "width": region[2], "height": region[3]}
        else:
            monitor = sct.monitors[1]  # primary monitor

        img_raw = sct.grab(monitor)
        img = _PIL.frombytes("RGB", img_raw.size, img_raw.bgra, "raw", "BGRX")

    if scale < 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), _PIL.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, img.width, img.height


def _screenshot_to_pil(region=None):
    """Take screenshot, return PIL Image object (for OCR)."""
    _ensure_deps()
    with _mss.mss() as sct:
        if region:
            monitor = {"left": region[0], "top": region[1],
                       "width": region[2], "height": region[3]}
        else:
            monitor = sct.monitors[1]
        img_raw = sct.grab(monitor)
        return _PIL.frombytes("RGB", img_raw.size, img_raw.bgra, "raw", "BGRX")


# ---------------------------------------------------------------------------
# OCR (fast path - no LLM needed)
# ---------------------------------------------------------------------------

def _ocr_screen(region=None, lang="eng+rus") -> str:
    """Extract text from screen using Tesseract OCR. Fast, no LLM cost."""
    _ensure_deps()
    img = _screenshot_to_pil(region)

    if _pytesseract is not None:
        try:
            text = _pytesseract.image_to_string(img, lang=lang)
            return text.strip()
        except Exception as e:
            log.warning("pytesseract OCR failed: %s", e)

    # Fallback: no OCR available
    return ("WARNING: OCR not available. Install Tesseract-OCR from "
            "https://github.com/UB-Mannheim/tesseract/wiki and pytesseract package.")


def _ocr_find_text(target: str, region=None, lang="eng+rus") -> List[Dict[str, Any]]:
    """Find text on screen using OCR, return list of {text, x, y, w, h, confidence}."""
    _ensure_deps()
    if _pytesseract is None:
        return []

    img = _screenshot_to_pil(region)

    try:
        data = _pytesseract.image_to_data(img, lang=lang, output_type=_pytesseract.Output.DICT)
    except Exception as e:
        log.warning("OCR find_text failed: %s", e)
        return []

    results = []
    n = len(data["text"])
    target_lower = target.lower()

    # Search individual words and consecutive word groups
    for i in range(n):
        word = data["text"][i].strip()
        if not word:
            continue
        # Try matching single word
        if target_lower in word.lower():
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            conf = int(data["conf"][i])
            cx = x + w // 2
            cy = y + h // 2
            if region:
                cx += region[0]
                cy += region[1]
            results.append({"text": word, "x": cx, "y": cy,
                            "w": w, "h": h, "confidence": conf})

    # Try matching multi-word target across consecutive words on same line
    if " " in target and not results:
        target_words = target_lower.split()
        for i in range(n - len(target_words) + 1):
            match = True
            for j, tw in enumerate(target_words):
                actual = data["text"][i + j].strip().lower()
                if tw not in actual:
                    match = False
                    break
                # Must be same line
                if j > 0 and data["line_num"][i + j] != data["line_num"][i]:
                    match = False
                    break
            if match:
                x1 = data["left"][i]
                y1 = data["top"][i]
                last = i + len(target_words) - 1
                x2 = data["left"][last] + data["width"][last]
                y2 = max(data["top"][k] + data["height"][k] for k in range(i, last + 1))
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if region:
                    cx += region[0]
                    cy += region[1]
                matched_text = " ".join(data["text"][i + k].strip() for k in range(len(target_words)))
                results.append({"text": matched_text, "x": cx, "y": cy,
                                "w": x2 - x1, "h": y2 - y1, "confidence": 80})

    return results


# ---------------------------------------------------------------------------
# RAM safety guard (emergency only)
# ---------------------------------------------------------------------------

# Minimum free RAM in GB before we refuse to load vision model.
# Below this threshold, vision calls are blocked and the vision model
# is unloaded from Ollama to free memory.
_RAM_CRITICAL_GB = 3.0


def _get_free_ram_gb() -> float:
    """Return available RAM in GB (Windows)."""
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        return 99.0  # if we can't check, assume OK


def _emergency_unload_vision() -> str:
    """Unload vision model from Ollama to free RAM. Emergency only."""
    vision_model = os.environ.get("OUROBOROS_VISION_MODEL", "minicpm-v:8b")
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import requests
        # Ollama: sending keep_alive=0 unloads the model from memory
        requests.post(
            f"{ollama_base}/api/generate",
            json={"model": vision_model, "keep_alive": 0},
            timeout=10,
        )
        log.warning("Emergency: unloaded vision model '%s' to free RAM", vision_model)
        return f"Unloaded {vision_model} from RAM"
    except Exception as e:
        log.warning("Failed to unload vision model: %s", e)
        return f"Failed to unload: {e}"


# ---------------------------------------------------------------------------
# Vision analysis (slow path - uses Ollama vision model)
# ---------------------------------------------------------------------------

def _analyze_screen_vision(ctx: ToolContext, prompt: str, region=None,
                           model: str = "", scale: float = 0.5) -> str:
    """Analyze screenshot using Ollama vision model.

    SLOW: may trigger model swap (~20-30s). Use OCR first when possible.
    scale=0.5 by default to reduce image size and speed up inference.
    """
    # --- RAM safety check (emergency only) ---
    free_gb = _get_free_ram_gb()
    if free_gb < _RAM_CRITICAL_GB:
        _emergency_unload_vision()
        return (
            f"WARNING: Vision blocked -- critically low RAM ({free_gb:.1f} GB free, "
            f"need at least {_RAM_CRITICAL_GB} GB). Vision model unloaded to protect system. "
            f"Use read_screen (OCR) instead, or close some applications to free memory."
        )

    b64, w, h = _take_screenshot(region=region, scale=scale)

    vision_model = model or os.environ.get("OUROBOROS_VISION_MODEL", "minicpm-v:8b")
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # Use Ollama native API (not OpenAI compat) for vision
    import requests

    try:
        resp = requests.post(
            f"{ollama_base}/api/chat",
            json={
                "model": vision_model,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 4096,
                    "num_predict": 1024,
                },
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("message", {}).get("content", "")

        # Post-call check: if RAM dropped critically during inference, unload
        free_after = _get_free_ram_gb()
        if free_after < _RAM_CRITICAL_GB:
            _emergency_unload_vision()
            log.warning("RAM critical after vision call (%.1f GB). Unloaded vision model.", free_after)

        return answer or "(empty response from vision model)"
    except Exception as e:
        return f"WARNING: Vision analysis failed: {e}"


# ---------------------------------------------------------------------------
# Window management (via win32gui)
# ---------------------------------------------------------------------------

def _list_windows() -> List[Dict[str, Any]]:
    """List all visible windows with titles, positions, and process info."""
    _ensure_deps()
    if _win32gui is None:
        return [{"error": "pywin32 not available"}]

    windows = []

    def enum_callback(hwnd, _):
        if not _win32gui.IsWindowVisible(hwnd):
            return
        title = _win32gui.GetWindowText(hwnd)
        if not title:
            return
        try:
            rect = _win32gui.GetWindowRect(hwnd)
            _, pid = _win32process.GetWindowThreadProcessId(hwnd)
            windows.append({
                "hwnd": hwnd,
                "title": title,
                "pid": pid,
                "x": rect[0], "y": rect[1],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
            })
        except Exception:
            pass

    _win32gui.EnumWindows(enum_callback, None)
    return windows


def _focus_window(title_substr: str) -> str:
    """Bring window to foreground by title substring match."""
    _ensure_deps()
    if _win32gui is None:
        return "WARNING: pywin32 not available"

    windows = _list_windows()
    target_lower = title_substr.lower()
    for w in windows:
        if target_lower in w["title"].lower():
            hwnd = w["hwnd"]
            try:
                # Un-minimize if needed
                if _win32gui.IsIconic(hwnd):
                    _win32gui.ShowWindow(hwnd, 9)  # SW_RESTORE
                _win32gui.SetForegroundWindow(hwnd)
                return f"OK: Focused window: {w['title']}"
            except Exception as e:
                return f"WARNING: Could not focus window: {e}"
    return f"WARNING: No window matching '{title_substr}'"


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _tool_screenshot(ctx: ToolContext, region: str = "", scale: float = 0.5) -> str:
    """Take a screenshot. Returns base64 PNG stored in browser_state for send_photo.

    scale: 0.5 = half resolution (faster, less RAM). 1.0 = full resolution.
    region: "left,top,width,height" or empty for full screen.
    """
    _ensure_deps()
    rgn = None
    if region:
        try:
            parts = [int(x.strip()) for x in region.split(",")]
            if len(parts) == 4:
                rgn = tuple(parts)
        except ValueError:
            return "WARNING: region format: left,top,width,height"

    b64, w, h = _take_screenshot(region=rgn, scale=scale)
    ctx.browser_state.last_screenshot_b64 = b64
    return (
        f"Screenshot captured: {w}x{h} ({len(b64)} bytes base64). "
        f"Use send_photo(image_base64='__last_screenshot__') to send to owner, "
        f"or analyze_desktop(prompt='...') for vision analysis."
    )


def _tool_click(ctx: ToolContext, x: int, y: int, button: str = "left",
                clicks: int = 1) -> str:
    """Click at screen coordinates."""
    _ensure_deps()
    try:
        _pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        return f"OK: {button} click at ({x}, {y}), clicks={clicks}"
    except Exception as e:
        return f"WARNING: Click failed: {e}"


def _clipboard_paste(text: str):
    """Set clipboard to text and paste via Ctrl+V. Works with any Unicode."""
    import ctypes

    CF_UNICODETEXT = 13
    GHND = 0x0042

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    user32.OpenClipboard(0)
    user32.EmptyClipboard()

    # Allocate global memory for the string
    encoded = text.encode("utf-16le") + b"\x00\x00"
    h_mem = kernel32.GlobalAlloc(GHND, len(encoded))
    p_mem = kernel32.GlobalLock(h_mem)
    ctypes.memmove(p_mem, encoded, len(encoded))
    kernel32.GlobalUnlock(h_mem)

    user32.SetClipboardData(CF_UNICODETEXT, h_mem)
    user32.CloseClipboard()

    _pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)


def _tool_type_text(ctx: ToolContext, text: str, interval: float = 0.02) -> str:
    """Type text at current cursor position. For special keys, use press_key."""
    _ensure_deps()
    if text.isascii():
        try:
            _pyautogui.typewrite(text, interval=interval)
            return f"OK: Typed {len(text)} chars"
        except Exception as e:
            return f"WARNING: Type failed: {e}"
    else:
        # Non-ASCII (Cyrillic, etc) — clipboard paste
        try:
            _clipboard_paste(text)
            return f"OK: Typed {len(text)} chars (via clipboard)"
        except Exception as e:
            return f"WARNING: Type failed: {e}"


def _tool_press_key(ctx: ToolContext, keys: str) -> str:
    """Press key or key combination. Examples: 'enter', 'ctrl+c', 'alt+tab', 'win+d'."""
    _ensure_deps()
    try:
        parts = [k.strip() for k in keys.split("+")]
        if len(parts) == 1:
            _pyautogui.press(parts[0])
        else:
            _pyautogui.hotkey(*parts)
        return f"OK: Pressed {keys}"
    except Exception as e:
        return f"WARNING: Key press failed: {e}"


def _tool_move_mouse(ctx: ToolContext, x: int, y: int) -> str:
    """Move mouse to coordinates without clicking."""
    _ensure_deps()
    _pyautogui.moveTo(x, y)
    return f"OK: Mouse moved to ({x}, {y})"


def _tool_scroll(ctx: ToolContext, amount: int = 3, x: int = -1, y: int = -1) -> str:
    """Scroll wheel. Positive = up, negative = down. At (x,y) or current position."""
    _ensure_deps()
    kwargs = {"clicks": amount}
    if x >= 0 and y >= 0:
        kwargs["x"] = x
        kwargs["y"] = y
    _pyautogui.scroll(**kwargs)
    direction = "up" if amount > 0 else "down"
    return f"OK: Scrolled {direction} by {abs(amount)}"


def _tool_drag(ctx: ToolContext, start_x: int, start_y: int,
               end_x: int, end_y: int, duration: float = 0.5) -> str:
    """Drag from (start_x, start_y) to (end_x, end_y)."""
    _ensure_deps()
    _pyautogui.moveTo(start_x, start_y)
    _pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
    return f"OK: Dragged ({start_x},{start_y}) -> ({end_x},{end_y})"


def _tool_read_screen(ctx: ToolContext, region: str = "", lang: str = "eng+rus") -> str:
    """Read text from screen using OCR. FAST, no LLM needed.

    region: "left,top,width,height" or empty for full screen.
    """
    rgn = None
    if region:
        try:
            parts = [int(x.strip()) for x in region.split(",")]
            if len(parts) == 4:
                rgn = tuple(parts)
        except ValueError:
            return "WARNING: region format: left,top,width,height"
    return _ocr_screen(region=rgn, lang=lang)


def _tool_find_on_screen(ctx: ToolContext, text: str, region: str = "",
                         lang: str = "eng+rus") -> str:
    """Find text on screen via OCR, return coordinates for clicking. FAST, no LLM.

    Returns JSON list of matches: [{text, x, y, w, h, confidence}, ...]
    x,y = center of the match (ready for click).
    """
    rgn = None
    if region:
        try:
            parts = [int(x.strip()) for x in region.split(",")]
            if len(parts) == 4:
                rgn = tuple(parts)
        except ValueError:
            return "WARNING: region format: left,top,width,height"

    results = _ocr_find_text(text, region=rgn, lang=lang)
    if not results:
        return f"Not found: '{text}'. Try a shorter/different search term."
    return json.dumps(results, ensure_ascii=False, indent=2)


def _tool_list_windows(ctx: ToolContext) -> str:
    """List all visible windows on the desktop."""
    windows = _list_windows()
    # Compact output: title + position
    compact = []
    for w in windows:
        compact.append({
            "hwnd": w["hwnd"],
            "title": w["title"][:80],
            "pid": w["pid"],
            "pos": f"{w['x']},{w['y']} {w['width']}x{w['height']}",
        })
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _tool_focus_window(ctx: ToolContext, title: str) -> str:
    """Bring a window to foreground by title substring match."""
    return _focus_window(title)


def _tool_resize_window(ctx: ToolContext, title: str, width: int, height: int,
                        x: int = -1, y: int = -1) -> str:
    """Resize and optionally move a window."""
    _ensure_deps()
    if _win32gui is None:
        return "WARNING: pywin32 not available"

    windows = _list_windows()
    target_lower = title.lower()
    for w in windows:
        if target_lower in w["title"].lower():
            hwnd = w["hwnd"]
            try:
                if x < 0:
                    x = w["x"]
                if y < 0:
                    y = w["y"]
                _win32gui.MoveWindow(hwnd, x, y, width, height, True)
                return f"OK: Resized '{w['title'][:40]}' to {width}x{height} at ({x},{y})"
            except Exception as e:
                return f"WARNING: Resize failed: {e}"
    return f"WARNING: No window matching '{title}'"


def _tool_minimize_window(ctx: ToolContext, title: str) -> str:
    """Minimize a window by title."""
    _ensure_deps()
    if _win32gui is None:
        return "WARNING: pywin32 not available"
    windows = _list_windows()
    for w in windows:
        if title.lower() in w["title"].lower():
            _win32gui.ShowWindow(w["hwnd"], 6)  # SW_MINIMIZE
            return f"OK: Minimized '{w['title'][:40]}'"
    return f"WARNING: No window matching '{title}'"


def _tool_analyze_desktop(ctx: ToolContext, prompt: str,
                          region: str = "", model: str = "",
                          scale: float = 0.5) -> str:
    """Analyze screen using Ollama vision model. SLOW (~20-60s).

    Use read_screen (OCR) first when possible. This tool is for complex
    visual analysis that OCR cannot handle (icons, images, layouts, etc).

    scale: 0.5 by default (half resolution, faster). Use 1.0 for detail.
    """
    rgn = None
    if region:
        try:
            parts = [int(x.strip()) for x in region.split(",")]
            if len(parts) == 4:
                rgn = tuple(parts)
        except ValueError:
            return "WARNING: region format: left,top,width,height"

    return _analyze_screen_vision(ctx, prompt, region=rgn, model=model, scale=scale)


def _tool_get_mouse_pos(ctx: ToolContext) -> str:
    """Get current mouse position."""
    _ensure_deps()
    pos = _pyautogui.position()
    return f"Mouse at ({pos.x}, {pos.y})"


def _tool_get_screen_size(ctx: ToolContext) -> str:
    """Get screen resolution."""
    _ensure_deps()
    size = _pyautogui.size()
    return f"Screen: {size.width}x{size.height}"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def get_tools() -> List[ToolEntry]:
    """Register all desktop tools. Only on Windows."""
    if os.name != "nt":
        return []

    return [
        # --- Screenshots ---
        ToolEntry("desktop_screenshot", {
            "name": "desktop_screenshot",
            "description": (
                "Take a screenshot of the real desktop (not headless browser). "
                "Returns base64 PNG. Use send_photo to deliver to owner. "
                "scale=0.5 for speed, 1.0 for full resolution."
            ),
            "parameters": {"type": "object", "properties": {
                "region": {"type": "string", "description": "left,top,width,height or empty for full screen"},
                "scale": {"type": "number", "default": 0.5, "description": "Image scale (0.5=half, 1.0=full)"},
            }, "required": []},
        }, _tool_screenshot),

        # --- Mouse ---
        ToolEntry("desktop_click", {
            "name": "desktop_click",
            "description": "Click at screen coordinates (x, y). button: left/right/middle.",
            "parameters": {"type": "object", "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                "clicks": {"type": "integer", "default": 1},
            }, "required": ["x", "y"]},
        }, _tool_click, is_code_tool=True),

        ToolEntry("desktop_move", {
            "name": "desktop_move",
            "description": "Move mouse cursor to (x, y) without clicking.",
            "parameters": {"type": "object", "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            }, "required": ["x", "y"]},
        }, _tool_move_mouse),

        ToolEntry("desktop_scroll", {
            "name": "desktop_scroll",
            "description": "Scroll mouse wheel. Positive=up, negative=down.",
            "parameters": {"type": "object", "properties": {
                "amount": {"type": "integer", "default": 3, "description": "Scroll amount (positive=up, negative=down)"},
                "x": {"type": "integer", "default": -1},
                "y": {"type": "integer", "default": -1},
            }, "required": []},
        }, _tool_scroll),

        ToolEntry("desktop_drag", {
            "name": "desktop_drag",
            "description": "Drag from one point to another.",
            "parameters": {"type": "object", "properties": {
                "start_x": {"type": "integer"}, "start_y": {"type": "integer"},
                "end_x": {"type": "integer"}, "end_y": {"type": "integer"},
                "duration": {"type": "number", "default": 0.5},
            }, "required": ["start_x", "start_y", "end_x", "end_y"]},
        }, _tool_drag, is_code_tool=True),

        # --- Keyboard ---
        ToolEntry("desktop_type", {
            "name": "desktop_type",
            "description": (
                "Type text at current cursor position. "
                "Supports Cyrillic and Unicode (via clipboard fallback). "
                "For special keys, use desktop_press_key."
            ),
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string"},
                "interval": {"type": "number", "default": 0.02},
            }, "required": ["text"]},
        }, _tool_type_text, is_code_tool=True),

        ToolEntry("desktop_press_key", {
            "name": "desktop_press_key",
            "description": (
                "Press a key or key combination. "
                "Examples: 'enter', 'tab', 'escape', 'ctrl+c', 'ctrl+v', "
                "'alt+tab', 'win+d', 'ctrl+shift+esc', 'f5', 'backspace'"
            ),
            "parameters": {"type": "object", "properties": {
                "keys": {"type": "string", "description": "Key or combination like 'ctrl+c'"},
            }, "required": ["keys"]},
        }, _tool_press_key, is_code_tool=True),

        # --- OCR (fast path, no LLM) ---
        ToolEntry("read_screen", {
            "name": "read_screen",
            "description": (
                "Extract ALL text from screen using OCR. FAST (no LLM needed). "
                "Use this first before analyze_desktop. Good for reading text, "
                "error messages, file contents visible on screen."
            ),
            "parameters": {"type": "object", "properties": {
                "region": {"type": "string", "description": "left,top,width,height or empty for full screen"},
                "lang": {"type": "string", "default": "eng+rus"},
            }, "required": []},
        }, _tool_read_screen),

        ToolEntry("find_on_screen", {
            "name": "find_on_screen",
            "description": (
                "Find text on screen via OCR and return click coordinates. "
                "FAST (no LLM needed). Returns {x, y} center of each match. "
                "Use with desktop_click to click on found text."
            ),
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "Text to find on screen"},
                "region": {"type": "string", "description": "left,top,width,height or empty for full screen"},
                "lang": {"type": "string", "default": "eng+rus"},
            }, "required": ["text"]},
        }, _tool_find_on_screen),

        # --- Vision (slow path, Ollama vision model) ---
        ToolEntry("analyze_desktop", {
            "name": "analyze_desktop",
            "description": (
                "Analyze desktop screenshot using Ollama vision model. "
                "SLOW (~20-60s, may trigger model swap). "
                "Use read_screen (OCR) FIRST when you just need to read text. "
                "Use this for: complex UI analysis, icon recognition, layout understanding, "
                "visual verification that OCR cannot handle."
            ),
            "parameters": {"type": "object", "properties": {
                "prompt": {"type": "string", "description": "What to analyze on screen"},
                "region": {"type": "string", "description": "left,top,width,height or empty for full screen"},
                "model": {"type": "string", "description": "Vision model (default: minicpm-v:8b)"},
                "scale": {"type": "number", "default": 0.5},
            }, "required": ["prompt"]},
        }, _tool_analyze_desktop, timeout_sec=180),

        # --- Windows ---
        ToolEntry("desktop_list_windows", {
            "name": "desktop_list_windows",
            "description": "List all visible windows with titles, positions, PIDs.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, _tool_list_windows),

        ToolEntry("desktop_focus_window", {
            "name": "desktop_focus_window",
            "description": "Bring a window to foreground by title substring.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string", "description": "Part of the window title"},
            }, "required": ["title"]},
        }, _tool_focus_window),

        ToolEntry("desktop_resize_window", {
            "name": "desktop_resize_window",
            "description": "Resize and optionally move a window.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"}, "width": {"type": "integer"},
                "height": {"type": "integer"},
                "x": {"type": "integer", "default": -1},
                "y": {"type": "integer", "default": -1},
            }, "required": ["title", "width", "height"]},
        }, _tool_resize_window),

        ToolEntry("desktop_minimize", {
            "name": "desktop_minimize",
            "description": "Minimize a window by title substring.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"},
            }, "required": ["title"]},
        }, _tool_minimize_window),

        # --- Utility ---
        ToolEntry("desktop_mouse_pos", {
            "name": "desktop_mouse_pos",
            "description": "Get current mouse position.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, _tool_get_mouse_pos),

        ToolEntry("desktop_screen_size", {
            "name": "desktop_screen_size",
            "description": "Get screen resolution.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, _tool_get_screen_size),
    ]
