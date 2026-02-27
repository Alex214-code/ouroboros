"""Strip all emoji/unicode symbols from Ouroboros source files.
Run once:  python strip_emoji.py
"""
import pathlib, sys

BASE = pathlib.Path(__file__).parent.resolve()

# All .py files to scan
targets = []
for pattern in ["*.py", "ouroboros/*.py", "ouroboros/tools/*.py", "supervisor/*.py", "desktop/*.py"]:
    targets.extend(BASE.glob(pattern))
targets = sorted(set(targets))

# Emoji -> ASCII replacements
REPLACEMENTS = {
    "\u26a0\ufe0f": "WARNING:",   # WARNING:
    "\u26a0": "WARNING:",          # WARNING:
    "\u2705": "OK:",               # OK:
    "\u274c": "FAIL:",             # FAIL:
    "\u2615": "",                   # 
    "\u231a": "",                   # 
    "\U0001f6d1": "PANIC:",        # PANIC:
    "\U0001f9e0": "[BG]",          # [BG]
    "\U0001f9ec": "[EVO]",         # [EVO]
    "\u267b\ufe0f": "RESTART:",    # RESTART:
    "\u267b": "RESTART:",           # RESTART:
    "\U0001f4ce": "[ATTACH]",      # [ATTACH]
    "\u23f1\ufe0f": "[TIMER]",     # [TIMER]
    "\u23f1": "[TIMER]",            # [TIMER]
    "\u23f0": "[TIMER]",            # [TIMER]
    "\u2699\ufe0f": ">",           # >
    "\u2699": ">",                  # >
    "\u2713": "v",                  # v
    "\u2714": "v",                  # v
    "\u00b7": ".",                  # .
    "\U0001f40d": "",               # 
    "\u2192": "->",                 # ->
    "\u2190": "<-",                 # <-
    "\U0001f525": "[HOT]",         # [HOT]
    "\U0001f4a1": "[IDEA]",        # [IDEA]
    "\U0001f4e6": "[PKG]",         # [PKG]
    "\U0001f50d": "[SEARCH]",      # [SEARCH]
    "\U0001f310": "[WEB]",         # [WEB]
    "\U0001f4dd": "[NOTE]",        # [NOTE]
    "\U0001f504": "[SYNC]",        # [SYNC]
    "\U0001f512": "[LOCK]",        # [LOCK]
    "\U0001f513": "[UNLOCK]",      # [UNLOCK]
    "\U0001f4ca": "[CHART]",       # [CHART]
    "\U0001f680": "[LAUNCH]",      # [LAUNCH]
    "\U0001f3af": "[TARGET]",      # [TARGET]
    "\u270f\ufe0f": "[EDIT]",      # [EDIT]
    "\u270f": "[EDIT]",             # [EDIT]
    "\U0001f4cb": "[LIST]",        # [LIST]
    "\U0001f916": "[BOT]",         # [BOT]
}

# Also remove any remaining emoji in common ranges
import re
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F77F"   # alchemical
    "\U0001F780-\U0001F7FF"   # geometric extended
    "\U0001F800-\U0001F8FF"   # supplemental arrows
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess symbols
    "\U0001FA70-\U0001FAFF"   # symbols extended-A
    "\u2600-\u26FF"            # misc symbols (but keep common ASCII-range)
    "\u2700-\u27BF"            # dingbats
    "\uFE0F"                   # variation selector
    "]+", re.UNICODE
)

changed_files = 0
total_replacements = 0

for fp in targets:
    try:
        text = fp.read_text(encoding="utf-8")
    except Exception:
        continue

    original = text
    count = 0

    # Apply known replacements
    for old, new in REPLACEMENTS.items():
        if old in text:
            n = text.count(old)
            text = text.replace(old, new)
            count += n

    # Fix double-prefix: "WARNING:" -> "WARNING:"
    while "WARNING:" in text:
        text = text.replace("WARNING:", "WARNING:")
        
    # Fix "WARNING: " at start when there's already a prefix
    # e.g. return "WARNING: SHELL_ERROR" is fine, but "WARNING: TIMEOUT" is not

    if text != original:
        fp.write_text(text, encoding="utf-8")
        rel = fp.relative_to(BASE)
        print(f"  [{count:3d} replacements] {rel}")
        changed_files += 1
        total_replacements += count

print(f"\nDone: {changed_files} files changed, {total_replacements} total replacements")
