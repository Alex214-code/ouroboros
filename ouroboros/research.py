"""
Research Journal for Ouroboros.
JSONL-based research tracking: hypotheses, experiments, findings, conclusions.
Replaces Brain + KnowledgeGraph with a simple, file-based approach.
"""

import json
import os
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

ENTRY_TYPES = {"hypothesis", "experiment", "finding", "conclusion", "question"}
STATUSES = {"open", "in_progress", "validated", "rejected", "archived"}


class ResearchJournal:
    """JSONL-backed research journal. Each entry is a line in the file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_entry(
        self,
        entry_type: str,
        title: str,
        content: str,
        status: str = "open",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Add a research entry. Returns the entry ID."""
        if entry_type not in ENTRY_TYPES:
            raise ValueError(f"Invalid type '{entry_type}'. Must be one of: {ENTRY_TYPES}")
        if status not in STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of: {STATUSES}")

        entry_id = uuid.uuid4().hex[:8]
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        entry = {
            "id": entry_id,
            "type": entry_type,
            "title": title,
            "content": content,
            "status": status,
            "tags": tags or [],
            "created_at": now,
            "updated_at": now,
        }
        self._append(entry)
        return entry_id

    def update_status(self, entry_id: str, new_status: str, note: str = "") -> bool:
        """Append a status update record for an existing entry."""
        if new_status not in STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of: {STATUSES}")

        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        update = {
            "id": entry_id,
            "type": "_status_update",
            "new_status": new_status,
            "note": note,
            "updated_at": now,
        }
        self._append(update)
        return True

    def get_entries(
        self,
        entry_type: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Read entries with optional filters. Applies status updates."""
        entries, updates = self._load()

        # Apply status updates
        for entry in entries:
            eid = entry["id"]
            if eid in updates:
                last_update = updates[eid][-1]
                entry["status"] = last_update["new_status"]
                entry["updated_at"] = last_update["updated_at"]
                if last_update.get("note"):
                    entry["status_note"] = last_update["note"]

        # Filter
        if entry_type:
            entries = [e for e in entries if e["type"] == entry_type]
        if status:
            entries = [e for e in entries if e["status"] == status]
        if tag:
            entries = [e for e in entries if tag in e.get("tags", [])]

        return entries[-limit:]

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Simple text search across titles and content."""
        query_lower = query.lower()
        entries = self.get_entries()
        results = []
        for entry in entries:
            text = (entry.get("title", "") + " " + entry.get("content", "")).lower()
            if query_lower in text:
                results.append(entry)
        return results[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """Statistical summary: counts by type and status."""
        entries = self.get_entries(limit=10000)
        by_type: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        for e in entries:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
            by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        return {"total": len(entries), "by_type": by_type, "by_status": by_status}

    def get_context_summary(self, max_entries: int = 20) -> str:
        """Text summary formatted for LLM context injection."""
        entries = self.get_entries(limit=max_entries)
        if not entries:
            return ""

        active = [e for e in entries if e["status"] in ("open", "in_progress")]
        resolved = [e for e in entries if e["status"] in ("validated", "rejected", "archived")]

        parts = []
        summary = self.get_summary()
        parts.append(f"Total entries: {summary['total']}. "
                      f"Active: {len(active)}, Resolved: {len(resolved)}.")

        if active:
            parts.append("\n### Active Research:")
            for e in active[-10:]:
                tags = f" [{', '.join(e['tags'])}]" if e.get("tags") else ""
                note = f" | Note: {e['status_note']}" if e.get("status_note") else ""
                parts.append(f"- [{e['type'].upper()}] **{e['title']}** ({e['status']}){tags}{note}")
                if e.get("content"):
                    preview = e["content"][:200]
                    parts.append(f"  {preview}")

        if resolved:
            parts.append("\n### Resolved:")
            for e in resolved[-10:]:
                parts.append(f"- [{e['type'].upper()}] {e['title']} → {e['status']}")

        return "\n".join(parts)

    # --- Internal ---

    def _append(self, record: Dict[str, Any]) -> None:
        """Append a JSON record to the journal file."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load(self):
        """Load all entries and status updates from the journal file."""
        entries: List[Dict[str, Any]] = []
        updates: Dict[str, List[Dict[str, Any]]] = {}

        if not self.path.exists():
            return entries, updates

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("type") == "_status_update":
                            eid = record["id"]
                            if eid not in updates:
                                updates[eid] = []
                            updates[eid].append(record)
                        else:
                            entries.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.warning(f"Failed to load research journal: {e}")

        return entries, updates
