"""
Research tools for Ouroboros.
Exposes ResearchJournal as LLM-callable tools.
"""

from __future__ import annotations

import json
from typing import List

from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.research import ResearchJournal


def _get_journal(ctx: ToolContext) -> ResearchJournal:
    """Get or create a ResearchJournal instance for the current context."""
    path = ctx.drive_path("research") / "journal.jsonl"
    return ResearchJournal(path=path)


def _research_add(ctx: ToolContext, entry_type: str, title: str, content: str,
                   status: str = "open", tags: str = "") -> str:
    """Add a research entry to the journal."""
    journal = _get_journal(ctx)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    try:
        entry_id = journal.add_entry(
            entry_type=entry_type,
            title=title,
            content=content,
            status=status,
            tags=tag_list,
        )
        return f"Added {entry_type} '{title}' with id={entry_id}"
    except ValueError as e:
        return f"WARNING: {e}"


def _research_list(ctx: ToolContext, entry_type: str = "", status: str = "",
                    tag: str = "", limit: int = 20) -> str:
    """List research entries with optional filters."""
    journal = _get_journal(ctx)
    entries = journal.get_entries(
        entry_type=entry_type or None,
        status=status or None,
        tag=tag or None,
        limit=limit,
    )
    if not entries:
        return "No entries found."

    lines = [f"Found {len(entries)} entries:\n"]
    for e in entries:
        tags = f" [{', '.join(e['tags'])}]" if e.get("tags") else ""
        note = f" | {e['status_note']}" if e.get("status_note") else ""
        lines.append(f"[{e['id']}] {e['type'].upper()}: {e['title']} ({e['status']}){tags}{note}")
        if e.get("content"):
            lines.append(f"  {e['content'][:300]}")
    return "\n".join(lines)


def _research_update(ctx: ToolContext, entry_id: str, new_status: str,
                      note: str = "") -> str:
    """Update the status of a research entry."""
    journal = _get_journal(ctx)
    try:
        journal.update_status(entry_id, new_status, note)
        return f"Updated {entry_id} -> {new_status}" + (f" ({note})" if note else "")
    except ValueError as e:
        return f"WARNING: {e}"


def _research_search(ctx: ToolContext, query: str, limit: int = 10) -> str:
    """Search research entries by text."""
    journal = _get_journal(ctx)
    results = journal.search(query, limit=limit)
    if not results:
        return f"No entries matching '{query}'."

    lines = [f"Found {len(results)} entries matching '{query}':\n"]
    for e in results:
        lines.append(f"[{e['id']}] {e['type'].upper()}: {e['title']} ({e['status']})")
        if e.get("content"):
            lines.append(f"  {e['content'][:200]}")
    return "\n".join(lines)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="research_add",
            schema={
                "name": "research_add",
                "description": (
                    "Add a research entry to the journal. "
                    "Types: hypothesis, experiment, finding, conclusion, question. "
                    "Statuses: open, in_progress, validated, rejected, archived."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entry_type": {
                            "type": "string",
                            "enum": ["hypothesis", "experiment", "finding", "conclusion", "question"],
                            "description": "Type of research entry",
                        },
                        "title": {
                            "type": "string",
                            "description": "Short title for the entry",
                        },
                        "content": {
                            "type": "string",
                            "description": "Detailed content, observations, or analysis",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["open", "in_progress", "validated", "rejected", "archived"],
                            "default": "open",
                            "description": "Initial status",
                        },
                        "tags": {
                            "type": "string",
                            "default": "",
                            "description": "Comma-separated tags for categorization",
                        },
                    },
                    "required": ["entry_type", "title", "content"],
                },
            },
            handler=_research_add,
            timeout_sec=10,
        ),
        ToolEntry(
            name="research_list",
            schema={
                "name": "research_list",
                "description": "List research journal entries with optional filters by type, status, or tag.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entry_type": {
                            "type": "string",
                            "default": "",
                            "description": "Filter by type (hypothesis, experiment, finding, conclusion, question)",
                        },
                        "status": {
                            "type": "string",
                            "default": "",
                            "description": "Filter by status (open, in_progress, validated, rejected, archived)",
                        },
                        "tag": {
                            "type": "string",
                            "default": "",
                            "description": "Filter by tag",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "description": "Max entries to return",
                        },
                    },
                },
            },
            handler=_research_list,
            timeout_sec=10,
        ),
        ToolEntry(
            name="research_update",
            schema={
                "name": "research_update",
                "description": "Update the status of a research entry (e.g., mark a hypothesis as validated or rejected).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entry_id": {
                            "type": "string",
                            "description": "ID of the entry to update",
                        },
                        "new_status": {
                            "type": "string",
                            "enum": ["open", "in_progress", "validated", "rejected", "archived"],
                            "description": "New status",
                        },
                        "note": {
                            "type": "string",
                            "default": "",
                            "description": "Optional note explaining the status change",
                        },
                    },
                    "required": ["entry_id", "new_status"],
                },
            },
            handler=_research_update,
            timeout_sec=10,
        ),
        ToolEntry(
            name="research_search",
            schema={
                "name": "research_search",
                "description": "Search research journal entries by text query across titles and content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search text",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Max results",
                        },
                    },
                    "required": ["query"],
                },
            },
            handler=_research_search,
            timeout_sec=10,
        ),
    ]
