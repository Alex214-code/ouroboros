"""Web search tool — supports DuckDuckGo (free, local) and OpenAI Responses API."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry

import logging
log = logging.getLogger(__name__)


def _web_search_ddg(query: str, max_results: int = 5) -> str:
    """Search via DDGS (DuckDuckGo / metasearch) — free, no API key needed."""
    try:
        from ddgs import DDGS
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ddgs"],
                       check=True)
        from ddgs import DDGS

    try:
        with DDGS(timeout=30) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return json.dumps({"query": query, "answer": "(no results)", "sources": []},
                              ensure_ascii=False, indent=2)

        # Format results into readable text
        answer_parts = []
        sources = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            answer_parts.append(f"{i}. {title}\n{body}")
            if href:
                sources.append({"title": title, "url": href})

        return json.dumps({
            "query": query,
            "answer": "\n\n".join(answer_parts),
            "result_count": len(results),
            "sources": sources,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        log.warning("DuckDuckGo search failed: %s", e)
        return json.dumps({"error": f"DuckDuckGo search failed: {e}"}, ensure_ascii=False)


def _web_search_openai(query: str) -> str:
    """Search via OpenAI Responses API (requires OPENAI_API_KEY)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return ""  # signal to fallback
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=os.environ.get("OUROBOROS_WEBSEARCH_MODEL", "gpt-5"),
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            input=query,
        )
        d = resp.model_dump()
        text = ""
        for item in d.get("output", []) or []:
            if item.get("type") == "message":
                for block in item.get("content", []) or []:
                    if block.get("type") in ("output_text", "text"):
                        text += block.get("text", "")
        return json.dumps({"answer": text or "(no answer)"}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def _web_search(ctx: ToolContext, query: str) -> str:
    """Web search: tries OpenAI first (if key available), falls back to DDGS."""
    backend = os.environ.get("OUROBOROS_LLM_BACKEND", "").lower()

    # For local (ollama) and openrouter backends, use DDGS directly
    if backend in ("ollama", "openrouter"):
        return _web_search_ddg(query)

    # Try OpenAI first (only for anthropic/openai backends)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        result = _web_search_openai(query)
        if result:
            return result

    # Fallback to DDGS
    return _web_search_ddg(query)


def _web_fetch(ctx: ToolContext, url: str) -> str:
    """Fetch a web page and return its text content. Useful for reading articles, docs, etc."""
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        # Try to extract text from HTML
        try:
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text_parts = []
                    self._skip = False
                    self._skip_tags = {"script", "style", "noscript", "nav", "footer", "header"}

                def handle_starttag(self, tag, attrs):
                    if tag in self._skip_tags:
                        self._skip = True

                def handle_endtag(self, tag):
                    if tag in self._skip_tags:
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip:
                        text = data.strip()
                        if text:
                            self.text_parts.append(text)

            parser = TextExtractor()
            parser.feed(resp.text)
            text = "\n".join(parser.text_parts)
        except Exception:
            text = resp.text

        # Truncate if too long
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)"

        return json.dumps({
            "url": url,
            "status": resp.status_code,
            "content": text,
            "length": len(text),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to fetch {url}: {e}"}, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": (
                "Search the web. Returns top results with titles, snippets, and URLs. "
                "Use for researching topics, finding documentation, learning new things."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search query"},
            }, "required": ["query"]},
        }, _web_search),
        ToolEntry("web_fetch", {
            "name": "web_fetch",
            "description": (
                "Fetch and read a web page by URL. Returns extracted text content. "
                "Use after web_search to read full articles, documentation, source code."
            ),
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            }, "required": ["url"]},
        }, _web_fetch),
    ]
