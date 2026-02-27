"""
Central Brain orchestration for Ouroboros.
Decides between local/cloud models, manages knowledge extraction,
and coordinates with the Knowledge Graph.
"""

import logging
import time
import os
from typing import Dict, Any, List, Optional, Tuple

from ouroboros.graph import KnowledgeGraph
import ouroboros.llm as llm_client

log = logging.getLogger(__name__)

# Constants for model routing
LOCAL_MODEL = "qwen2.5:0.5b"  # Ollama model for fast reflexes
DEEP_MODEL = "google/gemini-2.0-flash-thinking-exp:free" # Powerful model for learning
FAST_MODEL = "google/gemini-2.0-flash-exp:free" # Fast cloud model

class Brain:
    def __init__(self, repo_dir: str, drive_root: str):
        self.repo_dir = repo_dir
        self.drive_root = drive_root
        self.graph = KnowledgeGraph(os.path.join(drive_root, "memory/graph.json"))
        self.llm = llm_client.LLMClient()
        
    def process(self, prompt: str) -> Dict[str, Any]:
        """
        Analyzes the task and returns a strategy:
        - Which model to use
        - Relevant context from the Graph
        - Recommended reasoning effort
        """
        start_time = time.time()
        
        # 1. Retrieve context from Knowledge Graph
        stored_knowledge = self.graph.get_context(prompt)
        
        # 2. Heuristic Analysis
        word_count = len(prompt.split())
        is_complex = any(kw in prompt.lower() for kw in ["архитектура", "проектирование", "анализ", "strategy", "design"])
        needs_code = any(kw in prompt.lower() for kw in ["код", "python", "script", "refactor"])
        
        # 3. Model Routing
        if word_count < 10 and not is_complex and not needs_code:
            # Simple interaction -> Try local first if available
            strategy = {
                "model": LOCAL_MODEL,
                "backend": "ollama",
                "reasoning": "low",
                "route": "local_core"
            }
        elif is_complex or needs_code:
            strategy = {
                "model": DEEP_MODEL,
                "backend": "openrouter",
                "reasoning": "high",
                "route": "deep_reasoning"
            }
        else:
            strategy = {
                "model": FAST_MODEL,
                "backend": "openrouter",
                "reasoning": "medium",
                "route": "standard"
            }
            
        return {
            "strategy": strategy,
            "context": stored_knowledge,
            "analysis_ms": (time.time() - start_time) * 1000
        }

    async def learn(self, task: str, result: str):
        """
        Distills knowledge from a completed task and saves it to the Graph.
        Uses a fast cloud model to perform extraction.
        """
        try:
            extraction_prompt = f"""
            Analyze the task and result below. Extract key facts, strategies, or lessons learned.
            Format: A list of atomical knowledge points for a Knowledge Graph.
            
            Task: {task}
            Result: {result[:2000]} # Truncate for efficiency
            
            Return JSON: {{"knowledge": [{{"label": "concept", "summary": "brief explanation", "type": "fact/strategy"}}]}}
            """
            
            # Use a fast model for background learning
            messages = [{"role": "user", "content": extraction_prompt}]
            response, _ = self.llm.chat(messages, model=FAST_MODEL, max_tokens=1000)
            
            import json
            import re
            
            content = response.get("content", "")
            # Basic JSON extraction
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                for item in data.get("knowledge", []):
                    self.graph.add_node(
                        label=item["label"],
                        metadata={"summary": item["summary"], "type": item.get("type", "fact")}
                    )
                log.info(f"Brain learned {len(data.get('knowledge', []))} new items.")
                
        except Exception as e:
            log.warning(f"Brain learning failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "graph_nodes": len(self.graph.nodes),
            "graph_edges": len(self.edges),
            "local_model": LOCAL_MODEL
        }
