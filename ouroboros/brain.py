"""
Ouroboros Brain — Orchestrator for complex cognitive tasks.
Analyzes complexity and chooses components based on task domain.
Integrates cloud teacher (Gemini) and local core (Ollama).
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from ouroboros.graph import KnowledgeGraph

@dataclass
class CognitiveTask:
    id: str
    description: str
    domain: str = "general"
    complexity: str = "low"
    priority: int = 1
    status: str = "pending"
    result: Any = None
    steps: List[Dict[str, Any]] = field(default_factory=list)

class Brain:
    def __init__(self, drive_root: str):
        self.drive_root = drive_root
        # Synchronized path to data/local_state/graph.json
        self.graph_path = os.path.join(drive_root, "graph.json")
        self.graph = KnowledgeGraph(self.graph_path)
        self.active_tasks = []
        self._load_state()

    def _load_state(self):
        state_path = os.path.join(self.drive_root, "brain_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}
        else:
            self.state = {"total_tasks": 0, "cycles_completed": 0}

    def _save_state(self):
        state_path = os.path.join(self.drive_root, "brain_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def process(self, task_description: str) -> Dict[str, Any]:
        """Analyzes task and plans execution path."""
        # Query graph for context
        knowledge_context = self.graph.get_context(task_description)
        
        # Simple rule-based analysis (to be replaced with local LLM classifier)
        complexity = "low"
        if len(task_description) > 50 or any(kw in task_description.lower() for kw in ["архит", "рефакт", "исслед", "созд"]):
            complexity = "high"
            
        domain = "general"
        if any(kw in task_description.lower() for kw in ["код", "python", "скрипт", "ошибка"]):
            domain = "code"
        elif any(kw in task_description.lower() for kw in ["поиск", "узнай", "интернет"]):
            domain = "web"

        task = CognitiveTask(
            id=f"T-{int(time.time())}",
            description=task_description,
            domain=domain,
            complexity=complexity
        )
        
        self.active_tasks.append(task)
        
        result = {
            "task_id": task.id,
            "complexity": complexity,
            "domain": domain,
            "knowledge_context": knowledge_context,
            "strategy": "delegation" if complexity == "high" else "local_process"
        }
        
        return result

    def learn(self, task_id: str, result_summary: str):
        """Extracts knowledge from task result and updates graph."""
        # In the future, this will use Gemini Flash to distill experience.
        # For now, it adds a summary node to the graph.
        task = next((t for t in self.active_tasks if t.id == task_id), None)
        if task:
            self.graph.add_node(f"Task: {task.id}", {
                "description": task.description,
                "summary": result_summary,
                "domain": task.domain
            })
            self.graph.add_edge("Experience", f"Task: {task.id}", "learned_from")
            self._save_state()
            return True
        return False
