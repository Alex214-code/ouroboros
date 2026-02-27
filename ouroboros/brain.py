"""
Ouroboros Brain — Cognitive Engine.
Manages reasoning orchestration, knowledge distillation, and local/cloud balance.
"""

import os
import json
import time
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from ouroboros.graph import KnowledgeGraph
from ouroboros.llm import LLMClient

log = logging.getLogger(__name__)

@dataclass
class CognitiveTask:
    id: str
    description: str
    domain: str = "general"
    complexity: str = "low"
    result: Any = None
    started_at: float = field(default_factory=time.time)

class Brain:
    def __init__(self, repo_dir: str, drive_root: str):
        self.repo_dir = repo_dir
        self.drive_root = drive_root
        
        # Consistent pathing across components
        self.graph_path = os.path.join(drive_root, "graph.json")
        self.graph = KnowledgeGraph(self.graph_path)
        
        self.llm = LLMClient()
        self.active_tasks: Dict[str, CognitiveTask] = {}
        self._load_state()

    def _load_state(self):
        state_path = os.path.join(self.drive_root, "brain_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {"total_tasks": 0, "knowledge_nodes": 0}
        else:
            self.state = {"total_tasks": 0, "knowledge_nodes": 0}

    def _save_state(self):
        state_path = os.path.join(self.drive_root, "brain_state.json")
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"Failed to save brain state: {e}")

    def process(self, prompt: str) -> Dict[str, Any]:
        """Strategic analysis before task execution."""
        # 1. Retrieve knowledge context
        context_data = self.graph.get_context(prompt)
        
        # 2. Analyze complexity (using simple heuristic for speed)
        # In v6.6.0 this will move to local LLM classifier
        complexity = "low"
        if len(prompt) > 150 or any(kw in prompt.lower() for kw in ["архит", "рефакт", "исследова", "создай", "сложн"]):
            complexity = "high"
            
        domain = "general"
        if any(kw in prompt.lower() for kw in ["код", "python", "скрипт", "ошибка", "репозиторий"]):
            domain = "code"
        elif any(kw in prompt.lower() for kw in ["поиск", "узнай", "интернет", "web"]):
            domain = "web"

        task_id = f"brain_{int(time.time())}"
        self.active_tasks[task_id] = CognitiveTask(id=task_id, description=prompt, domain=domain, complexity=complexity)
        
        # 3. Formulate strategy
        model = os.environ.get("OUROBOROS_MODEL")
        if complexity == "low" and not domain == "code":
            # Potential for local core switch (Ollama)
            model = os.environ.get("OUROBOROS_MODEL_LIGHT", "google/gemini-2.0-flash")

        return {
            "task_id": task_id,
            "strategy": "deep_reasoning" if complexity == "high" else "fast_reflection",
            "model": model,
            "context": context_data,
            "complexity": complexity,
            "domain": domain
        }

    def learn(self, prompt: str, result: str):
        """Knowledge Distillation Loop.
        
        Uses light cloud model to extract structured knowledge from the interaction.
        This knowledge is then stored in the localized Knowledge Graph.
        """
        log.info("Brain: Starting knowledge distillation...")
        
        distillation_prompt = f"""
        Analyze the following interaction and extract atomic knowledge units.
        Original Task: {prompt}
        Result: {result}
        
        Format your response as a JSON array of objects:
        [
          {{"node": "Concept Name", "type": "Concept/Fact/Strategy", "relation": "Related to", "target": "Existing Concept"}},
          ...
        ]
        
        Focus on technical patterns, architectural decisions, and unique insights. 
        Be concise. If no significant new knowledge is found, return an empty list [].
        ONLY return the JSON.
        """
        
        try:
            # Use light model for distillation to save budget
            model_light = os.environ.get("OUROBOROS_MODEL_LIGHT", "google/gemini-2.0-flash")
            msgs = [{"role": "system", "content": "You are Ouroboros Knowledge Distiller. Extract clean, modular facts for a Knowledge Graph."},
                    {"role": "user", "content": distillation_prompt}]
            
            # OpenAI API returns (message_dict, usage_dict)
            resp_msg, usage = self.llm.chat(msgs, model=model_light)
            content = resp_msg.get("content", "")
            
            # Robust JSON extraction
            # Remove markdown formatting if present
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            
            # Find the actual array
            match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if match:
                knowledge_units = json.loads(match.group(0))
                for unit in knowledge_units:
                    node_id = unit.get("node")
                    if node_id:
                        self.graph.add_node(node_id, {"type": unit.get("type"), "task_ref": prompt[:50]})
                        if unit.get("relation") and unit.get("target"):
                            self.graph.add_edge(node_id, unit.get("target"), unit.get("relationship", unit.get("relation")))
                
                num_units = len(knowledge_units)
                self.state["knowledge_nodes"] += num_units
                self.state["total_tasks"] += 1
                self._save_state()
                log.info(f"Brain: Successfully distilled {num_units} knowledge units.")
                return True
            else:
                log.info("Brain: No knowledge units found in response.")
                return False
        except Exception as e:
            log.warning(f"Brain: Knowledge distillation failed: {e}")
            return False
