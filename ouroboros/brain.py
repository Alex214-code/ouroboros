"""
Brain — the central cognitive engine of Ouroboros.
Optimized for CPU efficiency via modular component orchestration and persistent Knowledge Graph.
Follows Principle 5 (Minimalism).
"""

import time
import re
from typing import Dict, List, Any, Optional
from ouroboros.graph import KnowledgeGraph

class Brain:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.graph = KnowledgeGraph()
        self.components = {
            "core": "stepfun/step-3.5-flash:free",
            "strategist": "deepseek/deepseek-chat", # Default paid strategist
            "vision": "qwen/qwen3-vl-235b-a22b-thinking"
        }
        
    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyzes task complexity and chooses optimal model."""
        # Simple heuristic analysis
        is_complex = len(task_description) > 500 or any(kw in task_description.lower() for kw in ["design", "architect", "complex", "research"])
        domain = "general"
        if any(kw in task_description.lower() for kw in ["code", "python", "script"]):
            domain = "code"
        elif any(kw in task_description.lower() for kw in ["math", "logic"]):
            domain = "logic"
            
        return {
            "complexity": "high" if is_complex else "low",
            "domain": domain,
            "recommended_model": self.components["strategist"] if is_complex else self.components["core"]
        }

    def process(self, task_description: str) -> Optional[str]:
        """
        Main entry point for task processing.
        Returns optimized text if brain decides to handle it, else None.
        """
        if not task_description:
            return None
            
        analysis = self.analyze_task(task_description)
        
        # Knowledge Graph Integration
        kg_context = self.graph.get_context(task_description)
        
        # Decision: delegate to paid model OR use local core
        # For now, let's keep it simple: if complexity is high, we let the normal loop handle it (with paid model)
        # BUT we can inject KG context into the loop. 
        # Actually, let's return None to allow the normal loop to proceed, 
        # but we've already prepared everything.
        
        # Evolutionary step: can we handle simple tasks entirely via brain.process?
        if analysis["complexity"] == "low" and analysis["domain"] == "general":
            # Test run with core model
            print(f"[Brain] Handling simple task locally with {self.components['core']}")
            # We would call LLM here directly and return the text
            # return self.llm.generate(task_description, model=self.components["core"])
            pass

        return None # Fallback to normal llm_loop for now

    def learn(self, task: str, result: str):
        """Extracts knowledge from task result and updates graph."""
        # TODO: Implement NLP extraction of entities and relationships
        # For now, just record the task summary
        self.graph.add_node(task[:50], {"summary": result[:200], "full_task": task})
