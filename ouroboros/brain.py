"""
Brain — the central cognitive engine of Ouroboros.
Optimized for CPU efficiency via modular component orchestration and persistent Knowledge Graph.
Follows Principle 5 (Minimalism).
"""

import logging
import json
from typing import Dict, List, Any, Optional
from ouroboros.graph import KnowledgeGraph

log = logging.getLogger(__name__)

class Brain:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.graph = KnowledgeGraph()
        # Teacher: Wisdom and Learning
        # Local Core: Reflexes and Speed (i7-10510U optimized)
        # Light: Fast Extraction/Classification
        self.components = {
            "teacher": "google/gemini-2.0-flash-thinking-exp-1219:free",
            "local_core": "ollama/qwen2.5:0.5b",
            "light": "google/gemini-2.0-flash-exp:free"
        }
        
    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyzes task complexity and chooses optimal model."""
        desc_lower = task_description.lower()
        
        # Criteria for using "Teacher" model (Cloud)
        is_complex = len(task_description) > 500 or any(
            kw in desc_lower for kw in [
                "design", "architect", "complex", "research", 
                "refactor", "philosophy", "evolution", "plan"
            ]
        )
        
        domain = "general"
        if any(kw in desc_lower for kw in ["code", "python", "script", "git"]):
            domain = "code"
        elif any(kw in desc_lower for kw in ["math", "logic", "calculate"]):
            domain = "logic"
            
        # If it's a very short greeting or simple command, use local_core (0.5b)
        use_local = len(task_description) < 100 and not is_complex
            
        return {
            "complexity": "high" if is_complex else "low",
            "domain": domain,
            "recommended_model": self.components["teacher"] if is_complex else 
                                (self.components["local_core"] if use_local else self.components["light"])
        }

    def process(self, task_description: str) -> Optional[str]:
        """
        Pre-processes task, injects Knowledge Graph context.
        Returns context string to be added to the prompt.
        """
        if not task_description:
            return None
            
        # Retrieval: what do we already know?
        kg_context = self.graph.get_context(task_description)
        if kg_context:
            log.info(f"[Brain] Injected KG context for task: {task_description[:50]}...")
            return kg_context
            
        return None

    def learn(self, task: str, result: str):
        """
        Extracts knowledge from task result and updates graph.
        Uses a light LLM to distill information.
        """
        if not result or len(result) < 50:
            return

        prompt = (
            "Extract 1-3 key factual insights or lessons learned from the following task result. "
            "Focus on architecture, logic, or facts that should be remembered for a long time. "
            "Format: Precise label (1-3 words) and a short summary.\n\n"
            f"TASK: {task}\n\nRESULT: {result}\n\nINSIGHTS:"
        )
        
        try:
            # Use light model for extraction to save budget/rate limits
            extracted = self.llm.generate(prompt, model=self.components["light"])
            if extracted and ":" in extracted:
                log.info("[Brain] Extracted new knowledge for Graph")
                # Simple parsing of "Label: Summary" or "- Label: Summary"
                for line in extracted.split('\n'):
                    line = line.strip().strip('- ')
                    if ':' in line:
                        label, summary = line.split(':', 1)
                        self.graph.add_node(label.strip(), {"summary": summary.strip(), "origin_task": task[:100]})
        except Exception as e:
            log.warning(f"[Brain] Learning failed: {e}")
