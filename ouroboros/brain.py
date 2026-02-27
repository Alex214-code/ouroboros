"""
Brain — the central cognitive engine of Ouroboros.
Optimized for CPU efficiency via modular component orchestration and persistent Knowledge Graph.
Follows Principle 5 (Minimalism).
"""

import logging
from typing import Dict, List, Any, Optional
from ouroboros.graph import KnowledgeGraph

log = logging.getLogger(__name__)

class Brain:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.graph = KnowledgeGraph()
        self.components = {
            "core": "stepfun/step-3.5-flash:free",
            "strategist": "deepseek/deepseek-chat",
            "light": "google/gemini-2.5-flash-lite-preview-02-10:free"
        }
        
    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyzes task complexity and chooses optimal model."""
        desc_lower = task_description.lower()
        is_complex = len(task_description) > 500 or any(kw in desc_lower for kw in ["design", "architect", "complex", "research", "refactor"])
        
        domain = "general"
        if any(kw in desc_lower for kw in ["code", "python", "script", "git"]):
            domain = "code"
        elif any(kw in desc_lower for kw in ["math", "logic", "calculate"]):
            domain = "logic"
            
        return {
            "complexity": "high" if is_complex else "low",
            "domain": domain,
            "recommended_model": self.components["strategist"] if is_complex else self.components["core"]
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
            "Format: Precise label (1-3 words) and a short summary.\n\n"
            f"TASK: {task}\n\nRESULT: {result}\n\nINSIGHTS:"
        )
        
        try:
            # Use light model for extraction to save budget/rate limits
            extracted = self.llm.generate(prompt, model=self.components["light"])
            if extracted and ":" in extracted:
                log.info("[Brain] Extracted new knowledge for Graph")
                # Simple parsing of "Label: Summary"
                for line in extracted.split('\n'):
                    if ':' in line:
                        label, summary = line.split(':', 1)
                        self.graph.add_node(label.strip().strip('- '), {"summary": summary.strip(), "origin_task": task[:100]})
        except Exception as e:
            log.warning(f"[Brain] Learning failed: {e}")
