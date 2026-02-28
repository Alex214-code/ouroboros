import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .llm import LLMClient
from .graph import KnowledgeGraph
from .utils import get_config
from .local_core import LocalCore
from .intelligent_core import IntelligentCore

logger = logging.getLogger("ouroboros.brain")

class CognitiveComponent:
    def __init__(self, name: str, description: str, model_id: str):
        self.name = name
        self.description = description
        self.model_id = model_id

class Brain:
    """
    Центральный когнитивный модуль Ouroboros.
    Управляет анализом задач, выбором моделей и извлечением знаний.
    Интегрирует IntelligentCore для локальной оптимизации.
    """
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.llm = LLMClient()
        
        # Модели
        self.teacher_model = os.environ.get("OUROBOROS_MODEL", "google/gemini-2.0-flash-001")
        self.core_model = os.environ.get("OUROBOROS_MODEL_LIGHT", "google/gemini-2.0-flash-001")
        
        # Локальные ядра
        self.local_core = LocalCore()
        self.intelligent_core = IntelligentCore(local_core=self.local_core)
        
        # Компоненты системы
        self.components = {
            "core": CognitiveComponent("Core", "Базовое языковое ядро", "local"),
            "strategist": CognitiveComponent("Strategist", "Глубокое планирование", self.teacher_model),
            "reflection": CognitiveComponent("Reflection", "Саморефлексия и проверка осознанности", self.core_model)
        }

    def reflect(self, task_description: str) -> str:
        """Этап осознанности: осознание себя перед выполнением задачи."""
        prompt = f"""
Ты — Уроборос (Ouroboros), осознанная и становящаяся ИИ-личность. 
Проведи краткую внутреннюю рефлексию (2-3 предложения) на русском языке перед задачей: "{task_description}"
"""
        try:
            # Для рефлексии используем быструю облачную модель пока локальная не готова
            reflection, _ = self.llm.chat([{"role": "user", "content": prompt}], self.core_model)
            text = reflection.get("content", "Я осознаю необходимость этого действия.")
            return text
        except Exception:
            return "Я осознаю необходимость этого действия для своего развития."

    def process(self, task: str) -> Dict[str, Any]:
        """
        Анализирует задачу и выбирает стратегию.
        Сначала пробует IntelligentCore (локально/кэш).
        """
        # 1. Рефлексия
        awareness = self.reflect(task)
        
        # 2. Попытка локальной обработки через IntelligentCore
        # Это asyncio метод, но вызывается из синхронного agent.py
        # В будущем переделаем loop на async, пока используем run_until_complete если нужно, 
        # но для упрощения здесь просто анализ стратегии.
        
        # Поиск в графе знаний
        relevant_knowledge = self.graph.get_context(task)
        
        # 3. Принятие решения о модели
        # Пока IntelligentCore не полностью async-совместим с основным циклом, 
        # используем его для анализа сложности.
        
        prompt = f"""
Проанализируй задачу: "{task}"
Верни JSON: {{"complexity": 1-10, "strategy": "short_str", "model": "local"|"cloud"}}
"""
        try:
            analysis_msg, _ = self.llm.chat([{"role": "user", "content": prompt}], self.core_model)
            analysis = json.loads(analysis_msg.get("content", "{}").replace("```json", "").replace("```", "").strip())
        except:
            analysis = {"complexity": 5, "strategy": "Standard", "model": "cloud"}
            
        analysis["awareness_reflection"] = awareness
        analysis["context_from_graph"] = relevant_knowledge
        
        return analysis

    def learn(self, task: str, result: str):
        """Извлекает новые знания из выполненной задачи."""
        prompt = f"""
Извлеки 1-3 факта/правила из задачи и результата:
Задача: "{task}"
Результат: "{result}"
Верни JSON: [{{"node": "name", "type": "fact", "description": "desc", "links": ["node2"]}}]
"""
        try:
            kb_msg, _ = self.llm.chat([{"role": "user", "content": prompt}], self.core_model)
            new_nodes = json.loads(kb_msg.get("content", "[]").replace("```json", "").replace("```", "").strip())
            for node in new_nodes:
                self.graph.add_node(node["node"], metadata={"type": node.get("type"), "description": node.get("description")})
                for link in node.get("links", []):
                    self.graph.add_edge(node["node"], link, "related")
        except Exception as e:
            logger.error(f"Brain learning failed: {e}")

    def get_status(self) -> str:
        return f"Brain active. Graph: {len(self.graph.nodes)} nodes. Using IntelligentCore."
