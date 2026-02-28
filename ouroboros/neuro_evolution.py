"""
Нейросимбиотическая эволюционная экосистема - ядро новой архитектуры.
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

log = logging.getLogger(__name__)

class ComponentType(Enum):
    LOCAL_FAST = "local_fast"
    CLOUD_MEDIUM = "cloud_medium"
    CLOUD_STRATEGIC = "cloud_strategic"
    RESEARCH_CHAIN = "research_chain"
    MEMORY_CACHE = "memory_cache"
    LEARNING_OPTIMIZER = "learning_optimizer"

@dataclass
class Component:
    id: str
    type: ComponentType
    name: str
    description: str
    cost_per_token: float
    speed_ms: int
    is_available: bool = True

@dataclass
class InferenceRequest:
    """Запрос на инференс к компоненту экосистемы."""
    prompt: str
    component_id: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    context: Optional[Dict[str, Any]] = None

@dataclass
class InferenceResult:
    """Результат инференса от компонента."""
    text: str
    component_id: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class NeuroSymbioticEcosystem:
    def __init__(self):
        self.components = self._init_components()
        self.task_history = []
        
    def _init_components(self) -> Dict[str, Component]:
        return {
            "local_fast": Component(
                id="local_fast",
                type=ComponentType.LOCAL_FAST,
                name="Локальный быстрый процессор",
                description="Быстрая обработка простых задач на CPU",
                cost_per_token=0.000001,
                speed_ms=50
            ),
            "cloud_medium": Component(
                id="cloud_medium",
                type=ComponentType.CLOUD_MEDIUM,
                name="Облачная модель среднего уровня",
                description="Баланс стоимости и качества",
                cost_per_token=0.00001,
                speed_ms=300
            ),
            "cloud_strategic": Component(
                id="cloud_strategic",
                type=ComponentType.CLOUD_STRATEGIC,
                name="Стратегическая облачная модель",
                description="Для сложных рассуждений и планирования",
                cost_per_token=0.00005,
                speed_ms=1000
            ),
            "research_chain": Component(
                id="research_chain",
                type=ComponentType.RESEARCH_CHAIN,
                name="Исследовательская цепочка",
                description="Многошаговая обработка с веб-поиском",
                cost_per_token=0.0001,
                speed_ms=5000
            )
        }
    
    async def analyze_task(self, prompt: str) -> Dict[str, Any]:
        word_count = len(prompt.split())
        
        if word_count < 10:
            complexity = "simple"
        elif word_count < 50:
            complexity = "moderate"
        elif word_count < 200:
            complexity = "complex"
        else:
            complexity = "strategic"
        
        return {
            "complexity": complexity,
            "estimated_tokens": word_count * 3,
            "word_count": word_count
        }
    
    async def select_component(self, analysis: Dict[str, Any]) -> str:
        complexity = analysis["complexity"]
        
        if complexity == "simple":
            return "local_fast"
        elif complexity == "moderate":
            return "cloud_medium"
        elif complexity == "complex":
            return "cloud_strategic"
        else:
            return "research_chain"
    
    async def route_task(self, prompt: str) -> Dict[str, Any]:
        analysis = await self.analyze_task(prompt)
        component_id = await self.select_component(analysis)
        component = self.components[component_id]
        
        await asyncio.sleep(component.speed_ms / 1000)
        
        response = f"[{component.name}] Обработал задачу сложности '{analysis['complexity']}': {prompt[:50]}..."
        
        self.task_history.append({
            "component": component_id,
            "complexity": analysis["complexity"],
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return {
            "component_id": component_id,
            "component_name": component.name,
            "response": response,
            "analysis": analysis
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tasks": len(self.task_history),
            "components": list(self.components.keys()),
            "recent_tasks": self.task_history[-10:] if self.task_history else []
        }
    
    async def shutdown(self):
        pass

def get_ecosystem() -> NeuroSymbioticEcosystem:
    global _ecosystem_instance
    if _ecosystem_instance is None:
        _ecosystem_instance = NeuroSymbioticEcosystem()
    return _ecosystem_instance

_ecosystem_instance = None
