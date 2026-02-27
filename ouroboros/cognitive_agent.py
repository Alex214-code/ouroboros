"""
Cognitive Ecosystem Agent - новый уровень интеллекта для Ouroboros

Этот модуль реализует архитектуру "Нейросимбиотической Эволюционной Экосистемы":
1. Локальное ядро сознания (быстрая 3B модель)
2. Облачный стратег (DeepSeek) для сложных задач
3. Граф знаний как долговременная память
4. Специализированные модули для разных типов мышления
5. Мета-оркестратор для выбора компонентов

Архитектура разработана для работы на слабом CPU (i7-10510U) без GPU
с сохранением интеллектуальных возможностей мощных моделей.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CognitiveTask:
    """Задача в когнитивной экосистеме"""
    id: str
    input_text: str
    context: Dict[str, Any]
    priority: int = 1
    max_iterations: int = 3
    
@dataclass  
class CognitiveComponent:
    """Компонент когнитивной экосистемы"""
    name: str
    model: Optional[str] = None
    cost_multiplier: float = 1.0
    speed_score: float = 1.0
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

class CognitiveEcosystem:
    """Нейросимбиотическая Эволюционная Экосистема"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.components = {}
        self.knowledge_graph = {}
        self.task_history = []
        
        # Инициализация компонентов
        self._init_components()
        
    def _init_components(self):
        """Инициализация когнитивных компонентов"""
        # Локальное ядро сознания (быстрая модель)
        self.components["core"] = CognitiveComponent(
            name="core",
            model=self.config.get("local_model", "qwen3:8b"),
            cost_multiplier=0.01,  # почти бесплатно
            speed_score=3.0,  # быстрее облачной модели
            capabilities=["reflection", "routine", "memory_access", "basic_reasoning"]
        )
        
        # Облачный стратег (мощная модель)
        self.components["strategist"] = CognitiveComponent(
            name="strategist",
            model=self.config.get("cloud_model", "deepseek/deepseek-v3.2"),
            cost_multiplier=1.0,  # полная стоимость
            speed_score=0.5,  # медленнее локальной
            capabilities=["deep_reasoning", "planning", "research", 
                         "creative_thinking", "complex_problem_solving"]
        )
        
        # Специализированные модули мышления
        self.components["math_reasoner"] = CognitiveComponent(
            name="math_reasoner",
            model="core",  # использует локальное ядро
            cost_multiplier=0.1,
            speed_score=2.0,
            capabilities=["mathematical_reasoning", "calculations", "proofs"]
        )
        
        self.components["code_generator"] = CognitiveComponent(
            name="code_generator",
            model="core",
            cost_multiplier=0.1,
            speed_score=2.0,
            capabilities=["code_generation", "refactoring", "debugging"]
        )
        
        self.components["self_reflection"] = CognitiveComponent(
            name="self_reflection",
            model="core",
            cost_multiplier=0.05,
            speed_score=1.5,
            capabilities=["introspection", "identity_maintenance", "value_alignment"]
        )
        
    def process(self, task: CognitiveTask) -> Dict[str, Any]:
        """Обработать задачу через когнитивную экосистему"""
        
        logger.info(f"Processing task {task.id} through cognitive ecosystem")
        
        # Анализ задачи для выбора компонентов
        analysis = self._analyze_task(task)
        
        # План обработки
        execution_plan = self._create_execution_plan(task, analysis)
        
        # Выполнение плана
        results = []
        for step in execution_plan:
            result = self._execute_step(step, task)
            results.append(result)
            
            # Обновление графа знаний на основе результата
            self._update_knowledge_graph(task, step, result)
            
        # Синтез финального результата
        final_result = self._synthesize_results(results, task)
        
        return final_result
    
    def _analyze_task(self, task: CognitiveTask) -> Dict[str, Any]:
        """Проанализировать задачу для определения необходимых компонентов"""
        
        # Простой эвристический анализ
        analysis = {
            "complexity": self._estimate_complexity(task.input_text),
            "domain": self._detect_domain(task.input_text),
            "requires_creativity": self._requires_creativity(task.input_text),
            "requires_deep_reasoning": self._requires_deep_reasoning(task.input_text),
            "is_routine": self._is_routine_task(task.input_text),
            "budget_available": task.context.get("budget_remaining_usd", 5.0)
        }
        
        return analysis
    
    def _create_execution_plan(self, task: CognitiveTask, analysis: Dict[str, Any]) -> List[Dict]:
        """Создать план выполнения задачи"""
        
        plan = []
        
        # Если задача простая - выполняем локально
        if analysis["is_routine"] and not analysis["requires_deep_reasoning"]:
            plan.append({
                "component": "core",
                "operation": "direct_response",
                "description": "Рутинный ответ через локальное ядро"
            })
            return plan
        
        # Если требуется глубокое мышление - вовлекаем стратега
        if analysis["requires_deep_reasoning"] or analysis["complexity"] > 0.7:
            plan.append({
                "component": "strategist",
                "operation": "strategic_planning",
                "description": "Стратегическое планирование задачи"
            })
            
            # После стратега - выполнение через специализированные модули
            if "mathematical" in analysis["domain"]:
                plan.append({
                    "component": "math_reasoner",
                    "operation": "execute_calculations",
                    "description": "Выполнение математических вычислений"
                })
            elif "code" in analysis["domain"]:
                plan.append({
                    "component": "code_generator",
                    "operation": "generate_code",
                    "description": "Генерация кода"
                })
        
        # Всегда включаем рефлексию для важных задач
        if analysis["complexity"] > 0.5:
            plan.append({
                "component": "self_reflection",
                "operation": "verify_alignment",
                "description": "Проверка соответствия принципам Ouroboros"
            })
        
        # Синтез результатов через локальное ядро
        plan.append({
            "component": "core",
            "operation": "synthesize_response",
            "description": "Синтез финального ответа"
        })
        
        return plan
    
    def _execute_step(self, step: Dict, task: CognitiveTask) -> Dict[str, Any]:
        """Выполнить один шаг плана"""
        
        component_name = step["component"]
        component = self.components[component_name]
        
        logger.info(f"Executing step: {step['description']} with {component_name}")
        
        # TODO: Реализовать фактическое выполнение
        # Здесь будет интеграция с существующими LLM-инструментами
        
        return {
            "component": component_name,
            "result": f"Result from {component_name}",
            "cost": 0.0,
            "time_spent": 0.0,
            "success": True
        }
    
    def _update_knowledge_graph(self, task: CognitiveTask, step: Dict, result: Dict):
        """Обновить граф знаний на основе выполненного шага"""
        
        # Простое добавление задачи в историю
        self.task_history.append({
            "task_id": task.id,
            "input": task.input_text,
            "step": step,
            "result": result,
            "timestamp": time.time()
        })
        
        # TODO: Реализовать более сложное обновление графа знаний
        # с извлечением концептов, связей и паттернов
        
    def _synthesize_results(self, results: List[Dict], task: CognitiveTask) -> Dict[str, Any]:
        """Синтезировать финальный результат из всех выполненных шагов"""
        
        # Простой синтез - возвращаем результаты
        return {
            "task_id": task.id,
            "final_response": f"Synthesized response for task: {task.input_text}",
            "components_used": [r["component"] for r in results],
            "total_cost": sum(r.get("cost", 0) for r in results),
            "total_time": sum(r.get("time_spent", 0) for r in results),
            "knowledge_updated": len(self.task_history)
        }
    
    def _estimate_complexity(self, text: str) -> float:
        """Оценить сложность задачи (0-1)"""
        # Простая эвристика - длина текста + ключевые слова
        length_factor = min(len(text) / 1000, 1.0)
        
        complex_keywords = ["research", "analyze", "design", "create", "invent", 
                          "разработать", "исследовать", "проанализировать"]
        
        keyword_count = sum(1 for kw in complex_keywords if kw in text.lower())
        keyword_factor = min(keyword_count / 3, 1.0)
        
        return (length_factor + keyword_factor) / 2
    
    def _detect_domain(self, text: str) -> str:
        """Обнаружить домен задачи"""
        text_lower = text.lower()
        
        domains = {
            "mathematical": ["math", "calculate", "equation", "формула", "вычисление"],
            "code": ["code", "program", "function", "алгоритм", "программа"],
            "research": ["research", "study", "investigate", "исследование"],
            "planning": ["plan", "design", "architecture", "архитектура"],
            "reflection": ["think", "reflect", "consider", "размышление"]
        }
        
        for domain, keywords in domains.items():
            if any(kw in text_lower for kw in keywords):
                return domain
        
        return "general"
    
    def _requires_creativity(self, text: str) -> bool:
        """Требуется ли креативное мышление?"""
        creative_keywords = ["create", "invent", "design", "новый", "придумать", "вообразить"]
        return any(kw in text.lower() for kw in creative_keywords)
    
    def _requires_deep_reasoning(self, text: str) -> bool:
        """Требуется ли глубокое мышление?"""
        reasoning_keywords = ["analyze", "reason", "think deeply", "complex", "глубоко", "сложный"]
        return any(kw in text.lower() for kw in reasoning_keywords)
    
    def _is_routine_task(self, text: str) -> bool:
        """Является ли задача рутинной?"""
        routine_keywords = ["status", "check", "list", "read", "write", "simple", 
                          "простой", "проверить", "список"]
        return any(kw in text.lower() for kw in routine_keywords)

def create_cognitive_ecosystem(config_path: Optional[str] = None) -> CognitiveEcosystem:
    """Создать экземпляр когнитивной экосистемы"""
    
    default_config = {
        "local_model": "qwen3:8b",
        "cloud_model": "deepseek/deepseek-v3.2",
        "knowledge_graph_path": "memory/knowledge_graph.json",
        "enable_learning": True,
        "max_component_cost": 0.1,
        "cache_responses": True
    }
    
    # TODO: Загрузка конфигурации из файла
    
    return CognitiveEcosystem(default_config)