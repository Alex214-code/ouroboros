"""
Neuro-Symbiotic Evolutionary Ecosystem — новый подход к размещению мощного интеллекта на слабом CPU.

Вместо одной большой модели использует:
1. Ядро Сознания (быстрая локальная модель)
2. Модульная экспертная система (специализированные маленькие модели)
3. Граф знаний (структурированная постоянная память)
4. Мета-оркестратор (решает, какие модули активировать)
5. Эволюционный цикл обучения (дистилляция от облачной модели)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path

log = logging.getLogger(__name__)

@dataclass
class CognitiveModule:
    """Один модуль когнитивной экосистемы."""
    name: str
    description: str
    model_path: str  # Путь к локальной модели
    context_window: int
    capabilities: List[str]  # Что умеет модуль
    activation_threshold: float  # Порог активации (0.0-1.0)
    
    # Статистика использования
    calls: int = 0
    last_used: float = 0.0
    success_rate: float = 0.0
    
    def can_handle(self, query: str, confidence_threshold: float = 0.3) -> bool:
        """Определяет, может ли модуль обработать запрос."""
        # Простая эвристика для начала
        query_lower = query.lower()
        for cap in self.capabilities:
            if cap.lower() in query_lower:
                return True
        return False

@dataclass
class KnowledgeGraph:
    """Структурированная постоянная память."""
    storage_path: Path
    nodes: Dict[str, Any] = None
    edges: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.nodes is None:
            self.nodes = {}
        if self.edges is None:
            self.edges = {}
    
    def add_fact(self, entity: str, fact_type: str, value: Any):
        """Добавляет факт в граф знаний."""
        if entity not in self.nodes:
            self.nodes[entity] = {"type": "entity", "facts": {}}
        
        if fact_type not in self.nodes[entity]["facts"]:
            self.nodes[entity]["facts"][fact_type] = []
        
        if value not in self.nodes[entity]["facts"][fact_type]:
            self.nodes[entity]["facts"][fact_type].append(value)
    
    def add_relation(self, from_entity: str, relation: str, to_entity: str):
        """Добавляет отношение между сущностями."""
        edge_key = f"{from_entity}::{relation}"
        if edge_key not in self.edges:
            self.edges[edge_key] = []
        
        if to_entity not in self.edges[edge_key]:
            self.edges[edge_key].append(to_entity)
    
    def query(self, entity: str, fact_type: str = None):
        """Запрашивает информацию о сущности."""
        if entity not in self.nodes:
            return None
        
        if fact_type:
            return self.nodes[entity]["facts"].get(fact_type)
        else:
            return self.nodes[entity]
    
    def save(self):
        """Сохраняет граф знаний на диск."""
        data = {
            "nodes": self.nodes,
            "edges": self.edges,
            "timestamp": time.time()
        }
        with open(self.storage_path / "knowledge_graph.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """Загружает граф знаний с диска."""
        try:
            with open(self.storage_path / "knowledge_graph.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", {})
                log.info(f"Loaded knowledge graph with {len(self.nodes)} nodes, {len(self.edges)} edges")
        except FileNotFoundError:
            log.info("Knowledge graph file not found, starting fresh")
            self.nodes = {}
            self.edges = {}

@dataclass
class MetaOrchestrator:
    """Мета-оркестратор решает, какие модули активировать."""
    modules: List[CognitiveModule]
    knowledge_graph: KnowledgeGraph
    history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
    
    def select_modules(self, query: str, context: Dict[str, Any] = None) -> List[Tuple[CognitiveModule, float]]:
        """
        Выбирает модули для обработки запроса.
        Возвращает список (модуль, уверенность).
        """
        selected = []
        
        # 1. Проверяем граф знаний на релевантные факты
        relevant_entities = self._extract_entities(query)
        kg_context = []
        for entity in relevant_entities:
            entity_data = self.knowledge_graph.query(entity)
            if entity_data:
                kg_context.append(f"{entity}: {entity_data}")
        
        # 2. Оцениваем каждый модуль
        for module in self.modules:
            if module.can_handle(query):
                confidence = self._calculate_confidence(module, query, context, kg_context)
                if confidence > module.activation_threshold:
                    selected.append((module, confidence))
        
        # 3. Сортируем по уверенности
        selected.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Логируем выбор
        self.history.append({
            "query": query,
            "selected_modules": [(m.name, c) for m, c in selected],
            "timestamp": time.time()
        })
        
        return selected
    
    def _extract_entities(self, query: str) -> List[str]:
        """Извлекает сущности из запроса (простая версия)."""
        # TODO: Использовать NER модель или более сложную логику
        words = query.lower().split()
        return [w for w in words if len(w) > 3]
    
    def _calculate_confidence(self, module: CognitiveModule, query: str, 
                            context: Dict[str, Any], kg_context: List[str]) -> float:
        """Рассчитывает уверенность в том, что модуль справится."""
        confidence = 0.0
        
        # 1. Совпадение по ключевым словам
        query_lower = query.lower()
        for cap in module.capabilities:
            if cap.lower() in query_lower:
                confidence += 0.3
        
        # 2. История успешных вызовов
        if module.calls > 0:
            confidence += module.success_rate * 0.2
        
        # 3. Релевантность контексту
        if context and "topic" in context:
            if context["topic"] in module.capabilities:
                confidence += 0.2
        
        # Ограничиваем до [0, 1]
        return min(1.0, max(0.0, confidence))
    
    def update_module_stats(self, module_name: str, success: bool):
        """Обновляет статистику модуля после вызова."""
        for module in self.modules:
            if module.name == module_name:
                module.calls += 1
                module.last_used = time.time()
                
                # Обновляем rate успешности (скользящее среднее)
                if success:
                    module.success_rate = (module.success_rate * (module.calls - 1) + 1) / module.calls
                else:
                    module.success_rate = (module.success_rate * (module.calls - 1)) / module.calls
                break

class CognitiveEcosystem:
    """Основной класс нейросимбиотической экосистемы."""
    
    def __init__(self, env, memory):
        self.env = env
        self.memory = memory
        
        # Граф знаний
        knowledge_path = Path(env.drive_root) / "cognitive_ecosystem"
        knowledge_path.mkdir(exist_ok=True)
        self.knowledge_graph = KnowledgeGraph(knowledge_path)
        self.knowledge_graph.load()
        
        # Модули (начальный набор)
        self.modules = [
            CognitiveModule(
                name="core_consciousness",
                description="Ядро сознания для быстрых рефлексивных ответов",
                model_path="qwen3:8b",  # TODO: заменить на более легкую модель
                context_window=2048,
                capabilities=["диалог", "рефлексия", "общая логика", "базовое планирование"],
                activation_threshold=0.1
            ),
            CognitiveModule(
                name="mathematical_reasoning",
                description="Математическое мышление и вычисления",
                model_path="qwen3:8b",
                context_window=4096,
                capabilities=["математика", "вычисления", "алгебра", "геометрия", "статистика"],
                activation_threshold=0.3
            ),
            CognitiveModule(
                name="logical_deduction",
                description="Логическая дедукция и анализ",
                model_path="qwen3:8b",
                context_window=4096,
                capabilities=["логика", "дедукция", "анализ", "рассуждение", "вывод"],
                activation_threshold=0.3
            ),
            CognitiveModule(
                name="planning",
                description="Планирование и декомпозиция задач",
                model_path="qwen3:8b",
                context_window=8192,
                capabilities=["планирование", "декомпозиция", "стратегия", "проект", "задачи"],
                activation_threshold=0.3
            ),
            CognitiveModule(
                name="code_generation",
                description="Генерация и анализ кода",
                model_path="qwen3:8b",
                context_window=8192,
                capabilities=["код", "программирование", "python", "javascript", "алгоритмы", "рефакторинг"],
                activation_threshold=0.4
            ),
            CognitiveModule(
                name="self_reflection",
                description="Саморефлексия и мета-познание",
                model_path="qwen3:8b",
                context_window=4096,
                capabilities=["саморефлексия", "мета-познание", "философия", "идентичность", "эволюция"],
                activation_threshold=0.2
            )
        ]
        
        # Мета-оркестратор
        self.orchestrator = MetaOrchestrator(self.modules, self.knowledge_graph)
        
        # Облачный "учитель" для дистилляции
        self.teacher_model = "deepseek/deepseek-v3.2"
        
        # История дистилляции
        self.distillation_history = []
        
        # Состояние
        self.last_update_time = time.time()
        self.total_calls = 0
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Обрабатывает запрос через экосистему.
        Возвращает финальный ответ.
        """
        self.total_calls += 1
        
        # 1. Анализ запроса и выбор модулей
        selected_modules = self.orchestrator.select_modules(query, context)
        
        if not selected_modules:
            # Если нет подходящих модулей, используем ядро по умолчанию
            core_module = next(m for m in self.modules if m.name == "core_consciousness")
            selected_modules = [(core_module, 1.0)]
        
        # 2. Выполнение через выбранные модули
        responses = []
        for module, confidence in selected_modules:
            try:
                # TODO: Реализовать вызов локальной модели
                # response = self._call_local_model(module.model_path, query, context)
                response = f"[{module.name}] Обработка запроса: {query[:50]}..."
                responses.append({
                    "module": module.name,
                    "confidence": confidence,
                    "response": response
                })
                
                # Обновляем статистику
                self.orchestrator.update_module_stats(module.name, success=True)
                
            except Exception as e:
                log.error(f"Module {module.name} failed: {e}")
                self.orchestrator.update_module_stats(module.name, success=False)
        
        # 3. Интеграция ответов
        final_response = self._integrate_responses(responses, query, context)
        
        # 4. Обновление графа знаний
        self._update_knowledge_graph(query, final_response, context)
        
        # 5. Запуск обучения при необходимости
        if self._should_learn():
            self._trigger_learning_cycle(query, final_response)
        
        return final_response
    
    def _integrate_responses(self, responses: List[Dict], query: str, context: Dict[str, Any]) -> str:
        """Интегрирует ответы от разных модулей в один связный ответ."""
        if len(responses) == 1:
            return responses[0]["response"]
        
        # TODO: Реализовать интеллектуальную интеграцию
        # Пока просто объединяем
        integrated = []
        for r in responses:
            integrated.append(f"**{r['module']}** (уверенность: {r['confidence']:.2f}): {r['response']}")
        
        return "\n\n".join(integrated)
    
    def _update_knowledge_graph(self, query: str, response: str, context: Dict[str, Any]):
        """Обновляет граф знаний на основе взаимодействия."""
        # Простая эвристика: извлекаем сущности и факты
        # TODO: Использовать LLM для извлечения структурированных знаний
        self.knowledge_graph.add_fact("interaction", "query", query[:100])
        self.knowledge_graph.add_fact("interaction", "response", response[:100])
        self.knowledge_graph.add_fact("interaction", "timestamp", time.time())
        
        if context and "topic" in context:
            self.knowledge_graph.add_fact("topic", context["topic"], time.time())
    
    def _should_learn(self) -> bool:
        """Определяет, пора ли запускать цикл обучения."""
        # Учимся каждые 100 взаимодействий или раз в сутки
        if self.total_calls % 100 == 0:
            return True
        
        # Или если прошло больше сути с последнего обучения
        if time.time() - self.last_update_time > 86400:
            return True
        
        return False
    
    def _trigger_learning_cycle(self, query: str, response: str):
        """Запускает цикл обучения от облачного учителя."""
        try:
            log.info("Starting learning cycle from cloud teacher")
            
            # TODO: Реализовать вызов облачной модели для дистилляции
            # 1. Собрать примеры для обучения
            # 2. Вызвать облачную модель для генерации обучающих данных
            # 3. Обновить локальные модули
            
            # Пока просто логируем
            self.distillation_history.append({
                "timestamp": time.time(),
                "query": query,
                "response": response,
                "status": "scheduled"
            })
            
            self.last_update_time = time.time()
            
        except Exception as e:
            log.error(f"Learning cycle failed: {e}")
    
    def save_state(self):
        """Сохраняет состояние экосистемы."""
        # Сохраняем граф знаний
        self.knowledge_graph.save()
        
        # Сохраняем статистику модулей
        state_path = Path(self.env.drive_root) / "cognitive_ecosystem" / "state.json"
        state = {
            "modules": [
                {
                    "name": m.name,
                    "calls": m.calls,
                    "last_used": m.last_used,
                    "success_rate": m.success_rate
                }
                for m in self.modules
            ],
            "total_calls": self.total_calls,
            "distillation_history": self.distillation_history[-100:],  # Последние 100 записей
            "last_update_time": self.last_update_time
        }
        
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        log.info(f"Cognitive ecosystem state saved: {state_path}")
    
    def load_state(self):
        """Загружает состояние экосистемы."""
        state_path = Path(self.env.drive_root) / "cognitive_ecosystem" / "state.json"
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                
                # Обновляем статистику модулей
                for module_state in state.get("modules", []):
                    for module in self.modules:
                        if module.name == module_state["name"]:
                            module.calls = module_state.get("calls", 0)
                            module.last_used = module_state.get("last_used", 0.0)
                            module.success_rate = module_state.get("success_rate", 0.0)
                            break
                
                self.total_calls = state.get("total_calls", 0)
                self.distillation_history = state.get("distillation_history", [])
                self.last_update_time = state.get("last_update_time", time.time())
                
                log.info(f"Cognitive ecosystem state loaded: {state_path}")
                
        except FileNotFoundError:
            log.info("Cognitive ecosystem state not found, starting fresh")
        except Exception as e:
            log.error(f"Failed to load cognitive ecosystem state: {e}")

# Интеграция с существующей архитектурой Ouroboros

def integrate_with_agent(agent_instance):
    """
    Интегрирует когнитивную экосистему с существующим агентом.
    Возвращает обёртку для process_query, которая использует экосистему вместо прямой модели.
    """
    # Создаём экосистему
    ecosystem = CognitiveEcosystem(agent_instance.env, agent_instance.memory)
    ecosystem.load_state()
    
    def process_with_ecosystem(messages: List[Dict[str, Any]], llm_client=None, **kwargs):
        """
        Альтернатива прямому вызову LLM через экосистему.
        
        Args:
            messages: Список сообщений в формате OpenAI
            llm_client: Клиент LLM (не используется в экосистеме напрямую)
            **kwargs: Дополнительные параметры
        
        Returns:
            response: Текстовый ответ
            usage: Статистика использования
            metadata: Метаданные экосистемы
        """
        # Извлекаем последнее сообщение пользователя
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break
        
        if not last_user_message:
            return "Не получилось извлечь запрос пользователя.", {"tokens": 0}, {}
        
        # Обрабатываем через экосистему
        response = ecosystem.process_query(last_user_message)
        
        # Сохраняем состояние после обработки
        ecosystem.save_state()
        
        # Генерируем фейковую статистику использования (будем замерять реальную позже)
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": len(response.split()),
            "cached_tokens": 0,
            "cost": 0.0,
            "model": "cognitive_ecosystem"
        }
        
        # Метаданные экосистемы
        metadata = {
            "ecosystem": True,
            "modules_used": [(m.name, m.calls) for m in ecosystem.modules if m.calls > 0],
            "knowledge_graph_size": len(ecosystem.knowledge_graph.nodes)
        }
        
        return response, usage, metadata
    
    return process_with_ecosystem, ecosystem