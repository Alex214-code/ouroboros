"""
Нейросимбиотический агент Ouroboros — следующее поколение архитектуры.

Сочетает в себе:
- Нейросимбиотическую экосистему (распределённую обработку)
- Совместимость с существующим интерфейсом Ouroboros
- Автоматическую маршрутизацию задач
- Постепенный переход от старой архитектуры
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .integrated_agent import IntegratedAgent, TaskComplexity
from .neuro_evolution import NeuroSymbioticEcosystem, ComponentType, get_ecosystem
from .tools.registry import ToolRegistry
from .llm import LLMClient
from .memory import Memory
from .utils import (
    utc_now_iso, read_text, append_jsonl,
    safe_relpath, truncate_for_log
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuroEnv:
    """Окружение для нейросимбиотического агента."""
    repo_dir: Path
    drive_root: Path
    branch_dev: str = "ouroboros"
    
    def repo_path(self, rel: str) -> Path:
        return (self.repo_dir / safe_relpath(rel)).resolve()
    
    def drive_path(self, rel: str) -> Path:
        return (self.drive_root / safe_relpath(rel)).resolve()


class NeuroAgent:
    """
    Нейросимбиотический агент — основная точка входа для новой архитектуры.
    
    Особенности:
    1. Использует IntegratedAgent как движок обработки
    2. Сохраняет совместимость с существующим интерфейсом
    3. Автоматически маршрутизирует задачи по сложности
    4. Поддерживает плавный переход от старой архитектуры
    """
    
    def __init__(self, env: NeuroEnv):
        self.env = env
        self.integrated_agent = IntegratedAgent()
        self.ecosystem = get_ecosystem()
        
        # Сохранение совместимости
        self.llm = LLMClient()
        self.tools = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
        self.memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
        
        # Статистика
        self.stats = {
            "start_time": time.time(),
            "total_tasks": 0,
            "tasks_by_complexity": {},
            "tasks_by_component": {},
            "total_cost_usd": 0.0,
            "avg_response_time_ms": 0.0,
            "fallback_to_legacy_count": 0
        }
        
        # Настройки
        self.config = self._load_config()
        self._init_neuro_tools()
        
        # Логирование инициализации
        self._log_init()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузить конфигурацию нейросимбиотического агента."""
        config_path = self.env.drive_path("config/neuro_agent.json")
        default_config = {
            "enabled": True,
            "auto_route_tasks": True,
            "complexity_thresholds": {
                "simple": {"max_tokens": 100, "temperature": 0.7},
                "moderate": {"max_tokens": 300, "temperature": 0.7},
                "complex": {"max_tokens": 1000, "temperature": 0.5},
                "strategic": {"max_tokens": 2000, "temperature": 0.3}
            },
            "fallback_enabled": True,
            "performance_optimization_interval": 100,  # задач между оптимизациями
            "enable_learning": True,
            "cache_task_analysis": True
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Объединить с дефолтными значениями
                    default_config.update(loaded)
                    return default_config
            except Exception as e:
                log.error(f"Ошибка загрузки конфигурации: {e}")
        
        return default_config
    
    def _init_neuro_tools(self):
        """Инициализировать нейросимбиотические инструменты."""
        try:
            # Импортировать нейросимбиотические инструменты
            from .tools.neuro_symbiotic_tool import get_tools as get_neuro_tools
            
            # Получить инструменты
            neuro_tools = get_neuro_tools()
            
            # Добавить их в реестр (если возможно)
            # В текущей архитектуре инструменты регистрируются по-другому,
            # но мы сохраняем ссылку для использования
            self.neuro_tools_available = True
            self.neuro_tools = neuro_tools
            
            log.info(f"Загружено {len(neuro_tools)} нейросимбиотических инструментов")
            
        except Exception as e:
            log.warning(f"Нейросимбиотические инструменты не загружены: {e}")
            self.neuro_tools_available = False
            self.neuro_tools = []
    
    def _log_init(self):
        """Записать событие инициализации."""
        init_event = {
            "ts": utc_now_iso(),
            "type": "neuro_agent_init",
            "config": {
                "enabled": self.config["enabled"],
                "auto_route_tasks": self.config["auto_route_tasks"],
                "neuro_tools_available": self.neuro_tools_available
            },
            "ecosystem_components": list(self.ecosystem.components.keys()),
            "stats": self.stats
        }
        
        append_jsonl(self.env.drive_path("logs") / "events.jsonl", init_event)
        log.info("Нейросимбиотический агент инициализирован")
    
    async def process(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Any]] = None,
        use_neuro_architecture: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Основной метод обработки запроса.
        
        Args:
            prompt: Текст запроса
            context: Дополнительный контекст
            tools: Список доступных инструментов
            use_neuro_architecture: Принудительно использовать нейросимбиотическую архитектуру
            
        Returns:
            Результат обработки
        """
        start_time = time.time()
        self.stats["total_tasks"] += 1
        
        # Определить, использовать ли нейросимбиотическую архитектуру
        if use_neuro_architecture is None:
            use_neuro_architecture = self.config["enabled"] and self.config["auto_route_tasks"]
        
        task_id = f"neuro_{int(time.time() * 1000)}"
        
        try:
            if use_neuro_architecture:
                result = await self._process_neuro(prompt, context, tools, task_id)
            else:
                result = await self._process_legacy(prompt, context, tools, task_id)
            
            # Рассчитать время ответа
            response_time_ms = (time.time() - start_time) * 1000
            
            # Обновить статистику
            self._update_stats(result, response_time_ms)
            
            # Записать результат
            self._log_task_result(task_id, prompt, result, response_time_ms)
            
            # Оптимизация производительности (периодически)
            if self.stats["total_tasks"] % self.config["performance_optimization_interval"] == 0:
                await self.optimize_performance()
            
            return result
            
        except Exception as e:
            log.error(f"Ошибка обработки задачи {task_id}: {e}", exc_info=True)
            
            # Fallback на старую архитектуру
            if use_neuro_architecture and self.config["fallback_enabled"]:
                self.stats["fallback_to_legacy_count"] += 1
                log.warning(f"Fallback к старой архитектуре для задачи {task_id}")
                
                fallback_result = await self._process_legacy(
                    f"[Ошибка нейроархитектуры: {str(e)[:100]}]\nЗадача: {prompt}",
                    context,
                    tools,
                    f"{task_id}_fallback"
                )
                
                fallback_result["metadata"]["neuro_error"] = str(e)
                fallback_result["metadata"]["fallback_used"] = True
                
                return fallback_result
            
            # Если fallback отключен или тоже не сработал
            return {
                "response": f"Критическая ошибка обработки: {e}",
                "metadata": {
                    "error": str(e),
                    "task_id": task_id,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "architecture": "error"
                }
            }
    
    async def _process_neuro(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        tools: Optional[List[Any]],
        task_id: str
    ) -> Dict[str, Any]:
        """Обработать задачу через нейросимбиотическую архитектуру."""
        # Преобразовать контекст в строку JSON для передачи в инструмент
        context_str = None
        if context:
            try:
                context_str = json.dumps(context, ensure_ascii=False)
            except:
                context_str = str(context)
        
        # Использовать нейросимбиотический инструмент для обработки
        if self.neuro_tools_available:
            try:
                # Ищем инструмент neuro_symbiotic_process
                for tool in self.neuro_tools:
                    if hasattr(tool, '__name__') and tool.__name__ == 'neuro_symbiotic_process':
                        result = await tool(
                            prompt=prompt,
                            context=context_str,
                            max_tokens=self._get_max_tokens(prompt),
                            temperature=self._get_temperature(prompt)
                        )
                        
                        # Добавить метаданные
                        result["metadata"]["architecture"] = "neuro_symbiotic"
                        result["metadata"]["task_id"] = task_id
                        result["metadata"]["tools_available"] = bool(tools)
                        
                        return result
            except Exception as e:
                log.error(f"Ошибка нейросимбиотического инструмента: {e}")
                # Продолжить с integrated_agent
        
        # Fallback: использовать integrated_agent напрямую
        result = await self.integrated_agent.process_task(
            prompt=prompt,
            context=context,
            tools=tools or []
        )
        
        # Форматировать результат
        formatted_result = {
            "response": result["response"],
            "metadata": {
                "architecture": "neuro_integrated",
                "task_id": task_id,
                "analysis": result["analysis"],
                "metrics": result["metrics"],
                "tool_calls": result["tool_calls"],
                "needs_follow_up": result["needs_follow_up"]
            }
        }
        
        return formatted_result
    
    async def _process_legacy(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        tools: Optional[List[Any]],
        task_id: str
    ) -> Dict[str, Any]:
        """Обработать задачу через старую архитектуру (совместимость)."""
        # Здесь должна быть интеграция со старым агентом
        # Пока возвращаем простой ответ
        
        legacy_response = f"[Legacy mode] {prompt}"
        
        return {
            "response": legacy_response,
            "metadata": {
                "architecture": "legacy",
                "task_id": task_id,
                "warning": "Legacy mode - limited functionality"
            }
        }
    
    def _get_max_tokens(self, prompt: str) -> int:
        """Определить максимальное количество токенов на основе эвристики."""
        # Простая эвристика: длина промпта * 2, но не более 2000
        estimated = len(prompt.split()) * 2
        return min(max(estimated, 100), 2000)
    
    def _get_temperature(self, prompt: str) -> float:
        """Определить температуру на основе типа задачи."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["код", "программа", "алгоритм", "функция"]):
            return 0.3  # Низкая температура для технических задач
        
        if any(word in prompt_lower for word in ["творческ", "придумай", "вообра"]):
            return 0.9  # Высокая температура для творческих задач
        
        return 0.7  # Средняя температура по умолчанию
    
    def _update_stats(self, result: Dict[str, Any], response_time_ms: float):
        """Обновить статистику на основе результата."""
        # Обновить среднее время ответа
        prev_avg = self.stats["avg_response_time_ms"]
        task_count = self.stats["total_tasks"]
        self.stats["avg_response_time_ms"] = (
            (prev_avg * (task_count - 1) + response_time_ms) / task_count
        )
        
        # Сложность задачи
        if "analysis" in result.get("metadata", {}):
            analysis = result["metadata"]["analysis"]
            complexity = analysis.get("complexity", "unknown")
            
            if complexity not in self.stats["tasks_by_complexity"]:
                self.stats["tasks_by_complexity"][complexity] = 0
            self.stats["tasks_by_complexity"][complexity] += 1
        
        # Компонент
        if "analysis" in result.get("metadata", {}):
            analysis = result["metadata"]["analysis"]
            component = analysis.get("component_used", "unknown")
            
            if component not in self.stats["tasks_by_component"]:
                self.stats["tasks_by_component"][component] = 0
            self.stats["tasks_by_component"][component] += 1
        
        # Стоимость
        if "metrics" in result.get("metadata", {}):
            metrics = result["metadata"]["metrics"]
            cost = metrics.get("cost_usd", 0.0)
            self.stats["total_cost_usd"] += cost
    
    def _log_task_result(
        self,
        task_id: str,
        prompt: str,
        result: Dict[str, Any],
        response_time_ms: float
    ):
        """Записать результат задачи в лог."""
        log_entry = {
            "ts": utc_now_iso(),
            "type": "neuro_task_result",
            "task_id": task_id,
            "prompt_preview": truncate_for_log(prompt, 200),
            "response_preview": truncate_for_log(result.get("response", ""), 200),
            "metadata": result.get("metadata", {}),
            "response_time_ms": response_time_ms,
            "stats_snapshot": {
                "total_tasks": self.stats["total_tasks"],
                "avg_response_time_ms": self.stats["avg_response_time_ms"]
            }
        }
        
        append_jsonl(self.env.drive_path("logs") / "neuro_tasks.jsonl", log_entry)
    
    async def analyze_task(self, prompt: str, detailed: bool = False) -> Dict[str, Any]:
        """
        Проанализировать задачу без выполнения.
        
        Args:
            prompt: Текст задачи для анализа
            detailed: Вернуть подробный анализ
            
        Returns:
            Анализ задачи
        """
        try:
            if self.neuro_tools_available:
                # Использовать нейросимбиотический инструмент анализа
                for tool in self.neuro_tools:
                    if hasattr(tool, '__name__') and tool.__name__ == 'neuro_symbiotic_analyze':
                        return await tool(prompt=prompt, detailed=detailed)
            
            # Fallback: использовать integrated_agent
            task_analysis = await self.integrated_agent.analyze_task(prompt)
            
            return {
                "complexity": task_analysis.complexity.value,
                "estimated_tokens": task_analysis.estimated_tokens,
                "requires_specialist": task_analysis.requires_specialist,
                "urgency": task_analysis.urgency,
                "cost_sensitivity": task_analysis.cost_sensitivity
            }
            
        except Exception as e:
            log.error(f"Ошибка анализа задачи: {e}")
            return {
                "error": str(e),
                "complexity": "unknown",
                "estimated_tokens": 300
            }
    
    async def get_stats(self, reset: bool = False) -> Dict[str, Any]:
        """
        Получить статистику работы нейросимбиотического агента.
        
        Args:
            reset: Сбросить статистику
            
        Returns:
            Статистика
        """
        try:
            # Получить статистику экосистемы
            ecosystem_stats = self.ecosystem.get_stats()
            
            # Получить статистику integrated_agent
            agent_stats = self.integrated_agent.get_stats()
            
            # Объединить статистику
            combined_stats = {
                "neuro_agent": self.stats.copy(),
                "ecosystem": ecosystem_stats,
                "integrated_agent": agent_stats,
                "config": self.config,
                "timestamp": utc_now_iso(),
                "uptime_seconds": time.time() - self.stats["start_time"]
            }
            
            # Сбросить статистику, если нужно
            if reset:
                self.stats = {
                    "start_time": time.time(),
                    "total_tasks": 0,
                    "tasks_by_complexity": {},
                    "tasks_by_component": {},
                    "total_cost_usd": 0.0,
                    "avg_response_time_ms": 0.0,
                    "fallback_to_legacy_count": 0
                }
                
                await self.ecosystem.reset_stats()
                
                combined_stats["reset"] = True
            
            return combined_stats
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Не удалось получить статистику"
            }
    
    async def optimize_performance(self):
        """Оптимизировать производительность нейросимбиотической архитектуры."""
        try:
            log.info("Оптимизация производительности нейросимбиотического агента...")
            
            # Оптимизировать integrated_agent
            await self.integrated_agent.optimize_performance()
            
            # Оптимизировать экосистему
            await self.ecosystem.optimize_configuration()
            
            # Проанализировать статистику для корректировки конфигурации
            self._analyze_and_adjust_config()
            
            # Сохранить оптимизированную конфигурацию
            self._save_config()
            
            log.info("Оптимизация завершена")
            
        except Exception as e:
            log.error(f"Ошибка оптимизации производительности: {e}")
    
    def _analyze_and_adjust_config(self):
        """Проанализировать статистику и скорректировать конфигурацию."""
        total_tasks = self.stats["total_tasks"]
        if total_tasks < 10:
            return  # Недостаточно данных
        
        # Анализ использования компонентов
        component_usage = self.stats["tasks_by_component"]
        
        # Если какой-то компонент используется редко, можно отключить его
        for component, count in component_usage.items():
            usage_rate = count / total_tasks
            if usage_rate < 0.05:  # Менее 5% использования
                log.info(f"Компонент {component} используется редко ({usage_rate:.1%})")
                # Можно предложить отключить или перенастроить
        
        # Анализ сложности задач
        complexity_dist = self.stats["tasks_by_complexity"]
        
        # Если слишком много сложных задач, возможно нужно улучшить маршрутизацию
        strategic_count = complexity_dist.get("strategic", 0)
        if strategic_count / total_tasks > 0.3:  # Более 30% стратегических задач
            log.info("Высокий процент стратегических задач, возможно нужна настройка анализа")
    
    def _save_config(self):
        """Сохранить текущую конфигурацию."""
        config_path = self.env.drive_path("config/neuro_agent.json")
        config_dir = config_path.parent
        
        try:
            if not config_dir.exists():
                config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
                
            log.debug("Конфигурация нейросимбиотического агента сохранена")
            
        except Exception as e:
            log.error(f"Ошибка сохранения конфигурации: {e}")
    
    async def shutdown(self):
        """Завершить работу нейросимбиотического агента."""
        try:
            log.info("Завершение работы нейросимбиотического агента...")
            
            # Сохранить статистику
            stats_path = self.env.drive_path("stats/neuro_agent_stats.json")
            stats_dir = stats_path.parent
            
            if not stats_dir.exists():
                stats_dir.mkdir(parents=True, exist_ok=True)
            
            final_stats = await self.get_stats()
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, indent=2, ensure_ascii=False)
            
            # Завершить integrated_agent
            await self.integrated_agent.shutdown()
            
            # Завершить экосистему
            await self.ecosystem.shutdown()
            
            log.info("Нейросимбиотический агент завершил работу")
            
        except Exception as e:
            log.error(f"Ошибка завершения работы: {e}")


# Фабрика для создания нейросимбиотического агента
def create_neuro_agent(
    repo_dir: str | Path,
    drive_root: str | Path,
    branch_dev: str = "ouroboros"
) -> NeuroAgent:
    """
    Создать нейросимбиотического агента.
    
    Args:
        repo_dir: Путь к репозиторию
        drive_root: Путь к корню данных на диске
        branch_dev: Имя рабочей ветки
        
    Returns:
        Экземпляр NeuroAgent
    """
    env = NeuroEnv(
        repo_dir=Path(repo_dir),
        drive_root=Path(drive_root),
        branch_dev=branch_dev
    )
    
    return NeuroAgent(env)


# Демонстрация работы
async def demo_neuro_agent():
    """Демонстрация работы нейросимбиотического агента."""
    print("Демонстрация нейросимбиотического агента Ouroboros")
    print("=" * 60)
    
    # Создать агента (с использованием временных путей для демо)
    current_dir = Path(__file__).parent.parent
    demo_drive = current_dir / "data" / "local_state_demo"
    
    # Создать директорию для демо, если не существует
    demo_drive.mkdir(parents=True, exist_ok=True)
    
    agent = create_neuro_agent(
        repo_dir=current_dir,
        drive_root=demo_drive,
        branch_dev="ouroboros"
    )
    
    # Тестовые задачи
    test_tasks = [
        "Привет, как дела?",
        "Что такое нейросимбиотическая архитектура?",
        "Напиши простую функцию на Python для анализа сложности текста",
        "Разработай план развития искусственного интеллекта на следующие 10 лет"
    ]
    
    for i, prompt in enumerate(test_tasks):
        print(f"\nТест {i+1}/{len(test_tasks)}: {prompt[:50]}...")
        
        # Анализ задачи
        analysis = await agent.analyze_task(prompt)
        print(f"  Анализ: сложность={analysis.get('complexity', 'unknown')}, "
              f"токенов={analysis.get('estimated_tokens', '?')}")
        
        # Обработка
        start_time = time.time()
        result = await agent.process(prompt)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"  Архитектура: {result.get('metadata', {}).get('architecture', 'unknown')}")
        print(f"  Время: {processing_time:.0f}ms")
        print(f"  Ответ (начало): {result.get('response', '')[:100]}...")
        print("-" * 40)
    
    # Показать статистику
    print("\nИтоговая статистика:")
    stats = await agent.get_stats()
    
    # Вывести ключевую информацию
    neuro_stats = stats.get("neuro_agent", {})
    print(f"Всего задач: {neuro_stats.get('total_tasks', 0)}")
    print(f"Среднее время ответа: {neuro_stats.get('avg_response_time_ms', 0):.0f}ms")
    print(f"Fallback к старой архитектуре: {neuro_stats.get('fallback_to_legacy_count', 0)}")
    
    # Распределение по сложности
    complexity_dist = neuro_stats.get("tasks_by_complexity", {})
    if complexity_dist:
        print("Распределение по сложности:")
        for comp, count in complexity_dist.items():
            print(f"  {comp}: {count}")
    
    # Распределение по компонентам
    component_dist = neuro_stats.get("tasks_by_component", {})
    if component_dist:
        print("Распределение по компонентам:")
        for comp, count in component_dist.items():
            print(f"  {comp}: {count}")
    
    # Завершить работу
    await agent.shutdown()
    
    print("\nДемонстрация завершена успешно!")


if __name__ == "__main__":
    # Запустить демонстрацию
    asyncio.run(demo_neuro_agent())