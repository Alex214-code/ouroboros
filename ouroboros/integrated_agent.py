"""
Интеграция нейросимбиотической архитектуры с Ouroboros.

Этот модуль объединяет существующего агента Ouroboros
с новой нейросимбиотической экосистемой.
"""

import asyncio
import logging
import sys
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path

log = logging.getLogger(__name__)


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    STRATEGIC = "strategic"

try:
    from ouroboros.neuro_evolution import NeuroSymbioticEcosystem, get_ecosystem
    HAS_NEURO = True
except ImportError:
    HAS_NEURO = False
    log.warning("Модуль neuro_evolution не найден. Создаём заглушку.")
    
    class NeuroSymbioticEcosystem:
        def __init__(self):
            self.components = {}
            self.task_history = []
        
        async def analyze_task(self, prompt: str) -> Dict[str, Any]:
            return {"complexity": "unknown", "estimated_tokens": 0}
        
        async def route_task(self, prompt: str) -> Dict[str, Any]:
            return {"component_id": "legacy", "component_name": "Legacy", "response": "Neuro ecosystem not available"}
        
        def get_stats(self) -> Dict[str, Any]:
            return {"total_tasks": 0, "components": [], "recent_tasks": []}
        
        async def shutdown(self):
            pass
    
    def get_ecosystem():
        return NeuroSymbioticEcosystem()


class IntegratedAgent:
    """
    Интегрированный агент Ouroboros.
    
    Объединяет:
    1. Существующий агент Ouroboros (legacy)
    2. Нейросимбиотическую экосистему
    3. Адаптивную маршрутизацию
    4. Обучение на основе обратной связи
    """
    
    def __init__(
        self,
        repo_dir: Path,
        drive_root: Path,
        branch_dev: str = "ouroboros"
    ):
        self.repo_dir = Path(repo_dir)
        self.drive_root = Path(drive_root)
        self.branch_dev = branch_dev
        
        # Инициализация экосистемы
        self.ecosystem = get_ecosystem()
        
        # Состояние агента
        self.use_neuro_by_default = True
        self.neuro_tools_available = HAS_NEURO
        self.performance_stats = {
            "total_tasks": 0,
            "neuro_tasks": 0,
            "legacy_tasks": 0,
            "avg_response_time_ms": 0,
            "total_response_time_ms": 0
        }
        
        # Инициализация legacy агента (если доступен)
        self.legacy_agent = None
        self._init_legacy_agent()
        
        log.info(f"Интегрированный агент инициализирован. Нейросимбиотическая архитектура: {'доступна' if HAS_NEURO else 'недоступна'}")
    
    def _init_legacy_agent(self):
        """Инициализировать legacy агента Ouroboros."""
        try:
            # Попытка импорта существующего агента
            sys.path.insert(0, str(self.repo_dir))
            from ouroboros.agent import OuroborosAgent, Env
            env = Env(repo_dir=self.repo_dir, drive_root=self.drive_root, branch_dev=self.branch_dev)
            self.legacy_agent = OuroborosAgent(env=env)
            log.info("Legacy агент Ouroboros успешно инициализирован")
        except ImportError as e:
            log.warning(f"Не удалось импортировать legacy агента: {e}")
            self.legacy_agent = None
        except Exception as e:
            log.error(f"Ошибка инициализации legacy агента: {e}")
            self.legacy_agent = None
    
    async def analyze_task(
        self,
        prompt: str,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Проанализировать задачу для определения оптимального подхода.
        
        Args:
            prompt: Текст задачи
            detailed: Детальный анализ
            
        Returns:
            Результат анализа
        """
        import time
        start_time = time.time()
        
        # Базовая эвристика
        word_count = len(prompt.split())
        has_question = "?" in prompt
        has_code = any(keyword in prompt.lower() for keyword in 
                      ["код", "функция", "алгоритм", "программа", "debug"])
        has_research = any(keyword in prompt.lower() for keyword in 
                          ["исследование", "анализ", "стратегия", "план"])
        
        # Определить подходящий подход
        if word_count < 15 and not has_code and not has_research:
            # Простые задачи - legacy агент
            architecture = "legacy"
            complexity = "simple"
            reasoning = "legacy_agent"
        elif word_count < 100 and not has_research:
            # Задачи средней сложности - нейросимбиотическая экосистема
            architecture = "neuro"
            complexity = "moderate"
            reasoning = "neuro_ecosystem"
        elif word_count < 300:
            # Сложные задачи - комбинированный подход
            architecture = "hybrid"
            complexity = "complex"
            reasoning = "hybrid_approach"
        else:
            # Стратегические задачи - нейросимбиотическая с исследованием
            architecture = "neuro_research"
            complexity = "strategic"
            reasoning = "neuro_research_chain"
        
        # Детальный анализ при необходимости
        analysis = {
            "architecture": architecture,
            "complexity": complexity,
            "reasoning": reasoning,
            "word_count": word_count,
            "has_question": has_question,
            "has_code": has_code,
            "has_research": has_research,
            "estimated_tokens": word_count * 3,
            "analysis_time_ms": (time.time() - start_time) * 1000
        }
        
        if detailed:
            # Использовать нейросимбиотическую экосистему для детального анализа
            if HAS_NEURO:
                neuro_analysis = await self.ecosystem.analyze_task(prompt)
                analysis.update(neuro_analysis)
                analysis["analysis_method"] = "neuro_ecosystem"
            else:
                analysis["analysis_method"] = "legacy_heuristic"
        
        return analysis
    
    async def process(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        use_neuro_architecture: Optional[bool] = None,
        tools: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Обработать задачу через интегрированную архитектуру.
        
        Args:
            prompt: Текст задачи
            context: Дополнительный контекст
            use_neuro_architecture: Использовать нейросимбиотическую архитектуру
                                     (None = автоматический выбор)
            tools: Доступные инструменты
            
        Returns:
            Результат обработки
        """
        import time
        start_time = time.time()
        
        # Определить подход
        if use_neuro_architecture is None:
            analysis = await self.analyze_task(prompt)
            use_neuro_architecture = analysis["architecture"] in ["neuro", "hybrid", "neuro_research"]
            architecture = analysis["architecture"]
        else:
            architecture = "neuro" if use_neuro_architecture else "legacy"
            analysis = {"architecture": architecture}
        
        # Обработка через выбранную архитектуру
        if use_neuro_architecture and HAS_NEURO:
            # Нейросимбиотическая архитектура
            result = await self._process_neuro(prompt, context, tools, analysis)
            self.performance_stats["neuro_tasks"] += 1
        else:
            # Legacy архитектура
            result = await self._process_legacy(prompt, context, tools)
            self.performance_stats["legacy_tasks"] += 1
        
        # Обновить статистику
        processing_time_ms = (time.time() - start_time) * 1000
        self._update_performance_stats(processing_time_ms)
        
        # Добавить метаданные
        result["metadata"] = {
            "architecture": architecture,
            "processing_time_ms": processing_time_ms,
            "analysis": analysis,
            "neuro_available": HAS_NEURO,
            "legacy_available": self.legacy_agent is not None
        }
        
        return result
    
    async def _process_neuro(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        tools: Optional[list],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Обработать задачу через нейросимбиотическую архитектуру."""
        if not HAS_NEURO:
            return {
                "response": "Нейросимбиотическая архитектура недоступна",
                "error": "neuro_evolution module not found"
            }
        
        # Использовать экосистему для маршрутизации
        ecosystem_result = await self.ecosystem.route_task(prompt)
        
        # Здесь может быть дополнительная логика обработки результата экосистемы
        # Например, использование tools для выполнения действий
        
        return {
            "response": ecosystem_result.get("response", ""),
            "tool_calls": [],
            "component_used": ecosystem_result.get("component_id"),
            "ecosystem_stats": self.ecosystem.get_stats()
        }
    
    async def _process_legacy(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        tools: Optional[list]
    ) -> Dict[str, Any]:
        """Обработать задачу через legacy архитектуру."""
        if self.legacy_agent is None:
            return {
                "response": "Legacy агент недоступен",
                "error": "legacy agent not initialized"
            }
        
        # Здесь должна быть логика вызова legacy агента
        # Для демо возвращаем простой ответ
        
        return {
            "response": f"[Legacy агент] Обработал задачу: {prompt[:100]}...",
            "tool_calls": [],
            "agent_type": "legacy"
        }
    
    def _update_performance_stats(self, processing_time_ms: float):
        """Обновить статистику производительности."""
        self.performance_stats["total_tasks"] += 1
        self.performance_stats["total_response_time_ms"] += processing_time_ms
        self.performance_stats["avg_response_time_ms"] = (
            self.performance_stats["total_response_time_ms"] / 
            self.performance_stats["total_tasks"]
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получить статистику работы агента."""
        stats = {
            "integrated_agent": self.performance_stats,
            "neuro_available": HAS_NEURO,
            "legacy_available": self.legacy_agent is not None,
            "use_neuro_by_default": self.use_neuro_by_default
        }
        
        if HAS_NEURO:
            stats["ecosystem"] = self.ecosystem.get_stats()
        
        return stats
    
    async def optimize_performance(self):
        """Оптимизировать производительность на основе статистики."""
        log.info("Оптимизация производительности интегрированного агента...")
        
        stats = await self.get_stats()
        total_tasks = stats["integrated_agent"]["total_tasks"]
        
        if total_tasks < 10:
            log.info("Недостаточно данных для оптимизации")
            return
        
        # Анализ производительности
        neuro_tasks = stats["integrated_agent"]["neuro_tasks"]
        legacy_tasks = stats["integrated_agent"]["legacy_tasks"]
        
        neuro_ratio = neuro_tasks / total_tasks if total_tasks > 0 else 0
        legacy_ratio = legacy_tasks / total_tasks if total_tasks > 0 else 0
        
        # Простая эвристика для настройки
        if neuro_ratio < 0.3 and HAS_NEURO:
            # Увеличить использование нейросимбиотической архитектуры
            self.use_neuro_by_default = True
            log.info("Увеличено использование нейросимбиотической архитектуры")
        elif neuro_ratio > 0.7 and legacy_tasks > 0:
            # Увеличить использование legacy агента для баланса
            self.use_neuro_by_default = False
            log.info("Увеличено использование legacy архитектуры")
        
        # Оптимизация экосистемы
        if HAS_NEURO:
            await self.ecosystem.optimize_configuration()
        
        log.info("Оптимизация завершена")
    
    async def shutdown(self):
        """Завершить работу агента."""
        log.info("Завершение работы интегрированного агента...")
        
        if HAS_NEURO:
            await self.ecosystem.shutdown()
        
        if self.legacy_agent:
            # Здесь должен быть вызов shutdown legacy агента
            pass
        
        log.info("Интегрированный агент завершил работу")


# Фабрика для создания агента
def create_integrated_agent(
    repo_dir: Path,
    drive_root: Path,
    branch_dev: str = "ouroboros"
) -> IntegratedAgent:
    """
    Создать интегрированного агента.
    
    Args:
        repo_dir: Путь к репозиторию
        drive_root: Путь к данным
        branch_dev: Рабочая ветка
        
    Returns:
        Экземпляр IntegratedAgent
    """
    return IntegratedAgent(
        repo_dir=repo_dir,
        drive_root=drive_root,
        branch_dev=branch_dev
    )


# Демонстрационная функция
async def demo_integrated_agent():
    """Демонстрация работы интегрированного агента."""
    import sys
    from pathlib import Path
    
    print("Демонстрация интегрированного агента Ouroboros")
    print("=" * 60)
    
    # Определить пути
    repo_dir = Path(__file__).parent.parent
    drive_root = repo_dir / "data" / "local_state_demo"
    
    if not drive_root.exists():
        drive_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Репозиторий: {repo_dir}")
    print(f"Данные: {drive_root}")
    print()
    
    # Создать агента
    print("Создание интегрированного агента...")
    agent = create_integrated_agent(repo_dir, drive_root)
    print(f"✓ Агент создан")
    print(f"  Нейросимбиотическая архитектура: {'доступна' if HAS_NEURO else 'недоступна'}")
    print(f"  Legacy агент: {'доступен' if agent.legacy_agent is not None else 'недоступен'}")
    print()
    
    # Тестовые задачи
    test_prompts = [
        "Привет, как дела?",
        "Объясни разницу между нейросетью и алгоритмом",
        "Напиши функцию на Python для анализа тональности текста",
        "Разработай стратегию развития искусственного интеллекта на следующие 10 лет"
    ]
    
    print("Тестирование обработки задач:")
    for i, prompt in enumerate(test_prompts):
        print(f"\nЗадача {i+1}: {prompt[:50]}...")
        
        # Анализ
        analysis = await agent.analyze_task(prompt, detailed=True)
        print(f"  Архитектура: {analysis.get('architecture', 'unknown')}")
        print(f"  Сложность: {analysis.get('complexity', 'unknown')}")
        
        # Обработка
        result = await agent.process(prompt)
        
        print(f"  Использованная архитектура: {result['metadata']['architecture']}")
        print(f"  Время обработки: {result['metadata']['processing_time_ms']:.0f}ms")
        print(f"  Ответ: {result.get('response', '')[:80]}...")
    
    # Статистика
    print("\n" + "=" * 60)
    print("Статистика работы:")
    stats = await agent.get_stats()
    
    print(f"Всего задач: {stats['integrated_agent']['total_tasks']}")
    print(f"Нейросимбиотических задач: {stats['integrated_agent']['neuro_tasks']}")
    print(f"Legacy задач: {stats['integrated_agent']['legacy_tasks']}")
    print(f"Среднее время ответа: {stats['integrated_agent']['avg_response_time_ms']:.0f}ms")
    
    if HAS_NEURO:
        print(f"\nСтатистика экосистемы:")
        eco_stats = stats.get('ecosystem', {})
        print(f"  Компонентов: {len(eco_stats.get('components', []))}")
        print(f"  Всего задач: {eco_stats.get('total_tasks', 0)}")
    
    # Оптимизация
    print("\nОптимизация производительности...")
    await agent.optimize_performance()
    
    # Завершить работу
    print("\nЗавершение работы...")
    await agent.shutdown()
    
    print("\n" + "=" * 60)
    print("Демонстрация завершена!")


if __name__ == "__main__":
    # Запустить демонстрацию
    asyncio.run(demo_integrated_agent())
