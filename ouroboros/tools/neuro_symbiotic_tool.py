"""
Инструмент для работы с нейросимбиотической архитектурой.

Позволяет Ouroboros использовать новую архитектуру через стандартный интерфейс инструментов.
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import asdict

try:
    from ..neuro_evolution import (
        NeuroSymbioticEcosystem, ComponentType,
        InferenceRequest, InferenceResult, get_ecosystem
    )
    from ..integrated_agent import IntegratedAgent, TaskComplexity
    _NEURO_AVAILABLE = True
except ImportError:
    _NEURO_AVAILABLE = False


def get_tools():
    """Возвращает список инструментов для нейросимбиотической архитектуры."""
    if not _NEURO_AVAILABLE:
        return []
    return [
        neuro_symbiotic_process,
        neuro_symbiotic_analyze,
        neuro_symbiotic_stats,
        neuro_symbiotic_configure,
        neuro_symbiotic_optimize,
        neuro_symbiotic_demo
    ]


async def neuro_symbiotic_process(
    prompt: str,
    context: Optional[str] = None,
    complexity_hint: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Обработать запрос через нейросимбиотическую архитектуру.
    
    Args:
        prompt: Текст запроса
        context: Дополнительный контекст (JSON строка)
        complexity_hint: Подсказка о сложности (simple, moderate, complex, strategic)
        max_tokens: Максимальное количество токенов
        temperature: Температура генерации
        
    Returns:
        Результат обработки с метаданными
    """
    start_time = time.time()
    
    try:
        # Инициализировать агента
        agent = IntegratedAgent()
        
        # Парсинг контекста
        parsed_context = None
        if context:
            try:
                parsed_context = json.loads(context)
            except json.JSONDecodeError:
                parsed_context = {"raw_context": context}
        
        # Если указана подсказка сложности, добавить в контекст
        if complexity_hint:
            if parsed_context is None:
                parsed_context = {}
            parsed_context["complexity_hint"] = complexity_hint
        
        # Обработать задачу
        result = await agent.process_task(
            prompt=prompt,
            context=parsed_context,
            tools=[]  # Пока без инструментов
        )
        
        # Рассчитать общее время
        total_time_ms = (time.time() - start_time) * 1000
        
        # Форматировать результат
        formatted_result = {
            "response": result["response"],
            "metadata": {
                "processing_time_ms": total_time_ms,
                "analysis": result["analysis"],
                "metrics": result["metrics"],
                "tool_calls": result["tool_calls"],
                "needs_follow_up": result["needs_follow_up"]
            }
        }
        
        return formatted_result
        
    except Exception as e:
        return {
            "error": str(e),
            "response": f"Ошибка обработки через нейросимбиотическую архитектуру: {e}",
            "metadata": {
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error_type": type(e).__name__
            }
        }


async def neuro_symbiotic_analyze(
    prompt: str,
    detailed: bool = False
) -> Dict[str, Any]:
    """
    Проанализировать задачу без выполнения.
    
    Args:
        prompt: Текст запроса для анализа
        detailed: Вернуть подробный анализ
        
    Returns:
        Анализ задачи
    """
    try:
        agent = IntegratedAgent()
        
        # Проанализировать задачу
        task_analysis = await agent.analyze_task(prompt)
        
        result = {
            "complexity": task_analysis.complexity.value,
            "estimated_tokens": task_analysis.estimated_tokens,
            "requires_specialist": task_analysis.requires_specialist,
            "urgency": task_analysis.urgency,
            "cost_sensitivity": task_analysis.cost_sensitivity,
            "recommended_component": "core_consciousness"  # Базовое значение
        }
        
        # Определить рекомендуемый компонент
        if task_analysis.complexity == TaskComplexity.SIMPLE:
            result["recommended_component"] = "core_consciousness"
        elif task_analysis.complexity == TaskComplexity.MODERATE:
            result["recommended_component"] = "core_consciousness"
        elif task_analysis.complexity == TaskComplexity.COMPLEX:
            result["recommended_component"] = "strategic_planner"
        elif task_analysis.complexity == TaskComplexity.STRATEGIC:
            result["recommended_component"] = "strategic_planner"
        
        # Если требуется специалист
        if task_analysis.requires_specialist:
            specialist_mapping = {
                "coding": "code_specialist",
                "vision": "vision_specialist",
                "research": "strategic_planner",
                "planning": "strategic_planner"
            }
            result["recommended_component"] = specialist_mapping.get(
                task_analysis.requires_specialist, 
                result["recommended_component"]
            )
        
        if detailed:
            # Добавить дополнительные детали
            ecosystem = get_ecosystem()
            result["available_components"] = list(ecosystem.components.keys())
            result["ecosystem_stats"] = ecosystem.get_stats()
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "complexity": "unknown",
            "estimated_tokens": 300
        }


async def neuro_symbiotic_stats(
    reset: bool = False
) -> Dict[str, Any]:
    """
    Получить статистику нейросимбиотической архитектуры.
    
    Args:
        reset: Сбросить статистику
        
    Returns:
        Статистика работы
    """
    try:
        agent = IntegratedAgent()
        ecosystem = get_ecosystem()
        
        stats = {
            "agent_stats": agent.get_stats(),
            "ecosystem_stats": ecosystem.get_stats(),
            "components": {}
        }
        
        # Информация о компонентах
        for name, component in ecosystem.components.items():
            stats["components"][name] = {
                "type": component.type.value,
                "model": component.model_name,
                "enabled": component.enabled,
                "cost_per_token": component.cost_per_token,
                "expected_latency_ms": component.expected_latency_ms
            }
        
        # Сбросить статистику, если нужно
        if reset:
            agent.stats = {
                "total_tasks": 0,
                "by_complexity": {c.value: 0 for c in TaskComplexity},
                "by_component": {},
                "total_cost": 0.0,
                "avg_response_time_ms": 0.0
            }
            
            ecosystem.stats = {
                "total_requests": 0,
                "total_cost": 0.0,
                "component_usage": {},
                "cache_hits": 0,
                "distillation_count": 0
            }
            
            stats["reset"] = True
        
        return stats
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Не удалось получить статистику"
        }


async def neuro_symbiotic_configure(
    component: str,
    enabled: Optional[bool] = None,
    model_name: Optional[str] = None,
    cost_per_token: Optional[float] = None,
    expected_latency_ms: Optional[int] = None
) -> Dict[str, Any]:
    """
    Настроить компонент нейросимбиотической архитектуры.
    
    Args:
        component: Имя компонента для настройки
        enabled: Включить/выключить компонент
        model_name: Имя модели для компонента
        cost_per_token: Стоимость за токен
        expected_latency_ms: Ожидаемая задержка в мс
        
    Returns:
        Результат настройки
    """
    try:
        ecosystem = get_ecosystem()
        
        if component not in ecosystem.components:
            return {
                "error": f"Компонент {component} не найден",
                "available_components": list(ecosystem.components.keys())
            }
        
        # Получить текущую конфигурацию
        current_config = ecosystem.components[component]
        changes = []
        
        # Применить изменения
        if enabled is not None:
            old_value = current_config.enabled
            current_config.enabled = enabled
            changes.append(f"enabled: {old_value} -> {enabled}")
        
        if model_name is not None:
            old_value = current_config.model_name
            current_config.model_name = model_name
            changes.append(f"model_name: {old_value} -> {model_name}")
        
        if cost_per_token is not None:
            old_value = current_config.cost_per_token
            current_config.cost_per_token = cost_per_token
            changes.append(f"cost_per_token: {old_value} -> {cost_per_token}")
        
        if expected_latency_ms is not None:
            old_value = current_config.expected_latency_ms
            current_config.expected_latency_ms = expected_latency_ms
            changes.append(f"expected_latency_ms: {old_value} -> {expected_latency_ms}")
        
        # Сохранить конфигурацию
        ecosystem.save_config("data/local_state/neuro_config.json")
        
        return {
            "success": True,
            "component": component,
            "changes": changes,
            "new_config": {
                "enabled": current_config.enabled,
                "model_name": current_config.model_name,
                "cost_per_token": current_config.cost_per_token,
                "expected_latency_ms": current_config.expected_latency_ms
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


async def neuro_symbiotic_optimize() -> Dict[str, Any]:
    """
    Оптимизировать нейросимбиотическую архитектуру на основе статистики.
    
    Returns:
        Результат оптимизации
    """
    try:
        agent = IntegratedAgent()
        ecosystem = get_ecosystem()
        
        # Оптимизировать агента
        await agent.optimize_performance()
        
        # Оптимизировать экосистему
        await ecosystem.optimize_configuration()
        
        # Сохранить оптимизированную конфигурацию
        ecosystem.save_config("data/local_state/neuro_config_optimized.json")
        
        return {
            "success": True,
            "optimizations_applied": [
                "agent_performance_optimization",
                "ecosystem_configuration_optimization"
            ],
            "new_stats": agent.get_stats()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


async def neuro_symbiotic_demo(
    test_cases: Optional[str] = None
) -> Dict[str, Any]:
    """
    Демонстрация работы нейросимбиотической архитектуры.
    
    Args:
        test_cases: JSON строка с тестовыми случаями
        
    Returns:
        Результаты демонстрации
    """
    try:
        # Парсинг тестовых случаев или использование по умолчанию
        if test_cases:
            cases = json.loads(test_cases)
        else:
            cases = [
                {"prompt": "Привет, как дела?", "complexity": "simple"},
                {"prompt": "Объясни концепцию нейросимбиотической архитектуры", "complexity": "moderate"},
                {"prompt": "Напиши код для классификации сложности текстовых задач", "complexity": "complex"},
                {"prompt": "Разработай стратегию развития ИИ на следующие 5 лет", "complexity": "strategic"}
            ]
        
        results = []
        agent = IntegratedAgent()
        
        for i, case in enumerate(cases):
            prompt = case["prompt"]
            expected_complexity = case.get("complexity", "unknown")
            
            print(f"\nДемо тест {i+1}/{len(cases)}: {prompt[:50]}...")
            
            # Обработать задачу
            result = await agent.process_task(
                prompt=prompt,
                context={"demo": True, "expected_complexity": expected_complexity}
            )
            
            results.append({
                "test_case": i + 1,
                "prompt": prompt[:100],
                "expected_complexity": expected_complexity,
                "actual_complexity": result["analysis"]["complexity"],
                "component_used": result["analysis"]["component_used"],
                "cost_usd": result["metrics"]["cost_usd"],
                "response_time_ms": result["metrics"]["total_response_time_ms"],
                "response_preview": result["response"][:150] + "..."
            })
        
        # Итоговая статистика
        total_cost = sum(r["cost_usd"] for r in results)
        avg_response_time = sum(r["response_time_ms"] for r in results) / len(results)
        
        return {
            "success": True,
            "test_count": len(results),
            "total_cost_usd": total_cost,
            "average_response_time_ms": avg_response_time,
            "results": results,
            "summary": {
                "components_used": {r["component_used"] for r in results},
                "complexity_distribution": {
                    r["actual_complexity"]: sum(1 for x in results if x["actual_complexity"] == r["actual_complexity"])
                    for r in results
                }
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


# Экспорт функций как инструментов
__all__ = [
    "neuro_symbiotic_process",
    "neuro_symbiotic_analyze",
    "neuro_symbiotic_stats",
    "neuro_symbiotic_configure",
    "neuro_symbiotic_optimize",
    "neuro_symbiotic_demo"
]