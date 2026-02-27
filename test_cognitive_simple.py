#!/usr/bin/env python3
"""
Простой тест когнитивной экосистемы без зависимостей.
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ouroboros.cognitive_agent import CognitiveEcosystem, CognitiveTask
    print("✅ Успешно импортирован CognitiveEcosystem и CognitiveTask")
    
    # Создаём простую конфигурацию
    config = {
        "enabled": True,
        "local_model": "qwen3:8b",
        "cloud_model": "deepseek/deepseek-v3.2"
    }
    
    # Создаём экосистему
    ecosystem = CognitiveEcosystem(config)
    print("✅ CognitiveEcosystem успешно создан")
    
    # Тестируем базовые функции
    print("\nТестирование базовых функций:")
    
    # Создаём тестовую задачу
    task = CognitiveTask(
        id="test_task_1",
        input_text="Создай архитектуру для нейросимбиотической экосистемы",
        context={"test_mode": True, "budget_remaining_usd": 10.0},
        priority=1,
        max_iterations=2
    )
    
    print(f"Задача создана: '{task.input_text[:50]}...'")
    
    # Тестируем оценку сложности
    complexity = ecosystem._estimate_complexity(task.input_text)
    print(f"Оценка сложности: {complexity:.2f}")
    
    # Тестируем определение домена
    domain = ecosystem._detect_domain(task.input_text)
    print(f"Определённый домен: {domain}")
    
    # Тестируем анализ задачи
    analysis = ecosystem._analyze_task(task)
    print(f"Анализ задачи:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Тестируем создание плана выполнения
    plan = ecosystem._create_execution_plan(task, analysis)
    print(f"\nПлан выполнения ({len(plan)} шагов):")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step['component']}: {step['description']}")
    
    # Тестируем обработку задачи
    print("\nЗапуск обработки задачи...")
    result = ecosystem.process(task)
    
    print(f"\nРезультат обработки:")
    print(f"  ID задачи: {result.get('task_id')}")
    print(f"  Использованные компоненты: {result.get('components_used', [])}")
    print(f"  Общая стоимость: ${result.get('total_cost', 0):.4f}")
    print(f"  Общее время: {result.get('total_time', 0):.2f} сек")
    print(f"  Обновлено записей в графе знаний: {result.get('knowledge_updated', 0)}")
    
    if result.get('final_response'):
        print(f"\nФинальный ответ (первые 300 символов):")
        print("-" * 80)
        print(result['final_response'][:300] + "...")
        print("-" * 80)
    
    print("\n✅ Все тесты пройдены успешно!")
    
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)