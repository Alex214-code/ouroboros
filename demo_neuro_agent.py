#!/usr/bin/env python3
"""
Демонстрация нейросимбиотического агента Ouroboros.

Этот скрипт показывает:
1. Как создаётся и инициализируется нейросимбиотический агент
2. Как работает маршрутизация задач по сложности
3. Как статистика собирается и анализируется
4. Как можно постепенно перейти на новую архитектуру
"""

import asyncio
import json
import sys
from pathlib import Path

# Добавить путь к модулям Ouroboros
sys.path.insert(0, str(Path(__file__).parent))

from ouroboros.neuro_agent import create_neuro_agent, demo_neuro_agent


async def demo_basic_functionality():
    """Базовая демонстрация функциональности."""
    print("=" * 70)
    print("Демонстрация нейросимбиотического агента Ouroboros")
    print("=" * 70)
    
    # Определить пути
    repo_dir = Path(__file__).parent
    drive_root = repo_dir / "data" / "local_state"
    
    # Убедиться, что директория данных существует
    if not drive_root.exists():
        drive_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Репозиторий: {repo_dir}")
    print(f"Данные: {drive_root}")
    print()
    
    # Создать агента
    print("1. Создание нейросимбиотического агента...")
    agent = create_neuro_agent(
        repo_dir=repo_dir,
        drive_root=drive_root,
        branch_dev="ouroboros"
    )
    print("   ✓ Агент создан")
    
    # Протестировать с задачами разной сложности
    test_cases = [
        {
            "name": "Простая приветственная задача",
            "prompt": "Привет, расскажи о себе кратко",
            "expected_complexity": "simple"
        },
        {
            "name": "Задача средней сложности",
            "prompt": "Объясни, что такое нейросимбиотическая архитектура и как она работает",
            "expected_complexity": "moderate"
        },
        {
            "name": "Сложная техническая задача",
            "prompt": "Напиши код на Python для классификации текстовых задач по сложности. "
                     "Используй машинное обучение и предоставь полную реализацию.",
            "expected_complexity": "complex"
        },
        {
            "name": "Стратегическая задача",
            "prompt": "Разработай стратегию развития искусственного интеллекта на следующие 5 лет "
                     "с учётом текущих технологических трендов и ограничений аппаратного обеспечения",
            "expected_complexity": "strategic"
        }
    ]
    
    results = []
    
    print("\n2. Тестирование маршрутизации задач по сложности:")
    for i, test_case in enumerate(test_cases):
        print(f"\n   Тест {i+1}: {test_case['name']}")
        print(f"   Промпт: {test_case['prompt'][:80]}...")
        
        # Анализ задачи
        print("   - Анализ задачи...")
        analysis = await agent.analyze_task(test_case['prompt'], detailed=True)
        
        actual_complexity = analysis.get('complexity', 'unknown')
        print(f"     Определённая сложность: {actual_complexity}")
        print(f"     Ожидаемая сложность: {test_case['expected_complexity']}")
        
        if 'requires_specialist' in analysis and analysis['requires_specialist']:
            print(f"     Требуется специалист: {analysis['requires_specialist']}")
        
        # Обработка задачи
        print("   - Обработка задачи...")
        start_time = asyncio.get_event_loop().time()
        result = await agent.process(test_case['prompt'])
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Сохранить результаты
        results.append({
            "name": test_case['name'],
            "prompt": test_case['prompt'],
            "analysis": analysis,
            "result_metadata": result.get("metadata", {}),
            "response_preview": result.get("response", "")[:150] + "...",
            "processing_time_ms": processing_time,
            "architecture": result.get("metadata", {}).get("architecture", "unknown")
        })
        
        print(f"     Архитектура: {result['metadata'].get('architecture', 'unknown')}")
        print(f"     Время обработки: {processing_time:.0f}ms")
        print(f"     Ответ (начало): {result.get('response', '')[:100]}...")
    
    # Показать статистику
    print("\n3. Статистика работы:")
    stats = await agent.get_stats()
    
    neuro_stats = stats.get("neuro_agent", {})
    print(f"   Всего обработано задач: {neuro_stats.get('total_tasks', 0)}")
    print(f"   Среднее время ответа: {neuro_stats.get('avg_response_time_ms', 0):.0f}ms")
    
    complexity_dist = neuro_stats.get("tasks_by_complexity", {})
    if complexity_dist:
        print("   Распределение по сложности:")
        for comp, count in complexity_dist.items():
            percentage = (count / neuro_stats.get('total_tasks', 1)) * 100
            print(f"     {comp}: {count} ({percentage:.1f}%)")
    
    component_dist = neuro_stats.get("tasks_by_component", {})
    if component_dist:
        print("   Распределение по компонентам:")
        for comp, count in component_dist.items():
            percentage = (count / neuro_stats.get('total_tasks', 1)) * 100
            print(f"     {comp}: {count} ({percentage:.1f}%)")
    
    # Показать детализацию по задачам
    print("\n4. Детализация по задачам:")
    for i, res in enumerate(results):
        print(f"\n   Задача {i+1}: {res['name']}")
        print(f"     Архитектура: {res['architecture']}")
        print(f"     Время: {res['processing_time_ms']:.0f}ms")
        
        analysis = res.get('analysis', {})
        if 'complexity' in analysis:
            print(f"     Сложность: {analysis['complexity']}")
        
        metadata = res.get('result_metadata', {})
        if 'component_used' in metadata.get('analysis', {}):
            component = metadata['analysis']['component_used']
            print(f"     Использованный компонент: {component}")
    
    # Завершить работу агента
    print("\n5. Завершение работы агента...")
    await agent.shutdown()
    print("   ✓ Агент завершил работу")
    
    # Сохранить результаты в файл
    output_path = drive_root / "neuro_demo_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "test_cases": test_cases,
            "results": results,
            "stats": stats
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nРезультаты сохранены в: {output_path}")
    print("\n" + "=" * 70)
    print("Демонстрация успешно завершена!")
    print("=" * 70)


async def demo_performance_optimization():
    """Демонстрация оптимизации производительности."""
    print("\n" + "=" * 70)
    print("Демонстрация оптимизации производительности")
    print("=" * 70)
    
    repo_dir = Path(__file__).parent
    drive_root = repo_dir / "data" / "local_state_perf"
    
    if not drive_root.exists():
        drive_root.mkdir(parents=True, exist_ok=True)
    
    # Создать агента
    agent = create_neuro_agent(
        repo_dir=repo_dir,
        drive_root=drive_root,
        branch_dev="ouroboros"
    )
    
    # Выполнить несколько задач для сбора статистики
    print("1. Выполнение задач для сбора статистики...")
    for i in range(15):
        prompt = f"Тестовая задача #{i+1}: объясни концепцию {['нейросетей', 'машинного обучения', 'глубокого обучения'][i % 3]}"
        await agent.process(prompt)
    
    # Получить статистику до оптимизации
    print("2. Статистика до оптимизации:")
    stats_before = await agent.get_stats()
    neuro_stats_before = stats_before.get("neuro_agent", {})
    
    avg_time_before = neuro_stats_before.get("avg_response_time_ms", 0)
    print(f"   Среднее время ответа: {avg_time_before:.0f}ms")
    print(f"   Всего задач: {neuro_stats_before.get('total_tasks', 0)}")
    
    # Выполнить оптимизацию
    print("3. Выполнение оптимизации...")
    await agent.optimize_performance()
    print("   ✓ Оптимизация выполнена")
    
    # Выполнить ещё несколько задач после оптимизации
    print("4. Выполнение задач после оптимизации...")
    for i in range(10):
        prompt = f"Оптимизированная задача #{i+1}: расскажи о {['производительности', 'эффективности', 'оптимизации'][i % 3]}"
        await agent.process(prompt)
    
    # Получить статистику после оптимизации
    print("5. Статистика после оптимизации:")
    stats_after = await agent.get_stats()
    neuro_stats_after = stats_after.get("neuro_agent", {})
    
    avg_time_after = neuro_stats_after.get("avg_response_time_ms", 0)
    print(f"   Среднее время ответа: {avg_time_after:.0f}ms")
    print(f"   Всего задач: {neuro_stats_after.get('total_tasks', 0)}")
    
    # Рассчитать улучшение
    if avg_time_before > 0:
        improvement = ((avg_time_before - avg_time_after) / avg_time_before) * 100
        print(f"   Улучшение производительности: {improvement:.1f}%")
    
    # Завершить работу
    print("6. Завершение работы...")
    await agent.shutdown()
    
    print("\nДемонстрация оптимизации завершена!")
    print("=" * 70)


async def demo_integration_with_existing_system():
    """Демонстрация интеграции с существующей системой Ouroboros."""
    print("\n" + "=" * 70)
    print("Демонстрация интеграции с существующей системой")
    print("=" * 70)
    
    repo_dir = Path(__file__).parent
    drive_root = repo_dir / "data" / "local_state"
    
    print("1. Проверка существующей системы Ouroboros...")
    
    # Проверить наличие ключевых файлов
    key_files = [
        "ouroboros/agent.py",
        "ouroboros/llm.py",
        "ouroboros/tools/__init__.py",
        "BIBLE.md",
        "VERSION"
    ]
    
    missing_files = []
    for file_path in key_files:
        full_path = repo_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"   ⚠ Отсутствуют файлы: {', '.join(missing_files)}")
        print("   Некоторые возможности могут быть ограничены")
    else:
        print("   ✓ Ключевые файлы системы найдены")
    
    # Создать нейросимбиотического агента
    print("\n2. Создание нейросимбиотического агента...")
    agent = create_neuro_agent(
        repo_dir=repo_dir,
        drive_root=drive_root,
        branch_dev="ouroboros"
    )
    
    # Проверить доступность нейросимбиотических инструментов
    print("3. Проверка доступности инструментов...")
    if hasattr(agent, 'neuro_tools_available') and agent.neuro_tools_available:
        print(f"   ✓ Доступно {len(agent.neuro_tools)} нейросимбиотических инструментов")
    else:
        print("   ⚠ Нейросимбиотические инструменты не доступны")
    
    # Проверить доступность компонентов экосистемы
    print("4. Проверка компонентов экосистемы...")
    components = list(agent.ecosystem.components.keys())
    print(f"   ✓ Доступно {len(components)} компонентов: {', '.join(components)}")
    
    # Тест интеграции: использовать старый и новый подходы
    print("\n5. Тест интеграции (старый vs новый подход):")
    
    test_prompt = "Объясни разницу между традиционной и нейросимбиотической архитектурой ИИ"
    
    # Использовать нейросимбиотический подход
    print("   - Нейросимбиотический подход...")
    result_neuro = await agent.process(test_prompt, use_neuro_architecture=True)
    print(f"     Архитектура: {result_neuro.get('metadata', {}).get('architecture', 'unknown')}")
    print(f"     Ответ (начало): {result_neuro.get('response', '')[:100]}...")
    
    # Использовать legacy подход (если доступен)
    print("   - Legacy подход...")
    result_legacy = await agent.process(test_prompt, use_neuro_architecture=False)
    print(f"     Архитектура: {result_legacy.get('metadata', {}).get('architecture', 'unknown')}")
    print(f"     Ответ (начало): {result_legacy.get('response', '')[:100]}...")
    
    print("\n6. Сравнение результатов:")
    neuro_response = result_neuro.get('response', '')
    legacy_response = result_legacy.get('response', '')
    
    # Простое сравнение длины
    print(f"   Нейросимбиотический ответ: {len(neuro_response)} символов")
    print(f"   Legacy ответ: {len(legacy_response)} символов")
    
    # Завершить работу
    print("\n7. Завершение работы...")
    await agent.shutdown()
    
    print("\nДемонстрация интеграции завершена!")
    print("=" * 70)


async def main():
    """Основная функция демонстрации."""
    print("Демонстрационный набор нейросимбиотического агента Ouroboros")
    print()
    
    # Создать меню выбора демонстрации
    print("Выберите демонстрацию:")
    print("1. Базовая функциональность")
    print("2. Оптимизация производительности")
    print("3. Интеграция с существующей системой")
    print("4. Все демонстрации последовательно")
    print("5. Выход")
    
    try:
        choice = input("\nВведите номер (1-5): ").strip()
        
        if choice == "1":
            await demo_basic_functionality()
        elif choice == "2":
            await demo_performance_optimization()
        elif choice == "3":
            await demo_integration_with_existing_system()
        elif choice == "4":
            await demo_basic_functionality()
            await demo_performance_optimization()
            await demo_integration_with_existing_system()
        elif choice == "5":
            print("Выход.")
            return
        else:
            print(f"Неизвестный выбор: {choice}")
            
    except KeyboardInterrupt:
        print("\n\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\nОшибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запустить основную функцию
    asyncio.run(main())