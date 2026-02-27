#!/usr/bin/env python3
"""
Тестирование когнитивной экосистемы.
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from ouroboros.cognitive_agent import CognitiveEcosystem, CognitiveTask
import json

def test_cognitive_ecosystem():
    """Тестирование когнитивной экосистемы."""
    
    # Загружаем конфигурацию
    config_path = Path("cognitive_config.json")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            "enabled": True,
            "local_model": "qwen3:8b",
            "cloud_model": "deepseek/deepseek-v3.2"
        }
    
    print("=" * 80)
    print("Тестирование когнитивной экосистемы")
    print("=" * 80)
    
    # Создаём экосистему
    ecosystem = CognitiveEcosystem(config)
    
    # Тестовые задачи разной сложности
    test_tasks = [
        {
            "id": "simple_question",
            "input_text": "Какая сегодня погода?",
            "expected_complexity": "low"
        },
        {
            "id": "math_problem",
            "input_text": "Реши уравнение: 2x^2 + 5x - 3 = 0",
            "expected_complexity": "medium"
        },
        {
            "id": "complex_research",
            "input_text": "Разработай архитектуру нейросимбиотической экосистемы для запуска на CPU без GPU",
            "expected_complexity": "high"
        },
        {
            "id": "self_reflection",
            "input_text": "Проанализируй свою текущую архитектуру и предложи улучшения",
            "expected_complexity": "high"
        }
    ]
    
    for test_task in test_tasks:
        print(f"\nТестовая задача: {test_task['id']}")
        print(f"Вопрос: {test_task['input_text']}")
        print(f"Ожидаемая сложность: {test_task['expected_complexity']}")
        
        # Создаём когнитивную задачу
        task = CognitiveTask(
            id=test_task["id"],
            input_text=test_task["input_text"],
            context={"test_mode": True},
            priority=1,
            max_iterations=2
        )
        
        # Обрабатываем через экосистему
        result = ecosystem.process(task)
        
        print(f"Результат анализа:")
        print(f"  Определённая сложность: {result.get('complexity', 'unknown')}")
        print(f"  Основной домен: {result.get('primary_domain', 'unknown')}")
        print(f"  Использованные компоненты: {result.get('components_used', [])}")
        print(f"  Итераций: {result.get('iterations', 0)}")
        print(f"  Стоимость: ${result.get('total_cost', 0):.4f}")
        print(f"  Время: {result.get('total_time', 0):.2f} сек")
        
        if result.get("final_response"):
            print(f"  Ответ: {result['final_response'][:200]}...")
        
        print("-" * 80)

def test_task_complexity_analysis():
    """Тестирование анализа сложности задач."""
    
    from ouroboros.cognitive_agent import TaskComplexityAnalyzer
    
    analyzer = TaskComplexityAnalyzer()
    
    test_cases = [
        ("Привет, как дела?", "low"),
        ("Напиши функцию для сложения двух чисел", "medium"),
        ("Разработай план исследования возможностей квантовых вычислений", "high"),
        ("Реши систему уравнений: x + y = 10, 2x - y = 5", "medium"),
        ("Спроектируй архитектуру распределённой системы для обработки больших данных", "high"),
        ("Что такое нейронные сети?", "low"),
        ("Объясни теорию относительности простыми словами", "medium"),
        ("Предложи новую парадигму для искусственного интеллекта следующего поколения", "high")
    ]
    
    print("\n" + "=" * 80)
    print("Тестирование анализатора сложности задач")
    print("=" * 80)
    
    for text, expected in test_cases:
        complexity = analyzer.analyze(text)
        print(f"Задача: {text[:50]}...")
        print(f"  Ожидаемая сложность: {expected}")
        print(f"  Определённая сложность: {complexity}")
        print(f"  Совпадение: {'✓' if complexity == expected else '✗'}")
        print()

def test_component_selection():
    """Тестирование выбора компонентов."""
    
    from ouroboros.cognitive_agent import ComponentSelector
    
    selector = ComponentSelector()
    
    test_scenarios = [
        {
            "complexity": "low",
            "domains": ["basic", "greeting"],
            "expected": ["core"]
        },
        {
            "complexity": "medium", 
            "domains": ["mathematical", "problem_solving"],
            "expected": ["core", "math_reasoner"]
        },
        {
            "complexity": "high",
            "domains": ["research", "architectural", "strategic"],
            "expected": ["strategist", "core"]
        },
        {
            "complexity": "high",
            "domains": ["self_reflection", "philosophical"],
            "expected": ["strategist", "self_reflection", "core"]
        }
    ]
    
    print("\n" + "=" * 80)
    print("Тестирование селектора компонентов")
    print("=" * 80)
    
    for scenario in test_scenarios:
        components = selector.select_components(
            complexity=scenario["complexity"],
            domains=scenario["domains"]
        )
        print(f"Сценарий: сложность={scenario['complexity']}, домены={scenario['domains']}")
        print(f"  Ожидаемые компоненты: {scenario['expected']}")
        print(f"  Выбранные компоненты: {components}")
        print(f"  Совпадение: {'✓' if set(components) == set(scenario['expected']) else '✗'}")
        print()

if __name__ == "__main__":
    print("Запуск тестов когнитивной экосистемы")
    print()
    
    try:
        test_task_complexity_analysis()
        test_component_selection()
        test_cognitive_ecosystem()
        
        print("\n" + "=" * 80)
        print("Все тесты пройдены успешно!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nОшибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)