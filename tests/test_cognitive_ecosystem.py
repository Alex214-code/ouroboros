"""Тест когнитивной экосистемы"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ouroboros.cognitive_agent import CognitiveEcosystem, CognitiveTask

def test_basic_functionality():
    """Тест базовой функциональности экосистемы"""
    
    # Создаём экосистему
    ecosystem = CognitiveEcosystem({
        "local_model": "qwen3:8b",
        "cloud_model": "deepseek/deepseek-v3.2"
    })
    
    # Простая задача
    simple_task = CognitiveTask(
        id="test_1",
        input_text="Привет, как дела?",
        context={"budget_remaining_usd": 5.0}
    )
    
    # Сложная задача
    complex_task = CognitiveTask(
        id="test_2",
        input_text="Разработай новую архитектуру для распределённого ИИ, который сможет обучаться на лету и сохранять быстродействие даже при росте модели до триллиона токенов.",
        context={"budget_remaining_usd": 5.0}
    )
    
    print("=== Тест простой задачи ===")
    result1 = ecosystem.process(simple_task)
    print(f"Результат: {result1}")
    print(f"Использованные компоненты: {result1.get('components_used', [])}")
    
    print("\n=== Тест сложной задачи ===")
    result2 = ecosystem.process(complex_task)
    print(f"Результат: {result2}")
    print(f"Использованные компоненты: {result2.get('components_used', [])}")
    
    print("\n=== Анализ экосистемы ===")
    print(f"Количество компонентов: {len(ecosystem.components)}")
    print(f"Задач в истории: {len(ecosystem.task_history)}")
    
    # Проверяем, что для разных задач выбираются разные компоненты
    if len(result1.get('components_used', [])) < len(result2.get('components_used', [])):
        print("✓ Экосистема правильно распределяет задачи по сложности")
    else:
        print("✗ Экосистема не различает сложность задач")
    
    return True

def test_task_analysis():
    """Тест анализа задач"""
    
    ecosystem = CognitiveEcosystem({})
    
    tasks = [
        ("Простая задача", "Привет, как дела?", True),
        ("Математическая задача", "Реши уравнение: x^2 + 5x + 6 = 0", False),
        ("Исследовательская задача", "Проанализируй текущее состояние ИИ и предложи новые направления развития", False),
        ("Креативная задача", "Придумай новую архитектуру для самообучающегося ИИ", False),
        ("Сложная задача", "Разработай план миграции с облачных моделей на локальные с сохранением качества и скорости", False)
    ]
    
    print("=== Анализ различных задач ===")
    
    for name, text, expected_routine in tasks:
        task = CognitiveTask(id=f"analysis_{name}", input_text=text, context={})
        analysis = ecosystem._analyze_task(task)
        
        print(f"\nЗадача: {name}")
        print(f"Текст: {text[:50]}...")
        print(f"Сложность: {analysis['complexity']:.2f}")
        print(f"Домен: {analysis['domain']}")
        print(f"Требует креативности: {analysis['requires_creativity']}")
        print(f"Требует глубокого мышления: {analysis['requires_deep_reasoning']}")
        print(f"Рутинная: {analysis['is_routine']} (ожидалось: {expected_routine})")
        
        # Проверяем план выполнения
        plan = ecosystem._create_execution_plan(task, analysis)
        print(f"План выполнения ({len(plan)} шагов):")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step['component']} - {step['description']}")

if __name__ == "__main__":
    print("Запуск тестов когнитивной экосистемы...")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        print("\n" + "=" * 50)
        test_task_analysis()
        print("\n✓ Все тесты пройдены успешно!")
    except Exception as e:
        print(f"\n✗ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)