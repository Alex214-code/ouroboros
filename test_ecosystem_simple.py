#!/usr/bin/env python3
"""Простой тест когнитивной экосистемы."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ouroboros.cognitive_agent import create_cognitive_ecosystem

def main():
    """Протестировать экосистему на разных задачах."""
    print("Создание когнитивной экосистемы...")
    ecosystem = create_cognitive_ecosystem()
    print("Экосистема создана!")
    
    # Тестовые задачи
    test_tasks = [
        "Какая сегодня дата?",
        "Создай программу на Python для вычисления чисел Фибоначчи",
        "Объясни, как работает квантовая запутанность",
        "Проанализируй текущую архитектуру Ouroboros и предложи улучшения",
        "Разработай план миграции на полностью локальное исполнение",
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'='*60}")
        print(f"Тест {i}: {task}")
        print(f"{'='*60}")
        
        try:
            result = ecosystem.process(task)
            print(f"Анализ сложности: {result.get('task_complexity', 'unknown')}")
            print(f"Домен: {result.get('task_domain', 'unknown')}")
            
            plan = result.get('execution_plan', [])
            if plan:
                components = [c.name for c in plan]
                print(f"Выбранные компоненты: {components}")
            else:
                print("План выполнения не создан")
            
            response = result.get('response', '')
            if response:
                preview = response[:300] + "..." if len(response) > 300 else response
                print(f"Результат: {preview}")
            else:
                print("Ответ не сгенерирован")
                
        except Exception as e:
            print(f"Ошибка при обработке задачи: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Тест завершён!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()