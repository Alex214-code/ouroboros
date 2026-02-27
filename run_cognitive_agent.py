#!/usr/bin/env python3
"""
Запуск Ouroboros с когнитивной экосистемой.

Этот скрипт запускает агента V2 с поддержкой когнитивной экосистемы.
Когнитивная экосистема автоматически анализирует задачи и решает,
использовать ли локальное ядро, облачного стратега или специализированные модули.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from ouroboros.agent_v2 import OuroborosAgentV2, Env

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cognitive_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.getLogger(__name__)

def main():
    """Запуск агента с когнитивной экосистемой."""
    
    # Определяем пути
    repo_dir = Path(os.getenv("OUROBOROS_REPO_DIR", Path(__file__).parent))
    drive_root = Path(os.getenv("OUROBOROS_DRIVE_ROOT", repo_dir / "data" / "local_state"))
    
    # Создаём окружение
    env = Env(
        repo_dir=repo_dir,
        drive_root=drive_root,
        branch_dev="ouroboros"
    )
    
    log.info(f"Starting Ouroboros V2 with Cognitive Ecosystem")
    log.info(f"Repo directory: {repo_dir}")
    log.info(f"Drive root: {drive_root}")
    
    # Проверяем конфигурацию когнитивной экосистемы
    config_path = repo_dir / "cognitive_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        log.info(f"Cognitive ecosystem config loaded: enabled={config.get('enabled', False)}")
    else:
        log.warning("Cognitive config not found, using defaults")
    
    # Создаём агента
    agent = OuroborosAgentV2(env)
    
    log.info("Agent initialized successfully")
    
    # Пример использования агента
    if len(sys.argv) > 1:
        # Если передана задача в аргументах
        task_text = " ".join(sys.argv[1:])
        task = {
            "id": "test_task",
            "text": task_text,
            "type": "direct_chat",
            "chat_id": 12345
        }
        
        log.info(f"Processing task: {task_text[:100]}...")
        result = agent.process_task(task)
        
        print("\n" + "="*80)
        print("Результат обработки:")
        print("="*80)
        
        if "cognitive_result" in result:
            print(f"Использована когнитивная экосистема:")
            print(f"  Компоненты: {result.get('usage', {}).get('components_used', [])}")
            print(f"  Стоимость: ${result.get('usage', {}).get('total_cost', 0):.4f}")
            print(f"  Время: {result.get('usage', {}).get('total_time', 0):.2f} сек")
            print()
        
        print(result.get("response", "Нет ответа"))
        print("="*80)
    else:
        # Интерактивный режим
        print("Ouroboros V2 with Cognitive Ecosystem")
        print("Введите задачу для обработки (или 'quit' для выхода):")
        
        while True:
            try:
                task_text = input("\n> ").strip()
                if task_text.lower() in ['quit', 'exit', 'выход']:
                    break
                
                if not task_text:
                    continue
                
                task = {
                    "id": f"interactive_{len(task_text)}",
                    "text": task_text,
                    "type": "direct_chat",
                    "chat_id": 12345
                }
                
                result = agent.process_task(task)
                
                if "error" in result:
                    print(f"Ошибка: {result['error']}")
                else:
                    print(f"\nОтвет: {result.get('response', 'Нет ответа')}")
                    
                    if result.get("usage", {}).get("cognitive_ecosystem"):
                        print(f"(Использована когнитивная экосистема)")
                
            except KeyboardInterrupt:
                print("\n\nВыход...")
                break
            except Exception as e:
                print(f"Ошибка: {e}")
                log.error(f"Error processing interactive task: {e}", exc_info=True)

if __name__ == "__main__":
    main()