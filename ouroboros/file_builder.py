\"""
Интеллектуальный конструктор файлов для нейросимбиотической архитектуры.

Оптимизирует создание больших файлов через:
1. Иерархическое планирование
2. Интеллектуальное разделение
3. Кэширование шаблонов
4. Пакетную обработку
\"""

import asyncio
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib

log = logging.getLogger(__name__)


@dataclass
class FilePlan:
    \"""План создания файла.\"""
    path: Path
    purpose: str
    estimated_size_kb: int
    structure: List[str]  # Разделы/компоненты
    dependencies: List[Path]  # Зависимости
    priority: int  # 1-10, где 10 - самый высокий
    complexity: str  # simple, moderate, complex, strategic


@dataclass
class FileChunk:
    \"""Часть файла для параллельной обработки.\"""
    chunk_id: str
    file_path: Path
    start_line: int
    end_line: int
    content: str
    dependencies: List[str]  # ID других чанков, от которых зависит


class FileBuilder:
    \"""
    Конструктор файлов для эффективной обработки больших файлов.
    
    Особенности:
    - Планирование перед созданием
    - Параллельное создание частей
    - Кэширование шаблонов
    - Автоматическое разделение на модули
    \"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".ouroboros" / "file_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Кэш шаблонов
        self.templates: Dict[str, str] = {}
        
        # История создания файлов
        self.build_history: List[Dict[str, Any]] = []
        
        # Статистика
        self.stats = {
            "files_created": 0,
            "total_size_kb": 0,
            "avg_time_per_kb": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        log.info(f"Инициализирован FileBuilder с кэшем в {self.cache_dir}")
    
    async def plan_file(
        self,
        file_path: Path,
        purpose: str,
        estimated_content: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> FilePlan:
        \"""
        Создать план файла на основе его цели.
        
        Args:
            file_path: Путь к файлу
            purpose: Назначение файла
            estimated_content: Предполагаемое содержимое (если известно)
            context: Дополнительный контекст
            
        Returns:
            FilePlan с структурой файла
        \"""
        # Анализ типа файла по расширению
        ext = file_path.suffix.lower()
        
        # Оценка размера
        estimated_size_kb = len(estimated_content.encode('utf-8')) / 1024 if estimated_content else 10
        
        # Определение сложности на основе размера
        if estimated_size_kb < 5:
            complexity = "simple"
        elif estimated_size_kb < 20:
            complexity = "moderate"
        elif estimated_size_kb < 100:
            complexity = "complex"
        else:
            complexity = "strategic"
        
        # Определение структуры на основе типа файла
        structure = self._determine_structure(ext, purpose, estimated_size_kb)
        
        # Определение зависимостей
        dependencies = self._find_dependencies(file_path, context)
        
        # Определение приоритета
        priority = self._determine_priority(purpose, complexity)
        
        return FilePlan(
            path=file_path,
            purpose=purpose,
            estimated_size_kb=estimated_size_kb,
            structure=structure,
            dependencies=dependencies,
            priority=priority,
            complexity=complexity
        )
    
    def _determine_structure(
        self,
        extension: str,
        purpose: str,
        size_kb: float
    ) -> List[str]:
        \"""Определить структуру файла на основе его типа.\"""
        structure = []
        
        if extension == ".py":
            # Python файлы
            structure = [
                "header_docstring",
                "imports",
                "constants",
                "classes",
                "functions",
                "main_block"
            ]
            
            if size_kb > 20:
                # Большие Python файлы делим на секции
                structure = [
                    "header_docstring",
                    "imports_section",
                    "types_section",
                    "classes_section",
                    "functions_section",
                    "utils_section",
                    "main_section"
                ]
        
        elif extension == ".md":
            # Markdown файлы
            structure = [
                "title",
                "toc",
                "introduction",
                "sections",
                "conclusion",
                "references"
            ]
        
        elif extension in [".json", ".yaml", ".yml", ".toml"]:
            # Конфигурационные файлы
            structure = [
                "root_object",
                "sections"
            ]
        
        else:
            # Общая структура для других файлов
            structure = [
                "header",
                "content",
                "footer"
            ]
        
        return structure
    
    def _find_dependencies(
        self,
        file_path: Path,
        context: Optional[Dict[str, Any]]
    ) -> List[Path]:
        \"""Найти зависимости файла.\"""
        dependencies = []
        
        if context and "dependencies" in context:
            deps = context.get("dependencies", [])
            for dep in deps:
                if isinstance(dep, str):
                    dep_path = Path(dep)
                    if dep_path.exists():
                        dependencies.append(dep_path)
        
        # Автоматическое определение зависимостей для Python файлов
        if file_path.suffix.lower() == ".py":
            # Проверка импортов в существующем файле (если есть)
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                for line in content.split('\n'):
                    if line.strip().startswith(("import ", "from ")):
                        # Извлечение имени модуля (упрощённо)
                        import_part = line.split()[1] if line.startswith("import ") else line.split()[1]
                        module_name = import_part.split('.')[0].split(',')[0].strip()
                        
                        # Поиск файла модуля
                        possible_paths = [
                            file_path.parent / f"{module_name}.py",
                            file_path.parent / module_name / "__init__.py",
                            file_path.parent / module_name / f"{module_name}.py"
                        ]
                        
                        for path in possible_paths:
                            if path.exists():
                                dependencies.append(path)
                                break
        
        return dependencies
    
    def _determine_priority(self, purpose: str, complexity: str) -> int:
        \"""Определить приоритет создания файла.\"""
        priority_map = {
            "core_module": 10,
            "config": 9,
            "main_entry": 8,
            "utils": 7,
            "tests": 6,
            "docs": 5,
            "examples": 4,
            "temp": 3,
            "backup": 2,
            "log": 1
        }
        
        # Определить категорию по назначению
        purpose_lower = purpose.lower()
        
        if any(keyword in purpose_lower for keyword in ["core", "main", "agent", "orchestrator"]):
            category = "core_module"
        elif any(keyword in purpose_lower for keyword in ["config", "settings", "options"]):
            category = "config"
        elif any(keyword in purpose_lower for keyword in ["entry", "main", "start", "launch"]):
            category = "main_entry"
        elif any(keyword in purpose_lower for keyword in ["util", "helper", "tool", "common"]):
            category = "utils"
        elif any(keyword in purpose_lower for keyword in ["test", "spec", "fixture"]):
            category = "tests"
        elif any(keyword in purpose_lower for keyword in ["doc", "readme", "manual"]):
            category = "docs"
        elif any(keyword in purpose_lower for keyword in ["example", "demo", "sample"]):
            category = "examples"
        elif any(keyword in purpose_lower for keyword in ["temp", "tmp", "scratch"]):
            category = "temp"
        elif any(keyword in purpose_lower for keyword in ["backup", "save", "snapshot"]):
            category = "backup"
        elif any(keyword in purpose_lower for keyword in ["log", "trace", "debug"]):
            category = "log"
        else:
            category = "utils"
        
        # Корректировка приоритета на основе сложности
        base_priority = priority_map.get(category, 5)
        
        if complexity == "strategic":
            return min(base_priority + 2, 10)
        elif complexity == "complex":
            return min(base_priority + 1, 10)
        elif complexity == "simple":
            return max(base_priority - 1, 1)
        else:  # moderate
            return base_priority
    
    async def create_file(
        self,
        file_path: Path,
        purpose: str,
        content_hint: str = "",
        context: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        \"""
        Интеллектуальное создание файла.
        
        Args:
            file_path: Путь к файлу
            purpose: Назначение файла
            content_hint: Подсказка о содержимом
            context: Дополнительный контекст
            force: Перезаписать существующий файл
            
        Returns:
            Результат создания
        \"""
        import time
        start_time = time.time()
        
        # Проверить, существует ли файл
        if file_path.exists() and not force:
            return {
                "status": "exists",
                "path": str(file_path),
                "size_kb": file_path.stat().st_size / 1024,
                "message": "Файл уже существует"
            }
        
        # Создать план
        plan = await self.plan_file(file_path, purpose, content_hint, context)
        
        # Проверить зависимости
        missing_deps = []
        for dep in plan.dependencies:
            if not dep.exists():
                missing_deps.append(str(dep))
        
        if missing_deps:
            log.warning(f"Отсутствуют зависимости: {missing_deps}")
        
        # Создать файл
        content = await self._generate_content(plan, content_hint, context)
        
        # Сохранить файл
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        
        # Обновить статистику
        file_size_kb = len(content.encode('utf-8')) / 1024
        processing_time = time.time() - start_time
        
        self.stats["files_created"] += 1
        self.stats["total_size_kb"] += file_size_kb
        
        if file_size_kb > 0:
            self.stats["avg_time_per_kb"] = (
                self.stats["avg_time_per_kb"] * (self.stats["files_created"] - 1) +
                processing_time / file_size_kb
            ) / self.stats["files_created"]
        
        # Записать в историю
        self.build_history.append({
            "timestamp": time.time(),
            "path": str(file_path),
            "size_kb": file_size_kb,
            "processing_time": processing_time,
            "plan": {
                "purpose": plan.purpose,
                "complexity": plan.complexity,
                "priority": plan.priority,
                "structure": plan.structure
            }
        })
        
        return {
            "status": "created",
            "path": str(file_path),
            "size_kb": file_size_kb,
            "processing_time_seconds": processing_time,
            "plan": {
                "purpose": plan.purpose,
                "complexity": plan.complexity,
                "priority": plan.priority,
                "structure": plan.structure,
                "estimated_size_kb": plan.estimated_size_kb,
                "dependencies": [str(dep) for dep in plan.dependencies]
            },
            "missing_dependencies": missing_deps
        }
    
    async def _generate_content(
        self,
        plan: FilePlan,
        content_hint: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        \"""Сгенерировать содержимое файла на основе плана.\"""
        # Определить тип файла
        ext = plan.path.suffix.lower()
        
        # Получить шаблон для типа файла
        template = self._get_template(ext, plan.purpose)
        
        # Заполнить шаблон
        content = template
        
        # Добавить содержимое на основе подсказки
        if content_hint:
            content += f"\n\n# Основное содержимое\n{content_hint}\n"
        
        # Добавить стандартные секции на основе структуры
        for section in plan.structure:
            section_content = self._generate_section(section, plan, context)
            if section_content:
                content += f"\n{section_content}\n"
        
        return content
    
    def _get_template(self, extension: str, purpose: str) -> str:
        \"""Получить шаблон для типа файла.\"""
        template_key = f"{extension}_{purpose}"
        
        if template_key in self.templates:
            self.stats["cache_hits"] += 1
            return self.templates[template_key]
        
        self.stats["cache_misses"] += 1
        
        # Генерация шаблона
        if extension == ".py":
            template = self._generate_python_template(purpose)
        elif extension == ".md":
            template = self._generate_markdown_template(purpose)
        elif extension == ".json":
            template = self._generate_json_template(purpose)
        else:
            template = self._generate_generic_template(purpose)
        
        # Кэшировать шаблон
        self.templates[template_key] = template
        
        # Сохранить в файл кэша
        cache_file = self.cache_dir / f"template_{hashlib.md5(template_key.encode()).hexdigest()[:8]}.txt"
        cache_file.write_text(template, encoding='utf-8')
        
        return template
    
    def _generate_python_template(self, purpose: str) -> str:
        \"""Сгенерировать шаблон Python файла.\"""
        purpose_lower = purpose.lower()
        
        if "module" in purpose_lower or "core" in purpose_lower:
            return f'''\"""
{purpose.title()}

Основной модуль системы.
\"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

log = logging.getLogger(__name__)


class {self._camel_case(purpose)}:
    \"""
    {purpose.title()}
    
    Класс для ...
    \"""
    
    def __init__(self):
        pass
    
    async def process(self):
        \"""Обработать ...\"""
        pass


async def main():
    \"""Основная функция.\"""
    pass


if __name__ == "__main__":
    asyncio.run(main())
'''
        
        elif "util" in purpose_lower or "helper" in purpose_lower:
            return f'''\"""
Утилиты для {purpose}.

Вспомогательные функции и инструменты.
\"""

import os
import sys
from pathlib import Path
from typing import Optional


def {self._snake_case(purpose)}(*args, **kwargs):
    \"""
    {purpose.title()}
    
    Args:
        *args: Аргументы
        **kwargs: Ключевые аргументы
        
    Returns:
        Результат
    \"""
    pass


# Константы
CONSTANT = "value"
'''
        
        else:
            return f'''\"""
{purpose.title()}
\"""

def main():
    \"""Основная функция.\"""
    pass


if __name__ == "__main__":
    main()
'''
    
    def _generate_markdown_template(self, purpose: str) -> str:
        \"""Сгенерировать шаблон Markdown файла.\"""
        return f'''# {purpose.title()}

## Описание

Здесь находится описание файла.

## Содержание

1. [Введение](#введение)
2. [Использование](#использование)
3. [Примеры](#примеры)
4. [API](#api)
5. [Примечания](#примечания)

## Введение

Введение в тему.

## Использование

Как использовать этот файл.

## Примеры

Примеры использования.

## API

Описание API (если применимо).

## Примечания

Дополнительные замечания.
'''
    
    def _generate_json_template(self, purpose: str) -> str:
        \"""Сгенерировать шаблон JSON файла.\"""
        return '''{
  "name": "example",
  "version": "1.0.0",
  "description": "JSON configuration file",
  "settings": {
    "enabled": true,
    "mode": "default"
  },
  "parameters": {},
  "metadata": {}
}
'''
    
    def _generate_generic_template(self, purpose: str) -> str:
        \"""Сгенерировать общий шаблон.\"""
        return f'''# {purpose.title()}

{self._get_date_header()}

## Описание

{purpose}

## Содержимое

Тут будет содержимое файла.
'''
    
    def _generate_section(
        self,
        section: str,
        plan: FilePlan,
        context: Optional[Dict[str, Any]]
    ) -> str:
        \"""Сгенерировать содержимое секции.\"""
        if section == "header_docstring":
            return self._generate_header_docstring(plan.purpose)
        elif section == "imports":
            return self._generate_imports_section()
        elif section == "classes":
            return self._generate_classes_section(plan.purpose)
        elif section == "functions":
            return self._generate_functions_section()
        elif section == "constants":
            return self._generate_constants_section()
        else:
            return f"# {section.replace('_', ' ').title()}\n\nСодержимое секции."
    
    def _generate_header_docstring(self, purpose: str) -> str:
        return f'"""\n{purpose.title()}\n\nСоздано FileBuilder.\n"""'
    
    def _generate_imports_section(self) -> str:
        return '''import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass'''
    
    def _generate_classes_section(self, purpose: str) -> str:
        class_name = self._camel_case(purpose)
        return f'''class {class_name}:
    """Класс для {purpose}."""
    
    def __init__(self):
        pass
    
    async def process(self):
        """Обработать задачу."""
        pass'''
    
    def _generate_functions_section(self) -> str:
        return '''def helper_function():
    """Вспомогательная функция."""
    pass

async def async_helper():
    """Асинхронная вспомогательная функция."""
    pass'''
    
    def _generate_constants_section(self) -> str:
        return '''# Константы
DEFAULT_CONFIG = {
    "enabled": True,
    "mode": "auto"
}

# Типы
from typing import TypeVar, Generic
T = TypeVar('T')'''
    
    def _camel_case(self, text: str) -> str:
        \"""Преобразовать текст в CamelCase.\"""
        return ''.join(word.capitalize() for word in text.replace('_', ' ').split())
    
    def _snake_case(self, text: str) -> str:
        \"""Преобразовать текст в snake_case.\"""
        return text.lower().replace(' ', '_')
    
    def _get_date_header(self) -> str:
        \"""Получить заголовок с датой.\"""
        from datetime import datetime
        return f"Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    async def create_large_file(
        self,
        file_path: Path,
        purpose: str,
        content_hint: str = "",
        max_chunk_size_kb: int = 50,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        \"""
        Создать большой файл через разделение на чанки.
        
        Args:
            file_path: Путь к файлу
            purpose: Назначение файла
            content_hint: Подсказка о содержимом
            max_chunk_size_kb: Максимальный размер чанка в КБ
            context: Дополнительный контекст
            
        Returns:
            Результат создания
        \"""
        import time
        start_time = time.time()
        
        # Создать план
        plan = await self.plan_file(file_path, purpose, content_hint, context)
        
        # Оценить необходимость разделения
        if plan.estimated_size_kb <= max_chunk_size_kb:
            # Файл достаточно мал - создавать целиком
            return await self.create_file(file_path, purpose, content_hint, context)
        
        # Разделить на чанки
        chunks = self._split_plan_into_chunks(plan, max_chunk_size_kb)
        
        log.info(f"Создание файла {file_path} через {len(chunks)} чанков")
        
        # Создать чанки параллельно
        chunk_contents = []
        tasks = []
        
        for chunk in chunks:
            task = self._generate_chunk_content(chunk, plan, context)
            tasks.append(task)
        
        # Запустить все задачи параллельно
        if tasks:
            chunk_contents = await asyncio.gather(*tasks)
        
        # Объединить чанки
        full_content = self._merge_chunks(chunks, chunk_contents)
        
        # Сохранить файл
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(full_content, encoding='utf-8')
        
        # Обновить статистику
        file_size_kb = len(full_content.encode('utf-8')) / 1024
        processing_time = time.time() - start_time
        
        self.stats["files_created"] += 1
        self.stats["total_size_kb"] += file_size_kb
        
        return {
            "status": "created_chunked",
            "path": str(file_path),
            "size_kb": file_size_kb,
            "processing_time_seconds": processing_time,
            "chunks_created": len(chunks),
            "plan": {
                "purpose": plan.purpose,
                "complexity": plan.complexity,
                "estimated_size_kb": plan.estimated_size_kb,
                "actual_size_kb": file_size_kb
            }
        }
    
    def _split_plan_into_chunks(
        self,
        plan: FilePlan,
        max_chunk_size_kb: int
    ) -> List[FileChunk]:
        \"""Разделить план файла на чанки.\"""
        chunks = []
        
        # Оценить размер каждой секции
        section_sizes = {}
        for section in plan.structure:
            # Простая эвристика для оценки размера
            if section in ["header_docstring", "imports", "constants"]:
                section_sizes[section] = 2  # КБ
            elif section in ["classes", "functions"]:
                section_sizes[section] = 10  # КБ
            else:
                section_sizes[section] = 5  # КБ
        
        # Группировать секции в чанки
        current_chunk_size = 0
        current_chunk_sections = []
        chunk_id = 1
        
        for section in plan.structure:
            section_size = section_sizes.get(section, 5)
            
            if current_chunk_size + section_size > max_chunk_size_kb and current_chunk_sections:
                # Создать чанк из текущих секций
                chunk = FileChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    file_path=plan.path,
                    start_line=0,  # Будет вычислено позже
                    end_line=0,    # Будет вычислено позже
                    content="",
                    dependencies=[]
                )
                chunks.append(chunk)
                
                current_chunk_size = 0
                current_chunk_sections = []
                chunk_id += 1
            
            current_chunk_sections.append(section)
            current_chunk_size += section_size
        
        # Добавить последний чанк
        if current_chunk_sections:
            chunk = FileChunk(
                chunk_id=f"chunk_{chunk_id}",
                file_path=plan.path,
                start_line=0,
                end_line=0,
                content="",
                dependencies=[]
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_chunk_content(
        self,
        chunk: FileChunk,
        plan: FilePlan,
        context: Optional[Dict[str, Any]]
    ) -> str:
        \"""Сгенерировать содержимое чанка.\"""
        # Для демо просто создаём заглушку
        # В реальной реализации здесь будет вызов модели для генерации
        return f"# Чанк {chunk.chunk_id}\n\nСодержимое чанка для файла {plan.path.name}.\n"
    
    def _merge_chunks(
        self,
        chunks: List[FileChunk],
        chunk_contents: List[str]
    ) -> str:
        \"""Объединить чанки в единое содержимое.\"""
        # Простое объединение
        return "\n\n".join(chunk_contents)
    
    def get_stats(self) -> Dict[str, Any]:
        \"""Получить статистику конструктора.\"""
        return {
            "files_created": self.stats["files_created"],
            "total_size_kb": self.stats["total_size_kb"],
            "avg_time_per_kb": self.stats["avg_time_per_kb"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_efficiency": (
                self.stats["cache_hits"] / 
                max(self.stats["cache_hits"] + self.stats["cache_misses"], 1)
            ),
            "build_history_count": len(self.build_history),
            "templates_count": len(self.templates)
        }
    
    async def clear_cache(self):
        \"""Очистить кэш шаблонов.\"""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates.clear()
        log.info("Кэш конструктора файлов очищен")


# Глобальный экземпляр
_file_builder_instance = None

def get_file_builder(cache_dir: Optional[Path] = None) -> FileBuilder:
    \"""
    Получить глобальный экземпляр конструктора файлов.
    
    Args:
        cache_dir: Директория для кэша
        
    Returns:
        Экземпляр FileBuilder
    \"""
    global _file_builder_instance
    
    if _file_builder_instance is None:
        _file_builder_instance = FileBuilder(cache_dir)
    
    return _file_builder_instance


# Демонстрационная функция
async def demo_file_builder():
    \"""Демонстрация работы конструктора файлов.\"""
    import tempfile
    
    print("Демонстрация интеллектуального конструктора файлов")
    print("=" * 60)
    
    # Создать временную директорию
    temp_dir = Path(tempfile.mkdtemp(prefix="ouroboros_file_builder_"))
    print(f"Временная директория: {temp_dir}")
    print()
    
    # Создать конструктор
    builder = FileBuilder()
    
    # Тест 1: Создать простой Python файл
    print("Тест 1: Создание простого Python файла")
    test_file1 = temp_dir / "simple_module.py"
    result1 = await builder.create_file(
        file_path=test_file1,
        purpose="Простой модуль для демонстрации",
        content_hint="Модуль содержит базовые функции и классы."
    )
    print(f"  Статус: {result1['status']}")
    print(f"  Путь: {result1['path']}")
    print(f"  Размер: {result1['size_kb']:.1f} КБ")
    print(f"  Сложность: {result1['plan']['complexity']}")
    print()
    
    # Тест 2: Создать Markdown файл
    print("Тест 2: Создание документации")
    test_file2 = temp_dir / "README.md"
    result2 = await builder.create_file(
        file_path=test_file2,
        purpose="Документация проекта",
        content_hint="Полное описание проекта и инструкции."
    )
    print(f"  Статус: {result2['status']}")
    print(f"  Путь: {result2['path']}")
    print(f"  Размер: {result2['size_kb']:.1f} КБ")
    print()
    
    # Тест 3: Создать JSON конфигурацию
    print("Тест 3: Создание конфигурационного файла")
    test_file3 = temp_dir / "config.json"
    result3 = await builder.create_file(
        file_path=test_file3,
        purpose="Конфигурация системы",
        content_hint='{"enabled": true, "mode": "auto"}'
    )
    print(f"  Статус: {result3['status']}")
    print(f"  Путь: {result3['path']}")
    print(f"  Размер: {result3['size_kb']:.1f} КБ")
    print()
    
    # Тест 4: Создать большой файл через чанки
    print("Тест 4: Создание большого файла через чанки")
    test_file4 = temp_dir / "large_module.py"
    result4 = await builder.create_large_file(
        file_path=test_file4,
        purpose="Большой модуль с множеством функций",
        content_hint="Модуль содержит много классов, функций и констант.",
        max_chunk_size_kb=10  # Искусственно маленький для демонстрации
    )
    print(f"  Статус: {result4['status']}")
    print(f"  Путь: {result4['path']}")
    print(f"  Размер: {result4['size_kb']:.1f} КБ")
    print(f"  Чанков создано: {result4.get('chunks_created', 1)}")
    print()
    
    # Статистика
    print("Статистика конструктора:")
    stats = builder.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Очистка
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("Демонстрация завершена!")


if __name__ == "__main__":
    # Запустить демонстрацию
    asyncio.run(demo_file_builder())
'''
