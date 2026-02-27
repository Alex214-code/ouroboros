"""
Интеллектуальное ядро для адаптивного роста.

Вместо того чтобы просто запускать модель, это ядро:
1. **Планирует заранее** — генерирует ответы на предсказуемые вопросы
2. **Оптимизирует мышление** — использует быстрые эвристики для простых задач
3. **Кэширует паттерны** — запоминает уже решённые задачи
4. **Учится на взаимодействиях** — становится умнее с каждым диалогом
5. **Распределяет нагрузку** — отправляет сложные задачи в облако

Ключевой принцип: качественный рост интеллекта ≠ количественный рост параметров.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

log = logging.getLogger(__name__)


@dataclass
class PatternCache:
    """Кэш паттернов для быстрого ответа."""
    pattern_hash: str
    question_pattern: str
    answer_template: str
    confidence: float  # 0.0-1.0
    usage_count: int = 0
    last_used: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class TaskComplexity:
    """Оценка сложности задачи."""
    task_hash: str
    estimated_tokens: int
    estimated_time_ms: int
    can_be_cached: bool
    suggested_approach: str  # "immediate", "planning", "cloud", "learn"


class IntelligentCore:
    """
    Интеллектуальное ядро с адаптивным ростом.
    
    Вместо линейного увеличения параметров модели:
    - Оптимизация через предвычисления
    - Кэширование паттернов мышления
    - Адаптивная маршрутизация задач
    - Постоянное обучение на взаимодействиях
    """
    
    def __init__(self, local_core_path: str):
        self.local_core_path = local_core_path
        
        # Паттерн-кэш
        self.pattern_cache: Dict[str, PatternCache] = {}
        self.pattern_limit = 1000
        
        # История решений
        self.solution_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Статистика
        self.stats = {
            "total_requests": 0,
            "pattern_hits": 0,
            "pattern_misses": 0,
            "local_answers": 0,
            "cloud_delegations": 0,
            "precomputed_answers": 0,
            "learning_cycles": 0,
        }
        
        # Эвристики для быстрого ответа
        self.heuristics = self._load_heuristics()
        
        # Сложность задач
        self.complexity_threshold = 50  # токенов, выше - в облако
        
        log.info("Инициализирован IntelligentCore с адаптивным ростом")
    
    def _load_heuristics(self) -> Dict[str, str]:
        """Загрузить базовые эвристики для быстрых ответов."""
        return {
            # Системные вопросы
            "статус": "Я работаю нормально. Все системы функционируют.",
            "как дела": "Всё хорошо, работаю над улучшением архитектуры.",
            "привет": "Привет! Чем могу помочь?",
            "здравствуй": "Здравствуйте! Готов к работе.",
            
            # Технические вопросы
            "версия": "Текущая версия системы указана в VERSION файле.",
            "бюджет": "Бюджет отслеживается автоматически.",
            "модель": "Используется нейросимбиотическая архитектура.",
            
            # Файловые операции
            "создать файл": "Готов создать файл. Укажите путь и содержимое.",
            "прочитать файл": "Готов прочитать файл. Укажите путь.",
            "записать файл": "Готов записать в файл. Укажите путь и содержимое.",
            
            # Git операции
            "коммит": "Сначала нужно изменить файлы, затем выполнить коммит.",
            "push": "После коммита можно выполнить push в ветку ouroboros.",
            "статус гита": "Используйте git_status для проверки статуса.",
        }
    
    async def analyze_task(self, task: str, context: Optional[Dict] = None) -> TaskComplexity:
        """
        Проанализировать задачу на сложность и предложить подход.
        
        Args:
            task: Текст задачи
            context: Дополнительный контекст
            
        Returns:
            TaskComplexity с оценкой и рекомендациями
        """
        task_hash = hashlib.md5(task.encode()).hexdigest()[:16]
        
        # Простая эвристика для оценки сложности
        words = task.split()
        word_count = len(words)
        
        # Ключевые слова, указывающие на сложность
        complexity_keywords = [
            "создать", "реализовать", "спроектировать", "разработать",
            "оптимизировать", "рефакторить", "проанализировать",
            "интегрировать", "архитектура", "алгоритм", "система"
        ]
        
        simple_keywords = [
            "статус", "информация", "список", "показать",
            "прочитать", "проверить", "найти", "поиск"
        ]
        
        # Оценка сложности
        complexity_score = 0
        for keyword in complexity_keywords:
            if keyword in task.lower():
                complexity_score += 3
        
        for keyword in simple_keywords:
            if keyword in task.lower():
                complexity_score -= 1
        
        # Учёт длины
        if word_count > 50:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1
        
        # Определить подход
        if complexity_score <= 1:
            approach = "immediate"  # Немедленный ответ
            estimated_tokens = min(50, word_count * 4)
            estimated_time = max(100, word_count * 5)  # мс
            cacheable = True
        elif complexity_score <= 3:
            approach = "planning"  # Требует планирования
            estimated_tokens = min(200, word_count * 6)
            estimated_time = max(500, word_count * 15)
            cacheable = True
        elif complexity_score <= 6:
            approach = "cloud"  # Требует облачного вычисления
            estimated_tokens = min(1000, word_count * 10)
            estimated_time = max(2000, word_count * 30)
            cacheable = False
        else:
            approach = "learn"  # Сложная задача, требующая обучения
            estimated_tokens = word_count * 15
            estimated_time = word_count * 50
            cacheable = False
        
        return TaskComplexity(
            task_hash=task_hash,
            estimated_tokens=estimated_tokens,
            estimated_time_ms=estimated_time,
            can_be_cached=cacheable,
            suggested_approach=approach,
        )
    
    async def generate_response(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        use_patterns: bool = True,
        learn_from_result: bool = True
    ) -> Dict[str, Any]:
        """
        Сгенерировать интеллектуальный ответ на задачу.
        
        Args:
            task: Текст задачи/вопроса
            context: Дополнительный контекст
            use_patterns: Использовать кэш паттернов
            learn_from_result: Учиться на результате
            
        Returns:
            Ответ с метаданными
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # 1. Проверить простые эвристики
        immediate_answer = self._check_heuristics(task)
        if immediate_answer:
            return self._format_response(
                text=immediate_answer,
                source="heuristic",
                processing_time_ms=int((time.time() - start_time) * 1000),
                complexity="simple",
            )
        
        # 2. Проверить паттерн-кэш
        if use_patterns:
            pattern_match = self._check_pattern_cache(task, context)
            if pattern_match:
                self.stats["pattern_hits"] += 1
                return self._format_response(
                    text=pattern_match,
                    source="pattern_cache",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    complexity="cached",
                )
        
        # 3. Проанализировать сложность задачи
        complexity = await self.analyze_task(task, context)
        
        # 4. Выбрать подход на основе сложности
        if complexity.suggested_approach == "immediate":
            # Немедленный ответ через локальное ядро
            response = await self._immediate_response(task, context)
            source = "local_core"
        
        elif complexity.suggested_approach == "planning":
            # Ответ с планированием
            response = await self._planned_response(task, context)
            source = "planned_local"
        
        elif complexity.suggested_approach == "cloud":
            # Делегировать в облако
            response = await self._cloud_response(task, context)
            source = "cloud"
            self.stats["cloud_delegations"] += 1
        
        elif complexity.suggested_approach == "learn":
            # Сложная задача - требует обучения
            response = await self._learning_response(task, context)
            source = "learning"
        
        else:
            # Fallback на локальное ядро
            response = await self._immediate_response(task, context)
            source = "fallback"
        
        # 5. Обновить паттерн-кэш если задача кэшируемая
        if learn_from_result and complexity.can_be_cached:
            self._update_pattern_cache(task, context, response["text"])
        
        # 6. Записать в историю решений
        self._record_solution(task, complexity, response)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return self._format_response(
            text=response["text"],
            source=source,
            processing_time_ms=processing_time_ms,
            complexity=complexity.suggested_approach,
            estimated_tokens=complexity.estimated_tokens,
            metadata={
                "task_hash": complexity.task_hash,
                "approach": complexity.suggested_approach,
                "cacheable": complexity.can_be_cached,
            }
        )
    
    def _check_heuristics(self, task: str) -> Optional[str]:
        """Проверить простые эвристики."""
        task_lower = task.lower().strip()
        
        # Проверить точные совпадения
        if task_lower in self.heuristics:
            return self.heuristics[task_lower]
        
        # Проверить частичные совпадения
        for pattern, answer in self.heuristics.items():
            if pattern in task_lower:
                return answer
        
        return None
    
    def _check_pattern_cache(self, task: str, context: Optional[Dict]) -> Optional[str]:
        """Проверить кэш паттернов."""
        # Создать ключ паттерна
        task_key = task.lower().strip()
        if context:
            context_str = json.dumps(context, sort_keys=True)
            task_key += "_" + hashlib.md5(context_str.encode()).hexdigest()[:8]
        
        pattern_hash = hashlib.md5(task_key.encode()).hexdigest()[:12]
        
        if pattern_hash in self.pattern_cache:
            pattern = self.pattern_cache[pattern_hash]
            pattern.usage_count += 1
            pattern.last_used = time.time()
            
            # Вернуть шаблон ответа (можно параметризовать)
            return pattern.answer_template
        
        return None
    
    async def _immediate_response(self, task: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Немедленный ответ через локальное ядро."""
        # Здесь будет интеграция с LocalCore
        # Пока заглушка
        return {
            "text": f"Ответ на задачу: {task[:100]}...",
            "success": True,
        }
    
    async def _planned_response(self, task: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Ответ с планированием."""
        # Здесь будет более сложная логика планирования
        # Пока заглушка
        return {
            "text": f"Планирую решение для: {task[:100]}...",
            "success": True,
        }
    
    async def _cloud_response(self, task: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Делегировать задачу в облако."""
        # Здесь будет интеграция с облачной моделью
        # Пока заглушка
        return {
            "text": f"Делегирую сложную задачу в облако: {task[:100]}...",
            "success": True,
        }
    
    async def _learning_response(self, task: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Ответ с обучением."""
        self.stats["learning_cycles"] += 1
        
        # Здесь будет логика обучения на задаче
        # Пока заглушка
        return {
            "text": f"Изучаю сложную задачу: {task[:100]}...",
            "success": True,
        }
    
    def _update_pattern_cache(self, task: str, context: Optional[Dict], answer: str):
        """Обновить кэш паттернов новым решением."""
        task_key = task.lower().strip()
        if context:
            context_str = json.dumps(context, sort_keys=True)
            task_key += "_" + hashlib.md5(context_str.encode()).hexdigest()[:8]
        
        pattern_hash = hashlib.md5(task_key.encode()).hexdigest()[:12]
        
        # Создать шаблон ответа
        # TODO: Извлечь паттерн из ответа
        answer_template = answer
        
        # Сохранить в кэш
        if len(self.pattern_cache) >= self.pattern_limit:
            # Найти наименее используемый паттерн
            oldest_key = min(
                self.pattern_cache.keys(),
                key=lambda k: self.pattern_cache[k].last_used
            )
            del self.pattern_cache[oldest_key]
        
        self.pattern_cache[pattern_hash] = PatternCache(
            pattern_hash=pattern_hash,
            question_pattern=task_key[:100],  # Усечённый паттерн
            answer_template=answer_template,
            confidence=0.7,  # Начальная уверенность
            usage_count=1,
            last_used=time.time(),
        )
        
        log.debug(f"Добавлен паттерн в кэш: {pattern_hash}")
    
    def _record_solution(self, task: str, complexity: TaskComplexity, response: Dict[str, Any]):
        """Записать решение в историю."""
        task_hash = complexity.task_hash
        
        self.solution_history[task_hash].append({
            "timestamp": time.time(),
            "task": task[:200],  # Усечённая задача
            "complexity": complexity.suggested_approach,
            "response_source": response.get("source", "unknown"),
            "success": response.get("success", False),
            "processing_time_ms": response.get("processing_time_ms", 0),
        })
        
        # Ограничить историю
        if len(self.solution_history[task_hash]) > 10:
            self.solution_history[task_hash] = self.solution_history[task_hash][-10:]
    
    def _format_response(
        self,
        text: str,
        source: str,
        processing_time_ms: int,
        complexity: str,
        estimated_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Форматировать ответ."""
        result = {
            "text": text,
            "source": source,
            "processing_time_ms": processing_time_ms,
            "complexity": complexity,
            "estimated_tokens": estimated_tokens,
            "success": True,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        
        # Логировать в зависимости от сложности
        if complexity in ["cached", "simple"]:
            log.debug(f"Быстрый ответ ({source}): {processing_time_ms}мс")
        elif complexity in ["planned_local", "cloud"]:
            log.info(f"Ответ средней сложности ({source}): {processing_time_ms}мс")
        else:
            log.warning(f"Сложный ответ ({source}): {processing_time_ms}мс")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            **self.stats,
            "pattern_cache_size": len(self.pattern_cache),
            "pattern_hit_rate": (
                self.stats["pattern_hits"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            ),
            "unique_tasks_in_history": len(self.solution_history),
            "heuristics_count": len(self.heuristics),
        }
    
    def learn_from_interaction(self, task: str, ideal_answer: str):
        """
        Обучение на взаимодействии.
        
        Args:
            task: Задача/вопрос
            ideal_answer: Идеальный/желаемый ответ
        """
        # Извлечь паттерн из задачи
        task_key = task.lower().strip()
        
        # Добавить в эвристики если задача простая
        if len(task.split()) < 10 and task_key not in self.heuristics:
            self.heuristics[task_key] = ideal_answer
            log.info(f"Добавлена новая эвристика: {task_key[:50]}...")
        
        # Обновить паттерн-кэш
        self._update_pattern_cache(task, None, ideal_answer)
    
    def optimize_thinking(self):
        """Оптимизировать процесс мышления."""
        # 1. Очистить старые редко используемые паттерны
        now = time.time()
        old_patterns = [
            key for key, pattern in self.pattern_cache.items()
            if now - pattern.last_used > 7 * 24 * 3600  # Старше недели
        ]
        
        for key in old_patterns:
            del self.pattern_cache[key]
        
        if old_patterns:
            log.info(f"Очищено {len(old_patterns)} устаревших паттернов")
        
        # 2. Повысить уверенность в часто используемых паттернах
        for pattern in self.pattern_cache.values():
            if pattern.usage_count > 10:
                pattern.confidence = min(1.0, pattern.confidence + 0.05)
        
        # 3. Удалить наименее полезные эвристики
        # TODO: Анализ эффективности эвристик
        
        log.debug("Оптимизация мышления завершена")
    
    async def precompute_responses(self, common_tasks: List[str]):
        """
        Предвычислить ответы на частые задачи.
        
        Args:
            common_tasks: Список частых задач/вопросов
        """
        for task in common_tasks:
            # Проанализировать задачу
            complexity = await self.analyze_task(task)
            
            # Если задача простая - предвычислить ответ
            if complexity.suggested_approach == "immediate":
                response = await self._immediate_response(task, None)
                self._update_pattern_cache(task, None, response["text"])
                self.stats["precomputed_answers"] += 1
        
        log.info(f"Предвычислено {len(common_tasks)} ответов на частые задачи")
    
    def export_knowledge(self, filepath: Path):
        """Экспортировать накопленные знания."""
        knowledge = {
            "heuristics": self.heuristics,
            "patterns": [
                {
                    "question": p.question_pattern,
                    "answer": p.answer_template,
                    "confidence": p.confidence,
                    "usage_count": p.usage_count,
                }
                for p in self.pattern_cache.values()
            ],
            "stats": self.stats,
            "exported_at": time.time(),
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(knowledge, indent=2, ensure_ascii=False), encoding='utf-8')
        
        log.info(f"Знания экспортированы в {filepath}")
    
    def import_knowledge(self, filepath: Path):
        """Импортировать знания из файла."""
        if not filepath.exists():
            log.warning(f"Файл знаний не найден: {filepath}")
            return
        
        try:
            data = json.loads(filepath.read_text(encoding='utf-8'))
            
            # Импортировать эвристики
            if "heuristics" in data:
                self.heuristics.update(data["heuristics"])
            
            # Импортировать паттерны
            if "patterns" in data:
                for pattern_data in data["patterns"]:
                    pattern_hash = hashlib.md5(pattern_data["question"].encode()).hexdigest()[:12]
                    
                    self.pattern_cache[pattern_hash] = PatternCache(
                        pattern_hash=pattern_hash,
                        question_pattern=pattern_data["question"],
                        answer_template=pattern_data["answer"],
                        confidence=pattern_data.get("confidence", 0.7),
                        usage_count=pattern_data.get("usage_count", 1),
                        last_used=time.time(),
                    )
            
            log.info(f"Знания импортированы из {filepath}")
            
        except Exception as e:
            log.error(f"Ошибка импорта знаний: {e}")


# Интеграция с существующей системой
class NeuroSymbioticProcessor:
    """
    Процессор нейросимбиотической архитектуры.
    
    Интегрирует IntelligentCore, LocalCore и облачные модели.
    """
    
    def __init__(self):
        self.intelligent_core = IntelligentCore(local_core_path="ollama")
        self.local_core = None  # Будет инициализирован позже
        self.cloud_core = None   # Будет инициализирован позже
        
        self.task_router = {
            "immediate": self._process_immediate,
            "planning": self._process_planning,
            "cloud": self._process_cloud,
            "learn": self._process_learn,
        }
        
        log.info("Инициализирован NeuroSymbioticProcessor")
    
    async def initialize(self):
        """Инициализировать все компоненты."""
        # Инициализировать IntelligentCore
        await self.intelligent_core.precompute_responses([
            "привет",
            "как дела",
            "статус",
            "помощь",
            "что ты умеешь",
        ])
        
        # TODO: Инициализировать LocalCore
        # TODO: Инициализировать CloudCore
        
        log.info("NeuroSymbioticProcessor инициализирован")
    
    async def process(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Обработать задачу через нейросимбиотическую архитектуру.
        
        Args:
            task: Задача/вопрос
            context: Дополнительный контекст
            
        Returns:
            Результат обработки
        """
        # 1. Анализ задачи
        complexity = await self.intelligent_core.analyze_task(task, context)
        
        # 2. Маршрутизация на основе сложности
        processor = self.task_router.get(
            complexity.suggested_approach,
            self._process_fallback
        )
        
        # 3. Обработка
        response = await processor(task, context, complexity)
        
        # 4. Обучение на результате
        if complexity.can_be_cached:
            self.intelligent_core.learn_from_interaction(task, response["text"])
        
        return response
    
    async def _process_immediate(self, task: str, context: Optional[Dict], complexity: TaskComplexity):
        """Обработка немедленных задач."""
        # Сначала проверить эвристики и кэш
        immediate = self.intelligent_core._check_heuristics(task)
        if immediate:
            return self.intelligent_core._format_response(
                text=immediate,
                source="heuristic",
                processing_time_ms=1,
                complexity="immediate",
            )
        
        # Затем локальное ядро
        # TODO: Вызвать LocalCore.generate()
        
        # Fallback
        return {
            "text": f"Быстрый ответ: {task[:100]}...",
            "success": True,
        }
    
    async def _process_planning(self, task: str, context: Optional[Dict], complexity: TaskComplexity):
        """Обработка задач, требующих планирования."""
        # TODO: Использовать LocalCore с планированием
        return {
            "text": f"Планирую решение: {task[:100]}...",
            "success": True,
        }
    
    async def _process_cloud(self, task: str, context: Optional[Dict], complexity: TaskComplexity):
        """Делегирование в облако."""
        # TODO: Использовать облачную модель
        return {
            "text": f"Делегирую в облако: {task[:100]}...",
            "success": True,
        }
    
    async def _process_learn(self, task: str, context: Optional[Dict], complexity: TaskComplexity):
        """Обработка сложных задач с обучением."""
        # TODO: Комплексная обработка с обучением
        return {
            "text": f"Изучаю сложную задачу: {task[:100]}...",
            "success": True,
        }
    
    async def _process_fallback(self, task: str, context: Optional[Dict], complexity: TaskComplexity):
        """Fallback обработчик."""
        return {
            "text": f"Fallback ответ: {task[:100]}...",
            "success": True,
        }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Получить детальную статистику."""
        intelligent_stats = self.intelligent_core.get_stats()
        
        return {
            "intelligent_core": intelligent_stats,
            "local_core_available": self.local_core is not None,
            "cloud_core_available": self.cloud_core is not None,
            "total_patterns": len(self.intelligent_core.pattern_cache),
            "total_heuristics": len(self.intelligent_core.heuristics),
        }