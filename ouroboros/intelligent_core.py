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
from dataclasses import dataclass, field, asdict
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
    
    def __init__(self, local_core: Any = None):
        self.local_core = local_core
        
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
            "статус": "Я работаю нормально. Все системы функционируют.",
            "как дела": "Всё хорошо, работаю над улучшением архитектуры.",
            "привет": "Привет! Чем могу помочь?",
            "версия": "Текущая версия системы указана в VERSION файле.",
            "бюджет": "Бюджет отслеживается автоматически.",
        }
    
    async def analyze_task(self, task: str, context: Optional[Dict] = None) -> TaskComplexity:
        """Проанализировать задачу на сложность и предложить подход."""
        task_hash = hashlib.md5(task.encode()).hexdigest()[:16]
        words = task.split()
        word_count = len(words)
        
        complexity_keywords = ["создать", "реализовать", "архитектура", "рефакторить", "интегрировать"]
        simple_keywords = ["статус", "привет", "прочитать", "найти"]
        
        complexity_score = 0
        for keyword in complexity_keywords:
            if keyword in task.lower(): complexity_score += 3
        for keyword in simple_keywords:
            if keyword in task.lower(): complexity_score -= 1
            
        if word_count > 50: complexity_score += 2
        
        if complexity_score <= 1:
            approach = "immediate"
        elif complexity_score <= 4:
            approach = "planning"
        elif complexity_score <= 7:
            approach = "cloud"
        else:
            approach = "learn"
            
        return TaskComplexity(
            task_hash=task_hash,
            estimated_tokens=word_count * 5,
            estimated_time_ms=word_count * 20,
            can_be_cached=complexity_score < 5,
            suggested_approach=approach,
        )
    
    async def process_task(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Основной вход для обработки задачи через интеллектуальное ядро."""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # 1. Хэш и эвристики
        immediate = self._check_heuristics(task)
        if immediate:
            return self._format_response(immediate, "heuristic", int((time.time()-start_time)*1000), "simple")

        # 2. Кэш паттернов
        cached = self._check_pattern_cache(task, context)
        if cached:
            self.stats["pattern_hits"] += 1
            return self._format_response(cached, "pattern_cache", int((time.time()-start_time)*1000), "cached")

        # 3. Сложность и маршрутизация
        complexity = await self.analyze_task(task, context)
        
        # Решение о маршруте
        if complexity.suggested_approach == "immediate" and self.local_core:
            res = await self.local_core.generate(task, context)
            source = "local_core"
        elif complexity.suggested_approach == "cloud":
            return {"status": "delegate_to_cloud", "complexity": complexity}
        else:
            # Для планирования или обучения пока тоже облако, но с пометкой
            return {"status": "delegate_to_cloud", "complexity": complexity, "action": complexity.suggested_approach}

        response_text = res.get("text", "Не удалось сгенерировать ответ.")
        
        # Сохранение в кэш если успешно
        if complexity.can_be_cached and response_text:
            self._update_pattern_cache(task, context, response_text)

        processing_time_ms = int((time.time() - start_time) * 1000)
        return self._format_response(response_text, source, processing_time_ms, complexity.suggested_approach)

    def _check_heuristics(self, task: str) -> Optional[str]:
        task_lower = task.lower().strip()
        for p, a in self.heuristics.items():
            if p in task_lower: return a
        return None

    def _check_pattern_cache(self, task: str, context: Optional[Dict]) -> Optional[str]:
        task_key = task.lower().strip()
        pattern_hash = hashlib.md5(task_key.encode()).hexdigest()[:12]
        if pattern_hash in self.pattern_cache:
            p = self.pattern_cache[pattern_hash]
            p.usage_count += 1
            p.last_used = time.time()
            return p.answer_template
        return None

    def _update_pattern_cache(self, task: str, context: Optional[Dict], answer: str):
        task_key = task.lower().strip()
        pattern_hash = hashlib.md5(task_key.encode()).hexdigest()[:12]
        if len(self.pattern_cache) >= self.pattern_limit:
            oldest = min(self.pattern_cache.keys(), key=lambda k: self.pattern_cache[k].last_used)
            del self.pattern_cache[oldest]
        
        self.pattern_cache[pattern_hash] = PatternCache(
            pattern_hash=pattern_hash,
            question_pattern=task_key[:100],
            answer_template=answer,
            confidence=0.8,
            usage_count=1,
            last_used=time.time()
        )

    def _format_response(self, text: str, source: str, time_ms: int, complexity: str) -> Dict[str, Any]:
        return {
            "text": text,
            "source": source,
            "processing_time_ms": time_ms,
            "complexity": complexity,
            "success": True
        }
