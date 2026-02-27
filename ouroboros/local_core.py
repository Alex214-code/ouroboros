"""
Локальное ядро для быстрого инференса на CPU.

Использует маленькие модели Ollama для молниеносных ответов.
Интегрируется с нейросимбиотической архитектурой как быстрый исполнитель.

Требования: Ollama с установленными маленькими моделями (smollm:1.7b, phi-3-mini, qwen3:8b)
"""

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib

log = logging.getLogger(__name__)


@dataclass
class LocalModelInfo:
    """Информация о локальной модели."""
    name: str
    size_gb: float
    context_window: int
    estimated_tokens_per_sec: int
    installed: bool = False
    available: bool = False
    avg_response_time: float = 0.0
    last_used: float = 0.0


class LocalCore:
    """
    Быстрое локальное ядро для рутинных задач.
    
    Особенности:
    - Автоматический выбор самой быстрой доступной модели
    - Кэширование частых запросов
    - Плавное падение на облако при недоступности локальных моделей
    - Мониторинг производительности
    """
    
    # Оптимизированные модели для CPU в порядке приоритета
    OPTIMAL_MODELS = [
        # Самые быстрые (1-3B параметров)
        LocalModelInfo("smollm:1.7b", 0.99, 2048, 30),
        LocalModelInfo("phi3:mini", 1.8, 4096, 25),
        LocalModelInfo("qwen2.5-coder:1.5b", 0.9, 32768, 20),
        LocalModelInfo("gemma2:2b", 1.5, 8192, 18),
        
        # Немного медленнее (4-8B параметров)
        LocalModelInfo("qwen3:8b", 4.7, 32768, 12),
        LocalModelInfo("llama3.2:3b", 1.8, 8192, 15),
        LocalModelInfo("deepseek-coder-v2-lite:1.3b", 0.8, 16384, 22),
    ]
    
    def __init__(self, ollama_path: str = "ollama"):
        self.ollama_path = ollama_path
        self.models: Dict[str, LocalModelInfo] = {}
        self.active_model: Optional[str] = None
        self.cache: Dict[str, str] = {}
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "local_responses": 0,
            "fallback_to_cloud": 0,
            "avg_response_time_ms": 0,
            "errors": 0,
        }
        
        # Кэш для ускорения повторяющихся запросов
        self.response_cache: Dict[str, str] = {}
        self.cache_size_limit = 1000
        
        log.info("Инициализирован LocalCore для быстрого CPU инференса")
    
    async def initialize(self) -> bool:
        """
        Инициализировать LocalCore: проверить доступные модели, измерить скорость.
        
        Returns:
            True если хотя бы одна модель доступна
        """
        try:
            # Проверить доступность Ollama
            result = await self._run_ollama_command(["list"])
            if result.returncode != 0:
                log.warning("Ollama не доступен, LocalCore будет использовать только облако")
                return False
            
            installed_models = self._parse_ollama_list(result.stdout)
            
            # Обновить информацию о моделях
            for model_info in self.OPTIMAL_MODELS:
                model_info.installed = model_info.name in installed_models
                model_info.available = model_info.installed
                self.models[model_info.name] = model_info
            
            # Если нет установленных моделей - попробовать установить самую лёгкую
            if not any(m.available for m in self.models.values()):
                log.info("Нет установленных локальных моделей. Попробую установить smollm:1.7b...")
                success = await self._install_model("smollm:1.7b")
                if success:
                    self.models["smollm:1.7b"].available = True
                    self.models["smollm:1.7b"].installed = True
            
            # Выбрать самую быструю доступную модель
            available_models = [
                m for m in self.models.values() 
                if m.available and m.estimated_tokens_per_sec > 0
            ]
            
            if not available_models:
                log.warning("Нет доступных локальных моделей")
                return False
            
            # Измерить реальную скорость для каждой доступной модели
            for model in available_models:
                speed = await self._benchmark_model(model.name)
                if speed > 0:
                    model.estimated_tokens_per_sec = speed
            
            # Выбрать модель с максимальной скоростью
            fastest_model = max(available_models, key=lambda m: m.estimated_tokens_per_sec)
            self.active_model = fastest_model.name
            
            log.info(f"Выбрана локальная модель: {self.active_model} "
                    f"({fastest_model.estimated_tokens_per_sec} токенов/сек)")
            
            return True
            
        except Exception as e:
            log.error(f"Ошибка инициализации LocalCore: {e}")
            return False
    
    async def _run_ollama_command(self, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Выполнить команду Ollama."""
        cmd = [self.ollama_path] + args
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout.decode('utf-8', errors='ignore'),
                stderr=stderr.decode('utf-8', errors='ignore'),
            )
        except asyncio.TimeoutError:
            log.error(f"Таймаут команды Ollama: {cmd}")
            raise
        except Exception as e:
            log.error(f"Ошибка выполнения команды Ollama: {e}")
            raise
    
    def _parse_ollama_list(self, output: str) -> List[str]:
        """Разобрать вывод 'ollama list'."""
        models = []
        lines = output.strip().split('\n')
        
        for line in lines[1:]:  # Пропустить заголовок
            if line.strip():
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    models.append(model_name)
        
        return models
    
    async def _install_model(self, model_name: str) -> bool:
        """Установить модель Ollama."""
        try:
            log.info(f"Установка модели {model_name}...")
            result = await self._run_ollama_command(["pull", model_name], timeout=300)
            
            if result.returncode == 0:
                log.info(f"Модель {model_name} успешно установлена")
                return True
            else:
                log.error(f"Ошибка установки модели {model_name}: {result.stderr}")
                return False
        except Exception as e:
            log.error(f"Исключение при установке модели {model_name}: {e}")
            return False
    
    async def _benchmark_model(self, model_name: str, test_prompt: str = "Hello, world!") -> int:
        """
        Замерить скорость модели в токенах/сек.
        
        Args:
            model_name: Имя модели
            test_prompt: Тестовый промпт
            
        Returns:
            Скорость в токенах/сек, или 0 если ошибка
        """
        try:
            start_time = time.time()
            
            # Простой тестовый запрос
            result = await self._run_ollama_command([
                "run", model_name, test_prompt
            ], timeout=15)
            
            if result.returncode != 0:
                return 0
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Оценить количество токенов (грубая оценка)
            # Средняя длина токена ~4 символа для английского
            response_length = len(result.stdout.strip())
            estimated_tokens = max(1, response_length // 4)
            
            tokens_per_sec = estimated_tokens / elapsed if elapsed > 0 else 0
            
            log.info(f"Бенчмарк {model_name}: {tokens_per_sec:.1f} токенов/сек "
                    f"({elapsed:.2f} сек на {estimated_tokens} токенов)")
            
            return int(tokens_per_sec)
            
        except Exception as e:
            log.error(f"Ошибка бенчмарка модели {model_name}: {e}")
            return 0
    
    def _get_cache_key(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Создать ключ кэша для запроса."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        
        if context:
            context_str = json.dumps(context, sort_keys=True)
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
            return f"{prompt_hash}_{context_hash}"
        
        return prompt_hash
    
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Сгенерировать ответ используя локальную модель.
        
        Args:
            prompt: Текст промпта
            context: Дополнительный контекст
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации
            use_cache: Использовать кэш
            
        Returns:
            Результат генерации
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # Проверить кэш
        if use_cache:
            cache_key = self._get_cache_key(prompt, context)
            if cache_key in self.response_cache:
                self.stats["cache_hits"] += 1
                log.debug(f"Кэш попадание для ключа: {cache_key[:16]}...")
                
                return {
                    "text": self.response_cache[cache_key],
                    "model": self.active_model,
                    "from_cache": True,
                    "response_time_ms": 1,
                    "estimated_tokens": len(self.response_cache[cache_key]) // 4,
                    "success": True,
                }
        
        # Если локальная модель недоступна - падение на облако
        if not self.active_model:
            self.stats["fallback_to_cloud"] += 1
            log.warning("Локальная модель недоступна, падаю на облако")
            
            return {
                "text": "",
                "model": None,
                "from_cache": False,
                "response_time_ms": 0,
                "estimated_tokens": 0,
                "success": False,
                "error": "Нет доступных локальных моделей",
                "fallback_required": True,
            }
        
        try:
            # Сформировать полный запрос
            full_prompt = prompt
            if context:
                context_str = json.dumps(context, indent=2)
                full_prompt = f"Контекст:\n{context_str}\n\nЗадача:\n{prompt}"
            
            # Выполнить запрос к локальной модели
            result = await self._run_ollama_command([
                "run", self.active_model,
                "--temperature", str(temperature),
                "--num-predict", str(max_tokens),
                full_prompt
            ], timeout=30)
            
            response_time = time.time() - start_time
            
            if result.returncode == 0:
                response_text = result.stdout.strip()
                
                # Сохранить в кэш
                if use_cache:
                    cache_key = self._get_cache_key(prompt, context)
                    if len(self.response_cache) >= self.cache_size_limit:
                        # Удалить самый старый элемент (FIFO)
                        oldest_key = next(iter(self.response_cache))
                        del self.response_cache[oldest_key]
                    self.response_cache[cache_key] = response_text
                
                # Обновить статистику модели
                if self.active_model in self.models:
                    self.models[self.active_model].last_used = time.time()
                    self.models[self.active_model].avg_response_time = (
                        self.models[self.active_model].avg_response_time * 0.9 + 
                        response_time * 0.1
                    )
                
                self.stats["local_responses"] += 1
                self.stats["avg_response_time_ms"] = (
                    self.stats["avg_response_time_ms"] * (self.stats["local_responses"] - 1) +
                    response_time * 1000
                ) / self.stats["local_responses"]
                
                estimated_tokens = len(response_text) // 4
                
                log.debug(f"Локальная модель ответила за {response_time:.2f} сек "
                         f"({estimated_tokens} токенов, "
                         f"{estimated_tokens/response_time:.1f} токенов/сек)")
                
                return {
                    "text": response_text,
                    "model": self.active_model,
                    "from_cache": False,
                    "response_time_ms": response_time * 1000,
                    "estimated_tokens": estimated_tokens,
                    "success": True,
                }
            else:
                self.stats["errors"] += 1
                log.error(f"Ошибка локальной модели: {result.stderr}")
                
                return {
                    "text": "",
                    "model": self.active_model,
                    "from_cache": False,
                    "response_time_ms": 0,
                    "estimated_tokens": 0,
                    "success": False,
                    "error": result.stderr,
                    "fallback_required": True,
                }
                
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"Исключение в LocalCore.generate: {e}")
            
            return {
                "text": "",
                "model": self.active_model,
                "from_cache": False,
                "response_time_ms": 0,
                "estimated_tokens": 0,
                "success": False,
                "error": str(e),
                "fallback_required": True,
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику использования."""
        return {
            **self.stats,
            "active_model": self.active_model,
            "available_models": [
                {
                    "name": m.name,
                    "installed": m.installed,
                    "available": m.available,
                    "speed_tps": m.estimated_tokens_per_sec,
                    "avg_response_time": m.avg_response_time,
                }
                for m in self.models.values()
            ],
            "cache_size": len(self.response_cache),
            "cache_hit_rate": (
                self.stats["cache_hits"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            ),
        }
    
    async def switch_model(self, model_name: str) -> bool:
        """
        Переключиться на другую локальную модель.
        
        Args:
            model_name: Имя модели для переключения
            
        Returns:
            True если переключение успешно
        """
        if model_name not in self.models:
            log.error(f"Модель {model_name} не найдена в списке")
            return False
        
        model_info = self.models[model_name]
        
        if not model_info.available:
            log.warning(f"Модель {model_name} не доступна")
            return False
        
        # Проверить скорость модели
        speed = await self._benchmark_model(model_name)
        if speed == 0:
            log.warning(f"Модель {model_name} не прошла бенчмарк")
            return False
        
        model_info.estimated_tokens_per_sec = speed
        self.active_model = model_name
        
        log.info(f"Переключился на модель {model_name} "
                f"({speed} токенов/сек)")
        
        return True
    
    def clear_cache(self):
        """Очистить кэш ответов."""
        self.response_cache.clear()
        log.info("Кэш LocalCore очищен")
    
    def is_available(self) -> bool:
        """Доступна ли локальная модель для использования."""
        return self.active_model is not None and self.active_model in self.models
    
    def get_best_model_name(self) -> Optional[str]:
        """Получить имя самой быстрой доступной модели."""
        if not self.active_model:
            available = [
                m for m in self.models.values() 
                if m.available and m.estimated_tokens_per_sec > 0
            ]
            if available:
                fastest = max(available, key=lambda m: m.estimated_tokens_per_sec)
                return fastest.name
        return self.active_model