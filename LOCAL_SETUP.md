# Ouroboros — Local Setup (Windows + Ollama)

## Предварительные требования

1. **Git** — установлен и доступен в PATH
2. **Python 3.10+** — установлен
3. **Ollama** — установлен ([ollama.com](https://ollama.com/))
4. **Telegram бот** — создан через @BotFather

## Быстрый старт

### 1. Запустите Ollama и скачайте модель

```bash
ollama serve
ollama pull gpt-oss:20b
```

### 2. Настройте конфигурацию

Отредактируйте `local_config.py`:
- `TELEGRAM_BOT_TOKEN` — токен вашего бота
- `GITHUB_USER` — ваш GitHub username  
- `GITHUB_TOKEN` — Personal Access Token с правами `repo`
- `AUTO_PUSH_INTERVAL_SEC` — интервал автопуша (по умолчанию 300 сек)

Или задайте переменные окружения (приоритетнее):

```bash
set TELEGRAM_BOT_TOKEN=your_token
set GITHUB_USER=your_username
set GITHUB_TOKEN=your_pat
```

### 3. Форкните репозиторий (если ещё нет)

Форкните https://github.com/razzant/ouroboros на свой GitHub.

### 4. Запустите

```bash
cd "C:\Users\morea\Рабочий стол\Программы на ПК\Ouroboros"
python local_launcher.py
```

### 5. Откройте Telegram

Напишите любое сообщение вашему боту. Первый написавший становится owner.

## Команды Telegram бота

| Команда | Описание |
|---------|----------|
| `/panic` | Аварийная остановка |
| `/restart` | Мягкий перезапуск |
| `/status` | Статус воркеров, очереди, бюджета |
| `/evolve` | Включить автоэволюцию |
| `/evolve stop` | Выключить автоэволюцию |
| `/review` | Запросить глубокий ревью кода |
| `/bg start` | Включить фоновое сознание |
| `/bg stop` | Выключить фоновое сознание |
| `/push` | Принудительный пуш на GitHub |

## Что было изменено vs оригинал

### `ouroboros/llm.py`
- Добавлена автодетекция бэкенда (Ollama / OpenRouter)
- Для Ollama: убраны OpenRouter-специфичные параметры (extra_body, provider pinning, cache_control)
- Стоимость для локальных моделей = 0

### `supervisor/git_ops.py`  
- Заменён `rm -rf` на `shutil.rmtree` (Windows-совместимость)
- `ensure_repo_present()` и `fetch` не падают без remote URL

### `local_launcher.py` (новый)
- Полная замена `colab_launcher.py` для локального запуска
- Не зависит от Google Colab / Google Drive
- Состояние хранится в `data/local_state/`
- Автопуш на GitHub в фоновом потоке (настраиваемый интервал)
- Проверка доступности Ollama при старте

### `local_config.py` (новый)
- Единый файл конфигурации для всех настроек

## Структура данных

```
data/local_state/
  state/          — state.json (основное состояние агента)
  logs/           — supervisor.jsonl, chat.jsonl
  memory/         — scratchpad, identity, chat memory
  index/          — knowledge base index
  locks/          — file locks
  archive/        — rescue snapshots
```

## Автопуш

Фоновый поток автоматически пушит на GitHub каждые N секунд
(настройка `AUTO_PUSH_INTERVAL_SEC` в `local_config.py`).
Также агент сам пушит после каждого коммита через свои git-тулзы.
Команда `/push` — принудительный ручной пуш.

## Troubleshooting

**Ollama не отвечает**: убедитесь, что `ollama serve` запущен.

**Модель не найдена**: выполните `ollama pull gpt-oss:20b`.

**Git push fails**: проверьте GITHUB_TOKEN (нужен scope `repo`).

**Агент зависает**: используйте `/panic` для остановки, затем перезапустите.
