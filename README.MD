# Image Processor on Steroids

Мультимодальный агент для редактирования изображений по текстовому запросу.

Пользователь загружает изображение, пишет инструкцию обычным языком, а система сама строит план редактирования, вызывает нужные инструменты, сохраняет промежуточные шаги, финальное изображение и подробные логи выполнения.

---

## Что делает проект

Проект принимает на вход изображение и текстовую инструкцию пользователя.

Примеры запросов:

```text
Сделай фото ярче, поверни его на 90 градусов вправо и добавь надпись HELLO
```

```text
Убери женщину с картинки
```

После этого агент отправляет изображение и пользовательский запрос в Qwen, получает JSON-план действий, валидирует его, последовательно выполняет инструменты, сохраняет промежуточные изображения, сохраняет финальную картинку и показывает результат в браузере.

---

## Главная идея

Qwen используется не как редактор изображений, а как VLM-планировщик.

Он не меняет пиксели напрямую. Он только понимает запрос пользователя и решает, какие инструменты нужно вызвать.

Редактирование изображения выполняется локально с помощью OpenCV, PIL и LaMa / IOPaint для inpainting.

---

## Архитектура

```text
Browser UI
   ↓
FastAPI /edit
   ↓
LangGraph agent
   ↓
Qwen planner
   ↓
JSON plan
   ↓
Plan validator
   ↓
Tool executor
   ↓
OpenCV / PIL / LaMa tools
   ↓
final.png + logs.json + logs.md
```

---

## Как работает агент

Пользователь отправляет изображение и текстовую инструкцию через веб-интерфейс.

FastAPI сохраняет изображение в папку конкретного запуска:

```text
outputs/run_YYYYMMDD_HHMMSS/
```

После этого запускается LangGraph pipeline:

```text
prepare_input
    ↓
planner
    ↓
parse_plan
    ↓
validate_plan
    ↓
execute_next_tool
    ↓
execute_next_tool
    ↓
...
    ↓
save_final
```

Planner вызывает Qwen и получает JSON-план.

Пример плана:

```json
{
  "task_type": "object_removal",
  "actions": [
    {
      "tool": "segment_object",
      "args": {
        "target": "woman",
        "padding": 20
      },
      "reason": "Нужно найти женщину на изображении и создать маску."
    },
    {
      "tool": "inpaint_lama",
      "args": {
        "mask_source": "last_mask",
        "device": "cpu"
      },
      "reason": "Нужно удалить объект по маске и восстановить фон."
    }
  ],
  "user_message": "Я удалю женщину с изображения."
}
```

Validator проверяет, что ответ является валидным JSON, в плане есть `actions`, каждый tool существует в списке доступных инструментов, а аргументы переданы в корректном формате.

Executor последовательно применяет инструменты к изображению и обновляет текущую версию картинки.

---

## Что умеет проект

Текущая версия поддерживает одну входную картинку и набор инструментов для обработки изображений.

Базовые операции:

- изменение яркости;
- изменение контраста;
- изменение насыщенности;
- повышение резкости;
- Gaussian blur;
- median blur;
- grayscale;
- threshold binary;
- Canny edge detection;
- поворот на 90 градусов вправо;
- поворот на 90 градусов влево;
- поворот на 180 градусов;
- инверсия цветов;
- histogram equalization;
- denoise;
- добавление текста.

Удаление объектов работает через pipeline:

```text
Qwen bbox → mask → LaMa inpainting
```

То есть Qwen находит объект по текстовому описанию, `segment_object` строит маску объекта, `inpaint_lama` удаляет объект по маске, а LaMa восстанавливает фон.

---

## Как работает удаление объекта

Пользователь пишет:

```text
Убери женщину с картинки
```

Qwen planner строит план:

```text
segment_object → inpaint_lama
```

Сначала вызывается `segment_object`.

Он получает:

```json
{
  "target": "woman",
  "padding": 20
}
```

Дальше `segment_object` отправляет изображение в Qwen, просит найти bounding box объекта, получает координаты `[x1, y1, x2, y2]`, расширяет bbox на `padding` и создаёт бинарную маску.

Маска — это чёрно-белое изображение:

```text
чёрное = оставить как есть
белое = область для удаления / восстановления
```

Маска сохраняется как:

```text
step_01_segment_object_mask.png
```

После этого вызывается `inpaint_lama`.

Он получает:

```json
{
  "mask_source": "last_mask",
  "device": "cpu"
}
```

`inpaint_lama` берёт текущее изображение, берёт последнюю созданную маску, запускает LaMa через IOPaint, восстанавливает фон в белой области маски и сохраняет результат.

Результат сохраняется как:

```text
step_02_inpaint_lama.png
```

Финальное изображение сохраняется как:

```text
final.png
```

---

## Примеры запросов

```text
Сделай фото ярче
```

```text
Повысь контраст изображения
```

```text
Сделай цвета более насыщенными
```

```text
Сделай изображение чёрно-белым
```

```text
Выдели границы объектов
```

```text
Поверни изображение на 90 градусов вправо
```

```text
Поверни изображение на 90 градусов влево
```

```text
Добавь сверху надпись HELLO
```

```text
Сделай фото ярче, поверни на 90 градусов вправо и добавь сверху надпись HELLO
```

```text
Убери человека справа с фотографии
```

```text
Убери женщину с картинки
```

---

## Ограничения

Текущая версия является прототипом.

Object removal сейчас работает через bbox-based mask:

```text
Qwen bbox → прямоугольная маска → LaMa
```

Это значит, что качество удаления зависит от точности bbox, размера padding, сложности фона и качества восстановления LaMa.

Пока не реализованы:

- точная SAM-сегментация объекта;
- GroundingDINO;
- работа с двумя изображениями;
- style transfer между изображениями;
- multi-turn editing без повторной загрузки;
- стабильная генеративная замена объекта на другой объект.

---

## Структура проекта

```text
image-processor-on-steroids/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env.example
├── requirements.txt
├── run_agent.py
├── prompts/
│   ├── planner_system.txt
│   └── planner_user_template.txt
├── src/
│   ├── api.py
│   ├── config.py
│   ├── executor.py
│   ├── graph.py
│   ├── image_io.py
│   ├── planner.py
│   ├── qwen_client.py
│   ├── reporting.py
│   ├── state.py
│   ├── tool_specs.py
│   ├── validator.py
│   └── tools/
│       ├── tools_from_cv_hw.py
│       ├── segmentation.py
│       └── inpainting.py
├── templates/
│   └── index.html
├── assets/
├── outputs/
└── model_cache/
```

---

## Основные файлы

`src/api.py` — FastAPI-приложение. Отвечает за отображение главной страницы, приём изображения и инструкции, запуск агента и возврат результата в браузер.

`src/graph.py` — LangGraph pipeline агента. Описывает порядок шагов: `prepare_input → planner → parse_plan → validate_plan → execute tools → save_final`.

`src/planner.py` — собирает prompt для Qwen и вызывает planner.

`src/validator.py` — проверяет JSON-план от Qwen.

`src/executor.py` — последовательно применяет tools к изображению.

`src/tool_specs.py` — список доступных инструментов, который передаётся Qwen planner-у.

`src/tools/tools_from_cv_hw.py` — основные OpenCV/PIL-инструменты обработки изображений.

`src/tools/segmentation.py` — инструмент для поиска объекта по текстовому описанию и построения маски.

`src/tools/inpainting.py` — инструмент для удаления объекта по маске через LaMa / IOPaint.

---

## Требования

Для запуска через Docker нужно:

- Git;
- Docker Desktop;
- Alibaba Model Studio API key.

Python и зависимости вручную ставить не нужно, если проект запускается через Docker.

---

## Переменные окружения

Создай `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

В `.env` нужно добавить Alibaba API key:

```env
DASHSCOPE_API_KEY=your_alibaba_api_key_here
```

Файл `.env` нельзя коммитить в GitHub.

---

## Запуск через Docker

Склонируй репозиторий:

```bash
git clone git@github.com:roshlight/image-processor-on-steroids.git
cd image-processor-on-steroids
```

Создай `.env`:

```bash
cp .env.example .env
```

Добавь ключ в `.env`:

```env
DASHSCOPE_API_KEY=your_alibaba_api_key_here
```

Собери образ:

```bash
docker compose build
```

Запусти приложение:

```bash
docker compose up
```

Открой в браузере:

```text
http://localhost:8000
```

---

## Первый запуск LaMa

При первом удалении объекта LaMa может скачать веса модели:

```text
big-lama.pt
```

Это может занять несколько минут.

Чтобы веса не скачивались заново при каждом запуске, в `docker-compose.yml` используется volume:

```yaml
- ./model_cache:/root/.cache
```

После первого скачивания веса будут храниться локально в папке:

```text
model_cache/
```

---

## Как пользоваться

1. Открыть `http://localhost:8000`.
2. Загрузить изображение.
3. Ввести текстовую инструкцию.
4. Нажать `Run agent`.
5. Дождаться результата.
6. Посмотреть финальное изображение и логи.

---

## Где сохраняются результаты

После каждого запуска создаётся папка:

```text
outputs/run_YYYYMMDD_HHMMSS/
```

Пример:

```text
outputs/run_20260609_200337/
├── input_image.jpeg
├── step_01_segment_object_mask.png
├── step_02_inpaint_lama.png
├── final.png
├── logs.json
└── logs.md
```

---

## Логи

Для каждого запуска сохраняются:

```text
logs.json
logs.md
```

В логах можно увидеть входное изображение, пользовательскую инструкцию, сырой JSON-план от Qwen, распарсенный план, выбранные tools, аргументы каждого tool-а, пути к промежуточным изображениям, путь к финальному изображению и ошибку, если она произошла.

---

## Запуск без Docker

Создай виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate
```

Установи зависимости:

```bash
pip install -r requirements.txt
```

Создай `.env`:

```bash
cp .env.example .env
```

Добавь ключ:

```env
DASHSCOPE_API_KEY=your_alibaba_api_key_here
```

Запусти FastAPI:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Открой:

```text
http://localhost:8000
```

---

## Полезные Docker-команды

Запустить:

```bash
docker compose up
```

Собрать образ:

```bash
docker compose build
```

Полностью пересобрать без cache:

```bash
docker compose build --no-cache
```

Остановить контейнер:

```bash
docker compose down
```

Запустить в фоне:

```bash
docker compose up -d
```

Посмотреть логи:

```bash
docker compose logs -f
```

---

## Что не коммитить

В репозиторий не нужно добавлять:

```text
.env
outputs/
__pycache__/
.DS_Store
.ipynb_checkpoints/
model_cache/
```

---

## Roadmap

Планируемые улучшения:

- SAM для точной сегментации объекта;
- GroundingDINO для поиска объектов по тексту;
- более точные маски вместо прямоугольных bbox-mask;
- Stable Diffusion Inpainting для замены объектов;
- работа с двумя изображениями;
- multi-turn editing без повторной загрузки;
- улучшенный UI;
- просмотр промежуточных шагов прямо в интерфейсе.

---

## Коротко

Это локальный image editing agent:

```text
Qwen понимает запрос пользователя
↓
LangGraph управляет пайплайном
↓
OpenCV/PIL/LaMa tools редактируют изображение
↓
FastAPI показывает результат в браузере
↓
Docker делает запуск воспроизводимым
```