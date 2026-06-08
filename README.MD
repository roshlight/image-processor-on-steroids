# Image Processor on Steroids

Локальный агент для редактирования изображений по текстовому запросу.

Пользователь загружает картинку, пишет инструкцию обычным языком, а система сама строит план редактирования и применяет нужные OpenCV/PIL-инструменты.

Qwen 3.5 Plus используется как VLM-планировщик: он понимает запрос и возвращает JSON-план. Само изображение редактируется локально.

---

## Что можно делать

Сейчас проект поддерживает одну входную картинку и базовые операции обработки изображений:

- изменить яркость;
- изменить контраст;
- изменить насыщенность;
- повысить резкость;
- размыть изображение;
- сделать изображение чёрно-белым;
- выделить границы объектов через Canny;
- повернуть изображение на 90 или 180 градусов;
- инвертировать цвета;
- добавить текст поверх изображения;
- применить denoise;
- сохранить финальное изображение и подробные логи.

Можно комбинировать несколько действий в одном запросе.

Примеры запросов:

text Сделай фото ярче 

text Повысь контраст и насыщенность 

text Поверни изображение на 90 градусов вправо 

text Добавь сверху надпись HELLO 

text Сделай фото ярче, поверни на 90 градусов вправо и добавь сверху надпись HELLO 

text Сделай изображение чёрно-белым, а потом выдели границы объектов 

---

## Архитектура

text Browser UI    ↓ FastAPI /edit    ↓ LangGraph agent    ↓ Qwen planner    ↓ JSON plan    ↓ Plan validator    ↓ OpenCV/PIL executor    ↓ final.png + logs.json + logs.md 

---

## Ограничения текущей версии

Пока проект не умеет стабильно:

- работать с двумя изображениями;
- удалять конкретный объект по тексту;
- точно выделять маску объекта;
- использовать SAM / GroundingDINO;
- использовать LaMa inpainting;
- делать style transfer;
- продолжать редактирование без повторной загрузки картинки.

---

## Запуск через Docker

### 1. Склонировать репозиторий

bash git clone git@github.com:roshlight/image-processor-on-steroids.git cd image-processor-on-steroids 

### 2. Создать .env

bash cp .env.example .env 

В .env нужно добавить Alibaba API key:

env DASHSCOPE_API_KEY=your_alibaba_api_key_here 

### 3. Запустить проект

bash docker compose up --build 

### 4. Открыть сайт

text http://localhost:8000 

---

## Как пользоваться

1. Открыть http://localhost:8000.
2. Загрузить изображение.
3. Написать инструкцию.
4. Нажать Run agent.
5. Дождаться результата.
6. Посмотреть итоговое изображение и логи.

---

## Где сохраняются результаты

После каждого запуска создаётся папка:

text outputs/run_YYYYMMDD_HHMMSS/ 

Внутри:

text input image step_01_... step_02_... final.png logs.json logs.md 

final.png — итоговое изображение.  
logs.json и logs.md — подробные логи работы агента.

---

## Запуск без Docker

bash python -m venv .venv source .venv/bin/activate pip install -r requirements.txt cp .env.example .env 

Добавить ключ в .env:

env DASHSCOPE_API_KEY=your_alibaba_api_key_here 

Запустить FastAPI:

bash uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload 

Открыть:

text http://localhost:8000 

---

## Полезные Docker-команды

Запустить:

bash docker compose up --build 

Остановить:

bash docker compose down 

Запустить без пересборки:

bash docker compose up 

Пересобрать без кеша:

bash docker compose build --no-cache 

---

## Что не коммитить

В репозиторий не нужно добавлять:

text .env outputs/ __pycache__/ .DS_Store .ipynb_checkpoints/ 

Эти файлы должны быть в .gitignore.