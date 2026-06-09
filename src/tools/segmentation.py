import base64
import json
import re

import cv2
import numpy as np

from src.qwen_client import call_qwen_chat


def image_array_to_data_url(image: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", image_bgr)

    if not success:
        raise RuntimeError("Failed to encode image to PNG.")

    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def extract_json(raw: str) -> dict:
    text = raw.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {raw}")

    return json.loads(text[start:end + 1])


def get_bbox_from_qwen(image: np.ndarray, target: str) -> list[int]:
    h, w = image.shape[:2]
    image_url = image_array_to_data_url(image)

    prompt = f"""
Ты должен найти объект на изображении и вернуть его bounding box.

Искомый объект:
{target}

Размер изображения:
width={w}, height={h}

Верни строго JSON без markdown:

{{
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0
}}

Правила:
- bbox должен быть в пикселях.
- x1, y1 — левый верхний угол.
- x2, y2 — правый нижний угол.
- Координаты должны лежать внутри изображения.
- Если объект не найден, верни bbox=null.
"""

    messages = [
        {
            "role": "system",
            "content": "Ты VLM-модель для object grounding. Отвечай только валидным JSON.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    raw = call_qwen_chat(messages=messages, temperature=0.0, max_tokens=500)
    parsed = extract_json(raw)

    bbox = parsed.get("bbox")

    if bbox is None:
        raise RuntimeError(f"Object not found by Qwen: {target}")

    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox from Qwen: {bbox}")

    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox coordinates: {bbox}")

    return [x1, y1, x2, y2]


def segment_object(image: np.ndarray, target: str, padding: int = 20) -> np.ndarray:
    h, w = image.shape[:2]

    x1, y1, x2, y2 = get_bbox_from_qwen(image=image, target=target)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return mask
