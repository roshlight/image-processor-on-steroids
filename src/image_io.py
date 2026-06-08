import base64
from pathlib import Path

import cv2
import numpy as np


def image_file_to_data_url(path: str) -> str:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    elif suffix == ".webp":
        mime_type = "image/webp"
    else:
        raise ValueError(f"Unsupported image format: {suffix}")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def read_rgb(path: str) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 2:
        cv2.imwrite(str(path), image)
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image_bgr)

    return str(path)
