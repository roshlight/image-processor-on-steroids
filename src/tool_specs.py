import json


TOOL_SPECS = [
    {
        "name": "adjust_brightness",
        "kind": "image_to_image",
        "description": "Изменяет яркость изображения. value > 0 делает ярче, value < 0 темнее.",
        "args": {"value": "int, например 30 или -30"},
    },
    {
        "name": "adjust_contrast",
        "kind": "image_to_image",
        "description": "Изменяет контраст изображения.",
        "args": {"factor": "float, например 1.2"},
    },
    {
        "name": "adjust_saturation",
        "kind": "image_to_image",
        "description": "Изменяет насыщенность изображения.",
        "args": {"factor": "float, например 1.3"},
    },
    {
        "name": "sharpen",
        "kind": "image_to_image",
        "description": "Повышает резкость изображения.",
        "args": {"amount": "float, например 1.0"},
    },
    {
        "name": "gaussian_blur",
        "kind": "image_to_image",
        "description": "Применяет гауссово размытие.",
        "args": {"kernel_size": "int odd, например 7", "sigma": "float, например 0"},
    },
    {
        "name": "median_blur",
        "kind": "image_to_image",
        "description": "Применяет медианный фильтр.",
        "args": {"kernel_size": "int odd, например 5"},
    },
    {
        "name": "to_grayscale",
        "kind": "image_to_image",
        "description": "Переводит изображение в градации серого.",
        "args": {},
    },
    {
        "name": "threshold_binary",
        "kind": "image_to_image",
        "description": "Бинаризует изображение по порогу.",
        "args": {"threshold": "int 0..255"},
    },
    {
        "name": "canny_edges",
        "kind": "image_to_image",
        "description": "Выделяет границы объектов методом Canny.",
        "args": {"threshold1": "int", "threshold2": "int"},
    },
    {
        "name": "rotate90_cw",
        "kind": "image_to_image",
        "description": "Поворачивает изображение на 90 градусов по часовой стрелке.",
        "args": {},
    },
    {
        "name": "rotate90_ccw",
        "kind": "image_to_image",
        "description": "Поворачивает изображение на 90 градусов против часовой стрелки.",
        "args": {},
    },
    {
        "name": "rotate180",
        "kind": "image_to_image",
        "description": "Поворачивает изображение на 180 градусов.",
        "args": {},
    },
    {
        "name": "invert_colors",
        "kind": "image_to_image",
        "description": "Инвертирует цвета изображения.",
        "args": {},
    },
    {
        "name": "equalize_hist_y_channel",
        "kind": "image_to_image",
        "description": "Эквализует яркостный канал изображения.",
        "args": {},
    },
    {
        "name": "denoise_nl_means",
        "kind": "image_to_image",
        "description": "Удаляет шум методом Non-local Means.",
        "args": {"h": "float, например 10"},
    },
    {
        "name": "add_text",
        "kind": "image_to_image",
        "description": "Добавляет текст на изображение.",
        "args": {
            "text": "str",
            "position": "top | center | bottom",
            "color": "RGB list, например [255, 255, 0]",
            "font_scale": "float, например 1.5",
            "thickness": "int, например 3",
        },
    },
    {
        "name": "create_center_mask",
        "kind": "mask_producer",
        "description": "Создаёт центральную прямоугольную маску для удаления объекта.",
        "args": {"rel_w": "float, например 0.25", "rel_h": "float, например 0.25"},
    },
    {
        "name": "create_bbox_mask",
        "kind": "mask_producer",
        "description": "Создаёт маску по bbox.",
        "args": {
            "bbox": {
                "x1": "int",
                "y1": "int",
                "x2": "int",
                "y2": "int",
            },
            "padding": "int, например 10",
        },
    },
    {
        "name": "inpaint_opencv",
        "kind": "image_mask_to_image",
        "description": "Удаляет объект по последней маске через OpenCV inpainting.",
        "args": {"radius": "int, например 3"},
    },
    {
        "name": "segment_object",
        "kind": "mask_producer",
        "description": (
            "Находит объект по текстовому описанию и создаёт маску. "
            "Используется перед inpaint_lama для удаления объектов."
        ),
        "args": {
            "target": "str, например 'ficus plant on the right'",
            "padding": "int, например 20"
        },
    },
    {
        "name": "inpaint_lama",
        "kind": "image_mask_to_image",
        "description": (
            "Удаляет объект по последней созданной маске с помощью LaMa/IOPaint. "
            "Используется после segment_object."
        ),
        "args": {
            "mask_source": "last_mask",
            "device": "cpu"
        },
    },
]


TOOL_KIND_BY_NAME = {tool["name"]: tool["kind"] for tool in TOOL_SPECS}
ALLOWED_TOOL_NAMES = set(TOOL_KIND_BY_NAME.keys())


def get_available_tools_json() -> str:
    return json.dumps(TOOL_SPECS, ensure_ascii=False, indent=2)
