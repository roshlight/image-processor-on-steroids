from typing import Any, TypedDict


class EditAgentState(TypedDict, total=False):
    user_instruction: str
    input_image_path: str
    current_image_path: str
    output_dir: str

    raw_plan: str
    plan: dict[str, Any]
    actions: list[dict[str, Any]]
    current_step: int

    last_mask_path: str | None # нужен для удаленя объектов
    intermediate_images: list[str] 
    final_image_path: str | None

    logs: list[dict[str, Any]]
    error: str | None
