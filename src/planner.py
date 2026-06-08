from src.config import read_prompt
from src.image_io import image_file_to_data_url
from src.qwen_client import call_qwen_chat
from src.tool_specs import get_available_tools_json


def build_planner_prompt(user_instruction: str) -> str:
    template = read_prompt("planner_user_template.txt")

    return template.format(
        user_instruction=user_instruction,
        available_tools_json=get_available_tools_json(),
    )


def call_planner(user_instruction: str, image_path: str) -> str:
    image_url = image_file_to_data_url(image_path)

    system_prompt = read_prompt("planner_system.txt")
    user_prompt = build_planner_prompt(user_instruction)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ],
        },
    ]

    return call_qwen_chat(
        messages=messages,
        temperature=0.0,
        max_tokens=1200,
    )
