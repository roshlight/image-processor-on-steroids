import json
import re
from typing import Any

from src.tool_specs import ALLOWED_TOOL_NAMES


def strip_json_markdown(raw: str) -> str:
    text = raw.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def parse_plan(raw: str) -> dict[str, Any]:
    cleaned = strip_json_markdown(raw)
    return json.loads(cleaned)


def validate_plan(plan: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(plan, dict):
        return False, "Plan must be a dict."

    actions = plan.get("actions")

    if actions is None:
        return False, "Plan must contain actions."

    if not isinstance(actions, list):
        return False, "actions must be a list."

    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            return False, f"Action {idx} must be a dict."

        tool = action.get("tool")

        if tool not in ALLOWED_TOOL_NAMES:
            return False, f"Unknown tool in action {idx}: {tool}"

        args = action.get("args", {})

        if not isinstance(args, dict):
            return False, f"args in action {idx} must be a dict."

    return True, None
