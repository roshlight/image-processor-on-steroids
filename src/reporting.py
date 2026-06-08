import json
from pathlib import Path
from typing import Any


def save_logs(logs: list[dict[str, Any]], output_dir: str) -> dict[str, str]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    json_path = output_dir_path / "logs.json"
    md_path = output_dir_path / "logs.md"

    json_path.write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# Agent logs\n"]

    for item in logs:
        lines.append(f"## {item.get('node', 'step')}\n")
        lines.append("```json\n")
        lines.append(json.dumps(item, ensure_ascii=False, indent=2))
        lines.append("\n```\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "logs_json": str(json_path),
        "logs_md": str(md_path),
    }
