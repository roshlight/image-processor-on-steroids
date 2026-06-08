from datetime import datetime
from pathlib import Path
import json

from src.graph import build_graph


def main():
    print("\n=== Image Processor on Steroids: console agent v0 ===\n")

    image_path = input("Path to image: ").strip()
    user_instruction = input("Instruction: ").strip()

    if not image_path:
        raise ValueError("Image path is empty.")

    if not user_instruction:
        raise ValueError("Instruction is empty.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"run_{run_id}"

    app = build_graph()

    result = app.invoke(
        {
            "user_instruction": user_instruction,
            "input_image_path": image_path,
            "output_dir": str(output_dir),
            "logs": [],
        }
    )

    print("\n=== PLAN ===")
    print(json.dumps(result.get("plan", {}), ensure_ascii=False, indent=2))

    print("\n=== RESULT ===")

    if result.get("error"):
        print("Status: failed")
        print("Error:", result["error"])
    else:
        print("Status: ok")
        print("Final image:", result.get("final_image_path"))

    print("Output dir:", output_dir)
    print("Logs:", output_dir / "logs.md")


if __name__ == "__main__":
    main()
    