import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def inpaint_lama(
    image: np.ndarray,
    mask: np.ndarray,
    mask_source: str = "last_mask",
    device: str = "cpu",
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        image_path = tmp_dir / "image.png"
        mask_path = tmp_dir / "mask.png"
        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image_bgr)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = mask.astype(np.uint8)
        cv2.imwrite(str(mask_path), mask)

        command = [
            "iopaint",
            "run",
            "--model",
            "lama",
            "--device",
            device,
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output",
            str(output_dir),
        ]

        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        candidates = sorted(output_dir.rglob("*.png"))

        if not candidates:
            raise RuntimeError(
                "LaMa did not produce output image.\n"
                f"Command: {' '.join(command)}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}\n"
                f"Output dir files: {[str(p) for p in output_dir.rglob('*')]}"
            )

        result_path = candidates[0]
        result_bgr = cv2.imread(str(result_path), cv2.IMREAD_COLOR)

        if result_bgr is None:
            raise RuntimeError(f"Could not read LaMa output image: {result_path}")

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        return result_rgb
    