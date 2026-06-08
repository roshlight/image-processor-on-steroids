from pathlib import Path
from typing import Any

from src.image_io import read_rgb, save_image
from src.tool_specs import TOOL_KIND_BY_NAME
from src.tools.tools_from_cv_hw import TOOL_REGISTRY


def normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    args = dict(args)

    if tool_name == "add_text":
        if "color" in args and isinstance(args["color"], list):
            args["color"] = tuple(args["color"])

    if tool_name == "sharpen":
        if "factor" in args and "amount" not in args:
            args["amount"] = args.pop("factor")

    if tool_name in {"gaussian_blur", "median_blur"}:
        if "kernel" in args and "kernel_size" not in args:
            args["kernel_size"] = args.pop("kernel")

    if tool_name == "inpaint_opencv":
        args.pop("mask_source", None)

    return args


def execute_action(
    current_image_path: str,
    output_dir: str,
    step_idx: int,
    action: dict[str, Any],
    last_mask_path: str | None = None,
) -> dict[str, Any]:
    tool_name = action["tool"]
    args = normalize_tool_args(tool_name, action.get("args", {}))

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool is not implemented in TOOL_REGISTRY: {tool_name}")

    tool_fn = TOOL_REGISTRY[tool_name]
    tool_kind = TOOL_KIND_BY_NAME[tool_name]

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    image = read_rgb(current_image_path)

    if tool_kind == "image_to_image":
        result_image = tool_fn(image, **args)
        output_path = output_dir_path / f"step_{step_idx:02d}_{tool_name}.png"
        save_image(result_image, str(output_path))

        return {
            "kind": tool_kind,
            "tool": tool_name,
            "output_image_path": str(output_path),
            "last_mask_path": last_mask_path,
        }

    if tool_kind == "mask_producer":
        mask = tool_fn(image, **args)
        mask_path = output_dir_path / f"step_{step_idx:02d}_{tool_name}_mask.png"
        save_image(mask, str(mask_path))

        return {
            "kind": tool_kind,
            "tool": tool_name,
            "output_image_path": current_image_path,
            "last_mask_path": str(mask_path),
        }

    if tool_kind == "image_mask_to_image":
        if last_mask_path is None:
            raise RuntimeError(
                f"Tool {tool_name} requires a mask, but last_mask_path is None."
            )

        mask = read_rgb(last_mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        result_image = tool_fn(image, mask=mask, **args)
        output_path = output_dir_path / f"step_{step_idx:02d}_{tool_name}.png"
        save_image(result_image, str(output_path))

        return {
            "kind": tool_kind,
            "tool": tool_name,
            "output_image_path": str(output_path),
            "last_mask_path": last_mask_path,
        }

    raise ValueError(f"Unknown tool kind: {tool_kind}")
