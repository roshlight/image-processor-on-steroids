"""
Tools extracted/adapted from cv_hw_1-2.ipynb for the image-editing-agent project.

Conventions:
- Functions accept numpy images.
- Most color functions expect RGB uint8 images.
- Return values are numpy arrays, usually uint8.
- These functions are designed to be used in a tool registry / agent executor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Clip image to [0, 255] and convert to uint8."""
    return np.clip(image, 0, 255).astype(np.uint8)


def _odd_kernel_size(k: int) -> int:
    """OpenCV blur kernels must be positive odd numbers."""
    k = int(k)
    if k <= 0:
        raise ValueError("kernel size must be positive")
    return k if k % 2 == 1 else k + 1


# ---------------------------------------------------------------------
# Numpy-only / geometry-like operations from the notebook
# ---------------------------------------------------------------------

def flip_vertical(image: np.ndarray) -> np.ndarray:
    """Vertical flip: from task_0_solution."""
    return image[::-1, :, :]


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """Horizontal flip."""
    return image[:, ::-1, :]


def flip_both(image: np.ndarray) -> np.ndarray:
    """Flip by both axes: from task_1_solution."""
    return image[::-1, ::-1, :]


def rotate90_ccw(image: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees counter-clockwise: from task_3_solution."""
    return np.rot90(image)


def rotate90_cw(image: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees clockwise."""
    return np.rot90(image, k=3)


def rotate180(image: np.ndarray) -> np.ndarray:
    """Rotate 180 degrees."""
    return np.rot90(image, k=2)


def mirror_concat_horizontal(image: np.ndarray) -> np.ndarray:
    """Concatenate image with its horizontal mirror: from task_5_solution."""
    return np.concatenate([image, image[:, ::-1, :]], axis=1)


def pad_constant(image: np.ndarray, pad: int = 100, value: int = 0) -> np.ndarray:
    """Constant padding around image: from task_6_solution."""
    return np.pad(
        image,
        ((pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=value,
    )


def pad_edge(image: np.ndarray, pad: int = 100) -> np.ndarray:
    """Edge padding around image: from task_7_solution."""
    return np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")


def roll_horizontal(image: np.ndarray, shift: int) -> np.ndarray:
    """Cyclic horizontal shift, generalized from task_8_solution."""
    return np.concatenate([image[:, shift:, :], image[:, :shift, :]], axis=1)


def permute_channels(image: np.ndarray, order: tuple[int, int, int] = (1, 2, 0)) -> np.ndarray:
    """Reorder channels, generalized from task_9_solution."""
    return image[:, :, list(order)]


# ---------------------------------------------------------------------
# Brightness / gamma / tone tools from the notebook
# ---------------------------------------------------------------------

def add_brightness(image: np.ndarray, value: int = 50) -> np.ndarray:
    """Increase brightness with saturation: from task_10_solution."""
    return cv2.add(image, int(value))


def subtract_brightness(image: np.ndarray, value: int = 117) -> np.ndarray:
    """Decrease brightness with saturation: from task_11_solution."""
    return cv2.subtract(image, int(value))


def adjust_brightness(image: np.ndarray, value: int = 30) -> np.ndarray:
    """
    Signed brightness adjustment.
    Positive value brightens, negative value darkens.
    """
    if value >= 0:
        return add_brightness(image, value)
    return subtract_brightness(image, -value)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma correction, generalized from task_12_solution/task_13_solution."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    normalized = image.astype(np.float32) / 255.0
    corrected = cv2.pow(normalized, float(gamma))
    return ensure_uint8(corrected * 255.0)


def adjust_value_gamma_hsv(image: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """
    Apply gamma correction only to HSV Value channel.
    Generalized from task_20_solution/task_21_solution.
    Expects RGB image.
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image_hsv)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    v = cv2.LUT(v, lut)
    image_hsv = np.stack([h, s, v], axis=-1)
    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)


def adjust_contrast(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """
    Contrast adjustment around 127.5.
    Extra generic tool needed by the project.
    """
    image_f = image.astype(np.float32)
    return ensure_uint8((image_f - 127.5) * factor + 127.5)


# ---------------------------------------------------------------------
# Color space / histogram tools from the notebook
# ---------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale: from task_14_solution."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def rgb_to_ycrcb(image: np.ndarray) -> np.ndarray:
    """Convert RGB to YCrCb: from task_15_solution."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def hsv_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert HSV to RGB. The notebook had an HSV2RGB conversion task."""
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def equalize_hist_gray(image: np.ndarray) -> np.ndarray:
    """Grayscale + histogram equalization: from task_17_solution."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(gray)


def equalize_hist_y_channel(image: np.ndarray) -> np.ndarray:
    """
    Histogram equalization on luminance channel, returns RGB.
    Useful for photo enhancement.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)


def adjust_saturation(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Saturation adjustment in HSV.
    Generalized from task_18_solution/task_19_solution.
    Expects RGB image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def shift_hue(image: np.ndarray, delta: int = 10) -> np.ndarray:
    """Hue shift in HSV. Extra project tool."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(delta)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------
# Morphology tools from the notebook
# ---------------------------------------------------------------------

def cross_kernel() -> np.ndarray:
    """Kernel from task_22/task_23/task_24."""
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def vertical_kernel() -> np.ndarray:
    """Kernel from task_26_solution."""
    return np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)


def erode(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological erosion: from task_22_solution."""
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, cross_kernel(), iterations=iterations)


def dilate(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological dilation: from task_23_solution."""
    return cv2.morphologyEx(image, cv2.MORPH_DILATE, cross_kernel(), iterations=iterations)


def opening(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological opening: from task_24_solution."""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, cross_kernel(), iterations=iterations)


def closing_vertical(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological closing with vertical kernel: from task_26_solution."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, vertical_kernel(), iterations=iterations)


# ---------------------------------------------------------------------
# Blur / filtering / edge tools from the notebook
# ---------------------------------------------------------------------

def gaussian_blur(image: np.ndarray, kernel_size: int = 13, sigma: float = 0) -> np.ndarray:
    """Gaussian blur: from task_27_solution."""
    k = _odd_kernel_size(kernel_size)
    return cv2.GaussianBlur(image, (k, k), sigma)


def horizontal_gaussian_blur(image: np.ndarray, kernel_size: int = 51, sigma: float = 0) -> np.ndarray:
    """Horizontal blur: from task_28_solution."""
    k = _odd_kernel_size(kernel_size)
    return cv2.GaussianBlur(image, (k, 1), sigma)


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median filter. Extra project tool."""
    k = _odd_kernel_size(kernel_size)
    return cv2.medianBlur(image, k)


def sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Sharpening via 3x3 kernel.
    The notebook had a sharpening task stub; this is a practical implementation for the project.
    """
    kernel = np.array([[0, -1, 0], [-1, 5 + amount, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def canny_edges(image: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Canny edge detection. Extra project tool."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return cv2.Canny(gray, threshold1, threshold2)


def threshold_binary(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Binary threshold. Extra project tool."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


# ---------------------------------------------------------------------
# Geometric transforms from the notebook
# ---------------------------------------------------------------------

def rotate_affine(image: np.ndarray, angle: float = 45, scale: float = 1.0) -> np.ndarray:
    """Affine rotation around center: from task_31_solution."""
    h, w = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)


def rotate_affine_around_bottom_left(image: np.ndarray, angle: float = 350, scale: float = 1.0) -> np.ndarray:
    """Affine rotation around bottom-left point: from task_32_solution."""
    h, w = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((0, h), angle, scale)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)


def perspective_warp_demo(image: np.ndarray, offset: int = 100) -> np.ndarray:
    """Perspective transform pattern adapted from task_33_solution."""
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32([[offset, offset], [w, 0], [w - offset, h - offset], [0, h]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (w, h))


def zoom_remap(image: np.ndarray, center_x: int, center_y: int, scale_factor: float = 1.8) -> np.ndarray:
    """Zoom-like remap around center: from task_35_solution."""
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx_new = center_x + (xx - center_x) / scale_factor
    yy_new = center_y + (yy - center_y) / scale_factor
    return cv2.remap(image, xx_new.astype(np.float32), yy_new.astype(np.float32), cv2.INTER_LINEAR)


def wave_warp(image: np.ndarray, amplitude: float = 50, period: float = 50) -> np.ndarray:
    """Wave deformation via remap: from task_36_solution."""
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx_new = xx + np.sin(yy / period) * amplitude
    yy_new = yy
    return cv2.remap(image, xx_new.astype(np.float32), yy_new.astype(np.float32), cv2.INTER_LINEAR)


def fisheye_warp(
    image: np.ndarray,
    center_x: int | None = None,
    center_y: int | None = None,
    alpha: float = 0.01,
    scale: float = 0.25,
) -> np.ndarray:
    """Fisheye-like effect via polar remap: from task_37_solution."""
    h, w = image.shape[:2]
    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    x_rel = xx - center_x
    y_rel = yy - center_y

    r = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(y_rel, x_rel)
    r_new = (r + alpha * (r**2)) * scale

    x_new = center_x + r_new * np.cos(theta)
    y_new = center_y + r_new * np.sin(theta)

    return cv2.remap(image, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)


# ---------------------------------------------------------------------
# Extra project-critical tools: crop, resize, text, inpaint masks
# ---------------------------------------------------------------------

def crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop image by bbox."""
    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return image[y1:y2, x1:x2]


def resize(image: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    """Resize while preserving aspect ratio when one dimension is omitted."""
    h, w = image.shape[:2]

    if width is None and height is None:
        raise ValueError("Either width or height must be provided")

    if width is None:
        scale = height / h
        width = int(w * scale)
    elif height is None:
        scale = width / w
        height = int(h * scale)

    return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)


def add_text(
    image: np.ndarray,
    text: str,
    position: Literal["top", "bottom", "center"] = "top",
    color: tuple[int, int, int] = (255, 255, 0),
    font_scale: float = 1.5,
    thickness: int = 3,
) -> np.ndarray:
    """Add text to image using OpenCV. Color is RGB."""
    result = image.copy()
    h, w = result.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if position == "top":
        x = (w - text_w) // 2
        y = text_h + 30
    elif position == "bottom":
        x = (w - text_w) // 2
        y = h - 30
    elif position == "center":
        x = (w - text_w) // 2
        y = (h + text_h) // 2
    else:
        raise ValueError(f"Unknown position: {position}")

    # cv2 uses BGR internally for drawing, but our image is RGB.
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.putText(result_bgr, text, (x, y), font, font_scale, bgr_color, thickness, cv2.LINE_AA)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def create_bbox_mask(image: np.ndarray, bbox: dict, padding: int = 0) -> np.ndarray:
    """
    Create binary mask from bbox.
    White area is intended for removal / inpainting.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(bbox["x1"]) - padding)
    y1 = max(0, int(bbox["y1"]) - padding)
    x2 = min(w, int(bbox["x2"]) + padding)
    y2 = min(h, int(bbox["y2"]) + padding)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def create_center_mask(image: np.ndarray, rel_w: float = 0.25, rel_h: float = 0.25) -> np.ndarray:
    """Create central rectangle mask."""
    h, w = image.shape[:2]
    box_w = int(w * rel_w)
    box_h = int(h * rel_h)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    return create_bbox_mask(image, {"x1": x1, "y1": y1, "x2": x1 + box_w, "y2": y1 + box_h})


def inpaint_opencv(image: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Object removal fallback using OpenCV Telea inpainting."""
    # OpenCV inpaint expects BGR image.
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.inpaint(image_bgr, mask, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def invert_colors(image: np.ndarray) -> np.ndarray:
    """Invert colors."""
    return 255 - image


def denoise_nl_means(image: np.ndarray, h: float = 10) -> np.ndarray:
    """Denoising using OpenCV Non-local Means."""
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.fastNlMeansDenoisingColored(image_bgr, None, h, h, 7, 21)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------
# Tool registry for the agent
# ---------------------------------------------------------------------

TOOL_REGISTRY = {
    # geometry / numpy
    "flip_vertical": flip_vertical,
    "flip_horizontal": flip_horizontal,
    "flip_both": flip_both,
    "rotate90_ccw": rotate90_ccw,
    "rotate90_cw": rotate90_cw,
    "rotate180": rotate180,
    "mirror_concat_horizontal": mirror_concat_horizontal,
    "pad_constant": pad_constant,
    "pad_edge": pad_edge,
    "roll_horizontal": roll_horizontal,
    "permute_channels": permute_channels,

    # tone / color
    "adjust_brightness": adjust_brightness,
    "add_brightness": add_brightness,
    "subtract_brightness": subtract_brightness,
    "adjust_contrast": adjust_contrast,
    "gamma_correction": gamma_correction,
    "adjust_value_gamma_hsv": adjust_value_gamma_hsv,
    "to_grayscale": to_grayscale,
    "rgb_to_ycrcb": rgb_to_ycrcb,
    "rgb_to_hsv": rgb_to_hsv,
    "hsv_to_rgb": hsv_to_rgb,
    "equalize_hist_gray": equalize_hist_gray,
    "equalize_hist_y_channel": equalize_hist_y_channel,
    "adjust_saturation": adjust_saturation,
    "shift_hue": shift_hue,
    "invert_colors": invert_colors,
    "denoise_nl_means": denoise_nl_means,

    # morphology / filters
    "erode": erode,
    "dilate": dilate,
    "opening": opening,
    "closing_vertical": closing_vertical,
    "gaussian_blur": gaussian_blur,
    "horizontal_gaussian_blur": horizontal_gaussian_blur,
    "median_blur": median_blur,
    "sharpen": sharpen,
    "canny_edges": canny_edges,
    "threshold_binary": threshold_binary,

    # transforms
    "rotate_affine": rotate_affine,
    "rotate_affine_around_bottom_left": rotate_affine_around_bottom_left,
    "perspective_warp_demo": perspective_warp_demo,
    "zoom_remap": zoom_remap,
    "wave_warp": wave_warp,
    "fisheye_warp": fisheye_warp,

    # project-critical
    "crop": crop,
    "resize": resize,
    "add_text": add_text,
    "create_bbox_mask": create_bbox_mask,
    "create_center_mask": create_center_mask,
    "inpaint_opencv": inpaint_opencv,
}
