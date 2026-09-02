import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

GEOMETRIC_TRANSFORMS = ("horizontal_flip", "small_rotation", "zoom_scale", "resize")
PHOTOMETRIC_TRANSFORMS = (
    "brightness",
    "contrast",
    "color_saturation",
    "gaussian_noise",
    "grayscale",
    "blur",
)


@dataclass
class AugmentationConfig:
    """Configuration for preprocessing augmentations.

    Geometric transformations are distinguished from photometric transformations because
    geometric transforms may change pixel coordinates and therefore must also update
    spatial annotations such as bounding boxes or keypoints. Photometric transforms only
    alter appearance and must not modify spatial annotations.
    """

    # Resize applied for both train and eval (deterministic)
    resize: Optional[Tuple[int, int]] = None  # (height, width)

    # Geometric augmentations: defaults are off to keep the augmentation policy explicit.
    enable_flip: bool = False
    flip_prob: float = 0.5

    enable_rotation: bool = False
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    rotation_prob: float = 0.5

    enable_zoom: bool = False
    zoom_range: Tuple[float, float] = (0.9, 1.1)
    zoom_prob: float = 0.5

    # Deprecated compatibility field retained for older API usage. The realistic policy
    # for cattle imagery is a small rotation range rather than a 90-degree rotation.
    enable_rotate90: bool = False
    rotate90_prob: float = 0.0

    # Photometric augmentations: defaults are off unless the training configuration opts in.
    enable_brightness: bool = False
    brightness_range: Tuple[float, float] = (0.85, 1.15)

    enable_contrast: bool = False
    contrast_range: Tuple[float, float] = (0.85, 1.15)

    enable_color: bool = False
    color_range: Tuple[float, float] = (0.8, 1.2)

    enable_noise: bool = False
    noise_std: float = 0.02

    enable_grayscale: bool = False
    grayscale_prob: float = 0.2

    enable_blur: bool = False
    blur_sigma: Tuple[float, float] = (0.1, 1.0)
    blur_prob: float = 0.2

    conservative: bool = False  # when True, use conservative defaults

    normalize_mean: Optional[Tuple[float, ...]] = None
    normalize_std: Optional[Tuple[float, ...]] = None


def _to_float01(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0,1]."""
    img = image.astype(np.float32)
    if img.max() > 2.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _normalize(img: np.ndarray, mean, std) -> np.ndarray:
    mean_arr = np.array(mean, dtype=np.float32).reshape((1, 1, -1))
    std_arr = np.array(std, dtype=np.float32).reshape((1, 1, -1))
    return (img - mean_arr) / std_arr


def _nearest_resize(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Simple nearest-neighbor resize without external deps."""
    h, w = img.shape[:2]
    if (h, w) == (new_h, new_w):
        return img
    row_idx = (np.floor(np.arange(new_h) * (h / new_h))).astype(int)
    col_idx = (np.floor(np.arange(new_w) * (w / new_w))).astype(int)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    return img[row_idx[:, None], col_idx[None, :]]


def _to_2d_box_array(boxes: Any) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape((0, 4))
    if arr.ndim == 1:
        if arr.shape[0] != 4:
            raise ValueError("Each box must have four coordinates (x1, y1, x2, y2)")
        return arr.reshape((1, 4))
    if arr.ndim == 2 and arr.shape[1] >= 4:
        return arr[:, :4]
    raise ValueError("Bounding boxes must be 1D or 2D with at least four coordinates")


def _update_boxes_for_flip(boxes: np.ndarray, width: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0], out[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    return out


def _rotate_point_matrix(points: np.ndarray, angle_deg: float, cx: float, cy: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = points[:, 0] - cx
    y = points[:, 1] - cy
    out_x = x * cos_a - y * sin_a + cx
    out_y = x * sin_a + y * cos_a + cy
    return np.column_stack((out_x, out_y))


def _update_boxes_for_rotation(boxes: np.ndarray, height: int, width: int, angle_deg: float) -> np.ndarray:
    cx, cy = width / 2.0, height / 2.0
    corners = np.array(
        [
            [boxes[:, 0], boxes[:, 1]],
            [boxes[:, 2], boxes[:, 1]],
            [boxes[:, 0], boxes[:, 3]],
            [boxes[:, 2], boxes[:, 3]],
        ],
        dtype=np.float32,
    )
    corners = np.moveaxis(corners, 0, 1).reshape((-1, 2))
    rotated = _rotate_point_matrix(corners, angle_deg, cx, cy)
    x0 = rotated[:, 0].reshape((-1, 4)).min(axis=1)
    y0 = rotated[:, 1].reshape((-1, 4)).min(axis=1)
    x1 = rotated[:, 0].reshape((-1, 4)).max(axis=1)
    y1 = rotated[:, 1].reshape((-1, 4)).max(axis=1)
    return np.column_stack((x0, y0, x1, y1))


def _update_boxes_for_zoom(boxes: np.ndarray, scale_x: float, scale_y: float, offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = out[:, 0] * scale_x + offset_x
    out[:, 2] = out[:, 2] * scale_x + offset_x
    out[:, 1] = out[:, 1] * scale_y + offset_y
    out[:, 3] = out[:, 3] * scale_y + offset_y
    return out


def _center_zoom_transform(height: int, width: int, scale: float) -> Tuple[float, float, float, float]:
    """Compute the exact affine transform applied to coordinates by ``_center_zoom``.

    The transform is ``x' = x * scale_x + offset_x`` and ``y' = y * scale_y + offset_y``.
    This mirrors ``_center_zoom`` exactly, including its centered crop/pad behavior and
    integer ``// 2`` offset computation, so annotations stay consistent with the image.
    """
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    scale_x = scale
    scale_y = scale
    if new_h >= height and new_w >= width:
        y0 = (new_h - height) // 2
        x0 = (new_w - width) // 2
        offset_x = -float(x0)
        offset_y = -float(y0)
    elif new_h <= height and new_w <= width:
        y0 = (height - new_h) // 2
        x0 = (width - new_w) // 2
        offset_x = float(x0)
        offset_y = float(y0)
    else:
        offset_x = 0.0
        offset_y = 0.0
    return scale_x, scale_y, offset_x, offset_y


def update_spatial_annotations(annotations: Any, *, transform_name: str, source_shape: Tuple[int, int], target_shape: Tuple[int, int], angle_deg: float = 0.0, zoom_scale: Optional[float] = None) -> Any:
    """Apply a single geometric transform to spatial annotations.

    This helper is intentionally compositional: the caller may apply multiple geometric
    transforms in sequence, and the same annotation object is updated in that exact order.
    Photometric transforms never call this helper.
    """
    if annotations is None:
        return None
    if not isinstance(annotations, dict):
        return annotations

    boxes = annotations.get("boxes", annotations.get("bboxes"))
    if boxes is not None:
        arr = _to_2d_box_array(boxes)
        if transform_name == "resize":
            scale_x = target_shape[1] / max(source_shape[1], 1)
            scale_y = target_shape[0] / max(source_shape[0], 1)
            arr = _update_boxes_for_zoom(arr, scale_x, scale_y)
        elif transform_name == "horizontal_flip":
            arr = _update_boxes_for_flip(arr, target_shape[1])
        elif transform_name == "small_rotation":
            arr = _update_boxes_for_rotation(arr, source_shape[0], source_shape[1], angle_deg)
        elif transform_name == "zoom_scale":
            scale = zoom_scale if zoom_scale is not None else target_shape[0] / max(source_shape[0], 1)
            scale_x, scale_y, offset_x, offset_y = _center_zoom_transform(source_shape[0], source_shape[1], scale)
            arr = _update_boxes_for_zoom(arr, scale_x, scale_y, offset_x, offset_y)
        annotations["boxes"] = arr
        if "bboxes" in annotations:
            annotations["bboxes"] = arr

    keypoints = annotations.get("keypoints")
    if keypoints is not None:
        arr = np.asarray(keypoints, dtype=np.float32)
        if transform_name == "resize":
            scale_x = target_shape[1] / max(source_shape[1], 1)
            scale_y = target_shape[0] / max(source_shape[0], 1)
            arr = arr.copy()
            arr[..., 0] *= scale_x
            arr[..., 1] *= scale_y
        elif transform_name == "horizontal_flip":
            arr = arr.copy()
            arr[..., 0] = target_shape[1] - arr[..., 0]
        elif transform_name == "small_rotation":
            cx = source_shape[1] / 2.0
            cy = source_shape[0] / 2.0
            points = arr.reshape((-1, 2))
            rotated = _rotate_point_matrix(points, angle_deg, cx, cy)
            arr = rotated.reshape(keypoints.shape)
        elif transform_name == "zoom_scale":
            scale = zoom_scale if zoom_scale is not None else target_shape[0] / max(source_shape[0], 1)
            scale_x, scale_y, offset_x, offset_y = _center_zoom_transform(source_shape[0], source_shape[1], scale)
            arr = arr.copy()
            arr[..., 0] = arr[..., 0] * scale_x + offset_x
            arr[..., 1] = arr[..., 1] * scale_y + offset_y
        annotations["keypoints"] = arr

    history = annotations.setdefault("_geometric_transform_history", [])
    history.append(
        {
            "transform": transform_name,
            "source_shape": source_shape,
            "target_shape": target_shape,
            "angle_deg": angle_deg,
        }
    )
    annotations["_spatial_annotations_updated"] = True
    annotations["_last_geometric_transform"] = transform_name
    return annotations


def _validate_config(config: AugmentationConfig, channels: int):
    if config.resize is not None:
        if (
            not isinstance(config.resize, tuple)
            or len(config.resize) != 2
            or not all(isinstance(x, int) and x > 0 for x in config.resize)
        ):
            raise ValueError("resize must be a tuple of two positive integers (height, width)")

    if not (0.0 <= config.flip_prob <= 1.0):
        raise ValueError("flip_prob must be in [0,1]")
    if not (0.0 <= config.rotation_prob <= 1.0):
        raise ValueError("rotation_prob must be in [0,1]")
    if not (0.0 <= config.zoom_prob <= 1.0):
        raise ValueError("zoom_prob must be in [0,1]")
    if not (0.0 <= config.rotate90_prob <= 1.0):
        raise ValueError("rotate90_prob must be in [0,1]")

    if config.enable_rotation:
        if (
            not isinstance(config.rotation_range, tuple)
            or len(config.rotation_range) != 2
            or not np.isfinite(config.rotation_range[0])
            or not np.isfinite(config.rotation_range[1])
            or config.rotation_range[0] > config.rotation_range[1]
            or abs(config.rotation_range[0]) > 90.0
            or abs(config.rotation_range[1]) > 90.0
        ):
            raise ValueError("rotation_range must be a (min,max) tuple in [-90,90] degrees")
    if config.enable_zoom:
        if (
            not isinstance(config.zoom_range, tuple)
            or len(config.zoom_range) != 2
            or config.zoom_range[0] <= 0
            or config.zoom_range[1] <= 0
            or config.zoom_range[0] > config.zoom_range[1]
        ):
            raise ValueError("zoom_range must be a (min,max) positive tuple with min<=max")

    if config.enable_brightness:
        if (
            not isinstance(config.brightness_range, tuple)
            or len(config.brightness_range) != 2
            or config.brightness_range[0] <= 0
            or config.brightness_range[1] <= 0
            or config.brightness_range[0] > config.brightness_range[1]
        ):
            raise ValueError("brightness_range must be a (min,max) tuple with positive numbers and min<=max")
    if config.enable_contrast:
        if (
            not isinstance(config.contrast_range, tuple)
            or len(config.contrast_range) != 2
            or config.contrast_range[0] <= 0
            or config.contrast_range[1] <= 0
            or config.contrast_range[0] > config.contrast_range[1]
        ):
            raise ValueError("contrast_range must be a (min,max) tuple with positive numbers and min<=max")
    if config.enable_color:
        if (
            not isinstance(config.color_range, tuple)
            or len(config.color_range) != 2
            or config.color_range[0] <= 0
            or config.color_range[1] <= 0
            or config.color_range[0] > config.color_range[1]
        ):
            raise ValueError("color_range must be a (min,max) tuple with positive numbers and min<=max")
    if config.enable_blur:
        if (
            not isinstance(config.blur_sigma, tuple)
            or len(config.blur_sigma) != 2
            or config.blur_sigma[0] < 0
            or config.blur_sigma[1] < 0
            or config.blur_sigma[0] > config.blur_sigma[1]
        ):
            raise ValueError("blur_sigma must be a (min,max) tuple with non-negative values and min<=max")
    if config.noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not (0.0 <= config.grayscale_prob <= 1.0):
        raise ValueError("grayscale_prob must be in [0,1]")
    if not (0.0 <= config.blur_prob <= 1.0):
        raise ValueError("blur_prob must be in [0,1]")
    if config.normalize_mean is not None:
        if len(config.normalize_mean) != channels:
            raise ValueError("normalize_mean length must match image channels")
    if config.normalize_std is not None:
        if len(config.normalize_std) != channels:
            raise ValueError("normalize_std length must match image channels")


def _apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return img
    radius = max(1, int(math.ceil(3.0 * sigma)))
    if radius % 2 == 0:
        radius += 1
    kernel = np.exp(-(np.arange(-(radius // 2), radius // 2 + 1) ** 2) / (2.0 * sigma * sigma))
    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum()

    out = img.copy()
    channels = img.shape[-1] if img.ndim == 3 else 1
    for channel_index in range(channels):
        if img.ndim == 3:
            channel = out[:, :, channel_index].copy()
        else:
            channel = out.copy()
        blurred_rows = np.empty_like(channel)
        for row in range(channel.shape[0]):
            row_vals = channel[row, :]
            padded = np.pad(row_vals, (radius // 2, radius // 2), mode="edge")
            blurred_rows[row, :] = np.convolve(padded, kernel, mode="valid")
        blurred_cols = np.empty_like(channel)
        for col in range(channel.shape[1]):
            col_vals = blurred_rows[:, col]
            padded = np.pad(col_vals, (radius // 2, radius // 2), mode="edge")
            blurred_cols[:, col] = np.convolve(padded, kernel, mode="valid")
        if img.ndim == 3:
            out[:, :, channel_index] = blurred_cols
        else:
            out = blurred_cols
    return np.clip(out, 0.0, 1.0)


def _center_zoom(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    scaled = _nearest_resize(img, new_h, new_w)
    if new_h >= h and new_w >= w:
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        return scaled[y0 : y0 + h, x0 : x0 + w, ...]
    if new_h <= h and new_w <= w:
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        canvas = np.zeros_like(img)
        canvas[y0 : y0 + new_h, x0 : x0 + new_w, ...] = scaled
        return canvas
    return scaled[:h, :w, ...]


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if angle_deg == 0.0:
        return img
    h, w = img.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            src_x = dx * cos_a + dy * sin_a + cx
            src_y = -dx * sin_a + dy * cos_a + cy
            x0 = int(round(src_x))
            y0 = int(round(src_y))
            if 0 <= x0 < w and 0 <= y0 < h:
                out[y, x] = img[y0, x0]
    return out


def training_preprocess(
    image: np.ndarray,
    config: AugmentationConfig,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    annotations: Any = None,
) -> np.ndarray:
    """Training-time preprocessing.

    Randomized transforms are allowed here. Geometric transforms are applied in the same
    sequence as the image operations, and the same sequence is applied to any provided
    spatial annotations. Photometric transforms do not modify annotations.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    img = _to_float01(image)
    current_shape = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    _validate_config(config, channels)

    if config.resize is not None:
        source_shape = img.shape[:2]
        img = _nearest_resize(img, config.resize[0], config.resize[1])
        if annotations is not None:
            update_spatial_annotations(
                annotations,
                transform_name="resize",
                source_shape=source_shape,
                target_shape=img.shape[:2],
            )
        current_shape = img.shape[:2]

    flip_prob = config.flip_prob if config.enable_flip else 0.0
    brightness_range = config.brightness_range if config.enable_brightness else (1.0, 1.0)
    contrast_range = config.contrast_range if config.enable_contrast else (1.0, 1.0)
    color_range = config.color_range if config.enable_color else (1.0, 1.0)
    noise_std = config.noise_std if config.enable_noise else 0.0
    rotation_prob = config.rotation_prob if config.enable_rotation else 0.0
    if config.enable_rotate90:
        rotation_prob = max(rotation_prob, config.rotate90_prob)
    zoom_prob = config.zoom_prob if config.enable_zoom else 0.0
    grayscale_prob = config.grayscale_prob if config.enable_grayscale else 0.0
    blur_prob = config.blur_prob if config.enable_blur else 0.0

    if config.conservative:
        flip_prob = min(flip_prob, 0.2)
        br_mid = (brightness_range[0] + brightness_range[1]) / 2.0
        brightness_range = (1.0 - (1.0 - br_mid) * 0.5, 1.0 + (brightness_range[1] - br_mid) * 0.5)
        contrast_range = (1.0 - (contrast_range[0] - 1.0) * 0.5, 1.0 + (contrast_range[1] - 1.0) * 0.5)
        color_range = (1.0 - (color_range[0] - 1.0) * 0.5, 1.0 + (color_range[1] - 1.0) * 0.5)
        noise_std *= 0.5
        rotation_prob = min(rotation_prob, 0.2)
        zoom_prob = min(zoom_prob, 0.2)
        grayscale_prob = min(grayscale_prob, 0.2)
        blur_prob = min(blur_prob, 0.2)

    angle_deg = 0.0

    if rng.random() < flip_prob:
        source_shape = img.shape[:2]
        img = img[:, ::-1, ...]
        if annotations is not None:
            update_spatial_annotations(
                annotations,
                transform_name="horizontal_flip",
                source_shape=source_shape,
                target_shape=img.shape[:2],
            )
        current_shape = img.shape[:2]

    if rng.random() < rotation_prob:
        source_shape = img.shape[:2]
        if config.enable_rotate90 and not config.enable_rotation:
            angle_deg = float(rng.uniform(-15.0, 15.0))
        else:
            angle_deg = float(rng.uniform(config.rotation_range[0], config.rotation_range[1]))
        img = _rotate_image(img, angle_deg)
        if annotations is not None:
            update_spatial_annotations(
                annotations,
                transform_name="small_rotation",
                source_shape=source_shape,
                target_shape=img.shape[:2],
                angle_deg=angle_deg,
            )
        current_shape = img.shape[:2]

    if rng.random() < zoom_prob:
        source_shape = img.shape[:2]
        scale = float(rng.uniform(config.zoom_range[0], config.zoom_range[1]))
        img = _center_zoom(img, scale)
        if annotations is not None:
            update_spatial_annotations(
                annotations,
                transform_name="zoom_scale",
                source_shape=source_shape,
                target_shape=img.shape[:2],
                zoom_scale=scale,
            )
        current_shape = img.shape[:2]

    if config.enable_brightness:
        bmin, bmax = brightness_range
        factor = float(rng.uniform(bmin, bmax))
        img = np.clip(img * factor, 0.0, 1.0)

    if config.enable_contrast:
        cmin, cmax = contrast_range
        factor = float(rng.uniform(cmin, cmax))
        img = np.clip((img - 0.5) * factor + 0.5, 0.0, 1.0)

    if config.enable_color:
        cmin, cmax = color_range
        sat = float(rng.uniform(cmin, cmax))
        if img.shape[-1] >= 3:
            gray = img.mean(axis=-1, keepdims=True)
            img = np.clip((1.0 - sat) * gray + sat * img, 0.0, 1.0)

    if noise_std > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    if config.enable_grayscale and rng.random() < grayscale_prob:
        if img.shape[-1] >= 3:
            img = np.repeat(img.mean(axis=-1, keepdims=True), img.shape[-1], axis=-1)

    if config.enable_blur and rng.random() < blur_prob:
        sigma = float(rng.uniform(config.blur_sigma[0], config.blur_sigma[1]))
        img = _apply_gaussian_blur(img, sigma)

    if config.normalize_mean is not None and config.normalize_std is not None:
        img = _normalize(img, config.normalize_mean, config.normalize_std)

    return img


def eval_preprocess(
    image: np.ndarray,
    config: AugmentationConfig,
    annotations: Any = None,
) -> np.ndarray:
    """Evaluation-time preprocessing: deterministic operations only (resize + normalization).
    Resize remains a geometric operation and therefore updates annotations when provided.
    """
    img = _to_float01(image)
    channels = img.shape[2] if img.ndim == 3 else 1
    _validate_config(config, channels)
    if config.resize is not None:
        source_shape = img.shape[:2]
        img = _nearest_resize(img, config.resize[0], config.resize[1])
        if annotations is not None:
            update_spatial_annotations(
                annotations,
                transform_name="resize",
                source_shape=source_shape,
                target_shape=img.shape[:2],
            )
    if config.normalize_mean is not None and config.normalize_std is not None:
        img = _normalize(img, config.normalize_mean, config.normalize_std)
    return img
