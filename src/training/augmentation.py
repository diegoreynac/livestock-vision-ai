from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AugmentationConfig:
    # Resize applied for both train and eval (deterministic)
    resize: Optional[Tuple[int, int]] = None  # (height, width)

    # Independent transform toggles and magnitudes
    enable_flip: bool = True
    flip_prob: float = 0.5

    enable_brightness: bool = True
    brightness_range: Tuple[float, float] = (0.9, 1.1)

    enable_noise: bool = True
    noise_std: float = 0.02

    enable_rotate90: bool = False
    rotate90_prob: float = 0.0  # probability to apply a 90/180/270 deg rotation

    conservative: bool = False  # when True, use conservative defaults

    normalize_mean: Optional[Tuple[float, ...]] = None
    normalize_std: Optional[Tuple[float, ...]] = None


def _to_float01(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0,1]."""
    img = image.astype(np.float32)
    # if values look like 0-255, scale down
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
    # compute source indices
    row_idx = (np.floor(np.arange(new_h) * (h / new_h))).astype(int)
    col_idx = (np.floor(np.arange(new_w) * (w / new_w))).astype(int)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    return img[row_idx[:, None], col_idx[None, :]]


def _validate_config(config: AugmentationConfig, channels: int):
    # resize
    if config.resize is not None:
        if (
            not isinstance(config.resize, tuple)
            or len(config.resize) != 2
            or not all(isinstance(x, int) and x > 0 for x in config.resize)
        ):
            raise ValueError("resize must be a tuple of two positive integers (height, width)")
    # flip prob
    if not (0.0 <= config.flip_prob <= 1.0):
        raise ValueError("flip_prob must be in [0,1]")
    # rotate prob
    if not (0.0 <= config.rotate90_prob <= 1.0):
        raise ValueError("rotate90_prob must be in [0,1]")
    # brightness
    if config.enable_brightness:
        if (
            not isinstance(config.brightness_range, tuple)
            or len(config.brightness_range) != 2
            or config.brightness_range[0] <= 0
            or config.brightness_range[1] <= 0
            or config.brightness_range[0] > config.brightness_range[1]
        ):
            raise ValueError("brightness_range must be a (min,max) tuple with positive numbers and min<=max")
    # noise
    if config.noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    # normalization lengths
    if config.normalize_mean is not None:
        if len(config.normalize_mean) != channels:
            raise ValueError("normalize_mean length must match image channels")
    if config.normalize_std is not None:
        if len(config.normalize_std) != channels:
            raise ValueError("normalize_std length must match image channels")


def training_preprocess(
    image: np.ndarray,
    config: AugmentationConfig,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Training-time preprocessing (randomized, reproducible via seed or rng).
    Deterministic operations: resize, normalization. Randomized ops only occur here.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    img = _to_float01(image)
    h0, w0 = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    _validate_config(config, channels)

    # apply resize deterministically
    if config.resize is not None:
        img = _nearest_resize(img, config.resize[0], config.resize[1])

    # copy augmentation parameters and apply conservative scaling
    flip_prob = config.flip_prob if config.enable_flip else 0.0
    brightness_range = config.brightness_range if config.enable_brightness else (1.0, 1.0)
    noise_std = config.noise_std if config.enable_noise else 0.0
    rotate_prob = config.rotate90_prob if config.enable_rotate90 else 0.0

    if config.conservative:
        flip_prob = min(flip_prob, 0.2)
        # move brightness range closer to 1.0
        br_mid = (brightness_range[0] + brightness_range[1]) / 2.0
        brightness_range = (1.0 - (1.0 - br_mid) * 0.5, 1.0 + (brightness_range[1] - br_mid) * 0.5)
        noise_std = noise_std * 0.5
        rotate_prob = min(rotate_prob, 0.2)

    # random flip
    if rng.random() < flip_prob:
        img = img[:, ::-1, ...]

    # random 90-degree rotation
    if rng.random() < rotate_prob:
        # choose k in {1,2,3}
        k = int(rng.integers(1, 4))
        img = np.rot90(img, k=k, axes=(0, 1))

    # random brightness
    if config.enable_brightness:
        bmin, bmax = brightness_range
        factor = float(rng.uniform(bmin, bmax))
        img = np.clip(img * factor, 0.0, 1.0)

    # gaussian noise
    if noise_std > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    # normalization
    if config.normalize_mean is not None and config.normalize_std is not None:
        img = _normalize(img, config.normalize_mean, config.normalize_std)

    return img


def eval_preprocess(
    image: np.ndarray,
    config: AugmentationConfig,
) -> np.ndarray:
    """
    Evaluation-time preprocessing: deterministic operations only (resize + normalization).
    No random augmentations are applied here.
    """
    img = _to_float01(image)
    channels = img.shape[2] if img.ndim == 3 else 1
    _validate_config(config, channels)
    if config.resize is not None:
        img = _nearest_resize(img, config.resize[0], config.resize[1])
    if config.normalize_mean is not None and config.normalize_std is not None:
        img = _normalize(img, config.normalize_mean, config.normalize_std)
    return img
