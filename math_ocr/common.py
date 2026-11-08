"""Shared helpers for every OCR model: loading images and picking a device."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

ImageInput = Union[str, os.PathLike, Image.Image, np.ndarray]


def load_image(image: ImageInput) -> Image.Image:
    """Return the image as an upright RGB PIL image with no transparency.

    Accepts a file path, a PIL image, or an RGB/grayscale numpy array.

    Two problems this fixes before any model sees the image:
    - Phone photos are often stored sideways, with an EXIF tag saying how to rotate
      them. exif_transpose applies that rotation so the formula is upright.
    - PNGs exported from a web canvas usually have a transparent background. Dropping
      the alpha channel would leave black ink on a black background, so transparent
      pixels are composited onto white paper instead.
    """
    if isinstance(image, (str, os.PathLike)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        img.load()
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    img = ImageOps.exif_transpose(img)

    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        # 16-bit grayscale (some scanners): scale down to 8-bit.
        arr = np.asarray(img, dtype=np.float32)
        img = Image.fromarray((arr / max(arr.max(), 1) * 255).astype(np.uint8))

    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        paper = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(paper, img)

    return img.convert("RGB")


def shrink_to_fit(img: Image.Image, max_pixels: int | None) -> Image.Image:
    """Downscale so width*height <= max_pixels, keeping the aspect ratio."""
    if not max_pixels:
        return img
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def pick_device(device: str | None = None) -> str:
    """'cuda' if a GPU is usable, else 'cpu', unless the caller chose one."""
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
