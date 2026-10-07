"""
Charles Tran
Sept 24, 2026
MURL Fall 2026
Convert handwritten character images into clean 28x28 training samples.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


class PreprocessingError(ValueError):
    """Raised when an image cannot be converted into a valid character sample."""


# PIPELINE CHECKPOINT P01 — LOAD IMAGE + GREYSCALE
# The pipeline begins by reading the source image, fixing orientation, and converting
# it to grayscale before any foreground extraction is attempted.
def _load_grayscale(source: str | Path | Image.Image) -> Image.Image:
    """Open an image, keep metadata-safe orientation, and normalize it to grayscale."""
    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).copy()

    # If the source uses transparency, flatten it onto a white background before conversion.
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        white_background = Image.new("RGBA", image.size, "white")
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        image = Image.alpha_composite(white_background, image)

    return image.convert("L")


def _otsu_threshold(values: np.ndarray) -> int:
    """Use Otsu's method to choose the best foreground/background split."""
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    levels = np.arange(256)

    cumulative_weight = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * levels)
    total_mean = cumulative_mean[-1]

    between_class_variance = (
        (total_mean * cumulative_weight - cumulative_mean) ** 2
        / (cumulative_weight * (1 - cumulative_weight) + 1e-12)
    )
    return int(np.argmax(between_class_variance))


# PIPELINE CHECKPOINT P03 — FIX POLARITY
# This stage decides the foreground/background polarity and converts the grayscale image
# into a binary mask that isolates the character.
def _binary_mask(image: Image.Image, use_background_correction: bool = True) -> np.ndarray:
    """Create a binary mask that isolates the character from the background."""
    values = np.asarray(ImageOps.autocontrast(image), dtype=np.uint8)
    threshold = _otsu_threshold(values)
    mask = values > threshold

    # If the page background is uneven or very noisy, estimate it with a blur and re-threshold.
    if use_background_correction and (mask.mean() > 0.65 or mask.mean() < 0.002):
        radius = max(8, min(image.size) // 12)
        blurred_background = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        corrected = np.asarray(image, dtype=np.float32) - blurred_background + 128
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        threshold = _otsu_threshold(corrected)
        mask = corrected > threshold

    # Most handwritten characters use a minority foreground, so flip the mask when needed.
    if mask.mean() > 0.5:
        mask = ~mask

    return mask


def _remove_small_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    """Delete tiny disconnected blobs that usually come from noise or borders."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)

    for row, column in zip(*np.where(mask & ~visited)):
        if visited[row, column]:
            continue

        stack = [(int(row), int(column))]
        visited[row, column] = True
        component = []

        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))

            for next_row in range(max(0, current_row - 1), min(height, current_row + 2)):
                for next_column in range(max(0, current_column - 1), min(width, current_column + 2)):
                    if mask[next_row, next_column] and not visited[next_row, next_column]:
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))

        if len(component) >= minimum_pixels:
            rows, columns = zip(*component)
            kept[rows, columns] = True

    return kept


# PIPELINE CHECKPOINT P05 — CROP TO CHARACTER
# Remove border noise and keep only the tight bounding box around the ink area.
def _crop_mask(mask: np.ndarray) -> np.ndarray:
    """Trim the blank border and keep only the true character bounding box."""
    height, width = mask.shape
    border = max(1, round(min(height, width) * 0.02))

    trimmed = mask.copy()
    trimmed[:border, :] = False
    trimmed[-border:, :] = False
    trimmed[:, :border] = False
    trimmed[:, -border:] = False

    cleaned = _remove_small_components(trimmed, max(2, mask.size // 10000))
    rows, columns = np.where(cleaned)

    if len(rows) == 0:
        raise PreprocessingError("no character pixels found")

    bounding_box_pixels = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
    if bounding_box_pixels > 0.9 * mask.size:
        raise PreprocessingError("character bounding box covers nearly the whole image")

    return cleaned[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]


def preprocess_image(
    source: str | Path | Image.Image,
    character_size: int = 20,
    output_size: int = 28,
    center_of_mass: bool = True,
    thicken: bool = True,
) -> Image.Image:
    """Return a validated, centered grayscale character image in a 28x28 canvas."""
    if character_size >= output_size or character_size < 1:
        raise ValueError("character_size must be positive and smaller than output_size")

    # PIPELINE CHECKPOINT P01 — LOAD IMAGE + GREYSCALE
    grayscale = _load_grayscale(source)

    # PIPELINE CHECKPOINT P03 — FIX POLARITY
    mask = _crop_mask(_binary_mask(grayscale))
    ink_ratio = float(mask.mean())

    # PIPELINE CHECKPOINT P06 — PAD TO SQUARE
    character = Image.fromarray(mask.astype(np.uint8) * 255)
    side = max(character.size)
    square = Image.new("L", (side, side), 0)
    square.paste(character, ((side - character.width) // 2, (side - character.height) // 2))

    # PIPELINE CHECKPOINT P08 — RESIZE TO 20x20
    character = square.resize((character_size, character_size), Image.Resampling.LANCZOS)

    # PIPELINE CHECKPOINT P07 — THICKEN IF THIN
    if thicken and ink_ratio < 0.12:
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.asarray(mask_image.filter(ImageFilter.MaxFilter(3)), dtype=np.uint8) > 0

    # PIPELINE CHECKPOINT P09 — CENTER ON A 28x28 CANVAS
    canvas = Image.new("L", (output_size, output_size), 0)
    offset = (output_size - character_size) // 2
    canvas.paste(character, (offset, offset))

    if center_of_mass:
        pixels = np.asarray(canvas, dtype=np.float32)
        total = pixels.sum()
        if total <= 0:
            raise PreprocessingError("character became blank after resizing")

        rows, columns = np.indices(pixels.shape)
        shift_row = round(output_size / 2 - (rows * pixels).sum() / total)
        shift_column = round(output_size / 2 - (columns * pixels).sum() / total)

        canvas = canvas.transform(
            canvas.size,
            Image.Transform.AFFINE,
            (1, 0, -shift_column, 0, 1, -shift_row),
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )

    # PIPELINE CHECKPOINT P10 — VALIDATE
    extrema = canvas.getextrema()
    fraction = np.count_nonzero(np.asarray(canvas)) / (output_size * output_size)

    if canvas.size != (output_size, output_size) or canvas.mode != "L" or extrema[1] == 0:
        raise PreprocessingError("output is blank or has the wrong format")
    if not 0.002 <= fraction <= 0.75:
        raise PreprocessingError(f"white-pixel fraction is out of range: {fraction:.3f}")

    return canvas


def preprocess_file(source: str | Path, destination: str | Path) -> None:
    """Process a single image and save it as a PNG file."""
    result = preprocess_image(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, format="PNG")


def preprocess_directory(source_directory: str | Path, output_directory: str | Path) -> Path:
    """Process every supported image in a folder and write a CSV log of results."""
    # PIPELINE CHECKPOINT P11 — SAVE
    source_directory = Path(source_directory)
    output_directory = Path(output_directory)
    rejected_directory = output_directory / "rejected"

    output_directory.mkdir(parents=True, exist_ok=True)
    rejected_directory.mkdir(parents=True, exist_ok=True)

    records = []
    for source in sorted(source_directory.iterdir()):
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue

        destination = output_directory / f"{source.stem}.png"
        try:
            preprocess_file(source, destination)
            records.append((source.name, "accepted", ""))
        except (OSError, PreprocessingError, ValueError) as error:
            records.append((source.name, "rejected", str(error)))

    log_path = output_directory / "preprocessing_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(("file", "status", "reason"))
        writer.writerows(records)

    return log_path


if __name__ == "__main__":
    preprocess_directory("charset", "output")


