"""Measure horizontal stroke widths for a reproducible sample of EMNIST O images."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from torchvision.datasets import EMNIST

import design


def measure_stroke_width(
    image: np.ndarray,
    row: int = 14,
    threshold: int = 127,
) -> int | None:
    """Return the distance from the first white pixel to the next black pixel."""
    if image.ndim != 2 or not 0 <= row < image.shape[0]:
        raise ValueError("image must be 2D and contain the requested row")

    white_columns = np.flatnonzero(image[row] > threshold)
    if white_columns.size == 0:
        return None

    first_white = int(white_columns[0])
    black_columns = np.flatnonzero(image[row, first_white + 1 :] <= threshold)
    if black_columns.size == 0:
        return None

    first_black_after_white = first_white + 1 + int(black_columns[0])
    return first_black_after_white - first_white


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure row-14 stroke widths for EMNIST O samples."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="number of O images to measure (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used to select a repeatable random sample (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("stroke_width_measurements.csv"),
        help="CSV file for per-image measurements",
    )
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")

    dataset = EMNIST(
        root=design.EMNISTDir,
        split=design.EMNIST_SPLIT,
        train=True,
        download=True,
    )
    try:
        o_label = dataset.classes.index("O")
    except ValueError as error:
        raise ValueError(
            f"The EMNIST '{design.EMNIST_SPLIT}' split has no 'O' class."
        ) from error

    target_values = dataset.targets.numpy()
    o_indices = np.flatnonzero(target_values == o_label).tolist()
    if len(o_indices) < args.samples:
        raise ValueError(
            f"Found only {len(o_indices)} O images; {args.samples} were requested."
        )

    random.Random(args.seed).shuffle(o_indices)
    measurements: list[tuple[int, int]] = []
    skipped = 0
    for dataset_index in o_indices:
        # EMNIST's stored images are transposed; match the orientation used by
        # the project's TransposeTransform before scanning the horizontal row.
        image = dataset.data[dataset_index].numpy().T
        width = measure_stroke_width(image)
        if width is None:
            skipped += 1
            continue

        measurements.append((dataset_index, width))
        if len(measurements) == args.samples:
            break

    if len(measurements) < args.samples:
        raise ValueError(
            f"Measured only {len(measurements)} valid O images; "
            f"{args.samples} were requested."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("dataset_index", "stroke_width_pixels"))
        writer.writerows(measurements)

    widths = [width for _, width in measurements]
    print(f"EMNIST split: {design.EMNIST_SPLIT} (training set)")
    print(f"Class: {dataset.classes[o_label]}")
    print(f"Images measured: {len(widths)}")
    print(f"Scan row: 14; white threshold: >127")
    print(f"Average stroke width: {np.mean(widths):.3f} pixels")
    print(f"Minimum / maximum: {min(widths)} / {max(widths)} pixels")
    print(f"Skipped images without both scan transitions: {skipped}")
    print(f"Per-image measurements saved to: {args.output}")


if __name__ == "__main__":
    main()
