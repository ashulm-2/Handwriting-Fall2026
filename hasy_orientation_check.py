"""Compare candidate HASYv2 orientations with actual EMNIST training inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.datasets import EMNIST

import design


CLASSIFICATION_ROOT = Path.home() / "Downloads" / "classification-task"
EXTRA_CHARACTERS = ("7", "R", "q")
EMNIST_F_INDEX = 76


def _target_index(character: str) -> int:
    matches = [
        index
        for index in range(47)
        if design.CToC(index) == character
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one bymerge index for {character!r}; found {matches!r}."
        )
    return matches[0]


def _find_hasy_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    wanted = {"F", *EXTRA_CHARACTERS}
    rows: dict[str, dict[str, str]] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "latex"}.issubset(reader.fieldnames or ()):
            raise RuntimeError(f"Unexpected CSV columns in {csv_path}")

        for row in reader:
            label = row["latex"]
            normalized_path = row["path"].replace("\\", "/")
            filename = Path(normalized_path).name

            if label == "F" and filename == "v2-00058.png":
                rows["F"] = row
            elif label in wanted and label not in rows:
                rows[label] = row

    if "F" not in rows:
        raise FileNotFoundError(
            f"Required HASYv2 F image v2-00058.png not found in {csv_path}"
        )
    return rows


def _resolve_hasy_path(csv_path: Path, row: dict[str, str]) -> Path:
    relative_path = Path(row["path"].replace("\\", "/"))
    image_path = (csv_path.parent / relative_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing HASYv2 image: {image_path}")
    return image_path


def _first_emnist_index(dataset: EMNIST, character: str) -> int | None:
    target = _target_index(character)
    matches = torch.nonzero(dataset.targets == target, as_tuple=False).flatten()
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


def _forward_input(
    tensor: torch.Tensor,
    description: str,
    device: torch.device,
) -> torch.Tensor:
    batch = tensor.unsqueeze(0).to(device)
    if tuple(batch.shape) != (1, 1, 28, 28):
        raise RuntimeError(
            f"{description} has unexpected pre-forward shape {tuple(batch.shape)!r}."
        )
    if batch.dtype != torch.float32:
        raise RuntimeError(
            f"{description} has dtype {batch.dtype!r}; expected torch.float32."
        )
    if not torch.isfinite(batch).all().item():
        raise RuntimeError(f"{description} contains non-finite values.")
    return batch


def main() -> None:
    if design.DATASET_TYPE != "EMNIST" or design.EMNIST_SPLIT != "bymerge":
        raise RuntimeError(
            "This diagnostic requires the configured EMNIST bymerge mapping."
        )

    fold1_csv = CLASSIFICATION_ROOT / "fold-1" / "test.csv"
    if not fold1_csv.is_file():
        raise FileNotFoundError(f"Missing HASYv2 test CSV: {fold1_csv}")
    hasy_rows = _find_hasy_rows(fold1_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emnist = EMNIST(
        root=design.EMNISTDir,
        split="bymerge",
        train=True,
        download=False,
        transform=design.IMAGE_TRANSFORM,
    )

    if len(emnist) <= EMNIST_F_INDEX:
        raise IndexError(f"EMNIST training index {EMNIST_F_INDEX} is unavailable.")

    candidates = [("F", EMNIST_F_INDEX)]
    for character in EXTRA_CHARACTERS:
        if character not in hasy_rows:
            print(f"Skipping optional {character!r}: no fold-1 HASYv2 sample.")
            continue
        emnist_index = _first_emnist_index(emnist, character)
        if emnist_index is None:
            print(f"Skipping optional {character!r}: no EMNIST training sample.")
            continue
        candidates.append((character, emnist_index))

    panels: list[tuple[str, str, torch.Tensor]] = []
    for character, emnist_index in candidates:
        target = emnist.targets[emnist_index]
        mapped_label = design.CToC(int(target))
        if mapped_label != character:
            raise RuntimeError(
                f"EMNIST index {emnist_index} maps to {mapped_label!r}, "
                f"not {character!r}."
            )

        emnist_tensor, item_target = emnist[emnist_index]
        if design.CToC(int(item_target)) != character:
            raise RuntimeError(
                f"Transformed EMNIST item {emnist_index} has an unexpected label."
            )

        hasy_row = hasy_rows[character]
        if hasy_row["latex"] != character:
            raise RuntimeError(
                f"HASYv2 row label {hasy_row['latex']!r} does not match {character!r}."
            )
        hasy_path = _resolve_hasy_path(fold1_csv, hasy_row)
        with Image.open(hasy_path) as opened:
            hasy_image = opened.convert("L")

        hasy_a = design.IMAGE_TRANSFORM(hasy_image)
        hasy_b = torch.transpose(hasy_a, 1, 2)

        panels.extend(
            (
                (
                    character,
                    f"EMNIST training transform (index {emnist_index})",
                    _forward_input(emnist_tensor, "EMNIST training input", device),
                ),
                (
                    character,
                    "HASY A: IMAGE_TRANSFORM",
                    _forward_input(hasy_a, "HASY A input", device),
                ),
                (
                    character,
                    "HASY B: transform + inference transpose",
                    _forward_input(hasy_b, "HASY B input", device),
                ),
            )
        )
        print(f"HASY {character} image: {hasy_path}")

    figure, axes = plt.subplots(
        len(candidates),
        3,
        figsize=(10, 3.2 * len(candidates)),
        squeeze=False,
    )
    for row_index, character in enumerate(character for character, _ in candidates):
        for column, (panel_character, title, tensor) in enumerate(
            panels[row_index * 3 : row_index * 3 + 3]
        ):
            if panel_character != character:
                raise RuntimeError("Panel order no longer matches sample order.")
            axis = axes[row_index, column]
            axis.imshow(
                tensor[0, 0].detach().cpu().numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                origin="upper",
                interpolation="nearest",
            )
            axis.set_title(f"{character}: {title}\nshape={tuple(tensor.shape)}")
            axis.axis("off")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
