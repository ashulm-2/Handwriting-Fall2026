"""Evaluate the existing EMNIST-bymerge CNN on the pooled HASYv2 test folds.

The visual orientation check for F, 7, R, and q showed upright HASYv2 tensors
with path B: IMAGE_TRANSFORM followed by MakePrediction's transpose. This is
visual orientation evidence, not an accuracy-based choice.
"""

from __future__ import annotations

import csv
import inspect
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

import CNN as cnn_architecture
import design
import preprocessing_final


FOLDS = tuple(range(1, 11))
CLASSIFICATION_ROOT = Path.home() / "Downloads" / "classification-task"
CHECKPOINT_PATH = Path(__file__).with_name("cnn_mnist_1.pt")
EXPECTED_ELIGIBLE_COUNT = 3772
BATCH_SIZE = 64

SUPPORTED_CLASSES = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + tuple(
    "abdefghnqr"
)
EXPECTED_BYMERGE = SUPPORTED_CLASSES + ("t",)
EXPECTED_TRANSFORM_ORDER = (
    "GrayscaleTransform",
    "ForegroundMaskTransform",
    "CropCharacterTransform",
    "PadToSquareTransform",
    "ResizeCharacterTransform",
    "CenterOnCanvasTransform",
    "ThickenToEMNISTWidthTransform",
    "ValidateCharacterTransform",
    "ToTensor",
    "TransposeTransform",
)
EXPECTED_TARGET_WIDTH = 4.345
IMAGE_READ_ERRORS = (
    UnidentifiedImageError,
    OSError,
    SyntaxError,
    Image.DecompressionBombError,
)


def _source_lines(function: Any, statement: str, expected_count: int = 1) -> list[int]:
    source, first_line = inspect.getsourcelines(function)
    matches = [
        first_line + offset
        for offset, text in enumerate(source)
        if text.strip() == statement
    ]
    if len(matches) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} occurrence(s) of {statement!r} in "
            f"{function.__qualname__}; found {len(matches)}."
        )
    return matches


def _source_line(function: Any, statement: str) -> int:
    return _source_lines(function, statement)[0]


def _verify_preprocessing_source() -> None:
    if design.DATASET_TYPE != "EMNIST":
        raise RuntimeError(f"Expected EMNIST; found {design.DATASET_TYPE!r}.")
    if design.EMNIST_SPLIT != "bymerge":
        raise RuntimeError(f"Expected bymerge; found {design.EMNIST_SPLIT!r}.")
    if len(SUPPORTED_CLASSES) != 46 or len(set(SUPPORTED_CLASSES)) != 46:
        raise RuntimeError("Supported class list must contain 46 unique labels.")

    actual_mapping = tuple(design.CToC(index) for index in range(47))
    if actual_mapping != EXPECTED_BYMERGE:
        raise RuntimeError("CToC mapping differs from the verified bymerge order.")

    transform_stages = tuple(design.IMAGE_TRANSFORM.transforms)
    stage_names = tuple(type(stage).__name__ for stage in transform_stages)
    if stage_names != EXPECTED_TRANSFORM_ORDER:
        raise RuntimeError(f"Unexpected transform composition: {stage_names!r}")

    mask_stages = [
        stage for stage in transform_stages
        if type(stage).__name__ == "ForegroundMaskTransform"
    ]
    thickener_stages = [
        stage for stage in transform_stages
        if type(stage).__name__ == "ThickenToEMNISTWidthTransform"
    ]
    if len(mask_stages) != 1 or len(thickener_stages) != 1:
        raise RuntimeError("Expected exactly one mask stage and one thickener stage.")

    mask_function = design._binary_mask
    if mask_function is not preprocessing_final._binary_mask:
        raise RuntimeError("design._binary_mask is not the inspected implementation.")

    project_root = Path(__file__).resolve().parent
    mask_file = Path(inspect.getsourcefile(mask_function)).resolve()
    thickener_method = type(thickener_stages[0]).__call__
    thickener_file = Path(inspect.getsourcefile(thickener_method)).resolve()
    if mask_file != project_root / "preprocessing_final.py":
        raise RuntimeError(f"Unexpected _binary_mask source file: {mask_file}")
    if thickener_file != project_root / "design.py":
        raise RuntimeError(f"Unexpected thickener source file: {thickener_file}")

    if not math.isclose(
        thickener_stages[0].target_width,
        EXPECTED_TARGET_WIDTH,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Unexpected stroke-width target: {thickener_stages[0].target_width!r}"
        )

    mask_lines = {
        "correction_condition": _source_line(
            mask_function,
            "if use_background_correction and (mask.mean() > 0.65 or mask.mean() < 0.002):",
        ),
        "correction_entry": _source_line(
            mask_function,
            "radius = max(8, min(image.size) // 12)",
        ),
        "correction_assignment": _source_line(
            mask_function,
            "mask = corrected > threshold",
        ),
        "correction_completion": _source_line(mask_function, "if mask.mean() > 0.5:"),
        "inversion": _source_line(mask_function, "mask = ~mask"),
        "return": _source_line(mask_function, "return mask"),
    }
    if not (
        mask_lines["correction_condition"]
        < mask_lines["correction_entry"]
        < mask_lines["correction_assignment"]
        < mask_lines["correction_completion"]
        < mask_lines["inversion"]
        < mask_lines["return"]
    ):
        raise RuntimeError("_binary_mask statement order changed.")

    width_assignments = _source_lines(
        thickener_method,
        "stroke_width = _horizontal_stroke_width(img)",
        expected_count=2,
    )
    thickener_lines = {
        "initial_width": width_assignments[0],
        "loop_condition": _source_line(
            thickener_method,
            "while stroke_width is not None and stroke_width < self.target_width:",
        ),
        "filter": _source_line(
            thickener_method,
            "img = img.filter(ImageFilter.MaxFilter(3))",
        ),
        "remeasurement": width_assignments[1],
        "return": _source_line(thickener_method, "return img"),
    }
    if not (
        thickener_lines["initial_width"]
        < thickener_lines["loop_condition"]
        < thickener_lines["filter"]
        < thickener_lines["remeasurement"]
        < thickener_lines["return"]
    ):
        raise RuntimeError("Thickener statement order changed.")

    foreground_method = type(mask_stages[0]).__call__
    if inspect.getsource(foreground_method).count("mask = _binary_mask(img)") != 1:
        raise RuntimeError("ForegroundMaskTransform no longer calls _binary_mask once.")


def _sample_context(sample: dict[str, Any]) -> str:
    return (
        f"fold-{sample['fold']}, CSV line {sample['csv_line']}, "
        f"character {sample['character']!r}, image {sample['image_path']}"
    )


def _load_population() -> list[dict[str, Any]]:
    samples = []
    if len(SUPPORTED_CLASSES) != 46 or len(set(SUPPORTED_CLASSES)) != 46:
        raise RuntimeError("Supported class list must contain 46 unique labels.")

    for fold in FOLDS:
        csv_path = CLASSIFICATION_ROOT / f"fold-{fold}" / "test.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing test CSV: {csv_path}")

        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"path", "latex"}.issubset(reader.fieldnames or ()):
                raise RuntimeError(f"Unexpected CSV columns in {csv_path}")

            for row in reader:
                label = row["latex"]
                if label not in SUPPORTED_CLASSES:
                    continue

                relative_path = Path(row["path"].replace("\\", "/"))
                if relative_path.is_absolute():
                    raise RuntimeError(
                        f"Expected a relative image path in {csv_path}, "
                        f"line {reader.line_num}: {row['path']!r}"
                    )

                samples.append(
                    {
                        "fold": fold,
                        "character": label,
                        "csv_line": reader.line_num,
                        "csv_path": csv_path,
                        "image_path": (csv_path.parent / relative_path).resolve(),
                    }
                )

    if len(samples) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_COUNT} eligible rows, "
            f"found {len(samples)}. Stop for investigation."
        )

    first_sample_by_path: dict[Path, dict[str, Any]] = {}
    duplicate_paths = []
    for sample in samples:
        path = sample["image_path"]
        if path in first_sample_by_path:
            duplicate_paths.append((path, first_sample_by_path[path], sample))
        else:
            first_sample_by_path[path] = sample

    if duplicate_paths:
        path, first, duplicate = duplicate_paths[0]
        raise RuntimeError(
            f"Found {len(duplicate_paths)} duplicate resolved image path(s). "
            f"Example: {path} occurs in fold-{first['fold']} line "
            f"{first['csv_line']} and fold-{duplicate['fold']} line "
            f"{duplicate['csv_line']}."
        )
    return samples


def _format_reason(exception: Exception) -> str:
    message = str(exception).strip()
    return (
        f"{type(exception).__name__}: {message}"
        if message
        else type(exception).__name__
    )


def _validate_preprocessed_tensor(
    tensor: Any,
    sample: dict[str, Any],
) -> torch.Tensor:
    context = _sample_context(sample)
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"Transform returned {type(tensor).__name__}, not a tensor, at {context}."
        )
    if tuple(tensor.shape) != (1, 28, 28):
        raise RuntimeError(
            f"Transform returned shape {tuple(tensor.shape)!r} at {context}."
        )
    if tensor.dtype != torch.float32:
        raise RuntimeError(
            f"Transform returned dtype {tensor.dtype!r}, not torch.float32, at {context}."
        )
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"Transform returned non-finite values at {context}.")
    return tensor


def _run_batch(
    model: torch.nn.Module,
    pending_tensors: list[torch.Tensor],
    pending_targets: list[int],
    pending_samples: list[dict[str, Any]],
    confusion: list[list[int]],
    counts: Counter,
) -> None:
    if not pending_tensors:
        return

    contexts = "; ".join(_sample_context(sample) for sample in pending_samples)
    try:
        batch = torch.stack(pending_tensors, dim=0)
        with torch.inference_mode():
            logits = model(batch)
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected inference exception for batch samples [{contexts}]: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    expected_shape = (len(pending_tensors), 47)
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected_shape:
        actual_shape = (
            tuple(logits.shape) if isinstance(logits, torch.Tensor) else type(logits).__name__
        )
        raise RuntimeError(
            f"Model returned {actual_shape!r}, expected logits {expected_shape}, "
            f"for batch samples [{contexts}]."
        )
    if not torch.isfinite(logits).all().item():
        raise RuntimeError(f"Model returned non-finite logits for batch samples [{contexts}].")

    predictions = torch.argmax(logits, dim=1).tolist()
    for target, prediction in zip(pending_targets, predictions):
        confusion[target][prediction] += 1
        counts["correct"] += int(target == prediction)
        counts["t_predictions"] += int(prediction == 46)
        counts["successful"] += 1

    pending_tensors.clear()
    pending_targets.clear()
    pending_samples.clear()


def _print_confusion_matrix(confusion: list[list[int]]) -> None:
    column_labels = EXPECTED_BYMERGE
    print("Confusion matrix: rows=true, columns=predicted")
    print("row\\column," + ",".join(column_labels))
    for character, row in zip(SUPPORTED_CLASSES, confusion):
        print(f"{character}," + ",".join(str(value) for value in row))


def _print_results(
    samples: list[dict[str, Any]],
    counts: Counter,
    confusion: list[list[int]],
    eligible_support: Counter,
    failure_reasons: Counter,
    failure_by_character: Counter,
    failure_by_fold: Counter,
) -> None:
    eligible = len(samples)
    successful = counts["successful"]
    failed = counts["failed"]
    correct = counts["correct"]

    if successful + failed != eligible:
        raise RuntimeError(
            f"Accounting mismatch: successful ({successful}) + failed ({failed}) "
            f"!= eligible ({eligible})."
        )
    matrix_sum = sum(sum(row) for row in confusion)
    if matrix_sum != successful:
        raise RuntimeError(
            f"Confusion-matrix sum ({matrix_sum}) != successful ({successful})."
        )
    diagonal_sum = sum(confusion[index][index] for index in range(46))
    if correct != diagonal_sum:
        raise RuntimeError(
            f"Correct count ({correct}) != confusion diagonal sum ({diagonal_sum})."
        )
    if sum(eligible_support.values()) != eligible:
        raise RuntimeError("Per-class eligible support does not sum to eligible count.")

    t_predictions = sum(row[46] for row in confusion)
    if counts["t_predictions"] != t_predictions:
        raise RuntimeError(
            f"t prediction count ({counts['t_predictions']}) != column 46 sum "
            f"({t_predictions})."
        )

    true_positives = [confusion[index][index] for index in range(46)]
    class_f1 = []
    for class_index, character in enumerate(SUPPORTED_CLASSES):
        tp = true_positives[class_index]
        fp = sum(confusion[row][class_index] for row in range(46)) - tp
        fn = eligible_support[character] - tp
        denominator = 2 * tp + fp + fn
        f1 = (2 * tp / denominator) if denominator else 0.0
        class_f1.append(f1)

    macro_f1 = sum(class_f1) / 46

    print("\nResults")
    print(f"eligible: {eligible}")
    print(f"preprocessing-successful: {successful}")
    print(f"failed: {failed}")
    print(
        f"end-to-end accuracy: {correct}/{eligible} "
        f"({correct / eligible:.6f})"
    )
    if successful:
        print(
            f"conditional accuracy: {correct}/{successful} "
            f"({correct / successful:.6f})"
        )
    else:
        print("conditional accuracy: undefined (no preprocessing successes)")
    print(f"macro-F1 (46 target classes): {macro_f1:.6f}")

    print("\nPer-class F1 and eligible support")
    print("character,F1,support")
    for character, f1 in zip(SUPPORTED_CLASSES, class_f1):
        print(f"{character},{f1:.6f},{eligible_support[character]}")

    _print_confusion_matrix(confusion)
    print(f"\nt predictions (index 46): {t_predictions}")
    print(f"failures by reason: {dict(sorted(failure_reasons.items())) or '(none)'}")
    print(
        "failures by character: "
        + str({character: failure_by_character[character] for character in SUPPORTED_CLASSES})
    )
    print(
        "failures by fold: "
        + str({fold: failure_by_fold[fold] for fold in FOLDS})
    )


def main() -> None:
    print("HASYv2 pooled CNN evaluation configuration")
    print(f"dataset root: {CLASSIFICATION_ROOT}")
    print("test CSVs:")
    for fold in FOLDS:
        test_csv = CLASSIFICATION_ROOT / f"fold-{fold}" / "test.csv"
        print(f"  {test_csv}")
    print(f"strict supported labels: {''.join(SUPPORTED_CLASSES)}")
    print(f"expected eligible count: {EXPECTED_ELIGIBLE_COUNT}")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print("checkpoint provenance: project source/configuration only; metadata unverified")
    print("device: CPU")
    print(
        "HASYv2 input order: IMAGE_TRANSFORM, transpose(dim 1, 2), then batch; "
        "visual evidence from F, 7, R, q"
    )

    _verify_preprocessing_source()
    samples = _load_population()

    char_to_target = {
        design.CToC(index): index
        for index in range(46)
    }
    eligible_support = Counter(
        sample["character"] for sample in samples
    )
    confusion = [[0 for _ in range(47)] for _ in range(46)]
    counts = Counter()
    failure_reasons = Counter()
    failure_by_character = Counter()
    failure_by_fold = Counter()

    model = cnn_architecture.CNN().to(torch.device("cpu"))
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("Model parameters are not on CPU.")

    pending_tensors: list[torch.Tensor] = []
    pending_targets: list[int] = []
    pending_samples: list[dict[str, Any]] = []

    for sample in samples:
        target = char_to_target[sample["character"]]
        try:
            with Image.open(sample["image_path"]) as opened:
                opened.load()
                image = opened.copy()
        except IMAGE_READ_ERRORS as exc:
            reason = _format_reason(exc)
            counts["failed"] += 1
            failure_reasons[reason] += 1
            failure_by_character[sample["character"]] += 1
            failure_by_fold[sample["fold"]] += 1
            continue
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected image-read exception at {_sample_context(sample)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            transformed = design.IMAGE_TRANSFORM(image)
            transformed = _validate_preprocessed_tensor(transformed, sample)
        except preprocessing_final.PreprocessingError as exc:
            reason = _format_reason(exc)
            counts["failed"] += 1
            failure_reasons[reason] += 1
            failure_by_character[sample["character"]] += 1
            failure_by_fold[sample["fold"]] += 1
            continue
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected preprocessing exception at {_sample_context(sample)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            inference_tensor = torch.transpose(transformed, 1, 2)
            inference_tensor = _validate_preprocessed_tensor(inference_tensor, sample)
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected inference-transform exception at {_sample_context(sample)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        pending_tensors.append(inference_tensor)
        pending_targets.append(target)
        pending_samples.append(sample)
        if len(pending_tensors) == BATCH_SIZE:
            _run_batch(
                model,
                pending_tensors,
                pending_targets,
                pending_samples,
                confusion,
                counts,
            )

    _run_batch(
        model,
        pending_tensors,
        pending_targets,
        pending_samples,
        confusion,
        counts,
    )
    _print_results(
        samples,
        counts,
        confusion,
        eligible_support,
        failure_reasons,
        failure_by_character,
        failure_by_fold,
    )


if __name__ == "__main__":
    main()
