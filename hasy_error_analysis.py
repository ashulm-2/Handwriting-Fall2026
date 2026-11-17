"""Display selected HASYv2 CNN errors and first correct controls.

The visual orientation check for F, 7, R, and q supported path B:
IMAGE_TRANSFORM followed by the existing inference transpose. This is visual
evidence, not an accuracy-based choice.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import hasy_evaluate as evaluation


CONFUSION_PAIRS = (
    ("O", "0"),
    ("2", "7"),
    ("9", "Y"),
    ("S", "5"),
    ("f", "F"),
)
EXPECTED_PAIR_ERROR_COUNTS = {
    ("O", "0"): 75,
    ("2", "7"): 53,
    ("9", "Y"): 42,
    ("S", "5"): 40,
    ("f", "F"): 32,
}
MAX_ERRORS_PER_PAIR = 3
EXPECTED_CORRECT = 2775


def _record(
    kind: str,
    sample: dict[str, Any],
    prediction: str,
    original: Image.Image,
    forward_input: torch.Tensor,
) -> dict[str, Any]:
    if tuple(forward_input.shape) != (1, 1, 28, 28):
        raise RuntimeError(
            f"Selected tensor has unexpected shape {tuple(forward_input.shape)!r} "
            f"at {evaluation._sample_context(sample)}."
        )
    return {
        "kind": kind,
        "true": sample["character"],
        "prediction": prediction,
        "sample": sample,
        "original": original.copy(),
        "forward_input": forward_input.detach().cpu().clone(),
    }


def _flush_batch(
    model: torch.nn.Module,
    pending_tensors: list[torch.Tensor],
    pending_originals: list[Image.Image],
    pending_samples: list[dict[str, Any]],
    char_to_index: dict[str, int],
    correct_controls: dict[str, dict[str, Any] | None],
    selected_errors: dict[tuple[str, str], list[dict[str, Any]]],
    pair_error_counts: Counter,
    counts: Counter,
) -> None:
    if not pending_tensors:
        return

    contexts = "; ".join(
        evaluation._sample_context(sample) for sample in pending_samples
    )
    try:
        batch = torch.stack(pending_tensors, dim=0)
        with torch.inference_mode():
            logits = model(batch)
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected inference exception for samples [{contexts}]: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    expected_shape = (len(pending_tensors), 47)
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected_shape:
        actual_shape = (
            tuple(logits.shape)
            if isinstance(logits, torch.Tensor)
            else type(logits).__name__
        )
        raise RuntimeError(
            f"Model returned {actual_shape!r}, expected logits {expected_shape}, "
            f"for samples [{contexts}]."
        )
    if not torch.isfinite(logits).all().item():
        raise RuntimeError(f"Model returned non-finite logits for [{contexts}].")

    predictions = torch.argmax(logits, dim=1).tolist()
    for offset, prediction_index in enumerate(predictions):
        sample = pending_samples[offset]
        true_label = sample["character"]
        prediction = evaluation.design.CToC(prediction_index)
        forward_input = batch[offset : offset + 1]

        counts["successful"] += 1
        if prediction_index == char_to_index[true_label]:
            counts["correct"] += 1
            if true_label in correct_controls and correct_controls[true_label] is None:
                correct_controls[true_label] = _record(
                    "correct control",
                    sample,
                    prediction,
                    pending_originals[offset],
                    forward_input,
                )
            continue

        pair = (true_label, prediction)
        if pair in selected_errors:
            pair_error_counts[pair] += 1
            if len(selected_errors[pair]) < MAX_ERRORS_PER_PAIR:
                selected_errors[pair].append(
                    _record(
                        "error",
                        sample,
                        prediction,
                        pending_originals[offset],
                        forward_input,
                    )
                )

    pending_tensors.clear()
    pending_originals.clear()
    pending_samples.clear()


def _display_examples(
    selected_errors: dict[tuple[str, str], list[dict[str, Any]]],
    correct_controls: dict[str, dict[str, Any] | None],
    pair_error_counts: Counter,
) -> None:
    for true_label, predicted_label in CONFUSION_PAIRS:
        pair = (true_label, predicted_label)
        errors = selected_errors[pair]
        control = correct_controls[true_label]

        print(
            f"{true_label}->{predicted_label}: "
            f"{pair_error_counts[pair]} errors found; displaying {len(errors)}."
        )
        if control is None:
            print(
                f"Missing correct control for true label {true_label!r}: "
                "no correctly classified example was found."
            )

        examples = list(errors)
        if control is not None:
            examples.append(control)
        if not examples:
            continue

        figure, axes = plt.subplots(
            len(examples),
            2,
            figsize=(12, 3.2 * len(examples)),
            squeeze=False,
        )
        figure.suptitle(
            f"True {true_label} -> predicted {predicted_label}; "
            "selected errors and correct control"
        )

        for row_index, example in enumerate(examples):
            sample = example["sample"]
            filename = Path(sample["image_path"]).name
            metadata = (
                f"{example['kind']}; true={example['true']}; "
                f"prediction={example['prediction']}; fold={sample['fold']}; "
                f"CSV line={sample['csv_line']}; file={filename}"
            )

            source_image = example["original"]
            if source_image.mode in {"1", "L", "I", "F"}:
                source_pixels = np.asarray(source_image.convert("L"))
                axes[row_index, 0].imshow(
                    source_pixels,
                    cmap="gray",
                    origin="upper",
                    aspect="equal",
                )
            else:
                source_pixels = np.asarray(source_image.convert("RGB"))
                axes[row_index, 0].imshow(
                    source_pixels,
                    origin="upper",
                    aspect="equal",
                )
            axes[row_index, 0].set_title(f"Original source image\n{metadata}")
            axes[row_index, 0].axis("off")

            tensor = example["forward_input"]
            axes[row_index, 1].imshow(
                tensor[0, 0].numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
            )
            axes[row_index, 1].set_title(
                f"Final path-B tensor to forward "
                f"(shape={tuple(tensor.shape)})\n{metadata}"
            )
            axes[row_index, 1].axis("off")

        figure.tight_layout()


def main() -> None:
    print(f"dataset root: {evaluation.CLASSIFICATION_ROOT}")
    print(f"checkpoint: {evaluation.CHECKPOINT_PATH}")
    print("device: CPU")
    print("sample order: fold order, then original CSV row order")
    print("HASYv2 transform: IMAGE_TRANSFORM, transpose(dim 1, 2), then batch")

    evaluation._verify_preprocessing_source()
    samples = evaluation._load_population()

    char_to_index = {
        evaluation.design.CToC(index): index
        for index in range(47)
    }
    selected_errors = {
        pair: []
        for pair in CONFUSION_PAIRS
    }
    correct_controls: dict[str, dict[str, Any] | None] = {
        true_label: None
        for true_label, _ in CONFUSION_PAIRS
    }
    pair_error_counts = Counter()
    failure_reasons = Counter()
    counts = Counter()

    model = evaluation.cnn_architecture.CNN().to(torch.device("cpu"))
    state_dict = torch.load(
        evaluation.CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("Model parameters are not on CPU.")

    pending_tensors: list[torch.Tensor] = []
    pending_originals: list[Image.Image] = []
    pending_samples: list[dict[str, Any]] = []

    for sample in samples:
        try:
            with Image.open(sample["image_path"]) as opened:
                opened.load()
                original = opened.copy()
        except evaluation.IMAGE_READ_ERRORS as exc:
            counts["failed"] += 1
            failure_reasons[evaluation._format_reason(exc)] += 1
            continue
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected image-read exception at "
                f"{evaluation._sample_context(sample)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            transformed = evaluation.design.IMAGE_TRANSFORM(original)
            transformed = evaluation._validate_preprocessed_tensor(
                transformed, sample
            )
        except evaluation.preprocessing_final.PreprocessingError as exc:
            counts["failed"] += 1
            failure_reasons[evaluation._format_reason(exc)] += 1
            continue
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected preprocessing exception at "
                f"{evaluation._sample_context(sample)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            path_b_tensor = torch.transpose(transformed, 1, 2)
            path_b_tensor = evaluation._validate_preprocessed_tensor(
                path_b_tensor, sample
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected path-B transform exception at "
                f"{evaluation._sample_context(sample)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        pending_tensors.append(path_b_tensor)
        pending_originals.append(original)
        pending_samples.append(sample)

        if len(pending_tensors) == evaluation.BATCH_SIZE:
            _flush_batch(
                model,
                pending_tensors,
                pending_originals,
                pending_samples,
                char_to_index,
                correct_controls,
                selected_errors,
                pair_error_counts,
                counts,
            )

    _flush_batch(
        model,
        pending_tensors,
        pending_originals,
        pending_samples,
        char_to_index,
        correct_controls,
        selected_errors,
        pair_error_counts,
        counts,
    )

    expected_pair_counts = {
        f"{true_label}->{predicted_label}": EXPECTED_PAIR_ERROR_COUNTS[
            (true_label, predicted_label)
        ]
        for true_label, predicted_label in CONFUSION_PAIRS
    }
    actual_pair_counts = {
        f"{true_label}->{predicted_label}": pair_error_counts[
            (true_label, predicted_label)
        ]
        for true_label, predicted_label in CONFUSION_PAIRS
    }
    expected = {
        "eligible": evaluation.EXPECTED_ELIGIBLE_COUNT,
        "successful": evaluation.EXPECTED_ELIGIBLE_COUNT,
        "failed": 0,
        "correct": EXPECTED_CORRECT,
        "confusion_pair_counts": expected_pair_counts,
    }
    actual = {
        "eligible": len(samples),
        "successful": counts["successful"],
        "failed": counts["failed"],
        "correct": counts["correct"],
        "confusion_pair_counts": actual_pair_counts,
    }
    if actual != expected:
        raise RuntimeError(
            "Baseline verification failed; no figures will be displayed.\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {actual!r}\n"
            f"Failures by reason: {dict(failure_reasons)!r}"
        )

    print(
        f"Baseline verified before display: eligible={actual['eligible']}, "
        f"successful={actual['successful']}, failed={actual['failed']}, "
        f"correct={actual['correct']}."
    )
    _display_examples(selected_errors, correct_controls, pair_error_counts)
    plt.show()


if __name__ == "__main__":
    main()
