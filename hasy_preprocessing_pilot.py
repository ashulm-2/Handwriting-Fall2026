"""Read-only, preprocessing-only diagnostic over HASYv2 test samples."""

import sys

if sys.gettrace() is not None:
    raise SystemExit("Aborting: a Python trace function is already active.")

import csv
import inspect
import math
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

import design
import preprocessing_final


FOLDS = tuple(range(1, 11))
CLASSIFICATION_ROOT = Path.home() / "Downloads" / "classification-task"
EXPECTED_ELIGIBLE_COUNT = 3772

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


def _source_lines(function, statement, expected_count=1):
    """Return exact source-line matches, requiring the expected number of them."""
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


def _source_line(function, statement):
    return _source_lines(function, statement)[0]


def _verify_preprocessing_source():
    if design.EMNIST_SPLIT != "bymerge":
        raise RuntimeError(f"Expected bymerge; found {design.EMNIST_SPLIT!r}.")

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

    mask_file = Path(inspect.getsourcefile(mask_function)).resolve()
    thickener_method = type(thickener_stages[0]).__call__
    thickener_file = Path(inspect.getsourcefile(thickener_method)).resolve()
    project_root = Path(__file__).resolve().parent

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
        "correction_completion_marker": _source_line(
            mask_function,
            "if mask.mean() > 0.5:",
        ),
        "inversion_entry": _source_line(mask_function, "mask = ~mask"),
        "inversion_completion_marker": _source_line(mask_function, "return mask"),
    }
    if not (
        mask_lines["correction_condition"]
        < mask_lines["correction_entry"]
        < mask_lines["correction_assignment"]
        < mask_lines["correction_completion_marker"]
        < mask_lines["inversion_entry"]
        < mask_lines["inversion_completion_marker"]
    ):
        raise RuntimeError("_binary_mask statement order changed.")

    width_assignment_lines = _source_lines(
        thickener_method,
        "stroke_width = _horizontal_stroke_width(img)",
        expected_count=2,
    )
    thickener_lines = {
        "initial_width_assignment": width_assignment_lines[0],
        "loop_condition": _source_line(
            thickener_method,
            "while stroke_width is not None and stroke_width < self.target_width:",
        ),
        "filter_assignment": _source_line(
            thickener_method,
            "img = img.filter(ImageFilter.MaxFilter(3))",
        ),
        "remeasurement": width_assignment_lines[1],
        "return": _source_line(thickener_method, "return img"),
    }
    if not (
        thickener_lines["initial_width_assignment"]
        < thickener_lines["loop_condition"]
        < thickener_lines["filter_assignment"]
        < thickener_lines["remeasurement"]
        < thickener_lines["return"]
    ):
        raise RuntimeError("Thickener statement order changed.")

    foreground_method = type(mask_stages[0]).__call__
    foreground_source = inspect.getsource(foreground_method)
    if foreground_source.count("mask = _binary_mask(img)") != 1:
        raise RuntimeError("ForegroundMaskTransform no longer calls _binary_mask once.")

    return {
        "mask_function": mask_function,
        "thickener_method": thickener_method,
        "thickener_instance": thickener_stages[0],
        "mask_lines": mask_lines,
        "thickener_lines": thickener_lines,
    }


def _load_population():
    samples = []
    eligible_by_fold = Counter()
    eligible_by_character = Counter()

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

                sample = {
                    "fold": fold,
                    "character": label,
                    "csv_line": reader.line_num,
                    "csv_path": csv_path,
                    "image_path": (csv_path.parent / relative_path).resolve(),
                }
                samples.append(sample)
                eligible_by_fold[fold] += 1
                eligible_by_character[label] += 1

    if len(samples) != EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_COUNT} eligible rows, "
            f"found {len(samples)}. Stop for investigation."
        )

    first_sample_by_path = {}
    duplicate_paths = []
    for sample in samples:
        path = sample["image_path"]
        if path in first_sample_by_path:
            duplicate_paths.append(
                (path, first_sample_by_path[path], sample)
            )
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

    return samples, eligible_by_fold, eligible_by_character


def _new_trace_state(kind):
    state = {
        "kind": kind,
        "exception_events": [],
        "return_events": 0,
    }
    if kind == "mask":
        state.update({
            "correction_entries": 0,
            "correction_completions": 0,
            "inversion_entries": 0,
            "inversion_completions": 0,
        })
    else:
        state.update({
            "initial_width_seen": False,
            "initial_width": None,
            "target_width": None,
            "loop_tests": 0,
            "filter_entries": 0,
            "filter_completions": 0,
            "filter_pending": False,
        })
    return state


def _make_tracer(preflight, sample):
    mask_function = preflight["mask_function"]
    thickener_method = preflight["thickener_method"]
    thickener_instance = preflight["thickener_instance"]
    mask_lines = preflight["mask_lines"]
    thickener_lines = preflight["thickener_lines"]
    code_to_kind = {
        mask_function.__code__: "mask",
        thickener_method.__code__: "thickener",
    }

    def global_trace(frame, event, arg):
        if event != "call":
            return None

        kind = code_to_kind.get(frame.f_code)
        if kind is None:
            return None

        state = _new_trace_state(kind)
        sample["trace_states"].append(state)

        if (
            kind == "thickener"
            and frame.f_locals.get("self") is not thickener_instance
        ):
            sample["trace_faults"].append(
                "Thickener code object called on an unexpected instance."
            )
            return None

        def local_trace(current_frame, local_event, local_arg):
            try:
                if local_event == "line":
                    line = current_frame.f_lineno
                    if kind == "mask":
                        if line == mask_lines["correction_entry"]:
                            state["correction_entries"] += 1
                        elif line == mask_lines["correction_completion_marker"]:
                            if state["correction_entries"]:
                                state["correction_completions"] += 1
                        elif line == mask_lines["inversion_entry"]:
                            state["inversion_entries"] += 1
                        elif line == mask_lines["inversion_completion_marker"]:
                            if state["inversion_entries"]:
                                state["inversion_completions"] += 1
                    else:
                        if line == thickener_lines["loop_condition"]:
                            state["loop_tests"] += 1
                            if not state["initial_width_seen"]:
                                state["initial_width_seen"] = True
                                state["initial_width"] = current_frame.f_locals.get(
                                    "stroke_width"
                                )
                                instance = current_frame.f_locals.get("self")
                                state["target_width"] = instance.target_width
                        elif line == thickener_lines["filter_assignment"]:
                            if state["filter_pending"]:
                                sample["trace_faults"].append(
                                    "Filter loop re-entered before prior filter completion."
                                )
                            state["filter_entries"] += 1
                            state["filter_pending"] = True
                        elif line == thickener_lines["remeasurement"]:
                            if not state["filter_pending"]:
                                sample["trace_faults"].append(
                                    "Reached stroke remeasurement without a pending filter."
                                )
                            else:
                                state["filter_completions"] += 1
                                state["filter_pending"] = False
                elif local_event == "exception":
                    state["exception_events"].append(local_arg[0].__name__)
                elif local_event == "return":
                    # Observation only; the outer transform result determines success.
                    state["return_events"] += 1
            except Exception as exc:
                sample["trace_faults"].append(
                    f"Tracer error: {type(exc).__name__}: {exc}"
                )
                return None

            return local_trace

        return local_trace

    return global_trace


def _check_sample_instrumentation(sample, transform_succeeded):
    states = sample["trace_states"]
    mask_states = [state for state in states if state["kind"] == "mask"]
    thickener_states = [state for state in states if state["kind"] == "thickener"]

    if len(mask_states) > 1 or len(thickener_states) > 1:
        sample["trace_faults"].append("A target function ran more than once.")

    for state in mask_states:
        for entry_key, completion_key in (
            ("correction_entries", "correction_completions"),
            ("inversion_entries", "inversion_completions"),
        ):
            entries = state[entry_key]
            completions = state[completion_key]
            if entries > 1 or completions > 1 or completions > entries:
                sample["trace_faults"].append(
                    f"Inconsistent {entry_key}/{completion_key} counts."
                )
            if entries and not completions and not state["exception_events"]:
                sample["trace_faults"].append(
                    f"{entry_key} occurred without completion or an exception observation."
                )
        if not state["return_events"] and not state["exception_events"]:
            sample["trace_faults"].append(
                "Mask invocation had neither return nor exception trace observations."
            )

    for state in thickener_states:
        if state["filter_completions"] > state["filter_entries"]:
            sample["trace_faults"].append("More completed filters than filter entries.")
        if state["filter_pending"] and not state["exception_events"]:
            sample["trace_faults"].append(
                "Filter entry lacked completion and an exception observation."
            )
        if not state["initial_width_seen"] and not state["exception_events"]:
            sample["trace_faults"].append(
                "Thickener invocation had no initial-width observation."
            )
        if not state["return_events"] and not state["exception_events"]:
            sample["trace_faults"].append(
                "Thickener invocation had neither return nor exception trace observations."
            )

    if transform_succeeded:
        if len(mask_states) != 1 or len(thickener_states) != 1:
            sample["trace_faults"].append(
                "Successful transform did not invoke each target exactly once."
            )
        for state in mask_states:
            if state["correction_entries"] != state["correction_completions"]:
                sample["trace_faults"].append(
                    "Successful transform has incomplete background-correction tracing."
                )
            if state["inversion_entries"] != state["inversion_completions"]:
                sample["trace_faults"].append(
                    "Successful transform has incomplete polarity-inversion tracing."
                )
        for state in thickener_states:
            if state["filter_entries"] != state["filter_completions"]:
                sample["trace_faults"].append(
                    "Successful transform has incomplete filter tracing."
                )
            if state["filter_pending"]:
                sample["trace_faults"].append(
                    "Successful transform has a filter still pending."
                )
        for state in mask_states + thickener_states:
            if not state["return_events"]:
                sample["trace_faults"].append(
                    "Outer transform succeeded but target return was not observed."
                )

    if sample["trace_faults"]:
        raise RuntimeError(
            f"Instrumentation inconsistency for fold-{sample['fold']} "
            f"{sample['character']} ({sample['image_path']}): "
            + " | ".join(sample["trace_faults"])
        )


def _reason(exception):
    message = str(exception).strip()
    return f"{type(exception).__name__}: {message}" if message else type(exception).__name__


def _sample_context(sample):
    return (
        f"fold-{sample['fold']}, CSV line {sample['csv_line']}, "
        f"character {sample['character']!r}, image {sample['image_path']}"
    )


def main():
    if sys.gettrace() is not None:
        raise SystemExit("Aborting: a Python trace function is already active.")

    preflight = _verify_preprocessing_source()
    samples, eligible_by_fold, eligible_by_character = _load_population()

    counts = Counter()
    failure_reasons = Counter()
    failure_by_character = Counter()
    failure_by_fold = Counter()
    failure_categories = Counter()
    success_by_character = Counter()
    success_by_fold = Counter()
    trace_totals = Counter()

    for sample in samples:
        sample.update({
            "trace_states": [],
            "trace_faults": [],
            "status": None,
        })

        try:
            with Image.open(sample["image_path"]) as opened:
                opened.load()
                image = opened.copy()
        except IMAGE_READ_ERRORS as exc:
            sample["status"] = "image_read_error"
            counts["failed"] += 1
            failure_categories["image_read_error"] += 1
            failure_reasons[_reason(exc)] += 1
            failure_by_character[sample["character"]] += 1
            failure_by_fold[sample["fold"]] += 1
            continue
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected image-read exception at {_sample_context(sample)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        previous_trace = sys.gettrace()
        if previous_trace is not None:
            raise SystemExit(
                "Aborting: a Python trace function became active before the scan."
            )

        result = None
        transform_succeeded = False
        unexpected_exception = None
        try:
            sys.settrace(_make_tracer(preflight, sample))
            try:
                result = design.IMAGE_TRANSFORM(image)
                if not isinstance(result, torch.Tensor):
                    raise RuntimeError(
                        f"Transform returned {type(result).__name__}, not a tensor."
                    )
                if tuple(result.shape) != (1, 28, 28):
                    raise RuntimeError(
                        f"Transform returned unexpected shape {tuple(result.shape)!r}."
                    )
                if result.dtype != torch.float32:
                    raise RuntimeError(
                        f"Transform returned unexpected dtype {result.dtype!r}; "
                        "expected torch.float32."
                    )
                if not torch.isfinite(result).all().item():
                    raise RuntimeError("Transform returned non-finite tensor values.")
                transform_succeeded = True
            except preprocessing_final.PreprocessingError as exc:
                sample["status"] = "preprocessing_error"
                sample["failure_reason"] = _reason(exc)
            except Exception as exc:
                unexpected_exception = exc
        finally:
            sys.settrace(previous_trace)

        instrumentation_error = None
        try:
            _check_sample_instrumentation(sample, transform_succeeded)
        except RuntimeError as exc:
            instrumentation_error = exc

        if unexpected_exception is not None:
            detail = (
                f"Unexpected preprocessing exception at {_sample_context(sample)}: "
                f"{type(unexpected_exception).__name__}: {unexpected_exception}"
            )
            if instrumentation_error is not None:
                detail += f"; instrumentation check also failed: {instrumentation_error}"
            raise RuntimeError(detail) from unexpected_exception

        if instrumentation_error is not None:
            raise instrumentation_error

        if transform_succeeded:
            sample["status"] = "success"
            counts["successful"] += 1
            success_by_character[sample["character"]] += 1
            success_by_fold[sample["fold"]] += 1
        else:
            counts["failed"] += 1
            failure_reasons[sample["failure_reason"]] += 1
            failure_by_character[sample["character"]] += 1
            failure_by_fold[sample["fold"]] += 1
            failure_categories["preprocessing_error"] += 1

        for state in sample["trace_states"]:
            if state["kind"] == "mask":
                trace_totals["mask_calls"] += 1
                trace_totals["background_correction_entered"] += bool(
                    state["correction_entries"]
                )
                trace_totals["background_correction_completed"] += bool(
                    state["correction_completions"]
                )
                trace_totals["polarity_inversion_entered"] += bool(
                    state["inversion_entries"]
                )
                trace_totals["polarity_inversion_completed"] += bool(
                    state["inversion_completions"]
                )
            else:
                trace_totals["thickener_calls"] += 1
                if state["initial_width_seen"]:
                    if state["initial_width"] is None:
                        trace_totals["width_unmeasurable"] += 1
                    else:
                        trace_totals["width_measurable"] += 1
                        if state["initial_width"] < state["target_width"]:
                            trace_totals["initial_width_below_target"] += 1
                else:
                    trace_totals["initial_width_not_observed"] += 1

                trace_totals["filter_entries"] += state["filter_entries"]
                trace_totals["filter_completions"] += state["filter_completions"]
                if state["filter_completions"]:
                    trace_totals["images_with_completed_thickening"] += 1

        del result, image

    eligible = len(samples)
    successful = counts["successful"]
    failed = counts["failed"]
    if successful + failed != eligible:
        raise RuntimeError(
            f"Accounting mismatch: successful ({successful}) + failed ({failed}) "
            f"!= eligible ({eligible})."
        )

    mask_calls = trace_totals["mask_calls"]
    thickener_calls = trace_totals["thickener_calls"]
    measurable = trace_totals["width_measurable"]

    print("HASYv2 preprocessing-only full scan")
    print(f"eligible samples: {eligible}")
    print(f"successfully processed: {successful} ({successful / eligible:.1%})")
    print(f"failed: {failed} ({failed / eligible:.1%})")
    print(f"failure categories: {dict(sorted(failure_categories.items()))}")
    print(f"failure reasons: {dict(sorted(failure_reasons.items()))}")

    print("eligible by fold:", dict(sorted(eligible_by_fold.items())))
    print("successful by fold:", dict(sorted(success_by_fold.items())))
    print("failed by fold:", dict(sorted(failure_by_fold.items())))
    print("eligible by character:", dict(sorted(eligible_by_character.items())))
    print("successful by character:", dict(sorted(success_by_character.items())))
    print("failed by character:", dict(sorted(failure_by_character.items())))

    print(f"mask-stage calls: {mask_calls}")
    if mask_calls:
        print(
            "background correction entered/completed: "
            f"{trace_totals['background_correction_entered']}/"
            f"{trace_totals['background_correction_completed']} "
            f"of {mask_calls} mask calls "
            f"({trace_totals['background_correction_entered'] / mask_calls:.1%} entered)"
        )
        print(
            "polarity inversion entered/completed: "
            f"{trace_totals['polarity_inversion_entered']}/"
            f"{trace_totals['polarity_inversion_completed']} "
            f"of {mask_calls} mask calls "
            f"({trace_totals['polarity_inversion_entered'] / mask_calls:.1%} entered)"
        )
    else:
        print("background correction and polarity inversion: no mask-stage calls")

    print(f"thickener calls: {thickener_calls}")
    print(f"initial stroke width measurable: {measurable}")
    print(
        f"initial stroke width below target: "
        f"{trace_totals['initial_width_below_target']} of {measurable}"
    )
    print(f"initial stroke width unmeasurable: {trace_totals['width_unmeasurable']}")
    print(
        f"initial stroke width not observed: "
        f"{trace_totals['initial_width_not_observed']} of {thickener_calls}"
    )
    print(
        f"filter loop entries/completions: "
        f"{trace_totals['filter_entries']}/{trace_totals['filter_completions']}"
    )
    print(
        f"images with completed thickening: "
        f"{trace_totals['images_with_completed_thickening']} of {measurable} "
        f"measurable widths"
    )
    if measurable:
        print(
            "thickened images / measurable widths: "
            f"{trace_totals['images_with_completed_thickening'] / measurable:.1%}"
        )


if __name__ == "__main__":
    main()