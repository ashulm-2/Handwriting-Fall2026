"""Benchmark the OCR models on a folder of labelled handwritten formulas.

The folder needs a labels.csv with two columns, `image` (path relative to the folder)
and `latex` (the correct LaTeX), e.g.

    image,latex
    images/001.jpg,\\frac{x^2+1}{3y}
    images/002.jpg,\\int_0^{\\pi} \\sin(2t)\\,dt

Each model is loaded once, run on every image, and scored with metrics.score (LaTeX is
normalized first, so x^2 and x^{2} count as the same). Writes to --out:
    predictions.csv  one row per (model, image): prediction, score, seconds
    summary.csv      one row per model
    summary.md       the same summary as a Markdown table

Usage:
    python evaluate.py --data benchmark/unimer_hwe
    python evaluate.py --data testset --models texo texteller --out results/testset
    python evaluate.py --data testset --resume          # skip pairs already in predictions.csv
                                                        # (and re-score them with metrics.py)
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from metrics import score  # noqa: E402
from ocr import MODEL_NAMES, load_model  # noqa: E402

PRED_FIELDS = ["model", "image", "truth", "prediction", "exact", "within1", "within2",
               "edit_distance", "ter", "similarity", "seconds", "error"]


def read_labels(data: Path, limit: int | None) -> list[dict]:
    with open(data / "labels.csv", newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("image") and r.get("latex", "").strip()]
    split = [r["image"] for r in rows if None in r]
    if split:
        # LaTeX like "\, dx" or "[0, 7)" contains commas; unquoted, the label gets cut off.
        raise SystemExit(f"labels.csv rows {split[:5]} have extra columns: put LaTeX that "
                         f'contains a comma in double quotes, e.g. "\\int f(x) \\, dx"')
    missing = [r["image"] for r in rows if not (data / r["image"]).is_file()]
    if missing:
        print(f"skipping {len(missing)} labels with no image file, e.g. {missing[:3]}")
        rows = [r for r in rows if (data / r["image"]).is_file()]
    return rows[:limit] if limit else rows


def summarize(name: str, rows: list[dict], load_seconds: float | None) -> dict:
    n = len(rows)
    pct = lambda key: 100 * sum(r[key] in (True, "True") for r in rows) / n  # noqa: E731
    secs = [float(r["seconds"]) for r in rows]
    return {
        "model": name,
        "n": n,
        "exact_%": round(pct("exact"), 1),
        "within1_%": round(pct("within1"), 1),
        "within2_%": round(pct("within2"), 1),
        "similarity": round(statistics.mean(float(r["similarity"]) for r in rows), 3),
        "token_error_rate": round(statistics.mean(float(r["ter"]) for r in rows), 3),
        "mean_s_per_image": round(statistics.mean(secs), 3),
        "median_s_per_image": round(statistics.median(secs), 3),
        "load_s": round(load_seconds, 1) if load_seconds is not None else "",
        "errors": sum(bool(r.get("error")) for r in rows),
    }


def write_summary(out: Path, summaries: list[dict], data: Path) -> None:
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    cols = list(summaries[0])
    lines = [f"# Results on `{data}`", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(s[c]) for c in cols) + " |" for s in summaries]
    lines += ["", "exact / within1 / within2: % of formulas with 0 / <=1 / <=2 token edits "
              "after normalization (CROHME ExpRate). similarity: 1 - edit distance / length "
              "(1 = perfect). Times are per image, excluding model loading."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing labels.csv and the images")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--out", help="output folder (default: results/<data folder name>)")
    ap.add_argument("--limit", type=int, help="only use the first N labelled images")
    ap.add_argument("--resume", action="store_true", help="keep existing predictions.csv rows")
    ap.add_argument("--qwen-max-pixels", type=int, help="override the Qwen image-size cap")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out or Path("results") / data.name)
    out.mkdir(parents=True, exist_ok=True)
    labels = read_labels(data, args.limit)
    print(f"{len(labels)} labelled images in {data}")

    pred_path = out / "predictions.csv"
    done: dict[tuple[str, str], dict] = {}
    old_load: dict[str, str] = {}
    if args.resume and pred_path.exists():
        # Saved predictions are re-scored with the current metrics.py, so a resume with
        # nothing left to run is also a cheap way to re-score after changing the metric.
        with open(pred_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r.update(score(r["prediction"], r["truth"]))
                done[(r["model"], r["image"])] = r
        if (out / "summary.csv").exists():
            with open(out / "summary.csv", newline="", encoding="utf-8") as f:
                old_load = {r["model"]: r["load_s"] for r in csv.DictReader(f)}
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        writer.writeheader()
        writer.writerows(r for r in done.values() if r["model"] not in args.models)

        summaries = []
        for name in args.models:
            todo = [r for r in labels if (name, r["image"]) not in done]
            rows = [done[(name, r["image"])] for r in labels if (name, r["image"]) in done]
            writer.writerows(rows)
            load_seconds = None
            if todo:
                print(f"\n== {name}: loading ...", flush=True)
                kwargs = {}
                if name.startswith("qwen") and args.qwen_max_pixels is not None:
                    kwargs["max_pixels"] = args.qwen_max_pixels or None
                start = time.perf_counter()
                model = load_model(name, **kwargs)
                load_seconds = time.perf_counter() - start
                print(f"   loaded in {load_seconds:.1f}s", flush=True)
            for i, label in enumerate(todo, 1):
                start = time.perf_counter()
                try:
                    pred, err = model.predict(data / label["image"]), ""
                except Exception as e:  # keep going; record the failure
                    pred, err = "", f"{type(e).__name__}: {e}"
                seconds = time.perf_counter() - start
                row = {"model": name, "image": label["image"], "truth": label["latex"],
                       "prediction": pred, "seconds": round(seconds, 3), "error": err,
                       **score(pred, label["latex"])}
                writer.writerow(row)
                f.flush()
                rows.append(row)
                mark = "ok " if row["exact"] else "   "
                print(f"   [{i}/{len(todo)}] {mark}{seconds:6.2f}s  {label['image']}: {pred[:70]}", flush=True)
            if rows:
                summary = summarize(name, rows, load_seconds)
                if load_seconds is None:
                    summary["load_s"] = old_load.get(name, "")
                summaries.append(summary)

    write_summary(out, summaries, data)
    print("\n" + (out / "summary.md").read_text(encoding="utf-8"))
    print(f"wrote {pred_path}, {out / 'summary.csv'}, {out / 'summary.md'}")


if __name__ == "__main__":
    main()
