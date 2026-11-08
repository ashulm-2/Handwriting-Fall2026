"""Texo weight-space experiment: choose a logit bias from dev error counts, score it.

    python run_texo.py dev  --alphas 0.5 1 2     # dev evaluations (counted against the budget)
    python run_texo.py test --alpha A            # one final test evaluation of the chosen edit

Every dev/test evaluation is appended to logs/texo_budget.csv, so the budget is auditable.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "math_ocr"))
from error_analysis import tally  # noqa: E402
from metrics import normalize, score  # noqa: E402
from texo_edit import TexoWithBias, bake  # noqa: E402

LOG = HERE / "logs" / "texo_budget.csv"


def read(split: str) -> list[dict]:
    with open(HERE / "data" / split / "labels.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bias_from_dev(ocr: TexoWithBias, alpha: float, min_gap: int = 4) -> tuple[np.ndarray, list]:
    """bias_k = -alpha * log((predicted_k + 1) / (true_k + 1)) for tokens with |gap| >= min_gap."""
    with open(HERE / "work" / "base_dev" / "predictions.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == "texo"]
    # Leave out runaway loops (>= 20 extra tokens): they say nothing about per-token preferences.
    rows = [r for r in rows if len(normalize(r["prediction"])) - len(normalize(r["truth"])) < 20]
    t = tally(rows)
    bias = np.zeros(687, dtype=np.float32)
    changed = []
    for tok in set(t["pred"]) | set(t["true"]):
        gap = t["pred"][tok] - t["true"][tok]
        if abs(gap) >= min_gap and tok in ocr.vocab:
            b = -alpha * math.log((t["pred"][tok] + 1) / (t["true"][tok] + 1))
            bias[ocr.token_id(tok)] = b
            changed.append((tok, gap, round(b, 3)))
    return bias, sorted(changed, key=lambda c: c[2])


def evaluate(ocr, split: str) -> dict:
    rows, scores, secs = read(split), [], []
    for r in rows:
        t0 = time.perf_counter()
        pred = ocr.predict(HERE / "data" / split / r["image"])
        secs.append(time.perf_counter() - t0)
        scores.append(score(pred, r["latex"]))
    n = len(scores)
    return {"n": n, "exact": 100 * sum(s["exact"] for s in scores) / n,
            "within2": 100 * sum(s["within2"] for s in scores) / n,
            "similarity": statistics.mean(s["similarity"] for s in scores),
            "median_ter": statistics.median(s["ter"] for s in scores), "s_per_img": statistics.mean(secs)}


def log(entry: dict) -> None:
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(entry))
        if new:
            w.writeheader()
        w.writerow(entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--alphas", type=float, nargs="*", default=[])
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--baseline", action="store_true", help="also score the unedited model")
    args = ap.parse_args()

    ocr = TexoWithBias()
    alphas = ([0.0] if args.baseline else []) + (args.alphas or ([args.alpha] if args.alpha is not None else []))
    for a in alphas:
        bias, changed = bias_from_dev(ocr, a) if a else (np.zeros(687, np.float32), [])
        ocr.bias = bias
        res = evaluate(ocr, args.split)
        entry = {"split": args.split, "alpha": a, "tokens_changed": len(changed),
                 **{k: round(v, 4) for k, v in res.items()}}
        log(entry)
        print(entry)
        if a and args.split == "dev":
            print("   biased tokens:", changed)
        if a and args.split == "test":
            out = bake(bias, HERE / "work" / "texo_edited" / "decoder_model_merged.onnx")
            (HERE / "edits").mkdir(exist_ok=True)
            np.save(HERE / "edits" / "texo_lm_head_bias.npy", bias)
            print("   baked edited decoder:", out)


if __name__ == "__main__":
    main()
