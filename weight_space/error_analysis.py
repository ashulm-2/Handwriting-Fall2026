"""Count a model's systematic mistakes: token substitutions, insertions and deletions.

Each prediction is aligned to its correct LaTeX (normalized tokens, Levenshtein alignment),
and every edit is tallied, so repeated patterns stand out from one-off misreads.

Usage:  python error_analysis.py work/base_dev/predictions.csv texo [--top 25]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "math_ocr"))
from metrics import normalize  # noqa: E402


def align(p: list[str], t: list[str]) -> list[tuple[str | None, str | None]]:
    """Levenshtein alignment as (predicted, truth) pairs; None marks an insertion/deletion."""
    n, m = len(p), len(t)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (p[i - 1] != t[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (p[i - 1] != t[j - 1]):
            out.append((p[i - 1], t[j - 1])); i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            out.append((p[i - 1], None)); i -= 1
        else:
            out.append((None, t[j - 1])); j -= 1
    return out[::-1]


def tally(rows: list[dict]) -> dict[str, Counter]:
    subs, ins, dels, pred_count, true_count = Counter(), Counter(), Counter(), Counter(), Counter()
    for r in rows:
        p, t = normalize(r["prediction"]), normalize(r["truth"])
        pred_count.update(p)
        true_count.update(t)
        for a, b in align(p, t):
            if a is not None and b is not None and a != b:
                subs[(b, a)] += 1  # (truth, predicted)
            elif b is None:
                ins[a] += 1  # predicted but not there
            elif a is None:
                dels[b] += 1  # there but not predicted
    return {"subs": subs, "ins": ins, "dels": dels, "pred": pred_count, "true": true_count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("model")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    with open(args.predictions, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == args.model]
    t = tally(rows)
    wrong = sum(r["exact"] != "True" for r in rows)
    print(f"{args.model}: {len(rows)} formulas, {wrong} not exact, "
          f"{sum(t['subs'].values())} substitutions, {sum(t['ins'].values())} extra tokens, {sum(t['dels'].values())} missing tokens")
    print("\ntop substitutions (truth -> predicted):")
    for (a, b), c in t["subs"].most_common(args.top):
        print(f"  {c:4d}  {a!r} -> {b!r}")
    print("\ntop extra tokens (predicted, not in truth):", t["ins"].most_common(args.top))
    print("\ntop missing tokens (in truth, not predicted):", t["dels"].most_common(args.top))
    over = sorted(((t["pred"][k] - t["true"][k], k) for k in set(t["pred"]) | set(t["true"])), reverse=True)
    print("\nmost over-produced tokens (count predicted - count true):", over[:12])
    print("most under-produced tokens:", over[-12:])


if __name__ == "__main__":
    main()
