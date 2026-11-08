"""Qwen weight-space experiment: fix a superscript -> subscript habit with an lm_head bias.

    python run_qwen.py instrument --variant qwen3.5-mathparser --limit 150
        one dev pass recording, at every ^/_ decision, the best superscript-token logit minus the best
        subscript-token logit (Ġ^ ^{ ^ ^{-  vs  Ġ_ _{ _); 1 dev pass of the budget
    python run_qwen.py choose --variant qwen3.5-mathparser
        pick the bias from those margins (no model runs)
    python run_qwen.py eval --variant qwen3.5-mathparser --split dev --limit 150 --bias 1.5 [--baseline]
        score the edit (each eval is one pass of the budget; logged to logs/qwen_budget.csv)

How the bias is chosen: adding +d to the "^" logit flips exactly the steps where the model chose
"_" with a margin smaller than d. Using the dev alignment against the correct LaTeX, each such flip
is a fix (truth has ^ there) or a break (truth has _ there); d maximizes fixes - breaks. This
ignores knock-on effects on later tokens, so the chosen d is then checked by a real dev pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "math_ocr"))
from error_analysis import align  # noqa: E402
from metrics import normalize, score  # noqa: E402

WORK = HERE / "work"
LOG = HERE / "logs" / "qwen_budget.csv"


def labels(split: str, limit: int | None) -> list[dict]:
    with open(HERE / "data" / split / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def cmd_instrument(a):
    from qwen_edit import QwenWithBias

    ocr = QwenWithBias(a.variant)
    sups, subs = known(ocr, a.sup), known(ocr, a.sub)
    sup_ids, sub_ids = {ocr.token_id(t) for t in sups}, {ocr.token_id(t) for t in subs}
    out = []
    for i, r in enumerate(labels("dev", a.limit), 1):
        pred, steps = ocr.predict_with_scores(HERE / "data" / "dev" / r["image"], sups + subs)
        keep = []
        for s in steps:
            if s["chosen"] in sup_ids | sub_ids:
                keep.append({"prefix": s["prefix"], "chose": "^" if s["chosen"] in sup_ids else "_",
                             "margin": max(s[t] for t in sups) - max(s[t] for t in subs)})
        out.append({"image": r["image"], "truth": r["latex"], "prediction": pred, "decisions": keep})
        print(f"[{i}] {len(keep)} ^/_ decisions  {pred[:60]}", flush=True)
    WORK.mkdir(exist_ok=True)
    (WORK / f"{a.variant}_instrument.json").write_text(json.dumps({"sup": sups, "sub": subs, "items": out}))
    log({"variant": a.variant, "split": "dev", "what": "instrumented base pass", "n": len(out)})


def known(ocr, tokens: list[str]) -> list[str]:
    """The tokens that exist as single entries in this model's vocabulary."""
    vocab = ocr.processor.tokenizer.get_vocab()
    return [t for t in tokens if t in vocab]


def labelled_decisions(variant: str) -> list[dict]:
    """Each ^/_ decision with its margin (logit ^ minus logit _) and what the truth has there."""
    data = json.loads((WORK / f"{variant}_instrument.json").read_text())
    res = []
    for item in data["items"]:
        p, t = normalize(item["prediction"]), normalize(item["truth"])
        pred_to_truth, pi = {}, 0
        for a, b in align(p, t):
            if a is not None:
                pred_to_truth[pi] = b
                pi += 1
        for d in item["decisions"]:
            # Strip an opening ```latex / $$ / \[ the model wrote (the final output has it removed).
            prefix = re.sub(r"^\s*(```(?:latex|tex)?\s*)?(\$\$|\$|\\\[|\\\()?", "", d["prefix"])
            k = len(normalize(prefix))
            if k < len(p) and p[k] == d["chose"]:
                res.append({"margin": d["margin"], "chose": d["chose"],
                            "truth": pred_to_truth.get(k)})
    return res


def cmd_choose(a):
    dec = labelled_decisions(a.variant)
    subs = [d for d in dec if d["chose"] == "_"]
    print(f"{len(dec)} ^/_ decisions; chose _ {len(subs)} times "
          f"({sum(d['truth'] == '^' for d in subs)} wrong: truth is ^)")
    best = (0, 0.0, 0, 0)
    for step in range(1, 41):
        d = step * 0.25
        flips = [s for s in subs if s["margin"] > -d]
        fixes = sum(s["truth"] == "^" for s in flips)
        breaks = sum(s["truth"] == "_" for s in flips)
        if fixes - breaks > best[0]:
            best = (fixes - breaks, d, fixes, breaks)
        if step % 4 == 0:
            print(f"  bias +{d:.2f} on ^: flips {len(flips)}, fixes {fixes}, breaks {breaks}")
    print(f"chosen bias +{best[1]:.2f} (fixes {best[2]}, breaks {best[3]})" if best[0] else "no bias helps on dev")


def cmd_eval(a):
    from qwen_edit import QwenWithBias

    ocr = QwenWithBias(a.variant)
    runs = ([0.0] if a.baseline else []) + ([a.bias] if a.bias else [])
    for b in runs:
        ocr.set_bias({t: b for t in known(ocr, a.sup)} if b else {})
        rows, scores, secs, preds = labels(a.split, a.limit), [], [], []
        for r in rows:
            t0 = time.perf_counter()
            pred = ocr.predict(HERE / "data" / a.split / r["image"])
            secs.append(time.perf_counter() - t0)
            scores.append(score(pred, r["latex"]))
            preds.append({"image": r["image"], "truth": r["latex"], "prediction": pred, "exact": scores[-1]["exact"]})
        n = len(scores)
        entry = {"variant": a.variant, "split": a.split, "what": f"bias +{b} on superscript tokens", "n": n,
                 "exact": round(100 * sum(s["exact"] for s in scores) / n, 2),
                 "within2": round(100 * sum(s["within2"] for s in scores) / n, 2),
                 "similarity": round(statistics.mean(s["similarity"] for s in scores), 4),
                 "s_per_img": round(statistics.mean(secs), 2)}
        log(entry)
        print(entry, flush=True)
        (WORK / f"{a.variant}_{a.split}_bias{b}.json").write_text(json.dumps(preds))


def log(entry: dict) -> None:
    LOG.parent.mkdir(exist_ok=True)
    fields = ["variant", "split", "what", "n", "exact", "within2", "similarity", "s_per_img"]
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: entry.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["instrument", "choose", "eval"])
    ap.add_argument("--variant", default="qwen3.5-mathparser")
    ap.add_argument("--sup", nargs="+", default=["Ġ^", "^{", "^", "^{-"], help="superscript tokens")
    ap.add_argument("--sub", nargs="+", default=["Ġ_", "_{", "_"], help="subscript tokens")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bias", type=float)
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    {"instrument": cmd_instrument, "choose": cmd_choose, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
