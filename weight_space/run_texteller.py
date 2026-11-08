r"""TexTeller weight-space experiment: penalize its text-mode habit on printed formulas.

    python run_texteller.py runnerup            # what TexTeller would write instead of each habit token
    python run_texteller.py eval --split printed_dev --limit 120 --bias -12 --baseline

The habit: on printed (matplotlib) formulas TexTeller writes \; or \text/\mbox/$ where a superscript,
subscript or plain math belongs. The edit is a negative bias on those tokens (and, here, on the
formatting tokens that were their runners-up) in a new output-layer bias. Every evaluation is
logged to logs/texteller_budget.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "math_ocr"))
from metrics import score  # noqa: E402
from texteller_edit import HABIT_TOKENS, TexTellerWithBias  # noqa: E402

# Formatting tokens that were the runner-up to a habit token on printed dev (see `runnerup`).
RUNNER_UP_TOKENS = ["char", "tiny", "textbf", "textit", "\\;\\", "Ġ\\;", "\\:", "\\,\\,", "\\,\\"]
LOG = HERE / "logs" / "texteller_budget.csv"


def labels(split: str, limit: int | None) -> list[dict]:
    with open(HERE / "data" / split / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def cmd_runnerup(a):
    m = TexTellerWithBias()
    rep = []
    for r in labels("printed_dev", a.limit or 120):
        rep += m.runner_up_report(HERE / "data" / "printed_dev" / r["image"], HABIT_TOKENS)
    (HERE / "work" / "texteller_runnerup.json").write_text(json.dumps(rep))
    gaps = sorted(x["gap"] for x in rep)
    print(len(rep), "habit tokens;", Counter(x["habit"] for x in rep).most_common())
    print("runner-up:", Counter(x["alternative"] for x in rep).most_common(12))
    print("median gap %.2f" % statistics.median(gaps))


def cmd_eval(a):
    m = TexTellerWithBias()
    family = [t for t in HABIT_TOKENS + RUNNER_UP_TOKENS if _known(m, t)]
    runs = ([("base", {})] if a.baseline else []) + ([(f"bias {a.bias} on {len(family)} tokens", {t: a.bias for t in family})] if a.bias else [])
    rows = labels(a.split, a.limit)
    for name, bias in runs:
        m.set_bias(bias)
        preds = [m.predict(HERE / "data" / a.split / r["image"]) for r in rows]
        s = [score(p, r["latex"]) for p, r in zip(preds, rows)]
        entry = {"split": a.split, "n": len(s), "what": name,
                 "exact": round(100 * sum(x["exact"] for x in s) / len(s), 2),
                 "similarity": round(statistics.mean(x["similarity"] for x in s), 4)}
        print(entry, flush=True)
        _log(entry)
        (HERE / "work" / f"texteller_{a.split}_{'base' if not bias else 'edit'}.json").write_text(
            json.dumps([{"image": r["image"], "truth": r["latex"], "prediction": p} for p, r in zip(preds, rows)]))


def _known(m, t):
    try:
        m.token_id(t)
        return True
    except KeyError:
        return False


def _log(entry):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "n", "what", "exact", "similarity"])
        if new:
            w.writeheader()
        w.writerow(entry)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["runnerup", "eval"])
    ap.add_argument("--split", default="printed_dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bias", type=float)
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    {"runnerup": cmd_runnerup, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
