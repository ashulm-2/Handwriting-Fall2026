"""Printed (matplotlib mathtext) versions of the dev and test splits' formulas.

TexTeller reads handwriting almost perfectly here but, on printed sans-serif formulas, it
often wraps output in formatting commands (\\mbox, \\hbox, \\texttt, one-line arrays). These
splits let that habit be measured on dev and a fix be scored on a separate test set.
Formulas mathtext cannot draw are skipped.

Writes data/printed_dev and data/printed_test (images/*.png + labels.csv).
Usage:  python render_printed_splits.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent


def render(src: str, dst: str) -> None:
    out = HERE / "data" / dst
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(HERE / "data" / src / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for r in rows:
        tex = r["latex"].replace(r"\limits", "")  # mathtext has no \limits
        if not tex.isascii():
            continue
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=26)
        try:
            fig.savefig(out / r["image"], dpi=100, bbox_inches="tight", pad_inches=0.25, facecolor="white")
            kept.append(r)
        except Exception:
            pass
        finally:
            plt.close(fig)
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "latex"])
        w.writeheader()
        w.writerows({"image": r["image"], "latex": r["latex"]} for r in kept)
    print(f"{dst}: rendered {len(kept)} of {len(rows)}")


if __name__ == "__main__":
    render("dev", "printed_dev")
    render("test", "printed_test")
