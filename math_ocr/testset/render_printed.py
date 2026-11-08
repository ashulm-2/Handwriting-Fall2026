"""Render the test-set formulas as clean printed images (a sanity check, not handwriting).

This lets the whole pipeline, including the "altered-famous" check, be run before anyone
has written the formulas by hand. Printed input is much easier than handwriting, so treat
the scores as an upper bound. Formulas that matplotlib's mathtext cannot draw (e.g.
matrices) are skipped.

Writes ../benchmark/printed/images/*.png and ../benchmark/printed/labels.csv.

Usage:  python render_printed.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE.parent / "benchmark" / "printed"


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    with open(HERE / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept, skipped = [], []
    for r in rows:
        name = "images/" + Path(r["image"]).with_suffix(".png").name
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${r['latex']}$", fontsize=28)
        try:
            fig.savefig(OUT / name, dpi=100, bbox_inches="tight", pad_inches=0.25, facecolor="white")
            kept.append({**r, "image": name})
        except ValueError:
            skipped.append(r["image"])
        finally:
            plt.close(fig)
    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(kept)
    print(f"rendered {len(kept)} formulas to {OUT}; skipped (mathtext can't draw them): {skipped}")


if __name__ == "__main__":
    main()
