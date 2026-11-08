"""Figures for the MURL Expo poster.

    python poster_figures.py pipeline  PHOTO [--emnist DIR]   -> results/poster_pipeline.png
    python poster_figures.py ocr       math_ocr/results/unimer_hwe_100 [--printed math_ocr/results/printed]
                                                               -> results/poster_formula_ocr.png

`pipeline` shows one handwritten character photo going through imgresize.toEmnist,
then the CNN's top-3 guesses. With --emnist (the folder torchvision downloaded EMNIST
into) the closest real EMNIST sample of the predicted class is shown for comparison.

`ocr` summarizes evaluate.py output for the formula-to-LaTeX models: accuracy and CPU
time per image, side by side (two separate axes, never one dual axis).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "results"

# Validated categorical palette (light mode), one fixed colour per model.
MODEL_COLORS = {"texo": "#2a78d6", "texteller": "#eb6834", "qwen3.5-mathparser": "#1baf7a", "qwen3-vl": "#eda100"}
MODEL_LABELS = {"texo": "Texo (20M)", "texteller": "TexTeller (298M)",
                "qwen3.5-mathparser": "Qwen3.5-2B MathParser", "qwen3-vl": "Qwen3-VL-2B (general)"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GOOD = "#008300"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _frame(img: np.ndarray, aspect: float = 1.45, margin: float = 0.08, bg: int = 0) -> np.ndarray:
    """Pad to a common height/width ratio with a small margin, centred."""
    h, w = img.shape
    m = int(margin * max(h, w))
    H, W = h + 2 * m, w + 2 * m
    H, W = max(H, int(W * aspect)), max(W, int(H / aspect))
    out = np.full((H, W), bg, dtype=img.dtype)
    top, left = (H - h) // 2, (W - w) // 2
    out[top:top + h, left:left + w] = img
    return out


def pipeline_figure(photo: str, emnist_dir: str | None) -> Path:
    import torch
    import torch.nn.functional as F

    import CNN
    import design
    import imgresize as M
    from steps import collectSteps

    steps = dict(collectSteps(photo))
    stages = [
        ("Photo", steps["1. loadGrayscale"]),
        ("Binarize\n(lighting-corrected)", steps["2. binarize"]),
        ("Remove specks,\ncrop to ink", steps["4. cropToInk"]),
        ("Match EMNIST\nstroke width", steps["7. matchStrokeWidth"]),
        ("28×28 input\nto the CNN", steps["10. toEmnist (best pass)"]),
    ]

    # Same path as predict.py: upright 28x28 -> design.IMAGE_TRANSFORM -> transpose back.
    from PIL import Image

    x = design.IMAGE_TRANSFORM(Image.fromarray(M.toEmnist(photo)))
    x = torch.transpose(x, 1, 2).unsqueeze(0)
    model = CNN.CNN()
    model.load_state_dict(torch.load(HERE / "CNNEMNIST.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 3)
    labels = [design.CToC(int(i)) for i in top.indices]
    values = [float(v) * 100 for v in top.values]

    real = None
    if emnist_dir:
        from torchvision.datasets import EMNIST

        # The real EMNIST test image of the predicted class that is closest (pixel-wise)
        # to our converted photo: shows the conversion produces EMNIST-like input.
        ds = EMNIST(emnist_dir, split="bymerge", train=False, download=False)
        ours = M.toEmnist(photo).astype(np.float32)
        same = ds.data[ds.targets == int(top.indices[0])].numpy().transpose(0, 2, 1).astype(np.float32)
        real = same[((same - ours) ** 2).sum(axis=(1, 2)).argmin()]

    n = len(stages) + (1 if real is not None else 0)
    fig = plt.figure(figsize=(2.3 * n + 3.6, 3.4))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.6], wspace=0.25)
    for k, (title, img) in enumerate(stages):
        ax = fig.add_subplot(gs[0, k])
        if img.shape[0] > 28:  # same frame for every stage, so none looks zoomed in
            img = _frame(img, bg=int(np.median(img[0])) if k == 0 else 0)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest" if img.shape[0] <= 28 else "antialiased")
        ax.set_title(f"{k + 1}. {title}", fontsize=10, color=INK)
        ax.axis("off")
        if k < len(stages) - 1:
            ax.annotate("", xy=(1.13, 0.5), xytext=(1.01, 0.5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
    if real is not None:
        ax = fig.add_subplot(gs[0, len(stages)])
        ax.imshow(real, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"Closest real\nEMNIST '{labels[0]}'", fontsize=10, color=MUTED)
        ax.axis("off")

    ax = fig.add_subplot(gs[0, n])
    ypos = np.arange(3)[::-1]
    ax.barh(ypos, values, height=0.55, color=["#2a78d6"] + ["#c8c7c2"] * 2)
    for y, v, lab in zip(ypos, values, labels):
        ax.text(v + 2, y, f"{v:.1f}%", va="center", fontsize=10, color=INK)
    ax.set_yticks(ypos, [f"'{lab}'" for lab in labels], fontsize=11, color=INK)
    ax.set_xlim(0, 118)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("CNN top-3 guesses", fontsize=10, color=INK)

    OUT.mkdir(exist_ok=True)
    path = OUT / "poster_pipeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _read_summary(folder: Path) -> dict[str, dict]:
    with open(folder / "summary.csv", newline="", encoding="utf-8") as f:
        return {r["model"]: r for r in csv.DictReader(f)}


def ocr_figure(results: str, printed: str | None) -> Path:
    hw = _read_summary(Path(results))
    models = [m for m in MODEL_COLORS if m in hw]
    pr = _read_summary(Path(printed)) if printed else {}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.12})
    y = np.arange(len(models))[::-1]
    names = [MODEL_LABELS[m] for m in models]

    exact = [float(hw[m]["exact_%"]) for m in models]
    within = [float(hw[m]["within2_%"]) for m in models]
    # Lighter bar = within 2 token edits; solid bar on top = exactly right.
    a1.barh(y, within, height=0.6, color=[MODEL_COLORS[m] for m in models], alpha=0.35)
    a1.barh(y, exact, height=0.6, color=[MODEL_COLORS[m] for m in models])
    for yi, e, w in zip(y, exact, within):
        a1.text(e - 1.5, yi, f"{e:.0f}%", va="center", ha="right", fontsize=10, color="white", fontweight="bold")
        a1.text(w + 1.5, yi, f"{w:.0f}%", va="center", fontsize=9, color=MUTED)
    a1.set_yticks(y, names, color=INK)
    a1.set_xlim(0, 108)
    a1.set_xlabel(f"% of handwritten formulas (n = {hw[models[0]]['n']})")
    a1.set_title("Accuracy: solid = exactly right, light = within 2 symbols", fontsize=11, color=INK, loc="left")
    a1.xaxis.grid(True, color=GRID)
    a1.set_axisbelow(True)

    secs = [float(hw[m]["mean_s_per_image"]) for m in models]
    a2.barh(y, secs, height=0.6, color=[MODEL_COLORS[m] for m in models])
    a2.set_xscale("log")
    for yi, s in zip(y, secs):
        a2.text(s * 1.12, yi, f"{s:.2f} s" if s < 1 else f"{s:.1f} s", va="center", fontsize=10, color=INK)
    a2.set_yticks(y, [""] * len(models))
    a2.set_xlim(0.02, 60)
    a2.set_xticks([0.1, 1, 10], ["0.1 s", "1 s", "10 s"])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlabel("seconds per image (log scale)")
    a2.set_title("Speed on a laptop CPU, no GPU", fontsize=11, color=INK, loc="left")
    a2.xaxis.grid(True, color=GRID, which="major")
    a2.set_axisbelow(True)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    OUT.mkdir(exist_ok=True)
    path = OUT / "poster_formula_ocr.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pipeline")
    p.add_argument("photo")
    p.add_argument("--emnist", help="folder EMNIST was downloaded into (optional)")
    o = sub.add_parser("ocr")
    o.add_argument("results")
    o.add_argument("--printed")
    args = ap.parse_args()
    if args.cmd == "pipeline":
        print("wrote", pipeline_figure(args.photo, args.emnist))
    else:
        sys.path.insert(0, str(HERE / "math_ocr"))
        print("wrote", ocr_figure(args.results, args.printed))
