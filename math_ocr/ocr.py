"""Image of a handwritten formula -> LaTeX, with any of the supported models.

Command line (prints one LaTeX string per image):
    python ocr.py --model texo      photo.png
    python ocr.py --model texteller photo1.png photo2.jpg
    python ocr.py --model all       photo.png          # every model, labelled
    python ocr.py --model texo --json photo.png        # {"image":..., "model":..., "latex":..., "seconds":...}

From Python (e.g. a web server). Load once at start-up, then call predict per request,
because loading is the slow part (Texo <1 s, TexTeller ~20 s, Qwen ~60 s):
    from ocr import load_model
    model = load_model("texteller")
    latex = model.predict("upload.png")        # path, PIL.Image or numpy array

Models:
    texo                20M params, ONNX, ~0.05 s/image on CPU   (AGPL-3.0)
    texteller           298M params,       ~1.5 s/image on CPU   (Apache-2.0)
    qwen3.5-mathparser  2B params VLM,     ~8 s/image on CPU     (Apache-2.0)
    qwen3-vl            2B params VLM,     ~6 s/image on CPU     (Apache-2.0)
(CPU times measured on a 16-core laptop CPU; see README.md.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAMES = ["texo", "texteller", "qwen3.5-mathparser", "qwen3-vl"]


def load_model(name: str, **kwargs):
    """Return a loaded model with a .predict(image) -> str method."""
    if name == "texo":
        from texo_model import TexoOCR

        return TexoOCR(**kwargs)
    if name == "texteller":
        from texteller_model import TexTellerOCR

        return TexTellerOCR(**kwargs)
    if name in ("qwen3.5-mathparser", "qwen3-vl"):
        from qwen_model import QwenOCR

        return QwenOCR(name, **kwargs)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files (png, jpg, ...)")
    ap.add_argument("--model", default="texteller", choices=MODEL_NAMES + ["all"])
    ap.add_argument("--json", action="store_true", help="print one JSON object per line")
    args = ap.parse_args()

    names = MODEL_NAMES if args.model == "all" else [args.model]
    for name in names:
        model = load_model(name)
        for path in args.images:
            start = time.perf_counter()
            latex = model.predict(path)
            seconds = time.perf_counter() - start
            if args.json:
                print(json.dumps({"image": path, "model": name, "latex": latex, "seconds": round(seconds, 3)}))
            elif len(names) > 1 or len(args.images) > 1:
                print(f"[{name}] {path}: {latex}")
            else:
                print(latex)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
