"""Texo: image of a formula -> LaTeX, with a 20M-parameter model.

Texo (https://github.com/alephpi/Texo, AGPL-3.0) is a distilled PP-FormulaNet-S: an
HGNetV2 image encoder and a 2-layer MBart decoder with a 687-token LaTeX vocabulary.
Its encoder is a custom architecture that plain `transformers` cannot load, so this
file runs the ONNX export published with the weights (alephpi/FormulaNet, folder
onnx/), the same files the in-browser demo uses. The only dependencies are
onnxruntime, tokenizers, numpy and Pillow (no PyTorch), and it runs comfortably on a CPU.

Preprocessing reproduces Texo's EvalMERImageProcessor (src/texo/data/processor/
image_processor.py): crop the white margin, scale to fit 384x384, pad with black,
grayscale, normalise with the UniMERNet mean/std.

Usage:
    python texo_model.py path/to/image.png [more images...]
"""

from __future__ import annotations

import re
import sys

import numpy as np
from PIL import Image, ImageOps

from common import ImageInput, load_image

REPO_ID = "alephpi/FormulaNet"
IMAGE_SIZE = 384
MEAN, STD = 0.7931, 0.1738
BOS, PAD, EOS = 0, 1, 2  # decoder_start / pad / eos token ids from generation_config.json
MAX_TOKENS = 1024


def _border_median(img: Image.Image) -> float:
    """Median brightness of a thin strip around the edge, i.e. the background."""
    g = np.asarray(img.convert("L"))
    k = max(1, round(0.05 * min(g.shape)))
    return float(np.median(np.concatenate([g[:k].ravel(), g[-k:].ravel(), g[:, :k].ravel(), g[:, -k:].ravel()])))


def _crop_margin(img: Image.Image) -> Image.Image:
    """Crop to the box around every pixel darker than 200 after contrast stretching."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lo, hi = gray.min(), gray.max()
    if hi == lo:
        return img
    ink = (gray - lo) / (hi - lo) * 255 < 200
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def _fit_and_pad(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Shorter side -> size, then shrink to fit size x size, then pad (black) to centre it."""
    w, h = img.size
    if w <= h:
        img = img.resize((size, int(size * h / w)), Image.BILINEAR)
    else:
        img = img.resize((int(size * w / h), size), Image.BILINEAR)
    img.thumbnail((size, size))
    dw, dh = size - img.width, size - img.height
    return ImageOps.expand(img, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2))


class TexoOCR:
    name = "texo"

    def __init__(self, threads: int | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]

        def session(name):
            return ort.InferenceSession(hf_hub_download(REPO_ID, f"onnx/{name}"), opts, providers=providers)

        self.encoder = session("encoder_model.onnx")
        self.decoder = session("decoder_model_merged.onnx")
        self.tokenizer = Tokenizer.from_file(hf_hub_download(REPO_ID, "tokenizer.json"))
        self.past_names = [i.name for i in self.decoder.get_inputs() if i.name.startswith("past_key_values")]
        self.present_names = [o.name for o in self.decoder.get_outputs() if o.name.startswith("present")]
        shape = self.decoder.get_inputs()[2].shape  # [batch, heads, seq, head_dim]
        self.heads, self.head_dim = shape[1], shape[3]

    def preprocess(self, image: ImageInput) -> np.ndarray:
        img = load_image(image)
        # The model expects dark ink on light paper. Texo-web inverts when most pixels
        # are < 200, but that also flips photos of grey-looking paper; judging by the
        # border instead (as imgresize.isDarkOnLight does) only flips a truly dark board.
        if _border_median(img) < 128:
            img = ImageOps.invert(img)
        img = _fit_and_pad(_crop_margin(img))
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        x = (gray - MEAN) / STD
        return np.repeat(x[None, None], 3, axis=1)  # 1 x 3 x 384 x 384

    def predict(self, image: ImageInput) -> str:
        hidden = self.encoder.run(None, {"pixel_values": self.preprocess(image)})[0]

        # Greedy decoding with a key/value cache. The merged decoder takes
        # use_cache_branch=False on the first step (empty cache) and True afterwards.
        empty = np.zeros((1, self.heads, 0, self.head_dim), dtype=np.float32)
        past = {name: empty for name in self.past_names}
        ids = [BOS]
        for step in range(MAX_TOKENS):
            feeds = {
                "input_ids": np.array([[ids[-1]]], dtype=np.int64),
                "encoder_hidden_states": hidden,
                "use_cache_branch": np.array([step > 0]),
                **past,
            }
            logits, *presents = self.decoder.run(None, feeds)
            next_id = int(logits[0, -1].argmax())
            if next_id == EOS:
                break
            ids.append(next_id)
            new = dict(zip(self.present_names, presents))
            for name in self.past_names:
                # Cross-attention (".encoder.") keys/values depend only on the image, so
                # they are computed once on step 0; afterwards the merged model returns
                # empty placeholders for them, which must not overwrite the real ones.
                if step == 0 or ".decoder." in name:
                    past[name] = new[name.replace("past_key_values", "present")]
        return _to_standard_latex(self.tokenizer.decode(ids[1:], skip_special_tokens=True))


# Texo's vocabulary includes a few KaTeX-only shorthands; Texo-web rewrites them the same way.
_KATEX_SHORTHANDS = {
    r"\infin": r"\infty", r"\rarr": r"\rightarrow", r"\larr": r"\leftarrow", r"\harr": r"\leftrightarrow",
    r"\Rarr": r"\Rightarrow", r"\Larr": r"\Leftarrow", r"\Harr": r"\Leftrightarrow",
    r"\darr": r"\downarrow", r"\uarr": r"\uparrow",
}


def _to_standard_latex(latex: str) -> str:
    return re.sub(r"\\(?:infin|rarr|larr|harr|Rarr|Larr|Harr|darr|uarr)(?![a-zA-Z])",
                  lambda m: _KATEX_SHORTHANDS[m.group(0)], latex)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ocr = TexoOCR()
    for p in sys.argv[1:]:
        print(ocr.predict(p))
