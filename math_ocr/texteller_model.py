"""TexTeller: image of a formula -> LaTeX.

TexTeller (https://github.com/OleehyO/TexTeller, Apache-2.0) is a ~300M-parameter
ViT encoder + RoBERTa-style decoder trained on 80M image/formula pairs, including
handwritten ones.

The official `texteller` pip package pins transformers==4.47 and pulls in Ray and
Streamlit, which clashes with the newer transformers the Qwen models need. So this
file loads the same Hugging Face weights with plain `transformers` and reproduces the
package's preprocessing and post-processing (texteller 1.0.2: utils/image.py
`transform`, utils/latex.py `remove_style` / `add_newlines`). The output was checked
against the official package; see math_ocr/README.md.

Usage:
    python texteller_model.py path/to/image.png [more images...]
"""

from __future__ import annotations

import re
import sys
from collections import Counter

import numpy as np

from common import ImageInput, load_image, pick_device

REPO_ID = "OleehyO/TexTeller"

# From texteller/constants.py
FIXED_IMG_SIZE = 448
IMAGE_MEAN = 0.9545467
IMAGE_STD = 0.15394445
MAX_TOKENS = 1024


def _trim_white_border(image: np.ndarray) -> np.ndarray:
    """Crop away the background, whose colour is taken from the most common corner."""
    import cv2

    corners = [tuple(image[0, 0]), tuple(image[0, -1]), tuple(image[-1, 0]), tuple(image[-1, -1])]
    bg = np.array(Counter(corners).most_common(1)[0][0], dtype=np.uint8)
    diff = cv2.absdiff(image, np.full_like(image, bg))
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:  # blank image: nothing to crop to
        return image
    return image[y : y + h, x : x + w]


def _remove_style(latex: str) -> str:
    """Strip \\bm, \\boldsymbol, \\textit, \\textbf, \\mathbf wrappers, keeping their content."""
    for cmd in (r"\bm", r"\boldsymbol", r"\textit", r"\textbf", r"\mathbf"):
        latex = _unwrap(latex, cmd)
    return latex.strip()


def _unwrap(s: str, cmd: str) -> str:
    """Replace every `cmd{...}` (balanced braces) with ` ... `."""
    out, i = [], 0
    while True:
        j = s.find(cmd + "{", i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k, depth = j + len(cmd) + 1, 1
        while k < len(s) and depth:
            if s[k] == "\\":
                k += 2
                continue
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        if depth:  # unbalanced: leave the rest untouched
            out.append(s[j:])
            return "".join(out)
        out.append(" " + _unwrap(s[j + len(cmd) + 1 : k - 1], cmd) + " ")
        i = k


def _add_newlines(latex: str) -> str:
    """Put \\begin/\\end and \\\\ on their own lines, as the official package does."""
    latex = re.sub(r"\s*(\\begin\{[^}]*\})\s*", r"\n\g<1>\n", latex)
    latex = re.sub(r"\s*(\\end\{[^}]*\})\s*", r"\n\g<1>\n", latex)
    latex = re.sub(r"\\\\(?!\n| )|\\\\ ", r"\\\\\n", latex)
    latex = re.sub(r"\n{2,}", "\n", latex)
    return latex.strip()


class TexTellerOCR:
    name = "texteller"

    def __init__(self, device: str | None = None, num_beams: int = 1, threads: int | None = None):
        import torch
        from torchvision.transforms import v2
        from transformers import AutoTokenizer, VisionEncoderDecoderModel

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = pick_device(device)
        self.num_beams = num_beams
        self.model = VisionEncoderDecoderModel.from_pretrained(REPO_ID).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Grayscale(),
                v2.Resize(
                    size=FIXED_IMG_SIZE - 1,
                    interpolation=v2.InterpolationMode.BICUBIC,
                    max_size=FIXED_IMG_SIZE,
                    antialias=True,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),
            ]
        )

    def preprocess(self, image: ImageInput):
        rgb = np.asarray(load_image(image))
        x = self.transform(_trim_white_border(rgb))
        # Pad right/bottom with zeros to a fixed 448x448, as the official package does.
        return self.torch.nn.functional.pad(
            x, (0, FIXED_IMG_SIZE - x.shape[2], 0, FIXED_IMG_SIZE - x.shape[1])
        )

    def predict(self, image: ImageInput) -> str:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[ImageInput]) -> list[str]:
        from transformers import GenerationConfig

        pixels = self.torch.stack([self.preprocess(im) for im in images]).to(self.device)
        config = GenerationConfig(
            max_new_tokens=MAX_TOKENS,
            num_beams=self.num_beams,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        with self.torch.inference_mode():
            ids = self.model.generate(pixels, generation_config=config)
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        return [_strip_display_math(_add_newlines(_remove_style(t))) for t in texts]


def _strip_display_math(latex: str) -> str:
    """TexTeller wraps formulas in \\[ ... \\]; return the bare LaTeX like the other models."""
    s = latex.strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()
    return s


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ocr = TexTellerOCR()
    for p in sys.argv[1:]:
        print(ocr.predict(p))
