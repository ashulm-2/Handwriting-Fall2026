"""Qwen vision-language models: image of a formula -> LaTeX.

Two checkpoints are wired up:
- "qwen3-vl"           Qwen/Qwen3-VL-2B-Instruct, the general-purpose VLM from Drew's script.
- "qwen3.5-mathparser" sugartai/Qwen3.5-2B-MathParser-pro, Qwen3.5-2B fine-tuned for
                       handwritten-math OCR (prompt taken from its model card).

Both have ~2B parameters, so they are the slowest of the models here, especially on CPU.
Most of that cost is the number of image tokens: a 12-megapixel phone photo becomes
thousands of vision tokens. The image is therefore downscaled to `max_pixels` first
(default ~0.5 MP, plenty for a handwritten formula), which is the biggest speed-up
available on a machine without a GPU.

Usage:
    python qwen_model.py [--model qwen3.5-mathparser|qwen3-vl] path/to/image.png [...]
"""

from __future__ import annotations

import argparse
import re

from common import ImageInput, load_image, pick_device, shrink_to_fit

CHECKPOINTS = {
    "qwen3-vl": {
        "repo": "Qwen/Qwen3-VL-2B-Instruct",
        "system": None,
        "template_kwargs": {},
        "prompt": (
            "Convert the handwritten mathematical expression in this image to LaTeX. "
            "Transcribe exactly what is written, even if it looks unusual or incorrect. "
            "Output only the LaTeX, with no $ delimiters and no explanation."
        ),
    },
    "qwen3.5-mathparser": {
        "repo": "sugartai/Qwen3.5-2B-MathParser-pro",
        "system": "You are a handwritten mathematical OCR model. Return only the LaTeX transcription.",
        "prompt": "Transcribe the handwritten mathematical formula into LaTeX only.",
        "template_kwargs": {"enable_thinking": False},  # answer directly, no <think> block
    },
}

DEFAULT_MAX_PIXELS = 512 * 1024


def clean_output(text: str) -> str:
    """Strip markdown fences and $ / \\[ \\] delimiters that chat models like to add."""
    text = text.strip()
    text = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", text).strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left) : -len(right)].strip()
            break
    return text


class QwenOCR:
    def __init__(
        self,
        variant: str = "qwen3.5-mathparser",
        device: str | None = None,
        dtype: str | None = None,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
        max_new_tokens: int = 512,
        threads: int | None = None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if variant not in CHECKPOINTS:
            raise ValueError(f"Unknown Qwen variant {variant!r}; choose from {list(CHECKPOINTS)}")
        if threads:
            torch.set_num_threads(threads)
        self.name = variant
        self.cfg = CHECKPOINTS[variant]
        self.torch = torch
        self.device = pick_device(device)
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens

        # bfloat16 halves memory and is fast on GPUs; float32 is the safe CPU default.
        if dtype is None:
            dtype = "bfloat16" if self.device.startswith("cuda") else "float32"
        self.processor = AutoProcessor.from_pretrained(self.cfg["repo"])
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg["repo"], dtype=getattr(torch, dtype)
        ).to(self.device).eval()

    def predict(self, image: ImageInput) -> str:
        img = shrink_to_fit(load_image(image), self.max_pixels)
        messages = []
        if self.cfg["system"]:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.cfg["system"]}]})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": self.cfg["prompt"]}],
            }
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.cfg["template_kwargs"],
        ).to(self.device)

        tok = self.processor.tokenizer
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return clean_output(tok.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="qwen3.5-mathparser", choices=list(CHECKPOINTS))
    args = ap.parse_args()
    ocr = QwenOCR(args.model)
    for p in args.images:
        print(ocr.predict(p))
