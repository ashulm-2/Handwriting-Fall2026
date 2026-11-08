"""Weight-space edit for the Qwen models: a bias on the output layer (lm_head).

Qwen3.5's lm_head is tied to its input embedding (tie_word_embeddings = true), so editing its
rows would also change how the model reads those tokens. Instead the edit adds a bias vector
(one value per vocabulary token) to lm_head: new weights that leave the embedding untouched.
Chosen from dev-set measurements only; no gradients.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "math_ocr"))
from common import load_image, shrink_to_fit  # noqa: E402
from qwen_model import QwenOCR, clean_output  # noqa: E402


class QwenWithBias(QwenOCR):
    def __init__(self, variant: str, token_bias: dict[str, float] | None = None, **kw):
        super().__init__(variant, **kw)
        import torch

        self.name = variant + "-edited"
        lm = self.model.get_output_embeddings()
        if lm.bias is None:
            lm.bias = torch.nn.Parameter(torch.zeros(lm.weight.shape[0], dtype=lm.weight.dtype), requires_grad=False)
        self.lm = lm
        self.set_bias(token_bias or {})

    def token_id(self, tok: str) -> int:
        return self.processor.tokenizer.convert_tokens_to_ids(tok)

    def set_bias(self, token_bias: dict[str, float]) -> None:
        with self.torch.no_grad():
            self.lm.bias.zero_()
            for tok, value in token_bias.items():
                self.lm.bias[self.token_id(tok)] = value

    def predict_with_scores(self, image, watch: list[str]) -> tuple[str, list[dict]]:
        """Greedy prediction plus, for every generated step, the logits of the `watch` tokens."""
        img = shrink_to_fit(load_image(image), self.max_pixels)
        messages = []
        if self.cfg["system"]:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.cfg["system"]}]})
        messages.append({"role": "user", "content": [{"type": "image", "image": img},
                                                     {"type": "text", "text": self.cfg["prompt"]}]})
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt",
                                                    **self.cfg["template_kwargs"]).to(self.device)
        tok = self.processor.tokenizer
        ids = [self.token_id(w) for w in watch]
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                                      pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                                      output_scores=True, return_dict_in_generate=True)
        new = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        steps = [{"chosen": new[k], "prefix": tok.decode(new[:k], skip_special_tokens=True),
                  **{w: float(s[0, i]) for w, i in zip(watch, ids)}} for k, s in enumerate(out.scores)]
        return clean_output(tok.decode(new, skip_special_tokens=True)), steps
