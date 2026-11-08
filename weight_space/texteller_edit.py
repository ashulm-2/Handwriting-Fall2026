"""Weight-space edit for TexTeller: a bias on the decoder's output layer.

TexTeller's output projection is tied to its token embedding and has no bias, so the edit adds a
15,000-value bias to it (new weights; the embedding is untouched). It lowers the tokens of one
habit: on printed formulas TexTeller writes a spacing command (\\;) or text-mode wrappers
(\\text, \\mbox, $...$) where a superscript, subscript or plain math belongs.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "math_ocr"))
from texteller_model import MAX_TOKENS, TexTellerOCR  # noqa: E402

HABIT_TOKENS = ["\\;", "$", "text", "mbox", "hbox", "emph", "mathbin", "mathsf", "kern", "phantom", "raise", "vbox"]


class TexTellerWithBias(TexTellerOCR):
    name = "texteller-edited"

    def __init__(self, token_bias: dict[str, float] | None = None, **kw):
        super().__init__(**kw)
        torch = self.torch
        self.proj = self.model.decoder.output_projection
        if self.proj.bias is None:
            self.proj.bias = torch.nn.Parameter(torch.zeros(self.proj.weight.shape[0]), requires_grad=False)
        self.set_bias(token_bias or {})

    def token_id(self, tok: str) -> int:
        i = self.tokenizer.convert_tokens_to_ids(tok)
        if i == self.tokenizer.unk_token_id:
            raise KeyError(tok)
        return i

    def set_bias(self, token_bias: dict[str, float]) -> None:
        with self.torch.no_grad():
            self.proj.bias.zero_()
            for tok, v in token_bias.items():
                self.proj.bias[self.token_id(tok)] = v

    def runner_up_report(self, image, habit: list[str]) -> list[dict]:
        """For each generated habit token: the best non-habit alternative and how far behind it was."""
        from transformers import GenerationConfig

        habit_ids = {self.token_id(t) for t in habit}
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        cfg = GenerationConfig(max_new_tokens=MAX_TOKENS, num_beams=1, do_sample=False,
                               pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                               bos_token_id=self.tokenizer.bos_token_id)
        with self.torch.inference_mode():
            out = self.model.generate(x, generation_config=cfg, output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0].tolist()[1:]  # drop the start token
        report = []
        for k, s in enumerate(out.scores):
            if k < len(seq) and seq[k] in habit_ids:
                logits = s[0].clone()
                chosen = float(logits[seq[k]])
                logits[list(habit_ids)] = float("-inf")
                alt = int(logits.argmax())
                report.append({"habit": self.tokenizer.convert_ids_to_tokens(seq[k]),
                               "alternative": self.tokenizer.convert_ids_to_tokens(alt),
                               "gap": chosen - float(logits[alt])})
        return report
