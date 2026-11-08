"""Weight-space edit for Texo: a per-token bias on the decoder's output layer.

Texo's lm_head is a bias-free 384x687 MatMul (one column per vocabulary token). The edit
adds a 687-value bias after it, i.e. new final-layer weights. During experiments the bias is
applied inside the decoding loop (identical maths: logits + bias); `bake()` then writes it
into a copy of decoder_model_merged.onnx as a real initializer + Add node.

No gradients are used anywhere: the bias values are chosen by reading dev-set error counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "math_ocr"))
import texo_model  # noqa: E402
from texo_model import BOS, EOS, MAX_TOKENS, TexoOCR, _to_standard_latex  # noqa: E402


class TexoWithBias(TexoOCR):
    name = "texo-edited"

    def __init__(self, bias: np.ndarray | None = None, **kw):
        super().__init__(**kw)
        self.bias = np.zeros(687, dtype=np.float32) if bias is None else bias.astype(np.float32)
        self.vocab = self.tokenizer.get_vocab()

    def token_id(self, tok: str) -> int:
        return self.vocab[tok]

    def predict(self, image) -> str:
        hidden = self.encoder.run(None, {"pixel_values": self.preprocess(image)})[0]
        empty = np.zeros((1, self.heads, 0, self.head_dim), dtype=np.float32)
        past = {name: empty for name in self.past_names}
        ids = [BOS]
        for step in range(MAX_TOKENS):
            feeds = {"input_ids": np.array([[ids[-1]]], dtype=np.int64), "encoder_hidden_states": hidden,
                     "use_cache_branch": np.array([step > 0]), **past}
            logits, *presents = self.decoder.run(None, feeds)
            next_id = int((logits[0, -1] + self.bias).argmax())
            if next_id == EOS:
                break
            ids.append(next_id)
            new = dict(zip(self.present_names, presents))
            for name in self.past_names:
                if step == 0 or ".decoder." in name:
                    past[name] = new[name.replace("past_key_values", "present")]
        return _to_standard_latex(self.tokenizer.decode(ids[1:], skip_special_tokens=True))


def bake(bias: np.ndarray, out_path: Path) -> Path:
    """Write a copy of Texo's merged decoder with `bias` added after lm_head (a real weight edit)."""
    import onnx
    from huggingface_hub import hf_hub_download
    from onnx import helper, numpy_helper

    model = onnx.load(hf_hub_download(texo_model.REPO_ID, "onnx/decoder_model_merged.onnx"))
    bias_name = "lm_head_bias_edit"
    added = 0

    def patch(graph):
        nonlocal added
        for node in list(graph.node):
            if node.op_type == "MatMul" and node.output and node.output[0] == "logits":
                raw = "logits_before_bias_edit"
                node.output[0] = raw
                graph.node.insert(list(graph.node).index(node) + 1,
                                  helper.make_node("Add", [raw, bias_name], ["logits"], name="lm_head_bias_edit_add"))
                added += 1
            for a in node.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    patch(a.g)

    patch(model.graph)
    model.graph.initializer.append(numpy_helper.from_array(bias.astype(np.float32), bias_name))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path
