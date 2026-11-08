"""Shared setup for the CNN experiments: the model, EMNIST bymerge data, and cached features.

dev  = the first 20,000 EMNIST bymerge TRAINING images (edits may be designed on these only).
test = all 116,323 EMNIST bymerge TEST images (final evaluation only).

`features(split)` returns the 128-d vector the CNN's final linear layer sees (after global
average pooling) plus the labels, cached in work/ so the conv layers run once per split.
Editing the final layer can then be scored without re-running the conv layers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
EMNIST_DIR = os.environ.get("EMNIST_DIR", r"C:\Users\notim\Documents\MURL-FA26---Sammy-Patel\data\emnist")
WORK = HERE / "work"
DEV_SIZE = 20_000

import CNN  # noqa: E402
import design  # noqa: E402

CLASSES = [design.CToC(i) for i in range(47)]


def load_cnn() -> "CNN.CNN":
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):  # CNN() prints its class count
        model = CNN.CNN()
    model.load_state_dict(torch.load(REPO / "CNNEMNIST.pt", map_location="cpu"))
    return model.eval()


def _dataset(split: str):
    from torchvision.datasets import EMNIST

    ds = EMNIST(EMNIST_DIR, split="bymerge", train=(split == "dev"), download=False, transform=design.IMAGE_TRANSFORM)
    if split == "dev":
        ds = torch.utils.data.Subset(ds, range(DEV_SIZE))
    return ds


def features(split: str, model=None) -> tuple[np.ndarray, np.ndarray]:
    """(N x 128 penultimate features, N labels) for 'dev' or 'test', cached."""
    WORK.mkdir(exist_ok=True)
    cache = WORK / f"cnn_feats_{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"]
    model = model or load_cnn()
    torch.set_num_threads(16)
    dl = torch.utils.data.DataLoader(_dataset(split), batch_size=2048, shuffle=False)
    xs, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            xs.append(model.conv_block(x).flatten(1).numpy())
            ys.append(y.numpy())
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.savez(cache, x=x, y=y)
    return x, y


def head(model) -> tuple[np.ndarray, np.ndarray]:
    """Final linear layer: W (47 x 128), b (47)."""
    lin = model.fc_block[1]
    return lin.weight.detach().numpy().copy(), lin.bias.detach().numpy().copy()


def accuracy(W: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    pred = (x @ W.T + b).argmax(1)
    per_class = np.array([(pred[y == c] == c).mean() for c in range(47)])
    return float((pred == y).mean()), per_class
