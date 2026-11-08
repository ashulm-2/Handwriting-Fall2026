"""CNN weight-space experiments (EMNIST bymerge), reproducing every number in README.md.

1. Read-only: predict the most-confused class pairs from the final layer's weights alone
   (cosine similarity of class rows), then check against the measured test confusions.
2. Is there hidden information? For each predicted pair, compare the current head with the
   best closed-form linear separator (Fisher LDA) fitted on one half of dev, scored on the other.
3. Edit: per-class bias shifts chosen on dev, scored once on test.

No gradients anywhere. Writes logs/cnn_results.json.   Usage:  python run_cnn.py
"""

import json
from pathlib import Path

import numpy as np

from cnn_common import CLASSES, accuracy, features, head, load_cnn

HERE = Path(__file__).parent
IDX = {c: i for i, c in enumerate(CLASSES)}
# Chosen on dev in the exploration: the bias shift that maximized dev accuracy for each pair.
BIAS_EDIT = {"O": 0.50, "L": 0.40, "I": -0.35, "f": 0.50}


def main():
    model = load_cnn()
    W, b = head(model)
    out = {}

    # 1. read-only prediction
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    S = Wn @ Wn.T
    np.fill_diagonal(S, -1)
    pairs = sorted({tuple(sorted((i, j))) for i in range(47) for j in range(47) if i != j}, key=lambda p: -S[p])
    predicted = [f"{CLASSES[i]}/{CLASSES[j]}" for i, j in pairs[:5]]
    x, y = features("test", model)
    pred = (x @ W.T + b).argmax(1)
    C = np.zeros((47, 47), int)
    np.add.at(C, (y, pred), 1)
    measured = sorted(((int(C[i, j] + C[j, i]), f"{CLASSES[i]}/{CLASSES[j]}") for i in range(47) for j in range(i + 1, 47)), reverse=True)
    errors = len(y) - np.trace(C)
    share = sum(c for c, p in measured if p in predicted) / errors
    out["read_only"] = {"predicted_from_weights": predicted, "measured_top6": [p for _, p in measured[:6]],
                        "share_of_test_errors_in_predicted_pairs": round(float(share), 4)}

    # 2. hidden information check
    xd, yd = features("dev", model)
    Ld = xd @ W.T + b
    A, B = np.arange(len(yd) // 2), np.arange(len(yd) // 2, len(yd))
    sep = {}
    for p in predicted + ["I/L"]:
        i, j = (IDX[c] for c in p.split("/"))
        xi, xj = xd[A][yd[A] == i], xd[A][yd[A] == j]
        w = np.linalg.solve(np.cov(xi.T) + np.cov(xj.T) + 1e-3 * np.eye(128), xj.mean(0) - xi.mean(0))
        thr = ((xi @ w).mean() + (xj @ w).mean()) / 2
        r = B[np.isin(yd[B], [i, j])]
        sep[p] = {"head": round(float(((Ld[r, j] > Ld[r, i]) == (yd[r] == j)).mean()), 3),
                  "fisher": round(float((((xd[r] @ w) > thr) == (yd[r] == j)).mean()), 3), "n": int(len(r))}
    out["pairwise_separability_heldout_dev"] = sep

    # 3. bias edit
    b2 = b.copy()
    for c, d in BIAS_EDIT.items():
        b2[IDX[c]] += d
    res = {}
    for split, (xs, ys) in (("dev", (xd, yd)), ("test", (x, y))):
        a0, p0 = accuracy(W, b, xs, ys)
        a1, p1 = accuracy(W, b2, xs, ys)
        res[split] = {"base": round(a0, 4), "edited": round(a1, 4),
                      "per_class": {c: [round(float(p0[IDX[c]]), 3), round(float(p1[IDX[c]]), 3)] for c in ("O", "0", "f", "F", "L", "1")}}
    out["bias_edit"] = {"shifts": BIAS_EDIT, **res}

    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "cnn_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
