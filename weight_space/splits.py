"""Make the dev / test splits for the weight-space experiments (UniMER-Test handwritten split).

- test_100: the 100 formulas used in math_ocr/results/unimer_hwe_100 (seed 0). Some of its
  errors were looked at while building math_ocr, so it is reported but NEVER used to design edits.
- dev:      400 formulas (seed 1). The ONLY handwritten data an edit may be designed on.
- test:     1,000 formulas (seed 2), disjoint from both. Not looked at until the final evaluation.

Writes data/<split>/images/*.png and data/<split>/labels.csv (image,latex).
Usage:  python splits.py
"""

import csv
import io
import random
from pathlib import Path

HERE = Path(__file__).parent


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    table = pq.read_table(hf_hub_download("alephpi/UniMER-Test", "data/hwe-00000-of-00001.parquet", repo_type="dataset"))
    images, texts = table.column("image").to_pylist(), table.column("text").to_pylist()
    all_idx = list(range(len(texts)))

    used = set(random.Random(0).sample(all_idx, 100))  # same draw as math_ocr/fetch_unimer_hwe.py
    rest = [i for i in all_idx if i not in used]
    dev = sorted(random.Random(1).sample(rest, 400))
    rest = [i for i in rest if i not in set(dev)]
    test = sorted(random.Random(2).sample(rest, 1000))
    assert not (set(dev) & set(test)) and not (used & (set(dev) | set(test)))

    for name, idx in (("dev", dev), ("test", test)):
        out = HERE / "data" / name
        (out / "images").mkdir(parents=True, exist_ok=True)
        with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "latex"])
            for i in idx:
                rel = f"images/hwe_{i:05d}.png"
                Image.open(io.BytesIO(images[i])).save(out / rel)
                w.writerow([rel, texts[i]])
        print(f"{name}: {len(idx)} formulas -> {out}")


if __name__ == "__main__":
    main()
