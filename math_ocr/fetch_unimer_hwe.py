"""Download a random sample of the UniMER-Test handwritten split as a ready-to-score test set.

UniMER-Test (https://huggingface.co/datasets/alephpi/UniMER-Test) is the standard
formula-recognition benchmark. Its "hwe" (handwritten expression) split has 6,332
handwritten formulas taken from the CROHME and HME100K test sets, each with its
correct LaTeX. It is a useful baseline until our own handwritten set is ready.

Writes DIR/images/*.png and DIR/labels.csv (columns: image,latex), the format
evaluate.py expects.

Usage:
    python fetch_unimer_hwe.py --n 100 --out benchmark/unimer_hwe
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path

PARQUET = "data/hwe-00000-of-00001.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="how many formulas to sample (0 = all 6,332)")
    ap.add_argument("--seed", type=int, default=0, help="random seed, so the sample is reproducible")
    ap.add_argument("--out", default="benchmark/unimer_hwe")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    table = pq.read_table(hf_hub_download("alephpi/UniMER-Test", PARQUET, repo_type="dataset"))
    images, texts = table.column("image").to_pylist(), table.column("text").to_pylist()

    indices = list(range(len(texts)))
    if args.n and args.n < len(indices):
        indices = sorted(random.Random(args.seed).sample(indices, args.n))

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "latex"])
        for i in indices:
            name = f"images/hwe_{i:05d}.png"
            Image.open(io.BytesIO(images[i])).save(out / name)
            writer.writerow([name, texts[i]])
    print(f"wrote {len(indices)} images and labels.csv to {out}")


if __name__ == "__main__":
    main()
