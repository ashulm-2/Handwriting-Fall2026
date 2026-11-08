# Math OCR: handwritten formula → LaTeX

Scripts that take a raw image of a handwritten formula (photo, scan or canvas export)
and print its LaTeX, for four models, plus a benchmark that scores them against the
correct LaTeX. Everything runs on a CPU; a GPU is used automatically if present.

| Model | Size | License | How it runs |
|---|---|---|---|
| `texo` | 20M params | AGPL-3.0 | ONNX Runtime, no PyTorch needed ([alephpi/Texo](https://github.com/alephpi/Texo)) |
| `texteller` | 298M params | Apache-2.0 | PyTorch/transformers ([OleehyO/TexTeller](https://github.com/OleehyO/TexTeller)) |
| `qwen3.5-mathparser` | 2B params | Apache-2.0 | Qwen3.5-2B fine-tuned for handwritten math ([sugartai/Qwen3.5-2B-MathParser-pro](https://huggingface.co/sugartai/Qwen3.5-2B-MathParser-pro)) |
| `qwen3-vl` | 2B params | Apache-2.0 | General VLM used in the original VLM script ([Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)) |

Model weights download from Hugging Face on first use (Texo ~80 MB, TexTeller ~1.2 GB,
each Qwen ~4.5 GB) and are cached in `~/.cache/huggingface`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Mac/Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r math_ocr/requirements.txt
```

## Image → LaTeX

```bash
cd math_ocr
python ocr.py --model texteller photo.jpg          # prints the LaTeX
python ocr.py --model texo --json a.png b.png      # one JSON object per image, with timing
python ocr.py --model all photo.jpg                # every model, labelled
```

From Python, e.g. inside the website's backend. **Load each model once at start-up**
and reuse it; loading is far slower than predicting:

```python
import sys; sys.path.insert(0, "math_ocr")
from ocr import load_model

texo = load_model("texo")
texteller = load_model("texteller")
latex = texteller.predict("upload.png")   # a path, a PIL.Image, or an RGB numpy array
```

Every model returns bare LaTeX (no `$`/`\[ \]` delimiters). All inputs go through the
same loader (`common.load_image`), which applies phone-photo EXIF rotation and puts
transparent canvas PNGs on a white background (otherwise the ink would be black on black).

## Why the VLM took ~5 minutes per image without a GPU

The Qwen models turn the image into vision tokens in proportion to its pixel count. A
12-megapixel phone photo becomes thousands of tokens. On this laptop's CPU, the same
formula took **372 s** at full resolution and **14 s** after downscaling to ~0.5 MP,
with no loss of legibility for a single formula. `qwen_model.py` therefore downscales
to `max_pixels = 512*1024` by default (`QwenOCR(max_pixels=None)` disables it). Texo
and TexTeller resize to 384×384 and 448×448 internally, so they are not affected.

## Benchmark

`evaluate.py` runs models over a folder containing `labels.csv` (`image,latex`) and
the images, and writes `predictions.csv`, `summary.csv` and `summary.md`.

```bash
python fetch_unimer_hwe.py --n 100                  # public handwritten benchmark sample
python evaluate.py --data benchmark/unimer_hwe --out results/unimer_hwe_100
python evaluate.py --data testset --models texo texteller   # our own handwriting
```

Scoring (`metrics.py`) first normalizes both LaTeX strings so that different spellings
of the same rendered formula match (`x^2` = `x^{2}`, `\dfrac` = `\frac`, `\left(` = `(`,
`\mathrm{d}x` = `dx`, `\sin` = `sin`, `\le` = `\leq`, ...). It then reports the CROHME
"ExpRate" measures, the percentage of formulas exactly right, and the percentages within
1 or 2 token edits, plus the mean similarity (1 − edit distance / length). `python
test_metrics.py` checks the normalizer.

### Results: 100 handwritten formulas (UniMER-Test `hwe`, random sample, seed 0)

Laptop CPU (AMD Ryzen AI Max+ 395, 16 cores, no GPU), Python 3.10, float32.

| Model | Exactly right | ≤1 edit | ≤2 edits | Similarity | Seconds / image | Load time |
|---|---|---|---|---|---|---|
| TexTeller | **94%** | 99% | 99% | 0.987 | 1.6 | 5 s |
| Qwen3.5-2B MathParser | 71% | 83% | 90% | 0.949 | 9.4 | 4 s* |
| Qwen3-VL-2B (general) | 65% | 76% | 84% | 0.939 | 8.6 | 4 s* |
| Texo | 57% | 75% | 81% | 0.906 | **0.05** | 1 s |

\*after the first download; a cold start reads ~4.5 GB from disk (60–70 s here).
Full per-image output: [`results/unimer_hwe_100/predictions.csv`](results/unimer_hwe_100/predictions.csv).

Caveats: these images come from the CROHME and HME100K **test** sets, but they are
public and widely used, so models trained on large scraped collections may have seen
them; TexTeller's lead should be confirmed on our own handwriting (`testset/`). A few
HME100K labels contain stray characters (e.g. Chinese punctuation), which cost every
model equally. Most remaining errors are genuinely ambiguous handwriting (g/q, 5/s,
D/P, 2/z).

### Sanity check: our 42 test formulas, printed (not handwritten)

`testset/render_printed.py` renders 41 of the 42 test-set formulas with matplotlib (it
cannot draw the matrix). Printed input is far easier than handwriting:

| Model | Exactly right (of 41) | Altered-famous formulas copied correctly |
|---|---|---|
| Qwen3-VL-2B | 100% | 7 / 7 |
| Qwen3.5-2B MathParser | 98% | 7 / 7 |
| Texo | 98% | 6 / 7 (garbled #42; did not "autocorrect" it) |
| TexTeller | 85% | 6 / 7 (garbled #41; did not "autocorrect" it) |

On clean print, **no model rewrote an altered famous formula back to the famous one**;
e.g. all four wrote `E = mc^{3}`. Whether that still holds when the handwriting is
ambiguous is what the handwritten test set is for. The ranking is also reversed relative
to handwriting: TexTeller is best on handwriting but weakest on matplotlib's sans-serif
print (it read γ as y and wrapped output in `\mbox`/`\hbox`). Results:
[`results/printed_testset/`](results/printed_testset/).

## Our own test set (`testset/`)

`testset/labels.csv` lists 42 formulas with their correct LaTeX. Most are deliberately
*unfamiliar*, so a model cannot get them right from memory. Seven are **altered famous
formulas**, such as `a^2+b^2=c^3`, `E=mc^3` and the quadratic formula with `+4ac`. A
model that reads the handwriting copies the change; a model that recognizes the famous
formula and "autocorrects" it gets these wrong.

1. Open `testset/prompt_sheet.html` (regenerate with `python testset/make_prompt_sheet.py`).
2. Write each formula by hand; save each photo as `testset/images/NN.jpg` (the name on its card).
3. `python evaluate.py --data testset`. Rows without a photo yet are skipped.

`testset/render_printed.py` renders the same formulas as clean printed images
(`benchmark/printed/`), so the whole pipeline can be tested before the handwriting exists.

## Files

| File | Purpose |
|---|---|
| `ocr.py` | command line + `load_model(name)` for all models |
| `texo_model.py`, `texteller_model.py`, `qwen_model.py` | one class per model, each with `.predict(image)`; each also runs on its own |
| `common.py` | image loading (EXIF, transparency), resizing, device choice |
| `metrics.py`, `test_metrics.py` | LaTeX normalization and scoring, and its checks |
| `evaluate.py` | the benchmark |
| `fetch_unimer_hwe.py` | downloads a sample of the UniMER-Test handwritten split |
| `testset/` | our own handwritten test set: labels, prompt sheet, printed renders |
| `results/` | benchmark outputs |

## Notes on the implementations

- **Texo**: the encoder is a custom HGNetV2 that plain `transformers` cannot load, so
  this runs the published ONNX files with a greedy decoder and key/value cache. The
  cached decoder was checked token-for-token against the uncached decoder. Preprocessing
  matches Texo's `EvalMERImageProcessor`. Texo's KaTeX shorthands (`\rarr`, `\infin`)
  are rewritten to standard LaTeX, as Texo-web does. **AGPL-3.0**: if Texo is served
  from a public website, the site's source code must be offered to its users.
- **TexTeller**: the official `texteller` package pins `transformers==4.47` (too old
  for Qwen3.5) and pulls in Ray and Streamlit, so `texteller_model.py` loads the same
  weights with plain `transformers` and reproduces its preprocessing and output
  clean-up. On 30 handwritten benchmark images, its output was identical to
  `texteller==1.0.2`'s `img2latex`, apart from the package's cosmetic line formatting
  and the `\[ \]` wrapper this version strips.
- **Qwen**: prompts are in `qwen_model.CHECKPOINTS`. The MathParser prompt is the one on
  its model card; thinking mode is disabled for Qwen3.5.
