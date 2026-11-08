# Weight-space programming on our handwriting models

An attempt to apply **weight-space programming**, an LLM reading and editing a network's
weights directly with no gradient descent ([Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/),
[logs](https://github.com/rikiyat/lm_understand-logs)), to every model in this repo: the EMNIST
CNN, Texo, TexTeller, Qwen3.5-2B-MathParser and Qwen3-VL-2B. The LLM doing the reading and
editing was Claude (Opus 5.5), working through Python on this laptop's CPU. The paper's
agent was a different model (OpenAI's "Astra"), so its results say nothing certain about ours.

**Result: reading weights worked; no edit improved any model.** The CNN's final layer alone
predicted the class pairs behind 65.7% of its test errors. Every edit that was tried either
moved errors between classes or replaced one wrong output with another. Diagnostics explain
why: in each model the errors come from genuinely ambiguous input, or from an ability the model
does not have. That is the limit the paper reports: editing "could remove an obstacle that
hid an ability, but it could not add an ability the model lacked."

## Rules

- **No gradients.** Edits come from reading weights, activations/logits and dev-set error
  counts, plus closed-form arithmetic (e.g. a log-ratio); nothing is trained.
- **Design on dev, judge on test.** Handwritten formulas from UniMER-Test `hwe`
  ([`splits.py`](splits.py)): `dev` 400, `test` 1,000, both disjoint from the 100 used in
  `math_ocr/results` (whose errors had already been looked at). CNN: dev = 20,000 EMNIST
  training images, test = all 116,323 EMNIST test images.
- **Budgeted and logged.** Every dev/test evaluation of an edit is logged in [`logs/`](logs/).
  An edit only goes to test if it helps on dev.

## Results

| Model | What was looked for | Edit tried | Dev | Test | Verdict |
|---|---|---|---|---|---|
| CNN (293k) | Which classes it confuses, from weights alone | none (read-only) | | top-5 predicted pairs = 65.7% of test errors; 4 of them in the measured top 6 | **reading works** |
| CNN | Hidden information about O/0, f/F, 1/L, 1/I, 5/S | bias shifts on O, L, I, f | 91.54% → 91.63% | 90.78% → 90.76% | moves errors (O 54→67%, 0 84→74%) |
| Texo (20M) | Token-level preferences | 687-value output bias (log-ratio of dev over/under-production), baked into the ONNX file | 51.25% → 51.5% | 50.8% → 51.2% (n=1,000) | noise |
| TexTeller (298M), handwriting | Any repeated mistake | none | 98.5% exact; 6 one-off errors | | nothing to fix |
| TexTeller, printed | Text-mode habit (`\;` for `^`, `\text`, `\mbox`, `$`) | −12 bias on 21 habit/runner-up tokens | 46.7% → 47.5% (6 fixed, 5 broken), similarity 0.828 → 0.763 | not run | rejected |
| Qwen3.5-2B MathParser | Superscript → subscript habit | + bias on superscript tokens | best bias fixes 1 decision in 150 formulas | not run | too small to measure |
| Qwen3-VL-2B | Same habit | same | no bias helps | not run | rejected |

### What each diagnostic showed

- **CNN.** The two most similar rows of the final layer are 0/O and F/f; the top five similar
  pairs account for 65.7% of all test errors. On held-out dev, the existing head separates
  each pair better than the best closed-form linear separator (Fisher LDA) on the same
  features, e.g. O/0 71.3% vs 68.0%. So the features hold no unused information; bymerge
  merges lowercase l into L and O/0 look the same in many hands. [`run_cnn.py`](run_cnn.py),
  [`logs/cnn_results.json`](logs/cnn_results.json).
- **Texo.** Its biggest error source was runaway repetition (990 extra "0" tokens), but it
  came from only 2 of 400 dev formulas: very wide formulas squeezed into its 384 px input
  until unreadable. That is input resolution, not weights. The remaining errors are scattered
  misreads (C/c, α/a, … vs ⋯). [`run_texo.py`](run_texo.py), [`texo_edit.py`](texo_edit.py).
- **TexTeller, printed.** On printed sans-serif formulas it drops superscripts (`32x\;6` for
  `32x^{6}`) and wraps math in `\text`/`\mbox`/`$`. Looking at its logits for 250 habit tokens
  on 120 printed dev formulas: the runner-up was a superscript/subscript token 0 times; it was
  mostly another formatting or spacing token (`\char` 34, `\tiny` 28, `\textbf` 20, `\textit`
  14, `\mathrm` 12, …), and the habit won by a median 8.2 logits. Penalizing them
  produced the next formatting token instead (`\ `, `\mathit`, `\texttt`, even
  `\eqref{eq:11.1}`) and one 2,000-character run of spaces. It does not read these superscripts,
  so there is no hidden ability to uncover. [`run_texteller.py`](run_texteller.py).
- **Qwen.** Both models write more `_` and fewer `^` than the truth, which looked like a habit.
  Recording the ^/_ logits at every decision shows otherwise: of 78 (Qwen3.5) and 72 (Qwen3-VL)
  subscript decisions, only 2 and 1 were wrong. One Qwen3.5 error was a near-miss (^ behind by
  1.3 logits), the single decision a +1.5 bias fixes; the others lost by 5.3 and 9.1, as
  confidently as correct choices, so a bias big enough to flip them breaks more than it fixes.
  The aggregate gap comes from larger structural misreads, not from a ^/_ bias. [`run_qwen.py`](run_qwen.py).

## How this compares with the paper

The paper's biggest win (Qwen3.5-0.8B, MATH-500 7.9% → 48.1%) removed one *behavioural
obstacle*: a thinking loop that hid answers the model could already produce. None of our
models had such an obstacle on the task that matters (handwriting): their errors are
ambiguous strokes or skills they lack, which the paper also found it could not fix (its
Gemma 3 270M result). Two conditions also differ: the paper used a frontier agent with up to
50 forward passes per experiment on GPU-free sandboxes; here edits were limited to output-layer
biases on small dev budgets (one CPU, ~10 s per Qwen formula), and edits deeper in the networks
were not tried.

## Reproduce

```bash
python splits.py                    # dev/test handwritten splits (needs pyarrow)
python render_printed_splits.py     # printed versions, for the TexTeller experiment
python run_cnn.py                   # needs EMNIST; set EMNIST_DIR
python run_texo.py dev --baseline --alphas 0.5 1 2 && python run_texo.py test --baseline --alpha 0.5
python run_texteller.py runnerup && python run_texteller.py eval --limit 120 --baseline --bias -12
python run_qwen.py instrument --variant qwen3.5-mathparser --limit 150 && python run_qwen.py choose --variant qwen3.5-mathparser
python run_qwen.py instrument --variant qwen3-vl --limit 150 && python run_qwen.py choose --variant qwen3-vl
python error_analysis.py <predictions.csv> <model>   # systematic errors in any evaluate.py output
```

`data/` and `work/` are regenerated by these scripts and not committed. The Texo bias is kept in
[`edits/texo_lm_head_bias.npy`](edits/texo_lm_head_bias.npy).
