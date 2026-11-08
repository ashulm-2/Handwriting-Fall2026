"""Build prompt_sheet.html: every formula in labels.csv, rendered, with its photo filename.

Open the HTML in a browser and print it (or keep it on screen), write each formula by
hand, and save the photo under the filename shown (e.g. images/07.jpg).

Usage:  python make_prompt_sheet.py
"""

import csv
import html
from pathlib import Path

HERE = Path(__file__).parent

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Handwriting Prompt Sheet</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111; background: #fff; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p.how {{ margin: 0 0 16px; color: #444; max-width: 70ch; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; }}
  .card {{ border: 1px solid #bbb; border-radius: 6px; padding: 10px 12px; break-inside: avoid; }}
  .head {{ display: flex; justify-content: space-between; font-size: 12px; color: #555; }}
  .math {{ font-size: 20px; padding: 10px 0 6px; overflow-x: auto; }}
  .note {{ font-size: 12px; color: #a33; }}
  @media print {{ body {{ margin: 8mm; }} .grid {{ grid-template-columns: 1fr 1fr; }} }}
</style></head><body>
<h1>Handwriting prompt sheet ({n} formulas)</h1>
<p class="how">Write each formula by hand in your normal handwriting, one per photo, on plain paper
(or on the online canvas). Save each photo with the filename in its box, inside
<code>testset/images/</code>, then run <code>python evaluate.py --data testset</code>.
Formulas marked "altered-famous" are deliberately changed versions of well-known formulas:
copy them exactly as shown, because they test whether a model reads what is written or
"autocorrects" it to the famous version.</p>
<div class="grid">
{cards}
</div>
<script>
  document.querySelectorAll('.math').forEach(el =>
    katex.render(el.dataset.tex, el, {{ displayMode: true, throwOnError: false }}));
</script>
</body></html>
"""


def main():
    with open(HERE / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cards = []
    for r in rows:
        note = f'<div class="note">{html.escape(r["note"])}</div>' if r.get("category") == "altered-famous" else ""
        cards.append(
            f'<div class="card"><div class="head"><b>{html.escape(r["image"])}</b>'
            f'<span>{html.escape(r["category"])}</span></div>'
            f'<div class="math" data-tex="{html.escape(r["latex"], quote=True)}"></div>{note}</div>'
        )
    (HERE / "prompt_sheet.html").write_text(PAGE.format(n=len(rows), cards="\n".join(cards)), encoding="utf-8")
    print(f"wrote {HERE / 'prompt_sheet.html'} ({len(rows)} formulas)")


if __name__ == "__main__":
    main()
