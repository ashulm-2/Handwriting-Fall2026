"""Checks for metrics.normalize/score. Run: python test_metrics.py"""

from metrics import normalize, normalized_string, score

SAME = [
    ("x^2", "x^{2}"),
    ("x^{2}", "x^{ {2} }"),
    (r"\frac12", r"\frac{1}{2}"),
    (r"\dfrac{a}{b}", r"\frac { a } { b }"),
    (r"\left( x \right)", "(x)"),
    (r"\left. \frac{d}{dx} \right|_{0}", r"\frac{d}{dx}|_0"),
    (r"\int_0^1 f(x)\,dx", r"\int_{0}^{1}f(x)dx"),
    (r"\int_0^1 f(x) \mathrm{d}x", r"\int_{0}^{1}f(x)dx"),
    (r"\sin x", r"\operatorname{sin} x"),
    (r"\sin x", "sin x"),
    (r"a \le b", r"a \leq b"),
    (r"f'(x)", r"f^{\prime}(x)"),
    (r"f''", r"f^{\prime\prime}"),
    (r"$x+1$", "x+1"),
    (r"\[ x+1 \]", "x+1"),
    (r"\displaystyle\sum_{i=1}^n i", r"\sum_{i=1}^{n} i"),
    (r"\sqrt[3]x", r"\sqrt[3]{x}"),
    (r"x^\frac12", r"x^{\frac{1}{2}}"),
    (r"\mathbf{v}\cdot\mathbf{w}", r"v \cdot w"),
    (r"\{ x \}", r"\lbrace x \rbrace"),
    (r"|w| < \varepsilon", r"|w| < \epsilon"),
    (r"P(A \mid B)", "P(A|B)"),
    (r"CE \bot OD", r"CE \perp OD"),
    (r"\overline{z} \cdot z", r"\bar z \cdot z"),
    ("\\begin{array}\n{l}\\lambda^{2}=0\n\\end{array}", r"\lambda^2=0"),
    (r"\lim_{x \to \infty}", r"\lim \limits_{x \rightarrow \infty}"),
    (r"\begin{pmatrix} 2 & -1 \\ 0 & 4 \end{pmatrix}", r"\begin{pmatrix}2&-1\\0&4\end{pmatrix}"),
]

DIFFERENT = [
    ("x^2", "x_2"),
    ("x^{2}", "x2"),
    (r"\frac{a}{b}", r"\frac{b}{a}"),
    (r"\alpha", "a"),
    ("a+b", "a-b"),
    (r"\begin{array}{l} a \\ b \end{array}", "a b"),  # two rows: must not be unwrapped
]


def main():
    failed = 0
    for a, b in SAME:
        if normalize(a) != normalize(b):
            failed += 1
            print(f"FAIL same: {a!r} -> {normalized_string(a)!r}  vs  {b!r} -> {normalized_string(b)!r}")
    for a, b in DIFFERENT:
        if normalize(a) == normalize(b):
            failed += 1
            print(f"FAIL different: {a!r} and {b!r} both -> {normalized_string(a)!r}")

    s = score(r"\frac{a}{b}+c", r"\frac{a}{b}+d")
    assert s["edit_distance"] == 1 and s["within1"] and not s["exact"], s
    s = score("", "x")
    assert s["edit_distance"] == 1 and s["ter"] == 1.0, s
    s = score(r"x^2", r"x^{2}")
    assert s["exact"] and s["similarity"] == 1.0, s

    total = len(SAME) + len(DIFFERENT)
    print(f"{total - failed}/{total} normalization checks passed; score() checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
