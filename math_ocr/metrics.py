"""Scoring a predicted LaTeX string against the correct one.

Two LaTeX strings can look different but render the same: `x^2` vs `x^{2}`, `\\dfrac`
vs `\\frac`, `\\left(` vs `(`, `\\mathrm{d}x` vs `dx`, `\\sin` vs `\\operatorname{sin}`.
Comparing raw strings would count those as errors, so both sides are first turned into
a canonical list of tokens:

1. Split into tokens: commands (\\frac), escaped symbols (\\{), and single characters.
2. Drop tokens that only change spacing or size (\\, \\quad \\left \\right \\displaystyle \\big ...).
3. Map synonyms to one spelling (\\le -> \\leq, \\dfrac -> \\frac, \\to -> \\rightarrow ...).
4. Unwrap font commands (\\mathrm{d} -> d) and spell out named functions (\\sin -> s i n).
5. Re-brace consistently: every command / ^ / _ argument gets exactly one pair of braces,
   and braces that are not arguments (which TeX does not draw) are removed.

Metrics per expression (the first three are the standard CROHME "ExpRate" measures):
- exact      normalized token sequences are identical
- within1/2  at most 1 / 2 token edits away
- ter        token error rate = edit distance / number of correct tokens
- similarity 1 - edit distance / length of the longer sequence  (1.0 = perfect)
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\\[a-zA-Z]+\*?|\\.|\S")

DROP = {
    r"\,", r"\;", r"\:", r"\!", r"\ ", "~", r"\quad", r"\qquad", r"\enspace", r"\thinspace",
    r"\displaystyle", r"\textstyle", r"\scriptstyle", r"\limits", r"\nolimits",
    r"\big", r"\Big", r"\bigg", r"\Bigg", r"\bigl", r"\Bigl", r"\biggl", r"\Biggl",
    r"\bigr", r"\Bigr", r"\biggr", r"\Biggr", r"\bigm", r"\Bigm",
    r"\rm", r"\it", r"\bf", r"\cal",
}
SIZE_DELIMS = {r"\left", r"\right", r"\middle"}

SYNONYMS = {
    r"\dfrac": r"\frac", r"\tfrac": r"\frac", r"\cfrac": r"\frac",
    r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq", r"\leqslant": r"\leq", r"\geqslant": r"\geq",
    r"\to": r"\rightarrow", r"\gets": r"\leftarrow", r"\implies": r"\Rightarrow",
    r"\lbrace": r"\{", r"\rbrace": r"\}", r"\lbrack": "[", r"\rbrack": "]",
    r"\vert": "|", r"\lvert": "|", r"\rvert": "|", r"\mid": "|", r"\Vert": r"\|", r"\lVert": r"\|", r"\rVert": r"\|",
    r"\lt": "<", r"\gt": ">", r"\ast": "*", r"\cdotp": r"\cdot", r"\colon": ":",
    r"\land": r"\wedge", r"\lor": r"\vee", r"\lnot": r"\neg",
    r"\dots": r"\ldots", r"\dotsc": r"\ldots", r"\dotsb": r"\cdots",
    r"\prime": "'", r"\owns": r"\ni", r"\dag": r"\dagger", r"\bot": r"\perp",
    # Handwritten forms of these are indistinguishable, so do not penalize either spelling.
    r"\varepsilon": r"\epsilon", r"\varphi": r"\phi", r"\vartheta": r"\theta",
    r"\overline": r"\bar",  # a hand-drawn bar over a symbol is the same either way
}

# TexTeller sometimes wraps a single line in a one-column array, which draws identically.
_ONE_LINE_ARRAY = re.compile(r"^\\begin\{array\}\s*\{[lcr]\}(.*)\\end\{array\}$", re.S)

# Wrappers that change the font but not what is written: keep only their argument.
UNWRAP = {
    r"\mathrm", r"\mathit", r"\mathbf", r"\mathsf", r"\mathtt", r"\boldsymbol", r"\bm",
    r"\text", r"\textrm", r"\textit", r"\textbf", r"\textsf", r"\mbox", r"\operatorname",
    r"\operatorname*", r"\mathnormal", r"\pmb",
}

# Named operators are written by hand as plain letters, so compare them letter by letter.
NAMED_OPS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh",
    "tanh", "coth", "log", "ln", "lg", "exp", "lim", "liminf", "limsup", "max", "min", "sup",
    "inf", "det", "dim", "ker", "deg", "gcd", "arg", "Pr", "hom", "mod", "bmod",
}

# Commands that take brace arguments, and how many.
ARITY = {
    r"\frac": 2, r"\binom": 2, r"\dbinom": 2, r"\tbinom": 2, r"\overset": 2, r"\underset": 2,
    r"\stackrel": 2, r"\sqrt": 1, r"\overline": 1, r"\underline": 1, r"\hat": 1, r"\widehat": 1,
    r"\bar": 1, r"\vec": 1, r"\dot": 1, r"\ddot": 1, r"\tilde": 1, r"\widetilde": 1,
    r"\overrightarrow": 1, r"\overleftarrow": 1, r"\overbrace": 1, r"\underbrace": 1,
    r"\mathbb": 1, r"\mathcal": 1, r"\mathfrak": 1, r"\mathscr": 1, r"\check": 1, r"\breve": 1,
    r"\acute": 1, r"\grave": 1, r"\boxed": 1, r"\begin": 1, r"\end": 1, r"\pmod": 1,
}


def tokenize(latex: str) -> list[str]:
    return _TOKEN_RE.findall(latex or "")


def _strip_delimiters(latex: str) -> str:
    s = (latex or "").strip()
    for left, right in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if s.startswith(left) and s.endswith(right) and len(s) >= len(left) + len(right):
            return s[len(left) : len(s) - len(right)].strip()
    return s


def _clean_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in SIZE_DELIMS:
            # \left. and \right. are invisible placeholders: drop both tokens.
            if i + 1 < len(tokens) and tokens[i + 1] == ".":
                i += 1
        elif t in DROP:
            pass
        elif t == r"\hspace" or t == r"\vspace":
            i = _skip_group(tokens, i + 1) - 1
        elif t.startswith("\\") and t[1:].rstrip("*") in NAMED_OPS:
            out.extend(t[1:].rstrip("*"))
        else:
            out.append(SYNONYMS.get(t, t))
        i += 1
    return out


def _skip_group(tokens: list[str], i: int) -> int:
    """Index just past the {...} group starting at i (or past one token)."""
    if i >= len(tokens):
        return i
    if tokens[i] != "{":
        return i + 1
    depth = 0
    while i < len(tokens):
        depth += {"{": 1, "}": -1}.get(tokens[i], 0)
        i += 1
        if depth == 0:
            break
    return i


def _parse(tokens: list[str], i: int, stop: str | None) -> tuple[list, int]:
    """Parse a sequence into a tree. Groups become lists; arguments are tagged tuples."""
    seq: list = []
    while i < len(tokens):
        t = tokens[i]
        if t == stop:
            return seq, i + 1
        if t == "}":  # stray closing brace: ignore it
            i += 1
            continue
        if t == "{":
            group, i = _parse(tokens, i + 1, "}")
            seq.append(group)  # bare group: will be flattened
            continue
        i += 1
        if t in UNWRAP:
            arg, i = _read_arg(tokens, i)
            seq.append(arg)  # flattened like a bare group
        elif t in ("^", "_"):
            arg, i = _read_arg(tokens, i)
            seq.append(("arg", t, [arg]))
        elif t in ARITY:
            args = []
            if t == r"\sqrt" and i < len(tokens) and tokens[i] == "[":
                opt, i = _parse(tokens, i + 1, "]")
                args.append(("opt", opt))
            for _ in range(ARITY[t]):
                arg, i = _read_arg(tokens, i)
                args.append(arg)
            seq.append(("arg", t, args))
        else:
            seq.append(t)
    return seq, i


def _read_arg(tokens: list[str], i: int) -> tuple[list, int]:
    while i < len(tokens) and tokens[i] in DROP:
        i += 1
    if i >= len(tokens):
        return [], i
    if tokens[i] == "{":
        return _parse(tokens, i + 1, "}")
    # An unbraced argument may itself be a command with arguments (x^\frac12).
    if tokens[i] in ARITY or tokens[i] in UNWRAP:
        node, j = _parse_one(tokens, i)
        return [node], j
    return [tokens[i]], i + 1


def _parse_one(tokens: list[str], i: int) -> tuple:
    """Parse exactly one element (a command with its arguments) starting at i."""
    t = tokens[i]
    i += 1
    if t in UNWRAP:
        arg, i = _read_arg(tokens, i)
        return arg, i
    args = []
    if t == r"\sqrt" and i < len(tokens) and tokens[i] == "[":
        opt, i = _parse(tokens, i + 1, "]")
        args.append(("opt", opt))
    for _ in range(ARITY[t]):
        arg, i = _read_arg(tokens, i)
        args.append(arg)
    return ("arg", t, args), i


def _emit(node, out: list[str]) -> None:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for n in node:
            _emit(n, out)
    elif node[0] == "opt":
        out.append("[")
        _emit(node[1], out)
        out.append("]")
    else:  # ("arg", command, [args])
        out.append(node[1])
        for a in node[2]:
            if isinstance(a, tuple) and a[0] == "opt":
                _emit(a, out)
            else:
                out.append("{")
                _emit(a, out)
                out.append("}")


def normalize(latex: str) -> list[str]:
    """Canonical token list for a LaTeX string (see module docstring)."""
    s = _strip_delimiters(latex)
    m = _ONE_LINE_ARRAY.match(s)
    if m and r"\\" not in m.group(1):
        s = m.group(1)
    tokens = _clean_tokens(tokenize(s))
    tree, _ = _parse(tokens, 0, None)
    out: list[str] = []
    _emit(tree, out)
    # f^{'} and f' render the same.
    joined = " ".join(out)
    joined = re.sub(r"\^ \{ ((?:' )+)\}", lambda m: m.group(1).strip() + " ", joined)
    return joined.split()


def normalized_string(latex: str) -> str:
    return " ".join(normalize(latex))


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two token lists."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(prediction: str, truth: str) -> dict:
    p, t = normalize(prediction), normalize(truth)
    ed = edit_distance(p, t)
    return {
        "exact": ed == 0,
        "within1": ed <= 1,
        "within2": ed <= 2,
        "edit_distance": ed,
        "ter": ed / max(len(t), 1),
        "similarity": 1 - ed / max(len(p), len(t), 1),
    }
