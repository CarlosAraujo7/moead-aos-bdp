"""make_latex.py -- gera a versao LaTeX (classe elsarticle, template do EJOR fornecido pelos autores) a partir do manuscrito montado em Markdown.

Entrada : Claude outputs/manuscript_draft_v2.md  (montado por make_manuscript_tables.py; as tabelas ja vem dos JSON de results/analysis)
          Claude outputs/supplementary_material.md (idem)
          Claude outputs/submission_materials_draft.md (destaques/highlights)
Saida   : latex/main.tex, latex/sections/*.tex, latex/referencias.bib, latex/figures/*.pdf, latex/supplement.tex
          (classe, estilo bibliografico e .sty vem do template: latex/elsarticle.cls, elsarticle-harv.bst, algorithm.sty, algorithmic.sty)

Nada e digitado a mao nas secoes: reexecutar apos qualquer mudanca no manuscrito. Autores, afiliacoes, conflitos de interesse, financiamento e
agradecimentos seguem o template (main.tex do artigo de corte a laser); CRediT usa os papeis do template (os autores devem conferir); a declaracao de IA
e a de dados sao as do manuscrito.

    python make_latex.py            # gera tudo
    python make_latex.py --build    # gera e compila (pdflatex + bibtex) main e supplement
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
MD = ROOT / "Claude outputs" / "manuscript_draft_v2.md"
SUP = ROOT / "Claude outputs" / "supplementary_material.md"
SUBM = ROOT / "Claude outputs" / "submission_materials_draft.md"
TPL = ROOT / "latex"
LATEX = ROOT / "latex"
FIGS = ROOT / "Claude outputs"

WARN: list[str] = []


# ----------------------------------------------------------------------------- caracteres
UNI = {
    "\u2212": "$-$", "\u2013": "--", "\u2014": "---", "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u00d7": r"$\times$", "\u2192": r"$\rightarrow$",
    "\u03b1": r"$\alpha$", "\u03b5": r"$\varepsilon$", "\u03bb": r"$\lambda$", "\u0394": r"$\Delta$", "\u03c4": r"$\tau$", "\u03a3": r"$\Sigma$",
    "\u03bc": r"$\mu$", "\u03bd": r"$\nu$", "\u03c7": r"$\chi$", "\u03c1": r"$\rho$", "\u2208": r"$\in$", "\u2286": r"$\subseteq$",
    "\u2248": r"$\approx$", "\u2217": r"$^{*}$", "\u00b7": r"$\cdot$", "\u00bc": r"\textonequarter{}", "\u00bd": r"\textonehalf{}", "\u2026": r"\ldots{}",
    "\u00c2": r"$\hat{A}$", "\u2308": r"$\lceil$", "\u2309": r"$\rceil$", "\u00b2": r"$^{2}$", "\u00b9": r"$^{1}$", "\u0161": r"\v{s}", "\u0107": r"\'c",
}
SUB = {"\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4", "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9"}
SUP_CH = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
          "\u207a": "+", "\u207b": "-"}


class Raw:
    """Armazena trechos LaTeX prontos, protegidos da codificacao por marcadores \x04n\x05."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def put(self, s: str) -> str:
        self.items.append(s)
        return f"\x04{len(self.items) - 1}\x05"

    def restore(self, s: str) -> str:
        for _ in range(4):        # marcadores podem estar aninhados
            s2 = re.sub(r"\x04(\d+)\x05", lambda m: self.items[int(m.group(1))], s)
            if s2 == s:
                break
            s = s2
        return s


def strip_acc(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# ----------------------------------------------------------------------------- bibliografia
class Ref:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.authors: list[tuple[str, str]] = []
        self.year = ""
        self.letter = ""
        self.key = ""
        self.bib = ""


def parse_refs(md_refs: list[str]) -> list[Ref]:
    refs = []
    for line in md_refs:
        r = Ref(line)
        m = re.match(r"(.+?)\s\((\d{4})([a-z]?)\)\.\s(.*)$", line)
        if not m:
            WARN.append(f"referencia nao analisada: {line[:80]}")
            continue
        auth, r.year, r.letter, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        auth = auth.replace(", &", ",").replace(" & ", ", ")
        auth = re.sub(r"\(Eds?\.\)", "", auth).strip().rstrip(",")
        pairs = re.findall(r"([^,]+?),\s*((?:[A-ZÁÀ-Ý]\.(?:-?[A-Z]\.)?\s?)+)", auth + ",")
        pairs = re.findall(r"\s*([^,]+?),\s*((?:[A-ZÁÀ-Ý][a-z]?\.\s?(?:-[A-Z]\.\s?)?)+)", auth)
        r.authors = [(a.strip(), b.strip()) for a, b in pairs]
        if not r.authors:
            WARN.append(f"autores nao analisados: {line[:80]}")
            continue
        r.key = re.sub(r"[^a-z]", "", strip_acc(r.authors[0][0]).lower()) + r.year + r.letter
        r.bib = make_bib_entry(r, rest)
        refs.append(r)
    return refs


def bib_escape(s: str) -> str:
    s = s.replace("&", r"\&").replace("%", r"\%")
    return s


_ACC = {chr(0x301): "'", chr(0x300): "`", chr(0x302): "^", chr(0x303): "~", chr(0x308): '"', chr(0x327): "c ", chr(0x30C): "v ", chr(0x304): "=", chr(0x307): "."}
_SPECIAL_BIB = {chr(0xF8): r"{\o}", chr(0xD8): r"{\O}", chr(0x142): r"{\l}", chr(0x141): r"{\L}", chr(0xDF): r"{\ss}"}


def bib_ascii(t: str) -> str:
    """Letras com acento -> comandos LaTeX (o BibTeX corta caracteres UTF-8 de varios bytes ao abreviar iniciais)."""
    out = []
    for ch in t:
        if ord(ch) < 128:
            out.append(ch)
            continue
        if ch == chr(0x2013):
            out.append("--")
            continue
        if ch == chr(0x2014):
            out.append("---")
            continue
        d = unicodedata.normalize("NFD", ch)
        if len(d) == 2 and d[1] in _ACC:
            acc = _ACC[d[1]]
            out.append("{" + chr(92) + acc + ("{" + d[0] + "}" if acc[-1] != " " else d[0]) + "}")
        elif ch in _SPECIAL_BIB:
            out.append(_SPECIAL_BIB[ch])
        else:
            out.append(ch)
            WARN.append(f"caractere nao tratado na bib: {ch!r}")
    return "".join(out)


def make_bib_entry(r: Ref, rest: str) -> str:
    doi = ""
    m = re.search(r"\s*doi:(\S+)\s*$", rest)
    if m:
        doi = m.group(1)
        rest = rest[:m.start()].rstrip()
    url = ""
    mu = re.search(r"(https?://\S+)", rest)
    if mu and not doi:
        url = mu.group(1).rstrip(".")
    authors = " and ".join(f"{a}, {b}" for a, b in r.authors)
    fields = {"author": authors, "year": r.year}
    typ = "misc"
    # artigo: Title. *Journal*, vol(issue), pages.
    m = re.match(r"(.+?)\.\s\*([^*]+)\*,\s*(\d+)(?:\(([^)]+)\))?,\s*([^.]+?)\.?$", rest)
    m_in = re.match(r"(.+?)\.\sIn\s\*([^*]+)\*(?:,\s*([^()]*?))?\s*(?:\(pp\.\s*([^)]+)\))?\.\s*(.*)$", rest)
    m_book = re.match(r"\*([^*]+)\*\.\s*(.*)$", rest)
    m_rep = re.match(r"\*([^*]+)\*\s*\(([^)]+)\)\.\s*(.*)$", rest)
    m_art2 = re.match(r"(.+?)\.\s\*([^*]+)\*,\s*(\d+)(?:\(([^)]+)\))?,\s*(?:paper\s+)?(\w+)\.?$", rest)
    if m_rep:
        typ = "techreport"
        fields.update(title=m_rep.group(1), type=m_rep.group(2), institution=m_rep.group(3).rstrip("."))
    elif m_book and not m:
        typ = "book"
        fields.update(title=m_book.group(1), publisher=m_book.group(2).rstrip("."))
    elif m_in:
        typ = "inproceedings"
        fields.update(title=m_in.group(1), booktitle=m_in.group(2))
        if m_in.group(3):
            fields["series"] = m_in.group(3).strip(" ,")
        if m_in.group(4):
            fields["pages"] = m_in.group(4).replace("–", "--")
        if m_in.group(5):
            fields["publisher"] = m_in.group(5).rstrip(".")
    elif m or m_art2:
        mm = m or m_art2
        typ = "article"
        fields.update(title=mm.group(1), journal=mm.group(2), volume=mm.group(3))
        if mm.group(4):
            fields["number"] = mm.group(4).replace("–", "--")
        pg = mm.group(5).replace("–", "--").strip()
        fields["pages" if re.search(r"--|-", pg) else "article-number"] = pg
        if fields.get("article-number"):
            fields["pages"] = fields.pop("article-number")
    else:
        mt = re.match(r"(.+?)\.\s*(.*)$", rest)
        fields["title"] = mt.group(1) if mt else rest
        if mt and mt.group(2):
            fields["note"] = mt.group(2)
        if not (url or doi):
            WARN.append(f"referencia como @misc: {r.key}: {rest[:70]}")
    if doi:
        fields["doi"] = doi
    if url:
        fields["url"] = url
    if "title" in fields:   # preserva maiusculas de siglas e nomes proprios (Pareto, MOEA/D, ...)
        fields["title"] = re.sub(r"\b([A-Z][A-Za-z/\-]*[A-Z][A-Za-z/\-]*|MOEA/D)\b", r"{\1}", fields["title"])
    body = ",\n".join(f"  {k:<10}= {{{bib_escape(v) if k not in ('doi', 'url') else v}}}" for k, v in fields.items())
    return bib_ascii(f"@{typ}{{{r.key},\n{body}\n}}\n")


def cite_tables(refs: list[Ref]):
    """Formas de citacao no texto: (rotulo_autores, ano+letra) -> chave."""
    forms = {}
    for r in refs:
        sur = [a for a, _ in r.authors]
        n = len(sur)
        label = sur[0] if n == 1 else (f"{sur[0]} and {sur[1]}" if n == 2 else f"{sur[0]} et al.")
        forms[(label, r.year + r.letter)] = r.key
        if n >= 3:   # tolerancia: a lista completa de tres autores
            forms[(f"{sur[0]}, {sur[1]} and {sur[2]}", r.year + r.letter)] = r.key
    return forms


def convert_citations(text: str, forms: dict, raw: Raw) -> str:
    labels = sorted({l for l, _ in forms}, key=len, reverse=True)
    lab_re = "|".join(re.escape(l) for l in labels)

    # 1) agrupadas entre parenteses: (Label, 2018; Label2, 2019a; ...)
    def group(m):
        inner = m.group(1)
        parts = [p.strip() for p in inner.split(";")]
        keys, pre = [], ""
        for i, p in enumerate(parts):
            mm = re.match(r"^((?:e\.g\.,?|see|cf\.)\s+)?(" + lab_re + r"),?\s+(\d{4}[a-z]?(?:,\s*\d{4}[a-z]?)*)$", p)
            if not mm:
                return m.group(0)
            if mm.group(1) and i == 0:
                pre = mm.group(1).strip()
            for y in re.split(r",\s*", mm.group(3)):
                k = forms.get((mm.group(2), y))
                if not k:
                    return m.group(0)
                keys.append(k)
        pre_opt = f"[{pre}][]" if pre else ""
        return raw.put(rf"\citep{pre_opt}{{{','.join(keys)}}}")

    text = re.sub(r"\(([^()]*\d{4}[a-z]?[^()]*)\)", group, text)

    # 2) narrativas: Label (2018) ou Label (2018a, 2018b)
    def narr(m):
        ys = [y.strip() for y in m.group(2).split(",")]
        keys = [forms.get((m.group(1), y)) for y in ys]
        if None in keys:
            return m.group(0)
        return raw.put(rf"\citet{{{','.join(keys)}}}")

    text = re.sub(r"(" + lab_re + r")\s+\((\d{4}[a-z]?(?:,\s*\d{4}[a-z]?)*)\)", narr, text)
    return text


# ----------------------------------------------------------------------------- inline
SPECIAL = [
    (r"HV_norm", r"$\mathrm{HV}_{\mathrm{norm}}$"),
    (r"Â12", r"$\hat{A}_{12}$"),
    (r"z\^min_k", r"$z^{\min}_k$"),
    (r"z∗_k", r"$z^{*}_k$"),
    (r"λ\^i", r"$\boldsymbol{\lambda}^{i}$"),
    (r"z∗_min", r"$z^{*}_{\min}$"),
    (r"z∗", r"$z^{*}$"),
    (r"z_min", r"$z_{\min}$"),
    (r"IGD\+", r"IGD$^{+}$"),
    (r"ε\+", r"$\varepsilon^{+}$"),
    (r"eps\+", r"$\varepsilon^{+}$"),
]


MATHID = re.compile(r"(?<![\w$])(d|c|k|g|p|f|z|λ|μ|ν|ΔHV|K)(∗)?_(\{[^}]*\}|[A-Za-z0-9]+)")
_GREEK = {"λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ΔHV": r"\Delta\mathrm{HV}"}


def mathid(m):
    base = _GREEK.get(m.group(1), m.group(1))
    sub = m.group(3).strip("{}")
    sub_tex = r"\mathrm{" + sub + "}" if re.fullmatch(r"[A-Za-z]{3,}", sub) else sub
    star = "^{*}" if m.group(2) else ""
    return RAW_HOLDER[0].put(f"${base}{star}_{{{sub_tex}}}$")


RAW_HOLDER = [None]


def inline(s: str, raw: Raw, forms: dict | None = None, table: bool = False) -> str:
    RAW_HOLDER[0] = raw
    # codigo
    def code(m):
        c = m.group(1)
        mt = re.match(r"^\[(TODO|PENDING)[:\s]*(.*)\]$", c)
        if mt:
            return raw.put(r"\todo{" + tex_escape(mt.group(2), raw) + "}")
        if c.startswith("[") and c.endswith("]"):          # [generic text: ...] e semelhantes: viram aviso visivel em vermelho
            return raw.put(r"\todo{" + tex_escape(c.strip("[]"), raw) + "}")
        e = tex_escape(c, raw).replace(r"\_", r"\_\allowbreak{}")
        return raw.put(r"\texttt{" + e + "}")

    s = re.sub(r"`([^`]+)`", code, s)
    # links
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", lambda m: raw.put(r"\href{" + m.group(2) + "}{" + tex_escape(m.group(1), raw) + "}"), s)
    s = re.sub(r"(?<![{(])(https?://[^\s)]+[^\s).,;])", lambda m: raw.put(r"\url{" + m.group(1) + "}"), s)
    if forms:
        s = convert_citations(s, forms, raw)
    # negrito e italico
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: "\x01B" + m.group(1) + "\x02", s)
    if not table:
        s = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", lambda m: "\x01I" + m.group(1) + "\x02", s)
    else:      # celula inteira em italico (*texto*): ex. linhas de referencia das tabelas; asteriscos soltos continuam sendo marcas
        s = re.sub(r"^\*([^*]+)\*$", lambda m: "\x01I" + m.group(1) + "\x02", s.strip())
    # aspas
    s = re.sub(r'"([^"]*)"', lambda m: "``" + m.group(1) + "''", s)
    for a, b in SPECIAL:
        s = re.sub(a, lambda m, b=b: raw.put(b), s)
    s = re.sub(r"Σ_\{([^}]*)\}", lambda m: raw.put(r"$\sum_{" + m.group(1).replace("∈", r"\in ").replace(";", r";\,") + "}$"), s)
    s = re.sub(MATHID, mathid, s) if not table else s
    s = tex_escape(s, raw)
    if table:
        s = s.replace("*", r"$^{*}$")
    s = s.replace("\x01B", r"\textbf{").replace("\x01I", r"\emph{").replace("\x02", "}")
    return raw.restore(s)


def tex_escape(s: str, raw: Raw) -> str:
    # sub/sobrescritos unicode
    s = re.sub("[" + "".join(SUB) + "]+", lambda m: raw.put("$_{" + "".join(SUB[c] for c in m.group(0)) + "}$"), s)
    s = re.sub("[" + "".join(k for k in SUP_CH if k not in ("\u00b9", "\u00b2")) + "]+",
               lambda m: raw.put("$^{" + "".join(SUP_CH[c] for c in m.group(0)) + "}$"), s)
    out = []
    for ch in s:
        if ch in "&%$#{}":
            out.append("\\" + ch)
        elif ch == "_":
            out.append(r"\_")
        elif ch == "^":
            out.append(r"\^{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in UNI:
            out.append(raw.put(UNI[ch]))
        else:
            out.append(ch)
    return "".join(out)


# ----------------------------------------------------------------------------- blocos
EQ = {
    "f₁(S) = Σ_{i<j; i,j∈S} d_ij  (MaxSum),  f₂(S) = min_{i<j; i,j∈S} d_ij  (MaxMin).":
        r"f_1(S) = \sum_{\substack{i<j\\ i,j\in S}} d_{ij}\;\; (\text{MaxSum}), \qquad f_2(S) = \min_{\substack{i<j\\ i,j\in S}} d_{ij}\;\; (\text{MaxMin}).",
    "g(S | λ, z∗) = max_k λ_k (z∗_k − f_k(S)),":
        r"g(S \mid \boldsymbol{\lambda}, z^{*}) = \max_{k}\; \lambda_k \bigl(z^{*}_k - f_k(S)\bigr),",
}

ALGO1 = r"""\begin{algorithm}[t]
\caption{MOEA/D with a pool of operators and a selector}
\label{alg:moead}
\begin{algorithmic}
\REQUIRE instance, $N$ weight vectors, neighbourhood size $T$, $G$ generations, pool of 8 operators, selector $\sigma$
\STATE for $i=1,\dots,N$: build $S_i$ by weight-guided GRASP and evaluate it
\STATE $A \leftarrow$ non-dominated set of $\{S_i\}$; $z^{*}\leftarrow$ component-wise maximum and $z_{\min}\leftarrow$ component-wise minimum of the objectives
\FOR{$g=1,\dots,G$}
\FOR{$i$ in a random permutation of $1,\dots,N$}
\STATE $k \leftarrow \sigma.\mathrm{select}()$ \COMMENT{uniform, UCB1, Thompson sampling or Gaussian sampler}
\STATE $j \leftarrow$ a random member of the neighbourhood of $i$ (probability 0.9) or of the population (0.1)
\STATE $S' \leftarrow \mathrm{operator}_k(S_j)$ (SubsetCrossover also takes a second parent from the neighbourhood); $S' \leftarrow \mathrm{repair}(S')$ if infeasible
\STATE $r \leftarrow$ reward of $S'$ (HV gain with respect to $A$, or Tchebycheff improvement over $S_j$); $\sigma.\mathrm{update}(k, r)$
\FORALL{$l$ in the neighbourhood of $i$}
\IF{$g(S' \mid \lambda_l, z^{*}) \leq g(S_l \mid \lambda_l, z^{*})$}
\STATE $S_l \leftarrow S'$
\ENDIF
\ENDFOR
\STATE update $z^{*}$ and $z_{\min}$; $A \leftarrow$ non-dominated$(A \cup \{S'\})$
\ENDFOR
\ENDFOR
\STATE optional: add the MaxMin anchor solution(s) of Section~\ref{sec:anchor} to $A$
\STATE return $A$
\end{algorithmic}
\end{algorithm}
"""


def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", strip_acc(t).lower()).strip("_")[:28]


def parse_table(rows: list[str]):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for i, r in enumerate(cells) if not (i == 1 and all(re.fullmatch(r":?-{2,}:?", c) for c in r))]
    return cells


def table_tex(cells: list[list[str]], caption: str, label: str, raw: Raw, forms) -> str:
    nc = max(len(r) for r in cells)
    rows = [r + [""] * (nc - len(r)) for r in cells]
    maxlen = [max(len(re.sub(r"[*`]", "", r[j])) for r in rows) for j in range(nc)]
    longcols = [j for j in range(nc) if maxlen[j] > 34 and j > 0]
    total = sum(min(m, 40) for m in maxlen)
    size = r"\footnotesize" if (nc >= 7 or total > 95) else r"\small"
    if nc >= 9 or total > 140:
        size = r"\scriptsize"
    spec = "l" + "".join(("X" if j in longcols else "c") for j in range(1, nc))
    if longcols:
        env_b, env_e = r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\end{tabularx}"
    else:
        env_b, env_e = r"\begin{tabular}{" + spec + "}", r"\end{tabular}"
    if caption:
        out = [r"\begin{table}[t]", r"\centering", size, r"\setlength{\tabcolsep}{4pt}", r"\caption{" + inline(caption, raw, forms) + r"}\label{" + label + "}"]
    else:        # tabela sem legenda (ex.: orcamentos do Apendice A): fica no fluxo do texto, sem flutuante nem numero
        out = [r"\begin{center}", size, r"\setlength{\tabcolsep}{4pt}"]
    body = [env_b, r"\toprule"]
    for i, r in enumerate(rows):
        body.append(" & ".join(inline(c, raw, forms, table=True) for c in r) + r" \\")
        if i == 0:
            body.append(r"\midrule")
    body += [r"\bottomrule", env_e]
    if not longcols and (nc >= 9 or total > 140):
        out += [r"\resizebox{\linewidth}{!}{%"] + body + ["}"]
    else:
        out += body
    out.append(r"\end{table}" if caption else r"\end{center}")
    return "\n".join(out)


def figure_tex(alt: str, path: str, label: str, raw: Raw, forms) -> str:
    cap = re.sub(r"^Figure\s+\d+\.\s*", "", alt)
    src = FIGS / path
    pdf = src.with_suffix(".pdf")
    use = pdf if pdf.exists() else src
    (LATEX / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(use, LATEX / "figures" / use.name)
    return "\n".join([r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\linewidth]{figures/" + use.name + "}",
                      r"\caption{" + inline(cap, raw, forms) + r"}\label{" + label + "}", r"\end{figure}"])


def blocks_to_tex(lines: list[str], raw: Raw, forms, ctx: dict) -> list[str]:
    out: list[str] = []
    para: list[str] = []
    i = 0
    pending_cap: str | None = None
    after_eq = False

    def flush():
        nonlocal para, after_eq
        if para:
            txt = " ".join(p.strip() for p in para).strip()
            if txt and not txt.startswith("*Working title") and not txt.startswith("*Supplement to"):
                out.append(("\\noindent " if after_eq else "") + inline(txt, raw, forms))
                after_eq = False
            para = []

    while i < len(lines):
        ln = lines[i]
        s = ln.rstrip()
        if not s.strip():
            flush()
            i += 1
            continue
        if s.startswith("```"):
            flush()
            j = i + 1
            block = []
            while j < len(lines) and not lines[j].startswith("```"):
                block.append(lines[j])
                j += 1
            if block and block[0].startswith("Algorithm 1"):
                out.append(ALGO1)
            else:
                out.append(r"\begin{small}\begin{verbatim}" + "\n" + "\n".join(block) + "\n" + r"\end{verbatim}\end{small}")
            i = j + 1
            continue
        if s.startswith("> "):                         # citacao: so a nota de estado (para os autores); nao vai ao PDF
            flush()
            while i < len(lines) and lines[i].startswith(">"):
                i += 1
            continue
        if s.startswith("###"):
            flush()
            m = re.match(r"(#{3,4})\s+(?:[\dA-Z]+(?:\.\d+)*\s+)?(.*)$", s)
            title = m.group(2)
            cmd = r"\subsection" if m.group(1) == "###" else r"\subsubsection"
            lab = f"sec:{slug(title)}"
            for pre_, l_ in (("MaxMin anchoring", "sec:anchor"), ("Covering the ends", "sec:anchor_lit")):
                if title.startswith(pre_):
                    lab = l_
            out.append(f"{cmd}{{{inline(title, raw, forms)}}}\\label{{{lab}}}")
            ctx["labels"][title] = lab
            i += 1
            continue
        if re.match(r"^\*\*Table \d+\.", s) and s.endswith("**"):
            flush()
            pending_cap = s[2:-2]
            pending_cap = re.sub(r"^Table\s+\d+\.\s*", "", pending_cap)
            i += 1
            continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            ctx["ntab"] += 1
            lab = f"tab:{ctx['prefix']}{ctx['ntab']}"
            out.append(table_tex(parse_table(rows), pending_cap or "", lab, raw, forms))
            pending_cap = None
            continue
        if s.startswith("!["):
            flush()
            m = re.match(r"!\[(.*)\]\(([^)]+)\)\s*$", s)
            ctx["nfig"] += 1
            out.append(figure_tex(m.group(1), m.group(2), f"fig:{ctx['prefix']}{ctx['nfig']}", raw, forms))
            i += 1
            continue
        if re.match(r"^\s*[-*]\s+", s) and not s.startswith("**"):
            flush()
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i]).rstrip()
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and not re.match(r"^\s*[-*]\s+", lines[i]):
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item)
            out.append(r"\begin{itemize}" + "\n" + "\n".join(r"\item " + inline(x, raw, forms) for x in items) + "\n" + r"\end{itemize}")
            continue
        if re.match(r"^\d+\.\s+", s):
            flush()
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i]):
                item = re.sub(r"^\d+\.\s+", "", lines[i]).rstrip()
                i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item)
            out.append(r"\begin{enumerate}" + "\n" + "\n".join(r"\item " + inline(x, raw, forms) for x in items) + "\n" + r"\end{enumerate}")
            continue
        if ln.startswith(" ") and not ln.startswith("  ") and ("=" in ln or "∈" in ln):
            flush()
            key = ln.strip()
            if key in EQ:
                out.append("\\begin{equation*}\n" + EQ[key] + "\n\\end{equation*}")
                after_eq = True
            else:
                WARN.append(f"equacao desconhecida: {key[:70]}")
                out.append(r"\begin{center}\texttt{" + tex_escape(key, raw) + "}\\end{center}")
            i += 1
            continue
        if s.strip() == "---":
            i += 1
            continue
        para.append(s)
        i += 1
    flush()
    return out


# ----------------------------------------------------------------------------- principal
def split_sections(md: str):
    pre, secs, cur = [], [], None
    for ln in md.splitlines():
        if ln.startswith("## "):
            cur = [ln[3:].strip(), []]
            secs.append(cur)
        elif cur is None:
            pre.append(ln)
        else:
            cur[1].append(ln)
    return pre, secs


def highlights() -> list[str]:
    t = SUBM.read_text(encoding="utf-8").split("## Highlights")[1].split("## Graphical")[0]
    return [re.sub(r"\s*\(\d+\)\s*$", "", m.group(1)).strip() for m in re.finditer(r"^\d+\.\s+(.*)$", t, flags=re.M)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    LATEX.mkdir(exist_ok=True)
    (LATEX / "sections").mkdir(exist_ok=True)
    md = MD.read_text(encoding="utf-8")
    pre, secs = split_sections(md)
    title = re.sub(r"^#\s+", "", pre[0]).strip()
    names = {s[0]: s[1] for s in secs}

    # --- bibliografia
    ref_lines = [re.sub(r"^-\s+", "", l) for l in names["References"] if l.startswith("- ")]
    refs = parse_refs(ref_lines)
    forms = cite_tables(refs)
    bib = ["% referencias.bib -- gerado por make_latex.py a partir da lista de referencias do manuscrito (conferir/editar a vontade)\n"]
    bib += [r.bib for r in refs]
    extra = TPL.parent  # nada
    (LATEX / "referencias.bib").write_text("\n".join(bib), encoding="utf-8", newline="\n")

    raw = Raw()
    ctx = {"ntab": 0, "nfig": 0, "labels": {}, "prefix": ""}
    body_files = []
    appendix = False
    for name, lines in secs:
        if name in ("Abstract", "References") or name.startswith(("CRediT", "Declaration", "Funding", "Data availability", "Acknowledgements")):
            continue
        m = re.match(r"^(?:(\d+)\.|Appendix ([A-Z])\.)\s+(.*?)(?:\s+`.*`)?$", name)
        if not m:
            WARN.append(f"secao ignorada: {name}")
            continue
        sec_title = m.group(3)
        tex = []
        if m.group(2) and not appendix:
            tex.append(r"\appendix")
            appendix = True
        lab = "sec:" + slug(sec_title)
        tex.append(r"\section{" + inline(sec_title, raw, forms) + "}\\label{" + lab + "}")
        tex += blocks_to_tex(lines, raw, forms, ctx)
        fn = f"{'app_' if m.group(2) else ''}{slug(sec_title)}.tex"
        text = raw.restore("\n\n".join(tex)) + "\n"
        (LATEX / "sections" / fn).write_text(text, encoding="utf-8", newline="\n")
        body_files.append(fn[:-4])

    # --- resumo e palavras-chave
    abs_txt = " ".join(l.strip() for l in names["Abstract"] if l.strip() and not l.startswith("**Keywords"))
    kw = next(l for l in names["Abstract"] if l.startswith("**Keywords:**"))
    kws = [k.strip() for k in kw.replace("**Keywords:**", "").split(";")]
    abstract_tex = raw.restore(inline(abs_txt, raw, forms))
    nwords = len(re.findall(r"\S+", re.sub(r"\([^)]*\d{4}[^)]*\)", "", abs_txt)))

    # --- declaracoes
    def sec_text(prefix):
        for n, l in secs:
            if n.startswith(prefix):
                return " ".join(x.strip() for x in l if x.strip())
        return ""

    ai_txt = raw.restore(inline(sec_text("Declaration of generative AI"), raw, forms))
    hl = highlights()
    main_tex = build_main(title, abstract_tex, kws, hl, body_files, ai_txt)
    (LATEX / "main.tex").write_text(main_tex, encoding="utf-8", newline="\n")

    # --- suplemento
    if SUP.exists():
        build_supplement(title, forms)

    print(f"LaTeX gerado em {LATEX}: {len(body_files)} secoes, {len(refs)} referencias, {ctx['ntab']} tabelas, {ctx['nfig']} figuras; resumo {nwords} palavras")
    for w in WARN:
        print("AVISO:", w)
    if a.build:
        build()


def build_main(title, abstract, kws, hl, body_files, ai_txt) -> str:
    tpl = (TPL / "main_template.tex").read_text(encoding="utf-8")
    inputs = "\n".join(f"\\input{{sections/{f}}}" for f in body_files)
    hl_tex = "\n".join(r"\item " + h.replace("&", r"\&") for h in hl)
    kw_tex = " \\sep\n".join(k for k in kws)
    out = (tpl.replace("%%TITLE%%", title).replace("%%HIGHLIGHTS%%", hl_tex).replace("%%ABSTRACT%%", abstract)
           .replace("%%KEYWORDS%%", kw_tex).replace("%%SECTIONS%%", inputs).replace("%%AIDECL%%", ai_txt))
    return out


def build_supplement(title, forms):
    md = SUP.read_text(encoding="utf-8")
    pre, secs = split_sections(md)
    raw = Raw()
    ctx = {"ntab": 0, "nfig": 0, "labels": {}, "prefix": "s"}
    parts = []
    for name, lines in secs:
        m = re.match(r"^(S\d+)\.\s+(.*)$", name)
        t = m.group(2) if m else name
        parts.append(r"\section{" + inline(t, raw, forms) + "}")
        parts += blocks_to_tex(lines, raw, forms, ctx)
    body = raw.restore("\n\n".join(parts))
    tpl = (TPL / "supplement_template.tex").read_text(encoding="utf-8")
    (LATEX / "supplement.tex").write_text(tpl.replace("%%TITLE%%", title).replace("%%BODY%%", body), encoding="utf-8", newline="\n")


def build():
    for doc in ("main", "supplement"):
        if not (LATEX / f"{doc}.tex").exists():
            continue
        for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):
            c = cmd + (["-interaction=nonstopmode", "-halt-on-error", doc] if cmd[0] == "pdflatex" else [doc])
            r = subprocess.run(c, cwd=str(LATEX), capture_output=True, text=True, errors="replace")
            if r.returncode != 0 and cmd[0] == "pdflatex":
                print(f"[{doc}] {' '.join(c)} falhou:\n" + "\n".join(r.stdout.splitlines()[-25:]))
                break
        print(f"[{doc}] compilacao concluida")


if __name__ == "__main__":
    main()
