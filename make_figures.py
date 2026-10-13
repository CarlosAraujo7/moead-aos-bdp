"""Figuras do manuscrito (matplotlib; PDF vetorial + PNG 300 dpi em `Claude outputs/figures/`). Nada e digitado a mao: tudo vem de
results/analysis/*.json e dos JSONs de runs. Paleta categorica de referencia (azul, laranja, verde-agua, ...) em ordem fixa; os
marcadores/hachuras tambem diferenciam as series (leitura em tons de cinza); grade discreta; sem eixo duplo.

    python make_figures.py [fig1 fig2 fig3 fig4 ...]     # sem argumentos: todas as figuras possiveis

fig1  ganho da normalizacao por instancia vs n (CDP, GDP; MOEA/D-Uniform, replica 0, hv_norm contra a referencia fixa)
fig2  frequencia real de selecao de operadores por modo (CDP n<=150, normalizado, 3 replicas)
fig3  extremo MaxMin / melhor valor publicado, com e sem ancora (CDP; results/analysis/anchor_lit_cdp.json)
fig4  diagramas de diferenca critica (Nemenyi) dos 3 problemas (catalogo completo, hv_norm, 3 replicas)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
OUT = ROOT / "Claude outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
A = ROOT / "results" / "analysis"

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})

LABEL = {"hypervolume": "UCB-HV", "thompson_hv": "TS-HV", "dyts_hv": "GTS-HV", "scalar": "UCB-Scal", "thompson_scalar": "TS-Scal",
         "dyts_scalar": "GTS-Scal", "random": "Uniform", "nsga2": "NSGA-II", "grasp": "GRASP-MO", "grasp_pr": "GRASP-PR",
         "vns_pareto": "VNS-Pareto", "vns_tchebycheff": "VNS-Tcheb."}
OPS = ["RandomSwap1x1", "BestSwap1x1", "RandomSwap2x2", "GraspGreedy", "BiObjBestSwap", "MaxMinSwap", "LargePerturbation", "SubsetCrossover"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("salvo:", OUT / f"{name}.pdf")


FIG1_MED = {}


def fig1():
    import reference_sets as RS
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, prob in zip(axes, ("CDP", "GDP")):
        ref = RS.load(prob)
        hist = ROOT / "results" / "capacitated" / prob / "runs"
        norm = ROOT / "results" / "capacitated_norm" / prob / "runs"
        rows = [r for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == prob]
        data = {}
        for r in rows:
            f0, f1 = hist / f"{r['name']}_random_rep00.json", norm / f"{r['name']}_random_rep00.json"
            if not (f0.exists() and f1.exists()):
                continue
            h0 = RS.indicators(np.array(json.load(open(f0))["pareto_front"]), ref[r["name"]])["hv_norm"]
            h1 = RS.indicators(np.array(json.load(open(f1))["pareto_front"]), ref[r["name"]])["hv_norm"]
            data.setdefault(int(r["n"]), []).append(h1 - h0)
        ns = sorted(data)
        print(prob, "mediana do ganho por n:", {n: round(float(np.median(data[n])), 3) for n in ns}, "| fracao > 0:", {n: round(float(np.mean(np.array(data[n]) > 0)), 2) for n in ns})
        FIG1_MED[prob] = {str(n): float(np.median(data[n])) for n in ns}
        for i, n in enumerate(ns):
            v = np.array(data[n]); x = i + (np.random.default_rng(1).random(len(v)) - 0.5) * 0.45
            ax.scatter(x, v, s=7, color=COL[0], alpha=0.45, linewidths=0)
            q1, md, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i - 0.28, i + 0.28], [md, md], color=INK, lw=1.6)
            ax.plot([i, i], [q1, q3], color=INK, lw=1.0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"n = {n}\n(N = {len(data[n])})" for n in ns])
        ax.set_title(prob, loc="left", fontsize=9, fontweight="bold")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("HV$_{norm}$ gain of normalisation\n(per instance, Uniform, replicate 0)")
    save(fig, "fig1_normalisation_gain_vs_n")
    (A / "fig1_medians.json").write_text(json.dumps(FIG1_MED, indent=1), encoding="utf-8")


def fig2():
    from experiment import ALL_MODES
    base = ROOT / "results" / "capacitated_norm" / "CDP" / "runs"
    names = [r["name"] for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == "CDP" and int(r["n"]) <= 150]
    modes = ["random", "hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar"]
    extra = {"insert reward (UCB)": "ucb_insert", "insert reward (TS)": "ts_insert"}
    p2 = ROOT / "results" / "phase2_aos" / "CDP" / "runs"
    fr, labels = [], []
    for m in modes:
        c = np.zeros(8); k = 0
        for n in names:
            for rep in range(3):
                f = base / f"{n}_{m}_rep{rep:02d}.json"
                if f.exists():
                    cc = np.array(json.load(open(f))["operator_counts"], float)
                    if cc.sum() > 0:
                        c += cc / cc.sum(); k += 1
        if k:
            fr.append(c / k); labels.append(f"{LABEL[m]}")
    for lab, m in extra.items():
        c = np.zeros(8); k = 0
        for n in names:
            for rep in range(3):
                f = p2 / f"{n}_{m}_rep{rep:02d}.json"
                if f.exists():
                    cc = np.array(json.load(open(f))["operator_counts"], float)
                    if cc.sum() > 0:
                        c += cc / cc.sum(); k += 1
        if k:
            fr.append(c / k); labels.append(lab)
    fr = np.array(fr)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    left = np.zeros(len(fr)); y = np.arange(len(fr))[::-1]
    for j in range(8):
        ax.barh(y, fr[:, j], left=left, color=COL[j], edgecolor="white", linewidth=1.2, height=0.7, label=OPS[j])
        left += fr[:, j]
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, 1); ax.set_xlabel("share of operator selections (CDP, n ≤ 150, 60 instances, 3 replicates)")
    ax.axvline(0.125, color=INK2, lw=0.6, ls=":")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), handlelength=1.0, columnspacing=1.0)
    save(fig, "fig2_operator_selection_shares")


def fig3():
    d = json.load(open(A / "anchor_lit_cdp.json"))
    groups = []
    for lab in ("n<=150", "n=500"):
        for g, v in d[lab].items():
            if g.startswith("FAM "):
                ms = list(v)
                groups.append((g[4:], v[ms[0]]["ratio_best"], v[ms[1]]["ratio_best"], v[ms[1]]["reach_best"], v[ms[1]]["N"]))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(groups)); w = 0.36
    b0 = ax.bar(x - w / 2, [g[1] for g in groups], w, color=COL[0], edgecolor="white", linewidth=1.2, label="Uniform, no anchor")
    b1 = ax.bar(x + w / 2, [g[2] for g in groups], w, color=COL[1], edgecolor="white", linewidth=1.2, hatch="////", label="Uniform + MaxMin anchor")
    for xi, g in zip(x, groups):
        ax.text(xi - w / 2, g[1] + 0.012, f"{g[1]:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.text(xi + w / 2, g[2] + 0.012, f"{g[2]:.2f}", ha="center", va="bottom", fontsize=7, color=INK, bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
    ax.axhline(1.0, color=INK, lw=1.0, ls="--")
    ax.text(len(groups) - 0.5, 1.012, "best published value", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"{g[0]}\n(N = {g[4]})" for g in groups], fontsize=7.5)
    ax.set_ylabel("MaxMin of the front's extreme /\nbest published value"); ax.set_ylim(0, 1.15)
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig3_extreme_maxmin_vs_published")


def cd_diagram(ax, ranks: dict, cd: float, title: str, top=None):
    names = sorted(ranks, key=ranks.get)
    k = len(names); lo, hi = 1, 12
    ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([lo, hi], [0.72, 0.72], color=INK, lw=1.0)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0.72, 0.76], color=INK, lw=1.0); ax.text(t, 0.8, str(t), ha="center", va="bottom", fontsize=7.5)
    half = (k + 1) // 2
    for i, nme in enumerate(names):
        r = ranks[nme]; left = i < half
        yy = 0.62 - 0.07 * (i if left else i - half)
        xt = lo - 0.15 if left else hi + 0.15
        ax.plot([r, r], [0.72, yy], color=INK2, lw=0.7); ax.plot([r, xt + (0.1 if left else -0.1)], [yy, yy], color=INK2, lw=0.7)
        col = COL[0] if (top and nme in top) else INK
        ax.text(xt, yy, f"{LABEL[nme]} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=7.5, color=col,
                fontweight="bold" if (top and nme in top) else "normal")
    # barra de CD e cliques (Nemenyi)
    ax.plot([lo, lo + cd], [0.93, 0.93], color=INK, lw=1.6); ax.text(lo + cd / 2, 0.95, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7.5)
    ordered = [ranks[n] for n in names]; ybar = 0.69; seen = []
    for i in range(k):
        j = i
        while j + 1 < k and ordered[j + 1] - ordered[i] < cd:
            j += 1
        if j > i and not any(s0 <= i and j <= s1 for s0, s1 in seen):
            seen.append((i, j)); ax.plot([ordered[i] - 0.03, ordered[j] + 0.03], [ybar, ybar], color=INK, lw=2.4, solid_capstyle="butt"); ybar -= 0.03
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", y=1.0)


def fig4():
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.2))
    for ax, (prob, hi) in zip(axes, (("CDP", 500), ("GDP", 500), ("MDP", 1000))):
        p = A / f"stats_{prob}_norm_n0-{hi}_reps3_hv_norm.json"
        if not p.exists():
            ax.axis("off"); continue
        s = json.load(open(p))
        cd_diagram(ax, s["avg_ranks"], s["cd"], f"{prob} (N = {s['N']}, HV$_{{norm}}$, 3 replicates; bold blue = top group, Wilcoxon–Holm)", s["top_group"])
    fig.subplots_adjust(hspace=0.15)
    save(fig, "fig4_cd_diagrams")


def aos_vs_size():
    """Diferenca de HV_norm entre UCB-HV/GTS-HV e Uniform por grupo de tamanho (media por instancia sobre as 3 replicas; referencia congelada;
    n=2000 do MDP vem do E10 com a referencia propria daquele experimento). Salva results/analysis/aos_vs_size.json."""
    from analyze_replicated import load_matrix
    from experiment import ALL_MODES
    iu, ih, ig = ALL_MODES.index("random"), ALL_MODES.index("hypervolume"), ALL_MODES.index("dyts_hv")
    meta_c = {r["name"]: int(r["n"]) for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv"))}
    meta_m = {r["name"]: int(r["n"]) for r in csv.DictReader(open(ROOT / "instances/catalog.csv"))}
    groups = {}   # problema -> lista de (rotulo, deltas_ucb, deltas_gts)
    for prob, hi in (("MDP", 100000), ("CDP", 500), ("GDP", 500)):
        names, cube = load_matrix(prob, hi, "hv_norm", 3)
        M = cube.mean(axis=2)
        meta = meta_m if prob == "MDP" else meta_c
        ns = np.array([meta[n] for n in names])
        bins = [("n ≤ 50", 0, 50), ("51–200", 51, 200), ("201–500", 201, 500), ("501–1000", 501, 1000)] if prob == "MDP" else                [("n = 50", 50, 50), ("n = 150", 150, 150), ("n = 500", 500, 500)]
        groups[prob] = []
        for lab, lo, hi_ in bins:
            idx = np.where((ns >= lo) & (ns <= hi_))[0]
            if prob == "GDP" and lo == 500:   # n=500 separado pelo fator de custo (k02 = orcamento apertado; k03 = folgado)
                for tag, suf in (("k02", "_k02"), ("k03", "_k03")):
                    ii = np.array([i for i in idx if names[i].endswith(suf)])
                    groups[prob].append(("n = 500"+chr(10)+("tight cost" if tag == "k02" else "loose cost"), M[ii, ih] - M[ii, iu], M[ii, ig] - M[ii, iu]))
            elif len(idx):
                groups[prob].append((lab, M[idx, ih] - M[idx, iu], M[idx, ig] - M[idx, iu]))
    e10 = json.load(open(A / "e10_budget_regime.json"))
    g = e10.get("MDP MDG-b n=2000 (x1)")
    if g:
        hv = g["hv"]["x1"]
        groups["MDP"].append(("n = 2000"+chr(10)+"(MDG-b)", np.array(hv["hypervolume"]) - np.array(hv["random"]), np.array(hv["dyts_hv"]) - np.array(hv["random"])))
    rng = np.random.default_rng(7)

    def ci(x):
        b = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(2000)]
        return float(np.mean(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), sharey=True, gridspec_kw={"width_ratios": [5, 3, 4]})
    summary = {}
    for ax, prob in zip(axes, ("MDP", "CDP", "GDP")):
        for j, (col, mk, name) in enumerate(((COL[0], "o", "UCB-HV"), (COL[1], "s", "GTS-HV"))):
            xs, ms, lo_, hi_ = [], [], [], []
            for i, (lab, du, dg) in enumerate(groups[prob]):
                m, l, h = ci(du if j == 0 else dg)
                xs.append(i + (j - 0.5) * 0.18); ms.append(m); lo_.append(m - l); hi_.append(h - m)
                summary.setdefault(prob, {}).setdefault(lab.replace(chr(10), " "), {})[name] = dict(mean=m, lo=l, hi=h, N=len(du))
            ax.errorbar(xs, ms, yerr=[lo_, hi_], fmt=mk, color=col, ms=4, capsize=2, lw=1.1, label=name)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(groups[prob]))); ax.set_xticklabels([f"{l}"+chr(10)+f"(N={len(d)})" for l, d, _ in groups[prob]], fontsize=6.5)
        ax.set_title(prob, loc="left", fontsize=9, fontweight="bold"); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("HV$_{norm}$(selector) − HV$_{norm}$(Uniform)"+chr(10)+"(mean over instances, 95% bootstrap CI)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig5_adaptive_selection_vs_size")
    (A / "aos_vs_size.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def graphical_abstract():
    """Resumo grafico (EJOR): quatro paineis com os resultados centrais, todos lidos de results/analysis/*.json."""
    med = json.load(open(A / "fig1_medians.json"))
    e11 = json.load(open(A / "e11_strong_baselines.json"))
    avs = json.load(open(A / "aos_vs_size.json"))
    anc = {p: json.load(open(A / f"anchor_v2_{p}.json")) for p in ("CDP", "GDP")}
    fig, axes = plt.subplots(1, 4, figsize=(13.3, 5.3), gridspec_kw={"width_ratios": [1, 1.15, 1.15, 1.15]})
    plt.rcParams.update({"font.size": 10})
    # (a) normalisation
    ax = axes[0]
    for prob, col, mk in (("CDP", COL[0], "o"), ("GDP", COL[1], "s")):
        ns = sorted(med[prob], key=int)
        ax.plot(range(len(ns)), [med[prob][n] for n in ns], marker=mk, color=col, lw=2, ms=7, label=prob)
        for i, n in enumerate(ns):
            ax.annotate(f"{med[prob][n]:+.2f}", (i, med[prob][n]), textcoords="offset points", xytext=((0, 9) if prob == "GDP" else (0, -17)), ha="center", fontsize=9, color=col)
    ax.set_xticks(range(3)); ax.set_xticklabels(["n = 50", "n = 150", "n = 500"]); ax.set_ylim(0, 0.36); ax.set_xlim(-0.35, 2.35)
    ax.set_ylabel("median gain in HV$_{norm}$"); ax.legend(frameon=False, loc="upper left"); ax.grid(axis="x", visible=False)
    ax.set_title("1  Scale the objectives inside\nthe Tchebycheff function", loc="left", fontsize=11, fontweight="bold")
    # (b) tuned baselines vs tuned MOEA/D at equal time
    ax = axes[1]
    cells = ["CDP-S", "CDP-L", "GDP-S", "GDP-L", "MDP-S", "MDP-L"]
    lab = ["CDP n≤150", "CDP n=500", "GDP n≤150", "GDP n=500", "MDP n≤150", "MDP n=500"]
    vals = []
    for c in cells:
        t = e11[c]["at_T_vs_tm"]["methods"]
        vals.append(max(t[m]["diff"] for m in ("t_nsga2", "t_grasp", "t_ts", "t_vns")))
    cols = [COL[0] if v < 0 else COL[1] for v in vals]
    ax.barh(range(6), vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + (0.006 if v >= 0 else -0.006), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color=INK2, lw=0.9); ax.set_yticks(range(6)); ax.set_yticklabels(lab); ax.invert_yaxis(); ax.set_xlim(-0.27, 0.22)
    ax.set_xlabel("best tuned baseline − tuned MOEA/D (HV$_{norm}$)\n← MOEA/D ahead          baseline ahead →", fontsize=9); ax.grid(axis="y", visible=False)
    ax.set_title("2  Against tuned baselines at\nequal time: depends on the problem", loc="left", fontsize=11, fontweight="bold")
    # (c) adaptive selection vs uniform
    ax = axes[2]
    pick = [("MDP", "51–200", "MDP\n51–200"), ("CDP", "n = 500", "CDP\n500"), ("GDP", "n = 500 loose cost", "GDP 500\nloose"),
            ("GDP", "n = 500 tight cost", "GDP 500\ntight"), ("MDP", "n = 2000 (MDG-b)", "MDP\n2000")]
    for j, (name, col, mk) in enumerate((("UCB-HV", COL[0], "o"), ("GTS-HV", COL[1], "s"))):
        xs = [i + (j - 0.5) * 0.2 for i in range(len(pick))]
        m = [avs[p][k][name]["mean"] for p, k, _ in pick]
        lo = [avs[p][k][name]["mean"] - avs[p][k][name]["lo"] for p, k, _ in pick]
        hi = [avs[p][k][name]["hi"] - avs[p][k][name]["mean"] for p, k, _ in pick]
        ax.errorbar(xs, m, yerr=[lo, hi], fmt=mk, color=col, ms=6, capsize=3, lw=1.4, label=name)
    ax.axhline(0, color=INK2, lw=0.9); ax.set_xticks(range(len(pick))); ax.set_xticklabels([x[2] for x in pick], fontsize=7.5)
    ax.set_ylabel("HV$_{norm}$(selector) − HV$_{norm}$(uniform)"); ax.legend(frameon=False, loc="upper left"); ax.grid(axis="x", visible=False)
    ax.set_title("3  A bandit beats uniform selection\nonly on large or tight instances", loc="left", fontsize=11, fontweight="bold")
    # (d) anchoring
    ax = axes[3]
    rows = [("CDP GKD-b150", "CDP", "n<=150", "GKD-b n150", "random_ref", "random_both"), ("CDP GKD-c", "CDP", "n=500", "GKD-c n500", "random", "random_both"),
            ("CDP MDG-b", "CDP", "n=500", "MDG-b n500", "random", "random_both"), ("GDP GKD-c", "GDP", "n=500", "GKD-c n500", "random", "random_both"),
            ("GDP MDG-b", "GDP", "n=500", "MDG-b n500", "random", "random_both")]
    for i, (name, prob, sz, fam, k0, k1) in enumerate(rows):
        a0, a1 = anc[prob][sz][k0]["ext_ratio"][fam], anc[prob][sz][k1]["ext_ratio"][fam]
        ax.plot([a0, a1], [i, i], color=INK2, lw=1.4, zorder=1)
        ax.scatter([a0], [i], color=COL[3], s=55, zorder=2, label="no anchor" if i == 0 else None)
        ax.scatter([a1], [i], color=COL[2], s=55, zorder=2, label="with MaxMin anchor" if i == 0 else None)
        ax.text(a1 + 0.03, i, f"{a1:.2f}", va="center", fontsize=9)
    ax.axvline(1.0, color=INK, lw=1.0, ls="--"); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0.4, 1.35); ax.set_xlabel("MaxMin end of the front / best published value\n(GDP: published 60-s heuristic)", fontsize=9)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=9); ax.grid(axis="y", visible=False)
    ax.set_title("4  A short MaxMin intensification\ncovers the end of the front", loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("Bi-objective (MaxSum + MaxMin) MDP, CDP and GDP: MOEA/D with a pool of eight operators", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "graphical_abstract")


def size_figure():
    """HV_norm medio (referencia propria do experimento) dos metodos principais do E18, no problema livre e no limitado (rho = 2), por celula."""
    d = json.load(open(A / "e18.json"))
    meths = [("moead", "MOEA/D-Uniform (8 operators)", COL[0], ""), ("moead_size", "+ size operators", COL[1], ""),
             ("moead_size_ucbhv", "UCB-HV + size operators", COL[2], "//"), ("moead_size_gtshv", "GTS-HV + size operators", COL[3], "\\\\"),
             ("vns_add", "VNS + insertions", COL[6], "xx"), ("ts_add", "Tabu + insertions", COL[4], "..")]
    cells = [("CDP-S", "CDP, n ≤ 150"), ("GDP-S", "GDP, n ≤ 150"), ("CDP-L", "CDP, n = 500"), ("GDP-L", "GDP, n = 500")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, exp, ttl in zip(axes, ("free", "bound2"), ("Free cardinality", "Cardinality bounded, |S| ≤ ⌈2 k$_{ref}$⌉")):
        w = 0.8 / len(meths)
        for j, (m, lab, col, hatch) in enumerate(meths):
            xs, ys = [], []
            for i, (c, _) in enumerate(cells):
                x = d.get(f"{exp}/{c}", {}).get("methods", {}).get(m)
                if x is not None:
                    xs.append(i + (j - (len(meths) - 1) / 2) * w); ys.append(x["mean"])
            ax.bar(xs, ys, width=w * 0.95, color=col, hatch=hatch, edgecolor="white", linewidth=0.4, label=lab if exp == "free" else None)
        ax.set_xticks(range(len(cells))); ax.set_xticklabels([l.replace(", ", chr(10)) for _, l in cells], fontsize=7)
        ax.set_title(ttl, loc="left", fontsize=9, fontweight="bold"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("HV$_{norm}$ against the reference front" + chr(10) + "of each experiment (mean over instances)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig6_size_operators")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": aos_vs_size, "fig6": size_figure, "ga": graphical_abstract}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(FIGS)):
        FIGS[k]()
