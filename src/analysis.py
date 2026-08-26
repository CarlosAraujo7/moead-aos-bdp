"""
analysis.py -- Analise estatistica dos resultados do MOEA/D-AOS.

Entrada esperada: results/summary.csv  (gerado por run_full.py / experiment.py)
Colunas obrigatorias: instance, family, n, m, mode, label,
                      hv_mean, hv_std, coverage_mean, coverage_std,
                      epsilon_mean, epsilon_std, pareto_size_mean, time_mean_s

Saidas (em results/analysis/)
------------------------------
    ranking_table.csv        -- ranking medio global por metodo
    ranking_by_family.csv    -- ranking medio por familia de instancias
    wilcoxon_full.csv        -- testes de Wilcoxon signed-rank (por par de metodos)
    fig_hv_by_family.png     -- HV medio por familia e metodo
    fig_ranking_boxplot.png  -- distribuicao do HV por metodo
    fig_hv_relative.png      -- HV relativo ao melhor por faixa de n
    fig_scalability.png      -- tempo de execucao vs n (por metodo)
    report_full.txt          -- relatorio interpretativo
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------

COLORS = {
    "hypervolume":      "#1E88E5",
    "scalar":           "#E53935",
    "thompson_hv":      "#00ACC1",
    "thompson_scalar":  "#F4511E",
    "random":           "#43A047",
    "nsga2":            "#8E24AA",
    "grasp":            "#FB8C00",
    "grasp_pr":         "#6D4C41",
}
LABELS = {
    "hypervolume":      "MOEAD-AOS-HV",
    "scalar":           "MOEAD-AOS-scalar",
    "thompson_hv":      "MOEAD-Thompson-HV",
    "thompson_scalar":  "MOEAD-Thompson-scalar",
    "random":           "MOEAD-random",
    "nsga2":            "NSGA-II",
    "grasp":            "GRASP-MO",
    "grasp_pr":         "GRASP-PR",
}
MODE_ORDER = [
    "hypervolume", "thompson_hv",
    "scalar", "thompson_scalar", "random",
    "nsga2", "grasp", "grasp_pr",
]


# ---------------------------------------------------------------------------
# Leitura de dados
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    float_cols = [
        "hv_mean", "hv_std", "hv_norm_mean", "hv_norm_std",
        "coverage_mean", "coverage_std",
        "epsilon_mean", "epsilon_std",
        "pareto_size_mean", "time_mean_s",
    ]
    for r in rows:
        for c in float_cols:
            try:
                r[c] = float(r[c])
            except (KeyError, ValueError):
                r[c] = 0.0
        r["n"] = int(r.get("n", 0))
        r["m"] = int(r.get("m", 0))
        # Garante coluna family
        if "family" not in r:
            r["family"] = "unknown"
    return rows


# ---------------------------------------------------------------------------
# Testes de Wilcoxon
# ---------------------------------------------------------------------------

PAIRS_MAIN = [
    # Comparacoes principais: proposta vs. baselines
    ("hypervolume",     "nsga2"),
    ("hypervolume",     "grasp_pr"),
    ("hypervolume",     "random"),
    # Ablacao: estrategia de selecao (UCB1 vs Thompson)
    ("hypervolume",     "thompson_hv"),
    # Ablacao: modo de recompensa (HV vs escalar)
    ("hypervolume",     "scalar"),
    ("thompson_hv",     "thompson_scalar"),
    # Baseline vs baseline
    ("grasp_pr",        "nsga2"),
    ("grasp_pr",        "grasp"),
    ("nsga2",           "random"),
]


def wilcoxon_pairs(rows: list[dict], metric: str, pairs: list[tuple]) -> list[dict]:
    # Agrega por instancia: um valor por instancia por metodo (media das replicas)
    inst_mode: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        inst_mode[(r["instance"], r["mode"])].append(r[metric])
    inst_mode_mean = {k: float(np.mean(v)) for k, v in inst_mode.items()}

    instances = sorted({r["instance"] for r in rows})
    results = []
    for a, b in pairs:
        va, vb = [], []
        for inst in instances:
            ka = inst_mode_mean.get((inst, a))
            kb = inst_mode_mean.get((inst, b))
            if ka is not None and kb is not None:
                va.append(ka)
                vb.append(kb)
        if len(va) < 2:
            continue
        diff = np.array(va) - np.array(vb)
        if np.all(diff == 0):
            results.append({
                "metric": metric, "A": a, "B": b,
                "n_instances": len(va),
                "mean_A": np.mean(va), "mean_B": np.mean(vb),
                "W": float("nan"), "p": float("nan"),
                "sig": "iguais", "winner": "empate",
            })
            continue
        try:
            stat, p = stats.wilcoxon(va, vb, alternative="two-sided")
        except ValueError:
            continue
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        if metric == "epsilon_mean":
            better = a if np.mean(va) < np.mean(vb) else b
        else:
            better = a if np.mean(va) > np.mean(vb) else b
        winner = better if p < 0.05 else "empate"
        results.append({
            "metric": metric, "A": a, "B": b,
            "n_instances": len(va),
            "mean_A": round(float(np.mean(va)), 5),
            "mean_B": round(float(np.mean(vb)), 5),
            "W": round(float(stat), 1), "p": round(float(p), 5),
            "sig": sig, "winner": winner,
        })
    return results


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _present_modes(rows: list[dict]) -> list[str]:
    """Retorna os modos presentes nos dados, na ordem padrao."""
    present = {r["mode"] for r in rows}
    return [m for m in MODE_ORDER if m in present]


def fig_hv_by_family(rows: list[dict], out: Path) -> None:
    """HV medio por familia de instancias, barras agrupadas por metodo."""
    families = sorted({r["family"] for r in rows})
    modes = _present_modes(rows)
    n_fam = len(families)
    n_mode = len(modes)
    w = 0.8 / n_mode

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_fam), 5))
    x = np.arange(n_fam)

    for j, mode in enumerate(modes):
        means = []
        errs  = []
        for fam in families:
            vals = [r["hv_mean"] for r in rows if r["family"] == fam and r["mode"] == mode]
            means.append(np.mean(vals) if vals else 0.0)
            errs.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        offset = (j - n_mode / 2 + 0.5) * w
        ax.bar(x + offset, means, w * 0.9,
               color=COLORS.get(mode, "#999"),
               label=LABELS.get(mode, mode),
               yerr=errs, capsize=2, alpha=0.85, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=9)
    ax.set_ylabel("Hipervolume medio")
    ax.set_title("HV medio por familia de instancias e metodo")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig_ranking_boxplot(rows: list[dict], out: Path) -> None:
    """Distribuicao do HV por metodo (boxplot)."""
    modes = _present_modes(rows)
    # Um valor por instancia: media das replicas do mesmo metodo
    inst_mode: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        inst_mode[(r["instance"], r["mode"])].append(r["hv_mean"])
    by_mode = {m: [float(np.mean(v)) for (inst, mm), v in inst_mode.items() if mm == m]
               for m in modes}

    fig, ax = plt.subplots(figsize=(11, 5))
    data   = [by_mode.get(m, []) for m in modes]
    labels = [LABELS.get(m, m) for m in modes]
    colors = [COLORS.get(m, "#999") for m in modes]

    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Hipervolume (media por instancia)")
    ax.set_title("Distribuicao do HV por metodo")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig_hv_relative(rows: list[dict], out: Path) -> None:
    """HV relativo ao melhor metodo por faixa de n."""
    ns = sorted({r["n"] for r in rows})
    # Agrupa em faixas
    def faixa(n):
        if n <= 30:   return "n<=30"
        if n <= 50:   return "31<=n<=50"
        if n <= 100:  return "51<=n<=100"
        if n <= 200:  return "101<=n<=200"
        if n <= 300:  return "201<=n<=300"
        return "n>300"

    faixas = ["n<=30","31<=n<=50","51<=n<=100","101<=n<=200","201<=n<=300","n>300"]
    faixas_present = [f for f in faixas if any(faixa(r["n"]) == f for r in rows)]
    if not faixas_present:
        return
    modes = _present_modes(rows)
    n_faixas = len(faixas_present)
    fig, axes = plt.subplots(1, n_faixas, figsize=(3.5 * n_faixas, 5), sharey=False)
    if n_faixas == 1:
        axes = [axes]

    for ax, fa in zip(axes, faixas_present):
        subset = [r for r in rows if faixa(r["n"]) == fa]
        insts  = sorted({r["instance"] for r in subset})
        mode_scores: dict[str, list[float]] = defaultdict(list)

        for inst in insts:
            best = max(
                (r["hv_mean"] for r in subset if r["instance"] == inst),
                default=0.0,
            )
            for r in [x for x in subset if x["instance"] == inst]:
                mode_scores[r["mode"]].append(
                    r["hv_mean"] / best if best > 0 else 0.0
                )

        for j, mode in enumerate(modes):
            vals = mode_scores.get(mode, [])
            if vals:
                ax.bar(j, np.mean(vals), 0.7,
                       color=COLORS.get(mode, "#999"),
                       alpha=0.85,
                       yerr=np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0,
                       capsize=3)
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(
            [LABELS.get(m, m).replace("MOEAD-", "")[:16] for m in modes],
            rotation=35, ha="right", fontsize=7,
        )
        ax.set_title(f"{fa} (n_inst={len(insts)})")
        ax.set_ylabel("HV / HV_melhor")
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.35)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle("HV relativo ao melhor metodo por faixa de n", y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def fig_scalability(rows: list[dict], out: Path) -> None:
    """Tempo medio de execucao vs n, por metodo."""
    modes = _present_modes(rows)
    ns    = sorted({r["n"] for r in rows})

    fig, ax = plt.subplots(figsize=(10, 5))
    for mode in modes:
        xs, ys = [], []
        for n in ns:
            times = [r["time_mean_s"] for r in rows if r["n"] == n and r["mode"] == mode]
            if times:
                xs.append(n)
                ys.append(np.mean(times))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5,
                    color=COLORS.get(mode, "#999"),
                    label=LABELS.get(mode, mode))

    ax.set_xlabel("Tamanho da instancia (n)")
    ax.set_ylabel("Tempo medio (s)")
    ax.set_title("Escalabilidade: tempo de execucao vs tamanho da instancia")
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Tabelas de ranking
# ---------------------------------------------------------------------------

def build_ranking(rows: list[dict], groupby: str | None = None) -> list[dict]:
    """Ranking medio global ou por familia."""
    modes = _present_modes(rows)
    groups = sorted({r.get(groupby, "all") for r in rows}) if groupby else ["all"]

    records = []
    for grp in groups:
        subset = [r for r in rows if (r.get(groupby, "all") == grp)] if groupby else rows
        for mode in modes:
            recs = [r for r in subset if r["mode"] == mode]
            if not recs:
                continue
            records.append({
                groupby or "group": grp,
                "mode":             mode,
                "label":            LABELS.get(mode, mode),
                "hv_mean":          round(float(np.mean([r["hv_mean"]          for r in recs])), 5),
                "hv_std":           round(float(np.mean([r["hv_std"]           for r in recs])), 5),
                "coverage_mean":    round(float(np.mean([r["coverage_mean"]    for r in recs])), 5),
                "epsilon_mean":     round(float(np.mean([r["epsilon_mean"]     for r in recs])), 5),
                "pareto_size_mean": round(float(np.mean([r["pareto_size_mean"] for r in recs])), 2),
                "time_mean_s":      round(float(np.mean([r["time_mean_s"]      for r in recs])), 4),
                "n_instances":      len(recs),
            })
    return records


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report(rows: list[dict], wilcoxon_results: list[dict]) -> str:
    sep = "=" * 72
    n_inst = len({r["instance"] for r in rows})
    n_mode = len({r["mode"] for r in rows})
    families = sorted({r["family"] for r in rows})
    modes = _present_modes(rows)

    lines = [
        sep,
        "  ANALISE COMPARATIVA -- MOEA/D-AOS vs. Baselines (MDP bi-objetivo)",
        sep,
        f"Instancias : {n_inst}",
        f"Familias   : {', '.join(families)}",
        f"Metodos    : {n_mode}",
        "",
    ]

    # Ranking global
    lines += [sep, "  1. RANKING GLOBAL POR METODO (media sobre todas as instancias)", sep]
    lines.append(f"  {'Metodo':<28} {'HV medio':>12} {'HV std':>10} {'Cobertura':>10} {'Epsilon':>10} {'Tempo(s)':>9}")
    lines.append("-" * 72)
    for mode in modes:
        recs = [r for r in rows if r["mode"] == mode]
        if not recs:
            continue
        lines.append(
            f"  {LABELS.get(mode, mode):<28}"
            f" {np.mean([r['hv_mean'] for r in recs]):>12.2f}"
            f" {np.mean([r['hv_std']  for r in recs]):>10.2f}"
            f" {np.mean([r['coverage_mean'] for r in recs]):>10.4f}"
            f" {np.mean([r['epsilon_mean']  for r in recs]):>10.4f}"
            f" {np.mean([r['time_mean_s']   for r in recs]):>9.3f}"
        )

    # Ranking por familia
    lines += ["", sep, "  2. HV MEDIO POR FAMILIA", sep]
    for fam in families:
        lines.append(f"  Familia: {fam}")
        for mode in modes:
            recs = [r for r in rows if r["family"] == fam and r["mode"] == mode]
            if not recs:
                continue
            lines.append(
                f"    {LABELS.get(mode, mode):<28}"
                f" HV={np.mean([r['hv_mean'] for r in recs]):.2f}"
                f" +/-{np.std([r['hv_mean'] for r in recs]):.2f}"
                f"  Cov={np.mean([r['coverage_mean'] for r in recs]):.4f}"
                f"  Eps={np.mean([r['epsilon_mean'] for r in recs]):.4f}"
            )
        lines.append("")

    # Wilcoxon
    lines += [sep, "  3. TESTES WILCOXON SIGNED-RANK (por instancia, bicaudal)", sep]
    for metric in sorted({r["metric"] for r in wilcoxon_results}):
        lines.append(f"  Metrica: {metric}")
        lines.append(f"  {'Par A':<22} {'Par B':<22} {'media_A':>9} {'media_B':>9} {'p':>8}  {'sig':>4}  vencedor")
        lines.append("  " + "-" * 80)
        for r in [x for x in wilcoxon_results if x["metric"] == metric]:
            lines.append(
                f"  {r['A']:<22} {r['B']:<22}"
                f" {r['mean_A']:>9.4f} {r['mean_B']:>9.4f}"
                f" {r['p']:>8.5f}  {r['sig']:>4}  {r['winner']}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wilcoxon com correcao Holm-Bonferroni (todos os pares)
# ---------------------------------------------------------------------------

def wilcoxon_all_pairs_holm(rows: list[dict], metric: str) -> list[dict]:
    """Testa todos os pares de metodos com correcao Holm-Bonferroni.

    Retorna lista com p-value bruto, p-value ajustado, e indicador de
    significancia para cada par. Usado para identificar diferencias
    estatisticas entre quaisquer dois algoritmos.
    """
    modes = _present_modes(rows)
    instances = sorted({r["instance"] for r in rows})
    inst_mode: dict[tuple, float] = {}
    for r in rows:
        key = (r["instance"], r["mode"])
        if key not in inst_mode:
            inst_mode[key] = r[metric]
        else:
            inst_mode[key] = (inst_mode[key] + r[metric]) / 2  # media simples

    raw_results = []
    for i, a in enumerate(modes):
        for b in modes[i+1:]:
            va, vb = [], []
            for inst in instances:
                ka = inst_mode.get((inst, a))
                kb = inst_mode.get((inst, b))
                if ka is not None and kb is not None:
                    va.append(ka)
                    vb.append(kb)
            if len(va) < 3:
                continue
            diff = np.array(va) - np.array(vb)
            if np.all(np.abs(diff) < 1e-12):
                continue
            try:
                stat, p = stats.wilcoxon(va, vb, alternative="two-sided")
                raw_results.append({"A": a, "B": b, "metric": metric,
                                     "n": len(va), "W": stat, "p_raw": p,
                                     "mean_A": float(np.mean(va)),
                                     "mean_B": float(np.mean(vb))})
            except ValueError:
                pass

    if not raw_results:
        return []

    # Correcao Holm-Bonferroni
    ps = [r["p_raw"] for r in raw_results]
    order = np.argsort(ps)
    n_tests = len(ps)
    p_adj = np.ones(n_tests)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(1.0, ps[idx] * (n_tests - rank))
    # Garante monotonia
    for rank in range(len(order) - 2, -1, -1):
        p_adj[order[rank]] = min(p_adj[order[rank]], p_adj[order[rank + 1]])

    for i, r in enumerate(raw_results):
        r["p_adj"] = round(float(p_adj[i]), 6)
        r["sig"] = "***" if p_adj[i] < 0.001 else ("**" if p_adj[i] < 0.01 else
                   ("*"  if p_adj[i] < 0.05  else "ns"))
        if metric in ("epsilon_mean",):
            r["winner"] = r["A"] if r["mean_A"] < r["mean_B"] else r["B"]
        else:
            r["winner"] = r["A"] if r["mean_A"] > r["mean_B"] else r["B"]
        if p_adj[i] >= 0.05:
            r["winner"] = "empate"

    return raw_results


# ---------------------------------------------------------------------------
# Teste de Friedman + post-hoc Nemenyi
# ---------------------------------------------------------------------------

def friedman_nemenyi(rows: list[dict], metric: str) -> dict:
    """Teste de Friedman para comparar k algoritmos em N instancias.

    Retorna:
        statistic  : estatistica chi-quadrado do Friedman
        p_value    : p-value do teste
        ranks      : rank medio de cada algoritmo
        cd         : critical difference (Nemenyi, alpha=0.05)
        comparisons: dict {(A, B): significativo}
    """
    modes = _present_modes(rows)
    instances = sorted({r["instance"] for r in rows})

    # Matriz N x K de valores
    matrix = []
    valid_insts = []
    for inst in instances:
        row_vals = {}
        for mode in modes:
            vals = [r[metric] for r in rows if r["instance"] == inst and r["mode"] == mode]
            if vals:
                row_vals[mode] = float(np.mean(vals))
        if len(row_vals) == len(modes):
            matrix.append([row_vals[m] for m in modes])
            valid_insts.append(inst)

    if len(matrix) < 3:
        return {"error": "Instancias insuficientes para Friedman"}

    M = np.array(matrix)  # shape (N, K)
    N, K = M.shape

    # Ranks por linha (instancia)
    if metric in ("epsilon_mean",):
        # Minimizacao: menor rank = melhor
        rank_matrix = np.array([stats.rankdata(row) for row in M])
    else:
        rank_matrix = np.array([stats.rankdata(-row) for row in M])

    avg_ranks = rank_matrix.mean(axis=0)

    # Estatistica de Friedman
    stat_result = stats.friedmanchisquare(*M.T.tolist())
    chi2_stat = float(stat_result.statistic)
    p_val     = float(stat_result.pvalue)

    # Critical Difference (Nemenyi, alpha=0.05)
    # q_alpha para k grupos, alpha=0.05 — valores tabelados
    q_table = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 12: 3.268}
    q_alpha = q_table.get(K, q_table.get(min(q_table.keys(), key=lambda x: abs(x - K)), 3.31))
    cd = q_alpha * math.sqrt(K * (K + 1) / (6 * N))

    # Pares significativos
    comparisons = {}
    for i in range(K):
        for j in range(i + 1, K):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            comparisons[(modes[i], modes[j])] = bool(diff > cd)

    return {
        "metric":    metric,
        "N":         N,
        "K":         K,
        "statistic": round(chi2_stat, 4),
        "p_value":   round(p_val, 6),
        "avg_ranks": {modes[i]: round(float(avg_ranks[i]), 4) for i in range(K)},
        "cd":        round(cd, 4),
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# CD-diagram
# ---------------------------------------------------------------------------

def fig_cd_diagram(friedman_result: dict, out: Path, title: str = "") -> None:
    """Gera o Critical Difference diagram (Demsar, 2006).

    Metodos com diferenca de rank medio <= CD sao conectados por uma linha
    horizontal, indicando ausencia de diferenca estatisticamente significativa.
    """
    if "error" in friedman_result:
        return

    avg_ranks = friedman_result["avg_ranks"]
    cd        = friedman_result["cd"]
    modes     = sorted(avg_ranks.keys(), key=lambda m: avg_ranks[m])
    n_methods = len(modes)

    fig_h = max(4, n_methods * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(-1, 2)
    ax.axis("off")

    # Linha principal (eixo de ranks)
    ax.plot([1, n_methods], [1, 1], "k-", lw=1.5)
    for r in range(1, n_methods + 1):
        ax.plot([r, r], [0.95, 1.05], "k-", lw=1)
        ax.text(r, 0.85, str(r), ha="center", va="top", fontsize=9)

    # Metodos: metade esquerda acima, metade direita abaixo
    half = n_methods // 2
    for idx, mode in enumerate(modes):
        r = avg_ranks[mode]
        norm_r = 1 + (r - 1) * (n_methods - 1) / (n_methods - 1)  # ja normalizado
        x_pos = r
        if idx < half:
            y_text, y_line = 1.55 - idx * 0.28, 1.05
            va = "bottom"
        else:
            y_text, y_line = -0.45 + (idx - half) * 0.28, 0.95
            va = "top"
        label = LABELS.get(mode, mode).replace("MOEAD-", "")
        ax.annotate(
            f"{label}\n(rank={avg_ranks[mode]:.2f})",
            xy=(x_pos, y_line), xytext=(x_pos, y_text),
            arrowprops=dict(arrowstyle="-", color=COLORS.get(mode, "#999"), lw=1),
            ha="center", va=va, fontsize=8,
            color=COLORS.get(mode, "#555"),
        )

    # Grupos (linhas CD)
    drawn = set()
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            mi, mj = modes[i], modes[j]
            if abs(avg_ranks[mi] - avg_ranks[mj]) <= cd:
                group_key = (mi, mj)
                if group_key not in drawn:
                    x1, x2 = avg_ranks[mi], avg_ranks[mj]
                    ax.plot([x1, x2], [1.15, 1.15], "k-", lw=3, alpha=0.4)
                    drawn.add(group_key)

    # CD bar
    ax.annotate("", xy=(1 + cd, 1.75), xytext=(1, 1.75),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1 + cd / 2, 1.82, f"CD={cd:.2f}", ha="center", va="bottom", fontsize=9)

    metric_name = friedman_result.get("metric", "")
    ax.set_title(
        f"{title or 'Critical Difference Diagram'}\n"
        f"Friedman: χ²={friedman_result['statistic']:.2f}, "
        f"p={friedman_result['p_value']:.4f}  "
        f"(N={friedman_result['N']} inst, α=0.05)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figura HVN por familia
# ---------------------------------------------------------------------------

def fig_hvn_by_family(rows: list[dict], out: Path) -> None:
    """HVN medio (normalizado) por familia de instancias."""
    metric = "hv_norm_mean" if "hv_norm_mean" in (rows[0] if rows else {}) else "hv_mean"
    families = sorted({r["family"] for r in rows})
    modes    = _present_modes(rows)
    n_fam, n_mode = len(families), len(modes)
    w = 0.8 / n_mode

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_fam), 5))
    x = np.arange(n_fam)

    for j, mode in enumerate(modes):
        means, errs = [], []
        for fam in families:
            vals = [r[metric] for r in rows if r["family"] == fam and r["mode"] == mode]
            means.append(np.mean(vals) if vals else 0.0)
            errs.append(np.std(vals) / np.sqrt(max(len(vals), 1)))
        offset = (j - n_mode / 2 + 0.5) * w
        ax.bar(x + offset, means, w * 0.9,
               color=COLORS.get(mode, "#999"),
               label=LABELS.get(mode, mode),
               yerr=errs, capsize=2, alpha=0.85, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=9)
    ylabel = "HVN medio (HV normalizado)" if metric == "hv_norm_mean" else "HV medio"
    ax.set_ylabel(ylabel)
    ax.set_title("Hipervolume Normalizado (HVN) por familia de instancias")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    if metric == "hv_norm_mean":
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Analise do comportamento AOS (dados de runs individuais)
# ---------------------------------------------------------------------------

def fig_aos_behavior(runs_dir: Path, out_dir: Path) -> None:
    """Analisa e plota o comportamento do AOS ao longo das geracoes.

    Le os arquivos JSON individuais em runs_dir/ e gera:
      - fig_operator_selection_freq.png : frequencia de selecao por operador
      - fig_operator_qvalue_evolution.png : evolucao dos Q-values por geracoes
    """
    import glob
    json_files = sorted(glob.glob(str(runs_dir / "*.json")))
    if not json_files:
        return

    # Coleta dados de um subconjunto de runs UCB-HV para analise
    ucb_runs = []
    for jf in json_files:
        try:
            with open(jf) as f:
                d = json.load(f)
            if d.get("mode") == "hypervolume" and d.get("operator_counts"):
                ucb_runs.append(d)
        except Exception:
            continue

    if not ucb_runs:
        return

    # --- Figura 1: frequencia de selecao por operador (media entre runs) ---
    from operators import OPERATOR_NAMES as OP_NAMES
    n_ops = len(OP_NAMES)

    # Agrega contagens
    total_counts = np.zeros(n_ops)
    for d in ucb_runs:
        counts = d.get("operator_counts", [])
        for k, c in enumerate(counts[:n_ops]):
            total_counts[k] += float(c)

    fig, ax = plt.subplots(figsize=(10, 4))
    freqs = total_counts / total_counts.sum() if total_counts.sum() > 0 else total_counts
    colors_op = plt.cm.tab10(np.linspace(0, 1, n_ops))
    bars = ax.bar(range(n_ops), freqs, color=colors_op, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(n_ops))
    ax.set_xticklabels([n.replace("_", "\n") for n in OP_NAMES[:n_ops]], fontsize=8)
    ax.set_ylabel("Frequencia de selecao relativa")
    ax.set_title(f"Selecao de Operadores — MOEAD-AOS-HV\n(media sobre {len(ucb_runs)} runs)")
    ax.axhline(1 / n_ops, color="red", linestyle="--", alpha=0.5, label=f"uniforme (1/{n_ops})")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_operator_selection_freq.png", dpi=150)
    plt.close()

    # --- Figura 2: Q-values finais por operador (violin/strip) ---
    qvals_by_op = [[] for _ in range(n_ops)]
    for d in ucb_runs:
        qv = d.get("operator_q_values", [])
        for k, q in enumerate(qv[:n_ops]):
            qvals_by_op[k].append(float(q))

    if any(qvals_by_op):
        fig, ax = plt.subplots(figsize=(10, 4))
        data_nonempty = [q for q in qvals_by_op if q]
        labels_nonempty = [OP_NAMES[k].replace("_", "\n") for k, q in enumerate(qvals_by_op) if q]
        if data_nonempty:
            bp = ax.boxplot(data_nonempty, patch_artist=True, tick_labels=labels_nonempty)
            for patch, color in zip(bp["boxes"], colors_op):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
        ax.set_ylabel("Q-value final (UCB)")
        ax.set_title("Distribuicao dos Q-values finais por operador — MOEAD-AOS-HV")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_operator_qvalue_dist.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Analise de fase dos operadores (exploracao -> intensificacao)
# ---------------------------------------------------------------------------

def fig_operator_phase_analysis(runs_dir: Path, out_dir: Path,
                                 n_phases: int = 3) -> None:
    """Plota a frequencia de selecao de cada operador por fase da busca.

    Divide a sequencia de selecoes em n_phases (inicio/meio/fim) e mostra
    como o AOS adapta o portfolio ao longo do tempo.
    Hipotese: operadores de perturbacao (random_swap, large_perturbation)
    dominam no inicio; operadores de intensificacao (best_swap, biobj)
    dominam no final.
    """
    import glob
    json_files = sorted(glob.glob(str(runs_dir / "*.json")))
    if not json_files:
        return

    try:
        from operators import OPERATOR_NAMES as OP_NAMES
    except Exception:
        OP_NAMES = [f"op_{k}" for k in range(7)]
    n_ops = len(OP_NAMES)

    phase_counts = np.zeros((n_phases, n_ops))
    n_runs = 0

    for jf in json_files:
        try:
            with open(jf) as f:
                d = json.load(f)
        except Exception:
            continue
        sels = d.get("operator_selections", [])
        if not sels or d.get("operator_selection", "").startswith("random"):
            continue
        n_s = len(sels)
        ps  = n_s // n_phases
        for p in range(n_phases):
            start = p * ps
            end   = (p + 1) * ps if p < n_phases - 1 else n_s
            for k in sels[start:end]:
                if 0 <= int(k) < n_ops:
                    phase_counts[p, int(k)] += 1
        n_runs += 1

    if n_runs == 0:
        return

    # Normaliza por fase
    for p in range(n_phases):
        t = phase_counts[p].sum()
        if t > 0:
            phase_counts[p] /= t

    phase_labels = ["Inicio (1/3)", "Meio (2/3)", "Fim (3/3)"][:n_phases]
    x = np.arange(n_ops)
    w = 0.8 / n_phases
    colors_phase = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n_phases))

    fig, ax = plt.subplots(figsize=(12, 5))
    for p in range(n_phases):
        offset = (p - n_phases / 2 + 0.5) * w
        ax.bar(x + offset, phase_counts[p], w * 0.9,
               color=colors_phase[p], alpha=0.85, label=phase_labels[p],
               edgecolor="white")

    ax.axhline(1 / n_ops, color="red", linestyle="--", alpha=0.5,
               label=f"Uniforme (1/{n_ops})")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in OP_NAMES], fontsize=8)
    ax.set_ylabel("Frequencia relativa de selecao")
    ax.set_title(f"Evolucao do Portfolio de Operadores por Fase\n"
                 f"(media sobre {n_runs} runs UCB-HV)")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_operator_phase_evolution.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Mapa de calor: complementaridade de operadores por regiao da frente
# ---------------------------------------------------------------------------

def fig_operator_complementarity(runs_dir: Path, out_dir: Path) -> None:
    """Heatmap mostrando qual operador contribui para qual regiao da frente.

    Para cada run UCB-HV, identifica o operador dominante e a regiao da
    frente de Pareto (MaxSum-dominated vs. MaxMin-dominated vs. knee).
    Mostra que operadores diferentes sao especializados em regioes diferentes.
    """
    import glob
    json_files = sorted(glob.glob(str(runs_dir / "*.json")))
    if not json_files:
        return

    try:
        from operators import OPERATOR_NAMES as OP_NAMES
    except Exception:
        OP_NAMES = [f"op_{k}" for k in range(7)]
    n_ops = len(OP_NAMES)

    # 3 regioes: MaxSum-extremo, knee, MaxMin-extremo
    region_labels = ["MaxSum\nExtreme", "Knee\nRegion", "MaxMin\nExtreme"]
    n_regions = len(region_labels)
    contribution = np.zeros((n_ops, n_regions))
    counts = np.zeros(n_ops)

    for jf in json_files:
        try:
            with open(jf) as f:
                d = json.load(f)
        except Exception:
            continue
        pf = d.get("pareto_front", [])
        op_counts = d.get("operator_counts", [])
        if not pf or not op_counts:
            continue

        pts = np.array(pf)
        if len(pts) < 2:
            continue

        # Normaliza a frente
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        rng_ = maxs - mins
        rng_[rng_ < 1e-9] = 1.0
        pts_n = (pts - mins) / rng_

        # Identifica regioes: ordena por MaxSum decrescente
        sorted_idx = np.argsort(-pts_n[:, 0])
        n_pts = len(sorted_idx)
        t1, t2 = n_pts // 3, 2 * n_pts // 3

        region_pts = np.zeros(n_pts, dtype=int)
        region_pts[sorted_idx[:t1]]    = 0  # MaxSum extreme
        region_pts[sorted_idx[t1:t2]]  = 1  # Knee
        region_pts[sorted_idx[t2:]]    = 2  # MaxMin extreme

        # Operador dominante
        op_arr = np.array(op_counts[:n_ops], dtype=float)
        if op_arr.sum() == 0:
            continue
        dom_op = int(np.argmax(op_arr))
        counts[dom_op] += 1

        # Peso do operador em cada regiao (proporcional a pontos naquela regiao)
        for r in range(n_regions):
            frac = (region_pts == r).sum() / max(n_pts, 1)
            contribution[dom_op, r] += frac

    # Normaliza
    for k in range(n_ops):
        if counts[k] > 0:
            contribution[k] /= counts[k]

    if contribution.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(contribution.T, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=contribution.max())
    ax.set_xticks(range(n_ops))
    ax.set_xticklabels([n.replace("_", "\n") for n in OP_NAMES], fontsize=7)
    ax.set_yticks(range(n_regions))
    ax.set_yticklabels(region_labels, fontsize=9)
    ax.set_title("Mapa de Complementaridade dos Operadores\n"
                 "(contribuicao por regiao da frente de Pareto)")
    plt.colorbar(im, ax=ax, label="Frequencia relativa de contribuicao")
    for i in range(n_ops):
        for j in range(n_regions):
            ax.text(i, j, f"{contribution[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if contribution[i,j] < 0.5 else "white")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_operator_complementarity.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Analise teorica: conflito MaxSum vs MaxMin
# ---------------------------------------------------------------------------

def fig_conflict_analysis(rows: list[dict], out_dir: Path) -> None:
    """Plota o grau de conflito estrutural entre MaxSum e MaxMin por familia.

    Usa a correlacao de Spearman entre os dois objetivos, calculada a partir
    dos resultados experimentais (HV medio serve como proxy de qualidade).
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from theory import analyze_conflict
        from problem import MDPInstance
        import csv as _csv
        from pathlib import Path as _Path

        _root = _Path(__file__).parent.parent
        catalog = _root / "instances" / "catalog.csv"
        inst_root = _root / "instances"

        with catalog.open() as fh:
            cat_rows = list(_csv.DictReader(fh))

        families = sorted({r["family"] for r in rows})
        rho_by_family: dict[str, list[float]] = {f: [] for f in families}

        # Amostra ate 3 instancias por familia
        for fam in families:
            fam_rows = [r for r in cat_rows if r["family"] == fam][:3]
            for cr in fam_rows:
                fp = inst_root / cr["filepath"]
                if not fp.exists():
                    continue
                try:
                    inst = MDPInstance(fp)
                    rep  = analyze_conflict(inst, n_sample=200)
                    rho_by_family[fam].append(rep.spearman_rho)
                except Exception:
                    continue

        families_with_data = [f for f in families if rho_by_family[f]]
        if not families_with_data:
            return

        means = [float(np.mean(rho_by_family[f])) for f in families_with_data]
        stds  = [float(np.std(rho_by_family[f]))  for f in families_with_data]

        fig, ax = plt.subplots(figsize=(max(6, len(families_with_data) * 1.5), 4))
        x = np.arange(len(families_with_data))
        colors_bar = ["#E53935" if m < -0.2 else "#FB8C00" if m < 0 else "#43A047"
                      for m in means]
        ax.bar(x, means, color=colors_bar, alpha=0.85,
               yerr=stds, capsize=4, error_kw={"linewidth": 0.8})
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(families_with_data, fontsize=9)
        ax.set_ylabel("Correlacao de Spearman (MaxSum, MaxMin)")
        ax.set_title("Conflito Estrutural entre Objetivos por Familia de Instancias\n"
                     "(rho < 0 = conflito; rho ~ -1 = conflito total)")
        ax.set_ylim(-1.1, 0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_objective_conflict.png", dpi=150)
        plt.close()

    except Exception as e:
        print("   [AVISO] fig_conflict_analysis: " + str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
