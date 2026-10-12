"""Analise estatistica com REPLICAS do regime normalizado (results/capacitated_norm/<P>/runs, rep00..rep04).

Para cada (instancia, metodo): indicador por replica contra a frente de referencia FIXA (src/reference_sets.py),
media sobre replicas -> matriz instancias x 12 metodos -> Friedman, ranks medios, CD de Nemenyi (q=3,268 p/ K=12),
Wilcoxon pareado com Holm em todos os 66 pares (criterio principal do projeto), AOS vs random com tamanho de efeito
(A12 sobre as instancias) e a variabilidade entre replicas (CV) por metodo.

    python src/reference_sets.py build                  # antes, para a referencia incluir as replicas
    python analyze_replicated.py --problem CDP --max-n 150 [--metric hv_norm|igd_plus|eps_plus]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
import reference_sets as RS  # noqa: E402
from experiment import ALL_MODES  # noqa: E402

AOS = ALL_MODES[:7]
Q05_K12 = 3.268  # q_{0,05}(K=12)/sqrt(2); mesma constante de analysis.py


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def a12(x, y):
    """P(x > y) + 0,5 P(x = y) sobre pares (tamanho de efeito de Vargha-Delaney)."""
    x = np.asarray(x); y = np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum(); eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def load_matrix(problem, max_n, metric, reps, min_n=0):
    ref = RS.load(problem)
    if problem == "MDP":
        base = ROOT / "results_norm" / "runs"
        cat = [r for r in csv.DictReader(open(ROOT / "instances" / "catalog.csv")) if min_n <= int(r["n"]) <= max_n]
    else:
        base = ROOT / "results" / "capacitated_norm" / problem / "runs"
        cat = [r for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")) if r["problem"] == problem and min_n <= int(r["n"]) <= max_n]
    names, cube = [], []
    for r in cat:
        nm = r["name"]
        paths = {(m, k): base / f"{nm}_{m}_rep{k:02d}.json" for m in ALL_MODES for k in range(reps)}
        if not all(p.exists() for p in paths.values()):
            continue
        cube.append([[RS.indicators(np.array(json.load(open(paths[(m, k)]))["pareto_front"]), ref[nm])[metric] for k in range(reps)] for m in ALL_MODES])
        names.append(nm)
    return names, np.array(cube)  # (N, 12, reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP", "MDP"], default="CDP")
    ap.add_argument("--max-n", type=int, default=150)
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus", "eps_plus"], default="hv_norm")
    a = ap.parse_args()
    higher_better = a.metric == "hv_norm"
    names, cube = load_matrix(a.problem, a.max_n, a.metric, a.reps, a.min_n)
    N, K, R = cube.shape
    if N < 5:
        print("poucas instancias completas:", N); return
    M = cube.mean(axis=2)                                       # medias sobre replicas
    sign = -1.0 if higher_better else 1.0
    ranks = np.array([rankdata(sign * row) for row in M])
    avg = ranks.mean(axis=0)
    chi2, p = friedmanchisquare(*[M[:, j] for j in range(K)])
    cd = Q05_K12 * math.sqrt(K * (K + 1) / (6 * N))
    print(f"{a.problem} {a.min_n}<=n<={a.max_n}  metrica={a.metric}  N={N} instancias x {K} metodos x {R} replicas")
    print(f"Friedman chi2={chi2:.1f} p={p:.2g} | CD Nemenyi={cd:.2f}\n")
    cv = np.nanmean(cube.std(axis=2, ddof=1) / np.where(cube.mean(axis=2) == 0, np.nan, np.abs(cube.mean(axis=2))), axis=0)
    print(f"{'metodo':16s} rank medio   {a.metric} medio   CV entre replicas")
    for j in np.argsort(avg):
        print(f"{ALL_MODES[j]:16s} {avg[j]:8.2f}   {M[:, j].mean():10.4f}   {cv[j]*100:8.1f}%")
    pairs = list(itertools.combinations(range(K), 2)); ps = []
    for i, j in pairs:
        d = M[:, i] - M[:, j]
        ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
    adj = holm(ps)
    sig_w = sum(x < 0.05 for x in adj); sig_n = sum(abs(avg[i] - avg[j]) > cd for i, j in pairs)
    print(f"\npares significativos: Wilcoxon-Holm {sig_w}/{len(pairs)} | Nemenyi {sig_n}/{len(pairs)}")
    # grupos: topo = metodos nao significativamente piores que o melhor (Wilcoxon-Holm)
    best = int(np.argmin(avg)); pm = {(i, j): adj[k] for k, (i, j) in enumerate(pairs)}
    top = [ALL_MODES[best]] + [ALL_MODES[j] for j in range(K) if j != best and pm[(min(best, j), max(best, j))] >= 0.05]
    print(f"grupo do topo (nao distinguiveis do 1o, {ALL_MODES[best]}): {', '.join(top)}")
    jr = ALL_MODES.index("random"); print("\nAOS vs random (Wilcoxon-Holm entre as 6 comparacoes; A12 = P(AOS melhor que random)):")
    ps2, rows = [], []
    for m in AOS:
        if m == "random": continue
        j = ALL_MODES.index(m); d = M[:, j] - M[:, jr]
        ps2.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
        x, y = (M[:, j], M[:, jr]) if higher_better else (-M[:, j], -M[:, jr])
        rows.append((m, d.mean() * (1 if higher_better else -1), a12(x, y)))
    aos_rows = []
    for (m, dm, aa), pa in zip(rows, holm(ps2)):
        print(f"   {m:16s} dif media {dm:+.4f}  A12={aa:.2f}  p_holm={pa:.4f}{' *' if pa < 0.05 else ''}")
        aos_rows.append({"method": m, "mean_diff": float(dm), "a12": float(aa), "p_holm": float(pa)})
    out = ROOT / "results" / "analysis" / f"stats_{a.problem}_norm_n{a.min_n}-{a.max_n}_reps{R}_{a.metric}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"problem": a.problem, "min_n": a.min_n, "max_n": a.max_n, "reps": R, "metric": a.metric, "N": N, "friedman_chi2": float(chi2), "friedman_p": float(p),
               "cd": cd, "avg_ranks": dict(zip(ALL_MODES, map(float, avg))), "wilcoxon_holm_sig": int(sig_w), "nemenyi_sig": int(sig_n),
               "top_group": top,
               "mean_metric": dict(zip(ALL_MODES, map(float, M.mean(axis=0)))),
               "cv_between_reps": dict(zip(ALL_MODES, map(float, cv))),
               "aos_vs_random": aos_rows,
               "pairwise_p_holm": {f"{ALL_MODES[i]}|{ALL_MODES[j]}": float(adj[k]) for k, (i, j) in enumerate(pairs)}}, open(out, "w"), indent=1)
    print("\nsalvo:", out)


if __name__ == "__main__":
    main()
