"""E10: seleção adaptativa (UCB-HV, GTS-HV) vs uniforme em funcao do orcamento (run_e10.py).

Para cada problema/tamanho e cada multiplicador de geracoes, compara UCB-HV e GTS-HV com o uniforme NO MESMO orcamento, pareado por instancia
(media das 3 replicas): Wilcoxon, Holm sobre as comparacoes do grupo, A12 e diferenca media do indicador (HV_norm; ou -IGD+ com --metric igd_plus,
de modo que positivo = melhor). Tambem mostra a concentracao da selecao (fracao em BiObjBestSwap+MaxMinSwap).

REFERENCIA (importante): o HV_norm depende do ponto de referencia. Uma referencia montada so com os modos comparados (sem os baselines fracos) tem o
ponto de referencia muito proximo da frente e ATE INVERTEU o sinal da diferenca no CDP n=500 (descoberto em 7 out; ver CLAUDE.md §2s). Por isso:
  CDP/GDP n=500: referencia CONGELADA (results/reference_fronts/CDP|GDP.json; uniao de todos os metodos e ablacoes, a mesma da analise principal);
  MDP MDG-b n=2000 (fora da referencia congelada): uniao dos runs do E6 (6 metodos, inclui os baselines fracos) e do E10 de cada instancia,
  com ponto de referencia 1% abaixo do pior valor (mesma regra de reference_sets.build).

    python analyze_e10.py [--metric hv_norm|igd_plus]
Saida: terminal + results/analysis/e10_budget_regime[_igd_plus].json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import reference_sets as RS  # noqa: E402
from metrics import pareto_front  # noqa: E402
from analyze_phase2 import a12, holm  # noqa: E402

E10 = ROOT / "results" / "e10_budget_regime"
E6 = ROOT / "results" / "e6_scalability" / "runs"
NORM = {"CDP": ROOT / "results/capacitated_norm/CDP/runs", "GDP": ROOT / "results/capacitated_norm/GDP/runs"}
MODES = ["random", "hypervolume", "dyts_hv"]
LAB = {"random": "Uniform", "hypervolume": "UCB-HV", "dyts_hv": "GTS-HV"}
REPS = 3


def front(path):
    return np.array(json.load(open(path))["pareto_front"], float).reshape(-1, 2)


def build_ref_local(paths):
    allp = np.vstack([front(p) for p in paths if p.exists()])
    rf = pareto_front(allp)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    rp = lo - 0.01 * np.maximum(hi - lo, [1.0, 1e-9])
    return {"front": rf.tolist(), "ref_point": rp.tolist()}


def score(fr, ref, metric):
    v = RS.indicators(fr, ref)[metric]
    return v if metric == "hv_norm" else -v


def group(title, files_by_mult, insts, refs, metric, out):
    mults = list(files_by_mult)
    H = {m: {md: [] for md in MODES} for m in mults}
    SHARE = {m: {md: [] for md in MODES} for m in mults}
    kept = []
    for n in insts:
        ps = {(m, md, k): files_by_mult[m][md](n, k) for m in mults for md in MODES for k in range(REPS)}
        if not all(p.exists() for p in ps.values()) or n not in refs:
            continue
        for m in mults:
            for md in MODES:
                H[m][md].append(np.mean([score(front(ps[(m, md, k)]), refs[n], metric) for k in range(REPS)]))
                sh = []
                for k in range(REPS):
                    c = np.array(json.load(open(ps[(m, md, k)])).get("operator_counts", [0] * 8), float)
                    if c.sum() > 0:
                        sh.append((c[4] + c[5]) / c.sum())
                SHARE[m][md].append(float(np.mean(sh)) if sh else float("nan"))
        kept.append(n)
    print(f"\n=== {title} [{metric}]: N={len(kept)} instancias completas, {REPS} replicas")
    if len(kept) < 3:
        return
    res, ps_all, keys = {}, [], []
    for m in mults:
        for md in ("hypervolume", "dyts_hv"):
            d = np.array(H[m][md]) - np.array(H[m]["random"])
            ps_all.append(wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else 1.0)
            keys.append((m, md))
    ph = dict(zip(keys, holm(ps_all)))
    for m in mults:
        row = {md: dict(hv=float(np.mean(H[m][md])), share=float(np.nanmean(SHARE[m][md]))) for md in MODES}
        line = f"  {m:8s} Uniform {row['random']['hv']:.3f} (sel. BiObj+MaxMin {row['random']['share']:.2f})"
        for md in ("hypervolume", "dyts_hv"):
            d = np.array(H[m][md]) - np.array(H[m]["random"])
            row[md].update(diff=float(d.mean()), a12=a12(H[m][md], H[m]["random"]), p_holm=float(ph[(m, md)]), better=int((d > 0).sum()))
            line += (f" | {LAB[md]} {row[md]['hv']:.3f} (dif {d.mean():+.3f}, A12 {row[md]['a12']:.2f}, melhor em {row[md]['better']}/{len(d)}, "
                     f"p_Holm {ph[(m, md)]:.3f}; sel. {row[md]['share']:.2f})")
        print(line)
        res[m] = row
    out[title] = dict(N=len(kept), mults=res, instances=kept,
                      hv={m: {md: [float(x) for x in H[m][md]] for md in MODES} for m in mults})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus"], default="hv_norm")
    a = ap.parse_args()
    out = {}
    cat = list(csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")))
    for prob, sl, tag in (("CDP", slice(None), ""), ("GDP", slice(0, None, 2), ", k02 (tight cost budget)"), ("GDP", slice(1, None, 2), ", k03")):
        insts = [r["name"] for r in cat if r["problem"] == prob and int(r["n"]) == 500][sl]
        d = E10 / prob / "runs"
        fm = {
            "x1/4": {md: (lambda n, k, md=md, d=d: d / f"{n}_q25_{md}_rep{k:02d}.json") for md in MODES},
            "x1/2": {md: (lambda n, k, md=md, d=d: d / f"{n}_q50_{md}_rep{k:02d}.json") for md in MODES},
            "x1": {md: (lambda n, k, md=md, p=prob: NORM[p] / f"{n}_{md}_rep{k:02d}.json") for md in MODES},
        }
        group(f"{prob} n=500{tag}", fm, insts, RS.load(prob), a.metric, out)
    mats = [r for r in csv.DictReader(open(ROOT / "instances/catalog.csv")) if r["family"] == "MDG-b" and int(r["n"]) == 2000]
    d = E10 / "MDP" / "runs"
    idx10 = [0, 2, 4, 6, 8, 10, 12, 15, 17, 19]
    refs = {}
    for i in idx10:
        n = mats[i]["name"]
        paths = [E6 / f"{n}_{m}_rep{k:02d}.json" for m in ("random", "hypervolume", "dyts_hv", "nsga2", "grasp_pr", "vns_tchebycheff") for k in range(REPS)]
        paths += list(d.glob(f"{n}_*_rep*.json"))
        refs[n] = build_ref_local(paths)
    fm1 = {"x1": {md: (lambda n, k, md=md: d / f"{n}_{md}_rep{k:02d}.json") for md in MODES}}
    group("MDP MDG-b n=2000 (x1)", fm1, [mats[i]["name"] for i in idx10], refs, a.metric, out)
    fm4 = {"x1": fm1["x1"], "x4": {md: (lambda n, k, md=md: d / f"{n}_x4_{md}_rep{k:02d}.json") for md in MODES}}
    group("MDP MDG-b n=2000 (x1 e x4, 3 instancias)", fm4, [mats[i]["name"] for i in (0, 10, 19)], refs, a.metric, out)
    suffix = "" if a.metric == "hv_norm" else "_igd_plus"
    (ROOT / f"results/analysis/e10_budget_regime{suffix}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
