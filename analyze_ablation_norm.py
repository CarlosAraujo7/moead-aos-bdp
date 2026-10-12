"""Analise da ablacao de normalizacao (run_ablation_norm.py).

Condicao A (historica): 12 modos do catalogo (Tchebycheff sem normalizar).
Condicao B (normalizada): 7 modos MOEA/D-AOS normalizados + os 5 baselines do catalogo.
Em cada condicao, hv_norm usa a frente de referencia agrupada dos 12 metodos da condicao.
Wilcoxon pareado (hv_norm) de cada estrategia AOS contra `random`, com correcao de Holm.

    python analyze_ablation_norm.py --problem CDP --max-n 150
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from experiment import ALL_MODES, build_ref_point, compute_indicators, compute_reference_front  # noqa: E402

AOS = ["hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar", "random"]


def _front(p: Path):
    return np.array(json.load(open(p))["pareto_front"])


def hv_norm(fronts: dict) -> dict:
    allf = list(fronts.values())
    rf, rp = compute_reference_front(allf), build_ref_point(allf)
    return {m: compute_indicators(f, rf, rp)["hv_norm"] for m, f in fronts.items()}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def report(label, H):
    names = [i for i in H]
    M = np.array([[H[i][m] for m in ALL_MODES] for i in names])        # hv_norm
    R = np.array([rankdata(-row) for row in M])                          # rank 1 = melhor
    avg = R.mean(axis=0)
    chi2, p = friedmanchisquare(*[R[:, j] for j in range(R.shape[1])])
    print(f"\n=== {label}  (N={len(names)})  Friedman chi2={chi2:.1f} p={p:.2g}")
    for m, a, h in sorted(zip(ALL_MODES, avg, M.mean(axis=0)), key=lambda t: t[1]):
        print(f"   {m:16s} rank medio {a:5.2f}   hv_norm medio {h:.4f}")
    # AOS vs random (hv_norm pareado)
    jr = ALL_MODES.index("random")
    ps, labs = [], []
    for m in AOS:
        if m == "random":
            continue
        d = M[:, ALL_MODES.index(m)] - M[:, jr]
        ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
        labs.append((m, d.mean()))
    adj = holm(np.array(ps))
    print("   -- cada AOS vs random (diferenca media de hv_norm; Wilcoxon pareado, Holm) --")
    for (m, dm), pa in zip(labs, adj):
        print(f"      {m:16s} {dm:+.4f}   p_holm={pa:.4f} {'*' if pa < 0.05 else 'ns'}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP"], default="CDP")
    ap.add_argument("--max-n", type=int, default=150)
    a = ap.parse_args()
    cat = list(csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")))
    names = [r["name"] for r in cat if r["problem"] == a.problem and int(r["n"]) <= a.max_n]
    base = ROOT / "results" / "capacitated" / a.problem / "runs"
    norm = ROOT / "results" / "ablation_normalized" / a.problem / "runs"

    HA, HB = {}, {}
    for nm in names:
        if not all((norm / f"{nm}_{m}_rep00.json").exists() for m in AOS):
            continue
        hist = {m: _front(base / f"{nm}_{m}_rep00.json") for m in ALL_MODES}
        new = dict(hist)
        for m in AOS:
            new[m] = _front(norm / f"{nm}_{m}_rep00.json")
        HA[nm], HB[nm] = hv_norm(hist), hv_norm(new)
    if not HA:
        print("sem instancias completas ainda"); return
    report("A) HISTORICO (sem normalizar)", HA)
    report("B) AOS NORMALIZADO + baselines do catalogo", HB)

    # efeito direto: hv_norm de cada AOS, historico vs normalizado (mesmo conjunto de instancias nao e a mesma
    # referencia, entao comparar via rank medio acima; aqui so o ganho bruto por instancia dentro da condicao B,
    # usando uma referencia comum com as duas versoes juntas)
    print("\n=== Ganho direto (referencia agrupada com historico + normalizado juntos) ===")
    for m in AOS:
        gains = []
        for nm in HA:
            fr = {"hist": _front(base / f"{nm}_{m}_rep00.json"), "norm": _front(norm / f"{nm}_{m}_rep00.json")}
            for b in ("nsga2", "grasp", "grasp_pr", "vns_pareto", "vns_tchebycheff"):
                fr[b] = _front(base / f"{nm}_{b}_rep00.json")
            h = hv_norm(fr)
            gains.append(h["norm"] - h["hist"])
        g = np.array(gains)
        print(f"   {m:16s} ganho medio {g.mean():+.4f}  melhorou em {int((g > 0).sum())}/{len(g)}")


if __name__ == "__main__":
    main()
