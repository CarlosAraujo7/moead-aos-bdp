"""Fase 2: variantes de AOS (run_phase2.py) vs os 12 metodos de results/capacitated_norm, mesmas instancias/replicas.
hv_norm (ou IGD+/eps+) contra a frente de referencia fixa; media sobre replicas; Wilcoxon pareado de cada variante contra
`random` e contra o melhor AOS existente, com Holm; A12. Rode antes: python src/reference_sets.py build

    python analyze_phase2.py --problem CDP --max-n 150 --reps 3 [--metric hv_norm]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
import reference_sets as RS  # noqa: E402
from experiment import ALL_MODES  # noqa: E402
from run_phase2 import VARIANTS  # noqa: E402


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def a12(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return float(((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP"], default="CDP")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=150)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus", "eps_plus"], default="hv_norm")
    a = ap.parse_args()
    hb = a.metric == "hv_norm"
    ref = RS.load(a.problem)
    norm = ROOT / "results" / "capacitated_norm" / a.problem / "runs"
    p2 = ROOT / "results" / "phase2_aos" / a.problem / "runs"
    names = [r["name"] for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv"))
             if r["problem"] == a.problem and a.min_n <= int(r["n"]) <= a.max_n]
    present = [v for v in VARIANTS if all((p2 / f"{nm}_{v}_rep{k:02d}.json").exists() for nm in names for k in range(a.reps))]
    methods = list(ALL_MODES) + present
    print("variantes completas:", ", ".join(present) or "nenhuma")
    rows, used = [], []
    for nm in names:
        paths = {(m, k): (norm if m in ALL_MODES else p2) / f"{nm}_{m}_rep{k:02d}.json" for m in methods for k in range(a.reps)}
        if not all(p.exists() for p in paths.values()):
            continue
        rows.append([np.mean([RS.indicators(np.array(json.load(open(paths[(m, k)]))["pareto_front"]), ref[nm])[a.metric] for k in range(a.reps)]) for m in methods])
        used.append(nm)
    M = np.array(rows)
    if len(M) < 5:
        print("poucas instancias completas:", len(M)); return
    sign = -1.0 if hb else 1.0
    R = np.array([rankdata(sign * r) for r in M]); avg = R.mean(0)
    print(f"{a.problem} {a.min_n}<=n<={a.max_n} metrica={a.metric}: N={len(M)} instancias x {len(methods)} metodos x {a.reps} replicas\n")
    for j in np.argsort(avg):
        tag = "  <-- NOVA" if methods[j] in present else ""
        print(f"  {methods[j]:18s} rank medio {avg[j]:6.2f}   {a.metric} medio {M[:, j].mean():.4f}{tag}")
    jr = methods.index("random")
    best_exist = min((m for m in ALL_MODES if m in ("hypervolume", "thompson_hv", "dyts_hv", "dyts_scalar")), key=lambda m: avg[methods.index(m)])
    jb = methods.index(best_exist)
    print(f"\nVariantes novas vs `random` e vs melhor AOS existente ({best_exist}); Wilcoxon pareado, Holm (Holm sobre as variantes completas); A12 = P(variante melhor)")
    for ref_name, jj in (("random", jr), (best_exist, jb)):
        ps, rr = [], []
        for v in present:
            j = methods.index(v); d = M[:, j] - M[:, jj]
            ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
            x, y = (M[:, j], M[:, jj]) if hb else (-M[:, j], -M[:, jj])
            rr.append((v, d.mean() * (1 if hb else -1), a12(x, y)))
        print(f"  --- vs {ref_name}")
        for (v, dm, aa), pa in zip(rr, holm(ps)):
            print(f"      {v:18s} dif media {dm:+.4f}  A12={aa:.2f}  p_holm={pa:.4f}{' *' if pa < 0.05 else ''}")


if __name__ == "__main__":
    main()
