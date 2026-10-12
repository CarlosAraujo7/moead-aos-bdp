"""E8: ablacao de operadores com replicas (run_e8.py): operador unico (op0..op7) e leave-one-out (no0..no7) em CDP e GDP (n <= 150).

Por problema: hv_norm (referencia congelada) medio sobre as 3 replicas por instancia; cada variante e comparada com o pool completo de selecao
uniforme (`random` de results/capacitated_norm, mesmas instancias e replicas): diferenca media, Wilcoxon pareado, Holm sobre as 16 comparacoes e Â12;
tambem o rank medio entre os 17 metodos e o tempo mediano por run relativo ao do operador-unico mais lento (os tempos so sao comparaveis dentro do E8).

    python analyze_e8.py [--metric hv_norm|igd_plus]
Saida: terminal + results/analysis/e8_operator_ablation[_igd_plus].json
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
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import reference_sets as RS  # noqa: E402
from analyze_phase2 import a12, holm  # noqa: E402
from operators import OPERATOR_NAMES  # noqa: E402

OUT = ROOT / "results" / "e8_operator_ablation"
TAGS = [f"op{k}" for k in range(8)] + [f"no{k}" for k in range(8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus"], default="hv_norm")
    a = ap.parse_args()
    hb = a.metric == "hv_norm"
    sfx = "" if hb else "_igd_plus"
    res = {}
    for prob in ("CDP", "GDP"):
        ref = RS.load(prob)
        names = [r["name"] for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == prob and int(r["n"]) <= 150]
        runs = OUT / prob / "runs"
        norm = ROOT / "results" / "capacitated_norm" / prob / "runs"
        M, T = {}, {}
        for tag in ["random"] + TAGS:
            d = norm if tag == "random" else runs
            vals, tms = [], []
            for nm in names:
                fs = [d / f"{nm}_{tag}_rep{k:02d}.json" for k in range(3)]
                if not all(f.exists() for f in fs):
                    vals = None
                    break
                js = [json.load(open(f)) for f in fs]
                vals.append(np.mean([RS.indicators(np.array(j["pareto_front"], float).reshape(-1, 2), ref[nm])[a.metric] for j in js]))
                tms.append(np.median([j["elapsed_s"] for j in js]))
            if vals is not None:
                M[tag] = np.array(vals); T[tag] = float(np.median(tms))
        done = [t for t in TAGS if t in M]
        if "random" not in M or not done:
            print(f"{prob}: dados incompletos ({len(done)}/16 variantes)")
            continue
        ps, rows = [], []
        for t in done:
            d = M[t] - M["random"]
            ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
            x, y = (M[t], M["random"]) if hb else (-M[t], -M["random"])
            rows.append((t, float(d.mean() * (1 if hb else -1)), a12(x, y)))
        adj = holm(ps)
        allm = ["random"] + done
        S = np.stack([M[m] for m in allm], axis=1)
        ranks = np.mean([rankdata(-r if hb else r) for r in S], axis=0)
        out = {}
        print(f"\n=== {prob} (N={len(names)}, 3 replicas, {a.metric}); media do pool uniforme {M['random'].mean():.4f}; rank medio entre {len(allm)} metodos")
        print(f"   {'variante':28s} {'media':>7s} {'dif':>8s} {'A12':>5s} {'p_holm':>8s} {'rank':>6s} {'t_med(s)':>8s}")
        print(f"   {'pool completo (uniforme)':28s} {M['random'].mean():7.4f} {'':8s} {'':5s} {'':8s} {ranks[0]:6.2f}")
        for (t, dm, aa), pa, rk in zip(rows, adj, ranks[1:]):
            nm = ("only " if t.startswith("op") else "pool without ") + OPERATOR_NAMES[int(t[2:])]
            out[t] = dict(label=nm, mean=float(M[t].mean()), diff=dm, a12=aa, p_holm=pa, rank=float(rk), time_median=T[t])
            print(f"   {nm:28s} {M[t].mean():7.4f} {dm:+8.4f} {aa:5.2f} {pa:8.4f} {rk:6.2f} {T[t]:8.1f}{' *' if pa < 0.05 else ''}")
        out["pool"] = dict(mean=float(M["random"].mean()), rank=float(ranks[0]), time_median=T["random"], N=len(names))
        res[prob] = out
    (ROOT / "results" / "analysis" / f"e8_operator_ablation{sfx}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
