"""Analise do piloto de n=500 (decisao D2): as rodadas grandes confirmam/alteram os achados de n<=150?

Para as instancias n=500 presentes em results/capacitated_norm/<P>/runs (rep00, 12 metodos),
compara, por problema:
  A) catalogo HISTORICO (results/capacitated, sem normalizar) nas MESMAS instancias;
  B) regime NORMALIZADO (results/capacitated_norm).
Em cada condicao: ranks medios (hv_norm, frente de referencia agrupada dos 12), Wilcoxon pareado de
cada AOS contra `random` (Holm) e ganho direto da normalizacao (referencia comum). N e pequeno
(~10 por problema): reportar como direcao do efeito, nao como prova.

    python analyze_n500_pilot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from experiment import ALL_MODES, build_ref_point, compute_indicators, compute_reference_front  # noqa: E402

AOS = ["hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar", "random"]
BASE = ["nsga2", "grasp", "grasp_pr", "vns_pareto", "vns_tchebycheff"]


def fr(p: Path):
    return np.array(json.load(open(p))["pareto_front"])


def hv_norm(fronts: dict) -> dict:
    allf = list(fronts.values())
    rf, rp = compute_reference_front(allf), build_ref_point(allf)
    return {m: compute_indicators(f, rf, rp)["hv_norm"] for m, f in fronts.items()}


def holm(ps):
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def report(label, H):
    M = np.array([[H[i][m] for m in ALL_MODES] for i in H])
    R = np.array([rankdata(-r) for r in M])
    print(f"\n  --- {label}  (N={len(M)})")
    for j in np.argsort(R.mean(0)):
        print(f"      {ALL_MODES[j]:16s} rank medio {R.mean(0)[j]:5.2f}   hv_norm medio {M.mean(0)[j]:.4f}")
    jr = ALL_MODES.index("random")
    names, ps, dm = [], [], []
    for m in AOS:
        if m == "random":
            continue
        d = M[:, ALL_MODES.index(m)] - M[:, jr]
        names.append(m)
        dm.append(d.mean())
        ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
    adj = holm(np.array(ps))
    print("      AOS vs random (dif. media hv; Wilcoxon pareado, Holm):")
    for n_, d_, p_ in zip(names, dm, adj):
        print(f"         {n_:16s} {d_:+.4f}  p_holm={p_:.3f}{' *' if p_ < 0.05 else ''}")


def main():
    for P in ("CDP", "GDP"):
        norm = ROOT / "results" / "capacitated_norm" / P / "runs"
        hist = ROOT / "results" / "capacitated" / P / "runs"
        names = sorted({f.name.rsplit("_" + m + "_rep00.json", 1)[0]
                        for f in norm.glob("*_n500_*_rep00.json") for m in ALL_MODES if f.name.endswith("_" + m + "_rep00.json")})
        full = [n for n in names if all((norm / f"{n}_{m}_rep00.json").exists() for m in ALL_MODES)]
        print(f"\n==================== {P}: {len(full)} instancias n=500 completas (de {len(names)} iniciadas)")
        if len(full) < 3:
            print("  poucas instancias; aguardar")
            continue
        HA, HB, gains = {}, {}, {m: [] for m in AOS}
        for n in full:
            fa = {m: fr(hist / f"{n}_{m}_rep00.json") for m in ALL_MODES}
            fb = {m: fr(norm / f"{n}_{m}_rep00.json") for m in ALL_MODES}
            HA[n], HB[n] = hv_norm(fa), hv_norm(fb)
            for m in AOS:
                h = hv_norm({"hist": fa[m], "norm": fb[m], **{b: fb[b] for b in BASE}})
                gains[m].append(h["norm"] - h["hist"])
        report("A) HISTORICO (sem normalizar), mesmas instancias", HA)
        report("B) NORMALIZADO (arvore capacitated_norm)", HB)
        print("\n  --- ganho direto da normalizacao (referencia comum historico+normalizado+baselines)")
        for m in AOS:
            g = np.array(gains[m])
            print(f"      {m:16s} ganho medio {g.mean():+.4f}  melhorou em {int((g > 0).sum())}/{len(g)}")
        t = {m: np.median([json.load(open(norm / f'{n}_{m}_rep00.json'))['elapsed_s'] for n in full]) for m in ALL_MODES}
        print("\n  --- tempo mediano por run em n=500 (s, sob a carga da maquina no momento)")
        print("      " + "  ".join(f"{m}={t[m]:.0f}" for m in ALL_MODES))


if __name__ == "__main__":
    main()
