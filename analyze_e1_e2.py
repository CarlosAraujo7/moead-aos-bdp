"""E1 (varredura de hiperparametros dos bandits), E1b (DYTS fiel / recompensa `replace`) e E2 (orcamento maior) -- CDP/GDP n<=150.

Compara cada variante de `run_phase2.py` com a selecao uniforme (`random`, de results/capacitated_norm; no E2, com o `random` do MESMO orcamento,
b2_random/b4_random) por hv_norm medio sobre replicas, Wilcoxon pareado + Holm (A12 = P(variante melhor)). Rode antes:
    python src/reference_sets.py build        (a referencia fixa inclui os runs de phase2_aos)

    python analyze_e1_e2.py --problem CDP [--part e1|e1b|e2|all] [--metric hv_norm] [--out results/analysis/e1_e2_CDP.json]
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
import run_phase2 as R2  # noqa: E402
from analyze_phase2 import a12, holm  # noqa: E402

E1 = [v for v in R2.HYPER if v.startswith("h_")]
E1B = ["fd_replace", "fd_replace_c30", "fd_replace_c300", "fd_scalar", "fd_hv", "ts_replace", "ucb_replace"]
E2 = [v for v in R2.GENMULT]
E3 = [v for v in R2.VARIANTS if v.startswith("c_")]
E15 = [v for v in R2.VARIANTS if v.startswith("frr_")]   # FRRMAB de Li et al. (2014)
E17 = ["init_a10", "init_a00"]   # alpha da inicializacao GRASP
E16 = [v for v in R2.VARIANTS if v.startswith("px_")]    # pool envenenado (8 operadores + 3 nocivos)
E16_CLEAN = {"px_ucbhv": "hypervolume", "px_dytshv": "dyts_hv", "px_ucbscal": "scalar", "px_dytsscal": "dyts_scalar", "px_ucbins": "ucb_insert", "px_fdrep": "fd_replace", "px_frr": "frr_default", "px_random": "random"}
E3_BASE = {"c_ucbins_w3": "ucb_insert", "c_ucbins_p3": "ucb_insert", "c_ucbins_w3p3": "ucb_insert", "c_tsins_w3p3": "ts_insert",
           "c_fdrep_w3p3": "fd_replace", "c_ucbhv_w3p3": "hypervolume"}   # contraparte NAO contextual (c_ucbsrel_w3p3: sem contraparte)
DEFAULT_OF = {   # configuracao padrao do catalogo correspondente a cada familia de E1 (para ver o efeito do hiperparametro)
    "h_ucbhv": "hypervolume", "h_tshv": "thompson_hv", "h_tssc": "thompson_scalar", "h_dytshv": "dyts_hv", "h_dytssc": "dyts_scalar",
    "h_ucbsrel": None, "h_ucbins": None}


def load_matrix(problem, names, reps, methods, metric, ref):
    norm = ROOT / "results/capacitated_norm" / problem / "runs"
    p2 = ROOT / "results/phase2_aos" / problem / "runs"
    out = {}
    for m in methods:
        d = p2 if m in R2.VARIANTS else norm
        vals = []
        for nm in names:
            fs = [d / f"{nm}_{m}_rep{k:02d}.json" for k in range(reps)]
            if not all(f.exists() for f in fs):
                vals = None; break
            vals.append(np.mean([RS.indicators(np.array(json.load(open(f))["pareto_front"]), ref[nm])[metric] for f in fs]))
        if vals is not None:
            out[m] = np.array(vals)
    return out


def compare(M, variants, base, hb=True):
    ps, rows = [], []
    for v in variants:
        d = M[v] - M[base]
        ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
        x, y = (M[v], M[base]) if hb else (-M[v], -M[base])
        rows.append((v, float(d.mean() * (1 if hb else -1)), a12(x, y)))
    return [(v, dm, aa, pa) for (v, dm, aa), pa in zip(rows, holm(ps))]


def show(title, res):
    print(f"\n{title}")
    for v, dm, aa, pa in res:
        print(f"   {v:20s} dif media {dm:+.4f}   A12={aa:.2f}   p_holm={pa:.4f}{' *' if pa < 0.05 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP"], default="CDP")
    ap.add_argument("--part", choices=["e1", "e1b", "e2", "e3", "e15", "e16", "e17", "all"], default="all")
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus", "eps_plus"], default="hv_norm")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hb = a.metric == "hv_norm"
    ref = RS.load(a.problem)
    names = [r["name"] for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv"))
             if r["problem"] == a.problem and int(r["n"]) <= 150]
    result = {}
    base_methods = ["random", "hypervolume", "thompson_hv", "thompson_scalar", "dyts_hv", "dyts_scalar", "scalar"]
    if a.part in ("e1", "all"):
        M = load_matrix(a.problem, names, a.reps, base_methods + E1, a.metric, ref)
        done = [v for v in E1 if v in M]
        print(f"=== E1 ({a.problem}, N={len(names)} instancias, {a.reps} replicas): {len(done)}/{len(E1)} variantes completas; metrica {a.metric}")
        print("referencia: media de", {m: round(float(M[m].mean()), 4) for m in base_methods if m in M})
        res = compare(M, done, "random", hb)
        show("--- cada variante vs `random` (uniforme):", res)
        # efeito do hiperparametro dentro de cada familia: vs configuracao padrao do catalogo
        for fam, dflt in DEFAULT_OF.items():
            vs = [v for v in done if v.startswith(fam + "_")]
            if vs and dflt in M:
                show(f"--- familia {fam} vs configuracao padrao ({dflt}):", compare(M, vs, dflt, hb))
        best = max(done, key=lambda v: M[v].mean()) if done else None
        if best:
            print(f"\nmelhor variante de E1 em {a.metric} medio: {best} ({M[best].mean():.4f}) vs random ({M['random'].mean():.4f})")
        result["e1"] = {v: dict(mean=float(M[v].mean()), diff_vs_random=dm, a12=aa, p_holm=pa) for v, dm, aa, pa in res}
    if a.part in ("e1b", "all"):
        M = load_matrix(a.problem, names, a.reps, base_methods + E1B, a.metric, ref)
        done = [v for v in E1B if v in M]
        if done:
            print(f"\n=== E1b ({a.problem}): {len(done)}/{len(E1B)} variantes completas")
            res = compare(M, done, "random", hb)
            show("--- DYTS fiel / recompensa replace vs `random`:", res)
            result["e1b"] = {v: dict(mean=float(M[v].mean()), diff_vs_random=dm, a12=aa, p_holm=pa) for v, dm, aa, pa in res}
    if a.part in ("e3", "all"):
        M = load_matrix(a.problem, names, a.reps, base_methods + E3 + sorted(set(E3_BASE.values()) - set(base_methods)), a.metric, ref)
        done = [v for v in E3 if v in M]
        if done:
            print(f"{chr(10)}=== E3 selecao contextual ({a.problem}): {len(done)}/{len(E3)} variantes completas")
            res = compare(M, done, "random", hb)
            show("--- contextual vs `random` (uniforme):", res)
            for v in done:
                b = E3_BASE.get(v)
                if b in M:
                    show(f"--- {v} vs contraparte nao contextual ({b}):", compare(M, [v], b, hb))
            result["e3"] = {v: dict(mean=float(M[v].mean()), diff_vs_random=dm, a12=aa, p_holm=pa) for v, dm, aa, pa in res}
    if a.part in ("e15", "all"):
        M = load_matrix(a.problem, names, a.reps, base_methods + E15, a.metric, ref)
        done = [v for v in E15 if v in M]
        if done:
            print(f"{chr(10)}=== E15 FRRMAB de Li et al. 2014 ({a.problem}): {len(done)}/{len(E15)} variantes completas")
            res = compare(M, done, "random", hb)
            show("--- FRRMAB vs `random` (uniforme):", res)
            result["e15"] = {v: dict(mean=float(M[v].mean()), diff_vs_random=dm, a12=aa, p_holm=pa) for v, dm, aa, pa in res}
    if a.part in ("e17", "all"):
        M = load_matrix(a.problem, names, a.reps, base_methods + E17, a.metric, ref)
        done = [v for v in E17 if v in M]
        if done:
            print(f"{chr(10)}=== E17 inicializacao GRASP ({a.problem}): {len(done)}/{len(E17)} variantes completas")
            show("--- alpha da inicializacao (1,0 = aleatoria; 0,0 = gulosa) vs padrao 0,3 (`random`):", compare(M, done, "random", hb))
            result["e17"] = {v: dict(mean=float(M[v].mean()), diff_vs_random=dm, a12=aa, p_holm=pa) for v, dm, aa, pa in compare(M, done, "random", hb)}
    if a.part in ("e16", "all"):
        M = load_matrix(a.problem, names, a.reps, base_methods + E16 + ["ucb_insert", "fd_replace", "frr_default"], a.metric, ref)
        done = [v for v in E16 if v in M]
        if done and "px_random" in M:
            print(f"{chr(10)}=== E16 pool envenenado ({a.problem}): {len(done)}/{len(E16)} variantes completas")
            show("--- pool envenenado: uniforme vs uniforme do pool limpo:", compare(M, ["px_random"], "random", hb))
            show("--- pool envenenado: cada seletor vs uniforme no MESMO pool envenenado (px_random):", compare(M, [v for v in done if v != "px_random"], "px_random", hb))
            show("--- pool envenenado: seletor vs uniforme do pool LIMPO (random):", compare(M, [v for v in done if v != "px_random"], "random", hb))
            for v in done:
                c = E16_CLEAN.get(v)
                if v != "px_random" and c in M:
                    show(f"--- {v} vs sua contraparte no pool limpo ({c}):", compare(M, [v], c, hb))
            result["e16"] = {"poisoned_uniform_vs_clean_uniform": compare(M, ["px_random"], "random", hb),
                             "selector_vs_poisoned_uniform": compare(M, [v for v in done if v != "px_random"], "px_random", hb),
                             "selector_vs_clean_uniform": compare(M, [v for v in done if v != "px_random"], "random", hb),
                             "means": {v: float(M[v].mean()) for v in M}}
    if a.part in ("e2", "all"):
        for mult in (2, 4):
            vs = [f"b{mult}_random", f"b{mult}_dytshv", f"b{mult}_ucbhv"]
            M = load_matrix(a.problem, names, a.reps, ["random", "dyts_hv", "hypervolume"] + vs, a.metric, ref)
            if f"b{mult}_random" in M:
                print(f"\n=== E2 orcamento x{mult} ({a.problem}): medias {{{', '.join(f'{k}: {M[k].mean():.4f}' for k in ['random', 'dyts_hv', 'hypervolume'] + vs if k in M)}}}")
                show(f"--- mesmo orcamento x{mult}: seletores vs uniforme (b{mult}_random):", compare(M, [v for v in vs[1:] if v in M], vs[0], hb))
                show(f"--- ganho de orcamento (x{mult} vs x1), uniforme:", compare(M, [vs[0]], "random", hb))
                result[f"e2_x{mult}"] = {k: float(M[k].mean()) for k in M}
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print("\nsalvo:", a.out)


if __name__ == "__main__":
    main()
