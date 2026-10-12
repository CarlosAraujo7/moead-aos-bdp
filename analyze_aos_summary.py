"""Resumo por FAMILIA das tentativas de reparar a selecao adaptativa (Secao 6.4 do manuscrito): CDP e GDP, n<=150, 3 replicas, referencia congelada.

Para cada familia de variantes de `run_phase2.py`, compara cada variante com a selecao uniforme (`random` de capacitated_norm; nas familias de
orcamento, com o `random` do mesmo orcamento): diferenca media de hv_norm, Wilcoxon pareado sobre instancias, Holm DENTRO da familia, Â12.
Saida: results/analysis/aos_summary.json (usado por make_manuscript_tables.py) e tabela no terminal.

    python analyze_aos_summary.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import reference_sets as RS  # noqa: E402
import run_phase2 as R2  # noqa: E402
from analyze_e1_e2 import E1, E1B, E3, E15, compare, load_matrix  # noqa: E402

REWARD = ["ucb_insert", "ts_insert", "dyts_insert", "ucb_insert_cost", "ucb_srel_cost", "ucb_insert_mix50", "ts_insert_mix50", "ucb_srel_mix50"]
FAMILIES = [("rewards", "Other rewards, cost-aware scaling, exploration floor", REWARD),
            ("dyts", "Faithful DYTS and the `replace` reward", E1B),
            ("context", "Contextual selection (by weight and/or phase)", E3),
            ("hyper", "Hyper-parameter sweep (c, window)", E1),
            ("frrmab", "FRRMAB (Li et al., 2014)", E15)]


def main():
    out = {}
    for prob in ("CDP", "GDP"):
        ref = RS.load(prob)
        names = [r["name"] for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == prob and int(r["n"]) <= 150]
        res = {}
        for key, label, vs in FAMILIES:
            M = load_matrix(prob, names, 3, ["random"] + vs, "hv_norm", ref)
            done = [v for v in vs if v in M]
            if not done:
                continue
            c = compare(M, done, "random", True)
            diffs = [x[1] for x in c]
            res[key] = dict(label=label, n_variants=len(done), n_expected=len(vs), diff_min=min(diffs), diff_max=max(diffs),
                            better=sum(1 for x in c if x[3] < 0.05 and x[1] > 0), worse=sum(1 for x in c if x[3] < 0.05 and x[1] < 0),
                            a12_max=max(x[2] for x in c), variants={x[0]: dict(diff=x[1], a12=x[2], p_holm=x[3]) for x in c})
        # orcamento maior (mesmo orcamento para o seletor e o uniforme)
        rows = []
        for mult in (2, 4):
            M = load_matrix(prob, names, 3, [f"b{mult}_random", f"b{mult}_dytshv", f"b{mult}_ucbhv"], "hv_norm", ref)
            if f"b{mult}_random" in M:
                rows += [(f"x{mult}:{v}", d, a, p) for v, d, a, p in compare(M, [f"b{mult}_dytshv", f"b{mult}_ucbhv"], f"b{mult}_random", True)]
        if rows:
            res["budget"] = dict(label="More generations (×2, ×4), same budget for selector and uniform", n_variants=len(rows), n_expected=4,
                                 diff_min=min(r[1] for r in rows), diff_max=max(r[1] for r in rows),
                                 better=sum(1 for r in rows if r[3] < 0.05 and r[1] > 0), worse=sum(1 for r in rows if r[3] < 0.05 and r[1] < 0),
                                 a12_max=max(r[2] for r in rows), variants={r[0]: dict(diff=r[1], a12=r[2], p_holm=r[3]) for r in rows})
        out[prob] = res
        print(f"\n=== {prob} (N={len(names)} instancias, 3 replicas)")
        for key, d in res.items():
            print(f"   {d['label'][:62]:62s} {d['n_variants']:2d}/{d['n_expected']:2d} variantes  dif [{d['diff_min']:+.3f}, {d['diff_max']:+.3f}]  melhor {d['better']}  pior {d['worse']}  A12max {d['a12_max']:.2f}")
    (ROOT / "results" / "analysis" / "aos_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
