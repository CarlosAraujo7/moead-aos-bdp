"""E11 -- ajuste dos baselines (random search + successive halving) por celula, em instancias de TREINO disjuntas das de teste.

Para cada (celula, metodo): 16 configuracoes candidatas (as padrao + amostras aleatorias do espaco de e11_common.sample_params);
estagio 1: todas nas 2 primeiras instancias de treino; estagio 2: as 6 melhores nas 3 seguintes; estagio 3: as 2 melhores nas 3 ultimas;
vence a de maior hv_norm medio sobre as instancias avaliadas. Cada avaliacao roda o metodo com limite de tempo T_i = tempo do
MOEA/D-Uniform na instancia (o MOEA/D NAO e ajustado: a comparacao e conservadora para ele). Retomavel (um arquivo por avaliacao).

    python run_e11_tune.py [--workers 8] [--cells CDP-S,GDP-S,...] [--methods nsga2,grasp,ts,vns]
Saida: results/e11_strong_baselines/tuning/<celula>/{train_T.json, evals/*.json, tuned_<metodo>.json}
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import e11_common as C  # noqa: E402

N_CAND = 16
STAGES = [(None, (0, 2)), (6, (2, 5)), (2, (5, 8))]     # (quantos sobrevivem a este estagio como entrada, fatia de instancias)


def tdir(cell):
    d = C.OUT / "tuning" / cell
    (d / "evals").mkdir(parents=True, exist_ok=True)
    return d


def candidates(cell, method):
    rng = np.random.default_rng(C.seed_for("cands", cell, method))
    cands = [dict(p) for p in C.DEFAULT[method]]
    while len(cands) < N_CAND:
        cands.append(C.sample_params(method, rng))
    return cands


# ---------------------------------------------------------------- tempos-alvo do treino
def _t_task(args):
    cell, problem, row = args
    d = tdir(cell)
    f = d / "train_T.json"
    inst = C.load_instance(problem, row)
    ts = []
    for rep in range(3 if problem != "MDP" else 1):
        t = C.e4_time(problem, row["name"], rep) if problem != "MDP" else None
        if t is not None:
            ts.append(t)
    if ts:
        return row["name"], float(np.median(ts)), "e4"
    _, el = C.moead_uniform_time(inst, 0)
    return row["name"], float(el), "fresco"


def prelude(cells_sel, workers):
    allc = C.cells()
    tasks = []
    for cell in cells_sel:
        f = tdir(cell) / "train_T.json"
        have = json.load(open(f)) if f.exists() else {}
        for row in allc[cell]["train"]:
            if row["name"] not in have:
                tasks.append((cell, allc[cell]["problem"], row))
    if tasks:
        print(f"[prelude] medindo T do MOEA/D-Uniform em {len(tasks)} instancias de treino", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (cell, _, _), (name, t, src) in zip(tasks, ex.map(_t_task, tasks, chunksize=1)):
                f = tdir(cell) / "train_T.json"
                have = json.load(open(f)) if f.exists() else {}
                have[name] = t
                f.write_text(json.dumps(have, indent=1))
                print(f"   {cell} {name}: T={t:.1f}s ({src})", flush=True)


# ---------------------------------------------------------------- avaliacao
_REF = {}


def _eval_task(args):
    cell, problem, method, ci, params, row, T = args
    d = tdir(cell) / "evals"
    out = d / f"{method}_c{ci:02d}_{row['name']}.json"
    if out.exists():
        return str(out), "skip"
    import reference_sets as RS
    if problem not in _REF:
        _REF[problem] = RS.load(problem)
    inst = C.load_instance(problem, row)
    front, el, it = C.run_method(method, params, inst, C.seed_for("tune", cell, method, ci, row["name"]), T)
    hv = C.hv_of(front, _REF[problem][row["name"]])
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(cell=cell, method=method, cand=ci, instance=row["name"], hv_norm=hv, elapsed_s=el, limit_s=T, iterations=it,
                                   pareto_size=int(len(front)), params=params)))
    tmp.replace(out)
    return str(out), f"hv={hv:.3f}"


def _scores(cell, method, ci, names):
    d = tdir(cell) / "evals"
    v = []
    for n in names:
        f = d / f"{method}_c{ci:02d}_{n}.json"
        if f.exists():
            v.append(json.load(open(f))["hv_norm"])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cells", default="CDP-S,GDP-S,MDP-S,CDP-L,GDP-L,MDP-L")
    ap.add_argument("--methods", default=",".join(C.METHODS))
    a = ap.parse_args()
    cells_sel = a.cells.split(",")
    methods = a.methods.split(",")
    allc = C.cells()
    prelude(cells_sel, a.workers)
    cand = {(c, m): candidates(c, m) for c in cells_sel for m in methods}
    alive = {(c, m): list(range(N_CAND)) for c in cells_sel for m in methods}
    for si, (keep, (lo, hi)) in enumerate(STAGES):
        tasks = []
        for (c, m), lst in alive.items():
            train = allc[c]["train"]
            T = json.load(open(tdir(c) / "train_T.json"))
            for ci in lst:
                for row in train[lo:hi]:
                    tasks.append((T[row["name"]], (c, allc[c]["problem"], m, ci, cand[(c, m)][ci], row, T[row["name"]])))
        tasks.sort(key=lambda t: -t[0])
        print(f"[estagio {si + 1}/{len(STAGES)}] {len(tasks)} avaliacoes", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for _ in ex.map(_eval_task, [t[1] for t in tasks], chunksize=1):
                done += 1
                if done % 50 == 0:
                    print(f"   [{done}/{len(tasks)}]", flush=True)
        # sobreviventes do proximo estagio
        if si + 1 < len(STAGES):
            nxt = STAGES[si + 1][0]
            for (c, m), lst in alive.items():
                names = [r["name"] for r in allc[c]["train"][: STAGES[si][1][1]]]
                ranked = sorted(lst, key=lambda ci: -statistics.mean(_scores(c, m, ci, names)))
                alive[(c, m)] = ranked[:nxt]
    # resultado final
    for (c, m), lst in alive.items():
        train = allc[c]["train"]
        names = [r["name"] for r in train[:STAGES[-1][1][1]]]
        finals = sorted(lst, key=lambda ci: -statistics.mean(_scores(c, m, ci, names)))
        best = finals[0]
        table = {}
        for ci in range(N_CAND):
            sc = _scores(c, m, ci, names)
            if sc:
                table[ci] = dict(params=cand[(c, m)][ci], n_eval=len(sc), mean_hv=float(statistics.mean(sc)))
        dflt = [ci for ci in range(len(C.DEFAULT[m]))]
        d_first2 = {ci: float(statistics.mean(_scores(c, m, ci, [r["name"] for r in train[:2]]))) for ci in dflt}
        (tdir(c) / f"tuned_{m}.json").write_text(json.dumps(dict(
            cell=c, method=m, best=cand[(c, m)][best], best_index=best, best_mean_train_hv=float(statistics.mean(_scores(c, m, best, names))),
            default_first2_mean_hv=d_first2, train=[r["name"] for r in train], table=table), indent=1))
        print(f"{c:6s} {m:5s} melhor c{best:02d} hv_treino={statistics.mean(_scores(c, m, best, names)):.3f}  params={cand[(c, m)][best]}", flush=True)
    print("FIM E11-TUNE", flush=True)


if __name__ == "__main__":
    main()
