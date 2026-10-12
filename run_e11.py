"""E11 -- teste: baselines AJUSTADOS (run_e11_tune.py) e Tabu/VNS de Colmenar vs MOEA/D-Uniform, a tempo igual e a 5x o tempo.

Para cada (instancia de TESTE, replica) o tempo-alvo T e o do MOEA/D-Uniform normalizado (CDP/GDP: o do E4, mesma amostra; MDP: rodado aqui).
Modos gravados (results/e11_strong_baselines/<celula>/runs/<inst>_<modo>_rep<kk>.json):
  t_<m>  : baseline m ajustado, limite T (m em nsga2, grasp, ts, vns, sms)        l_<m> : ajustado, limite 5T (n<=150: todas as tarefas;
  d_ts, d_vns : Tabu/VNS com a configuracao do irace de Colmenar et al., limite T; d_sms: SMS-EMOA padrao    n=500: so replica 0 e instancias de indice par)
  MDP apenas: d_uniform (MOEA/D-Uniform, define T), d_nsga2, d_grasp, d_grasp_pr (padrao do projeto, limite T)
Todos os limites CONTAM a inicializacao. Retomavel (um arquivo por execucao).

    python run_e11.py [--workers 8] [--cells ...] [--reps 3] [--no-long]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import e11_common as C  # noqa: E402

LONG_FACTOR = 5.0


def runs_dir(cell):
    d = C.OUT / cell / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(out: Path, d: dict):
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(out)


def _uniform_task(args):
    cell, problem, row, rep = args
    out = runs_dir(cell) / f"{row['name']}_d_uniform_rep{rep:02d}.json"
    if out.exists():
        return str(out)
    inst = C.load_instance(problem, row)
    pf, el = C.moead_uniform_time(inst, rep)
    _write(out, dict(instance=row["name"], rep=rep, mode="d_uniform", time_target_s=el, elapsed_s=el, pareto_size=int(len(pf)), pareto_front=pf.tolist()))
    return str(out)


def target_time(cell, problem, name, rep):
    if problem == "MDP":
        f = runs_dir(cell) / f"{name}_d_uniform_rep{rep:02d}.json"
        return float(json.load(open(f))["time_target_s"]) if f.exists() else None
    return C.e4_time(problem, name, rep)


def _group_task(args):
    """Um grupo = (instancia, replica) com varios modos: a instancia e carregada UMA vez (o carregamento pesa para n pequeno)."""
    cell, problem, row, rep, items = args
    inst = None
    done = 0
    for mode, factor, T, params in items:
        out = runs_dir(cell) / f"{row['name']}_{mode}_rep{rep:02d}.json"
        if out.exists():
            continue
        if inst is None:
            inst = C.load_instance(problem, row)
        method = mode.split("_", 1)[1]
        method = "grasp" if method.startswith("grasp") else method
        front, el, it = C.run_method(method, params, inst, C.seed_for("e11", mode, row["name"], rep), factor * T)
        _write(out, dict(instance=row["name"], rep=rep, mode=mode, time_target_s=T, limit_s=factor * T, elapsed_s=el, iterations=it,
                         pareto_size=int(len(front)), pareto_front=front.tolist(), params=params))
        done += 1
    return row["name"], rep, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cells", default="CDP-S,GDP-S,MDP-S,CDP-L,GDP-L,MDP-L")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-long", action="store_true")
    ap.add_argument("--long-reps", type=int, default=1, help="celulas S: execucoes de 5T so nas primeiras N replicas (padrao 1)")
    ap.add_argument("--limit", type=int, default=0, help="so as N primeiras instancias de teste de cada celula (depuracao)")
    a = ap.parse_args()
    allc = C.cells()
    sel = a.cells.split(",")
    # fase A: MOEA/D-Uniform das celulas MDP (define T)
    ua = []
    for c in sel:
        if allc[c]["problem"] == "MDP":
            test = allc[c]["test"][: a.limit] if a.limit else allc[c]["test"]
            ua += [(c, "MDP", r, k) for r in test for k in range(a.reps)]
    if ua:
        print(f"[fase A] {len(ua)} execucoes do MOEA/D-Uniform (MDP) para fixar T", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, _ in enumerate(ex.map(_uniform_task, ua, chunksize=1), 1):
                if i % 20 == 0:
                    print(f"   [{i}/{len(ua)}]", flush=True)
    # fase B: todos os modos como tarefas independentes
    jobs = []
    for c in sel:
        prob = allc[c]["problem"]
        test = allc[c]["test"][: a.limit] if a.limit else allc[c]["test"]
        tuned = {}
        for m in C.METHODS:
            f = C.OUT / "tuning" / c / f"tuned_{m}.json"
            if not f.exists():
                print(f"[aviso] {c}: sem ajuste de {m} ({f}); celula ignorada", flush=True)
                break
            tuned[m] = json.load(open(f))["best"]
        else:
            large = c.endswith("-L")
            for ii, row in enumerate(test):
                for k in range(a.reps):
                    T = target_time(c, prob, row["name"], k)
                    if T is None:
                        print(f"[aviso] sem T para {row['name']} rep{k}", flush=True)
                        continue
                    long_ok = (not a.no_long) and ((k < a.long_reps) if not large else (k == 0 and ii % 2 == 0))
                    items = []
                    for m in C.METHODS:
                        items.append((f"t_{m}", 1.0, T, tuned[m]))
                    items.append(("d_ts", 1.0, T, C.DEFAULT["ts"][0]))
                    items.append(("d_vns", 1.0, T, C.DEFAULT["vns"][0]))
                    items.append(("d_sms", 1.0, T, C.DEFAULT["sms"][0]))
                    if prob == "MDP":
                        items.append(("d_nsga2", 1.0, T, C.DEFAULT["nsga2"][0]))
                        items.append(("d_grasp", 1.0, T, C.DEFAULT["grasp"][0]))
                        items.append(("d_grasp_pr", 1.0, T, C.DEFAULT["grasp"][1]))
                    if long_ok:
                        for m in C.METHODS:
                            items.append((f"l_{m}", LONG_FACTOR, T, tuned[m]))
                    cost = sum(f * T for _, f, T, _ in items)
                    jobs.append((cost, (c, prob, row, k, items)))
    jobs.sort(key=lambda t: -t[0])
    print(f"[fase B] {len(jobs)} grupos (instancia x replica)", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for _ in ex.map(_group_task, [j[1] for j in jobs], chunksize=1):
            done += 1
            if done % 25 == 0:
                print(f"   [{done}/{len(jobs)}]", flush=True)
    print("FIM E11", flush=True)


if __name__ == "__main__":
    main()
