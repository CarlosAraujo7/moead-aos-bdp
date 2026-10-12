"""E11 -- baselines mais fortes (ajustados + Tabu Search/VNS no estilo de Colmenar et al. 2018): definicoes comuns.

Celulas (problema x classe de tamanho), particao treino/teste, espacos de parametros, configuracoes padrao e o executor `run_method`.
Tudo e deterministico (sem sorteio fora de sementes derivadas por crc32). Usado por run_e11_tune.py, run_e11.py e analyze_e11.py.

Protocolo (ver CLAUDE.md §2s, E11):
  * Cada baseline (NSGA-II, GRASP[-PR], TS e VNS de Colmenar) e AJUSTADO por celula com busca aleatoria + successive halving em
    instancias de TREINO disjuntas das de TESTE, com o orcamento de tempo T_i = tempo do MOEA/D-Uniform na instancia (sem ajustar o MOEA/D).
  * Teste: por (instancia, replica) o tempo-alvo T e o do MOEA/D-Uniform medido no mesmo regime (CDP/GDP: E4; MDP: rodado aqui);
    baselines com limite de T (todos contando a inicializacao) e, em parte das tarefas, 5T.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "e11_strong_baselines"
METHODS = ["nsga2", "grasp", "ts", "vns", "moead", "sms"]
LABEL = {"nsga2": "NSGA-II", "grasp": "GRASP(-PR)", "ts": "Tabu (Colmenar)", "vns": "VNS (Colmenar)", "moead": "MOEA/D-Uniform", "sms": "SMS-EMOA"}


# ----------------------------------------------------------------------------- celulas
def _cat(problem):
    f = "catalog.csv" if problem == "MDP" else "catalog_capacitated.csv"
    rows = list(csv.DictReader(open(ROOT / "instances" / f)))
    return rows if problem == "MDP" else [r for r in rows if r["problem"] == problem]


def _pick(rows, idx):
    return [rows[i] for i in idx]


def cells() -> dict:
    """{cell: dict(problem, train=[rows], test=[rows], long_every=k)}. Linhas do catalogo (name/family/n/filepath)."""
    out = {}
    for prob in ("CDP", "GDP"):
        rows = _cat(prob)
        small = [r for r in rows if int(r["n"]) <= 150]
        big = [r for r in rows if int(r["n"]) == 500]
        tr_idx = list(range(3, len(small), 7))[:8]
        out[f"{prob}-S"] = dict(problem=prob, train=_pick(small, tr_idx), test=[r for i, r in enumerate(small) if i not in tr_idx])
        # n=500: teste = amostra do E4 (CDP: [0::4]+[1::4]; GDP: [0::4]+[3::4]); treino sai do resto
        if prob == "CDP":
            test_idx = sorted(set(range(0, len(big), 4)) | set(range(1, len(big), 4)))
        else:
            test_idx = sorted(set(range(0, len(big), 4)) | set(range(3, len(big), 4)))
        if prob == "GDP":
            # treino com as MESMAS combinacoes (b02k02 / b03k03) do teste: 8 instancias saem da amostra do E4 (teste fica com 32)
            tr_idx = test_idx[::5][:8]
            test_idx = [i for i in test_idx if i not in tr_idx]
        else:
            rest = [i for i in range(len(big)) if i not in test_idx]
            tr_idx = rest[::3]
        out[f"{prob}-L"] = dict(problem=prob, train=_pick(big, tr_idx), test=_pick(big, test_idx))
    mdp = _cat("MDP")
    small = [r for r in mdp if (r["family"] == "SOM-a" and int(r["n"]) >= 100) or (r["family"] == "MDG-a") or (r["family"] == "MDG-b" and int(r["n"]) == 100)]
    big = [r for r in mdp if (r["family"] == "GKD-c" and int(r["n"]) == 500) or (r["family"] == "MDG-b" and int(r["n"]) == 500)]
    tr_idx = list(range(3, len(small), 7))[:8]
    out["MDP-S"] = dict(problem="MDP", train=_pick(small, tr_idx), test=[r for i, r in enumerate(small) if i not in tr_idx])
    tr_idx = list(range(1, len(big), 4))[:8]
    out["MDP-L"] = dict(problem="MDP", train=_pick(big, tr_idx), test=[r for i, r in enumerate(big) if i not in tr_idx])
    return out


def load_instance(problem, row):
    if problem == "MDP":
        from problem import MDPInstance
        inst = MDPInstance(ROOT / "instances" / row["filepath"])
    else:
        from capacitated_problem import CDPInstance, GDPInstance
        inst = (CDPInstance if problem == "CDP" else GDPInstance)(ROOT / "instances" / row["filepath"])
    inst.family, inst.name = row["family"], row["name"]
    return inst


def seed_for(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) % (2**31)


# ----------------------------------------------------------------------------- parametros
DEFAULT = {
    "nsga2": [dict(pop_size=None, crossover_prob=0.9, mutation_prob=0.1, init_mode="grasp", grasp_alpha=0.2)],
    # SMS-EMOA (estado estacionario, selecao por contribuicao de hipervolume; src/smsemoa.py): mesmos operadores e padroes do NSGA-II
    "sms": [dict(pop_size=None, crossover_prob=0.9, mutation_prob=0.1, init_mode="grasp", grasp_alpha=0.2)],
    "grasp": [dict(alpha=0.2, local_search_iter=1, path_relink=False, pr_pairs=5), dict(alpha=0.2, local_search_iter=1, path_relink=True, pr_pairs=5)],
    # Configuracoes do irace de Colmenar et al. 2018 (Tabela 6, conferida na versao publicada): TS alpha 0,94, CAltBwS, LSAlt, T=64 (%), maxItNoImpr 70;
    # VNS alpha 0,19, CAltInS, LSPR, kmax=29,49 (% do tamanho da solucao), maxIters 7. T e kmax sao PORCENTAGENS no artigo.
    "ts": [dict(alpha=0.94, constructive="altbws", local_search="alt", tenure_frac=0.64, max_no_improve=70)],
    "vns": [dict(alpha=0.19, constructive="altins", local_search="pr", kmax_frac=0.2949, restart_after=7)],
    # MOEA/D-Uniform: n_weights/nbr None = regra de budget_for(n) (a configuracao padrao do projeto)
    "moead": [dict(n_weights=None, nbr=None, grasp_alpha=0.3)],
    # Os dois metodos de Colmenar et al. 2018 que faltavam (E22), com a configuracao do irace deles (Tabela 6): SPEA2 pop 340, mutacao 0,14, cruzamento 0,15;
    # Iterated Greedy alpha 0,95, CAltBwS, destruicao de 55,39% (o numero de geracoes/destruicoes vira o limite de tempo)
    "spea2": [dict(pop_size=340, crossover_prob=0.15, mutation_prob=0.14)],
    "ig": [dict(alpha=0.95, constructive="altbws", pct=55.39)],
}


def sample_params(method: str, rng: np.random.Generator) -> dict:
    lg = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if method in ("nsga2", "sms"):
        return dict(pop_size=int(round(lg(20, 400))), crossover_prob=float(rng.uniform(0.2, 1.0)), mutation_prob=lg(0.02, 0.9),
                    init_mode=str(rng.choice(["grasp", "random"])), grasp_alpha=float(rng.uniform(0.05, 0.5)))
    if method == "grasp":
        return dict(alpha=float(rng.uniform(0.0, 0.8)), local_search_iter=int(rng.choice([0, 1, 2, 3])), path_relink=bool(rng.integers(2)),
                    pr_pairs=int(rng.choice([2, 5, 10, 20])))
    if method == "ts":
        return dict(alpha=float(rng.uniform(0.3, 1.0)), constructive=str(rng.choice(["altbws", "altins", "rnd"])),
                    local_search=str(rng.choice(["pr", "alt", "dom"])), tenure=int(round(lg(5, 200))), max_no_improve=int(round(lg(5, 300))))
    if method == "vns":
        return dict(alpha=float(rng.uniform(0.0, 1.0)), constructive=str(rng.choice(["altbws", "altins", "rnd"])),
                    local_search=str(rng.choice(["pr", "alt", "dom"])), kmax=int(round(lg(2, 40))), restart_after=int(rng.integers(1, 21)))
    if method == "moead":
        nw = int(round(lg(10, 200)))
        return dict(n_weights=nw, nbr=max(3, int(round(float(rng.uniform(0.05, 0.4)) * nw))), grasp_alpha=float(rng.uniform(0.05, 0.8)))
    raise ValueError(method)


# ----------------------------------------------------------------------------- execucao
def run_method(method: str, params: dict, inst, seed: int, time_limit_s: float):
    """Roda um baseline com limite de tempo de parede que CONTA a inicializacao. Retorna (frente Nx2, tempo total, iteracoes)."""
    from experiment import budget_for
    t0 = time.perf_counter()
    if method == "nsga2":
        from nsga2 import NSGA2, NSGA2Config
        pop = params.get("pop_size") or budget_for(inst.n)["nsga2_pop"]
        r = NSGA2(inst, NSGA2Config(pop_size=int(pop), max_gen=10**7, crossover_prob=params["crossover_prob"],
                                    mutation_prob=params["mutation_prob"], init_mode=params["init_mode"], grasp_alpha=params["grasp_alpha"],
                                    seed=seed, time_limit_s=time_limit_s, include_init_time=True)).run()
        it = r.n_gen_executed
    elif method == "sms":
        from smsemoa import SMSEMOA, SMSEMOAConfig
        pop = params.get("pop_size") or budget_for(inst.n)["nsga2_pop"]
        r = SMSEMOA(inst, SMSEMOAConfig(pop_size=int(pop), crossover_prob=params["crossover_prob"], mutation_prob=params["mutation_prob"],
                                        init_mode=params["init_mode"], grasp_alpha=params["grasp_alpha"], seed=seed, time_limit_s=time_limit_s)).run()
        it = r.n_evals
    elif method == "grasp":
        from grasp_mo import GRASPMO, GRASPMOConfig
        r = GRASPMO(inst, GRASPMOConfig(max_iter=10**7, alpha=params["alpha"], local_search_iter=params["local_search_iter"],
                                        path_relink=params["path_relink"], pr_pairs=params["pr_pairs"], seed=seed, time_limit_s=time_limit_s)).run()
        it = r.n_gen_executed
    elif method == "moead":
        from moead_aos import MOEAD_AOS, MOEADConfig
        b = budget_for(inst.n)
        r = MOEAD_AOS(inst, MOEADConfig(n_weights=int(params.get("n_weights") or b["T"]), neighborhood_size=int(params.get("nbr") or b["nbr"]),
                                        max_gen=10**7, operator_selection="random", reward_mode="scalar", normalize_objectives=True,
                                        grasp_alpha=params["grasp_alpha"], seed=seed, time_limit_s=time_limit_s,
                                        polish_frac=float(params.get("polish_frac", 0.0)))).run()
        it = 0
    elif method == "spea2":
        from spea2 import SPEA2, SPEA2Config
        r = SPEA2(inst, SPEA2Config(pop_size=int(params["pop_size"]), crossover_prob=params["crossover_prob"], mutation_prob=params["mutation_prob"],
                                    seed=seed, time_limit_s=time_limit_s)).run()
        it = r.n_gen_executed
    elif method == "ig":
        from bo_trajectory import BOTrajConfig, BOTrajectory
        r = BOTrajectory(inst, BOTrajConfig(method="ig", seed=seed, time_limit_s=time_limit_s, **params)).run()
        it = r.n_iter_executed
    else:
        from bo_trajectory import BOTrajConfig, BOTrajectory
        base = dict(method=("tabu" if method == "ts" else "vns"), seed=seed, time_limit_s=time_limit_s)
        r = BOTrajectory(inst, BOTrajConfig(**base, **params)).run()
        it = r.n_iter_executed
    return np.asarray(r.pareto_front, float).reshape(-1, 2), time.perf_counter() - t0, int(it)


def hv_of(front, ref) -> float:
    import reference_sets as RS
    return float(RS.indicators(np.asarray(front, float).reshape(-1, 2), ref)["hv_norm"])


def moead_uniform_time(inst, rep: int):
    """Roda o MOEA/D-Uniform normalizado (orcamento padrao) e devolve (frente, tempo). Mesma semente do experimento principal."""
    from experiment import ExperimentConfig, run_one
    pf, el, *_ = run_one(inst, ExperimentConfig(normalize_objectives=True), "random", rep)
    return np.asarray(pf, float), float(el)


def e4_time(problem: str, name: str, rep: int):
    f = OUT.parent / "e4_equal_time" / problem / "runs" / f"{name}_e4_uniform_rep{rep:02d}.json"
    return float(json.load(open(f))["time_target_s"]) if f.exists() else None


if __name__ == "__main__":
    for c, d in cells().items():
        print(f"{c}: treino {len(d['train'])}, teste {len(d['test'])}")
        print("   treino:", [r["name"] for r in d["train"]])
