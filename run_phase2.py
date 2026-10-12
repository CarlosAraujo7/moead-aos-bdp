"""Fase 2 do plano: variantes de AOS com recompensas novas (CLAUDE.md §2p). Regime normalizado.

Variantes (selecao, recompensa, cost_aware):
  ucb_insert, ts_insert, dyts_insert      -- recompensa "entrou no arquivo nao-dominado" (nao satura como o HV)
  ucb_insert_cost                         -- idem + recompensa dividida pelo custo relativo do operador
  ucb_srel_cost                           -- melhora relativa do Tchebycheff em [0,1] + custo relativo
Grava em results/phase2_aos/<PROBLEMA>/runs/<inst>_<variante>_rep<kk>.json (retomavel; nao toca o catalogo).
Comparar com `random` e os demais de results/capacitated_norm (mesmas instancias).

    python run_phase2.py --problem CDP --max-n 150 --reps 3 --workers 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

VARIANTS = {   # nome: (selecao, recompensa, cost_aware, explore_mix)
    "ucb_insert":      ("ucb", "insert", False, 0.0),
    "ts_insert":       ("thompson", "insert", False, 0.0),
    "dyts_insert":     ("dyts", "insert", False, 0.0),
    "ucb_insert_cost": ("ucb", "insert", True, 0.0),
    "ucb_srel_cost":   ("ucb", "scalar_rel", True, 0.0),
    # rodada 2 (piso de exploracao + referencia na mesma condicao de carga)
    "ucb_insert_mix50": ("ucb", "insert", False, 0.5),
    "ts_insert_mix50":  ("thompson", "insert", False, 0.5),
    "ucb_srel_mix50":   ("ucb", "scalar_rel", False, 0.5),
    "random_ref":       ("random", "scalar", False, 0.0),
}
# rodada 3: pools enxutos com selecao uniforme (indices: 0 random_swap_1x1, 1 best_swap_1x1, 2 random_swap_2x2, 3 grasp_greedy,
# 4 biobj_best_swap, 5 maxmin_swap, 6 large_perturbation, 7 subset_crossover). Valor = operadores EXCLUIDOS.
POOLS = {
    "pool7_no_biobj":            (4,),
    "pool6_no_biobj_bestswap":   (4, 1),
    "pool6_no_biobj_grasp":      (4, 3),
    "pool5_no_biobj_bs_grasp":   (4, 1, 3),
}
for _n, _ex in POOLS.items():
    VARIANTS[_n] = ("random", "scalar", False, 0.0)
# rodada 4: ancoragem de MaxMin (src/anchor.py) com 2000 it x 3 reinicios; valor = (iteracoes, reinicios)
ANCHORS = {"random_anchor": (2000, 3), "dyts_hv_anchor": (2000, 3), "random_anchor6k": (6000, 3)}
VARIANTS["random_anchor"] = ("random", "scalar", False, 0.0)
VARIANTS["dyts_hv_anchor"] = ("dyts", "hypervolume", False, 0.0)
VARIANTS["random_anchor6k"] = ("random", "scalar", False, 0.0)
# E7: ancora com motor SBTS-like (src/sbts_anchor.py). "both" = tabu historico + SBTS (os dois resultados entram no arquivo).
# Orcamento por tamanho: n<=150: tabu 2000x3, SBTS 3000 it; n>150: tabu 6000x3, SBTS 30000 it.
ANCHOR_V2 = {"random_sbts": "sbts", "random_both": "both", "dyts_hv_sbts": "sbts", "dyts_hv_both": "both"}
for _v in ANCHOR_V2:
    VARIANTS[_v] = ("dyts", "hypervolume", False, 0.0) if _v.startswith("dyts_hv") else ("random", "scalar", False, 0.0)


def anchor_kwargs(variant, n):
    if variant in ANCHOR_V2:
        small = n <= 150
        return dict(anchor_iters=2000 if small else 6000, anchor_restarts=3, anchor_method=ANCHOR_V2[variant], sbts_iters=3000 if small else 30000)
    it, rs = ANCHORS.get(variant, (0, 3))
    return dict(anchor_iters=it, anchor_restarts=rs)

# E1 (varredura de hiperparametros dos bandits; base: ucb_c=0.5, ucb_window=50; a janela vale p/ UCB, Thompson e DYTS)
HYPER = {}
GENMULT = {}


def _hp(name, sel, rw, **kw):
    VARIANTS[name] = (sel, rw, False, 0.0)
    HYPER[name] = kw


for _c in (0.05, 2.0):
    _hp(f"h_ucbhv_c{str(_c).replace('.', '')}", "ucb", "hypervolume", ucb_c=_c)
for _w in (20, 200):
    _hp(f"h_ucbhv_w{_w}", "ucb", "hypervolume", ucb_window=_w)
    _hp(f"h_tshv_w{_w}", "thompson", "hypervolume", ucb_window=_w)
    _hp(f"h_tssc_w{_w}", "thompson", "scalar", ucb_window=_w)
    _hp(f"h_dytshv_w{_w}", "dyts", "hypervolume", ucb_window=_w)
    _hp(f"h_dytssc_w{_w}", "dyts", "scalar", ucb_window=_w)
for _c in (0.05, 0.5, 2.0):
    _hp(f"h_ucbsrel_c{str(_c).replace('.', '')}", "ucb", "scalar_rel", ucb_c=_c)
for _c in (0.05, 2.0):
    _hp(f"h_ucbins_c{str(_c).replace('.', '')}", "ucb", "insert", ucb_c=_c)

# E2 (orcamento maior: multiplicador de geracoes; compara cada seletor com `random` NO MESMO orcamento)
for _m in (2, 4):
    for _tag, _sel, _rw in (("random", "random", "scalar"), ("dytshv", "dyts", "hypervolume"), ("ucbhv", "ucb", "hypervolume")):
        VARIANTS[f"b{_m}_{_tag}"] = (_sel, _rw, False, 0.0)
        GENMULT[f"b{_m}_{_tag}"] = _m


# E1b: DYTS FIEL a Sun & Li (2020) (Beta, limiar C=100, recompensa binaria) e as recompensas "replace" (FI>0 do artigo: o filho melhora
# estritamente o Tchebycheff de algum subproblema da vizinhanca). `dyts`/`dyts_*` do catalogo e um amostrador GAUSSIANO de janela
# deslizante, NAO o DYTS do artigo (ver src/mab.py). fd_* = faithful DYTS.
for _name, _sel, _rw, _kw in (
        ("fd_replace", "dyts_beta", "replace", {}),                 # configuracao do artigo (C=100)
        ("fd_replace_c30", "dyts_beta", "replace", {"dyts_c": 30.0}),
        ("fd_replace_c300", "dyts_beta", "replace", {"dyts_c": 300.0}),
        ("fd_scalar", "dyts_beta", "scalar", {}),                   # binario: melhora o pai no subproblema
        ("fd_hv", "dyts_beta", "hypervolume", {}),                  # binario: ganho de HV > 0
        ("ts_replace", "thompson", "replace", {}),                  # nossos Thompson/UCB com a recompensa do artigo
        ("ucb_replace", "ucb", "replace", {})):
    _hp(_name, _sel, _rw, **_kw)


# E3: selecao contextual (um seletor por contexto: faixa do peso do subproblema e/ou fase da busca; src/mab.ContextualSelector)
for _name, _sel, _rw, _ctx in (
        ("c_ucbins_w3", "ucb", "insert", "weight"),
        ("c_ucbins_p3", "ucb", "insert", "phase"),
        ("c_ucbins_w3p3", "ucb", "insert", "weight_phase"),
        ("c_tsins_w3p3", "thompson", "insert", "weight_phase"),
        ("c_fdrep_w3p3", "dyts_beta", "replace", "weight_phase"),
        ("c_ucbsrel_w3p3", "ucb", "scalar_rel", "weight_phase"),
        ("c_ucbhv_w3p3", "ucb", "hypervolume", "weight_phase")):
    _hp(_name, _sel, _rw, context=_ctx)


# E15: FRRMAB de Li et al. (2014) (fitness-rate-rank-based MAB; src/mab.FRRMABSelector). frr_default = parametros do artigo (W = 0,5 N, C = 5, D = 1);
# w25 = janela de 2,5 N (as 2,5 geracoes que o artigo cobre com W = 0,5 N avaliacoes por geracao de N/5); c1/c03 = menos exploracao.
for _name, _kw in (("frr_default", {}), ("frr_w25", {"frr_window_frac": 2.5}), ("frr_c1", {"frr_c": 1.0}), ("frr_c03", {"frr_c": 0.3}),
                   ("frr_w25_c03", {"frr_window_frac": 2.5, "frr_c": 0.3})):
    _hp(_name, "frrmab", "frr", **_kw)


# E16: "pool envenenado" -- o pool de 8 operadores mais 3 operadores nocivos/inuteis (random_restart, half_replace, no_op; src/operators.EXTRA_OPERATORS).
# Testa se a selecao adaptativa evita operadores ruins que a selecao uniforme desperdica (o resultado "uniforme basta" pode depender de um pool bem curado).
PX_EXTRA = ("random_restart", "half_replace", "no_op")
for _name, _sel, _rw in (("px_random", "random", "scalar"), ("px_ucbhv", "ucb", "hypervolume"), ("px_dytshv", "dyts", "hypervolume"),
                         ("px_ucbscal", "ucb", "scalar"), ("px_dytsscal", "dyts", "scalar"), ("px_ucbins", "ucb", "insert"),
                         ("px_fdrep", "dyts_beta", "replace"), ("px_frr", "frrmab", "frr")):
    _hp(_name, _sel, _rw, extra_operators=PX_EXTRA)


# E17: papel da inicializacao GRASP guiada por peso: alpha = 1 (construcao praticamente aleatoria) e alpha = 0 (puramente gulosa) contra o padrao 0,3
_hp("init_a10", "random", "scalar", grasp_alpha=1.0)
_hp("init_a00", "random", "scalar", grasp_alpha=0.0)


def _work(args):
    problem, name, family, filepath, variant, rep, outdir = args
    out = Path(outdir) / f"{name}_{variant}_rep{rep:02d}.json"
    if out.exists():
        try:
            json.load(open(out))
            return name, variant, rep, "skip"
        except Exception:
            pass
    import time
    from capacitated_problem import CDPInstance, GDPInstance
    from experiment import budget_for, seed_for
    from moead_aos import MOEAD_AOS, MOEADConfig

    inst = (CDPInstance if problem == "CDP" else GDPInstance)(ROOT / "instances" / filepath)
    inst.family, inst.name = family, name
    b = budget_for(inst.n)
    sel, rw, ca, mix = VARIANTS[variant]
    excl = POOLS.get(variant, ())
    anc_kw = anchor_kwargs(variant, inst.n)
    cfg = MOEADConfig(n_weights=b["T"], neighborhood_size=b["nbr"], max_gen=b["max_gen"] * GENMULT.get(variant, 1), operator_selection=sel,
                      reward_mode=rw, cost_aware=ca, explore_mix=mix, excluded_operators=excl, **anc_kw, seed=seed_for(name, variant, rep), normalize_objectives=True, **HYPER.get(variant, {}))
    t0 = time.perf_counter()
    res = MOEAD_AOS(inst, cfg).run()
    el = time.perf_counter() - t0
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "instance": name, "family": family, "n": inst.n, "mode": variant, "rep": rep, "seed": cfg.seed,
        "elapsed_s": round(el, 4), "pareto_size": len(res.pareto_front), "pareto_front": res.pareto_front.tolist(),
        "operator_counts": res.operator_counts.tolist(), "normalize_objectives": True,
        "config": {"selection": sel, "reward_mode": rw, "cost_aware": ca, "explore_mix": mix, "excluded_operators": list(excl), **anc_kw},
    }))
    tmp.replace(out)   # escrita atomica: queda no meio nao deixa JSON truncado
    return name, variant, rep, f"{el:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP"], default="CDP")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=150)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv"))
            if r["problem"] == a.problem and a.min_n <= int(r["n"]) <= a.max_n]
    outdir = ROOT / "results" / "phase2_aos" / a.problem / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    vs = [v for v in a.variants.split(",") if v in VARIANTS]
    # ordem: replica-major (rep0 de tudo primeiro), instancias grandes antes para evitar cauda longa
    tasks = [(a.problem, r["name"], r["family"], r["filepath"], v, k, str(outdir))
             for k in range(a.reps) for r in sorted(rows, key=lambda r: -int(r["n"])) for v in vs]
    print(f"{len(rows)} instancias {a.problem} x {len(vs)} variantes x {a.reps} replicas = {len(tasks)} tarefas", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, v, k, status in ex.map(_work, tasks, chunksize=1):
            done += 1
            if done % 25 == 0:
                print(f"[{done}/{len(tasks)}] {name} {v} rep{k} {status}", flush=True)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
