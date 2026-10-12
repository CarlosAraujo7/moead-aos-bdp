"""
vns.py — Variable Neighborhood Search Bi-objetivo para o BODP.

Implementa duas variantes:
  1. BVNS-Pareto  : VNS com arquivo de Pareto externo. Aceita uma nova solucao
                   se ela nao e dominada pelo arquivo atual (criterio Pareto).
                   Usa 3 estruturas de vizinhanca crescentes: 1-opt, 2-opt, k-opt.

  2. BVNS-Tch    : VNS multi-start escalarizado via Tchebycheff com vetor de
                   pesos distribuidos uniformemente. Equivalente ao MOEA/D sem
                   operadores adaptativos — serve como ablacao do MOEA/D.

Ambas as variantes sao benchmarks legitimos e frequentemente usados em artigos
EJOR de busca local bi-objetivo (Liefooghe et al., 2018; Paquete & Stutzle, 2006).

Compatibilidade CDP/GDP
------------------------
As 3 estruturas de vizinhanca (_neighborhood_1opt/2opt/kopt) operam sobre
sol.size (tamanho real da solucao recebida), nao mais sobre um inst.m fixo
-- funcionam identicamente no MDP (sol.size == m sempre) e em CDP/GDP
(tamanho livre). Nenhuma delas garante viabilidade por si so (uma troca pode
derrubar a capacidade minima ou, no GDP, estourar orcamento) -- cada uma
aplica um "repair-net" duck-typed (if hasattr(inst, "repair")) antes do
evaluate final, mesmo padrao usado em nsga2.py/moead_aos.py/operators.py.
instance aceita MDPInstance, CDPInstance ou GDPInstance
(capacitated_problem.py); type hint continua MDPInstance por consistencia
com o resto do projeto (contrato real e duck-typed).

Referencias
-----------
Hansen & Mladenovic (2001). Variable neighborhood search: Principles and
    applications. EJOR, 130(3), 449-467.
Liefooghe, A. et al. (2018). A survey on bi-objective combinatorial
    optimization with variable neighborhood search. RAIRO-OR, 52, 59-78.
Paquete, L., Stutzle, T. (2006). A study of stochastic local search for the
    multiobjective assignment problem. EJOR, 169(3), 836-854.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from problem import MDPInstance
from metrics import hypervolume, pareto_front
from operators import OPERATOR_NAMES


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

@dataclass
class BVNSConfig:
    """Hiperparametros do BVNS bi-objetivo."""

    # Criterio de parada
    max_iter: int = 500          # iteracoes totais (cada iteracao = uma vizinhanca)
    max_no_improve: int = 150    # parada antecipada se archive nao muda

    # Vizinhancas (k_max): 1=1-opt, 2=2-opt, 3=k-opt(k=3-5)
    k_max: int = 3

    # Variante
    variant: Literal["pareto", "tchebycheff"] = "pareto"

    # Para variante Tchebycheff
    n_weights: int = 20          # numero de vetores de peso
    # Normaliza os objetivos pela amplitude observada no Tchebycheff (padrao False
    # = comportamento historico; ver MOEADConfig.normalize_objectives).
    normalize_objectives: bool = False

    # Logging
    log_every: int = 0

    # Semente
    seed: int | None = None

    # Limite de tempo de parede (s) para comparacoes a TEMPO IGUAL (E4); None = so max_iter/max_no_improve (historico).
    time_limit_s: float | None = None


@dataclass
class BVNSResult:
    pareto_front: np.ndarray
    hv_history: list[float]
    n_iter_executed: int
    operator_counts: np.ndarray   # compatibilidade com experiment.py
    operator_q_values: np.ndarray # compatibilidade com experiment.py
    best_hv: float
    config: BVNSConfig


# ---------------------------------------------------------------------------
# Estruturas de vizinhanca
# ---------------------------------------------------------------------------

def _neighborhood_1opt(sol: np.ndarray, inst: MDPInstance, rng: np.random.Generator,
                        f=None):
    """Melhor vizinho 1-opt (troca um elemento pelo melhor externo por MaxSum)."""
    n = inst.n
    m = sol.size   # tamanho real da solucao (== inst.m no MDP, livre no CDP/GDP)
    D = inst.D

    out_idx = int(rng.integers(m))
    mask = np.ones(m, dtype=bool); mask[out_idx] = False
    remaining = sol[mask]

    in_sol = np.zeros(n, dtype=bool); in_sol[sol] = True
    cands = np.where(~in_sol)[0]
    if len(cands) == 0:
        return sol.copy(), inst.evaluate(sol)

    contrib = D[np.ix_(cands, remaining)].sum(axis=1)
    best = int(cands[np.argmax(contrib)])
    new_sol = sol.copy(); new_sol[out_idx] = best
    if hasattr(inst, "repair"):
        new_sol = inst.repair(new_sol)
    return new_sol, inst.evaluate(new_sol)


def _neighborhood_2opt(sol: np.ndarray, inst: MDPInstance, rng: np.random.Generator,
                        f=None):
    """Troca 2 elementos aleatorios pelos 2 melhores externos (greedy)."""
    n = inst.n
    m = sol.size
    D = inst.D

    out_idxs = rng.choice(m, size=min(2, m), replace=False)
    mask = np.ones(m, dtype=bool); mask[out_idxs] = False
    remaining = sol[mask]

    in_sol = np.zeros(n, dtype=bool); in_sol[sol] = True
    cands = np.where(~in_sol)[0]
    if len(cands) < 2:
        return sol.copy(), inst.evaluate(sol)

    # Insere greedy: escolhe o melhor, depois o segundo melhor dado o primeiro
    contrib = D[np.ix_(cands, remaining)].sum(axis=1)
    b1_pos = int(np.argmax(contrib))
    b1 = int(cands[b1_pos])
    remaining2 = np.append(remaining, b1)
    mask2 = np.ones(len(cands), dtype=bool); mask2[b1_pos] = False
    cands2 = cands[mask2]
    if len(cands2) == 0:
        new_sol = sol.copy(); new_sol[out_idxs[0]] = b1
        if hasattr(inst, "repair"):
            new_sol = inst.repair(new_sol)
        return new_sol, inst.evaluate(new_sol)
    contrib2 = D[np.ix_(cands2, remaining2)].sum(axis=1)
    b2 = int(cands2[np.argmax(contrib2)])

    new_sol = sol.copy()
    new_sol[out_idxs[0]] = b1
    if len(out_idxs) > 1:
        new_sol[out_idxs[1]] = b2
    if hasattr(inst, "repair"):
        new_sol = inst.repair(new_sol)
    return new_sol, inst.evaluate(new_sol)


def _neighborhood_kopt(sol: np.ndarray, inst: MDPInstance, rng: np.random.Generator,
                        f=None, k: int = 4):
    """Remove k elementos aleatorios e reinsere greedily."""
    n = inst.n
    m = sol.size
    D = inst.D
    k = min(k, m - 1)

    out_idxs = rng.choice(m, size=k, replace=False)
    keep_mask = np.ones(m, dtype=bool); keep_mask[out_idxs] = False
    kept = sol[keep_mask].tolist()

    in_sol = np.zeros(n, dtype=bool); in_sol[kept] = True
    selected = list(kept)
    contrib = D[selected].sum(axis=0) if selected else np.zeros(n)

    for _ in range(k):
        cands = np.where(~in_sol)[0]
        if len(cands) == 0:
            break
        chosen = int(cands[np.argmax(contrib[cands])])
        selected.append(chosen)
        in_sol[chosen] = True
        contrib += D[chosen]

    new_sol = np.sort(np.array(selected, dtype=np.intp))
    if hasattr(inst, "repair"):
        new_sol = inst.repair(new_sol)
    return new_sol, inst.evaluate(new_sol)


NEIGHBORHOODS = [_neighborhood_1opt, _neighborhood_2opt, _neighborhood_kopt]


# ---------------------------------------------------------------------------
# Arquivo de Pareto
# ---------------------------------------------------------------------------

def _dominates(fa, fb):
    """Retorna True se fa domina fb (maximizacao)."""
    return (fa[0] >= fb[0] and fa[1] >= fb[1] and
            (fa[0] > fb[0] or fa[1] > fb[1]))


def _update_archive(archive_sols, archive_objs, new_sol, new_f):
    """Atualiza o arquivo com (new_sol, new_f). Retorna True se inserido."""
    for fa in archive_objs:
        if _dominates(fa, new_f):
            return False   # dominado -> rejeita
        if abs(fa[0] - new_f[0]) < 1e-9 and abs(fa[1] - new_f[1]) < 1e-9:
            return False   # duplicado

    # Remove solucoes dominadas por new_f
    to_remove = [i for i, fa in enumerate(archive_objs) if _dominates(new_f, fa)]
    for i in sorted(to_remove, reverse=True):
        archive_sols.pop(i)
        archive_objs.pop(i)

    archive_sols.append(new_sol.copy())
    archive_objs.append(new_f)
    return True


# ---------------------------------------------------------------------------
# BVNS-Pareto
# ---------------------------------------------------------------------------

class BVNS:
    """Variable Neighborhood Search Bi-objetivo com arquivo de Pareto."""

    def __init__(self, instance: MDPInstance, config: BVNSConfig | None = None):
        self.inst = instance
        self.cfg  = config or BVNSConfig()
        self.rng  = np.random.default_rng(self.cfg.seed)

    def run(self) -> BVNSResult:
        cfg  = self.cfg
        inst = self.inst
        rng  = self.rng

        if cfg.variant == "tchebycheff":
            return self._run_tchebycheff()

        # --- BVNS-Pareto ---
        # Inicializacao: solucao aleatoria
        sol = inst.random_solution(rng)
        f   = inst.evaluate(sol)

        archive_sols: list[np.ndarray] = [sol.copy()]
        archive_objs: list[tuple]      = [f]

        hv_history: list[float] = []
        ref_pt = self._ref_point(archive_objs)

        no_improve = 0
        n_iter = 0

        import time as _time
        _t0 = _time.perf_counter()
        for it in range(cfg.max_iter):
            if cfg.time_limit_s is not None and _time.perf_counter() - _t0 >= cfg.time_limit_s:
                break
            improved_global = False

            # Shake: escolhe uma solucao do arquivo aleatoriamente como ponto de partida
            if len(archive_sols) > 1:
                idx = int(rng.integers(len(archive_sols)))
                cur_sol = archive_sols[idx].copy()
                cur_f   = archive_objs[idx]
            else:
                cur_sol = archive_sols[0].copy()
                cur_f   = archive_objs[0]

            k = 1
            while k <= min(cfg.k_max, len(NEIGHBORHOODS)):
                nb_fn = NEIGHBORHOODS[k - 1]
                extra = {"k": 3 + k} if k == 3 else {}
                new_sol, new_f = nb_fn(cur_sol, inst, rng, cur_f, **extra)

                inserted = _update_archive(archive_sols, archive_objs, new_sol, new_f)
                if inserted:
                    cur_sol = new_sol
                    cur_f   = new_f
                    k = 1   # reinicia do vizinho mais proximo
                    improved_global = True
                else:
                    k += 1

            n_iter = it + 1
            if improved_global:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= cfg.max_no_improve:
                break

            # Logging HV
            if cfg.log_every > 0 and (it + 1) % cfg.log_every == 0:
                ref_pt = self._ref_point(archive_objs)
                hv_history.append(float(hypervolume(archive_objs, ref_pt)))

        ref_pt     = self._ref_point(archive_objs)
        final_pts  = np.array([list(f) for f in archive_objs])
        final_front = pareto_front(final_pts)
        best_hv     = float(hypervolume(final_front, ref_pt))

        return BVNSResult(
            pareto_front=final_front,
            hv_history=hv_history,
            n_iter_executed=n_iter,
            operator_counts=np.zeros(len(OPERATOR_NAMES)),   # sem pool de operadores
            operator_q_values=np.zeros(len(OPERATOR_NAMES)),
            best_hv=best_hv,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Variante Tchebycheff (MOEA/D sem AOS — ablacao pura)
    # ------------------------------------------------------------------

    def _run_tchebycheff(self) -> BVNSResult:
        cfg  = self.cfg
        inst = self.inst
        rng  = self.rng
        N    = cfg.n_weights

        # Gera pesos e inicializa com solucoes aleatorias
        lambdas = np.zeros((N, 2))
        for i in range(N):
            t = i / (N - 1) if N > 1 else 0.5
            lambdas[i, 0] = 1e-3 + (1 - 2e-3) * t
            lambdas[i, 1] = 1.0 - lambdas[i, 0]

        solutions  = [inst.random_solution(rng) for _ in range(N)]
        objectives = [inst.evaluate(s) for s in solutions]
        z_star = np.array([max(f[0] for f in objectives),
                           max(f[1] for f in objectives)])
        z_min = np.array([min(f[0] for f in objectives),
                          min(f[1] for f in objectives)])
        w_inv = np.ones(2)

        def _refresh_w_inv():
            if cfg.normalize_objectives:
                r_ = z_star - z_min
                w_inv[:] = np.where(r_ > 1e-12, 1.0 / np.maximum(r_, 1e-12), 1.0)

        _refresh_w_inv()

        archive_sols = list(solutions)
        archive_objs = list(objectives)

        hv_history: list[float] = []
        ref_pt = self._ref_point(archive_objs)

        iters_per_weight = max(1, cfg.max_iter // N)
        import time as _time
        _t0 = _time.perf_counter()
        n_done = 0

        for it in range(cfg.max_iter):
            if cfg.time_limit_s is not None and _time.perf_counter() - _t0 >= cfg.time_limit_s:
                break
            n_done = it + 1
            i = it % N
            lam = lambdas[i]
            cur_sol = solutions[i].copy()
            cur_f   = objectives[i]

            # VNS local com criterio Tchebycheff
            k = 1
            g_cur = float(np.max(lam * w_inv * (z_star - np.array(cur_f))))

            while k <= min(cfg.k_max, len(NEIGHBORHOODS)):
                nb_fn = NEIGHBORHOODS[k - 1]
                extra = {"k": 3 + k} if k == 3 else {}
                new_sol, new_f = nb_fn(cur_sol, inst, rng, cur_f, **extra)
                g_new = float(np.max(lam * w_inv * (z_star - np.array(new_f))))

                if g_new < g_cur:
                    cur_sol, cur_f, g_cur = new_sol, new_f, g_new
                    k = 1
                else:
                    k += 1

            solutions[i]  = cur_sol
            objectives[i] = cur_f
            z_star[0] = max(z_star[0], cur_f[0])
            z_star[1] = max(z_star[1], cur_f[1])
            z_min[0] = min(z_min[0], cur_f[0])
            z_min[1] = min(z_min[1], cur_f[1])
            _refresh_w_inv()
            _update_archive(archive_sols, archive_objs, cur_sol, cur_f)

            if cfg.log_every > 0 and (it + 1) % cfg.log_every == 0:
                ref_pt = self._ref_point(archive_objs)
                hv_history.append(float(hypervolume(archive_objs, ref_pt)))

        ref_pt      = self._ref_point(archive_objs)
        final_pts   = np.array([list(f) for f in archive_objs])
        final_front = pareto_front(final_pts)
        best_hv     = float(hypervolume(final_front, ref_pt))

        return BVNSResult(
            pareto_front=final_front,
            hv_history=hv_history,
            n_iter_executed=n_done if cfg.time_limit_s is not None else cfg.max_iter,
            operator_counts=np.zeros(len(OPERATOR_NAMES)),
            operator_q_values=np.zeros(len(OPERATOR_NAMES)),
            best_hv=best_hv,
            config=cfg,
        )

    def _ref_point(self, archive_objs):
        if not archive_objs:
            return np.zeros(2)
        fc = 0.9
        return np.array([min(f[0] for f in archive_objs) * fc,
                         min(f[1] for f in archive_objs) * fc])
