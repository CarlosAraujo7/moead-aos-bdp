"""
moead_aos.py -- MOEA/D com Adaptive Operator Selection (UCB1/Thompson) para o MDP bi-objetivo.

Decomposicao MOEA/D por Tchebycheff (maximizacao):
    g_tch(x | lam, z*) = max( lam[0]*(z0*-f0(x)), lam[1]*(z1*-f1(x)) )

Modos de operacao:
    operator_selection="ucb"      -> MOEA/D-AOS via UCB1 (contribuicao principal)
    operator_selection="thompson" -> MOEA/D-AOS via Thompson Sampling
    operator_selection="random"   -> MOEA/D-random (baseline ablativo)

Modo de recompensa (quando AOS ativo):
    reward_mode="hypervolume" -> ganho de hipervolume (recomendado)
    reward_mode="scalar"      -> melhoria escalar Tchebycheff

Inicializacao:
    Warm-start GRASP guiado pelo vetor de pesos de cada subproblema,
    gerando populacao inicial que ja cobre a frente de Pareto desde a
    primeira geracao.

Referencias:
    Zhang & Li (2007) MOEA/D. IEEE TEC 11(6):712-731.
    DaCosta et al. (2008) AOS with MAB. GECCO 2008.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

from problem import MDPInstance
from operators import OPERATOR_POOL, OPERATOR_NAMES
from mab import UCBSelector, RandomSelector, ThompsonSelector, DYTSSelector, scalar_reward, hypervolume_reward
from metrics import hypervolume, pareto_front


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

@dataclass
class MOEADConfig:
    """Hiperparametros do MOEA/D-AOS."""

    # Decomposicao
    n_weights: int = 20         # T: numero de subproblemas / vetores de peso
    neighborhood_size: int = 5  # tamanho da vizinhanca por subproblema

    # Criterio de parada
    max_gen: int = 300

    # Selecao de operadores
    operator_selection: Literal["ucb", "random", "thompson", "dyts"] = "ucb"

    # Modo de recompensa (quando AOS ativo)
    reward_mode: Literal["scalar", "hypervolume"] = "hypervolume"

    # Parametros UCB
    ucb_c: float = 0.5
    ucb_window: int = 50

    # Inicializacao warm-start GRASP
    grasp_alpha: float = 0.3   # 0=guloso, 1=aleatorio

    # Ponto de referencia para HV
    hv_ref_factor: float = 0.9

    # Logging
    log_every: int = 0

    # Semente
    seed: int | None = None


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class MOEADResult:
    pareto_front: np.ndarray
    hv_history: list[float]
    operator_selections: list[int]
    operator_counts: np.ndarray
    operator_q_values: np.ndarray
    n_gen_executed: int
    best_hv: float
    config: MOEADConfig


# ---------------------------------------------------------------------------
# Algoritmo
# ---------------------------------------------------------------------------

class MOEAD_AOS:
    """MOEA/D com suporte a AOS (UCB1/Thompson) ou selecao aleatoria."""

    def __init__(self, instance: MDPInstance, config: MOEADConfig | None = None) -> None:
        self.inst = instance
        self.cfg = config or MOEADConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def run(self) -> MOEADResult:
        cfg = self.cfg
        inst = self.inst
        rng = self.rng

        N = cfg.n_weights
        T = min(cfg.neighborhood_size, N)

        lambdas = self._generate_weights(N)
        neighborhoods = self._compute_neighborhoods(lambdas, T)

        # Warm-start: cada subproblema inicializado com GRASP guiado pelo seu peso
        solutions, objectives = self._initialize(N, lambdas)

        z_star = np.array([
            max(f[0] for f in objectives),
            max(f[1] for f in objectives),
        ])

        archive_sols = list(solutions)
        archive_objs = list(objectives)
        archive_sols, archive_objs = self._update_archive(archive_sols, archive_objs, [], [])

        if cfg.operator_selection == "ucb":
            selector = UCBSelector(
                n_operators=len(OPERATOR_POOL),
                c=cfg.ucb_c,
                window_size=cfg.ucb_window,
            )
        elif cfg.operator_selection == "thompson":
            selector = ThompsonSelector(
                n_operators=len(OPERATOR_POOL),
                window_size=cfg.ucb_window,
            )
        elif cfg.operator_selection == "dyts":
            selector = DYTSSelector(
                n_operators=len(OPERATOR_POOL),
                window_size=cfg.ucb_window,
            )
        else:
            selector = RandomSelector(n_operators=len(OPERATOR_POOL))

        ref_pt = self._compute_ref_point(archive_objs)
        hv_history: list[float] = []
        op_selections: list[int] = []

        for gen in range(cfg.max_gen):
            order = rng.permutation(N)

            for i in order:
                k = selector.select(rng)
                op_selections.append(k)

                if rng.random() < 0.9:
                    j = int(rng.choice(neighborhoods[i]))
                else:
                    j = int(rng.integers(N))

                parent_sol = solutions[j].copy()
                parent_f   = objectives[j]

                operator = OPERATOR_POOL[k]
                # Passa lam e z_star para operadores bi-objetivo que os utilizem
                import inspect as _insp
                _sig = _insp.signature(operator.apply)
                _kw: dict = {}
                if "lam" in _sig.parameters:
                    _kw["lam"] = lambdas[i]
                if "z_star" in _sig.parameters:
                    _kw["z_star"] = z_star.copy()
                new_sol, new_f = operator.apply(parent_sol, inst, rng, parent_f, **_kw)

                if cfg.operator_selection in ("ucb", "thompson", "dyts"):
                    if cfg.reward_mode == "scalar":
                        old_g = self._tchebycheff(parent_f, lambdas[i], z_star)
                        new_g = self._tchebycheff(new_f,    lambdas[i], z_star)
                        reward = scalar_reward(old_g, new_g)
                    else:
                        dominated = any(
                            fa[0] >= new_f[0] and fa[1] >= new_f[1]
                            and (fa[0] > new_f[0] or fa[1] > new_f[1])
                            for fa in archive_objs
                        )
                        if dominated:
                            reward = 0.0
                        else:
                            old_hv  = hypervolume(archive_objs, ref_pt)
                            new_hv  = hypervolume(archive_objs + [new_f], ref_pt)
                            reward  = hypervolume_reward(old_hv, new_hv)
                    selector.update(k, reward)

                for j in neighborhoods[i]:
                    g_old = self._tchebycheff(objectives[j], lambdas[j], z_star)
                    g_new = self._tchebycheff(new_f,         lambdas[j], z_star)
                    if g_new <= g_old:
                        solutions[j]  = new_sol.copy()
                        objectives[j] = new_f

                if new_f[0] > z_star[0]:
                    z_star[0] = new_f[0]
                if new_f[1] > z_star[1]:
                    z_star[1] = new_f[1]

                archive_sols, archive_objs = self._update_archive(
                    archive_sols, archive_objs, [new_sol], [new_f]
                )

            ref_pt = self._compute_ref_point(archive_objs)

            if cfg.log_every > 0 and (gen + 1) % cfg.log_every == 0:
                hv_history.append(float(hypervolume(archive_objs, ref_pt)))

        final_pts  = np.array([list(f) for f in archive_objs])
        final_front = pareto_front(final_pts)
        final_hv    = float(hypervolume(final_front, ref_pt))

        return MOEADResult(
            pareto_front=final_front,
            hv_history=hv_history,
            operator_selections=op_selections,
            operator_counts=selector.counts(),
            operator_q_values=selector.q_values(),
            n_gen_executed=cfg.max_gen,
            best_hv=final_hv,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Inicializacao
    # ------------------------------------------------------------------

    def _initialize(
        self, N: int, lambdas: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """Warm-start GRASP guiado por peso: cada subproblema recebe uma solucao
        construida com o criterio lam[0]*deltaSum + lam[1]*minDist, garantindo
        cobertura diversa da frente de Pareto desde a primeira geracao."""
        solutions, objectives = [], []
        for i in range(N):
            sol = self._grasp_init(lambdas[i])
            f   = self.inst.evaluate(sol)
            solutions.append(sol)
            objectives.append(f)
        return solutions, objectives

    def _grasp_init(self, lam: np.ndarray) -> np.ndarray:
        """Construcao GRASP guiada pelo vetor de pesos lam."""
        inst  = self.inst
        n, m  = inst.n, inst.m
        alpha = self.cfg.grasp_alpha
        rng   = self.rng
        D     = inst.D

        first = int(rng.integers(n))
        selected: list[int] = [first]
        available = list(range(n))
        available.remove(first)

        for _ in range(m - 1):
            sel_arr = np.array(selected, dtype=np.intp)
            scores  = np.empty(len(available))
            for k, cand in enumerate(available):
                delta_sum = float(D[cand, sel_arr].sum())
                min_dist  = float(D[cand, sel_arr].min())
                scores[k] = lam[0] * delta_sum + lam[1] * min_dist

            s_max, s_min = scores.max(), scores.min()
            if s_max > s_min:
                threshold = s_max - alpha * (s_max - s_min)
                rcl = [available[k] for k in range(len(available)) if scores[k] >= threshold]
            else:
                rcl = available[:]

            chosen = int(rng.choice(rcl))
            selected.append(chosen)
            available.remove(chosen)

        return np.sort(np.array(selected, dtype=np.intp))

    # ------------------------------------------------------------------
    # Auxiliares estaticos
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_weights(N: int) -> np.ndarray:
        eps = 1e-3
        lambdas = np.zeros((N, 2))
        for i in range(N):
            t = i / (N - 1) if N > 1 else 0.5
            lambdas[i, 0] = eps + (1 - 2 * eps) * t
            lambdas[i, 1] = 1.0 - lambdas[i, 0]
        return lambdas

    @staticmethod
    def _compute_neighborhoods(lambdas: np.ndarray, T: int) -> list[np.ndarray]:
        N = len(lambdas)
        return [np.argsort(np.linalg.norm(lambdas - lambdas[i], axis=1))[:T] for i in range(N)]

    @staticmethod
    def _tchebycheff(f: tuple[float, float], lam: np.ndarray, z_star: np.ndarray) -> float:
        return float(np.max(lam * (z_star - np.array(f))))

    @staticmethod
    def _update_archive(
        archive_sols: list[np.ndarray],
        archive_objs: list[tuple[float, float]],
        new_sols: list[np.ndarray],
        new_objs: list[tuple[float, float]],
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        for sol, f in zip(new_sols, new_objs):
            dominated_by_new = []
            dominated = False
            duplicate = False

            for idx, fa in enumerate(archive_objs):
                if abs(f[0]-fa[0]) < 1e-9 and abs(f[1]-fa[1]) < 1e-9:
                    duplicate = True
                    break
                if f[0] >= fa[0] and f[1] >= fa[1]:
                    if f[0] > fa[0] or f[1] > fa[1]:
                        dominated_by_new.append(idx)
                elif fa[0] >= f[0] and fa[1] >= f[1]:
                    if fa[0] > f[0] or fa[1] > f[1]:
                        dominated = True
                        break

            if not dominated and not duplicate:
                for idx in sorted(dominated_by_new, reverse=True):
                    archive_sols.pop(idx)
                    archive_objs.pop(idx)
                archive_sols.append(sol.copy())
                archive_objs.append(f)

        return archive_sols, archive_objs

    def _compute_ref_point(self, archive_objs: list[tuple[float, float]]) -> np.ndarray:
        if not archive_objs:
            return np.zeros(2)
        fc = self.cfg.hv_ref_factor
        return np.array([
            min(f[0] for f in archive_objs) * fc,
            min(f[1] for f in archive_objs) * fc,
        ])
