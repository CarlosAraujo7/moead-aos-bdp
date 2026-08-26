"""
nsga2.py -- NSGA-II para o MDP bi-objetivo.

Implementa o NSGA-II (Deb et al., 2002) adaptado para o Problema de
Diversidade Maxima bi-objetivo (MaxSum + MaxMin), utilizado como
baseline de comparacao com o MOEA/D-AOS.

Representacao
-------------
Cada individuo e um array numpy de inteiros de tamanho m (indices dos
elementos selecionados), ordenado. O espaco de busca e combinatorial:
C(n, m) solucoes possiveis.

Operadores geneticos
--------------------
- Crossover de subconjunto: elementos comuns sao herdados; os restantes
  sao repartidos aleatoriamente entre os filhos.
- Mutacao por troca (RandomSwap1x1): um elemento aleatorio de S e
  substituido por um elemento aleatorio fora de S.

Criterio de selecao
-------------------
Torneio binario baseado em (rank, distancia de crowding), conforme
o NSGA-II original.

Referencias
-----------
Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002). A fast and
    elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions
    on Evolutionary Computation, 6(2), 182-197.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from problem import MDPInstance
from metrics import pareto_front, hypervolume


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

@dataclass
class NSGA2Config:
    """Hiperparametros do NSGA-II para MDP bi-objetivo."""

    pop_size: int = 50
    max_gen: int = 120          # 50 x 120 = 6000 avaliacoes (mesmo budget MOEA/D)
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1  # probabilidade de aplicar mutacao a cada individuo
    init_mode: Literal["grasp", "random"] = "grasp"
    grasp_alpha: float = 0.2
    hv_ref_factor: float = 0.9
    log_every: int = 0
    seed: int | None = None


# ---------------------------------------------------------------------------
# Resultado (compativel com MOEADResult)
# ---------------------------------------------------------------------------

@dataclass
class NSGA2Result:
    pareto_front: np.ndarray
    hv_history: list[float]
    operator_selections: list[int]   # vazio para NSGA-II
    operator_counts: np.ndarray      # zeros
    operator_q_values: np.ndarray    # zeros
    n_gen_executed: int
    best_hv: float
    config: NSGA2Config


# ---------------------------------------------------------------------------
# Algoritmo
# ---------------------------------------------------------------------------

class NSGA2:
    """NSGA-II adaptado para subconjuntos de tamanho fixo (MDP)."""

    def __init__(self, instance: MDPInstance, config: NSGA2Config | None = None) -> None:
        self.inst = instance
        self.cfg = config or NSGA2Config()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------
    # Interface principal
    # ------------------------------------------------------------------

    def run(self) -> NSGA2Result:
        cfg = self.cfg
        inst = self.inst
        rng = self.rng
        N = cfg.pop_size

        # Inicializacao
        pop = self._initialize(N)
        objs = [inst.evaluate(sol) for sol in pop]

        archive_sols = list(pop)
        archive_objs = list(objs)
        archive_sols, archive_objs = self._update_archive(archive_sols, archive_objs, [], [])

        hv_history: list[float] = []

        for gen in range(cfg.max_gen):
            # Ordenacao por nao-dominancia + crowding
            ranks, crowding = self._fast_non_dominated_sort(objs)

            # Geracao de filhos via selecao + crossover + mutacao
            offspring_sols = []
            offspring_objs = []
            while len(offspring_sols) < N:
                p1 = self._tournament_select(ranks, crowding, rng)
                p2 = self._tournament_select(ranks, crowding, rng)
                if rng.random() < cfg.crossover_prob:
                    c1, c2 = self._crossover(pop[p1], pop[p2], rng)
                else:
                    c1, c2 = pop[p1].copy(), pop[p2].copy()
                for child in [c1, c2]:
                    if rng.random() < cfg.mutation_prob:
                        child = self._mutate(child, rng)
                    offspring_sols.append(child)
                    offspring_objs.append(inst.evaluate(child))

            offspring_sols = offspring_sols[:N]
            offspring_objs = offspring_objs[:N]

            # Atualiza arquivo de nao-dominados
            archive_sols, archive_objs = self._update_archive(
                archive_sols, archive_objs, offspring_sols, offspring_objs
            )

            # Combinacao pais + filhos para selecao sobreviventes
            combined_sols = pop + offspring_sols
            combined_objs = objs + offspring_objs

            ranks_c, crowding_c = self._fast_non_dominated_sort(combined_objs)
            pop, objs = self._survivor_selection(combined_sols, combined_objs, ranks_c, crowding_c, N)

            if cfg.log_every > 0 and (gen + 1) % cfg.log_every == 0:
                ref_pt = self._ref_point(archive_objs)
                hv_val = hypervolume(archive_objs, ref_pt)
                hv_history.append(float(hv_val))

        final_front_pts = np.array([list(f) for f in archive_objs])
        final_front = pareto_front(final_front_pts)
        ref_pt = self._ref_point(list(final_front))
        final_hv = hypervolume(final_front, ref_pt)

        return NSGA2Result(
            pareto_front=final_front,
            hv_history=hv_history,
            operator_selections=[],
            operator_counts=np.zeros(4),
            operator_q_values=np.zeros(4),
            n_gen_executed=cfg.max_gen,
            best_hv=float(final_hv),
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Inicializacao
    # ------------------------------------------------------------------

    def _initialize(self, N: int) -> list[np.ndarray]:
        solutions = []
        for _ in range(N):
            if self.cfg.init_mode == "grasp":
                sol = self.inst.greedy_solution(alpha=self.cfg.grasp_alpha)
            else:
                sol = self.inst.random_solution(self.rng)
            solutions.append(sol)
        return solutions

    # ------------------------------------------------------------------
    # Operadores geneticos
    # ------------------------------------------------------------------

    @staticmethod
    def _crossover(s1: np.ndarray, s2: np.ndarray, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Crossover de subconjunto: elementos comuns sao preservados;
        os exclusivos sao particionados aleatoriamente entre os filhos."""
        set1, set2 = set(s1.tolist()), set(s2.tolist())
        common = np.array(sorted(set1 & set2), dtype=np.int64)
        exc1 = np.array(sorted(set1 - set2), dtype=np.int64)
        exc2 = np.array(sorted(set2 - set1), dtype=np.int64)

        pool = np.concatenate([exc1, exc2])
        rng.shuffle(pool)
        need = len(s1) - len(common)

        c1 = np.sort(np.concatenate([common, pool[:need]]))
        c2 = np.sort(np.concatenate([common, pool[need:]]))
        return c1, c2

    def _mutate(self, sol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mutacao por troca: um elemento aleatorio de sol por um de fora."""
        inst = self.inst
        sol_set = set(sol.tolist())
        outside = np.array([i for i in range(inst.n) if i not in sol_set], dtype=np.int64)
        if len(outside) == 0:
            return sol.copy()
        out_idx = int(rng.integers(len(sol)))
        in_idx = int(rng.choice(outside))
        new_sol = sol.copy()
        new_sol[out_idx] = in_idx
        return np.sort(new_sol)

    # ------------------------------------------------------------------
    # Classificacao por nao-dominancia (NSGA-II, O(N^2) em 2 objetivos)
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_non_dominated_sort(
        objectives: list[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna ranks e distancias de crowding para toda a populacao."""
        N = len(objectives)
        objs = np.array(objectives, dtype=np.float64)  # shape (N, 2)

        # Calcula dominancia par a par
        n_dominated = np.zeros(N, dtype=int)    # n_p: qtd que domina p
        dominated_by: list[list[int]] = [[] for _ in range(N)]  # S_p

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # j domina i (maximizacao)?
                if objs[j, 0] >= objs[i, 0] and objs[j, 1] >= objs[i, 1]:
                    if objs[j, 0] > objs[i, 0] or objs[j, 1] > objs[i, 1]:
                        n_dominated[i] += 1
                        dominated_by[j].append(i)

        ranks = np.full(N, -1, dtype=int)
        front_id = 0
        current_front = [i for i in range(N) if n_dominated[i] == 0]
        for i in current_front:
            ranks[i] = 0

        while current_front:
            next_front = []
            for p in current_front:
                for q in dominated_by[p]:
                    n_dominated[q] -= 1
                    if n_dominated[q] == 0:
                        ranks[q] = front_id + 1
                        next_front.append(q)
            front_id += 1
            current_front = next_front

        # Crowding distance por frente
        crowding = np.zeros(N, dtype=float)
        for r in range(ranks.max() + 1):
            front_idx = np.where(ranks == r)[0]
            if len(front_idx) <= 2:
                crowding[front_idx] = np.inf
                continue
            front_objs = objs[front_idx]
            for m in range(2):
                order = np.argsort(front_objs[:, m])
                sorted_idx = front_idx[order]
                f_min = front_objs[order[0], m]
                f_max = front_objs[order[-1], m]
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                if f_max - f_min < 1e-12:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    crowding[sorted_idx[k]] += (
                        (front_objs[order[k + 1], m] - front_objs[order[k - 1], m])
                        / (f_max - f_min)
                    )

        return ranks, crowding

    # ------------------------------------------------------------------
    # Selecao
    # ------------------------------------------------------------------

    @staticmethod
    def _tournament_select(
        ranks: np.ndarray,
        crowding: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Torneio binario: vence quem tem menor rank; em empate, maior crowding."""
        N = len(ranks)
        a, b = int(rng.integers(N)), int(rng.integers(N))
        if ranks[a] < ranks[b]:
            return a
        if ranks[b] < ranks[a]:
            return b
        return a if crowding[a] >= crowding[b] else b

    @staticmethod
    def _survivor_selection(
        combined_sols: list[np.ndarray],
        combined_objs: list[tuple[float, float]],
        ranks: np.ndarray,
        crowding: np.ndarray,
        N: int,
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """Seleciona os N melhores individuos da populacao combinada."""
        # Ordena por (rank asc, crowding desc)
        order = sorted(range(len(combined_sols)),
                       key=lambda i: (ranks[i], -crowding[i]))
        selected = order[:N]
        return [combined_sols[i] for i in selected], [combined_objs[i] for i in selected]

    # ------------------------------------------------------------------
    # Arquivo de nao-dominados (igual ao MOEAD_AOS)
    # ------------------------------------------------------------------

    @staticmethod
    def _update_archive(
        archive_sols, archive_objs, new_sols, new_objs
    ):
        for sol, f in zip(new_sols, new_objs):
            dominated_by_new = []
            dominated = False
            duplicate = False
            for idx, f_arch in enumerate(archive_objs):
                if abs(f[0] - f_arch[0]) < 1e-9 and abs(f[1] - f_arch[1]) < 1e-9:
                    duplicate = True
                    break
                if f[0] >= f_arch[0] and f[1] >= f_arch[1]:
                    if f[0] > f_arch[0] or f[1] > f_arch[1]:
                        dominated_by_new.append(idx)
                elif f_arch[0] >= f[0] and f_arch[1] >= f[1]:
                    if f_arch[0] > f[0] or f_arch[1] > f[1]:
                        dominated = True
                        break
            if not dominated and not duplicate:
                for idx in sorted(dominated_by_new, reverse=True):
                    archive_sols.pop(idx)
                    archive_objs.pop(idx)
                archive_sols.append(sol.copy())
                archive_objs.append(f)
        return archive_sols, archive_objs

    def _ref_point(self, archive_objs) -> np.ndarray:
        if not archive_objs:
            return np.zeros(2)
        f1 = [f[0] for f in archive_objs]
        f2 = [f[1] for f in archive_objs]
        fc = self.cfg.hv_ref_factor
        return np.array([min(f1) * fc, min(f2) * fc])

    def __repr__(self) -> str:
        return (f"NSGA2(pop={self.cfg.pop_size}, gen={self.cfg.max_gen}, "
                f"n={self.inst.n}, m={self.inst.m})")


# ---------------------------------------------------------------------------
# Teste direto
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv, time
    from pathlib import Path

    root = Path(__file__).parent.parent
    catalog = root / "instances" / "catalog.csv"
    with catalog.open() as fh:
        rows = list(csv.DictReader(fh))

    targets = [r for r in rows if r["family"] == "GKD-d"
               and int(r["n"]) <= 50
               and r.get("bi_obj_ok", "True") == "True"][:2]

    for row in targets:
        fname = row["name"] + ".txt"
        matches = list((root / "instances").rglob(fname))
        if not matches:
            continue
        inst = MDPInstance(matches[0])
        print(f"\n{'='*60}\nInstancia: {inst}")
        for pop, gen in [(50, 120), (100, 60)]:
            cfg = NSGA2Config(pop_size=pop, max_gen=gen, seed=42)
            t0 = time.perf_counter()
            result = NSGA2(inst, cfg).run()
            elapsed = time.perf_counter() - t0
            print(f"  pop={pop}, gen={gen}: |PF|={len(result.pareto_front):3d}  "
                  f"HV={result.best_hv:.4f}  t={elapsed:.3f}s")
