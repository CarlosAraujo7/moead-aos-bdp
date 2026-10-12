"""
spea2.py -- SPEA2 bi-objetivo (Zitzler, Laumanns & Thiele, 2001) para MDP, CDP e GDP.

Sexto metodo do estudo de Colmenar, Marti & Duarte (2018) que faltava no nosso conjunto de baselines (junto com o Iterated Greedy, em bo_trajectory.py).
Fitness F(i) = R(i) + D(i): R = soma das "forcas" (numero de individuos dominados) dos que dominam i; D = 1 / (sigma_k + 2), sigma_k = distancia (no espaco dos
objetivos normalizado) ao k-esimo vizinho, k = sqrt(|P u A|). Selecao ambiental: o novo arquivo A (tamanho fixo `archive_size`) recebe os nao-dominados
(F < 1); se faltam, os melhores dominados por F; se sobram, trunca-se iterativamente o individuo mais proximo de outro (lexicografico nas distancias).
Selecao de acasalamento: torneio binario sobre o arquivo por F. Variacao: a MESMA do nosso NSGA-II (cruzamento de subconjuntos, mutacao por troca, reparo de
viabilidade, inicializacao GRASP) para isolar o mecanismo de selecao (Colmenar et al. usam cruzamento de um ponto e mutacao uniforme com reparo de repeticoes).
Parametros do irace deles (Tabela 6): populacao 340, mutacao 0,14, cruzamento 0,15, 12.805 geracoes (aqui substituidas por limite de tempo).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from nsga2 import NSGA2, NSGA2Config


@dataclass
class SPEA2Config:
    pop_size: int = 340
    archive_size: int | None = None        # None = pop_size
    max_gen: int = 10**9
    crossover_prob: float = 0.15
    mutation_prob: float = 0.14
    init_mode: Literal["grasp", "random"] = "random"      # Colmenar et al.: populacao inicial aleatoria
    grasp_alpha: float = 0.2
    seed: int | None = None
    time_limit_s: float | None = None
    include_init_time: bool = True


@dataclass
class SPEA2Result:
    pareto_front: np.ndarray
    n_gen_executed: int


class SPEA2:
    def __init__(self, instance, config: SPEA2Config | None = None) -> None:
        self.inst = instance
        self.cfg = config or SPEA2Config()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._base = NSGA2(instance, NSGA2Config(pop_size=self.cfg.pop_size, crossover_prob=self.cfg.crossover_prob, mutation_prob=self.cfg.mutation_prob,
                                                 init_mode=self.cfg.init_mode, grasp_alpha=self.cfg.grasp_alpha, seed=self.cfg.seed))
        self._base.rng = self.rng

    @staticmethod
    def _fitness(O: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """F = R + D e R (R == 0 <=> nao-dominado), maximizacao."""
        M = len(O)
        ge = (O[:, None, :] >= O[None, :, :]).all(axis=2)
        gt = (O[:, None, :] > O[None, :, :]).any(axis=2)
        dom = ge & gt                                    # dom[i, j]: i domina j
        strength = dom.sum(axis=1).astype(float)
        R = dom.T @ strength                             # R[j] = soma das forcas dos que dominam j
        span = np.maximum(O.max(axis=0) - O.min(axis=0), 1e-12)
        Z = (O - O.min(axis=0)) / span
        dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        k = min(max(int(np.sqrt(M)), 1), M - 1) if M > 1 else 1
        sigma = np.sort(dist, axis=1)[:, k - 1] if M > 1 else np.zeros(M)
        return R + 1.0 / (sigma + 2.0), R

    @staticmethod
    def _truncate(Z: np.ndarray, n_keep: int) -> list[int]:
        """Remove iterativamente o individuo com menor distancia ao vizinho mais proximo (desempate lexicografico) ate sobrar n_keep."""
        idx = list(range(len(Z)))
        dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        while len(idx) > n_keep:
            sub = dist[np.ix_(idx, idx)]
            srt = np.sort(sub, axis=1)
            order = np.lexsort(srt.T[::-1])              # lexicografico: 1a coluna (vizinho mais proximo) primeiro
            idx.pop(int(order[0]))
        return idx

    def run(self) -> SPEA2Result:
        cfg, inst, rng, base = self.cfg, self.inst, self.rng, self._base
        t_run0 = time.perf_counter()
        N = cfg.pop_size
        NA = cfg.archive_size or N
        P = base._initialize(N)
        PO = [inst.evaluate(x) for x in P]
        A, AO = [], []
        arch_s, arch_f = list(P), list(PO)
        arch_s, arch_f = base._update_archive(arch_s, arch_f, [], [])
        t0 = t_run0 if cfg.include_init_time else time.perf_counter()
        gen = 0
        while gen < cfg.max_gen:
            if cfg.time_limit_s is not None and time.perf_counter() - t0 >= cfg.time_limit_s:
                break
            gen += 1
            U, UO = P + A, PO + AO
            O = np.array([list(f) for f in UO], dtype=float)
            F, R = self._fitness(O)
            nd = [i for i in range(len(U)) if R[i] == 0]
            if len(nd) < NA:
                rest = sorted((i for i in range(len(U)) if R[i] > 0), key=lambda i: F[i])
                sel = nd + rest[: NA - len(nd)]
            elif len(nd) > NA:
                span = np.maximum(O.max(axis=0) - O.min(axis=0), 1e-12)
                Z = (O[nd] - O.min(axis=0)) / span
                sel = [nd[j] for j in self._truncate(Z, NA)]
            else:
                sel = nd
            A, AO = [U[i] for i in sel], [UO[i] for i in sel]
            Fa = F[sel]
            kids, kobj = [], []
            late = False
            while len(kids) < N:
                if cfg.time_limit_s is not None and time.perf_counter() - t0 >= cfg.time_limit_s:     # evita estourar o limite dentro de uma geracao grande
                    late = True
                    break
                a, b = (int(rng.integers(len(A))) for _ in range(2))
                p1 = a if Fa[a] <= Fa[b] else b
                a, b = (int(rng.integers(len(A))) for _ in range(2))
                p2 = a if Fa[a] <= Fa[b] else b
                if rng.random() < cfg.crossover_prob:
                    c1, c2 = base._crossover(A[p1], A[p2], rng)
                else:
                    c1, c2 = A[p1].copy(), A[p2].copy()
                for ch in (c1, c2):
                    if rng.random() < cfg.mutation_prob:
                        ch = base._mutate(ch, rng)
                    if hasattr(inst, "repair"):
                        ch = inst.repair(ch)
                    kids.append(ch)
                    kobj.append(inst.evaluate(ch))
            arch_s, arch_f = base._update_archive(arch_s, arch_f, kids[:N], kobj[:N])
            if late:
                break
            P, PO = kids[:N], kobj[:N]
        pts = np.array([list(f) for f in arch_f], dtype=float).reshape(-1, 2)
        return SPEA2Result(pareto_front=pts, n_gen_executed=gen)
