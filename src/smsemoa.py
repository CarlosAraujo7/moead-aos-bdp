"""
smsemoa.py -- SMS-EMOA bi-objetivo (steady-state, selecao por contribuicao de hipervolume) para MDP, CDP e GDP.

SMS-EMOA (Beume, Naujoks & Emmerich, 2007): populacao de mu individuos; a cada passo um filho (selecao aleatoria dos pais, cruzamento e mutacao)
entra na populacao e o individuo da pior frente com a MENOR contribuicao de hipervolume e removido. Em duas dimensoes a contribuicao de um ponto de
uma frente ordenada e o retangulo entre seus vizinhos; os extremos usam o ponto de referencia. Reutiliza do NSGA-II (nsga2.py) a inicializacao,
o cruzamento de subconjuntos, a mutacao por troca, o reparo de viabilidade e o arquivo externo, de modo que a comparacao isola o mecanismo de selecao
(indicador de hipervolume em vez de rank + crowding). Para comparacoes a TEMPO IGUAL: `time_limit_s` com `include_init_time` (como no NSGA-II).

Referencia: Beume, N., Naujoks, B., Emmerich, M. (2007). SMS-EMOA: Multiobjective selection based on dominated hypervolume. EJOR 181(3):1653-1669.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from metrics import pareto_front
from nsga2 import NSGA2, NSGA2Config


@dataclass
class SMSEMOAConfig:
    pop_size: int = 50
    max_evals: int = 10**9          # com time_limit_s, o limite efetivo e o tempo
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    init_mode: Literal["grasp", "random"] = "grasp"
    grasp_alpha: float = 0.2
    seed: int | None = None
    time_limit_s: float | None = None
    include_init_time: bool = True


@dataclass
class SMSEMOAResult:
    pareto_front: np.ndarray
    n_evals: int


class SMSEMOA:
    def __init__(self, instance, config: SMSEMOAConfig | None = None) -> None:
        self.inst = instance
        self.cfg = config or SMSEMOAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        # o NSGA-II fornece inicializacao, cruzamento, mutacao e arquivo (nao e executado)
        self._base = NSGA2(instance, NSGA2Config(pop_size=self.cfg.pop_size, crossover_prob=self.cfg.crossover_prob, mutation_prob=self.cfg.mutation_prob,
                                                 init_mode=self.cfg.init_mode, grasp_alpha=self.cfg.grasp_alpha, seed=self.cfg.seed))
        self._base.rng = self.rng

    @staticmethod
    def _contributions(F: np.ndarray) -> np.ndarray:
        """Contribuicao de hipervolume (maximizacao, 2D) de cada ponto de uma frente: F (k x 2) normalizado, ponto de referencia (-0.1, -0.1)."""
        k = len(F)
        if k == 1:
            return np.array([np.inf])
        order = np.argsort(-F[:, 0], kind="stable")      # f1 decrescente => f2 crescente numa frente nao dominada
        f1, f2 = F[order, 0], F[order, 1]
        c = np.empty(k)
        ref = -0.1
        for i in range(k):
            right = f1[i + 1] if i + 1 < k else ref      # proximo com f1 menor
            up = f2[i - 1] if i > 0 else ref             # vizinho anterior (f1 maior, f2 menor)
            c[order[i]] = (f1[i] - right) * (f2[i] - up)
        return c

    @staticmethod
    def _ranks_2d(P: np.ndarray) -> np.ndarray:
        """Indice da frente de nao dominancia (0 = melhor) de cada ponto, maximizacao 2D, em O(k log k): ordena por f1 decrescente (empate: f2
        decrescente); o posto de um ponto e 1 + o maior posto cujo maior f2 ja visto e >= ao seu f2 (pontos identicos ficam na mesma frente)."""
        import bisect
        k = len(P)
        order = np.lexsort((-P[:, 1], -P[:, 0]))
        ranks = np.zeros(k, dtype=int)
        neg_best: list[float] = []          # -(maior f2 de cada frente); nao decrescente em r
        prev = None
        prev_rank = 0
        for idx in order:
            f1, f2 = P[idx, 0], P[idx, 1]
            if prev is not None and f1 == prev[0] and f2 == prev[1]:
                ranks[idx] = prev_rank
                continue
            r = bisect.bisect_right(neg_best, -f2)       # numero de frentes cujo maior f2 e >= f2  => posto = r
            if r == len(neg_best):
                neg_best.append(-f2)
            elif -f2 < neg_best[r]:
                neg_best[r] = -f2
            ranks[idx] = r
            prev, prev_rank = (f1, f2), r
        return ranks

    def run(self) -> SMSEMOAResult:
        cfg, inst, rng = self.cfg, self.inst, self.rng
        t0 = time.perf_counter()
        pop = self._base._initialize(cfg.pop_size)
        objs = [inst.evaluate(s) for s in pop]
        arch_s, arch_o = [], []
        arch_s, arch_o = self._base._update_archive(arch_s, arch_o, pop, objs)
        n_evals = len(pop)
        mu = len(pop)
        while n_evals < cfg.max_evals:
            if cfg.time_limit_s is not None and time.perf_counter() - t0 >= cfg.time_limit_s:
                break
            i, j = rng.integers(mu), rng.integers(mu)
            if rng.random() < cfg.crossover_prob:
                c1, _ = self._base._crossover(pop[i], pop[j], rng)
            else:
                c1 = pop[i].copy()
            if rng.random() < cfg.mutation_prob:
                c1 = self._base._mutate(c1, rng)
            if hasattr(inst, "repair"):
                c1 = inst.repair(c1)
            f1 = inst.evaluate(c1)
            n_evals += 1
            arch_s, arch_o = self._base._update_archive(arch_s, arch_o, [c1], [f1])
            pop.append(c1)
            objs.append(f1)
            # remocao: pior frente (nao dominacao), menor contribuicao de HV
            P = np.array(objs, float)
            lo, hi = P.min(0), P.max(0)
            Pn = (P - lo) / np.where(hi > lo, hi - lo, 1.0)
            rk = self._ranks_2d(P)
            last = np.flatnonzero(rk == rk.max())
            if len(last) == 1:
                victim = int(last[0])
            else:
                victim = int(last[int(np.argmin(self._contributions(Pn[last])))])
            pop.pop(victim)
            objs.pop(victim)
        return SMSEMOAResult(pareto_front=np.asarray(pareto_front(np.array([list(f) for f in arch_o])), float).reshape(-1, 2), n_evals=n_evals)
