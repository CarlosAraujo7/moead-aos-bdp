"""Ancoragem da ponta de MaxMin da frente (CDP/GDP): busca tabu mono-objetivo de MaxMin.

Motivo (CLAUDE.md §2j/§2k): a ponta de MaxMin das frentes do MOEA/D (mesmo normalizado) fica bem abaixo dos
melhores valores conhecidos (MDG-b n=500: ~0,6 do BKS), enquanto um construtivo + tabu simples chega a 0,85-1,0
do BKS em ~30 s. Este modulo implementa essa busca para ser usada como passo final opcional do MOEAD_AOS
(`MOEADConfig.anchor_iters > 0`; padrao desligado = comportamento historico) ou isoladamente.

Estrutura do problema explorada: remover um no NUNCA reduz o MaxMin, entao as boas solucoes sao conjuntos
minimos em capacidade. Movimento: tira um no de um par gargalo, reconstroi pelo ponto mais distante (com tabu
nos nos recem-removidos e leve aleatoriedade) ate recuperar a capacidade B, e poda nos redundantes (gargalo
primeiro). No GDP a reconstrucao respeita o orcamento de custo (`inst._extendable`); se ficar presa, a
tentativa e descartada. Estilo da busca: destruicao/reconstrucao parcial com tabu (cf. Gomez et al. 2026,
Symmetry 18(1):110, e Martí et al. 2021 para o CDP).

Nao se aplica ao MDP (cardinalidade fixa m, sem capacidade): retorna None.
"""
from __future__ import annotations

import numpy as np


def _supported(inst) -> bool:
    return hasattr(inst, "min_capacity") and hasattr(inst, "capacity") and hasattr(inst, "_extendable")


def _cost_of(inst, S) -> float:
    return float(inst.cost[np.asarray(S, dtype=np.intp)].sum()) if hasattr(inst, "cost") else 0.0


def _maxmin(D: np.ndarray, S: list[int]) -> float:
    sub = D[np.ix_(S, S)]
    return float(sub[np.triu_indices(len(S), k=1)].min())


def _construct(inst, S: list[int], banned: np.ndarray, rng: np.random.Generator):
    """Adiciona o ponto mais distante do conjunto (com leve aleatoriedade) ate cumprir a capacidade minima."""
    D, cap, B, n = inst.D, inst.capacity, inst.min_capacity, inst.n
    S = list(S)
    ins = np.zeros(n, dtype=bool)
    ins[S] = True
    minv = D[:, S].min(axis=1)
    cap_s = float(cap[S].sum())
    cost = _cost_of(inst, S)
    all_idx = np.arange(n)
    while cap_s < B - 1e-9:
        ok = (~ins) & (~banned) & inst._extendable(cost, all_idx)
        if not ok.any():
            return None
        v = np.where(ok, minv, -np.inf)
        top = np.argsort(-v)[:3]
        top = top[np.isfinite(v[top])]
        c = int(top[rng.integers(len(top))]) if (rng.random() < 0.3 and len(top) > 1) else int(top[0])
        S.append(c)
        ins[c] = True
        cap_s += float(cap[c])
        cost += float(inst.cost[c]) if hasattr(inst, "cost") else 0.0
        minv = np.minimum(minv, D[c])
    return S


def _prune(inst, S: list[int]) -> list[int]:
    """Remove nos redundantes (capacidade >= B mantida), os de gargalo primeiro. Nunca piora o MaxMin."""
    D, cap, B = inst.D, inst.capacity, inst.min_capacity
    S = list(S)
    while len(S) > 2:
        tot = float(cap[S].sum())
        sub = D[np.ix_(S, S)].copy()
        np.fill_diagonal(sub, np.inf)
        order = np.argsort(sub.min(axis=1))
        for k in order:
            if tot - float(cap[S[int(k)]]) >= B - 1e-9:
                S.pop(int(k))
                break
        else:
            return S
    return S


def maxmin_anchor(inst, rng: np.random.Generator, iters: int = 2000, restarts: int = 3,
                  start: np.ndarray | None = None):
    """Busca tabu de MaxMin. Retorna (solucao ordenada, (MaxSum, MaxMin)) viavel, ou None se nao se aplica/falha.

    `start`: solucao inicial opcional (ex.: a de maior MaxMin do arquivo do MOEA/D) usada na 1a reinicializacao;
    as demais partem do par mais distante + reconstrucao."""
    if not _supported(inst):
        return None
    D, n = inst.D, inst.n
    iu, iv = np.triu_indices(n, k=1)
    k0 = int(np.argmax(D[iu, iv]))
    pair = [int(iu[k0]), int(iv[k0])]

    best, best_S = -np.inf, None
    for r in range(max(1, restarts)):
        if r == 0 and start is not None:
            S = list(map(int, start))
        else:
            S = _construct(inst, pair, np.zeros(n, dtype=bool), rng)
            if S is None:
                continue
        S = _prune(inst, S)
        if len(S) < 2 or not inst.is_feasible(np.array(S, dtype=np.intp)):
            continue
        cur = _maxmin(D, S)
        if cur > best:
            best, best_S = cur, list(S)
        tabu_until = np.zeros(n, dtype=int)
        for it in range(iters):
            sub = D[np.ix_(S, S)].copy()
            np.fill_diagonal(sub, np.inf)
            a, b = np.unravel_index(np.argmin(sub), sub.shape)
            u = S[int(a) if rng.random() < 0.5 else int(b)]
            banned = tabu_until > it
            banned[u] = True
            tabu_until[u] = it + 10 + int(rng.integers(10))
            S2 = _construct(inst, [x for x in S if x != u], banned, rng)
            if S2 is None:
                continue
            S2 = _prune(inst, S2)
            if len(S2) < 2:
                continue
            v = _maxmin(D, S2)
            if v >= cur - 1e-12 or rng.random() < 0.02:
                S, cur = S2, v
            if cur > best + 1e-12:
                best, best_S = cur, list(S)
    if best_S is None:
        return None
    sol = np.array(sorted(best_S), dtype=np.intp)
    if not inst.is_feasible(sol):
        return None
    f = inst.evaluate(sol)
    return sol, (float(f[0]), float(f[1]))
