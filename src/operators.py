"""
operators.py — Pool de operadores de variação para o MOEA/D-AOS.

Quatro operadores são definidos, todos com a assinatura uniforme:

    apply(solution, instance, rng, f=None) -> (new_solution, new_f)

onde:
    solution  : np.ndarray de inteiros, shape (m,) — solução atual (não modificada)
    instance  : MDPInstance
    rng       : np.random.Generator
    f         : (maxsum, maxmin) ou None — valores atuais (usados para avaliação
                incremental quando disponíveis)

    new_solution : np.ndarray shape (m,) — nova solução proposta
    new_f        : (float, float) — avaliação bi-objetivo da nova solução

Operadores implementados
------------------------
1. RandomSwap1x1   — remove 1 elemento aleatório, insere 1 aleatório
2. BestSwap1x1     — remove 1 elemento, busca a melhor substituição (best-improvement)
                     usando critério escalar lexicográfico (MaxSum primeiro)
3. RandomSwap2x2   — remove 2 elementos aleatórios, insere 2 aleatórios
4. GraspGreedy     — constrói nova solução do zero via GRASP (alpha ∈ [0, 0.3])
"""

from __future__ import annotations

import numpy as np
from problem import MDPInstance


# ---------------------------------------------------------------------------
# Tipo de retorno
# ---------------------------------------------------------------------------

Solution = np.ndarray                     # shape (m,), dtype intp
Objectives = tuple[float, float]          # (maxsum, maxmin)


# ---------------------------------------------------------------------------
# Operador 1 — Troca 1×1 aleatória
# ---------------------------------------------------------------------------

class RandomSwap1x1:
    """Remove um elemento aleatório da solução e insere um elemento aleatório externo."""

    name: str = "random_swap_1x1"

    def apply(
        self,
        solution: Solution,
        instance: MDPInstance,
        rng: np.random.Generator,
        f: Objectives | None = None,
    ) -> tuple[Solution, Objectives]:
        sol = solution.copy()
        m = instance.m
        n = instance.n

        out_idx = int(rng.integers(m))

        # Candidatos externos
        in_solution = np.zeros(n, dtype=bool)
        in_solution[sol] = True
        candidates = np.where(~in_solution)[0]
        in_elem = int(rng.choice(candidates))

        sol[out_idx] = in_elem
        new_f = instance.evaluate(sol)
        return sol, new_f


# ---------------------------------------------------------------------------
# Operador 2 — Troca 1×1 best-improvement
# ---------------------------------------------------------------------------

class BestSwap1x1:
    """Remove um elemento aleatório e insere o externo que maximiza MaxSum."""

    name: str = "best_swap_1x1"

    def apply(
        self,
        solution: Solution,
        instance: MDPInstance,
        rng: np.random.Generator,
        f: Objectives | None = None,
    ) -> tuple[Solution, Objectives]:
        sol = solution.copy()
        m = instance.m
        n = instance.n
        D = instance.D

        out_idx = int(rng.integers(m))
        out_elem = sol[out_idx]

        mask = np.ones(m, dtype=bool)
        mask[out_idx] = False
        remaining = sol[mask]

        # Para cada candidato externo, calcula contribuição marginal ao MaxSum
        in_solution = np.zeros(n, dtype=bool)
        in_solution[sol] = True
        candidates = np.where(~in_solution)[0]

        # contrib[c] = Σ_{j ∈ remaining} D[c, j]
        contrib = D[np.ix_(candidates, remaining)].sum(axis=1)

        best_pos = int(np.argmax(contrib))
        in_elem = int(candidates[best_pos])

        sol[out_idx] = in_elem
        new_f = instance.evaluate(sol)
        return sol, new_f


# ---------------------------------------------------------------------------
# Operador 3 — Troca 2×2 aleatória
# ---------------------------------------------------------------------------

class RandomSwap2x2:
    """Remove 2 elementos aleatórios e insere 2 elementos aleatórios externos."""

    name: str = "random_swap_2x2"

    def apply(
        self,
        solution: Solution,
        instance: MDPInstance,
        rng: np.random.Generator,
        f: Objectives | None = None,
    ) -> tuple[Solution, Objectives]:
        sol = solution.copy()
        m = instance.m
        n = instance.n

        # Sorteia 2 posições distintas para remover
        out_idxs = rng.choice(m, size=2, replace=False)

        in_solution = np.zeros(n, dtype=bool)
        in_solution[sol] = True
        candidates = np.where(~in_solution)[0]

        # Sorteia 2 externos distintos
        in_elems = rng.choice(candidates, size=2, replace=False)

        sol[out_idxs] = in_elems
        new_f = instance.evaluate(sol)
        return sol, new_f


# ---------------------------------------------------------------------------
# Operador 4 — GRASP greedy (reconstrução completa)
# ---------------------------------------------------------------------------

class GraspGreedy:
    """Constrói nova solução do zero via GRASP com alpha ∈ [0, alpha_max].

    alpha=0 → puramente guloso (MaxSum); alpha_max=0.3 por padrão.
    O alpha é amostrado uniformemente em cada chamada para diversidade.
    """

    name: str = "grasp_greedy"

    def __init__(self, alpha_max: float = 0.3) -> None:
        self.alpha_max = alpha_max

    def apply(
        self,
        solution: Solution,
        instance: MDPInstance,
        rng: np.random.Generator,
        f: Objectives | None = None,
    ) -> tuple[Solution, Objectives]:
        alpha = float(rng.uniform(0.0, self.alpha_max))
        n = instance.n
        m = instance.m
        D = instance.D

        # Passo 1: par inicial com maior distância
        iu, iv = np.triu_indices(n, k=1)
        best_pair = int(np.argmax(D[iu, iv]))
        selected = [int(iu[best_pair]), int(iv[best_pair])]
        in_sol = np.zeros(n, dtype=bool)
        in_sol[selected] = True

        # Contribuição acumulada de cada elemento para os já selecionados
        contrib = D[selected].sum(axis=0)   # shape (n,)

        while len(selected) < m:
            candidates = np.where(~in_sol)[0]
            c_vals = contrib[candidates]

            c_min, c_max = c_vals.min(), c_vals.max()
            if c_max > c_min:
                threshold = c_max - alpha * (c_max - c_min)
                rcl = candidates[c_vals >= threshold]
            else:
                rcl = candidates

            chosen = int(rng.choice(rcl))
            selected.append(chosen)
            in_sol[chosen] = True
            contrib += D[chosen]

        sol = np.array(selected, dtype=np.intp)
        new_f = instance.evaluate(sol)
        return sol, new_f


# ---------------------------------------------------------------------------
# Operador 5 — Troca 1×1 bi-objetivo (guiada por Tchebycheff)
# ---------------------------------------------------------------------------

class BiObjBestSwap:
    """Troca 1×1 guiada pela funcao de Tchebycheff do subproblema atual.

    Remove um elemento aleatorio e insere o candidato externo que minimiza
    g_tch(new_f | lam, z_star). Quando lam e z_star nao sao fornecidos,
    usa um vetor de pesos uniforme (0.5, 0.5) e z_star estimado a partir de f.

    Este operador e bi-objetivo consciente — diferentemente do BestSwap1x1
    que so maximiza MaxSum — o que o torna mais adequado ao contexto MOEA/D.
    """

    name: str = "biobj_best_swap"

    def apply(
        self,
        solution: "Solution",
        instance: "MDPInstance",
        rng: np.random.Generator,
        f: "Objectives | None" = None,
        lam: np.ndarray | None = None,
        z_star: np.ndarray | None = None,
    ) -> "tuple[Solution, Objectives]":
        sol = solution.copy()
        m = instance.m
        n = instance.n
        D = instance.D

        if f is None:
            f = instance.evaluate(sol)
        if lam is None:
            lam = np.array([0.5, 0.5])
        if z_star is None:
            z_star = np.array([f[0] * 1.05 + 1.0, f[1] * 1.05 + 1.0])

        out_idx = int(rng.integers(m))
        out_elem = sol[out_idx]

        mask = np.ones(m, dtype=bool)
        mask[out_idx] = False
        remaining = sol[mask]

        in_solution = np.zeros(n, dtype=bool)
        in_solution[sol] = True
        candidates = np.where(~in_solution)[0]

        best_g  = np.inf
        best_in = int(candidates[0])

        for cand in candidates:
            new_f = instance.evaluate_swap(sol, f, out_idx, int(cand))
            g = float(np.max(lam * (z_star - np.array(new_f))))
            if g < best_g:
                best_g  = g
                best_in = int(cand)

        sol[out_idx] = best_in
        new_f = instance.evaluate(sol)
        return sol, new_f


# ---------------------------------------------------------------------------
# Operador 6 -- Troca MaxMin (remove o gargalo da distancia minima)
# ---------------------------------------------------------------------------

class MaxMinSwap:
    """Remove o elemento que causa a menor distancia interna (gargalo MaxMin)
    e insere o candidato externo com maior contribuicao a MaxMin.
    """

    name: str = "maxmin_swap"

    def apply(
        self,
        solution,
        instance,
        rng,
        f=None,
    ):
        sol = solution.copy()
        m = instance.m
        n = instance.n
        D = instance.D

        sub = D[np.ix_(sol, sol)]
        rows, cols = np.triu_indices(m, k=1)
        vals = sub[rows, cols]
        min_idx = int(np.argmin(vals))
        pair = [int(sol[rows[min_idx]]), int(sol[cols[min_idx]])]
        out_elem = int(rng.choice(pair))
        out_idx  = int(np.where(sol == out_elem)[0][0])

        mask = np.ones(m, dtype=bool)
        mask[out_idx] = False
        remaining = sol[mask]

        in_solution = np.zeros(n, dtype=bool)
        in_solution[sol] = True
        candidates = np.where(~in_solution)[0]
        if len(candidates) == 0:
            return sol, instance.evaluate(sol)

        best_min = -1.0
        best_in  = int(candidates[0])
        for cand in candidates:
            md = float(D[cand, remaining].min())
            if md > best_min:
                best_min = md
                best_in  = int(cand)

        sol[out_idx] = best_in
        new_f = instance.evaluate(sol)
        return sol, new_f


# ---------------------------------------------------------------------------
# Operador 7 -- Perturbacao grande (remocao e reinsercao de k elementos)
# ---------------------------------------------------------------------------

class LargePerturbation:
    """Remove k elementos aleatorios e reinsercao gulosa (GRASP-like).

    Permite escapar de otimos locais profundos onde trocas 1x1 e 2x2
    nao conseguem melhorar. k e amostrado uniformemente em [k_min, k_max].
    """

    name: str = "large_perturbation"

    def __init__(self, k_min=3, k_max=5, alpha=0.2):
        self.k_min  = k_min
        self.k_max  = k_max
        self.alpha  = alpha

    def apply(
        self,
        solution,
        instance,
        rng,
        f=None,
    ):
        sol = solution.copy()
        m   = instance.m
        n   = instance.n
        D   = instance.D

        k = int(rng.integers(self.k_min, min(self.k_max + 1, m)))
        remove_idxs = rng.choice(m, size=k, replace=False)
        keep_mask   = np.ones(m, dtype=bool)
        keep_mask[remove_idxs] = False
        kept = sol[keep_mask].tolist()

        in_sol = np.zeros(n, dtype=bool)
        in_sol[kept] = True
        selected = list(kept)

        alpha = self.alpha
        contrib = D[selected].sum(axis=0) if selected else np.zeros(n)

        for _ in range(k):
            cands = np.where(~in_sol)[0]
            c_vals = contrib[cands]
            c_min, c_max = c_vals.min(), c_vals.max()
            if c_max > c_min:
                threshold = c_max - alpha * (c_max - c_min)
                rcl = cands[c_vals >= threshold]
            else:
                rcl = cands
            chosen = int(rng.choice(rcl))
            selected.append(chosen)
            in_sol[chosen] = True
            contrib += D[chosen]

        new_sol = np.sort(np.array(selected, dtype=np.intp))
        new_f   = instance.evaluate(new_sol)
        return new_sol, new_f


# ---------------------------------------------------------------------------
# Registro global do pool
# ---------------------------------------------------------------------------

OPERATOR_POOL = [
    RandomSwap1x1(),
    BestSwap1x1(),
    RandomSwap2x2(),
    GraspGreedy(alpha_max=0.3),
    BiObjBestSwap(),
    MaxMinSwap(),
    LargePerturbation(k_min=3, k_max=5),
]

OPERATOR_NAMES = [op.name for op in OPERATOR_POOL]


if __name__ == "__main__":
    from pathlib import Path
    import csv
    root = Path(__file__).parent.parent
    catalog = root / "instances" / "catalog.csv"
    with catalog.open() as fh:
        rows = list(csv.DictReader(fh))
    row = next((r for r in rows if r["family"] == "GKD-c"), None)
    if row:
        fname = row["name"] + ".txt"
        matches = list((root / "instances").rglob(fname))
        if matches:
            from problem import MDPInstance
            inst = MDPInstance(matches[0])
            rng = np.random.default_rng(0)
            sol = inst.random_solution(rng)
            f = inst.evaluate(sol)
            print("Instance:", inst)
            for op in OPERATOR_POOL:
                new_sol, new_f = op.apply(sol, inst, rng, f)
                print(f"  [{op.name}] MaxSum={new_f[0]:.4f} MaxMin={new_f[1]:.4f}")
