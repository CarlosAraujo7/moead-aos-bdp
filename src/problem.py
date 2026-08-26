"""
problem.py -- Carregamento e avaliacao bi-objetivo do Maximum Diversity Problem.

Formato de instancia (MDPLib 2.0):

    linha 1  : n m
    linhas 2+: i j d    (lista de arestas, triangulo superior, 0-indexado)

Avaliacao bi-objetivo
---------------------
Dado um conjunto solucao S com |S| = m:
    MaxSum(S) = sum_{i<j, i,j in S} D[i,j]
    MaxMin(S) = min_{i<j, i,j in S} D[i,j]
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Sequence


class MDPInstance:
    """Instancia do Maximum Diversity Problem no formato MDPLib 2.0."""

    def __init__(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Instancia nao encontrada: {filepath}")

        self.filepath = filepath
        self.name = filepath.stem
        self.family: str = filepath.parent.name

        with filepath.open("r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        header = lines[0].split()
        self.n: int = int(header[0])
        self.m: int = int(header[1])

        if self.m < 2:
            raise ValueError(f"m deve ser >= 2 (m={self.m})")
        if self.m >= self.n:
            raise ValueError(f"m ({self.m}) deve ser < n ({self.n})")

        # Constroi matriz simetrica a partir da lista de arestas (i j d)
        D = np.zeros((self.n, self.n), dtype=np.float64)
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != 3:
                continue
            i, j, d = int(parts[0]), int(parts[1]), float(parts[2])
            D[i, j] = d
            D[j, i] = d

        self.D: np.ndarray = D

        # Pre-computa indices do triangulo superior (reutilizados em evaluate)
        self._triu_rows, self._triu_cols = np.triu_indices(self.m, k=1)

    # ------------------------------------------------------------------
    # Avaliacao
    # ------------------------------------------------------------------

    def evaluate(self, solution: Sequence[int] | np.ndarray) -> tuple[float, float]:
        """Avalia uma solucao bi-objetivo (MaxSum, MaxMin)."""
        sol = np.asarray(solution, dtype=np.intp)
        if sol.shape != (self.m,):
            raise ValueError(
                f"Solucao deve ter exatamente m={self.m} elementos "
                f"(recebidos {sol.shape[0]})"
            )
        sub  = self.D[np.ix_(sol, sol)]
        vals = sub[self._triu_rows, self._triu_cols]
        return float(vals.sum()), float(vals.min())

    def evaluate_swap(
        self,
        solution: np.ndarray,
        f: tuple[float, float],
        out_idx: int,
        in_elem: int,
    ) -> tuple[float, float]:
        """Avalia incrementalmente o efeito de uma troca (out_idx -> in_elem).

        MaxSum: delta exato em O(m).
        MaxMin: recalculo sobre a nova solucao (sem formula fechada eficiente).
        """
        out_elem = int(solution[out_idx])
        mask = np.ones(self.m, dtype=bool)
        mask[out_idx] = False
        remaining = solution[mask]

        gain = float(self.D[in_elem, remaining].sum())
        loss = float(self.D[out_elem, remaining].sum())
        new_maxsum = f[0] + gain - loss

        new_sol = solution.copy()
        new_sol[out_idx] = in_elem
        sub = self.D[np.ix_(new_sol, new_sol)]
        new_maxmin = float(sub[self._triu_rows, self._triu_cols].min())

        return float(new_maxsum), new_maxmin

    # ------------------------------------------------------------------
    # Geracao de solucoes
    # ------------------------------------------------------------------

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Gera uma solucao aleatoria (m elementos distintos de {0,...,n-1})."""
        return rng.choice(self.n, size=self.m, replace=False).astype(np.intp)

    def greedy_solution(self, alpha: float = 0.0) -> np.ndarray:
        """Construcao GRASP-greedy para MaxSum (alpha=0 guloso, alpha=1 aleatorio)."""
        rng = np.random.default_rng()
        iu, iv = np.triu_indices(self.n, k=1)
        best_pair = int(np.argmax(self.D[iu, iv]))
        selected = [int(iu[best_pair]), int(iv[best_pair])]
        in_sol = np.zeros(self.n, dtype=bool)
        in_sol[selected] = True
        contrib = self.D[selected].sum(axis=0)

        while len(selected) < self.m:
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
            contrib += self.D[chosen]

        return np.array(selected, dtype=np.intp)

    def __repr__(self) -> str:
        return (
            f"MDPInstance(name={self.name!r}, family={self.family!r}, "
            f"n={self.n}, m={self.m})"
        )


if __name__ == "__main__":
    import math, time
    from pathlib import Path
    from collections import defaultdict

    root = Path(__file__).parent.parent
    tested, errors = 0, 0
    counts: dict = defaultdict(int)

    for txt in sorted((root / "instances").rglob("*.txt")):
        fam = txt.parent.name
        if counts[fam] >= 2:
            continue
        counts[fam] += 1
        try:
            t0 = time.perf_counter()
            inst = MDPInstance(txt)
            t_load = time.perf_counter() - t0
        except Exception as e:
            print(f"  [ERR] {txt.name}: {e}")
            errors += 1
            continue
        rng = np.random.default_rng(42)
        sol = inst.random_solution(rng)
        f1, f2 = inst.evaluate(sol)
        pairs = math.comb(inst.m, 2)
        ok = f1 >= 0 and f2 >= 0 and f1 >= f2 * pairs - 1e-9
        status = "OK" if ok else "ERR"
        if not ok:
            errors += 1
        print(f"  [{status}] {inst}  MaxSum={f1:.2f}  MaxMin={f2:.4f}  t={t_load:.3f}s")
        tested += 1

    print(f"Testadas: {tested} | Erros: {errors}")
