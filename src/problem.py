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

import io
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

        # Constroi matriz simetrica a partir da lista de arestas (i j d).
        # CORRIGIDO nesta sessao -- performance: o parsing original fazia um
        # loop Python linha-a-linha (int()/float() por aresta), custando
        # ~2.4s so de parsing para as maiores instancias do catalogo (MDG-c,
        # n=3000, ~4.5M arestas). run_full.py recarrega a instancia do zero a
        # cada replica (nao ha cache entre modo/rep em _task()), entao esse
        # custo se repete centenas de vezes por instancia grande ao longo de
        # um experimento completo. Caminho rapido: parse vetorizado via
        # np.loadtxt (~2x mais rapido nas maiores instancias do catalogo --
        # verificado produzindo matriz D IDENTICA, np.array_equal, ao metodo
        # antigo em instancias de varios tamanhos antes de substituir). Com
        # validacao defensiva (exatamente 3 colunas, indices inteiros dentro
        # de [0,n)) -- qualquer falha cai automaticamente no loop linha-a-
        # linha original, preservando o comportamento exato para arquivos
        # atipicos ou malformados.
        D = np.zeros((self.n, self.n), dtype=np.float64)
        body_lines = lines[1:]
        fast_ok = False
        if body_lines:
            try:
                arr = np.loadtxt(io.StringIO("\n".join(body_lines)))
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    ii_f, jj_f, dd = arr[:, 0], arr[:, 1], arr[:, 2]
                    if np.all(ii_f == np.round(ii_f)) and np.all(jj_f == np.round(jj_f)):
                        ii = ii_f.astype(np.intp)
                        jj = jj_f.astype(np.intp)
                        if (ii.size and ii.min() >= 0 and ii.max() < self.n
                                and jj.min() >= 0 and jj.max() < self.n):
                            D[ii, jj] = dd
                            D[jj, ii] = dd
                            fast_ok = True
            except (ValueError, IndexError):
                fast_ok = False

        if not fast_ok:
            D[:] = 0.0
            for line in body_lines:
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

    def evaluate_swaps_batch(
        self,
        solution: np.ndarray,
        f: tuple[float, float],
        out_idx: int,
        cands,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Versao em lote de evaluate_swap para TODOS os candidatos `cands` de uma vez.

        MaxSum igual a evaluate_swap. MaxMin: se out_elem NAO participa de nenhum par
        que define o MaxMin atual f[1], entao MaxMin(S sem out) == f[1] exatamente e
        MaxMin(S') = min(f[1], min_k D[in, remaining]) em O(m) (mesma prova de
        CDPInstance.evaluate_swap); so no caso raro em que out_elem define o minimo
        recalcula a submatriz completa, candidato a candidato. Resultado numericamente
        identico a chamar evaluate_swap em laco (testado bit a bit). Antes, o MDP
        recalculava O(m^2) para cada candidato, o que dominava o custo em MDG-c (n=3000).
        Requer f[1] consistente com `solution` (como produzido por evaluate())."""
        solution = np.asarray(solution, dtype=np.intp)
        cands = np.asarray(cands, dtype=np.intp)
        out_elem = int(solution[out_idx])
        mask = np.ones(self.m, dtype=bool)
        mask[out_idx] = False
        remaining = solution[mask]

        Dr = self.D[np.ix_(cands, remaining)]
        gain = Dr.sum(axis=1)
        loss = float(self.D[out_elem, remaining].sum())
        new_maxsum = f[0] + gain - loss

        out_involved = bool(np.any(
            np.isclose(self.D[out_elem, remaining], f[1], rtol=0.0, atol=1e-9)
        ))
        if not out_involved:
            rem_min = f[1]
        else:
            # ver CDPInstance.evaluate_swaps_batch: termo comum calculado uma vez, exato.
            if remaining.size >= 2:
                iu = np.triu_indices(remaining.size, k=1)
                rem_min = float(self.D[np.ix_(remaining, remaining)][iu].min())
            else:
                rem_min = np.inf
        new_maxmin = np.minimum(rem_min, Dr.min(axis=1))
        return new_maxsum, new_maxmin

    # ------------------------------------------------------------------
    # Geracao de solucoes
    # ------------------------------------------------------------------

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Gera uma solucao aleatoria (m elementos distintos de {0,...,n-1})."""
        return rng.choice(self.n, size=self.m, replace=False).astype(np.intp)

    def greedy_solution(self, alpha: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        """Construcao GRASP-greedy para MaxSum (alpha=0 guloso, alpha=1 aleatorio).

        rng opcional (mesmo padrao de CDPInstance/GDPInstance.greedy_solution em
        capacitated_problem.py) -- se None, cria um gerador novo a cada chamada
        (comportamento original, preservado para chamadores que nao passam rng).
        Adicionado para corrigir nsga2.py::_initialize(): no init_mode="grasp" (o
        default), este metodo nunca aceitava um gerador semeado, entao a
        populacao inicial do NSGA2 nunca era exatamente reproduzivel so pela
        semente -- achado documentado na secao 11 do artefato de diagnostico."""
        if rng is None:
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
