"""
operators.py — Pool de operadores de variação para o MOEA/D-AOS.

Oito operadores são definidos, todos com a assinatura mínima uniforme:

    apply(solution, instance, rng, f=None) -> (new_solution, new_f)

onde:
    solution  : np.ndarray de inteiros, shape (m,) — solução atual (não modificada)
    instance  : MDPInstance, CDPInstance ou GDPInstance (capacitated_problem.py) --
                os 8 operadores sao duck-typed: quando a instancia expoe
                min_capacity (CDP/GDP, tamanho de subconjunto LIVRE em vez
                de m fixo), cada operador usa solution.size no lugar de
                instance.m, e GraspGreedy/SubsetCrossover delegam a
                construcao/reparo aos metodos proprios da instancia
                (greedy_solution/build_from_seed). O loop principal do
                MOEA/D-AOS (moead_aos.py) chama instance.repair() depois
                de QUALQUER operador quando ela existe, garantindo
                capacidade minima e orcamento (GDP) independente de qual
                operador gerou a solucao. Nada disso e exercido para
                MDPInstance (nao define min_capacity nem repair) -- o
                comportamento fixo em m permanece inalterado.
    rng       : np.random.Generator
    f         : (maxsum, maxmin) ou None — valores atuais (usados para avaliação
                incremental quando disponíveis)

    new_solution : np.ndarray shape (m,) — nova solução proposta
    new_f        : (float, float) — avaliação bi-objetivo da nova solução

Alguns operadores aceitam parâmetros extras opcionais (injetados via
inspect.signature em moead_aos.py quando presentes na assinatura):
    lam, z_star        -- vetor de peso e ponto ideal do subproblema (BiObjBestSwap)
    partner, partner_f -- segunda solução-pai, extraída da mesma vizinhança
                          MOEA/D (SubsetCrossover)

Operadores implementados
------------------------
1. RandomSwap1x1     — remove 1 elemento aleatório, insere 1 aleatório
2. BestSwap1x1       — remove 1 elemento, busca a melhor substituição (best-improvement)
                       usando critério escalar lexicográfico (MaxSum primeiro)
3. RandomSwap2x2     — remove 2 elementos aleatórios, insere 2 aleatórios
4. GraspGreedy       — constrói nova solução do zero via GRASP (alpha ∈ [0, 0.3])
5. BiObjBestSwap     — troca 1x1 guiada pela escalarização de Tchebycheff do subproblema
6. MaxMinSwap        — remove o gargalo da distância mínima (MaxMin), reinsere o melhor candidato
7. LargePerturbation — remove e reconstrói k∈[3,5] elementos via GRASP (escape de ótimo local)
8. SubsetCrossover   — recombinação de dois pais (interseção preservada, diferença
                       simétrica particionada aleatoriamente); único operador do
                       pool que de fato recombina duas soluções distintas
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
        m = solution.size  # generalizado p/ CDP/GDP (sem .m); MDP: solution.size == instance.m sempre
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
        m = solution.size  # generalizado p/ CDP/GDP (sem .m); MDP: solution.size == instance.m sempre
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
        m = solution.size  # generalizado p/ CDP/GDP (sem .m); MDP: solution.size == instance.m sempre
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

        if hasattr(instance, "min_capacity"):
            # CDP/GDP: tamanho de solucao variavel, capacidade minima em
            # vez de m fixo -- delega para o GRASP guloso proprio da
            # instancia (capacitated_problem.py), que ja mistura
            # diversidade+eficiencia e para na capacidade minima em vez
            # de num m fixo. Caminho MDP abaixo fica 100% inalterado.
            sol = instance.greedy_solution(alpha, rng=rng)
            new_f = instance.evaluate(sol)
            return sol, new_f

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
        m = solution.size  # generalizado p/ CDP/GDP (sem .m); MDP: solution.size == instance.m sempre
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

        if hasattr(instance, "evaluate_swaps_batch"):
            # Caminho em lote (CDP/GDP): mesmo resultado do laco abaixo, bem mais rapido.
            ms, mm = instance.evaluate_swaps_batch(sol, f, out_idx, candidates)
            g_all = np.max(lam * (z_star - np.stack([ms, mm], axis=1)), axis=1)
            best_in = int(candidates[int(np.argmin(g_all))])
        else:
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
        m = solution.size  # generalizado p/ CDP/GDP (sem .m); MDP: solution.size == instance.m sempre
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
        m   = solution.size  # generalizado p/ CDP/GDP (sem .m); MDP: solution.size == instance.m sempre
        n   = instance.n
        D   = instance.D

        # CORRIGIDO -- bug real encontrado rodando o experimento completo:
        # instancias com m pequeno (ex. SOM-a com m=2) causavam
        # "ValueError: low >= high" aqui, porque min(k_max+1, m) podia cair
        # <= k_min quando m < k_min. Agora k e sempre um inteiro valido em
        # [1, m]: usa [k_min, k_max] quando m comporta, e cai para
        # perturbar todos os m elementos (o unico "grande" possivel) quando
        # a instancia e pequena demais para o intervalo pedido.
        lo = max(1, min(self.k_min, m))
        hi = max(lo, min(self.k_max, m)) + 1
        k = int(rng.integers(lo, hi))
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
# Operador 8 -- Crossover de subconjunto (recombinacao de dois pais)
# ---------------------------------------------------------------------------

class SubsetCrossover:
    """Recombina a solucao atual com uma parceira (partner): os elementos
    comuns aos dois pais sao preservados; os elementos exclusivos de cada
    um sao particionados aleatoriamente para compor o filho.

    Fiel ao MOEA/D original (Zhang & Li, 2007), que recombina solucoes de
    subproblemas vizinhos -- ao contrario dos demais operadores do pool
    (mutacao/reconstrucao de um unico pai), este e o unico operador de
    recombinacao de fato. Logica de particao identica a NSGA2._crossover
    (nsga2.py), adaptada para produzir um unico filho por chamada -- o que
    basta para o loop principal do MOEA/D-AOS, que substitui um individuo
    por vez.

    Quando nenhuma parceira e fornecida (partner=None, fora do contexto
    MOEA/D), cai de volta para uma solucao aleatoria da propria instancia,
    garantindo que o operador nunca falhe -- ex.: smoke test isolado em
    __main__.
    """

    name: str = "subset_crossover"

    def apply(
        self,
        solution: Solution,
        instance: MDPInstance,
        rng: np.random.Generator,
        f: Objectives | None = None,
        partner: Solution | None = None,
        partner_f: Objectives | None = None,
    ) -> tuple[Solution, Objectives]:
        if partner is None:
            partner = instance.random_solution(rng)

        set1 = set(solution.tolist())
        set2 = set(partner.tolist())
        common = np.array(sorted(set1 & set2), dtype=np.int64)
        exc1 = np.array(sorted(set1 - set2), dtype=np.int64)
        exc2 = np.array(sorted(set2 - set1), dtype=np.int64)

        pool = np.concatenate([exc1, exc2])
        rng.shuffle(pool)

        if hasattr(instance, "min_capacity"):
            # CDP/GDP: tamanho de solucao livre -- nao existe instance.m
            # (nao ha "need" fixo). Constroi a partir do "seed" comum aos
            # dois pais, adicionando genes exclusivos na ordem ja
            # aleatorizada acima ate restaurar capacidade minima
            # (respeita orcamento no GDP via build_from_seed --
            # capacitated_problem.py -- com fallback garantido).
            child = instance.build_from_seed(common, pool, rng=rng)
            new_f = instance.evaluate(child)
            return child, new_f

        need = instance.m - len(common)

        child = np.sort(np.concatenate([common, pool[:need]])).astype(np.intp)
        new_f = instance.evaluate(child)
        return child, new_f


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
    SubsetCrossover(),
]

OPERATOR_NAMES = [op.name for op in OPERATOR_POOL]


# ---------------------------------------------------------------------------
# Operadores EXTRAS, opt-in (E16, "pool envenenado"): nocivos ou inuteis de proposito, para testar se a selecao adaptativa os evita.
# Nao fazem parte de OPERATOR_POOL; entram so via MOEADConfig.extra_operators.
# ---------------------------------------------------------------------------

class RandomRestart:
    """Descarta a solucao e devolve uma solucao viavel aleatoria (destrutivo)."""

    name: str = "random_restart"

    def apply(self, solution, instance, rng, f=None):
        child = np.asarray(instance.random_solution(rng), dtype=np.intp)
        return child, instance.evaluate(child)


class HalfReplace:
    """Remove metade dos elementos ao acaso e insere o mesmo numero de elementos externos ao acaso (muito disruptivo)."""

    name: str = "half_replace"

    def apply(self, solution, instance, rng, f=None):
        sol = solution.copy()
        m = solution.size
        k = max(1, m // 2)
        out_idx = rng.choice(m, size=k, replace=False)
        in_solution = np.zeros(instance.n, dtype=bool)
        in_solution[sol] = True
        cand = np.where(~in_solution)[0]
        k = min(k, cand.size)
        sol[out_idx[:k]] = rng.choice(cand, size=k, replace=False)
        return sol, instance.evaluate(sol)


class NoOp:
    """Nao altera a solucao (desperdica a geracao do filho)."""

    name: str = "no_op"

    def apply(self, solution, instance, rng, f=None):
        sol = solution.copy()
        return sol, (f if f is not None else instance.evaluate(sol))


# ---------------------------------------------------------------------------
# Operadores que MUDAM O TAMANHO do conjunto (E18/E19; so CDP/GDP, onde o tamanho e livre). Opt-in via extra_operators.
# No CDP/GDP o MaxSum e monotono no tamanho (distancias >= 0): inserir um elemento cuja distancia minima aos membros e >= MaxMin atual nao
# reduz o MaxMin e aumenta o MaxSum, isto e, DOMINA (ver CLAUDE.md, "tamanho como dimensao escondida"). Os 8 operadores do pool trocam 1-por-1
# (ou reconstroem ate a capacidade) e nunca exploram essa direcao.
# ---------------------------------------------------------------------------

class BiObjGrow:
    """Insere o elemento externo que minimiza a funcao de Tchebycheff do subproblema (lam, z_star), aceitando perda de MaxMin se ela
    compensar. Respeita o orcamento (GDP) e o limite de cardinalidade (instance.max_size, se definido). Sem candidato, devolve a solucao."""

    name: str = "biobj_grow"

    def apply(self, solution, instance, rng, f=None, lam=None, z_star=None):
        sol = np.asarray(solution, dtype=np.intp).copy()
        if not hasattr(instance, "min_capacity"):          # MDP: tamanho fixo, operador nao se aplica
            return sol, (f if f is not None else instance.evaluate(sol))
        if f is None:
            f = instance.evaluate(sol)
        max_size = getattr(instance, "max_size", None)
        if max_size is not None and sol.size >= max_size:
            return sol, f
        if instance.n - sol.size <= 3:     # deixa >= 3 candidatos externos: os operadores de troca (2x2, perturbacao) exigem candidatos
            return sol, f
        in_solution = np.zeros(instance.n, dtype=bool)
        in_solution[sol] = True
        cands = np.flatnonzero(~in_solution)
        if hasattr(instance, "_extendable"):
            total_cost = float(sum(instance._cost(int(i)) for i in sol))
            cands = cands[instance._extendable(total_cost, cands)]
        if cands.size == 0:
            return sol, f
        if lam is None:
            lam = np.array([0.5, 0.5])
        if z_star is None:
            z_star = np.array([f[0] * 1.05 + 1.0, f[1] * 1.05 + 1.0])
        Dc = instance.D[np.ix_(cands, sol)]
        ms = f[0] + Dc.sum(axis=1)
        mm = np.minimum(f[1], Dc.min(axis=1))
        g = np.max(lam * (z_star - np.stack([ms, mm], axis=1)), axis=1)
        best = np.flatnonzero(g == g.min())
        pick = int(cands[int(rng.choice(best))])
        child = np.append(sol, pick).astype(np.intp)
        return child, instance.evaluate(child)


class BottleneckShrink:
    """Remove um extremo do par gargalo (o de menor distancia do conjunto) se o restante continua viavel (capacidade minima, >= 2 elementos);
    escolhe o extremo cuja saida deixa o maior MaxMin. Se nenhum pode sair, cai para MaxMinSwap (troca do gargalo)."""

    name: str = "bottleneck_shrink"

    def apply(self, solution, instance, rng, f=None):
        sol = np.asarray(solution, dtype=np.intp).copy()
        if not hasattr(instance, "min_capacity") or sol.size <= 2:
            return MaxMinSwap().apply(sol, instance, rng, f)
        m = sol.size
        sub = instance.D[np.ix_(sol, sol)]
        rows, cols = np.triu_indices(m, k=1)
        k = int(np.argmin(sub[rows, cols]))
        pair = [int(rows[k]), int(cols[k])]
        cap_tot = float(instance.capacity[sol].sum())
        opts = []
        for idx in pair:
            if cap_tot - float(instance.capacity[sol[idx]]) >= instance.min_capacity - 1e-9:
                rest = np.delete(sol, idx)
                if rest.size >= 2:
                    opts.append((float(instance.D[np.ix_(rest, rest)][np.triu_indices(rest.size, k=1)].min()), idx))
        if not opts:
            return MaxMinSwap().apply(sol, instance, rng, f)
        best_mm = max(o[0] for o in opts)
        idx = int(rng.choice([o[1] for o in opts if o[0] >= best_mm]))
        child = np.delete(sol, idx).astype(np.intp)
        return child, instance.evaluate(child)


# ---------------------------------------------------------------------------
# Busca local escalarizada (opt-in; E20, MOEA/D memetico): sobe a funcao de Tchebycheff do subproblema por trocas ate um otimo local.
# Os 8 operadores do pool dao UM movimento por filho; no MDP isso deixa o MOEA/D abaixo do Tabu/VNS ajustados (E11) nos extremos da frente.
# ---------------------------------------------------------------------------

class TchebLocalSearch:
    """Hill-climb de primeira melhora (por elemento removido, na ordem aleatoria; melhor candidato de entrada) sobre g_tch(lam, z_star) ate
    max_steps trocas ou otimo local. Troca 1x1 (tamanho constante). Em CDP/GDP so aceita trocas que mantem capacidade e orcamento."""

    name: str = "tcheb_ls"

    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = int(max_steps)

    def apply(self, solution, instance, rng, f=None, lam=None, z_star=None):
        sol = np.asarray(solution, dtype=np.intp).copy()
        if f is None:
            f = instance.evaluate(sol)
        if lam is None:
            lam = np.array([0.5, 0.5])
        if z_star is None:
            z_star = np.array([f[0] * 1.05 + 1.0, f[1] * 1.05 + 1.0])
        n = instance.n
        cap = getattr(instance, "capacity", None)
        B = getattr(instance, "min_capacity", None)
        cost = getattr(instance, "cost", None)
        K = getattr(instance, "max_budget", None)
        g_cur = float(np.max(lam * (z_star - np.asarray(f))))
        for _ in range(self.max_steps):
            in_sol = np.zeros(n, dtype=bool)
            in_sol[sol] = True
            cands = np.flatnonzero(~in_sol)
            if cands.size == 0:
                break
            capS = float(cap[sol].sum()) if cap is not None else 0.0
            costS = float(cost[sol].sum()) if (cost is not None and K is not None) else 0.0
            improved = False
            for oi in rng.permutation(sol.size):
                ms, mm = instance.evaluate_swaps_batch(sol, f, int(oi), cands)
                g = np.max(lam * (z_star - np.stack([ms, mm], axis=1)), axis=1)
                if cap is not None and B is not None:
                    o = int(sol[oi])
                    ok = (capS - cap[o] + cap[cands]) >= B - 1e-9
                    if cost is not None and K is not None:
                        ok &= (costS - cost[o] + cost[cands]) <= K + 1e-9
                    g = np.where(ok, g, np.inf)
                k = int(np.argmin(g))
                if g[k] < g_cur - 1e-12 * max(1.0, abs(g_cur)):
                    sol[oi] = int(cands[k])
                    f = instance.evaluate(sol)
                    g_cur = float(np.max(lam * (z_star - np.asarray(f))))
                    improved = True
                    break
            if not improved:
                break
        return sol, f


EXTRA_OPERATORS = {"random_restart": RandomRestart(), "half_replace": HalfReplace(), "no_op": NoOp(),
                   "biobj_grow": BiObjGrow(), "bottleneck_shrink": BottleneckShrink(),
                   "tcheb_ls": TchebLocalSearch(max_steps=10)}


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
