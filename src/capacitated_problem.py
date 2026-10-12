"""
capacitated_problem.py -- Carregamento e avaliacao do Capacitated Dispersion
Problem (CDP) e do Generalized Dispersion Problem (GDP, modelo "constante").

Ambos sao variantes do Maximum Diversity Problem (problem.py) em que, em vez
de selecionar exatamente m elementos, seleciona-se um subconjunto de TAMANHO
LIVRE que satisfaz uma restricao de capacidade minima (e, no GDP, tambem uma
restricao de orcamento maximo). Formulacao verificada contra a literatura
(nao adivinhada -- ver referencias abaixo):

CDP (Capacitated Dispersion Problem)
    Marti, Martinez-Gavara & Sanchez-Oro, "The Capacitated Dispersion
    Problem. A mathematical model and a scatter search metaheuristic"
    (2021); ver tambem Peiro et al., "A Learnheuristic Algorithm for the
    Capacitated Dispersion Problem under Dynamic Conditions", Algorithms
    16(12):532 (2023).
        max  min_{i<j, i,j in S} d_ij           (MaxMin -- objetivo original)
        s.a. sum_{i in S} c_i >= B              (capacidade minima)
             S subconjunto de V, SEM restricao de cardinalidade
    Esta implementacao AMPLIA para bi-objetivo (MaxSum, MaxMin), replicando o
    par de objetivos usado no resto deste projeto para o MDP -- nao e o que
    o artigo original do CDP faz (que e mono-objetivo, so MaxMin).

GDP (Generalized Dispersion Problem), modelo "constante"
    Lozano-Osorio, Martinez-Gavara, Marti & Duarte, "Max-min dispersion with
    capacity and cost for a practical location problem", Expert Systems with
    Applications 202:117237 (2022).
        max  min_{i<j, i,j in S} d_ij
        s.a. sum_{i in S} c_i >= B              (capacidade minima)
             sum_{i in S} a_i <= K1             (orcamento maximo)
    Tambem ampliado para bi-objetivo aqui, pelo mesmo motivo do CDP acima.

    O arquivo de instancia GDP tambem grava um segundo custo por elemento
    (b: custo por unidade de material armazenado) e um segundo orcamento
    (K2: orcamento extra no "modelo variavel" citado em
    instances/instance_format_GDP.txt). NAO implementei o modelo variavel --
    carreguei b/K2 e exponho como atributos (.unit_cost, .extra_budget) mas
    evaluate()/is_feasible() usam so o modelo constante (a, c, K1, B) acima,
    que consegui verificar com confianca contra a literatura (Lozano-Osorio
    et al. 2022). Nao encontrei o artigo que define precisamente o modelo
    variavel (quais variaveis de decisao, que papel exato b/K2 tem na
    funcao objetivo/restricoes) -- carreguei os campos sem usa-los, em vez
    de adivinhar a formulacao. Ver secao 10 do artefato de diagnostico.

Formato de arquivo -- CONFIRMADO por inspecao programada de 100% dos
arquivos (100 CDP + 200 GDP), nao so por amostragem (ver validacao no
bloco __main__ deste arquivo):

CDP (instances/CDP (Const.)/*.txt) -- formato JA REFORMATADO em disco, NAO
bate com instances/instance_format_CDP.txt (que descreve uma lista de
arestas em triangulo -- alguem reformatou os dados em algum momento antes
deste projeto herda-los):
    linha 1        : n
    linha 2        : (em branco)
    linha 3        : B                          (1 valor)
    linha 4        : (em branco)
    linha 5        : c_1 c_2 ... c_n             (n valores, 1 linha)
    linha 6        : (em branco)
    linhas 7..6+n  : matriz D densa n x n, simetrica, diagonal zero

GDP (instances/GDP (Const.)/*.txt) -- formato ORIGINAL, bate exatamente com
instances/instance_format_GDP.txt (lista de arestas em triangulo superior):
    linha 1                     : n
    linhas 2..1+n(n-1)/2        : i j d            (arestas, 1-indexado)
    proximas n linhas           : i a_i b_i c_i    (1-indexado)
    ultima linha                : K1 K2 B
"""

from __future__ import annotations

import io
import zlib
import numpy as np
from pathlib import Path
from typing import Sequence


class _DispersionInstanceBase:
    """Base comum a CDPInstance e GDPInstance: matriz de distancia D (n x n)
    e avaliacao bi-objetivo (MaxSum, MaxMin) sobre subconjuntos de tamanho
    VARIAVEL, mais geracao de solucoes viaveis (respeitando capacidade
    minima e, quando aplicavel, orcamento maximo)."""

    n: int
    D: np.ndarray
    capacity: np.ndarray
    min_capacity: float
    # Limite de cardinalidade OPCIONAL (variante 'cardinality-bounded' do CDP/GDP, E19). None = problema original (tamanho livre,
    # dirigido pela capacidade). Ver set_max_size(). Com None nada muda (bit a bit).
    max_size = None
    filepath: Path
    name: str
    family: str

    # -- avaliacao ------------------------------------------------------

    def evaluate(self, solution: Sequence[int] | np.ndarray) -> tuple[float, float]:
        """Avalia uma solucao bi-objetivo (MaxSum, MaxMin). Tamanho livre
        (>=2) -- diferente de MDPInstance.evaluate(), que exige |S|==m."""
        sol = np.asarray(solution, dtype=np.intp)
        if sol.size < 2:
            raise ValueError(f"Solucao precisa de >=2 elementos (recebidos {sol.size})")
        sub = self.D[np.ix_(sol, sol)]
        iu = np.triu_indices(sol.size, k=1)
        vals = sub[iu]
        return float(vals.sum()), float(vals.min())

    def evaluate_swap(
        self,
        solution: Sequence[int] | np.ndarray,
        f: tuple[float, float],
        out_idx: int,
        in_elem: int,
    ) -> tuple[float, float]:
        """Avaliacao incremental de uma troca (out_idx -> in_elem), mesmo
        padrao de MDPInstance.evaluate_swap -- MaxSum em O(|S|). Generalizado
        para tamanho de solucao livre (usa sol.size em vez de self.m). Usado
        por BiObjBestSwap (operators.py) e por GRASPMO._local_search
        (grasp_mo.py).

        MaxMin: otimizado para O(|S|) no caso comum (antes: sempre O(|S|^2)).
        Verifica primeiro, em O(|S|) vetorizado, se out_elem participava de
        algum par que definia o MaxMin atual (f[1]). Se NAO participava,
        MaxMin(S \\ {out_elem}) == f[1] exatamente -- prova: remover um
        elemento so pode aumentar o min sobre os pares que sobram (um
        subconjunto de pares tem min >= o min do conjunto todo), e como o
        par que definia f[1] nao envolvia out_elem, ele continua presente
        em S \\ {out_elem}, entao o min sobre o que sobra tambem e <= f[1];
        logo e exatamente f[1]. Da MaxMin(S') = min(f[1], min_k D[in_elem,k]
        para k em remaining) -- O(|S|), sem reconstruir submatriz nenhuma.
        So cai na reconstrucao completa original (O(|S|^2)) no caso raro em
        que out_elem participava do par minimo -- resultado numerico
        identico ao codigo anterior nos dois casos, nao e uma aproximacao.
        Ganho relevante em CDP/GDP (solucoes maiores, guiadas por
        capacidade); no MDP o ganho pratico e menor mas o resultado e o
        mesmo."""
        sol = np.asarray(solution, dtype=np.intp)
        out_elem = int(sol[out_idx])
        mask = np.ones(sol.size, dtype=bool)
        mask[out_idx] = False
        remaining = sol[mask]

        gain = float(self.D[in_elem, remaining].sum())
        loss = float(self.D[out_elem, remaining].sum())
        new_maxsum = f[0] + gain - loss

        out_involved = bool(np.any(
            np.isclose(self.D[out_elem, remaining], f[1], rtol=0.0, atol=1e-9)
        ))
        if not out_involved:
            new_maxmin = min(f[1], float(self.D[in_elem, remaining].min()))
        else:
            new_sol = sol.copy()
            new_sol[out_idx] = in_elem
            sub = self.D[np.ix_(new_sol, new_sol)]
            iu = np.triu_indices(new_sol.size, k=1)
            new_maxmin = float(sub[iu].min())

        return float(new_maxsum), new_maxmin

    def evaluate_swaps_batch(
        self,
        solution: Sequence[int] | np.ndarray,
        f: tuple[float, float],
        out_idx: int,
        cands: Sequence[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Versao em lote de evaluate_swap: avalia a troca (out_idx -> c) para TODOS
        os candidatos c de uma vez. Resultado numericamente identico a chamar
        evaluate_swap em laco (verificado bit a bit em ~89 mil candidatos, incluindo o
        ramo raro em que out_elem define o MaxMin), mas calcula uma unica vez o que so
        depende de out_elem (loss, teste de participacao no MaxMin) e usa uma
        submatriz D[cands, remaining] em vez de 1 chamada Python por candidato."""
        sol = np.asarray(solution, dtype=np.intp)
        cands = np.asarray(cands, dtype=np.intp)
        out_elem = int(sol[out_idx])
        mask = np.ones(sol.size, dtype=bool)
        mask[out_idx] = False
        remaining = sol[mask]

        Dr = self.D[np.ix_(cands, remaining)]
        gain = Dr.sum(axis=1)
        loss = float(self.D[out_elem, remaining].sum())
        new_maxsum = f[0] + gain - loss

        out_involved = bool(np.any(
            np.isclose(self.D[out_elem, remaining], f[1], rtol=0.0, atol=1e-9)
        ))
        if not out_involved:
            rem_min = f[1]   # provado em evaluate_swap: MaxMin(S sem out) == f[1]
        else:
            # MaxMin(S') = min(MaxMin(S sem out), min_k D[in, k]); o 1o termo e comum a todos os
            # candidatos -> 1 calculo O(m^2) em vez de 1 por candidato. min e exato (sem aritmetica).
            if remaining.size >= 2:
                iu = np.triu_indices(remaining.size, k=1)
                rem_min = float(self.D[np.ix_(remaining, remaining)][iu].min())
            else:
                rem_min = np.inf
        new_maxmin = np.minimum(rem_min, Dr.min(axis=1))
        return new_maxsum, new_maxmin

    def build_from_seed(
        self,
        seed_elements: Sequence[int] | np.ndarray,
        candidate_pool: Sequence[int] | np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Parte de seed_elements (ja incluidos, nao removidos) e adiciona
        elementos de candidate_pool NA ORDEM DADA (o chamador decide se
        random/por eficiencia), pulando os que quebrariam orcamento, ate
        restaurar viabilidade (tamanho>=2 -- exigido por evaluate(), ver
        is_feasible -- e capacidade minima, e orcamento no GDP). Se o
        pool se esgotar antes de alcancar B, cai para
        _efficiency_greedy_fill() como garantia (ignora seed_elements
        nesse caso -- reconstroi do zero, mas so acontece se o pool
        fornecido for insuficiente, o que nao deveria ocorrer na pratica
        com um pool que cobre os elementos restantes). Usado por
        SubsetCrossover (operators.py) e por repair() abaixo."""
        selected = [int(x) for x in seed_elements]
        in_sol = set(selected)
        total_cost = sum(self._cost(i) for i in selected)
        total_cap = self.capacity_of(selected) if selected else 0.0
        if len(in_sol) >= 2 and total_cap >= self.min_capacity - 1e-9:
            return np.array(sorted(in_sol), dtype=np.intp)
        for raw in candidate_pool:
            idx = int(raw)
            if idx in in_sol:
                continue
            if self._can_extend(total_cost, idx):
                selected.append(idx)
                in_sol.add(idx)
                total_cost += self._cost(idx)
                total_cap += float(self.capacity[idx])
                if len(in_sol) >= 2 and total_cap >= self.min_capacity - 1e-9:
                    return np.array(sorted(in_sol), dtype=np.intp)
        return self._efficiency_greedy_fill()

    def repair(self, solution: Sequence[int] | np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Repara capacidade/orcamento (ver _repair_core) e, se max_size estiver definido, reduz o conjunto ate o limite de
        cardinalidade. Com max_size=None e identico a _repair_core."""
        out = self._repair_core(solution, rng)
        if self.max_size is not None and out.size > self.max_size:
            out = self._shrink_to_max(out)
        return out

    def _repair_core(self, solution: Sequence[int] | np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Se solution ja e viavel (tamanho>=2, capacidade minima e, no
        GDP, orcamento maximo -- ver is_feasible), retorna como esta.
        Caso contrario, repara em ate duas fases:

        Fase 1 (relevante so no GDP -- _budget_ok e sempre True na base,
        CDP nunca estoura orcamento): se o custo total excede o orcamento
        maximo, remove elementos em ordem CRESCENTE de eficiencia
        (capacidade/custo) -- descarta primeiro os que menos contribuem
        por unidade de orcamento -- ate caber no orcamento.

        Fase 2 (CDP e GDP): se a capacidade resultante ficar abaixo do
        minimo (a Fase 1 pode ter removido capacidade demais, ou a
        solucao de entrada so precisava mesmo de mais elementos), adiciona
        elementos externos em ordem DECRESCENTE de eficiencia (respeitando
        orcamento) via build_from_seed, com fallback garantido em
        _efficiency_greedy_fill.

        So remove elementos quando estritamente necessario para caber no
        orcamento (Fase 1) -- nunca para "economizar" quando ja viavel.
        Chamado genericamente pelo loop principal do MOEA/D-AOS
        (moead_aos.py) apos cada operator.apply() quando a instancia
        define repair (deteccao por duck-typing -- MDPInstance nao define
        este metodo, entao o MOEA/D para MDP nao chama isto e seu
        comportamento fica inalterado). Tambem usado por _grasp_init
        (moead_aos.py) como rede de seguranca da inicializacao guiada por
        peso, no caso raro de o RCL guiado por lam esgotar candidatos
        antes de alcancar B."""
        sol = np.asarray(solution, dtype=np.intp)
        if self.is_feasible(sol):
            return sol

        current = list(set(int(x) for x in sol.tolist()))
        if not self._budget_ok(sum(self._cost(i) for i in current)):
            current.sort(key=lambda i: self._efficiency(i))  # menos eficiente primeiro -> removido primeiro
            while current and not self._budget_ok(sum(self._cost(i) for i in current)):
                current.pop(0)

        in_sol = set(current)
        if len(in_sol) >= 2 and self.capacity_of(current) >= self.min_capacity - 1e-9:
            return np.array(sorted(in_sol), dtype=np.intp)

        remaining = [i for i in range(self.n) if i not in in_sol]
        remaining.sort(key=lambda i: -self._efficiency(i))
        return self.build_from_seed(current, remaining)

    def capacity_of(self, solution: Sequence[int] | np.ndarray) -> float:
        sol = np.asarray(solution, dtype=np.intp)
        return float(self.capacity[sol].sum())

    def is_feasible(self, solution: Sequence[int] | np.ndarray) -> bool:
        """CDP: tamanho >=2 (exigido pela propria definicao de MaxMin
        sobre pares) e capacidade minima. GDP sobrescreve para tambem
        checar orcamento maximo."""
        sol = np.asarray(solution, dtype=np.intp)
        return (sol.size >= 2 and self.capacity_of(sol) >= self.min_capacity - 1e-9
                and (self.max_size is None or sol.size <= self.max_size))

    def cost_of_element(self, idx: int) -> float:
        """Wrapper publico de _cost -- custo de incluir um unico elemento
        (0.0 no CDP, custo real no GDP). Usado por chamadores externos
        (grasp_mo.py) que precisam decidir, elemento a elemento, se cabe no
        orcamento durante uma construcao gulosa-randomica (RCL) nativamente
        ciente de orcamento, em vez de so reparar no final. Nao muda
        comportamento nenhum -- so expoe publicamente o que _cost ja fazia."""
        return self._cost(idx)

    def can_extend(self, current_total_cost: float, idx: int) -> bool:
        """Wrapper publico de _can_extend -- True se adicionar idx a uma
        solucao parcial com custo acumulado current_total_cost ainda
        respeita o orcamento maximo (sempre True no CDP, que nao tem
        orcamento). Mesmo motivo de cost_of_element acima."""
        return self._can_extend(current_total_cost, idx)

    # -- pontos de extensao (GDP sobrescreve) ----------------------------

    def _cost(self, idx: int) -> float:
        return 0.0

    def _can_extend(self, current_total_cost: float, idx: int) -> bool:
        return True

    def _extendable(self, current_total_cost: float, idxs: np.ndarray) -> np.ndarray:
        """Mascara booleana vetorizada equivalente a [_can_extend(total, i) for i in idxs]
        (mesma aritmetica; ver GDPInstance._extendable). Base/CDP: sempre True."""
        return np.ones(len(idxs), dtype=bool)

    def _best_pair(self) -> tuple[int, int]:
        """Par (i, j) de maior distancia (argmax de D sobre o triangulo superior). Depende so da
        instancia -> calculado uma vez e guardado, em vez de O(n^2) a cada greedy_solution()."""
        bp = getattr(self, "_best_pair_cache", None)
        if bp is None:
            iu, iv = np.triu_indices(self.n, k=1)
            k = int(np.argmax(self.D[iu, iv]))
            bp = (int(iu[k]), int(iv[k]))
            self._best_pair_cache = bp
        return bp

    def _efficiency_vec(self) -> np.ndarray:
        """np.array([_efficiency(i) for i in range(n)]), calculado uma vez por instancia."""
        ev = getattr(self, "_efficiency_cache", None)
        if ev is None:
            ev = np.array([self._efficiency(i) for i in range(self.n)])
            self._efficiency_cache = ev
        return ev

    def _budget_ok(self, total_cost: float) -> bool:
        """True se total_cost respeita o orcamento maximo (sempre True na
        base -- CDP nao tem orcamento). GDP sobrescreve. Usado por
        repair() para decidir se e preciso remover elementos (Fase 1)."""
        return True

    def _efficiency(self, idx: int) -> float:
        """Proxy de 'progresso em direcao a capacidade minima por unidade
        de orcamento gasto'. Usado para enviesar a construcao de solucoes
        (nao a avaliacao). CDP nao tem orcamento -- retorna so a
        capacidade (qualquer ordem eventualmente alcanca B, sem restricao
        de custo). GDP sobrescreve para capacity/cost."""
        return float(self.capacity[idx])

    def set_max_size(self, rho: float | None) -> int | None:
        """Ativa a variante com limite de cardinalidade: max_size = ceil(rho * k_ref), k_ref = tamanho da solucao viavel gulosa por
        eficiencia (_efficiency_greedy_fill; viavel por construcao, tambem no GDP). rho >= 1 garante que existe solucao viavel
        dentro do limite. rho=None desliga (problema original). Devolve o limite."""
        if rho is None:
            self.max_size = None
            return None
        if rho < 1.0:
            raise ValueError("rho precisa ser >= 1")
        self.max_size = None            # k_ref e calculado sem limite
        k_ref = int(len(self._efficiency_greedy_fill()))
        self.k_ref = k_ref
        self.max_size = int(np.ceil(rho * k_ref - 1e-9))
        return self.max_size

    def _shrink_to_max(self, solution: Sequence[int] | np.ndarray) -> np.ndarray:
        """Reduz a solucao a no maximo max_size elementos mantendo a viabilidade: remove, um a um, o elemento de MENOR contribuicao
        de MaxSum (soma das distancias aos demais) entre os que podem sair sem violar a capacidade minima; se nenhum puder sair,
        recorre a solucao gulosa por eficiencia (tamanho k_ref <= max_size). Deterministica (sem RNG)."""
        sol = [int(x) for x in np.asarray(solution, dtype=np.intp).tolist()]
        cap = self.capacity[np.asarray(sol, dtype=np.intp)].astype(float)
        tot_cap = float(cap.sum())
        while len(sol) > self.max_size:
            arr = np.asarray(sol, dtype=np.intp)
            contrib = self.D[np.ix_(arr, arr)].sum(axis=1)
            can = (tot_cap - cap) >= self.min_capacity - 1e-9
            if len(sol) <= 2 or not np.any(can):
                saved, self.max_size = self.max_size, None
                try:
                    return np.sort(self._efficiency_greedy_fill())
                finally:
                    self.max_size = saved
            idx = int(np.argmin(np.where(can, contrib, np.inf)))
            tot_cap -= float(cap[idx])
            del sol[idx]
            cap = np.delete(cap, idx)
        return np.array(sorted(sol), dtype=np.intp)

    def _efficiency_greedy_fill(self) -> np.ndarray:
        """Construcao guiada 100% por eficiencia (capacidade por orcamento
        gasto), ignorando diversidade -- usada como ultimo recurso quando
        random_solution/greedy_solution/build_from_seed nao encontram uma
        ordem viavel a tempo. Percorre os elementos em ordem decrescente
        de eficiencia e para assim que capacidade minima E tamanho>=2 sao
        atingidos; se nem essa ordem (a mais favoravel possivel a
        capacidade/orcamento) alcanca B, a instancia e de fato inviavel
        para o orcamento dado."""
        order = sorted(range(self.n), key=lambda i: -self._efficiency(i))
        selected: list[int] = []
        total_cap = 0.0
        total_cost = 0.0
        for idx in order:
            if self._can_extend(total_cost, idx):
                selected.append(idx)
                total_cap += float(self.capacity[idx])
                total_cost += self._cost(idx)
                if len(selected) >= 2 and total_cap >= self.min_capacity - 1e-9:
                    return np.array(selected, dtype=np.intp)
        raise RuntimeError(
            f"Instancia aparentemente inviavel (capacidade maxima alcancavel "
            f"dentro do orcamento = {total_cap:.2f} < B={self.min_capacity:.2f}): "
            f"{self.filepath}"
        )

    # -- geracao de solucoes ---------------------------------------------

    def random_solution(self, rng: np.random.Generator, max_attempts: int = 20) -> np.ndarray:
        sol = self._random_solution_core(rng, max_attempts)
        if self.max_size is not None and sol.size > self.max_size:
            sol = self._shrink_to_max(sol)
        return sol

    def _random_solution_core(self, rng: np.random.Generator, max_attempts: int = 20) -> np.ndarray:
        """Ordem aleatoria PONDERADA por eficiencia (amostragem de
        Efraimidis-Spirakis, ponderada por _efficiency), adiciona
        gulosamente (pulando os que quebrariam orcamento) ate satisfazer a
        capacidade minima. A ponderacao preserva aleatoriedade genuina mas
        evita gastar o orcamento cedo em elementos caros/pouco eficientes
        -- sem ela, boa parte das instancias GDP com orcamento apertado
        nunca alcancava B na pratica (verificado: com ordem uniforme, 50
        das 200 instancias GDP do catalogo falhavam; todas comprovadamente
        viaveis por um guloso puro de capacidade/custo -- ver __main__
        deste arquivo). Com a ponderacao por eficiencia mais o fallback de
        _efficiency_greedy_fill abaixo, as 300 instancias (100 CDP + 200
        GDP) constroem solucao viavel. Tenta ate max_attempts ordens
        ponderadas antes de cair no fallback garantido."""
        weights = np.array([max(self._efficiency(i), 1e-12) for i in range(self.n)])
        for _ in range(max_attempts):
            u = np.clip(rng.random(self.n), 1e-12, 1.0)
            keys = -np.log(u) / weights
            order = np.argsort(keys)
            selected: list[int] = []
            total_cap = 0.0
            total_cost = 0.0
            for raw_idx in order:
                idx = int(raw_idx)
                if self._can_extend(total_cost, idx):
                    selected.append(idx)
                    total_cap += float(self.capacity[idx])
                    total_cost += self._cost(idx)
                    if len(selected) >= 2 and total_cap >= self.min_capacity - 1e-9:
                        return np.array(selected, dtype=np.intp)
        # Nenhuma das tentativas aleatorias (ponderadas) alcancou B --
        # cai para a construcao 100% por eficiencia como garantia.
        return self._efficiency_greedy_fill()

    def greedy_solution(self, alpha: float = 0.0, rng: np.random.Generator | None = None,
                         efficiency_weight: float = 0.5) -> np.ndarray:
        sol = self._greedy_solution_core(alpha, rng, efficiency_weight)
        if self.max_size is not None and sol.size > self.max_size:
            sol = self._shrink_to_max(sol)
        return sol

    def _greedy_solution_core(self, alpha: float = 0.0, rng: np.random.Generator | None = None,
                              efficiency_weight: float = 0.5) -> np.ndarray:
        """GRASP-guloso por um escore misto: contribuicao de diversidade
        (mesmo padrao de MDPInstance.greedy_solution, normalizada em
        [0,1]) E eficiencia (_efficiency, tambem normalizada), parando
        assim que a capacidade minima e atingida em vez de num m fixo. Sem
        o termo de eficiencia (efficiency_weight=0, comportamento
        original), o guloso persegue so diversidade e tende a esgotar o
        orcamento em candidatos caros/pouco eficientes antes de alcancar B
        -- verificado: so com o termo de diversidade, 102 das 200
        instancias GDP falhavam, todas comprovadamente viaveis (ver
        random_solution). Com o termo de eficiencia mais o fallback de
        _efficiency_greedy_fill abaixo, as 300 instancias do catalogo
        constroem solucao viavel. efficiency_weight=1 e guloso puro por
        eficiencia (sem diversidade)."""
        if rng is None:
            rng = np.random.default_rng()
        i0, j0 = self._best_pair()
        selected = [i0, j0]
        in_sol = np.zeros(self.n, dtype=bool)
        in_sol[selected] = True
        contrib = self.D[selected].sum(axis=0)
        total_cost = self._cost(i0) + self._cost(j0)
        cap = self.capacity_of(selected)
        efficiency = self._efficiency_vec()

        while cap < self.min_capacity - 1e-9:
            remaining = np.where(~in_sol)[0]
            candidates = remaining[self._extendable(total_cost, remaining)].astype(np.intp, copy=False)
            if candidates.size == 0:
                # A construcao mista (diversidade+eficiencia) ficou presa
                # -- provavel que escolhas anteriores tenham comprometido
                # orcamento demais. Cai para a construcao 100% por
                # eficiencia como garantia, em vez de falhar numa
                # instancia que e comprovadamente viavel.
                return self._efficiency_greedy_fill()
            c_vals = contrib[candidates]
            e_vals = efficiency[candidates]
            c_rng = c_vals.max() - c_vals.min()
            e_rng = e_vals.max() - e_vals.min()
            c_norm = (c_vals - c_vals.min()) / c_rng if c_rng > 0 else np.zeros_like(c_vals)
            e_norm = (e_vals - e_vals.min()) / e_rng if e_rng > 0 else np.zeros_like(e_vals)
            score = (1.0 - efficiency_weight) * c_norm + efficiency_weight * e_norm
            s_min, s_max = score.min(), score.max()
            if s_max > s_min:
                threshold = s_max - alpha * (s_max - s_min)
                rcl = candidates[score >= threshold]
            else:
                rcl = candidates
            chosen = int(rng.choice(rcl))
            selected.append(chosen)
            in_sol[chosen] = True
            contrib = contrib + self.D[chosen]
            total_cost += self._cost(chosen)
            cap += float(self.capacity[chosen])

        return np.array(selected, dtype=np.intp)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, family={self.family!r}, "
            f"n={self.n}, B={self.min_capacity})"
        )


class CDPInstance(_DispersionInstanceBase):
    """Instancia do Capacitated Dispersion Problem (formato reformatado em
    disco -- ver docstring do modulo)."""

    def __init__(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Instancia CDP nao encontrada: {filepath}")
        self.filepath = filepath
        self.name = filepath.stem
        self.family = filepath.parent.name

        with filepath.open("r") as fh:
            lines = fh.readlines()
        if len(lines) < 6:
            raise ValueError(f"Arquivo CDP truncado: {filepath}")

        n = int(lines[0].split()[0])
        if lines[1].strip() != "":
            raise ValueError(f"Formato CDP inesperado (linha 2 deveria ser vazia): {filepath}")
        b_tok = lines[2].split()
        if len(b_tok) != 1:
            raise ValueError(f"Formato CDP inesperado (linha 3 deveria ter so B): {filepath}")
        min_capacity = float(b_tok[0])
        if lines[3].strip() != "":
            raise ValueError(f"Formato CDP inesperado (linha 4 deveria ser vazia): {filepath}")
        cap_tok = lines[4].split()
        if len(cap_tok) != n:
            raise ValueError(
                f"Formato CDP inesperado (linha 5 tem {len(cap_tok)} valores, esperado n={n}): {filepath}"
            )
        capacity = np.array([float(x) for x in cap_tok], dtype=np.float64)
        if lines[5].strip() != "":
            raise ValueError(f"Formato CDP inesperado (linha 6 deveria ser vazia): {filepath}")

        matrix_lines = lines[6:6 + n]
        if len(matrix_lines) != n:
            raise ValueError(
                f"Formato CDP inesperado (matriz com {len(matrix_lines)} linhas, esperado n={n}): {filepath}"
            )
        try:
            D = np.loadtxt(io.StringIO("".join(matrix_lines)))
        except ValueError as exc:
            raise ValueError(f"Falha ao ler matriz densa CDP: {filepath} ({exc})") from exc
        if D.shape != (n, n):
            raise ValueError(f"Matriz CDP com shape {D.shape}, esperado ({n},{n}): {filepath}")

        self.n = n
        self.D = D.astype(np.float64)
        self.capacity = capacity
        self.min_capacity = min_capacity


class GDPInstance(_DispersionInstanceBase):
    """Instancia do Generalized Dispersion Problem, modelo constante
    (formato original em disco, lista de arestas -- ver docstring do
    modulo). Carrega tambem os campos do modelo variavel (unit_cost,
    extra_budget) sem usa-los em evaluate()/is_feasible() -- ver docstring."""

    def __init__(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Instancia GDP nao encontrada: {filepath}")
        self.filepath = filepath
        self.name = filepath.stem
        self.family = filepath.parent.name

        with filepath.open("r") as fh:
            lines = [ln for ln in fh.readlines()]
        if not lines:
            raise ValueError(f"Arquivo GDP vazio: {filepath}")

        n = int(lines[0].split()[0])
        n_edges = n * (n - 1) // 2
        expected_total = 1 + n_edges + n + 1
        if len(lines) != expected_total:
            raise ValueError(
                f"Formato GDP inesperado ({len(lines)} linhas, esperado {expected_total} "
                f"para n={n}): {filepath}"
            )

        edge_lines = lines[1:1 + n_edges]
        try:
            edges = np.loadtxt(io.StringIO("".join(edge_lines)))
        except ValueError as exc:
            raise ValueError(f"Falha ao ler arestas GDP: {filepath} ({exc})") from exc
        if edges.ndim == 1:
            edges = edges.reshape(1, -1)
        if edges.shape != (n_edges, 3):
            raise ValueError(f"Arestas GDP com shape {edges.shape}, esperado ({n_edges},3): {filepath}")
        ii = edges[:, 0].astype(np.intp) - 1  # 1-indexado no arquivo -> 0-indexado
        jj = edges[:, 1].astype(np.intp) - 1
        dd = edges[:, 2]
        if ii.min() < 0 or ii.max() >= n or jj.min() < 0 or jj.max() >= n:
            raise ValueError(f"Indices de aresta GDP fora de [0,n): {filepath}")
        D = np.zeros((n, n), dtype=np.float64)
        D[ii, jj] = dd
        D[jj, ii] = dd

        site_lines = lines[1 + n_edges:1 + n_edges + n]
        sites = np.array([[float(x) for x in ln.split()] for ln in site_lines], dtype=np.float64)
        if sites.shape != (n, 4):
            raise ValueError(f"Linhas de site GDP com shape {sites.shape}, esperado ({n},4): {filepath}")
        site_idx = sites[:, 0].astype(np.intp) - 1
        order = np.argsort(site_idx)
        if not np.array_equal(site_idx[order], np.arange(n)):
            raise ValueError(f"IDs de site GDP nao formam 1..n exatamente: {filepath}")
        cost = sites[order, 1]
        unit_cost = sites[order, 2]
        capacity = sites[order, 3]

        last = lines[-1].split()
        if len(last) != 3:
            raise ValueError(f"Ultima linha GDP deveria ter K1 K2 B (3 valores): {filepath}")
        max_budget, extra_budget, min_capacity = (float(x) for x in last)

        self.n = n
        self.D = D
        self.cost = cost
        self.unit_cost = unit_cost
        self.capacity = capacity
        self.max_budget = max_budget
        self.extra_budget = extra_budget
        self.min_capacity = min_capacity

    def _cost(self, idx: int) -> float:
        return float(self.cost[idx])

    def _can_extend(self, current_total_cost: float, idx: int) -> bool:
        return self._budget_ok(current_total_cost + float(self.cost[idx]))

    def _extendable(self, current_total_cost: float, idxs: np.ndarray) -> np.ndarray:
        # Mesma aritmetica de _can_extend/_budget_ok, vetorizada: (total + cost[i]) <= K + 1e-9
        return (current_total_cost + self.cost[np.asarray(idxs, dtype=np.intp)]) <= self.max_budget + 1e-9

    def _budget_ok(self, total_cost: float) -> bool:
        return total_cost <= self.max_budget + 1e-9

    def _efficiency(self, idx: int) -> float:
        return float(self.capacity[idx]) / max(float(self.cost[idx]), 1e-9)

    def is_feasible(self, solution: Sequence[int] | np.ndarray) -> bool:
        sol = np.asarray(solution, dtype=np.intp)
        if sol.size < 2:
            return False
        cap_ok = self.capacity_of(sol) >= self.min_capacity - 1e-9
        budget_ok = self._budget_ok(float(self.cost[sol].sum()))
        size_ok = self.max_size is None or sol.size <= self.max_size
        return bool(cap_ok and budget_ok and size_ok)


if __name__ == "__main__":
    import time
    from collections import defaultdict

    root = Path(__file__).parent.parent

    def _validate(cls, folder_name, label):
        files = sorted((root / "instances" / folder_name).glob("*.txt"))
        print(f"=== {label}: {len(files)} arquivos ===")
        tested = parse_err = feas_rand_fail = feas_greedy_fail = eval_err = 0
        t0 = time.perf_counter()
        by_n: dict = defaultdict(int)
        for fp in files:
            try:
                inst = cls(fp)
            except Exception as e:
                print(f"  [PARSE ERR] {fp.name}: {e}")
                parse_err += 1
                continue
            by_n[inst.n] += 1
            tested += 1
            rng = np.random.default_rng(zlib.crc32(inst.name.encode("utf-8")) % (2**31))
            try:
                sol_r = inst.random_solution(rng)
                if not inst.is_feasible(sol_r):
                    print(f"  [INFEASIBLE random] {fp.name}")
                    feas_rand_fail += 1
                f1r, f2r = inst.evaluate(sol_r)
            except Exception as e:
                print(f"  [RANDOM ERR] {fp.name}: {e}")
                feas_rand_fail += 1
                f1r = f2r = float("nan")
            try:
                sol_g = inst.greedy_solution(alpha=0.3, rng=rng)
                if not inst.is_feasible(sol_g):
                    print(f"  [INFEASIBLE greedy] {fp.name}")
                    feas_greedy_fail += 1
                f1g, f2g = inst.evaluate(sol_g)
                if f1g < f2g - 1e-6:
                    print(f"  [EVAL ERR] {fp.name}: MaxSum={f1g} < MaxMin={f2g}")
                    eval_err += 1
            except Exception as e:
                print(f"  [GREEDY ERR] {fp.name}: {e}")
                feas_greedy_fail += 1
        dt = time.perf_counter() - t0
        print(f"  tamanhos (n): {dict(sorted(by_n.items()))}")
        print(
            f"  testadas={tested} parse_err={parse_err} "
            f"falha_construcao_random={feas_rand_fail} falha_construcao_greedy={feas_greedy_fail} "
            f"eval_err={eval_err}  tempo={dt:.2f}s"
        )
        return parse_err + feas_rand_fail + feas_greedy_fail + eval_err

    total_bad = 0
    total_bad += _validate(CDPInstance, "CDP (Const.)", "CDP")
    total_bad += _validate(GDPInstance, "GDP (Const.)", "GDP")
    print()
    print("TUDO OK" if total_bad == 0 else f"PROBLEMAS ENCONTRADOS: {total_bad}")
