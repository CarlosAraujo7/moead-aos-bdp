"""
grasp_mo.py -- GRASP multi-objetivo com arquivo de Pareto para MDP, CDP e GDP.

Implementa um GRASP multi-start adaptado ao contexto bi-objetivo (MaxSum + MaxMin),
utilizando um arquivo externo de solucoes nao-dominadas. Em cada iteracao, a
construcao e guiada por um escalar de pesos aleatorio (scalarizacao por Tchebycheff),
garantindo diversidade ao longo da frente de Pareto. Apos a construcao, uma busca
local por melhor melhoria e aplicada.

O path relinking e implementado como intensificacao final: para cada par de solucoes
no arquivo, os caminhos intermediarios sao percorridos e as solucoes nao-dominadas
adicionadas ao arquivo.

Compatibilidade CDP/GDP
------------------------
No MDP (instance.m definido), _construct() adiciona exatamente m-1 elementos via
RCL -- comportamento inalterado, inclusive a sequencia de chamadas de rng (mesma
seed reproduz os mesmos resultados de antes desta adaptacao).

No CDP/GDP (sem m), _construct() para quando a capacidade minima e atingida
(>=2 elementos), filtrando a CADA passo do RCL os candidatos que ainda respeitam
o orcamento maximo (inst.can_extend) -- construcao nativamente viavel na maior
parte do tempo, nao so reparada no final. Se ficar presa (RCL vazio antes de
alcancar capacidade), cai para inst.repair() como rede de seguranca, mesmo
padrao ja usado por _grasp_init (moead_aos.py) e por greedy_solution/
random_solution (capacitated_problem.py).

_local_search() usa sol.size no lugar de m (identico em espirito ao MDP) e ja
usa inst.evaluate_swap(), que capacitated_problem.py generaliza para tamanho
livre (mesmo metodo usado por BiObjBestSwap em operators.py) -- sem mudanca
estrutural. Como uma troca 1-por-1 pode derrubar capacidade/orcamento, cada
troca aceita passa por um "repair-net" duck-typed antes de ser adotada.

_generate_path() (path relinking) tambem recebe o mesmo repair-net a cada
passo. Quando as duas solucoes do par tem tamanhos diferentes (possivel em
CDP/GDP), o caminho pode nao alcancar exatamente a solucao-guia -- o loop so
para mais cedo ou deixa de incluir algum elemento exclusivo dela; isso nao
quebra nada (os pontos intermediarios continuam avaliados e podem entrar no
arquivo), so significa que o "caminho" nem sempre conecta literalmente as
duas pontas quando os tamanhos diferem.

instance aceita MDPInstance, CDPInstance ou GDPInstance
(capacitated_problem.py); type hint continua MDPInstance por consistencia
com o resto do projeto (contrato real e duck-typed).

Referencias
-----------
Colmenar, J.M., Martin-Moreno, R., Duarte, A., Garzon-Leiva, I. (2018).
    Grasp con busqueda local basada en trayectorias para el problema de la
    diversidad maxima biobjetivo. MAEB 2018.
Resende, M.G.C., Ribeiro, C.C. (2016). Optimization by GRASP: Greedy
    Randomized Adaptive Search Procedures. Springer.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

from problem import MDPInstance
from metrics import pareto_front, hypervolume


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

@dataclass
class GRASPMOConfig:
    """Hiperparametros do GRASP multi-objetivo para o MDP bi-objetivo."""

    max_iter: int = 300           # numero de construcoes GRASP
    alpha: float = 0.2            # grau de greedy na construcao (0=guloso, 1=aleatório)
    local_search_iter: int = 1    # passes BestImprovement por construcao (0 = desligado)
    path_relink: bool = True      # ativar path relinking na intensificacao
    pr_pairs: int = 5             # pares de solucoes para path relinking
    hv_ref_factor: float = 0.9
    log_every: int = 0
    seed: int | None = None
    # Limite de tempo de parede (s) do laco de construcoes, para comparacoes a TEMPO IGUAL (E4); None = so max_iter (historico).
    # O path relinking final nao e limitado (custo pequeno perto do laco).
    time_limit_s: float | None = None


# ---------------------------------------------------------------------------
# Resultado (compativel com MOEADResult/NSGA2Result)
# ---------------------------------------------------------------------------

@dataclass
class GRASPMOResult:
    pareto_front: np.ndarray
    hv_history: list[float]
    operator_selections: list[int]
    operator_counts: np.ndarray
    operator_q_values: np.ndarray
    n_gen_executed: int
    best_hv: float
    config: GRASPMOConfig


# ---------------------------------------------------------------------------
# Algoritmo
# ---------------------------------------------------------------------------

class GRASPMO:
    """GRASP multi-objetivo com arquivo de Pareto e path relinking."""

    def __init__(self, instance: MDPInstance, config: GRASPMOConfig | None = None) -> None:
        self.inst = instance
        self.cfg = config or GRASPMOConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------
    # Interface principal
    # ------------------------------------------------------------------

    def run(self) -> GRASPMOResult:
        cfg = self.cfg
        inst = self.inst
        rng = self.rng

        archive_sols: list[np.ndarray] = []
        archive_objs: list[tuple[float, float]] = []
        hv_history: list[float] = []
        import time as _time
        _t0 = _time.perf_counter()

        n_done = 0
        for it in range(cfg.max_iter):
            if cfg.time_limit_s is not None and _time.perf_counter() - _t0 >= cfg.time_limit_s:
                break
            n_done = it + 1
            # Scalarizacao aleatoria (pesos uniformes no simplex)
            lam = rng.dirichlet([1.0, 1.0])

            # Construcao GRASP guiada pelo escalar atual
            sol = self._construct(lam, rng)
            f = inst.evaluate(sol)

            # Busca local por melhor melhoria (scalarizacao)
            sol, f = self._local_search(sol, f, lam)

            # Atualiza arquivo
            archive_sols, archive_objs = self._update_archive(
                archive_sols, archive_objs, [sol], [f]
            )

            if cfg.log_every > 0 and (it + 1) % cfg.log_every == 0:
                ref_pt = self._ref_point(archive_objs)
                hv_val = hypervolume(archive_objs, ref_pt)
                hv_history.append(float(hv_val))

        # Path relinking: intensificacao final
        if cfg.path_relink and len(archive_sols) >= 2:
            archive_sols, archive_objs = self._path_relinking(
                archive_sols, archive_objs, rng
            )

        final_pts = np.array([list(f) for f in archive_objs])
        final_front = pareto_front(final_pts)
        ref_pt = self._ref_point(list(final_front))
        final_hv = hypervolume(final_front, ref_pt)

        return GRASPMOResult(
            pareto_front=final_front,
            hv_history=hv_history,
            operator_selections=[],
            operator_counts=np.zeros(4),
            operator_q_values=np.zeros(4),
            n_gen_executed=n_done if cfg.time_limit_s is not None else cfg.max_iter,
            best_hv=float(final_hv),
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Construcao GRASP
    # ------------------------------------------------------------------

    def _construct(self, lam: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Constroi uma solucao GRASP guiada pela scalarizacao linear com pesos lam.

        MDP (inst.m definido): adiciona exatamente m-1 elementos via RCL, igual
        a antes desta adaptacao -- mesma sequencia de chamadas de rng, zero
        mudanca de comportamento.

        CDP/GDP (sem m): para quando a capacidade minima e atingida (>=2
        elementos), filtrando a cada passo os candidatos do RCL que ainda
        respeitam o orcamento maximo (inst.can_extend) -- ver docstring do
        modulo. Se o RCL ficar vazio antes de alcancar a capacidade, cai para
        inst.repair() como rede de seguranca.
        """
        inst = self.inst
        n = inst.n
        m = getattr(inst, "m", None)
        alpha = self.cfg.alpha

        selected: list[int] = []
        available = list(range(n))

        # Adiciona primeiro elemento: escolha aleatoria.
        first = int(rng.integers(n))
        selected.append(first)
        available.remove(first)

        total_cost = 0.0
        total_cap = 0.0
        if m is None:
            total_cost = inst.cost_of_element(first)
            total_cap = float(inst.capacity[first])

        def _done() -> bool:
            if m is not None:
                return len(selected) >= m
            return len(selected) >= 2 and total_cap >= inst.min_capacity - 1e-9

        while not _done():
            if m is None:
                candidates = [a for a in available if inst.can_extend(total_cost, a)]
            else:
                candidates = available

            if not candidates:
                # RCL preso (orcamento esgotado antes da capacidade minima) --
                # inst.repair() completa em vez de falhar aqui.
                break

            # Avalia contribuicao de cada candidato
            scores = self._candidate_scores(selected, candidates, lam)

            # RCL: elementos com score >= max - alpha*(max - min)
            s_max, s_min = scores.max(), scores.min()
            threshold = s_max - alpha * (s_max - s_min)
            rcl = [candidates[i] for i in range(len(candidates)) if scores[i] >= threshold]

            chosen = int(rng.choice(rcl))
            selected.append(chosen)
            available.remove(chosen)
            if m is None:
                total_cost += inst.cost_of_element(chosen)
                total_cap += float(inst.capacity[chosen])

        sol = np.sort(np.array(selected, dtype=np.int64))
        if m is None and hasattr(inst, "repair"):
            sol = inst.repair(sol, rng)
        return sol

    def _candidate_scores(
        self,
        selected: list[int],
        available: list[int],
        lam: np.ndarray,
    ) -> np.ndarray:
        """Avalia o ganho escalar de adicionar cada candidato ao subconjunto atual."""
        inst = self.inst
        D = inst.D
        scores = np.zeros(len(available))

        if not selected:
            return scores

        sel_arr = np.array(selected)
        # Vetorizado (era 1 chamada Python de sum+min por candidato); a soma por linha da
        # submatriz equivale ao sum() do vetor D[cand, sel] -- verificado bit a bit.
        # delta_sum = contribuicao ao MaxSum; min_dist = contribuicao ao MaxMin.
        Dc = D[np.ix_(np.asarray(available, dtype=np.intp), sel_arr)]
        scores = lam[0] * Dc.sum(axis=1) + lam[1] * Dc.min(axis=1)

        return scores

    # ------------------------------------------------------------------
    # Busca local
    # ------------------------------------------------------------------

    def _local_search(
        self,
        sol: np.ndarray,
        f: tuple[float, float],
        lam: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Busca local BestImprovement via trocas 1x1 com avaliacao incremental.

        Utiliza inst.evaluate_swap para calcular o delta de MaxSum em
        O(sol.size) (em vez de O(sol.size**2)) -- metodo generalizado para
        tamanho livre em capacitated_problem.py (mesmo usado por
        BiObjBestSwap, operators.py), entao funciona em MDP e CDP/GDP sem
        mudanca estrutural aqui.
        """
        if self.cfg.local_search_iter <= 0:
            return sol, f

        inst = self.inst
        n = inst.n

        # Mascara booleana de pertencimento
        in_sol = np.zeros(n, dtype=bool)
        in_sol[sol] = True
        outside = np.where(~in_sol)[0]

        improved = True
        iters = 0
        while improved and iters < self.cfg.local_search_iter:
            improved = False
            iters += 1
            best_delta = 1e-9          # melhoria minima exigida
            best_out_idx = -1
            best_in_el = -1
            best_new_f: tuple[float, float] = f
            cur_scalar = self._scalar(f, lam)

            if hasattr(inst, "evaluate_swaps_batch") and outside.size > 0:
                # Em lote por elemento removido. O laco sequencial com `delta > best_delta`
                # estrito equivale a escolher a PRIMEIRA ocorrencia do maximo na ordem
                # (out_idx, in_el crescentes); aqui: argmax (1a ocorrencia) por out_idx e
                # atualiza so se o maximo local for estritamente maior -> mesmo resultado.
                for out_idx in range(sol.size):
                    ms, mm = inst.evaluate_swaps_batch(sol, f, out_idx, outside)
                    deltas = (lam[0] * ms + lam[1] * mm) - cur_scalar
                    k = int(np.argmax(deltas))
                    if deltas[k] > best_delta:
                        best_delta = float(deltas[k])
                        best_out_idx = out_idx
                        best_in_el = int(outside[k])
                        best_new_f = (float(ms[k]), float(mm[k]))
            else:
                for out_idx in range(sol.size):
                    for in_el in outside:
                        new_f = inst.evaluate_swap(sol, f, out_idx, int(in_el))
                        delta = self._scalar(new_f, lam) - cur_scalar
                        if delta > best_delta:
                            best_delta = delta
                            best_out_idx = out_idx
                            best_in_el = int(in_el)
                            best_new_f = new_f

            if best_out_idx >= 0:
                new_sol = sol.copy()
                new_sol[best_out_idx] = best_in_el
                new_f = best_new_f
                if hasattr(inst, "repair"):
                    # A troca pode ter derrubado capacidade/orcamento -- repara
                    # e reavalia (repair pode inclusive mudar o tamanho).
                    new_sol = inst.repair(new_sol)
                    new_f = inst.evaluate(new_sol)
                sol = new_sol
                f = new_f
                in_sol = np.zeros(n, dtype=bool)
                in_sol[sol] = True
                outside = np.where(~in_sol)[0]
                improved = True

        return sol, f

    @staticmethod
    def _scalar(f: tuple[float, float], lam: np.ndarray) -> float:
        return lam[0] * f[0] + lam[1] * f[1]

    # ------------------------------------------------------------------
    # Path relinking
    # ------------------------------------------------------------------

    def _path_relinking(
        self,
        archive_sols: list[np.ndarray],
        archive_objs: list[tuple[float, float]],
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """Percorre caminhos entre pares de solucoes no arquivo e insere
        solucoes intermediarias nao-dominadas."""
        inst = self.inst
        n_pairs = min(self.cfg.pr_pairs, len(archive_sols) * (len(archive_sols) - 1) // 2)

        for _ in range(n_pairs):
            if len(archive_sols) < 2:
                # O arquivo pode encolher DENTRO deste laco -- cada
                # _update_archive() abaixo pode podar solucoes dominadas
                # (inclusive as usadas como s_init/s_guide em iteracoes
                # anteriores). n_pairs foi calculado uma unica vez, no
                # tamanho do arquivo ANTES do laco comecar, entao deixa de
                # ser valido se o arquivo encolher demais no meio do
                # caminho. Sem esta guarda, rng.choice(..., replace=False)
                # quebra com ValueError quando sobra menos de 2 solucoes --
                # bug real, pre-existente (nao introduzido na adaptacao
                # CDP/GDP), reproduzido com GRASPMOConfig(max_iter=5,
                # path_relink=True, seed=12345) na instancia CDP
                # GKD-c_01_n500_b02_m50. Parar cedo aqui e o comportamento
                # correto: nao ha mais pares distintos pra path-relink.
                break
            idx = rng.choice(len(archive_sols), size=2, replace=False)
            s_init = archive_sols[idx[0]].copy()
            s_guide = archive_sols[idx[1]].copy()

            # Gera solucoes ao longo do caminho de s_init -> s_guide
            path_sols, path_objs = self._generate_path(s_init, s_guide)
            archive_sols, archive_objs = self._update_archive(
                archive_sols, archive_objs, path_sols, path_objs
            )

        return archive_sols, archive_objs

    def _generate_path(
        self,
        s_init: np.ndarray,
        s_guide: np.ndarray,
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """Gera o caminho entre s_init e s_guide via trocas elementares.

        A cada passo, um elemento exclusivo de s_guide e adicionado e o
        elemento de s_init com menor contribuicao ao MaxSum e removido.
        """
        inst = self.inst
        D = inst.D
        path_sols = []
        path_objs = []

        current = s_init.copy()
        add_set = set(s_guide.tolist()) - set(current.tolist())
        add_queue = list(add_set)

        for add_el in add_queue:
            remove_set = set(current.tolist()) - set(s_guide.tolist())
            if not remove_set:
                break

            # Remove o elemento de pior contribuicao marginal ao MaxSum
            remove_set_arr = np.array(sorted(remove_set))
            others = np.array([e for e in current if e not in remove_set], dtype=np.int64)
            if len(others) > 0:
                contribs = np.array([D[e, others].sum() for e in remove_set_arr])
                remove_el = int(remove_set_arr[contribs.argmin()])
            else:
                remove_el = int(remove_set_arr[0])

            new_sol = current.copy()
            new_sol = new_sol[new_sol != remove_el]
            new_sol = np.sort(np.append(new_sol, add_el))
            if hasattr(inst, "repair"):
                new_sol = inst.repair(new_sol)
            current = new_sol

            f = inst.evaluate(current)
            path_sols.append(current.copy())
            path_objs.append(f)

        return path_sols, path_objs

    # ------------------------------------------------------------------
    # Arquivo
    # ------------------------------------------------------------------

    @staticmethod
    def _update_archive(archive_sols, archive_objs, new_sols, new_objs):
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
        return (f"GRASPMO(iter={self.cfg.max_iter}, alpha={self.cfg.alpha}, "
                f"pr={self.cfg.path_relink})")


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

    # BUGFIX: "GKD-b" nunca existiu no catalogo (nao e uma familia real do
    # MDPLIB 2.0 -- ver instances/MDPLIB_2_0.pdf; o prefixo "GKD-b" so aparece
    # nos conjuntos separados CDP/GDP, que tem formato de arquivo incompativel
    # com MDPInstance). O filtro abaixo silenciosamente so retornava SOM-a.
    # Trocado por duas familias que realmente tem instancias com n<=50.
    targets = [r for r in rows
               if r["family"] in ("SOM-a", "real_team")
               and int(r["n"]) <= 50][:2]

    for row in targets:
        from problem import MDPInstance
        inst = MDPInstance(root / "instances" / row["filepath"])
        print(f"\nInstancia: {inst}")
        for pr in [False, True]:
            cfg = GRASPMOConfig(max_iter=250, alpha=0.3, path_relink=pr, seed=42)
            t0 = time.perf_counter()
            result = GRASPMO(inst, cfg).run()
            elapsed = time.perf_counter() - t0
            label = "GRASP+PR" if pr else "GRASP   "
            print(f"  {label}: |PF|={len(result.pareto_front):3d}  "
                  f"HV={result.best_hv:.4f}  t={elapsed:.3f}s")
