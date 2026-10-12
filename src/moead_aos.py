"""
moead_aos.py -- MOEA/D com Adaptive Operator Selection (UCB1/Thompson) para o MDP bi-objetivo.

Decomposicao MOEA/D por Tchebycheff (maximizacao):
    g_tch(x | lam, z*) = max( lam[0]*(z0*-f0(x)), lam[1]*(z1*-f1(x)) )

Modos de operacao:
    operator_selection="ucb"      -> MOEA/D-AOS via UCB1 (contribuicao principal)
    operator_selection="thompson" -> MOEA/D-AOS via Thompson Sampling
    operator_selection="random"   -> MOEA/D-random (baseline ablativo)

Modo de recompensa (quando AOS ativo):
    reward_mode="hypervolume" -> ganho de hipervolume (recomendado)
    reward_mode="scalar"      -> melhoria escalar Tchebycheff

Inicializacao:
    Warm-start GRASP guiado pelo vetor de pesos de cada subproblema,
    gerando populacao inicial que ja cobre a frente de Pareto desde a
    primeira geracao.

Referencias:
    Zhang & Li (2007) MOEA/D. IEEE TEC 11(6):712-731.
    DaCosta et al. (2008) AOS with MAB. GECCO 2008.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

from problem import MDPInstance
from operators import OPERATOR_POOL, OPERATOR_NAMES, EXTRA_OPERATORS
from mab import UCBSelector, RandomSelector, FixedSelector, SubsetRandomSelector, ThompsonSelector, DYTSSelector, DYTSBetaSelector, FRRMABSelector, ContextualSelector, scalar_reward, hypervolume_reward
from metrics import hypervolume, pareto_front


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

# Tolerancia RELATIVA para decidir se uma melhora/inclusao e real. O mesmo conjunto de elementos avaliado em outra ordem (p.ex. um filho do
# SubsetCrossover igual a um membro do arquivo) da objetivos que diferem ~1e-15 em termos relativos; sem tolerancia isso contava como 'sucesso'
# para seletores binarios (Thompson, DYTS) e para a recompensa 'insert' (descoberto em 10 out 2026; ver CLAUDE.md).
_REL_TOL = 1e-9


@dataclass
class MOEADConfig:
    """Hiperparametros do MOEA/D-AOS."""

    # Decomposicao
    n_weights: int = 20         # T: numero de subproblemas / vetores de peso
    neighborhood_size: int = 5  # tamanho da vizinhanca por subproblema

    # Criterio de parada
    max_gen: int = 300

    # Selecao de operadores
    operator_selection: Literal["ucb", "random", "thompson", "dyts", "dyts_beta", "frrmab"] = "ucb"

    # Modo de recompensa (quando AOS ativo)
    # "scalar"/"hypervolume": historicos (ganho ABSOLUTO). Novos, opt-in (Fase 2, CLAUDE.md §2p):
    # "scalar_rel" = melhora relativa do Tchebycheff em [0,1]; "insert" = 1 se o filho entra no arquivo nao-dominado.
    # "replace" (so com dyts_beta/ucb/thompson; fiel a Sun & Li 2020): 1 se o filho melhora ESTRITAMENTE o Tchebycheff de pelo menos um
    # subproblema da vizinhanca (FI > 0), 0 caso contrario; calculado na etapa de substituicao. Incompativel com cost_aware.
    # "frr" (so com frrmab; Li et al. 2014): FIR = soma, sobre os vizinhos cujo Tchebycheff o filho melhora, de (g_antigo - g_novo)/g_antigo.
    reward_mode: Literal["scalar", "hypervolume", "scalar_rel", "insert", "replace", "frr"] = "hypervolume"

    # Divide a recompensa pelo custo relativo (tempo medio movel) do operador. Padrao False.
    cost_aware: bool = False

    # Piso de exploracao: com probabilidade explore_mix o operador e sorteado UNIFORMEMENTE; senao, pelo seletor.
    # 0.0 = desligado (padrao; nao consome RNG extra). Fase 2: o pool uniforme e quase otimo (CLAUDE.md §2h/§2p).
    explore_mix: float = 0.0

    # Operadores extras (E16, 'pool envenenado'; opt-in): nomes de operators.EXTRA_OPERATORS acrescentados ao pool de 8. () = pool padrao.
    extra_operators: tuple = ()

    # Parametros UCB
    ucb_c: float = 0.5
    ucb_window: int = 50
    dyts_c: float = 100.0   # limiar C do DYTS de Sun & Li (selecao 'dyts_beta')
    # FRRMAB (Li et al. 2014; operator_selection='frrmab' + reward_mode='frr'): janela W = frr_window_frac*N, fator de escala C e decaimento D
    frr_window_frac: float = 0.5   # W = frr_window_frac * N (0,5 = padrao do artigo)
    frr_c: float = 5.0
    frr_d: float = 1.0

    # Selecao contextual (E3; opt-in): um seletor independente por contexto. context in {"none", "weight", "phase", "weight_phase"}:
    # "weight" = faixa do peso do subproblema (ctx_weight_bins faixas pelo peso de MaxSum), "phase" = terco/fase da busca
    # (ctx_phases fases por geracao). So com operator_selection em ucb/thompson/dyts/dyts_beta.
    context: str = "none"
    ctx_weight_bins: int = 3
    ctx_phases: int = 3

    # Inicializacao warm-start GRASP
    grasp_alpha: float = 0.3   # 0=guloso, 1=aleatorio

    # Ponto de referencia para HV
    hv_ref_factor: float = 0.9

    # Normaliza os objetivos pela amplitude observada (z_star - z_min) dentro
    # do Tchebycheff. Padrao False = comportamento historico (sem normalizar;
    # MaxSum ~1e4-1e5 domina MaxMin ~1e0-1e2 e achata os subproblemas).
    normalize_objectives: bool = False

    # Ablacao de operador unico: indice do operador a usar sempre (None = desligado).
    # Com fixed_operator definido, operator_selection e ignorado (nao ha AOS).
    fixed_operator: int | None = None

    # Ablacao leave-one-out: operadores excluidos da selecao uniforme (so com operator_selection="random").
    excluded_operators: tuple = ()

    # Ancoragem da ponta de MaxMin (src/anchor.py): se > 0, apos o laco principal roda uma busca tabu mono-objetivo
    # de MaxMin (iteracoes por reinicio) e insere o resultado no arquivo. 0 = desligado (comportamento historico).
    # So CDP/GDP (instancias com capacidade); ignorado no MDP.
    anchor_iters: int = 0
    anchor_restarts: int = 3
    # Motor da ancora: "tabu" = anchor.maxmin_anchor (historico); "sbts" = sbts_anchor.sbts_anchor (busca tabu baseada em solucoes, estilo
    # SBTS de Lu et al. 2023; usa `sbts_iters` se > 0, senao `anchor_iters`); "both" = roda as duas e insere os dois resultados no arquivo
    # (tabu com anchor_iters x anchor_restarts; SBTS com sbts_iters, 1 reinicio). So tem efeito com anchor_iters > 0.
    anchor_method: str = "tabu"
    sbts_iters: int = 0

    # Semeadura por busca de trajetoria (E20, hibrido para o MDP): se > 0 e houver time_limit_s, roda ANTES do laco principal um Tabu/VNS bi-objetivo
    # (bo_trajectory.BOTrajectory; parametros em seed_traj_params, ex. os ajustados no E11) durante seed_traj_frac * time_limit_s (a fracao CONTA
    # no limite) e injeta o arquivo resultante na populacao (cada solucao substitui os subproblemas que ela melhora) e no arquivo.
    seed_traj_frac: float = 0.0
    seed_traj_params: dict | None = None

    # Logging
    log_every: int = 0

    # Limite de tempo de parede (s), contado desde o INICIO de run() (inclui a inicializacao); verificado a cada geracao.
    # None = so max_gen (comportamento historico). Usado no E11 (MOEA/D ajustado a tempo igual).
    time_limit_s: float | None = None
    # E12 (opt-in): fracao do time_limit_s reservada a um POLIMENTO final do arquivo por busca local (bo_trajectory.polish): o laco principal
    # para em (1 - polish_frac) * time_limit_s e o resto e gasto intensificando o arquivo (extremos de MaxMin e MaxSum + path relinking).
    # 0.0 = desligado (padrao). Requer time_limit_s.
    polish_frac: float = 0.0

    # E14 (opt-in): estatisticas por operador (aplicacoes, filhos inviaveis antes do reparo, filhos que melhoram algum subproblema da vizinhanca,
    # filhos que entram no arquivo) devolvidas em MOEADResult.operator_stats. Nao consome RNG nem altera o resultado.
    track_ops: bool = False

    # Semente
    seed: int | None = None


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class MOEADResult:
    pareto_front: np.ndarray
    hv_history: list[float]
    operator_selections: list[int]
    operator_counts: np.ndarray
    operator_q_values: np.ndarray
    n_gen_executed: int
    best_hv: float
    config: MOEADConfig
    operator_stats: dict | None = None
    archive_sizes: list | None = None      # |S| de cada membro do arquivo (diagnostico; E18/E19)


# ---------------------------------------------------------------------------
# Algoritmo
# ---------------------------------------------------------------------------

class MOEAD_AOS:
    """MOEA/D com suporte a AOS (UCB1/Thompson) ou selecao aleatoria."""

    def __init__(self, instance: MDPInstance, config: MOEADConfig | None = None) -> None:
        self.inst = instance
        self.cfg = config or MOEADConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def run(self) -> MOEADResult:
        cfg = self.cfg
        inst = self.inst
        rng = self.rng
        _pool = list(OPERATOR_POOL) + [EXTRA_OPERATORS[_n] for _n in cfg.extra_operators]
        import time as _tl_time
        _tl_t0 = _tl_time.perf_counter()

        N = cfg.n_weights
        T = min(cfg.neighborhood_size, N)

        lambdas = self._generate_weights(N)
        neighborhoods = self._compute_neighborhoods(lambdas, T)

        # Warm-start: cada subproblema inicializado com GRASP guiado pelo seu peso
        solutions, objectives = self._initialize(N, lambdas)

        z_star = np.array([
            max(f[0] for f in objectives),
            max(f[1] for f in objectives),
        ])

        z_min = np.array([
            min(f[0] for f in objectives),
            min(f[1] for f in objectives),
        ])
        w_inv = np.ones(2)   # fator de normalizacao (1 = sem normalizar)

        def _refresh_w_inv():
            if cfg.normalize_objectives:
                rng_ = z_star - z_min
                w_inv[:] = np.where(rng_ > 1e-12, 1.0 / np.maximum(rng_, 1e-12), 1.0)

        _refresh_w_inv()

        archive_sols = list(solutions)
        archive_objs = list(objectives)
        archive_sols, archive_objs = self._update_archive(archive_sols, archive_objs, [], [])

        if cfg.seed_traj_frac > 0.0 and cfg.time_limit_s is not None:
            from bo_trajectory import BOTrajConfig, BOTrajectory
            _sp = dict(cfg.seed_traj_params or {})
            _meth = _sp.pop("method", "tabu")
            _rem = cfg.time_limit_s * cfg.seed_traj_frac - (_tl_time.perf_counter() - _tl_t0)
            _alg = BOTrajectory(inst, BOTrajConfig(method=_meth, seed=int(rng.integers(2**31 - 1)), time_limit_s=max(_rem, 0.0), **_sp))
            _alg.run()
            _ss = [np.asarray(x, dtype=np.intp) for x in _alg.arch_s]
            _so = [(float(a_), float(b_)) for a_, b_ in _alg.arch_f]
            for _f in _so:
                z_star[0] = max(z_star[0], _f[0]); z_star[1] = max(z_star[1], _f[1])
                z_min[0] = min(z_min[0], _f[0]); z_min[1] = min(z_min[1], _f[1])
            _refresh_w_inv()
            for _s, _f in zip(_ss, _so):
                for _i in range(N):
                    if self._tchebycheff(_f, lambdas[_i] * w_inv, z_star) < self._tchebycheff(objectives[_i], lambdas[_i] * w_inv, z_star):
                        solutions[_i], objectives[_i] = _s.copy(), _f
            archive_sols, archive_objs = self._update_archive(archive_sols, archive_objs, _ss, _so)

        if cfg.context != "none":
            if cfg.fixed_operator is not None or cfg.excluded_operators or cfg.operator_selection not in ("ucb", "thompson", "dyts", "dyts_beta"):
                raise ValueError("context so e suportado com seletores adaptativos (ucb/thompson/dyts/dyts_beta) sem fixed/excluded")
            _use_w = cfg.context in ("weight", "weight_phase")
            _use_p = cfg.context in ("phase", "weight_phase")
            _nw = cfg.ctx_weight_bins if _use_w else 1
            _np = cfg.ctx_phases if _use_p else 1

            def _factory():
                K_ = len(_pool)
                if cfg.operator_selection == "ucb":
                    return UCBSelector(n_operators=K_, c=cfg.ucb_c, window_size=cfg.ucb_window)
                if cfg.operator_selection == "thompson":
                    return ThompsonSelector(n_operators=K_, window_size=cfg.ucb_window)
                if cfg.operator_selection == "dyts":
                    return DYTSSelector(n_operators=K_, window_size=cfg.ucb_window)
                return DYTSBetaSelector(n_operators=K_, C=cfg.dyts_c)

            selector = ContextualSelector(_factory, _nw * _np)
            _wbin = np.minimum((np.argsort(np.argsort(lambdas[:, 0])) * _nw) // N, _nw - 1) if _use_w else np.zeros(N, dtype=int)
        elif cfg.fixed_operator is not None:
            selector = FixedSelector(len(_pool), cfg.fixed_operator)
        elif cfg.excluded_operators:
            if cfg.operator_selection != "random":
                raise ValueError("excluded_operators so e suportado com operator_selection='random'")
            selector = SubsetRandomSelector(len(_pool), cfg.excluded_operators)
        elif cfg.operator_selection == "ucb":
            selector = UCBSelector(
                n_operators=len(_pool),
                c=cfg.ucb_c,
                window_size=cfg.ucb_window,
            )
        elif cfg.operator_selection == "thompson":
            selector = ThompsonSelector(
                n_operators=len(_pool),
                window_size=cfg.ucb_window,
            )
        elif cfg.operator_selection == "dyts":
            selector = DYTSSelector(
                n_operators=len(_pool),
                window_size=cfg.ucb_window,
            )
        elif cfg.operator_selection == "dyts_beta":
            selector = DYTSBetaSelector(n_operators=len(_pool), C=cfg.dyts_c)
        elif cfg.operator_selection == "frrmab":
            selector = FRRMABSelector(n_operators=len(_pool), window=max(1, int(round(cfg.frr_window_frac * N))), C=cfg.frr_c, D=cfg.frr_d)
        else:
            selector = RandomSelector(n_operators=len(_pool))
        if (cfg.reward_mode == "frr") != (cfg.operator_selection == "frrmab"):
            raise ValueError("reward_mode='frr' so combina com operator_selection='frrmab' (e vice-versa)")
        _deferred = (cfg.reward_mode in ("replace", "frr")) and cfg.operator_selection in ("ucb", "thompson", "dyts", "dyts_beta", "frrmab")
        if _deferred and cfg.cost_aware:
            raise ValueError("reward_mode='replace' e incompativel com cost_aware")

        ref_pt = self._compute_ref_point(archive_objs)
        hv_history: list[float] = []
        op_selections: list[int] = []
        op_ema = np.zeros(len(_pool))   # tempo medio movel por operador (so com cost_aware)
        _st = {key: np.zeros(len(_pool)) for key in ("applied", "infeasible", "improved", "inserted")} if cfg.track_ops else None

        for gen in range(cfg.max_gen):
            if cfg.time_limit_s is not None and _tl_time.perf_counter() - _tl_t0 >= cfg.time_limit_s * (1.0 - cfg.polish_frac):
                break
            order = rng.permutation(N)

            for i in order:
                if cfg.context != "none":
                    selector.set_context(int(_wbin[i]) * _np + (min(_np * gen // cfg.max_gen, _np - 1) if _use_p else 0))
                if cfg.explore_mix > 0.0 and rng.random() < cfg.explore_mix:
                    k = int(rng.integers(len(_pool)))
                else:
                    k = selector.select(rng)
                op_selections.append(k)

                if rng.random() < 0.9:
                    j = int(rng.choice(neighborhoods[i]))
                else:
                    j = int(rng.integers(N))

                parent_sol = solutions[j].copy()
                parent_f   = objectives[j]

                operator = _pool[k]
                # Passa lam/z_star/partner para operadores que os utilizem
                import inspect as _insp
                _sig = _insp.signature(operator.apply)
                _kw: dict = {}
                if "lam" in _sig.parameters:
                    _kw["lam"] = lambdas[i] * w_inv
                if "z_star" in _sig.parameters:
                    _kw["z_star"] = z_star.copy()
                if "partner" in _sig.parameters:
                    # Segundo pai distinto de j, extraido da mesma vizinhanca
                    # (fiel ao MOEA/D original, que recombina solucoes vizinhas)
                    _cands_j2 = [int(x) for x in neighborhoods[i] if int(x) != j]
                    j2 = int(rng.choice(_cands_j2)) if _cands_j2 else j
                    _kw["partner"] = solutions[j2].copy()
                    _kw["partner_f"] = objectives[j2]
                if cfg.cost_aware:
                    import time as _time
                    _t_op0 = _time.perf_counter()
                new_sol, new_f = operator.apply(parent_sol, inst, rng, parent_f, **_kw)

                # Reparo centralizado para CDP/GDP (tamanho de solucao
                # variavel): garante capacidade minima (e orcamento
                # maximo no GDP) apos QUALQUER um dos 8 operadores,
                # independente de qual produziu new_sol. No-op para
                # MDPInstance (nao define repair() -- hasattr abaixo e
                # False, comportamento do MOEA/D para MDP fica 100%
                # inalterado). So reavalia new_f quando repair()
                # efetivamente mudou a solucao (evita reavaliacao
                # O(|S|^2) redundante quando new_sol ja era viavel).
                if _st is not None:
                    _st["applied"][k] += 1
                    if hasattr(inst, "is_feasible") and not inst.is_feasible(new_sol):
                        _st["infeasible"][k] += 1
                if hasattr(inst, "repair"):
                    repaired = inst.repair(new_sol, rng)
                    if repaired.size != new_sol.size or not np.array_equal(repaired, new_sol):
                        new_sol = repaired
                        new_f = inst.evaluate(new_sol)

                if cfg.operator_selection in ("ucb", "thompson", "dyts", "dyts_beta", "frrmab") and not _deferred:
                    _scale = 0.0
                    if cfg.reward_mode == "scalar":
                        old_g = self._tchebycheff(parent_f, lambdas[i] * w_inv, z_star)
                        new_g = self._tchebycheff(new_f,    lambdas[i] * w_inv, z_star)
                        reward = scalar_reward(old_g, new_g)
                        _scale = old_g
                    elif cfg.reward_mode == "scalar_rel":
                        old_g = self._tchebycheff(parent_f, lambdas[i] * w_inv, z_star)
                        new_g = self._tchebycheff(new_f,    lambdas[i] * w_inv, z_star)
                        reward = min(1.0, max(0.0, (old_g - new_g) / old_g)) if old_g > 0 else 0.0
                    elif cfg.reward_mode == "insert":
                        _t0_, _t1_ = _REL_TOL * abs(new_f[0]), _REL_TOL * abs(new_f[1])
                        _blocked = any(
                            (fa[0] >= new_f[0] - _t0_ and fa[1] >= new_f[1] - _t1_)   # dominado ou duplicado (a menos de ruido de ponto flutuante)
                            for fa in archive_objs
                        )
                        reward = 0.0 if _blocked else 1.0
                    else:
                        dominated = any(
                            fa[0] >= new_f[0] and fa[1] >= new_f[1]
                            and (fa[0] > new_f[0] or fa[1] > new_f[1])
                            for fa in archive_objs
                        )
                        if dominated:
                            reward = 0.0
                        else:
                            old_hv  = hypervolume(archive_objs, ref_pt)
                            new_hv  = hypervolume(archive_objs + [new_f], ref_pt)
                            reward  = hypervolume_reward(old_hv, new_hv)
                            _scale = old_hv
                    # Seletores de sucesso BINARIO (reward > 0): ganhos de ruido de ponto flutuante nao contam como sucesso
                    if cfg.operator_selection in ("thompson", "dyts_beta") and reward <= _REL_TOL * _scale:
                        reward = 0.0
                    if cfg.cost_aware:
                        _dt = _time.perf_counter() - _t_op0
                        op_ema[k] = _dt if op_ema[k] == 0.0 else 0.9 * op_ema[k] + 0.1 * _dt
                        _m = op_ema[op_ema > 0.0].mean()
                        reward = reward * (_m / op_ema[k])
                    selector.update(k, reward)

                _improved = False
                _fir = 0.0
                for j in neighborhoods[i]:
                    g_old = self._tchebycheff(objectives[j], lambdas[j] * w_inv, z_star)
                    g_new = self._tchebycheff(new_f,         lambdas[j] * w_inv, z_star)
                    if g_new < g_old and (g_old - g_new) > _REL_TOL * abs(g_old):
                        _improved = True
                        if g_old > 0.0:
                            _fir += (g_old - g_new) / g_old
                    if g_new <= g_old:
                        solutions[j]  = new_sol.copy()
                        objectives[j] = new_f
                if _deferred:
                    selector.update(k, _fir if cfg.reward_mode == "frr" else (1.0 if _improved else 0.0))
                if _st is not None and _improved:
                    _st["improved"][k] += 1

                if new_f[0] > z_star[0]:
                    z_star[0] = new_f[0]
                if new_f[1] > z_star[1]:
                    z_star[1] = new_f[1]
                if new_f[0] < z_min[0]:
                    z_min[0] = new_f[0]
                if new_f[1] < z_min[1]:
                    z_min[1] = new_f[1]
                _refresh_w_inv()

                archive_sols, archive_objs = self._update_archive(
                    archive_sols, archive_objs, [new_sol], [new_f]
                )
                if _st is not None and archive_objs and archive_objs[-1] is new_f:   # _update_archive anexa o proprio objeto new_f quando insere
                    _st["inserted"][k] += 1

            ref_pt = self._compute_ref_point(archive_objs)

            if cfg.log_every > 0 and (gen + 1) % cfg.log_every == 0:
                hv_history.append(float(hypervolume(archive_objs, ref_pt)))

        if cfg.anchor_iters > 0:
            kb = int(np.argmax([f[1] for f in archive_objs]))
            _res_anchor = []
            if cfg.anchor_method in ("tabu", "both"):
                from anchor import maxmin_anchor
                _res_anchor.append(maxmin_anchor(inst, rng, iters=cfg.anchor_iters, restarts=cfg.anchor_restarts,
                                                 start=archive_sols[kb]))
            if cfg.anchor_method in ("sbts", "both"):
                from sbts_anchor import sbts_anchor
                _res_anchor.append(sbts_anchor(inst, rng, iters=cfg.sbts_iters or cfg.anchor_iters,
                                               restarts=1 if cfg.anchor_method == "both" else cfg.anchor_restarts,
                                               start=archive_sols[kb]))
            for res_a in _res_anchor:
                if res_a is not None:
                    archive_sols, archive_objs = self._update_archive(archive_sols, archive_objs, [res_a[0]], [res_a[1]])
        if cfg.polish_frac > 0.0 and cfg.time_limit_s is not None:
            from bo_trajectory import BOTrajConfig, BOTrajectory
            _rem = cfg.time_limit_s - (_tl_time.perf_counter() - _tl_t0)
            _pol = BOTrajectory(inst, BOTrajConfig(method="tabu", local_search="pr", seed=int(rng.integers(2**31 - 1)), time_limit_s=max(_rem, 0.0)))
            _ps, _po = _pol.polish(archive_sols, archive_objs)
            archive_sols, archive_objs = list(_ps), list(_po)
        final_pts  = np.array([list(f) for f in archive_objs])
        final_front = pareto_front(final_pts)
        final_hv    = float(hypervolume(final_front, ref_pt))

        return MOEADResult(
            pareto_front=final_front,
            hv_history=hv_history,
            operator_selections=op_selections,
            # Total de selecoes por operador na execucao inteira. ATENCAO: ate 1 out 2026
            # isto era selector.counts() = observacoes na JANELA deslizante (<= W por operador),
            # que satura em W para todos e fica ~uniforme (1/K) independente da politica real.
            # Runs gravados antes dessa data tem a contagem de janela (inutil p/ analise de uso).
            operator_counts=np.bincount(op_selections, minlength=len(_pool)).astype(float),
            operator_q_values=selector.q_values(),
            n_gen_executed=cfg.max_gen,
            best_hv=final_hv,
            config=cfg,
            operator_stats=({key: v.tolist() for key, v in _st.items()} if _st is not None else None),
            archive_sizes=[int(len(x)) for x in archive_sols],
        )

    # ------------------------------------------------------------------
    # Inicializacao
    # ------------------------------------------------------------------

    def _initialize(
        self, N: int, lambdas: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """Warm-start GRASP guiado por peso: cada subproblema recebe uma solucao
        construida com o criterio lam[0]*deltaSum + lam[1]*minDist, garantindo
        cobertura diversa da frente de Pareto desde a primeira geracao."""
        solutions, objectives = [], []
        for i in range(N):
            sol = self._grasp_init(lambdas[i])
            f   = self.inst.evaluate(sol)
            solutions.append(sol)
            objectives.append(f)
        return solutions, objectives

    def _grasp_init(self, lam: np.ndarray) -> np.ndarray:
        """Construcao GRASP guiada pelo vetor de pesos lam.

        MDP (inst.m definido): comportamento ORIGINAL, byte-a-byte
        identico -- para em exatamente m elementos.

        CDP/GDP (sem .m, tem .min_capacity): mesmo criterio de score
        ponderado (lam[0]*delta_sum + lam[1]*min_dist) e RCL/alpha, mas
        para quando a capacidade minima e atingida em vez de num m fixo,
        filtrando candidatos por orcamento (_can_extend) a cada passo
        quando a instancia define esse metodo (GDP; CDP nao tem orcamento
        e _can_extend base sempre retorna True). Se o RCL guiado por lam
        esgotar os candidatos externos antes de alcancar a capacidade
        minima (orcamento muito apertado para a heuristica de diversidade
        pura), cai para instance.repair(), que usa o criterio de
        eficiencia (capacidade/custo) para garantir viabilidade -- mesma
        rede de seguranca usada apos cada operator.apply() no loop
        principal (run(), mais acima)."""
        inst  = self.inst
        n     = inst.n
        alpha = self.cfg.grasp_alpha
        rng   = self.rng
        D     = inst.D

        first = int(rng.integers(n))
        selected: list[int] = [first]
        available = list(range(n))
        available.remove(first)

        fixed_m = getattr(inst, "m", None)
        budget_aware = fixed_m is None and hasattr(inst, "_can_extend")
        n_steps = (fixed_m - 1) if fixed_m is not None else (n - 1)

        for _ in range(n_steps):
            if fixed_m is None:
                if inst.capacity_of(selected) >= inst.min_capacity - 1e-9:
                    break
                if not available:
                    break

            sel_arr = np.array(selected, dtype=np.intp)
            cand_pool = available
            if budget_aware:
                total_cost = sum(inst._cost(i) for i in selected)
                if hasattr(inst, "_extendable"):   # vetorizado, mesma aritmetica de _can_extend
                    _av = np.asarray(available, dtype=np.intp)
                    filtered = _av[inst._extendable(total_cost, _av)].tolist()
                else:
                    filtered = [c for c in available if inst._can_extend(total_cost, c)]
                if filtered:
                    cand_pool = filtered
                # se filtered vazio, mantem cand_pool=available (deixa
                # instance.repair() tratar o estouro de orcamento depois)

            # Vetorizado (era 1 chamada Python de sum+min por candidato: ~1e6 por construcao
            # em n=3000). A soma por linha da submatriz equivale ao sum() do vetor D[cand, sel]
            # (mesma ordem e mesmo algoritmo de reducao); verificado bit a bit.
            Dc      = D[np.ix_(np.asarray(cand_pool, dtype=np.intp), sel_arr)]
            scores  = lam[0] * Dc.sum(axis=1) + lam[1] * Dc.min(axis=1)

            s_max, s_min = scores.max(), scores.min()
            if s_max > s_min:
                threshold = s_max - alpha * (s_max - s_min)
                rcl = [cand_pool[k] for k in np.flatnonzero(scores >= threshold)]
            else:
                rcl = cand_pool[:]

            chosen = int(rng.choice(rcl))
            selected.append(chosen)
            available.remove(chosen)

        sol = np.array(selected, dtype=np.intp)
        if fixed_m is None and hasattr(inst, "repair"):
            sol = inst.repair(sol, rng)
        return np.sort(np.asarray(sol, dtype=np.intp))

    # ------------------------------------------------------------------
    # Auxiliares estaticos
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_weights(N: int) -> np.ndarray:
        eps = 1e-3
        lambdas = np.zeros((N, 2))
        for i in range(N):
            t = i / (N - 1) if N > 1 else 0.5
            lambdas[i, 0] = eps + (1 - 2 * eps) * t
            lambdas[i, 1] = 1.0 - lambdas[i, 0]
        return lambdas

    @staticmethod
    def _compute_neighborhoods(lambdas: np.ndarray, T: int) -> list[np.ndarray]:
        N = len(lambdas)
        return [np.argsort(np.linalg.norm(lambdas - lambdas[i], axis=1))[:T] for i in range(N)]

    @staticmethod
    def _tchebycheff(f: tuple[float, float], lam: np.ndarray, z_star: np.ndarray) -> float:
        return float(np.max(lam * (z_star - np.array(f))))

    @staticmethod
    def _update_archive(
        archive_sols: list[np.ndarray],
        archive_objs: list[tuple[float, float]],
        new_sols: list[np.ndarray],
        new_objs: list[tuple[float, float]],
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        for sol, f in zip(new_sols, new_objs):
            dominated_by_new = []
            dominated = False
            duplicate = False

            for idx, fa in enumerate(archive_objs):
                if abs(f[0]-fa[0]) < 1e-9 and abs(f[1]-fa[1]) < 1e-9:
                    duplicate = True
                    break
                if f[0] >= fa[0] and f[1] >= fa[1]:
                    if f[0] > fa[0] or f[1] > fa[1]:
                        dominated_by_new.append(idx)
                elif fa[0] >= f[0] and fa[1] >= f[1]:
                    if fa[0] > f[0] or fa[1] > f[1]:
                        dominated = True
                        break

            if not dominated and not duplicate:
                for idx in sorted(dominated_by_new, reverse=True):
                    archive_sols.pop(idx)
                    archive_objs.pop(idx)
                archive_sols.append(sol.copy())
                archive_objs.append(f)

        return archive_sols, archive_objs

    def _compute_ref_point(self, archive_objs: list[tuple[float, float]]) -> np.ndarray:
        if not archive_objs:
            return np.zeros(2)
        fc = self.cfg.hv_ref_factor
        return np.array([
            min(f[0] for f in archive_objs) * fc,
            min(f[1] for f in archive_objs) * fc,
        ])
