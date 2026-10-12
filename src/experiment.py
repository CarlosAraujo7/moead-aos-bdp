"""
experiment.py -- Orquestracao de experimentos para o MOEA/D-AOS e metodos comparados.

Metodos suportados:
    MOEA/D-AOS-HV        -- UCB1 com recompensa de hipervolume  (proposta principal)
    MOEA/D-AOS-scalar    -- UCB1 com recompensa escalar
    MOEA/D-Thompson-HV   -- Thompson Sampling com recompensa HV
    MOEA/D-Thompson-scal -- Thompson Sampling com recompensa escalar
    MOEA/D-DYTS-HV       -- Dynamic Thompson Sampling (Sun & Li, 2020) + recompensa HV
    MOEA/D-DYTS-scalar   -- Dynamic Thompson Sampling + recompensa escalar
    MOEA/D-random        -- selecao uniforme (baseline sem AOS)
    NSGA-II              -- baseline evolutivo (Deb et al., 2002)
    GRASP-MO             -- GRASP multi-objetivo
    GRASP-PR             -- GRASP + path relinking
    VNS-Pareto           -- Variable Neighborhood Search c/ arquivo de Pareto (Hansen & Mladenovic, 2001)
    VNS-Tchebycheff      -- VNS escalarizado via Tchebycheff -- ablacao: MOEA/D sem AOS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import sys

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from problem import MDPInstance
from capacitated_problem import CDPInstance, GDPInstance
from moead_aos import MOEAD_AOS, MOEADConfig
from nsga2 import NSGA2, NSGA2Config
from grasp_mo import GRASPMO, GRASPMOConfig
from vns import BVNS, BVNSConfig
from metrics import hypervolume, set_coverage, epsilon_indicator, pareto_front
from operators import OPERATOR_NAMES


ALL_MODES = [
    "hypervolume",     # MOEA/D-AOS (UCB1) + recompensa HV   -- contribuicao principal
    "thompson_hv",     # MOEA/D-Thompson + recompensa HV      -- variante proposta
    "dyts_hv",         # MOEA/D-DYTS (Sun & Li, 2020) + recompensa HV      -- variante proposta
    "scalar",          # MOEA/D-AOS (UCB1) + recompensa escalar -- ablacao: reward mode
    "thompson_scalar", # MOEA/D-Thompson + recompensa escalar  -- ablacao combinada
    "dyts_scalar",     # MOEA/D-DYTS + recompensa escalar      -- ablacao combinada
    "random",          # MOEA/D sem AOS (uniforme)            -- ablacao: decomposicao pura
    "nsga2",           # NSGA-II                              -- baseline evolutivo
    "grasp",           # GRASP multi-objetivo                 -- baseline construtivo
    "grasp_pr",        # GRASP + path relinking               -- baseline construtivo intensificado
    "vns_pareto",      # VNS c/ arquivo de Pareto              -- baseline busca local (Hansen & Mladenovic, 2001)
    "vns_tchebycheff", # VNS escalarizado via Tchebycheff      -- ablacao: MOEA/D sem AOS (busca local pura)
]
# NOTA: "dyts_hv"/"dyts_scalar" adicionados nesta sessao -- DYTSSelector ja
# existia em mab.py e MOEAD_AOS ja aceitava operator_selection="dyts", mas
# nenhum modo de experiment.py o expunha (a lista ALL_MODES parava em 8
# entradas, sem DYTS). Resultados existentes em results/runs/ sao anteriores
# a esta mudanca e nao contem estes dois modos.

CAPACITATED_COMPATIBLE_MODES = (
    "hypervolume", "scalar", "random",
    "thompson_hv", "thompson_scalar",
    "dyts_hv", "dyts_scalar",
    "nsga2", "grasp", "grasp_pr",
    "vns_pareto", "vns_tchebycheff",
)
# NOTA (atualizada ao adaptar nsga2.py/grasp_mo.py/vns.py para CDP/GDP):
# os 3 arquivos NAO tinham escopo comparavel entre si, como uma nota
# anterior aqui sugeria -- verificado arquivo por arquivo antes de mexer.
# nsga2.py: unico bloqueio real era o __repr__ (self.inst.m); crossover e
# mutacao ja eram size-agnostic (usa len()/sol.size), so faltava um
# repair-net apos gerar cada filho. vns.py: as 3 vizinhancas (1opt/2opt/
# kopt) so usavam inst.m pra dimensionar indices -- trocado por sol.size,
# mesmo repair-net em cada uma. grasp_mo.py: _local_search era mecanico
# igual aos outros dois (m -> sol.size, ja usava inst.evaluate_swap que
# capacitated_problem.py ja generaliza); _construct() era o unico caso
# de verdade -- nao existe "m" pra perseguir em CDP/GDP, entao o RCL
# guloso-randomico agora filtra candidatos por orcamento a cada passo
# (inst.can_extend) e para por capacidade minima (inst.capacity_of), com
# inst.repair() como rede de seguranca se ficar preso -- decisao tomada
# com o Carlos (opcao "nativamente ciente de orcamento" em vez de "repara
# so no final"). Todos os 12 modos rodam em CDP/GDP agora. Validacao:
# ver artefato secao 10 (~300 instancias CDP+GDP + regressao zero no MDP).

def seed_for(inst_name: str, mode: str, rep: int) -> int:
    """Semente deterministica por (instancia, modo, replica).

    Substitui hash((nome, modo, rep)): hash() de str e randomizado por processo
    (PYTHONHASHSEED), entao re-rodar o experimento dava sementes diferentes.
    """
    return zlib.crc32(f"{inst_name}|{mode}|{rep}".encode("utf-8")) % (2**31)


# Incrementar quando uma mudanca de CODIGO alterar os resultados numericos de algum modo
# (otimizacoes que preservam resultado bit a bit, como evaluate_swaps_batch, NAO contam).
CODE_VERSION = 1


def config_fingerprint(cfg: "ExperimentConfig", mode: str, n: int) -> dict:
    """Parametros EFETIVOS que determinam o resultado de (modo, instancia de tamanho n).

    Guardado em cada JSON de run; run_full.py compara o hash com o da configuracao atual
    e re-executa se mudou (o checkpoint antigo so olhava se o arquivo existia, entao uma
    mudanca de orcamento/regime deixava resultados velhos valendo para sempre)."""
    if cfg.use_budget_scaling:
        bgt = budget_for(n)
        T, maxgen, nbr = bgt["T"], bgt["max_gen"], bgt["nbr"]
        n2pop, n2gen, giter = bgt["nsga2_pop"], bgt["nsga2_gen"], bgt["grasp_iter"]
    else:
        T, maxgen, nbr = cfg.n_weights, cfg.max_gen, cfg.neighborhood_size
        n2pop, n2gen, giter = cfg.nsga2_pop, cfg.nsga2_gen, cfg.grasp_iter
    fp = {"code_version": CODE_VERSION, "mode": mode}
    if mode in ("hypervolume", "scalar", "random", "thompson_hv", "thompson_scalar", "dyts_hv", "dyts_scalar"):
        fp.update(T=T, max_gen=maxgen, nbr=nbr, ucb_c=cfg.ucb_c, ucb_window=cfg.ucb_window,
                  grasp_alpha=cfg.grasp_alpha, normalize_objectives=bool(cfg.normalize_objectives))
        if mode in ("thompson_hv", "thompson_scalar"):   # 10 out 2026: tolerancia relativa no 'sucesso' binario (ruido de ponto flutuante); invalida so estes runs
            fp.update(binary_reward_rel_tol=1e-9)
        if cfg.anchor_iters > 0:   # so entra no hash quando ligado: nao invalida runs existentes
            fp.update(anchor_iters=int(cfg.anchor_iters), anchor_restarts=int(cfg.anchor_restarts))
    elif mode == "nsga2":
        fp.update(pop=n2pop, gen=n2gen)
    elif mode in ("grasp", "grasp_pr"):
        fp.update(max_iter=giter, alpha=cfg.grasp_alpha)
    elif mode in ("vns_pareto", "vns_tchebycheff"):
        fp.update(max_iter=giter, T=T, normalize_objectives=bool(cfg.normalize_objectives))
    return fp


def config_hash(fp: dict) -> str:
    return hashlib.sha1(json.dumps(fp, sort_keys=True).encode("utf-8")).hexdigest()[:12]


MODE_LABEL = {
    "hypervolume":      "MOEAD-AOS-HV",
    "thompson_hv":      "MOEAD-Thompson-HV",
    "dyts_hv":          "MOEAD-DYTS-HV",
    "scalar":           "MOEAD-AOS-scalar",
    "thompson_scalar":  "MOEAD-Thompson-scalar",
    "dyts_scalar":      "MOEAD-DYTS-scalar",
    "random":           "MOEAD-random",
    "nsga2":            "NSGA-II",
    "grasp":            "GRASP-MO",
    "grasp_pr":         "GRASP-PR",
    "vns_pareto":       "VNS-Pareto",
    "vns_tchebycheff":  "VNS-Tchebycheff",
}


# ---------------------------------------------------------------------------
# Orcamento escalado por tamanho de instancia
# ---------------------------------------------------------------------------

def budget_for(n: int) -> dict:
    """Parametros de orcamento escalonados por n.

    Criterio de dimensionamento: o produto T*max_gen (total de avaliacoes MOEA/D)
    cresce com n, e os budgets de NSGA-II e GRASP sao calibrados para tempo de CPU
    comparavel. Todas as 6 colunas sao monotonicas nao-decrescentes em n -- nenhum
    parametro de orcamento cai quando a instancia cresce.

    Redesenhada em 3 set 2026 (diagnostico, secao 12): a tabela anterior nao era
    monotonica -- max_gen/nsga2_gen/grasp_iter SUBIAM ate n<=300/500 e depois
    CAIAM para n maior (grasp_iter chegava a voltar ao mesmo valor de n<=30 para
    n>1000). O comentario original alegava "T*max_gen cresce suavemente com n" e
    budgets "deliberadamente reduzidos para viabilizar o experimento completo em
    ~2h com 8 workers" para n>=500 -- nem a alegacao de crescimento suave de
    T*max_gen se sustentava (o produto tambem caia: 45000 em n<=500 -> 40000 em
    n<=1000/n>1000), nem o "~2h" batia com tempo real medido (o sense-check com
    a tabela antiga, mesmos 7 modos MOEA/D-AOS, 4 familias, 1 replica, levou
    6h22min em CDP + 14h00min em GDP com 10 workers -- so uma fracao do escopo
    total do projeto). Correcao aplicada: maximo cumulativo de cada coluna ao
    longo de n crescente (isotonic simples) -- preserva exatamente os valores de
    n<=300 (ja monotonicos) e so eleva os de n>300 ate o teto ja alcancado antes.
    Efeito colateral aceito conscientemente: instancias grandes (n>500,
    especialmente n>1000) ficam mais caras de rodar do que na tabela antiga --
    troca deliberada por comparabilidade entre todos os tamanhos de instancia,
    nao mais por limitar tempo de parede.

    ATENCAO -- comparabilidade com resultados ja gerados: todo resultado salvo
    antes desta mudanca (os 3.036 do sense-check preliminar: 936 MDP + 700 CDP +
    1.400 GDP) foi computado com a tabela ANTERIOR (nao-monotonica). O mecanismo
    de checkpoint de run_full.py (pula (instancia,modo,replica) ja salvos) NAO
    detecta essa mudanca de orcamento -- resultados antigos permanecem no disco
    indefinidamente sob o regime antigo, mesmo depois desta correcao. Qualquer
    novo resultado para um (instancia,modo,replica) ainda nao existente vai usar
    a tabela nova. Comparar um modo rodado antes desta mudanca com outro rodado
    depois, na MESMA instancia, compara sob orcamentos diferentes -- nao e uma
    comparacao modo-a-modo valida. Ver secao 12 do diagnostico para a decisao
    registrada sobre como tratar isso.

    Tabela (T*gen = T*max_gen, total de avaliacoes MOEA/D):
        n       | T   | max_gen | T*gen   | nsga2_pop | nsga2_gen | grasp_iter
        --------+-----+---------+---------+-----------+-----------+-----------
        <=30    |  15 |    200  |   3 000 |        20 |        80 |        150
        <=50    |  20 |    300  |   6 000 |        30 |       120 |        200
        <=100   |  30 |    400  |  12 000 |        50 |       180 |        300
        <=150   |  40 |    500  |  20 000 |        60 |       200 |        400
        <=300   |  60 |    600  |  36 000 |        80 |       250 |        500
        <=500   |  75 |    600  |  45 000 |        80 |       250 |        500
        <=1000  |  80 |    600  |  48 000 |       100 |       250 |        500
        >1000   | 100 |    600  |  60 000 |       120 |       250 |        500
    """
    if n <= 30:
        return dict(T=15,  max_gen=200, nbr=5,  nsga2_pop=20,  nsga2_gen=80,  grasp_iter=150)
    if n <= 50:
        return dict(T=20,  max_gen=300, nbr=5,  nsga2_pop=30,  nsga2_gen=120, grasp_iter=200)
    if n <= 100:
        return dict(T=30,  max_gen=400, nbr=7,  nsga2_pop=50,  nsga2_gen=180, grasp_iter=300)
    if n <= 150:
        return dict(T=40,  max_gen=500, nbr=10, nsga2_pop=60,  nsga2_gen=200, grasp_iter=400)
    if n <= 300:
        return dict(T=60,  max_gen=600, nbr=12, nsga2_pop=80,  nsga2_gen=250, grasp_iter=500)
    if n <= 500:
        return dict(T=75,  max_gen=600, nbr=15, nsga2_pop=80,  nsga2_gen=250, grasp_iter=500)
    if n <= 1000:
        return dict(T=80,  max_gen=600, nbr=15, nsga2_pop=100, nsga2_gen=250, grasp_iter=500)
    return         dict(T=100, max_gen=600, nbr=20, nsga2_pop=120, nsga2_gen=250, grasp_iter=500)


# ---------------------------------------------------------------------------
# Configuracao do experimento
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Filtro de instancias (aceita lista separada por virgula)
    families: str = "SOM-a,SOM-b,GKD-c,MDG-a,MGPO"
    max_n: int = 9999
    min_n: int = 0
    n_reps: int = 10
    modes: tuple = tuple(ALL_MODES)

    # Parametros fixos (usados quando use_budget_scaling=False)
    n_weights: int = 20
    neighborhood_size: int = 5
    max_gen: int = 300
    ucb_c: float = 0.5
    ucb_window: int = 50
    grasp_alpha: float = 0.3
    nsga2_pop: int = 50
    nsga2_gen: int = 200
    grasp_iter: int = 300

    # Orcamento automatico por n
    use_budget_scaling: bool = True

    # Normaliza objetivos no Tchebycheff (MOEA/D-AOS e VNS-Tchebycheff). Padrao False =
    # catalogo historico; ver MOEADConfig.normalize_objectives.
    normalize_objectives: bool = False

    # Ancoragem da ponta de MaxMin no MOEA/D-AOS (src/anchor.py); 0 = desligado (padrao = catalogo historico).
    anchor_iters: int = 0
    anchor_restarts: int = 3

    # Logging
    log_every: int = 0
    results_dir: Path = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Carregamento de instancias
# ---------------------------------------------------------------------------

def load_instances(catalog_path, instances_root, families_str, min_n, max_n):
    fam_set = {f.strip() for f in families_str.split(",")}
    instances = []
    with catalog_path.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row["family"] not in fam_set:
            continue
        n = int(row["n"])
        if n < min_n or n > max_n:
            continue
        filepath = instances_root / row["filepath"]
        if not filepath.exists():
            warnings.warn(f"Arquivo nao encontrado: {filepath}")
            continue
        inst = MDPInstance(filepath)
        # BUGFIX: MDPInstance infere .family a partir do nome da PASTA
        # (filepath.parent.name). Isso funciona para a maioria das familias
        # MDPLIB (uma pasta por familia), mas quebra para instances/real_world/,
        # onde 4 familias semanticamente distintas do catalogo (real_sensor,
        # real_facility, real_portfolio, real_team) compartilham a mesma pasta
        # fisica -- sem esta correcao, todas seriam gravadas como "real_world"
        # em run_full.py (instances_meta usa i.family), colapsando os 4
        # cenarios reais em um so grupo em summary.csv e nas figuras por familia.
        # O catalogo (row["family"]/row["name"]) e sempre a fonte de verdade.
        inst.family = row["family"]
        inst.name   = row["name"]
        instances.append(inst)
    return instances


def load_capacitated_instances(catalog_path, instances_root, problem, families_str, min_n, max_n):
    """Carrega instancias CDP ou GDP do catalogo proprio
    (instances/catalog_capacitated.csv) -- analogo a load_instances(),
    mas para instancias de tamanho de subconjunto VARIAVEL (sem m fixo).

    IMPORTANTE: catalog_capacitated.csv usa os MESMOS nomes de familia que
    catalog.csv (GKD-b, GKD-c, MDG-b, SOM-a) para um PROBLEMA DIFERENTE --
    confirmado nao haver colisao de nomes de instancia entre os dois
    catalogos nem entre CDP e GDP (GDP sempre tem sufixo _k02/_k03, CDP
    nao). Filtrar so por family seria ambiguo -- por isso este filtro usa
    SEMPRE problem (CDP ou GDP) em conjunto com family, nunca family
    sozinho.
    """
    fam_set = {f.strip() for f in families_str.split(",")}
    cls_by_problem = {"CDP": CDPInstance, "GDP": GDPInstance}
    if problem not in cls_by_problem:
        raise ValueError(f"problem deve ser 'CDP' ou 'GDP', recebido: {problem!r}")
    cls = cls_by_problem[problem]

    instances = []
    with catalog_path.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row["problem"] != problem:
            continue
        if row["family"] not in fam_set:
            continue
        n = int(row["n"])
        if n < min_n or n > max_n:
            continue
        filepath = instances_root / row["filepath"]
        if not filepath.exists():
            warnings.warn(f"Arquivo nao encontrado: {filepath}")
            continue
        inst = cls(filepath)
        inst.family = row["family"]
        inst.name   = row["name"]
        instances.append(inst)
    return instances


# ---------------------------------------------------------------------------
# Execucao de uma replica
# ---------------------------------------------------------------------------

def run_one(inst: "MDPInstance | CDPInstance | GDPInstance", cfg: ExperimentConfig, mode: str, rep: int):
    """Executa uma replica de um modo sobre uma instancia.

    Quando use_budget_scaling=True, os parametros sao escalonados por inst.n.

    Para CDP/GDP (sem cardinalidade fixa m -- deteccao via hasattr(inst,"m")),
    os modos compativeis sao os listados em CAPACITATED_COMPATIBLE_MODES: os
    7 da familia MOEA/D-AOS mais nsga2/grasp/grasp_pr/vns_pareto/
    vns_tchebycheff, estes ultimos adaptados via o padrao "repair-net" (ver
    nota acima de ALL_MODES). Nesta versao, todo o conjunto de modos do
    projeto e compativel com CDP/GDP.
    """
    if not hasattr(inst, "m") and mode not in CAPACITATED_COMPATIBLE_MODES:
        raise ValueError(
            f"Modo {mode!r} nao suporta instancias de tamanho variavel (CDP/GDP, "
            f"sem cardinalidade fixa m). Use um dos modos compativeis: "
            f"{CAPACITATED_COMPATIBLE_MODES}."
        )
    seed = seed_for(inst.name, mode, rep)

    if cfg.use_budget_scaling:
        bgt    = budget_for(inst.n)
        T      = bgt["T"]
        maxgen = bgt["max_gen"]
        nbr    = bgt["nbr"]
        n2pop  = bgt["nsga2_pop"]
        n2gen  = bgt["nsga2_gen"]
        giter  = bgt["grasp_iter"]
    else:
        T      = cfg.n_weights
        maxgen = cfg.max_gen
        nbr    = cfg.neighborhood_size
        n2pop  = cfg.nsga2_pop
        n2gen  = cfg.nsga2_gen
        giter  = cfg.grasp_iter

    if mode in ("hypervolume", "scalar", "random",
                "thompson_hv", "thompson_scalar",
                "dyts_hv", "dyts_scalar"):
        if mode == "random":
            op_sel, rw = "random", "scalar"
        elif mode == "thompson_hv":
            op_sel, rw = "thompson", "hypervolume"
        elif mode == "thompson_scalar":
            op_sel, rw = "thompson", "scalar"
        elif mode == "dyts_hv":
            op_sel, rw = "dyts", "hypervolume"
        elif mode == "dyts_scalar":
            op_sel, rw = "dyts", "scalar"
        else:
            op_sel, rw = "ucb", mode   # "hypervolume" ou "scalar"
        moead_cfg = MOEADConfig(
            n_weights=T, neighborhood_size=nbr, max_gen=maxgen,
            operator_selection=op_sel, reward_mode=rw,
            ucb_c=cfg.ucb_c, ucb_window=cfg.ucb_window,
            grasp_alpha=cfg.grasp_alpha,
            log_every=cfg.log_every, seed=seed,
            normalize_objectives=cfg.normalize_objectives,
            anchor_iters=cfg.anchor_iters, anchor_restarts=cfg.anchor_restarts,
        )
        t0 = time.perf_counter()
        result = MOEAD_AOS(inst, moead_cfg).run()
        elapsed = time.perf_counter() - t0
        pf = result.pareto_front
        hv_history = result.hv_history
        op_counts  = result.operator_counts.tolist()
        op_qvals   = result.operator_q_values.tolist()

    elif mode == "nsga2":
        nsga2_cfg = NSGA2Config(
            pop_size=n2pop, max_gen=n2gen,
            log_every=cfg.log_every, seed=seed,
        )
        t0 = time.perf_counter()
        result = NSGA2(inst, nsga2_cfg).run()
        elapsed = time.perf_counter() - t0
        pf = result.pareto_front
        hv_history = result.hv_history
        op_counts  = result.operator_counts.tolist()
        op_qvals   = result.operator_q_values.tolist()

    elif mode in ("grasp", "grasp_pr"):
        grasp_cfg = GRASPMOConfig(
            max_iter=giter, alpha=cfg.grasp_alpha,
            path_relink=(mode == "grasp_pr"),
            log_every=cfg.log_every, seed=seed,
        )
        t0 = time.perf_counter()
        result = GRASPMO(inst, grasp_cfg).run()
        elapsed = time.perf_counter() - t0
        pf = result.pareto_front
        hv_history = result.hv_history
        op_counts  = result.operator_counts.tolist()
        op_qvals   = result.operator_q_values.tolist()

    elif mode in ("vns_pareto", "vns_tchebycheff"):
        # Reusa o orcamento de grasp_iter (giter): GRASP-PR e VNS sao ambos
        # metodos de busca local iterativa com custo por iteracao da mesma
        # ordem de grandeza (uma troca/perturbacao + reavaliacao). Para
        # vns_tchebycheff, n_weights=T usa o MESMO T do MOEA/D -- e a
        # ablacao "MOEA/D sem AOS" (ver vns.py), entao a granularidade da
        # decomposicao precisa ser identica para a comparacao ser justa.
        vns_cfg = BVNSConfig(
            max_iter=giter, max_no_improve=max(50, int(giter * 0.3)),
            k_max=3, variant=("tchebycheff" if mode == "vns_tchebycheff" else "pareto"),
            n_weights=T,
            log_every=cfg.log_every, seed=seed,
            normalize_objectives=cfg.normalize_objectives,
        )
        t0 = time.perf_counter()
        result = BVNS(inst, vns_cfg).run()
        elapsed = time.perf_counter() - t0
        pf = result.pareto_front
        hv_history = result.hv_history
        op_counts  = result.operator_counts.tolist()
        op_qvals   = result.operator_q_values.tolist()

    else:
        raise ValueError(f"Modo desconhecido: {mode}")

    return pf, elapsed, hv_history, op_counts, op_qvals


def run_instance(inst, cfg, mode, results_dir):
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for rep in range(cfg.n_reps):
        pf, elapsed, hv_history, op_counts, op_qvals = run_one(inst, cfg, mode, rep)

        run_file = runs_dir / f"{inst.name}_{mode}_rep{rep:02d}.json"
        run_data = {
            "instance": inst.name,
            "family":   inst.family,
            "n": inst.n, "m": inst.m,
            "mode": mode,
            "label": MODE_LABEL.get(mode, mode),
            "rep": rep,
            "seed": seed_for(inst.name, mode, rep),
            "elapsed_s": round(elapsed, 4),
            "pareto_size": len(pf),
            "pareto_front": pf.tolist(),
            "hv_history": hv_history,
            "operator_counts": op_counts,
            "operator_q_values": op_qvals,
            "operator_names": OPERATOR_NAMES,
            "config": config_fingerprint(cfg, mode, inst.n),
            "config_hash": config_hash(config_fingerprint(cfg, mode, inst.n)),
        }
        with open(run_file, "w") as fh:
            json.dump(run_data, fh, indent=2)

        records.append({
            "instance": inst.name, "family": inst.family,
            "n": inst.n, "m": inst.m,
            "mode": mode, "label": MODE_LABEL.get(mode, mode),
            "rep": rep, "elapsed_s": elapsed,
            "pareto_size": len(pf), "_pf": pf,
        })

    return records


# ---------------------------------------------------------------------------
# Executor principal
# ---------------------------------------------------------------------------

def compute_reference_front(all_pfs):
    return pareto_front(np.vstack(all_pfs))

def build_ref_point(all_fronts):
    """Ponto de referencia nadir+margem.

    Usa o pior valor em cada objetivo entre TODAS as solucoes de TODOS os
    metodos como ponto nadir, deslocado 1% para baixo. Isso garante que
    toda solucao domina o ponto de referencia e que o HVN fica em [0, 1].
    """
    all_pts = np.vstack([f for f in all_fronts if len(f) > 0])
    f1_min = all_pts[:, 0].min()
    f1_max = all_pts[:, 0].max()
    f2_min = all_pts[:, 1].min()
    f2_max = all_pts[:, 1].max()
    margin = 0.01
    ref = np.array([
        f1_min - margin * max(f1_max - f1_min, 1.0),
        f2_min - margin * max(f2_max - f2_min, 1e-9),
    ])
    return ref

def compute_indicators(approx, reference, ref_point):
    """Calcula indicadores de qualidade.

    hv_norm (HVN) = HV(approx) / HV(reference_front)  in [0, 1]
    Permite comparacao entre instancias de tamanhos diferentes.
    """
    hv_approx = hypervolume(approx, ref_point)
    hv_ref    = hypervolume(reference, ref_point)
    return {
        "hv": hv_approx,
        "hv_norm": hv_approx / hv_ref if hv_ref > 0 else 0.0,
        "coverage": set_coverage(approx, reference),
        "epsilon":  epsilon_indicator(approx, reference),
    }


def run_experiment(cfg: ExperimentConfig) -> None:
    root = Path(__file__).parent.parent
    catalog = root / "instances" / "catalog.csv"
    instances_root = root / "instances"
    results_dir = cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(catalog, instances_root, cfg.families, cfg.min_n, cfg.max_n)
    if not instances:
        print("[ERRO] Nenhuma instancia encontrada.")
        return
    print(f"Instancias: {len(instances)}  |  Familias: {cfg.families}")
    print(f"Modos: {list(cfg.modes)}  |  Replicas: {cfg.n_reps}\n")

    all_records = []
    all_fronts_by_inst = {}

    for inst in instances:
        print(f"  [{inst.name}] n={inst.n} m={inst.m}")
        fronts = []
        for mode in cfg.modes:
            recs = run_instance(inst, cfg, mode, results_dir)
            all_records.extend(recs)
            for r in recs:
                fronts.append(r["_pf"])
        all_fronts_by_inst[inst.name] = fronts

    # Calcula indicadores
    summary_rows = []
    for inst in instances:
        fronts = all_fronts_by_inst[inst.name]
        if not fronts:
            continue
        ref_front = compute_reference_front(fronts)
        ref_pt    = build_ref_point(fronts)
        recs_inst = [r for r in all_records if r["instance"] == inst.name]

        for mode in cfg.modes:
            mode_recs = [r for r in recs_inst if r["mode"] == mode]
            if not mode_recs:
                continue
            hvs, hvns, covs, epss, sizes, times = [], [], [], [], [], []
            for r in mode_recs:
                ind = compute_indicators(r["_pf"], ref_front, ref_pt)
                hvs.append(ind["hv"])
                hvns.append(ind["hv_norm"])
                covs.append(ind["coverage"])
                epss.append(ind["epsilon"])
                sizes.append(r["pareto_size"])
                times.append(r["elapsed_s"])
            summary_rows.append({
                "instance": inst.name, "family": inst.family,
                "n": inst.n, "m": inst.m,
                "mode": mode, "label": MODE_LABEL.get(mode, mode),
                "hv_mean":           round(float(np.mean(hvs)),   6),
                "hv_std":            round(float(np.std(hvs)),    6),
                "hv_norm_mean":      round(float(np.mean(hvns)),  6),
                "hv_norm_std":       round(float(np.std(hvns)),   6),
                "coverage_mean":     round(float(np.mean(covs)),  5),
                "coverage_std":      round(float(np.std(covs)),   5),
                "epsilon_mean":      round(float(np.mean(epss)),  5),
                "epsilon_std":       round(float(np.std(epss)),   5),
                "pareto_size_mean":  round(float(np.mean(sizes)), 2),
                "time_mean_s":       round(float(np.mean(times)), 4),
            })

    if summary_rows:
        out_csv = results_dir / "summary.csv"
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[k for k in summary_rows[0] if not k.startswith("_")])
            writer.writeheader()
            writer.writerows(summary_rows)
        print("\nSummary salvo: " + str(out_csv) + "  (" + str(len(summary_rows)) + " linhas)")
