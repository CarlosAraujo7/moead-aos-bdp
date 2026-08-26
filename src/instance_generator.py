"""
instance_generator.py — Gerador de instancias BODP com motivacao real.

Gera instancias do Bi-objective Diversity Problem (BODP) a partir de
cenarios de aplicacao reais, fornecendo motivacao pratica para o artigo EJOR.

Cenarios implementados
----------------------
1. portfolio_diversity   : selecao de ativos financeiros diversos
   D[i,j] = 1 - |rho[i,j]|  (1 menos correlacao absoluta)
   Maximizar MaxSum = diversificacao total; MaxMin = pior correlacao no portfolio

2. team_diversity        : formacao de equipes com competencias diversas
   D[i,j] = distancia de competencias entre candidatos i e j
   Maximizar MaxSum = cobertura de competencias; MaxMin = evitar candidatos muito similares

3. sensor_placement      : posicionamento de sensores em area geografica
   D[i,j] = distancia euclidiana entre posicoes candidatas i e j
   Maximizar MaxSum = cobertura total; MaxMin = separacao minima (evitar redundancia)

4. facility_location     : localizacao de instalacoes com diversidade
   D[i,j] = distancia de viagem entre locais candidatos
   Aplicacao: Maximizar cobertura e garantir separacao minima entre instalacoes

Formato de saida
----------------
Compativel com MDPInstance (mesmo formato do MDPLIB 2.0):
  Linha 1: n m
  Linhas seguintes: matriz D (triangular superior)

Referencias
-----------
Duarte, A. et al. (2015). GRASP with exterior path relinking for differential
    dispersion minimization. Information Sciences, 296, 46-60.
Prokopyev, O. et al. (2009). The equitable dispersion problem. EJOR, 197, 59-67.
Palubeckis, G. (2007). Iterated tabu search for the maximum diversity problem.
    Applied Mathematics and Computation, 189, 371-383.
"""

from __future__ import annotations

import csv
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Estrutura de instancia real
# ---------------------------------------------------------------------------

@dataclass
class RealInstance:
    name: str
    scenario: str
    n: int
    m: int
    D: np.ndarray
    metadata: dict   # informacoes especificas do cenario


# ---------------------------------------------------------------------------
# Geradores por cenario
# ---------------------------------------------------------------------------

def generate_portfolio_diversity(
    n: int = 50,
    m: int = 10,
    seed: int = 42,
    n_assets_base: int | None = None,
) -> RealInstance:
    """Instancia de diversificacao de portfolio de ativos.

    Gera uma matriz de correlacoes simuladas usando o modelo fator:
        r_i = beta_i * F + epsilon_i
    onde F e um fator de mercado e epsilon_i e ruido idiossincratio.

    D[i,j] = 1 - |correlation(i,j)|: alta diversificacao = baixa correlacao.

    Problema real: selecionar m ativos de n candidatos maximizando
    (1) soma das diversificacoes par-a-par (MaxSum = baixo risco sistematico)
    (2) minima diversificacao entre qualquer par (MaxMin = nenhum par muito correlacionado)
    """
    rng = np.random.default_rng(seed)

    # Gera retornos sinteticos via modelo fator
    n_periods = 252   # 1 ano de dados diarios
    market_factor = rng.standard_normal(n_periods)
    betas  = rng.uniform(0.3, 1.5, size=n)
    sigmas = rng.uniform(0.1, 0.4, size=n)

    returns = np.outer(betas, market_factor) + \
              rng.standard_normal((n, n_periods)) * sigmas[:, None]

    # Matriz de correlacao
    corr = np.corrcoef(returns)

    # D[i,j] = 1 - |corr[i,j]|  (0=identicos, 1=sem correlacao)
    D = 1.0 - np.abs(corr)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 1.0)

    return RealInstance(
        name=f"portfolio_n{n}_m{m}_s{seed}",
        scenario="portfolio_diversity",
        n=n, m=m, D=D,
        metadata={
            "description": "Selecao de portfolio de ativos financeiros diversos",
            "n_periods": n_periods,
            "beta_range": (float(betas.min()), float(betas.max())),
            "avg_correlation": float(np.abs(corr[np.triu_indices(n, k=1)]).mean()),
        },
    )


def generate_team_diversity(
    n: int = 40,
    m: int = 8,
    n_skills: int = 15,
    seed: int = 42,
) -> RealInstance:
    """Instancia de formacao de equipes com diversidade de competencias.

    Cada candidato i tem um perfil de competencias skill_i in [0,1]^K.
    D[i,j] = distancia de cosseno entre perfis (diversidade de competencias).

    Problema real: selecionar m candidatos de n para formar equipe que
    (1) maximize a soma de diferencas de competencias (cobertura ampla)
    (2) garanta que nenhum par seja demasiado similar (evitar redundancia)
    """
    rng = np.random.default_rng(seed)

    # Gera perfis de competencia com estrutura de clusters (areas de especializacao)
    n_clusters = max(3, n_skills // 3)
    cluster_assign = rng.integers(0, n_clusters, size=n)
    skills = np.zeros((n, n_skills))

    for i in range(n):
        c = cluster_assign[i]
        # Alta competencia na area de cluster, baixa nas demais
        base = rng.uniform(0.0, 0.3, size=n_skills)
        expertise_start = (c * (n_skills // n_clusters)) % n_skills
        expertise_end   = expertise_start + (n_skills // n_clusters)
        base[expertise_start:expertise_end] = rng.uniform(0.6, 1.0,
                                                           size=expertise_end - expertise_start)
        skills[i] = np.clip(base + rng.normal(0, 0.05, n_skills), 0, 1)

    # Distancia de cosseno: D[i,j] = 1 - cos(skill_i, skill_j)
    norms = np.linalg.norm(skills, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    skills_n = skills / norms
    D = 1.0 - skills_n @ skills_n.T
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0) / 2.0   # normaliza para [0,1]

    return RealInstance(
        name=f"team_n{n}_m{m}_k{n_skills}_s{seed}",
        scenario="team_diversity",
        n=n, m=m, D=D,
        metadata={
            "description": "Formacao de equipes com diversidade de competencias",
            "n_skills": n_skills,
            "n_clusters": n_clusters,
            "avg_diversity": float(D[np.triu_indices(n, k=1)].mean()),
        },
    )


def generate_sensor_placement(
    n: int = 60,
    m: int = 12,
    area_size: float = 100.0,
    obstacle_frac: float = 0.1,
    seed: int = 42,
) -> RealInstance:
    """Instancia de posicionamento de sensores em area geografica.

    n posicoes candidatas em uma area 2D. D[i,j] = distancia euclidiana.
    Obstaculos penalizam a distancia efetiva (modelagem simplificada).

    Problema real: selecionar m posicoes que
    (1) maximizem a cobertura total (MaxSum = soma de distancias)
    (2) garantam separacao minima (MaxMin = nenhum sensor demasiado proximo)
    """
    rng = np.random.default_rng(seed)

    # Posicoes candidatas (clusters para simular areas de interesse)
    n_clusters = max(3, n // 10)
    centers = rng.uniform(10, area_size - 10, size=(n_clusters, 2))

    positions = np.zeros((n, 2))
    for i in range(n):
        c = rng.integers(n_clusters)
        positions[i] = centers[c] + rng.normal(0, area_size / (4 * n_clusters), 2)
    positions = np.clip(positions, 0, area_size)

    # Distancias euclidianas
    diff = positions[:, None, :] - positions[None, :, :]
    D = np.sqrt((diff**2).sum(axis=2))

    # Penaliza pares com obstaculo no caminho (simplificado: 10% dos pares)
    if obstacle_frac > 0:
        n_pairs = n * (n - 1) // 2
        n_obstacles = int(n_pairs * obstacle_frac)
        rows_idx, cols_idx = np.triu_indices(n, k=1)
        obstacle_pairs = rng.choice(len(rows_idx), size=n_obstacles, replace=False)
        for idx in obstacle_pairs:
            r, c = rows_idx[idx], cols_idx[idx]
            D[r, c] *= rng.uniform(0.3, 0.7)   # obstaculo reduz distancia efetiva
            D[c, r] = D[r, c]

    np.fill_diagonal(D, 0.0)

    return RealInstance(
        name=f"sensor_n{n}_m{m}_area{int(area_size)}_s{seed}",
        scenario="sensor_placement",
        n=n, m=m, D=D,
        metadata={
            "description": "Posicionamento de sensores em area geografica",
            "area_size": area_size,
            "n_clusters": n_clusters,
            "obstacle_fraction": obstacle_frac,
            "positions": positions.tolist(),
        },
    )


def generate_facility_location(
    n: int = 80,
    m: int = 15,
    n_cities: int = 20,
    seed: int = 42,
) -> RealInstance:
    """Instancia de localizacao de instalacoes com diversidade geografica.

    n locais candidatos, distancias baseadas em estrutura de cidade real
    (distancias de rua, nao euclidianas).

    Problema real: selecionar m locais para instalacoes (lojas, clinicas,
    escolas) que maximizem tanto cobertura total quanto separacao minima
    (evitar concentracao em uma unica regiao).
    """
    rng = np.random.default_rng(seed)

    # Gera mapa simplificado: grid + perturbacao + bairros
    n_bairros = max(4, n_cities // 3)
    bairro_centers = rng.uniform(5, 95, (n_bairros, 2))

    positions = np.zeros((n, 2))
    for i in range(n):
        b = rng.integers(n_bairros)
        positions[i] = bairro_centers[b] + rng.normal(0, 8, 2)
    positions = np.clip(positions, 0, 100)

    # Distancia de rua: aprox. L1 (Manhattan) com fator aleatorio
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Distancia Manhattan + perturbacao para simular congestionamento
            d_manhattan = np.abs(positions[i] - positions[j]).sum()
            traffic_factor = rng.uniform(1.0, 1.5)
            d = d_manhattan * traffic_factor
            D[i, j] = D[j, i] = d

    np.fill_diagonal(D, 0.0)

    return RealInstance(
        name=f"facility_n{n}_m{m}_cities{n_cities}_s{seed}",
        scenario="facility_location",
        n=n, m=m, D=D,
        metadata={
            "description": "Localizacao de instalacoes com diversidade geografica",
            "n_bairros": n_bairros,
            "n_cities": n_cities,
            "positions": positions.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Salvamento no formato MDPLIB
# ---------------------------------------------------------------------------

def save_instance_mdplib(inst: RealInstance, output_path: Path) -> None:
    """Salva a instancia no formato MDPLIB 2.0 (compativel com MDPInstance)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as fh:
        fh.write(f"{inst.n} {inst.m}\n")
        for i in range(inst.n):
            for j in range(i + 1, inst.n):
                fh.write(f"{i} {j} {inst.D[i, j]:.6f}\n")

    # Salva metadados em JSON separado
    import json
    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w") as fh:
        json.dump({
            "name": inst.name,
            "scenario": inst.scenario,
            "n": inst.n, "m": inst.m,
            **{k: v for k, v in inst.metadata.items() if k != "positions"},
        }, fh, indent=2)


# ---------------------------------------------------------------------------
# Geracao em lote para o artigo
# ---------------------------------------------------------------------------

REAL_INSTANCE_CATALOG = [
    # Portfolio diversity: n=50,100 ; m=10,20
    {"scenario": "portfolio", "n": 50,  "m": 10, "seed": 1},
    {"scenario": "portfolio", "n": 50,  "m": 10, "seed": 2},
    {"scenario": "portfolio", "n": 100, "m": 20, "seed": 1},
    {"scenario": "portfolio", "n": 100, "m": 20, "seed": 2},
    # Team diversity: n=40,80 ; m=8,15
    {"scenario": "team",      "n": 40,  "m":  8, "seed": 1},
    {"scenario": "team",      "n": 40,  "m":  8, "seed": 2},
    {"scenario": "team",      "n": 80,  "m": 15, "seed": 1},
    {"scenario": "team",      "n": 80,  "m": 15, "seed": 2},
    # Sensor placement: n=60,120 ; m=12,20
    {"scenario": "sensor",    "n": 60,  "m": 12, "seed": 1},
    {"scenario": "sensor",    "n": 60,  "m": 12, "seed": 2},
    {"scenario": "sensor",    "n": 120, "m": 20, "seed": 1},
    {"scenario": "sensor",    "n": 120, "m": 20, "seed": 2},
    # Facility location: n=80,150 ; m=15,25
    {"scenario": "facility",  "n": 80,  "m": 15, "seed": 1},
    {"scenario": "facility",  "n": 80,  "m": 15, "seed": 2},
    {"scenario": "facility",  "n": 150, "m": 25, "seed": 1},
    {"scenario": "facility",  "n": 150, "m": 25, "seed": 2},
]


def generate_all_real_instances(output_dir: Path) -> list[dict]:
    """Gera todas as instancias reais e atualiza o catalogo."""
    real_dir = output_dir / "real_world"
    real_dir.mkdir(parents=True, exist_ok=True)

    catalog_rows = []
    for spec in REAL_INSTANCE_CATALOG:
        sc   = spec["scenario"]
        n, m = spec["n"], spec["m"]
        seed = spec["seed"]

        if sc == "portfolio":
            inst = generate_portfolio_diversity(n=n, m=m, seed=seed)
        elif sc == "team":
            inst = generate_team_diversity(n=n, m=m, seed=seed)
        elif sc == "sensor":
            inst = generate_sensor_placement(n=n, m=m, seed=seed)
        else:
            inst = generate_facility_location(n=n, m=m, seed=seed)

        fname = f"{inst.name}.txt"
        fpath = real_dir / fname
        save_instance_mdplib(inst, fpath)

        catalog_rows.append({
            "name":    inst.name,
            "family":  f"real_{sc}",
            "n": n, "m": m,
            # BUGFIX: o caminho deve ser relativo a output_dir (=instances/),
            # que e exatamente a raiz usada por load_instances() em experiment.py
            # (instances_root / row["filepath"]). A versao anterior usava
            # output_dir.parent (raiz do projeto), gravando "instances/real_world/..."
            # em vez de "real_world/...", o que fazia load_instances() procurar
            # "instances/instances/real_world/..." (inexistente) e descartar
            # silenciosamente as 16 linhas real_* do catalogo.
            "filepath": str(fpath.relative_to(output_dir)),
        })
        print(f"  Gerada: {inst.name}  ({inst.scenario})")

    return catalog_rows


# ---------------------------------------------------------------------------
# Adiciona instancias reais ao catalog.csv existente
# ---------------------------------------------------------------------------

def append_to_catalog(new_rows: list[dict], catalog_path: Path) -> None:
    """Adiciona novas instancias ao catalog.csv sem duplicar.

    BUGFIX: as linhas novas agora sao escritas na MESMA ordem de colunas do
    cabecalho ja existente no arquivo. A versao anterior usava a ordem das
    chaves do dict (name, family, n, m, filepath), diferente da ordem real
    do catalogo (family, name, n, m, filepath) -- isso gravava as colunas
    family/name trocadas para toda linha adicionada por esta funcao.
    """
    existing_names = set()
    fields = None
    if catalog_path.exists() and catalog_path.stat().st_size > 0:
        with catalog_path.open() as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames
            for row in reader:
                existing_names.add(row["name"])

    to_add = [r for r in new_rows if r["name"] not in existing_names]
    if not to_add:
        print("Nenhuma instancia nova para adicionar ao catalogo.")
        return

    if fields is None:
        fields = list(to_add[0].keys())

    with catalog_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if catalog_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(to_add)
    print(f"Adicionadas {len(to_add)} instancias ao catalogo.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = Path(__file__).parent.parent
    output_dir  = root / "instances"
    catalog_path = root / "instances" / "catalog.csv"

    print("Gerando instancias com motivacao real para o artigo EJOR...")
    new_rows = generate_all_real_instances(output_dir)
    append_to_catalog(new_rows, catalog_path)
    print(f"\nTotal gerado: {len(new_rows)} instancias em {output_dir / 'real_world'}/")
    print("Catalogo atualizado:", catalog_path)
