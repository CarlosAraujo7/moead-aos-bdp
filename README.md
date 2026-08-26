# MOEA/D-AOS para o Problema Bi-objetivo da Máxima Diversidade (BMDP)

Pesquisa em andamento com foco em submissão ao **European Journal of Operational
Research (EJOR)**: seleção adaptativa de operadores via bandidos multi-braço
(Multi-Armed Bandits) dentro do MOEA/D, aplicada ao Problema da Máxima
Diversidade bi-objetivo (MaxSum + MaxMin).

## O problema

Dado um conjunto de `n` elementos com distâncias par-a-par `d(i,j)`, selecionar
um subconjunto `S` com `|S| = m` que maximize simultaneamente:

- **MaxSum(S)** = soma de todas as distâncias par-a-par dentro de `S`
- **MaxMin(S)** = menor distância par-a-par dentro de `S`

Os dois objetivos são conflitantes (NP-difícil), o que caracteriza um problema
multiobjetivo genuíno — não uma soma ponderada disfarçada.

## Abordagem

- **MOEA/D** (Zhang & Li, 2007) com decomposição por escalarização de
  Tchebycheff.
- **Seleção adaptativa de operadores (AOS)** via bandidos multi-braço: UCB1
  (janela deslizante), Thompson Sampling (binário) e Dynamic Thompson Sampling
  / DYTS (Sun & Li, 2020 — recompensas contínuas, prior Normal-inv-qui-quadrado).
- 7 operadores de busca local/perturbação (`src/operators.py`), incluindo um
  operador GRASP-guloso original usado tanto como operador quanto como
  construção inicial (warm-start) do MOEA/D.
- Baselines de comparação: NSGA-II, GRASP-MO/GRASP-PR (construção + path
  relinking), MOEA/D sem AOS (seleção uniforme).
- Indicadores de qualidade: hipervolume exato 2D (HV e HV normalizado/HVN),
  Set Coverage, indicador epsilon aditivo.
- Testes estatísticos: Wilcoxon com correção de Holm-Bonferroni, Friedman +
  Nemenyi com diagramas de diferença crítica (CD).

## Estrutura do repositório

```
src/
  problem.py            Carregamento/avaliação de instâncias (formato MDPLIB)
  moead_aos.py           MOEA/D com seleção adaptativa de operadores
  mab.py                 Seletores: UCB1, Random, Thompson, DYTS
  operators.py           Pool de 7 operadores de busca/perturbação
  metrics.py             Indicadores de qualidade (HV, coverage, epsilon)
  analysis.py            Testes estatísticos e geração de figuras
  theory.py              Análises teóricas (conflito de objetivos, regret bound etc.)
  experiment.py          Orquestração de experimentos, orçamento por tamanho
  nsga2.py               Baseline NSGA-II
  grasp_mo.py            Baseline GRASP-MO / GRASP-PR
  vns.py                 Baseline VNS (implementado, ainda não conectado ao pipeline)
  instance_generator.py  Gerador das 4 famílias de instâncias com motivação real

run_full.py       Executor paralelo do experimento completo (multiprocessing)
export_results.py Exporta summary.csv a partir dos checkpoints em results/runs/
cleanup.py        Remove resultados/checkpoints obsoletos antes de um novo run

instances/
  catalog.csv       Catálogo mestre (family,name,n,m,filepath) -- única fonte
                     de verdade sobre quais instâncias o pipeline enxerga
  MDPLIB_2_0.pdf     Documentação oficial da biblioteca de benchmarks
  <familia>/         Uma pasta por família (não versionadas -- ver .gitignore)
```

## Instalação

```bash
pip install -r requirements.txt
```

Testado com Python 3.10.12. Único ambiente de execução verificado até o
momento é o desktop do autor; nenhuma dependência de SO específica foi
identificada.

## Uso básico

```bash
# 1. (opcional) remove resultados/checkpoints obsoletos
python cleanup.py --dry-run     # confere o que seria removido
python cleanup.py               # remove de fato

# 2. roda o experimento completo (paraleliza por processo)
python run_full.py --families GKD-c,SOM-a,MDG-a,SOM-b,MGPO --workers 8

# 3. exporta summary.csv + ranking a partir dos checkpoints
python export_results.py --analyze
```

`run_full.py` salva um checkpoint JSON por réplica em `results/runs/` assim
que ela termina -- interrupções (Ctrl+C) são seguras e retomáveis: réplicas já
concluídas são puladas na próxima execução.

## Catálogo de instâncias

183 instâncias em 11 famílias (`instances/catalog.csv`), combinando MDPLIB 2.0
(Martí, Duarte & Martínez-Gavara, 2021) com 4 famílias de instâncias com
motivação real geradas para este trabalho (portfólio de investimentos, formação
de equipes, posicionamento de sensores, localização de instalações).

Famílias MDPLIB ainda **não incluídas** no catálogo, com o motivo específico:

- **GKD-d**: cabeçalho do arquivo traz apenas `n` (sem `m`) -- precisa de um
  adaptador que decida `m` por convenção antes de poder ser carregada.
- **ORLIB**: formato `.sparse` de BQP (Beasley), com coeficientes que podem
  ser **negativos** -- não são distâncias de dissimilaridade e não podem ser
  reaproveitados diretamente sem uma decisão de modelagem.
- **PI**: matriz densa completa em formato linha-a-linha (não é a lista de
  arestas `i j d` que `problem.py` espera) -- precisa de um parser dedicado,
  além do mesmo problema de `m` ausente do GKD-d.
- **CDP/GDP (Const.)**: MDPLIB documenta estes como um problema **diferente**
  (Constrained/Generalized Dispersion Problem, com restrição de orçamento por
  item) -- confirmado por inspeção direta: o formato de arquivo embute custo/
  capacidade, não é so uma variação de cabeçalho. Decisão pendente com o autor
  sobre se isso entra como uma extensão do artigo ou fica de fora do escopo.

## Status atual / pendências conhecidas

- Os 6199 checkpoints existentes em `results/runs/` (não versionados) são de
  uma versão **anterior** do código, com um pool de apenas 4 operadores --
  o pool atual tem 7 (+ DYTS). Esses resultados **não são válidos** para o
  artigo final e precisam ser regenerados com o código atual.
- Citações a verificar/já verificadas: ver histórico de conversa com o
  assistente para o detalhamento de cada uma.

## Citação (biblioteca de instâncias)

Martí, R., Duarte, A., Martínez-Gavara, A. (2021). *The MDPLIB 2.0 Library of
Benchmark Instances for Diversity Problems.* Universidade de Valência.
