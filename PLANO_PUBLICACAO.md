# Plano de consolidação para publicação (EJOR / IEEE CEC)

Escrito em 1 out 2026, a partir do que foi verificado na sessão de 1 out (detalhes e números em `CLAUDE.md` §2–§2h). Em português; termos técnicos ficam como estão. Estimativas de custo computacional usam `elapsed_s` medido sob contenção (10–11 workers em 12 núcleos) e têm incerteza de ~2×.

## 0. Onde estamos, em cinco linhas

- O catálogo CDP/GDP é consistente (o alarme de contaminação por `budget_for()` era falso), mas **toda estatística atual vem do regime sem normalização de objetivos**, que é claramente pior (+0,12 a +0,20 de hv_norm quando normalizado).
- **Nenhuma estratégia AOS supera `random`** (CDP e GDP, com ou sem normalização; três ficam significativamente piores no GDP normalizado). O ganho vem do **pool de operadores dentro do MOEA/D**: todo operador isolado é significativamente pior que o pool, e o pool é redundante (tirar `biobj_best_swap` mantém a qualidade com ~40% menos tempo).
- A diferença estrutural CDP-vs-GDP é em boa parte explicada pela restrição de orçamento (relaxá-la move GRASP ~1,8 posições, NSGA-II −1,5), mas a reversão é **parcial**.
- Tudo é **1 réplica**, e as ablações são só n≤150. n=500 e MDP completo ficaram de fora.
- O rascunho do artigo (§4.1–4.7) está desatualizado; metodologia e discussão não existem.

## 1. Estratégia de venue

| | IEEE CEC 2027 | EJOR |
|---|---|---|
| Prazo | artigo completo **28 fev 2027**, curto 15 mar 2027, sem prorrogação; Edimburgo, 25–28 jul 2027 ([fonte](https://ieeecec.org/2027), [FAQ](https://ieeecec.org/2027/about/general-faq)) | fluxo contínuo |
| Tipo de contribuição | empírica/algorítmica focada | metodologia de PO com rigor computacional; exige Highlights, graphical abstract, CRediT e data availability statement ([guia](https://www.sciencedirect.com/journal/european-journal-of-operational-research/publish/guide-for-authors)); trabalhos de metaheurísticas entram em "Computational Intelligence" ou "Discrete Optimization" |
| Encaixe com o que temos hoje | **bom**: "pool vs seleção adaptativa" + "normalização importa" é um achado enxuto e defensável | **só se** houver método melhor ou análise muito completa (3 problemas, réplicas, baselines fortes, benchmark público) |

**Recomendação (decisão do Carlos, D1):** estratégia em dois tempos. (1) **CEC 2027** como marco com data firme, focado em análise de componentes (pool × seleção × normalização) em MDP + CDP/GDP. (2) **EJOR** depois, como artigo estendido com os três problemas, método aprimorado (se a Fase 2 render) e benchmark/dados públicos. Antes de decidir, **conferir a política da Elsevier/EJOR sobre versão estendida de artigo de conferência** e o limite de páginas do CEC na chamada oficial (não verifiquei). Se preferir EJOR direto, o plano abaixo vale igual; só sai o marco de fevereiro.

## 2. Pontos fracos, em ordem de gravidade

### Críticos (um revisor rejeita por causa deles)

| # | Ponto fraco | Evidência | Correção |
|---|---|---|---|
| C1 | A tese "MOEA/D-AOS domina" não vale no ponto que o nome promete: o AOS não supera `random` | §2g: HV-reward ≈ uniforme por construção (recompensa satura); scalar adapta mas é o pior e fica < `random` no GDP | Reenquadrar (pool-centric) **ou** melhorar o AOS (Fase 2, com gate). Não submeter com o título "AOS" sem uma dessas duas |
| C2 | Regime sem normalização (Tchebycheff com MaxSum ~1e4–1e5 vs MaxMin ~1e0–1e2) | §2e: melhora em 12/12 (piloto), 48–56/60 (CDP), 79–101/120 (GDP); GRASP sai do topo no CDP | Tornar o regime normalizado o principal e re-rodar os 7 AOS (baselines não dependem). Reportar o histórico como ablação |
| C3 | 1 réplica por (instância, método); n=500 só em piloto de 6 casos | todos os `rep00` | Réplicas (Fase 1); estatística em dois níveis (média por instância → Friedman; mais tamanho de efeito Â12) |
| C4 | Baselines vulneráveis a "espantalho": VNS faz da ordem de 1.500 avaliações (500 iterações × ~3 movimentos; o contador de vizinhança reinicia a cada melhora, então não é um teto exato); NSGA-II usa operadores genéricos com reparo; GRASP-MO/PR foram criados por nós | §2b, §2b': VNS fica 10–12º mesmo com ×400 iterações e normalizado | Baselines mais fortes (Fase 3): NSGA-II com os mesmos operadores do pool; MOEA/D sem AOS com o melhor pool (já é `random`); âncora com a literatura |

### Altos

| # | Ponto fraco | Correção |
|---|---|---|
| A1 | Validade do bi-objetivo: CDP/GDP originais são mono-objetivo (só MaxMin); o artigo do GDP (Lozano-Osorio 2022) nunca foi lido; MDG-c tem MaxMin = 0 | Obter o artigo (acesso institucional/contato com autores); **âncora**: o extremo MaxMin da nossa frente vs melhores valores publicados de CDP/GDP e o extremo MaxSum vs MDPLIB. Se não bater, o benchmark não é comparável |
| A2 | MDP incompleto: 78/183 instâncias; K=12 nunca rodado no catálogo completo; resultados em `results/runs/` obsoletos; MDP **não** foi testado com normalização | Rodar MDP K=12 normalizado com critério de inclusão explícito (ex.: n≤1000; MDG-c n=3000 só se o custo permitir) |
| A3 | Métrica: `hv_norm` usa frente de referência agrupada **dentro do conjunto de métodos comparados** (muda com o conjunto → risco de inversão de ranks). Nas ablações o conjunto variou (12, 14, 20 métodos) | Fixar uma frente de referência única (união de todos os runs de todas as ablações + nadir fixo) e reportar também IGD+ e ε. Testar sensibilidade do ranking ao ponto de referência |
| A4 | Justiça de orçamento só testada em CDP (6 instâncias, 1 semente); tempos medidos sob contenção | Protocolo oficial: reportar nº de avaliações **e** tempo em condição limpa; piloto de tempo igual ampliado (GDP, 20+ instâncias, 3 sementes) |
| A5 | Teoria: das 4 Proposições, 1 e 4 já foram rebaixadas/corrigidas; a 3 (regret do UCB com janela) assume recompensa em [0,1] e estacionária, mas as recompensas reais são absolutas e não limitadas; `operator_complementarity_matrix` e `operator_phase_analysis` (theory.py) usam `operator_counts` corrompido nos runs antigos (§2g). Hipótese a verificar: a análise de densidade de pesos (Prop. 4, `sweep_weight_density`) pode depender da escala dos objetivos, logo da normalização | Auditar cada Proposição contra o regime normalizado; refazer as duas análises de operadores com contagens corretas; manter só o que tem prova ou evidência direta |

### Médios

| # | Ponto fraco | Correção |
|---|---|---|
| M1 | Hipótese CDP-vs-GDP só com suporte parcial, n≤150, regime não normalizado | Repetir normalizado, incluir n=500; ablação intermediária (orçamento ×1,5, ×2) para curva dose-resposta |
| M2 | Custo: `biobj_best_swap` ≈ metade do tempo do pool; HV da recompensa recalculado a cada filho não dominado | Profilar antes de otimizar (numba/vetorização do HV; atualização incremental); ganho esperado não medido |
| M3 | Throughput: contenção medida ~3,2× com 10 workers | Benchmark de 6/8/10/12 workers numa tarefa fixa (≈30 min); pode economizar dias |
| M4 | Reprodutibilidade/engenharia: checkpoint não detecta mudança de parâmetro; ~1.200+ linhas sem commit (Carlos pediu para não commitar); dados/código para o data availability statement | Gravar hash da configuração em cada JSON e invalidar o checkpoint quando mudar; **commit só quando o Carlos autorizar**; preparar pacote de dados |
| M5 | Manuscrito: rascunho desatualizado; falta metodologia e discussão; inglês; peças do EJOR | Fase 5 |

## 3. Fases e cronograma (hoje = 1 out 2026; CEC = 28 fev 2027)

| Fase | Período | Conteúdo | Saída / critério |
|---|---|---|---|
| **0. Decisões e infraestrutura** | 1–15 out | D1 (venue), D2 (normalização principal). Hash de configuração no checkpoint; benchmark de workers (M3); profiling (M2); fixar frente de referência única (A3) | decisões registradas; `run_full.py` pronto para réplicas sem risco de misturar regimes |
| **1. Consolidar o regime principal** | 15 out–30 nov | Regime normalizado como padrão (`run_full.py --normalize`, 4 workers). Re-rodar os 7 AOS em CDP/GDP: **n≤150 com 5 réplicas** (~1 h de parede por réplica) e **n=500 com 3 réplicas** (~20 h de parede por réplica, ~2,5 dias no total, já incluindo GRASP/NSGA-II). MDP K=12 normalizado em n≤1000 (~7 h de parede por réplica) + amostra de escalabilidade em MDG-b n=2000 / MDG-c | tabelas §9 refeitas; Friedman+Nemenyi e Wilcoxon-Holm (critério principal) por problema; Â12 |
| **2. Melhorar o AOS (time-box de 5 semanas)** | 15 nov–20 dez | Candidatos, em ordem de custo/benefício: (a) **recompensa normalizada em [0,1] e ciente do custo** (ganho por segundo; os operadores diferem ~14× em custo); (b) pool enxuto (sem `biobj_best_swap`, sem `best_swap_1x1`) com bandit; (c) bandit com piso de exploração/probability matching; (d) bandit por subproblema. Testar em n≤150 com 5 réplicas | **Gate D3 (20 dez):** algum AOS supera `random` com p_holm<0,05 em CDP **e** GDP? Sim → enquadramento positivo. Não → enquadramento "pool-centric" e o AOS vira resultado negativo bem documentado |
| **3. Validade e baselines** | 1 dez–15 jan (em paralelo) | A1 (âncora com a literatura), C4 (NSGA-II com operadores do pool; revisar VNS), A4 (tempo igual em GDP), A3 (IGD+, ε), M1 (n=500, dose-resposta), A5 (teoria) | seção de validação; tabela "nosso extremo vs publicado" |
| **4. Congelar experimentos** | **31 jan 2027** | Nenhuma rodada nova depois; análise final | conjunto de resultados citável + figuras |
| **5. Escrita** | 1–20 fev (CEC); mar–mai (EJOR) | CEC: artigo curto focado (pool × seleção × normalização), submissão com folga de ~8 dias. EJOR: metodologia, discussão/conclusão, Highlights, graphical abstract, CRediT, data availability, repositório de dados | submetido |

Buffer: com o código otimizado a Fase 1 inteira cabe em ~1 semana de máquina, o que deixa folga para repetir rodadas se a Fase 0/3 achar algum problema. As fases 1 e 2 ainda competem por CPU (satura em 3–4 workers).

## 4. Orçamento computacional (reestimado em 1 out e CORRIGIDO com a medição real de 2 out 2026)

**Medido na execução real** (8 workers, i5-12450H, regime normalizado, K=12 métodos, código otimizado): CDP n=500 com 3 réplicas (1.320 tarefas novas) levou **~3,5 h de parede** (00:44→04:13), ou ~1,2 h por réplica; o GDP n=500 (80 instâncias, o dobro) leva ~2× isso. **A estimativa anterior (~20 h de parede por réplica de CDP+GDP) estava ~4–5× pessimista**: ela usava ~195 s por run de AOS, enquanto a média real com as otimizações é bem menor (a vazão medida com 8 workers é ~3,8× a de 1 processo). Ordem de grandeza real: **n=500, CDP+GDP, 3 réplicas ≈ 11–12 h de parede**; n≤150 com 5 réplicas ≈ 12 h (CDP ~4 h + GDP ~8 h, incluindo a pausa por hibernação). Todo o Phase 1 de CDP/GDP coube em ~1 dia de máquina.

**MDP (catálogo de 183 instâncias; estimativa por classe de tamanho, serial, 7 AOS, ver §2i do CLAUDE.md; provavelmente também pessimista)**: n≤150 e reais ~2,5 h; n=500 (30 instâncias) ~14 h; n=1000 (5) ~3,5 h; n=2000 (MDG-b, 20) ~71 h; n=3000 (MDG-c, 20) ~123 h. As duas famílias grandes são ~77% do custo. Recomendação mantida: comparação K=12 principal em n≤1000 (143 instâncias) e MDG-b n=2000 / MDG-c como estudo de escalabilidade com amostra de 3–5 instâncias.

## 5. Decisões que dependem do Carlos

- **D1 (até 8 out):** estratégia de venue (CEC + EJOR em dois tempos, ou só EJOR).
- **D2 (até 15 out):** normalização como regime principal (recomendado: sim).
- **D3 (20 dez):** resultado do gate do AOS → enquadramento positivo ou pool-centric.
- **D4 (15 jan):** go/no-go do CEC com o que existir; congelamento em 31 jan.
- **Commits:** autorizar (ou não) um commit de checkpoint antes da Fase 1 (hoje há muito código novo sem versionar; instrução vigente é não commitar).
- **Escopo MDP:** quais famílias/tamanhos entram na comparação K=12.

## 6. Riscos

1. **O AOS não vence o `random` mesmo melhorado.** Mitigação: o enquadramento pool-centric já está sustentado por evidência direta (operador único, leave-one-out) e é publicável como análise de componentes; só o título e a narrativa mudam.
2. **A âncora com a literatura não bate** (A1) — **parcialmente materializado em 1 out 2026**: dados e avaliador estão corretos (200/200 instâncias idênticas, 80/80 soluções publicadas reproduzidas), mas o extremo MaxMin das nossas frentes fica abaixo do BKS (CDP −5% a −12%; MDG-b n500 −52%; GDP GKD-c competitivo). Não invalida a comparação entre métodos (todos sofrem do mesmo viés de escala), mas um revisor questionará a cobertura das pontas da frente. Mitigação: normalização (fecha parte do gap, ver `CLAUDE.md` §2e/§2j) + semeadura/ancoragem mono-objetivo das pontas; repetir a âncora no regime normalizado antes de congelar os experimentos.
3. **Tempo de máquina estoura** (n=500 e MDP). Mitigação: reduzir réplicas de n=500 para 3, restringir MDP por tamanho, otimizar o hot path.
4. **Mudança de regime invalida tabelas já escritas** (§4.3/§4.4 do rascunho). Esperado; por isso o texto só deve ser reescrito depois da Fase 1.
5. **Política de duplicidade conferência→periódico.** Conferir antes de submeter ao CEC.

## 7. Próximos passos imediatos

Concluídos em 1 out 2026 (detalhes em `CLAUDE.md` §2i):
1. ~~Benchmark de workers~~ — satura em ~4 workers; padrão agora 4. ~~Profiling~~ — o gargalo era `BiObjBestSwap`/`evaluate_swap`, não o HV; otimização bit-exata, **~2,5–3,5× por geração**.
2. ~~Hash de configuração + invalidação de checkpoint~~, com `--normalize` no `run_full.py`.
3. ~~Frente de referência única + IGD+/ε+~~ — ranking robusto (tau 0,76–1,00; ver §2i).

Concluídos depois (1 out 2026, detalhes em `CLAUDE.md` §2j–§2l):
- **Âncora com a literatura (A1)**: dados/avaliador validados; a ponta de MaxMin das frentes estava abaixo do BKS (CDP −5% a −12%; MDG-b n500 −52%).
- **Piloto de n=500 (D2)**: efeito da normalização cresce com n (+0,38 a +0,50 de hv_norm no CDP n=500); GRASP sai do grupo do topo; nenhum AOS supera `random`. Recomendação: rodar n=500 normalizado com 3 réplicas.
- **Ancoragem de MaxMin (`src/anchor.py`, opt-in)**: leva a ponta a ~1,0 do BKS no GKD-c e ~0,8 no MDG-b (n=500) sem mudar os ranks entre métodos. Decisão pendente: parte do método ou análise complementar.
- **Benchmark de workers limpo**: a vazão continua subindo até 12 workers (4,3 → 13,2 → 19,7 runs/min com 1/4/12); o "satura em 4" anterior estava contaminado por outra sessão. Padrão agora `--workers 8`.

- **Fase 1 — andamento (2 out 2026)**: CDP e GDP n≤150, 5 réplicas, K=12, regime normalizado: **concluídos e analisados** (`analyze_replicated.py`; CDP N=60 e GDP N=120; Wilcoxon-Holm 42–52/66 pares significativos; grupo do topo `{random, hypervolume, thompson_hv, dyts_hv, dyts_scalar}`; nenhum AOS supera `random`; robusto a hv_norm/IGD+/ε+, exceto GRASP-PR no topo do CDP sob ε+). **n=500 com 3 réplicas em todas as instâncias (CDP 40, GDP 80): em execução** desde 2 out 00:44 (decisões D2 e ancoragem acordadas em 2 out; ver `CLAUDE.md` §2o). Próximas: MDP K=12 normalizado em n≤1000; Fase 2 (melhorar o AOS) e Fase 3 (baselines/validade).

Pendentes:
4. ~~Piloto de custo do MDP K=12 normalizado~~ — feito (§4 acima).
5. ~~Âncora com a literatura (A1)~~ — **feita em 1 out 2026** (ver `CLAUDE.md` §2j): dados e avaliador validados (200/200 instâncias idênticas; 80/80 soluções publicadas reproduzidas); **mas a ponta de MaxMin das nossas frentes fica abaixo do estado da arte** (CDP −5% a −12%, MDG-b −52%; GDP GKD-c competitivo, MDG-b ~0,5×). Novo item de alto impacto: cobrir as pontas da frente (normalização + semeadura/ancoragem mono-objetivo).
6. ~~Reestimar o orçamento computacional~~ — feito (§4 acima).
