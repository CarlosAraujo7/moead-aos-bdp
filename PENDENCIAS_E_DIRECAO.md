# Pendências e recomendações de direção

Montado em 3 out 2026, às 20h40, a partir do estado real dos dados e do manuscrito (`Claude outputs/manuscript_draft_v2.md`). Substitui as listas de pendências anteriores do `CLAUDE.md` §12 e do `PLANO_PUBLICACAO.md`. **Custos de máquina são estimativas** (calibradas pelos tempos medidos com o código otimizado e 8 workers; a máquina é compartilhada com outros trabalhos).

## 1. Onde estamos

- **Dados da Fase 1 completos e analisados**: 443 instâncias (MDP 143, CDP 100, GDP 200), K=12 métodos, 3 réplicas (5 em n≤150 de CDP/GDP), regime normalizado, referência fixa, 3 indicadores.
- **Resultados que sustentam o artigo**: (1) normalizar os objetivos é decisivo (+0,12 a +0,59 de hv_norm; efeito cresce com n); (2) o pool de 8 operadores no MOEA/D fica significativamente à frente de NSGA-II, GRASP e VNS nos 3 problemas; (3) nenhum seletor adaptativo supera a seleção uniforme (3 seletores × 4 recompensas + custo + piso de exploração), e os de recompensa escalar são significativamente piores; (4) cada operador isolado é pior que o pool, e o pool aguenta podas; (5) a ancoragem de MaxMin leva a ponta da frente ao nível dos melhores valores publicados e melhora o hv_norm (positivo em 6 de 6 situações, significativo em 4); (6) dados e avaliador validados contra os benchmarks públicos (200/200 instâncias GDP idênticas; 80/80 soluções publicadas reproduzidas).
- **Manuscrito**: rascunho completo de 6,4 mil palavras, com 12 `[TODO]` e 15 `[CITE?]`.

### Estado em 5 out 2026 (noite) — o que mudou desde 3 out

- **Feitos**: E1 (varredura de hiperparâmetros: nenhuma variante supera a seleção uniforme; 5 piores), E9 (por família; Apêndice C gerado), conferência dos 15 PDFs (autorias, BKS atualizado, achado do `dyts_*` ≠ DYTS), figuras 1–4 do manuscrito, lista de referências com status de verificação, materiais de submissão (highlights, declaração de dados, carta; limites do EJOR a confirmar), primeira busca sistemática (web).
- **Em execução** (fila única `python run_queue.py 8`, aba `c7`; idempotente): E1b → E3 → E4 → E7 (n≤150) → E2 → E7 (n=500). Previsão ~12 h de máquina.
- **Achado importante da busca (M3)**: o MDP bi-objetivo MaxSum+MaxMin **já existe** (Colmenar, Martí & Duarte 2018, ESWA 108:193–205: NSGA-II, SPEA2, GRASP, Iterated Greedy, Tabu, VNS em instâncias do MDPLIB) e Vera et al. 2018 (MOEA, 5 medidas). A novidade fica restrita a CDP/GDP bi-objetivo, ao estudo de pool/AOS/normalização/âncora e ao protocolo. **Você precisa conseguir o artigo do Colmenar 2018** (e os pedidos em `REFERENCIAS_A_CONSEGUIR.md` §0b) para eu ler os resultados deles e ajustar o texto.
- **Risco novo para o EJOR**: o periódico pede "a major new research finding or novel approach". Com o MDP bi-objetivo não sendo novo, o argumento de novidade precisa de CDP/GDP + normalização + resultado negativo robusto + âncora. Vale conversar com o orientador/coautor sobre o enquadramento (e sobre o CEC como alternativa realista).
- **Decisões que ficam com você**: incluir ou não a amostra de escalabilidade (E6; custo estimado de ~30 h de máquina em n=2000 do GDP GKD-d com 40 instâncias publicadas); investir mais na âncora do MDG-b (hoje 0,70 do melhor valor; E7 testa a busca SBTS-like); quando congelar a referência e commitar.

## 2. Pendências

### 2.1 Experimentos (ordenados por impacto na aceitação)

| # | Experimento | Por que importa | Custo estimado | Prioridade |
|---|---|---|---|---|
| E1 | **[LANÇADO 3 out, aba `c2`]** **Varredura de hiperparâmetros dos bandits** (UCB: c ∈ {0,05; 0,5; 2}, janela ∈ {20; 50; 200}; janela/desconto do Thompson e do DYTS) | O ataque mais provável de um revisor ao resultado negativo: "os seletores estavam mal afinados". Sem isto, a conclusão central é vulnerável | ~10–15 min de parede por configuração (CDP/GDP n≤150, 3 réplicas); ~2–3 h no total | **P1** |
| E1b | **[ENFILEIRADO 3 out, aba `c3`]** **DYTS fiel a Sun & Li (2020)** (Beta, C=100, recompensa binária) + UCB/Thompson com a recompensa `replace` do artigo, CDP e GDP n≤150 | Descobrimos que o nosso `dyts_*` era um amostrador gaussiano, não o DYTS do artigo; sem isto um revisor que conhece o DYTS ataca o resultado negativo | ~1–2 h | **P1** |
| E2 | **[LANÇADO 3 out, aba `c2`, após E1]** **Orçamento maior** (2× e 4× gerações) para uniforme e os melhores seletores | "O AOS compensa em buscas longas?" Hoje só testamos o orçamento padrão | 2× e 4× o custo base; ~1–2 h num subconjunto (CDP n≤150) | **P1** |
| E3 | **Bandit contextual** (política por subproblema ou por fase da busca) | É a única família de seletor ainda não testada; fecha a pergunta RQ2 de vez | ~1 dia de implementação + ~1 h de máquina | **P1** |
| E4 | **Tempo igual para os baselines no regime normalizado**, no GDP, com 20+ instâncias e 3 réplicas (hoje: piloto de 6 instâncias CDP, regime bruto, 1 réplica) | Responde "os baselines só perdem por orçamento". As otimizações de código mudaram os custos relativos | ~3–5 h | **P1** |
| E5 | **Ablação do orçamento CDP vs GDP no regime normalizado com réplicas** (hoje: regime bruto, 1 réplica, n≤150) | Sustenta a seção 6.9 do texto; o regime bruto contaminou a primeira leitura | n≤150: ~1–2 h; n=500 só numa amostra de 20 instâncias (~8 h) | P2 |
| E6 | **Amostra de escalabilidade** MDG-b n=2000 e MDG-c n=3000 (3 instâncias, 3 réplicas, subconjunto de métodos) | Hoje o maior n do MDP é 1000; revisores de EJOR esperam ver tamanho realista | ~5–10 h (estimativa antiga de ~0,5–0,9 h por run serial é pessimista) | P2 |
| E7 | **Ancoragem melhor** no MDG-b do CDP (**0,70** do melhor valor publicado depois da conferência de 3 out; 0/20 atingem; SBTS/VNS/MNSA valem ~0,91 da média SBTS-10 s) — orçamento 20k×3, tabu aprimorado ou algo do tipo SBTS (hash de soluções + vizinhanças inserir/remover/trocar); e uma versão para o MDP (cardinalidade fixa, hoje não se aplica) | Fecha a única lacuna que sobrou contra a literatura; é a lacuna mais visível para um revisor de EJOR | ~0,5–1 dia de implementação + ~1 h | **P1–P2** |
| E8 | **Ablação de operador único/leave-one-out com réplicas** no regime normalizado, GDP e n=500 (hoje: CDP n≤150, 1 réplica) | Reforça a seção 6.5 | ~2–3 h | P2 |
| E9 | **MDP por família** (análise sobre dados já gerados) | Mostra onde os baselines chegam perto (VNS-Tchebycheff no MDP) | minutos | P2 |
| E10 | IGD⁺/ε⁺ nos recortes por tamanho; segunda instância de teste de significância | Acabamento | minutos | P3 |

### 2.2 Manuscrito

| # | Item | Observação |
|---|---|---|
| M1 | **Você ler o rascunho e confirmar enquadramento e título** | Bloqueia o resto; é a decisão que mais pesa |
| M2 | **Conferir as 15 referências `[CITE?]`** (Peiró, Lu, Rosati & Schaerf 2024, Landete 2023, Gomez et al., EJOR 2026, DYTS, UCB/AOS, NSGA-II, GRASP, MOEA/D) | Precisa dos artigos; eu não tenho acesso a todos. Você já tem o *Symmetry* e o pacote do Zenodo |
| M3 | **Busca sistemática de trabalhos prévios** em otimização multiobjetivo de CDP/GDP/diversidade — **PRIMEIRA PASSADA FEITA em 5 out (web)**: achou o BODP (Colmenar, Martí & Duarte 2018) e Vera et al. 2018 → novidade restrita a CDP/GDP e ao estudo de pool/AOS; repetir em Scopus/WoS e ler Colmenar 2018 antes de submeter | O texto já foi ajustado; falta a leitura do artigo (Carlos precisa conseguir) |
| M4 | **Fechar os 12 `[TODO]`**: dados (E5, E6, E4), figuras, DOI | Posso fechar os de dados quando os experimentos rodarem |
| M5 | **Figuras**: diagramas CD (3 problemas), rank vs n, frequência de seleção por recompensa, cobertura da ponta com e sem âncora, ganho da normalização vs n | Eu gero; você valida o desenho |
| M6 | **Discussão e relacionados mais profundos; apêndice por família; apêndice de hiperparâmetros** | Depende de E1, E9 |
| M7 | **Requisitos do EJOR**: Highlights, graphical abstract, CRediT, data availability statement, cover letter | Segundo o guia de autores do EJOR |
| M8 | Revisão de inglês e leitura crítica por orientador/coautor | |

### 2.3 Engenharia e reprodutibilidade

| # | Item | Observação |
|---|---|---|
| G1 | **Git**: 11 arquivos modificados e dezenas de novos sem commit | Você pediu para não commitar; fica pendente até você autorizar. Antes de submeter é obrigatório |
| G2 | **Limpeza**: `_to_delete/`, arquivos de debug avulsos, `results/runs` obsoleto, `.zip` de 1,2 GB em `a pedidos/` (4,1 GB no total); `.gitignore` para `a pedidos/` | Posso fazer quando você autorizar |
| G3 | **Congelamento**: reconstruir a referência fixa, rerodar todas as análises, regenerar tabelas, fixar `CODE_VERSION` e versões de pacotes | Último passo antes de submeter; qualquer rodada nova depois disso obriga a refazer |
| G4 | **Pacote de dados e código com DOI** (ex.: Zenodo) e README | Exigido pela data availability do EJOR; os `results/` têm ~140 MB |
| G5 | **Script único de regressão** (reexecuta N runs com a semente gravada e compara bit a bit) | Hoje feito à mão; vale automatizar para o pacote público |

## 3. Decisões que dependem de você

1. **Enquadramento**: "pool de operadores + normalização + ancoragem" com o resultado negativo sobre seleção adaptativa como achado central. **DECIDIDO em 3 out 2026: seguir o recomendado.**
2. **Venue e data**: EJOR como alvo principal; IEEE CEC 2027 (artigo completo até 28 fev 2027) só como plano B. *Recomendo decidir até 15 nov.*
3. **Âncora nas tabelas principais**: linha extra "Pool + Âncora" e tabela de cobertura das pontas; baselines sem âncora; cenário "âncora em todos os métodos" no apêndice. *Recomendo assim.*
4. **Escalabilidade**: incluir a amostra de MDG-b n=2000 / MDG-c (E6) ou declarar como limitação. *Recomendo incluir, com 3 instâncias.*
5. **Quando commitar e publicar o repositório** (G1, G4).
6. **Revisão interna**: quem lê o rascunho antes da submissão, e quando.

## 4. Recomendação de direção

1. **Não tentar salvar o AOS; consolidar o achado.** A evidência é forte, replicada e validada contra a literatura, e o artigo tem quatro achados positivos além do negativo (normalização, superioridade do pool sobre os baselines, ancoragem, dados e protocolo abertos). O risco não é o resultado negativo em si, e sim ele parecer produto de seletores mal configurados.
2. **Blindar o resultado negativo antes de submeter** com E1 a E4. São baratos (cerca de um dia de máquina somados) e respondem às três perguntas que um revisor faria primeiro: afinação, orçamento e política dependente de estado.
3. **Submeter ao EJOR**, não ao CEC, como artigo principal. O trabalho é grande demais para 8 páginas (3 problemas, 443 instâncias, réplicas, validação) e se encaixa no perfil de metodologia computacional do periódico. Manter o CEC só como plano B se o cronograma escorregar. Evitar um artigo curto com o recorte "AOS negativo" em paralelo, pelo risco de publicação duplicada.
4. **Ordem de trabalho**: experimentos de blindagem (E1–E4) → decisão de enquadramento (você) → figuras e discussão → fechamento dos dados (E5–E9) → congelamento e pacote de dados → submissão.

## 5. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| "AOS mal afinado" | E1, E2, E3 |
| "Baselines caseiros ou fracos" | Já mostrado: nossos extremos igualam ou superam os valores publicados; E4 (tempo igual); assumir a limitação no texto |
| "Resultado negativo sem novidade" | Colocar normalização, ancoragem, protocolo e dados abertos como contribuições de primeira linha |
| Novidade frente ao *Symmetry* 2026 e outros | M3 (busca sistemática); frisar que o 2º objetivo e os métodos comparados são diferentes |
| Reprodutibilidade | G3, G4, G5 |
| "Hardware único; orçamento por gerações" | E4; relatar também o número de avaliações; assumir a limitação |

## 6. Cronograma proposto (referência; ajustar às suas datas)

| Período | Entregas |
|---|---|
| 4–17 out | E1–E4 e E9 (eu); M1, M2 e M3 (você); implementar E3 e E7 |
| 18 out – 15 nov | E5, E6, E7, E8; figuras; discussão e relacionados; primeira rodada de revisão interna; decisão de venue |
| 16 nov – 15 dez | Congelamento (G3); refazer tabelas; inglês; pacote de dados e DOI; requisitos do EJOR; limpeza (G2) |
| 15 dez – 15 jan | **Submissão ao EJOR (alvo)** |
| 28 fev 2027 | Prazo do CEC 2027 (plano B) |
