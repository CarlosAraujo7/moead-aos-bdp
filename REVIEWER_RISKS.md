# Objeções prováveis de um revisor do EJOR e como o trabalho responde (rascunho de 10 out 2026)

Cada linha: a objeção, o que o manuscrito faz a respeito (seção, tabela/figura, script), e o risco residual. Todos os experimentos pendentes em 10 out (E8, E14–E17, SMS-EMOA) foram concluídos e integrados em 11 out, depois da correção de um artefato de ponto flutuante (ver linha própria na seção D).

## A. Novidade e posicionamento

| Objeção | Resposta | Risco residual |
|---|---|---|
| "O MDP bi-objetivo (MaxSum + MaxMin) já existe: Colmenar, Martí & Duarte (2018)." | Reconhecido na introdução, §2 e §4.6–4.7; o texto não reivindica o MDP. A novidade reivindicada é: CDP/GDP bi-objetivo com MOEAs, estudo de pool/seleção de operadores, normalização, âncora, e a comparação a tempo igual com baselines ajustados (incluindo o Tabu/VNS de Colmenar reimplementados). | Médio: o EJOR exige "major new finding"; o argumento depende do conjunto, não de um único item. Repetir a busca (Scopus/WoS) antes de submeter. |
| "Resultado negativo (AOS não ajuda) não é contribuição." | §6.3–6.4 e §6.11 mapeiam *quando* ajuda (n grande, orçamento de custo apertado, orçamento curto) com dose-resposta; §6.4 testa 5 famílias de remédios (Tabela 5 e Suplemento S1: 48 variantes, nenhuma supera o uniforme); §6.5 testa pool com 3 operadores nocivos (o uniforme não perde, nenhum seletor melhora). | Baixo–médio. |
| "Os problemas CDP/GDP originais são mono-objetivo; sua versão bi-objetivo é mal posta (tamanho livre → frente ilimitada)." | §3 e §6.13 declaram e quantificam: o tamanho é dimensão escondida; as frentes são uma fatia; rankings entre métodos no mesmo regime continuam válidos; HV_norm não é "fração da frente verdadeira". Formulação alternativa fica como trabalho futuro. | Médio (limitação assumida). |

## B. Baselines e justiça da comparação

| Objeção | Resposta | Risco residual |
|---|---|---|
| "Baselines sem ajuste e com orçamento pequeno." | E11 (§4.7, §6.12, Tabelas 11–12): NSGA-II, GRASP, Tabu e VNS de Colmenar, e o próprio MOEA/D ajustados pelo mesmo protocolo; a tempo T e 5T. Resultado depende do problema (MOEA/D à frente no GDP e a tempo igual no CDP; atrás no MDP). | Baixo. |
| "Tabu/VNS são reimplementações; não são o código deles." | Declarado (§4.7, limitações); testados com os parâmetros do irace deles (linhas "irace setting", corrigidas em 10 out) e ajustados. Não há código público para CDP/GDP. | Médio-baixo. |
| "Falta um MOEA moderno (SMS-EMOA, IBEA)." | SMS-EMOA adicionado ao protocolo E11 (§4.7, §6.12): ajustado, fica 0,11–0,37 abaixo do MOEA/D ajustado em todas as 6 células, a T e a 5T, no nível do NSGA-II. | Baixo. |
| "Tempo em Python; hardware diferente dos valores publicados." | Declarado: comparações com valores publicados são indicações (C++ vs Python); a comparação entre métodos usa o mesmo código/máquina e o tempo do MOEA/D padrão. | Baixo. |
| "Tamanho amostral n=500 (N=10–40) dá pouca potência." | Reportado em cada tabela; conclusões de n=500 qualificadas; TS 5T com N=10 no CDP-L descrito como "não significativo". | Médio (limitação assumida). |

## C. Metodologia estatística

| Objeção | Resposta | Risco residual |
|---|---|---|
| "HV depende do ponto de referência." | Frente de referência fixa e congelada (6 out); IGD+ e ε+ como robustez (Tabela 8); lição documentada (§5; referência montada só com métodos comparados invertia o sinal no CDP n=500). | Baixo. |
| "Múltiplos testes." | Holm dentro de cada família (Tabela 5 do resumo); p sem correção rotulados como exploratórios (§6.11). | Baixo. |
| "Replicatas = pseudo-replicação; só 3." | Médias por instância sobre réplicas; CV reportado (Tabela 2); instâncias são a unidade. | Baixo. |
| "Ruído entre execuções do mesmo método." | `random_ref` (2ª execução do uniforme) calibra o piso de ruído (≈1 posição de rank). | Baixo. |

## D. Seleção adaptativa de operadores (RQ2)

| Objeção | Resposta | Risco residual |
|---|---|---|
| "Seus bandits estão mal ajustados." | 17 configurações de c/janela, 4 recompensas, DYTS fiel (C = 30/100/300), contextual, 2×/4× gerações, **FRRMAB de Li et al. (2014)** (5 configurações; com C=5 vira round-robin; nenhuma difere do uniforme). | Baixo. |
| "O pool foi curado de modo que o uniforme já seja bom; AOS ajudaria com operadores ruins." | E16 "pool envenenado" (8 operadores + 3 nocivos): uniforme não perde (CDP +0,018, GDP +0,002; elitismo da substituição evita dano) e nenhum seletor melhora; E8 (3 réplicas, CDP+GDP): cada operador sozinho é pior que o pool; leave-one-out: MaxMinSwap indispensável (e SubsetCrossover/RandomSwap2×2 no GDP), os demais dispensáveis. | Baixo. |
| "Seus seletores têm algum erro de implementação?" | Achado e corrigido em 10 out: ruído de ponto flutuante (~1e-15) contava como sucesso nos seletores binários e na recompensa `insert`; variantes afetadas reexecutadas (§4.3 descreve a tolerância; regressão 70/70). Efeito: TS-HV no MDP deixou de ser "significativamente pior"; conclusões principais inalteradas. O catálogo bruto histórico não foi refeito (declarado em §6.1 e nas limitações). | Baixo (declarado). |
| "Por que a seleção ajuda em GDP apertado e MDG-b n=2000?" | E10 (dose-resposta no orçamento) + E14 (mecanismo): orçamento apertado → 34% de filhos inviáveis antes do reparo; a hipótese de "taxas de sucesso mais dispersas" foi **refutada**; o que difere é a distância da convergência (ganho do uniforme ¼→1×: 0,139 vs 0,068/0,058); correlação por instância fraca. Mecanismo estabelecido para grupos, aberto por instância. | Médio (explicitamente declarado). |

## E. Ancoragem e comparação com a literatura

| Objeção | Resposta | Risco residual |
|---|---|---|
| "Os valores publicados vêm de outro hardware/linguagem." | Declarado; razões ao melhor valor publicado, com comparação às médias de 10 s/300 s; GDP: o "publicado" é heurística de 60 s, não BKS. | Baixo. |
| "A âncora não é nova (tabu/SBTS)." | Apresentada como passo de cobertura, não como contribuição algorítmica principal; versão SBTS reimplementada a partir de Lu et al. (2023), declarada. | Baixo. |
| "MDG-b do CDP continua em 0,85–0,90 do melhor valor publicado." | Declarado como lacuna restante (§6.6, conclusões). | Médio-baixo. |

## F. Reprodutibilidade e ética

| Objeção | Resposta | Risco residual |
|---|---|---|
| "Código/dados?" | `README_REPLICATION.md`, `DATA_PACKAGE_MANIFEST.md`, `regression_check.py` (66/66 bit a bit), referência congelada, hash de configuração por run. DOI pendente. | Pendente (autores). |
| "Uso de IA generativa." | Seção de declaração (rascunho) antes das referências; **os autores precisam confirmar**. | Pendente (autores). |
| "Submissão duplicada / versão estendida de conferência." | O EJOR não considera versões estendidas de artigos de conferência: **não submeter estes resultados ao CEC 2027 antes** (ou assumir o CEC como alternativa e não o EJOR). | Decisão dos autores. |

## G. Forma

| Objeção | Resposta | Risco residual |
|---|---|---|
| "Texto longo e denso." | Detalhes de §6.4 e por família movidos ao suplemento; tabelas-resumo geradas; revisão do inglês pendente com os coautores. | Médio: ~17 mil palavras no corpo antes da movimentação; reavaliar após os resultados finais. |
