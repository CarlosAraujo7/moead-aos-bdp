"""
mab.py — Adaptive Operator Selection via UCB1 com janela deslizante.

Implementa o seletor MAB-UCB utilizado pelo MOEA/D-AOS para escolher
dinamicamente entre os operadores do pool a cada geração.

Referências
-----------
Auer et al. (2002) — "Finite-time Analysis of the Multiarmed Bandit
    Problem". Machine Learning, 47, 235–256.
DaCosta et al. (2008) — "Adaptive operator selection with dynamic multi-armed
    bandits". GECCO 2008.

Formulação UCB
--------------
Para o operador k no instante t:

    UCB(k, t) = Q(k) + c · sqrt( log(Σ N(k)) / N(k) )

onde:
    Q(k)  = média das recompensas observadas na janela deslizante de tamanho W
    N(k)  = número de vezes que o operador k foi selecionado na janela
    c     = parâmetro de exploração (padrão 0.5)
    W     = tamanho da janela deslizante (padrão 50)

Inicialização aquecida: cada operador é selecionado uma vez antes de começar
o regime UCB, garantindo N(k) > 0 para todos os k.

Recompensa
----------
Suportados dois modos (ablação):
    'scalar'      — melhoria na função escalar de Tchebycheff do subproblema
    'hypervolume' — melhoria na contribuição ao hipervolume da frente de Pareto
                    parcial (calculada externamente e passada para update())

Em ambos os casos a recompensa é clipada em [0, ∞) (recompensa nula se não
houve melhoria, não há penalidade negativa).
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque
from typing import Deque


class UCBSelector:
    """Seletor de operadores via UCB1 com janela deslizante.

    Parameters
    ----------
    n_operators : int
        Número de operadores no pool.
    c : float
        Parâmetro de exploração UCB (quanto maior, mais exploratório).
    window_size : int
        Tamanho W da janela deslizante. Use 0 para janela infinita (estacionária).
    """

    def __init__(
        self,
        n_operators: int,
        c: float = 0.5,
        window_size: int = 50,
    ) -> None:
        if n_operators < 1:
            raise ValueError("n_operators deve ser ≥ 1")

        self.K = n_operators
        self.c = c
        self.W = window_size

        # Histórico deslizante: cada deque contém recompensas observadas
        # (uma entrada por uso do operador dentro da janela)
        self._history: list[Deque[float]] = [
            deque(maxlen=window_size if window_size > 0 else None)
            for _ in range(n_operators)
        ]

        # Contador global de seleções (para o termo log)
        self._t: int = 0

        # Fase de aquecimento: quais operadores ainda precisam de 1ª seleção
        self._warmup_queue: list[int] = list(range(n_operators))

    # ------------------------------------------------------------------
    # Interface principal
    # ------------------------------------------------------------------

    def select(self, rng: np.random.Generator) -> int:
        """Retorna o índice do operador selecionado.

        Durante o aquecimento, os operadores são retornados em ordem para
        garantir pelo menos uma observação de cada um antes de começar UCB.
        """
        if self._warmup_queue:
            return self._warmup_queue.pop(0)

        scores = self._ucb_scores()
        # Desempate aleatório entre os melhores
        max_score = scores.max()
        best = np.where(np.isclose(scores, max_score))[0]
        return int(rng.choice(best))

    def update(self, operator_idx: int, reward: float) -> None:
        """Registra a recompensa obtida pelo operador `operator_idx`.

        Parameters
        ----------
        operator_idx : int
            Índice do operador que foi aplicado.
        reward : float
            Recompensa (≥ 0). Valores negativos são clipados para 0.
        """
        r = max(0.0, float(reward))
        self._history[operator_idx].append(r)
        self._t += 1

    # ------------------------------------------------------------------
    # Estatísticas
    # ------------------------------------------------------------------

    def q_values(self) -> np.ndarray:
        """Retorna Q(k) = média da janela para cada operador."""
        q = np.zeros(self.K)
        for k, hist in enumerate(self._history):
            if hist:
                q[k] = float(np.mean(hist))
        return q

    def counts(self) -> np.ndarray:
        """Retorna N(k) = número de observações na janela para cada operador."""
        return np.array([len(h) for h in self._history], dtype=float)

    def ucb_scores(self) -> np.ndarray:
        """Expõe os scores UCB publicamente (para logging e análise)."""
        return self._ucb_scores()

    def selection_probabilities(self) -> np.ndarray:
        """Frequência relativa de seleção na janela atual (para diagnóstico)."""
        n = self.counts()
        total = n.sum()
        if total == 0:
            return np.ones(self.K) / self.K
        return n / total

    def reset(self) -> None:
        """Reinicia o seletor (útil entre execuções independentes)."""
        for h in self._history:
            h.clear()
        self._t = 0
        self._warmup_queue = list(range(self.K))

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    def _ucb_scores(self) -> np.ndarray:
        n = self.counts()
        q = self.q_values()
        total = n.sum()

        scores = np.full(self.K, np.inf)
        for k in range(self.K):
            if n[k] > 0 and total > 0:
                exploration = self.c * math.sqrt(math.log(total) / n[k])
                scores[k] = q[k] + exploration
        return scores

    # ------------------------------------------------------------------
    # Representação
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UCBSelector(K={self.K}, c={self.c}, W={self.W}, "
            f"t={self._t}, warmup={'sim' if self._warmup_queue else 'nao'})"
        )


# ---------------------------------------------------------------------------
# Baseline: seleção uniforme aleatória (sem aprendizado)
# ---------------------------------------------------------------------------

class RandomSelector:
    """Seleciona operadores de forma uniforme aleatória — baseline sem AOS.

    Mantém a mesma interface do UCBSelector para uso intercambiável no
    MOEAD_AOS, mas ignora completamente as recompensas recebidas.

    Parameters
    ----------
    n_operators : int
        Número de operadores no pool.
    """

    def __init__(self, n_operators: int) -> None:
        self.K = n_operators
        self._counts = np.zeros(n_operators, dtype=float)

    def select(self, rng: np.random.Generator) -> int:
        k = int(rng.integers(self.K))
        self._counts[k] += 1
        return k

    def update(self, operator_idx: int, reward: float) -> None:
        """Recebe recompensa mas não faz nada com ela (baseline)."""
        pass

    def q_values(self) -> np.ndarray:
        return np.zeros(self.K)

    def counts(self) -> np.ndarray:
        return self._counts.copy()

    def ucb_scores(self) -> np.ndarray:
        return np.ones(self.K) / self.K

    def selection_probabilities(self) -> np.ndarray:
        total = self._counts.sum()
        if total == 0:
            return np.ones(self.K) / self.K
        return self._counts / total

    def reset(self) -> None:
        self._counts[:] = 0

    def __repr__(self) -> str:
        return f"RandomSelector(K={self.K})"



class FixedSelector(RandomSelector):
    """Sempre o mesmo operador (ablacao de operador unico; nao e um metodo do catalogo)."""

    def __init__(self, n_operators: int, fixed: int) -> None:
        super().__init__(n_operators)
        if not 0 <= fixed < n_operators:
            raise ValueError(f"operador fixo {fixed} fora de [0,{n_operators})")
        self.fixed = fixed

    def select(self, rng: np.random.Generator) -> int:
        self._counts[self.fixed] += 1
        return self.fixed

    def __repr__(self) -> str:
        return f"FixedSelector(K={self.K}, fixed={self.fixed})"


class SubsetRandomSelector(RandomSelector):
    """Selecao uniforme restrita a um subconjunto de operadores (ablacao leave-one-out)."""

    def __init__(self, n_operators: int, excluded) -> None:
        super().__init__(n_operators)
        self.allowed = [k for k in range(n_operators) if k not in set(excluded)]
        if not self.allowed:
            raise ValueError("todos os operadores foram excluidos")

    def select(self, rng: np.random.Generator) -> int:
        k = self.allowed[int(rng.integers(len(self.allowed)))]
        self._counts[k] += 1
        return k

    def __repr__(self) -> str:
        return f"SubsetRandomSelector(K={self.K}, allowed={self.allowed})"


# ---------------------------------------------------------------------------
# Variante proposta: Thompson Sampling com janela deslizante
# ---------------------------------------------------------------------------

class ThompsonSelector:
    """Selecao de operadores via Thompson Sampling binario com janela deslizante.

    Cada operador k e modelado por uma distribuicao Beta(alpha_k, beta_k),
    onde alpha_k = numero de recompensas positivas e beta_k = numero de
    recompensas nulas na janela de tamanho W. A selecao amostra de cada
    distribuicao Beta e escolhe o operador com maior amostra.

    Vantagem em relacao ao UCB: o balanco exploracao/explotacao e adaptativo
    e probabilistico, sem hiperparametro c a ajustar.

    References
    ----------
    Thompson, W.R. (1933). On the likelihood that one unknown probability
        exceeds another in view of the evidence of two samples. Biometrika,
        25(3-4), 285-294.
    Chapelle, O., Li, L. (2011). An empirical evaluation of Thompson Sampling.
        NeurIPS 2011.
    """

    def __init__(
        self,
        n_operators: int,
        window_size: int = 50,
    ) -> None:
        if n_operators < 1:
            raise ValueError("n_operators deve ser >= 1")
        self.K = n_operators
        self.W = window_size

        # Janela deslizante: 1 = recompensa positiva, 0 = recompensa nula
        from collections import deque
        self._history: list[deque] = [
            deque(maxlen=window_size if window_size > 0 else None)
            for _ in range(n_operators)
        ]
        self._t: int = 0
        self._warmup_queue: list[int] = list(range(n_operators))

    def select(self, rng: np.random.Generator) -> int:
        if self._warmup_queue:
            return self._warmup_queue.pop(0)
        # Amostra de Beta(alpha_k + 1, beta_k + 1) para cada operador
        samples = np.zeros(self.K)
        for k, hist in enumerate(self._history):
            alpha_k = sum(hist) + 1.0      # sucessos + prior
            beta_k = len(hist) - sum(hist) + 1.0  # fracassos + prior
            samples[k] = float(rng.beta(alpha_k, beta_k))
        return int(samples.argmax())

    def update(self, operator_idx: int, reward: float) -> None:
        binary = 1.0 if reward > 0.0 else 0.0
        self._history[operator_idx].append(binary)
        self._t += 1

    def q_values(self) -> np.ndarray:
        q = np.zeros(self.K)
        for k, hist in enumerate(self._history):
            if hist:
                q[k] = float(np.mean(list(hist)))
        return q

    def counts(self) -> np.ndarray:
        return np.array([len(h) for h in self._history], dtype=float)

    def ucb_scores(self) -> np.ndarray:
        return self.q_values()

    def selection_probabilities(self) -> np.ndarray:
        n = self.counts()
        total = n.sum()
        if total == 0:
            return np.ones(self.K) / self.K
        return n / total

    def reset(self) -> None:
        for h in self._history:
            h.clear()
        self._t = 0
        self._warmup_queue = list(range(self.K))

    def __repr__(self) -> str:
        return (f"ThompsonSelector(K={self.K}, W={self.W}, "
                f"t={self._t}, warmup={'sim' if self._warmup_queue else 'nao'})")


# ---------------------------------------------------------------------------
# DYTS — Dynamic Thompson Sampling (Sun & Li, 2020)
# ---------------------------------------------------------------------------

class DYTSSelector:
    """Adaptive Operator Selection via Dynamic Thompson Sampling (DYTS).

    Cada operador k e modelado por uma distribuicao Normal com media
    desconhecida. Usa um prior Normal-inverse-chi-squared conjugado
    atualizado com recompensas continuas dentro de uma janela deslizante.

    Ao contrario do ThompsonSelector binario, este seletor usa a magnitude
    real da recompensa.

    ATENCAO (conferido contra o artigo em 3 out 2026): apesar do nome `dyts`, este
    seletor NAO e o DYTS de Sun & Li (2020), que usa posteriores Beta, recompensa
    binaria e um limiar de decaimento C (ver `DYTSBetaSelector`, fiel ao artigo). Este
    e um amostrador gaussiano de janela deslizante; nos artigos/tabelas chama-se GTS.
    O nome `dyts` foi mantido nos modos/arquivos para nao invalidar o catalogo.

    Referencias
    -----------
    Sun, L., Li, K. (2020). Adaptive Operator Selection Based on Dynamic
        Thompson Sampling for MOEA/D. PPSN 2020 (arXiv:2004.10874) -- NAO implementado aqui.
    """

    def __init__(
        self,
        n_operators: int,
        window_size: int = 50,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        if n_operators < 1:
            raise ValueError("n_operators deve ser >= 1")
        self.K = n_operators
        self.W = window_size

        # Prior Normal-inverse-chi-squared: (mu0, kappa0, alpha0, beta0)
        self._mu0    = mu0
        self._kappa0 = kappa0
        self._alpha0 = alpha0
        self._beta0  = beta0

        from collections import deque
        self._history: list[deque] = [
            deque(maxlen=window_size if window_size > 0 else None)
            for _ in range(n_operators)
        ]
        self._t: int = 0
        self._warmup_queue: list[int] = list(range(n_operators))

    # ------------------------------------------------------------------
    # Interface principal
    # ------------------------------------------------------------------

    def select(self, rng: np.random.Generator) -> int:
        if self._warmup_queue:
            return self._warmup_queue.pop(0)

        samples = np.zeros(self.K)
        for k in range(self.K):
            mu_n, kappa_n, alpha_n, beta_n = self._posterior(k)
            # Marginal de mu e uma Student-t com 2*alpha_n graus de liberdade
            # Parametrizacao: mu_n + t * sqrt(beta_n / (alpha_n * kappa_n))
            dof   = max(2.0 * alpha_n, 1e-6)
            scale = math.sqrt(float(beta_n) / max(float(alpha_n * kappa_n), 1e-12))
            try:
                t_sample = float(rng.standard_t(dof))
            except Exception:
                t_sample = 0.0
            samples[k] = mu_n + t_sample * scale

        return int(np.argmax(samples))

    def update(self, operator_idx: int, reward: float) -> None:
        """Registra recompensa continua para o operador operator_idx."""
        self._history[operator_idx].append(float(reward))
        self._t += 1

    # ------------------------------------------------------------------
    # Posterior Normal-inverse-chi-squared
    # ------------------------------------------------------------------

    def _posterior(self, k: int) -> tuple[float, float, float, float]:
        """Calcula os parametros posteriores para o operador k.

        Dado n observacoes x_1, ..., x_n com media x_bar e soma de quadrados S:
            mu_n    = (kappa0 * mu0 + n * x_bar) / (kappa0 + n)
            kappa_n = kappa0 + n
            alpha_n = alpha0 + n / 2
            beta_n  = beta0 + S/2 + (kappa0*n*(x_bar - mu0)^2) / (2*(kappa0+n))
        """
        obs = list(self._history[k])
        n   = len(obs)

        if n == 0:
            return (self._mu0, self._kappa0, self._alpha0, self._beta0)

        mu0    = self._mu0
        kappa0 = self._kappa0
        alpha0 = self._alpha0
        beta0  = self._beta0

        x_arr  = np.array(obs, dtype=float)
        x_bar  = float(x_arr.mean())
        S      = float(((x_arr - x_bar) ** 2).sum())

        kappa_n = kappa0 + n
        mu_n    = (kappa0 * mu0 + n * x_bar) / kappa_n
        alpha_n = alpha0 + n / 2.0
        beta_n  = (beta0 + S / 2.0
                   + (kappa0 * n * (x_bar - mu0) ** 2) / (2.0 * kappa_n))

        return (mu_n, kappa_n, alpha_n, beta_n)

    # ------------------------------------------------------------------
    # Interface de consulta (compativel com UCBSelector / RandomSelector)
    # ------------------------------------------------------------------

    def q_values(self) -> np.ndarray:
        """Retorna a media posterior (mu_n) de cada operador como Q-value."""
        return np.array([self._posterior(k)[0] for k in range(self.K)])

    def counts(self) -> np.ndarray:
        """Retorna o numero de observacoes na janela de cada operador."""
        return np.array([len(self._history[k]) for k in range(self.K)], dtype=float)


# ---------------------------------------------------------------------------
# Funcoes de recompensa (usadas por moead_aos.py)
# ---------------------------------------------------------------------------

class DYTSBetaSelector:
    """Dynamic Thompson sampling EXATAMENTE como em Sun & Li (2020), Algoritmos 1-2 (arXiv:2004.10874).

    Cada operador k tem uma Beta(alpha_k, beta_k), ambos iniciados em 1; a selecao amostra uma Beta por operador e escolhe o maior
    valor (nao ha aquecimento). Recompensa BINARIA r em {0,1}. Atualizacao (Alg. 1): se alpha+beta < C, atualizacao classica
    (alpha += r; beta += 1 - r); senao, ambos sao reescalados por C/(C+1) apos a atualizacao, o que da peso geometrico decrescente
    as recompensas antigas (nao-estacionariedade). C = 100 no artigo original.

    NAO confundir com `DYTSSelector` deste modulo, que (apesar do nome) e um amostrador gaussiano de janela deslizante
    (prior normal-inversa-qui-quadrado, recompensas continuas) e NAO e o DYTS de Sun & Li.
    """

    def __init__(self, n_operators: int, C: float = 100.0) -> None:
        if n_operators < 1:
            raise ValueError("n_operators deve ser >= 1")
        self.K = n_operators
        self.C = float(C)
        self.alpha = np.ones(n_operators)
        self.beta = np.ones(n_operators)

    def select(self, rng: np.random.Generator) -> int:
        return int(np.argmax(rng.beta(self.alpha, self.beta)))

    def update(self, operator_idx: int, reward: float) -> None:
        r = 1.0 if reward > 0.0 else 0.0
        k = operator_idx
        if self.alpha[k] + self.beta[k] < self.C:
            self.alpha[k] += r
            self.beta[k] += 1.0 - r
        else:
            f = self.C / (self.C + 1.0)
            self.alpha[k] = (self.alpha[k] + r) * f
            self.beta[k] = (self.beta[k] + 1.0 - r) * f

    def q_values(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)


class FRRMABSelector:
    """FRRMAB de Li, Fialho, Kwong & Zhang (2014, IEEE TEVC 18(1):114-130), Algoritmos 1-2.

    Janela deslizante FIFO de tamanho W com pares (operador, FIR); Reward_i = soma dos FIR do operador i na janela; os Reward sao
    ordenados em ordem decrescente e Decay_i = D**Rank_i * Reward_i; FRR_i = Decay_i / sum_j Decay_j; selecao UCB
    argmax_i FRR_i + C * sqrt(2 ln(sum_j n_j) / n_i), com n_i = numero de aplicacoes do operador i NA JANELA. Enquanto algum operador
    nunca foi aplicado, escolhe-se uniformemente entre os nunca aplicados. Padroes do artigo: W = 0,5 N, C = 5, D = 1.
    O FIR (fitness improvement rate) de um filho e a soma, sobre os subproblemas cuja solucao ele substitui, de (g_antigo - g_novo)/g_antigo
    (calculado pelo MOEA/D com reward_mode='frr').
    """

    def __init__(self, n_operators: int, window: int, C: float = 5.0, D: float = 1.0) -> None:
        if n_operators < 1 or window < 1:
            raise ValueError("n_operators e window devem ser >= 1")
        self.K = n_operators
        self.W = int(window)
        self.C = float(C)
        self.D = float(D)
        self._ops: list[int] = []
        self._fir: list[float] = []
        self._applied = np.zeros(n_operators, dtype=bool)

    def _frr(self) -> tuple[np.ndarray, np.ndarray]:
        reward = np.zeros(self.K)
        n = np.zeros(self.K)
        for o, f in zip(self._ops, self._fir):
            reward[o] += f
            n[o] += 1.0
        order = np.argsort(-reward, kind="stable")
        rank = np.empty(self.K)
        rank[order] = np.arange(self.K)
        decay = (self.D ** rank) * reward
        s = decay.sum()
        return (decay / s if s > 0.0 else np.zeros(self.K)), n

    def select(self, rng: np.random.Generator) -> int:
        if not self._applied.all():
            cand = np.flatnonzero(~self._applied)
            return int(rng.choice(cand))
        frr, n = self._frr()
        total = n.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = np.where(n > 0, self.C * np.sqrt(2.0 * np.log(max(total, 1.0)) / np.maximum(n, 1.0)), np.inf)
        score = frr + bonus
        best = np.flatnonzero(score >= score.max() - 1e-12)
        return int(rng.choice(best)) if len(best) > 1 else int(best[0])

    def update(self, operator_idx: int, reward: float) -> None:
        self._applied[operator_idx] = True
        self._ops.append(int(operator_idx))
        self._fir.append(float(reward))
        if len(self._ops) > self.W:
            self._ops.pop(0)
            self._fir.pop(0)

    def q_values(self) -> np.ndarray:
        return self._frr()[0]

    def counts(self) -> np.ndarray:
        return self._frr()[1]


class ContextualSelector:
    """Seletor contextual (E3): um seletor independente por contexto; `set_context(c)` escolhe qual esta ativo.
    O MOEA/D define o contexto antes de cada selecao/atualizacao (regiao do vetor de pesos do subproblema e/ou fase da busca).
    Nenhuma informacao e compartilhada entre contextos (ver CLAUDE.md §2r/E3)."""

    def __init__(self, factory, n_contexts: int) -> None:
        self.sels = [factory() for _ in range(n_contexts)]
        self.K = self.sels[0].K
        self.ctx = 0

    def set_context(self, c: int) -> None:
        self.ctx = int(c)

    def select(self, rng: np.random.Generator) -> int:
        return self.sels[self.ctx].select(rng)

    def update(self, operator_idx: int, reward: float) -> None:
        self.sels[self.ctx].update(operator_idx, reward)

    def q_values(self) -> np.ndarray:
        return np.mean([s.q_values() for s in self.sels], axis=0)


def scalar_reward(old_g: float, new_g: float) -> float:
    """Recompensa baseada em melhoria escalar Tchebycheff.

    old_g, new_g sao valores da funcao de Tchebycheff (minimizacao, >0).
    Retorna a melhoria relativa clipada em [0, inf).
    """
    improvement = old_g - new_g           # positivo = melhoria
    return max(0.0, float(improvement))


def hypervolume_reward(old_hv: float, new_hv: float) -> float:
    """Recompensa baseada em ganho de hipervolume.

    Retorna o ganho absoluto de HV clipado em [0, inf).
    """
    return max(0.0, float(new_hv - old_hv))
