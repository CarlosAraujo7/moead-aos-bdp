"""
metrics.py — Indicadores de qualidade para comparação de fronteiras de Pareto.

Implementa três indicadores padrão da literatura multiobjetivo:

    1. Hypervolume (HV, 2D exato)
       Área dominada pela frente de Pareto em relação a um ponto de referência.
       Algoritmo: ordena por f1 decrescente; varredura linear O(|F| log |F|).

    2. C-metric / Set Coverage (C(A,B))
       Fração das soluções de B dominadas (fracamente) por alguma solução de A.
       Valor ∈ [0,1]. C(A,B) > C(B,A) indica que A é melhor.

    3. Epsilon Indicator Aditivo (I_ε+)
       Menor valor ε tal que cada solução de B tem, em A, uma solução ε-dominante.
       Valor menor indica melhor frente.

Convenção
---------
- Todas as funções recebem arrays de shape (|F|, 2) onde cada linha é (f1, f2).
- Problema de maximização: f1 = MaxSum, f2 = MaxMin.
- Dominância: a domina b ↔ a[0] ≥ b[0] e a[1] ≥ b[1] com pelo menos uma estrita.

Referências
-----------
Zitzler et al. (2003) — "Performance Assessment of Multiobjective Optimizers:
    An Analysis and Review". IEEE TEC, 7(2), 117–132.
While et al. (2006) — "A faster algorithm for calculating hypervolume".
    IEEE TEC, 10(1), 29–38.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def _to_array(front: list | np.ndarray) -> np.ndarray:
    """Converte para np.ndarray shape (N, 2) com dtype float64."""
    arr = np.asarray(front, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Esperado shape (N, 2); recebido {arr.shape}")
    return arr


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Filtra e retorna apenas as soluções não dominadas (fronteira de Pareto).

    Parameters
    ----------
    points : np.ndarray shape (N, 2)
        Conjunto de soluções; cada linha é (f1, f2). Maximização.

    Returns
    -------
    np.ndarray shape (k, 2)
        Subconjunto não dominado de `points`, ordenado por f1 decrescente.
    """
    pts = _to_array(points)
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # j domina i?
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    dominated[i] = True
                    break

    non_dom = pts[~dominated]
    # Ordena por f1 decrescente
    order = np.argsort(-non_dom[:, 0])
    return non_dom[order]


# ---------------------------------------------------------------------------
# 1. Hypervolume (2D exato)
# ---------------------------------------------------------------------------

def hypervolume(
    front: list | np.ndarray,
    reference: tuple[float, float] | np.ndarray,
) -> float:
    """Calcula o hipervolume 2D exato dominado pela frente em relação ao ponto
    de referência.

    Parameters
    ----------
    front : array-like shape (N, 2)
        Frente de Pareto (não precisa ser filtrada previamente; a função
        aplica filtragem interna).
    reference : (ref_f1, ref_f2)
        Ponto de referência — deve ser dominado por todas as soluções na frente.
        Tipicamente definido como (worst_f1 − δ, worst_f2 − δ).

    Returns
    -------
    float
        Valor do hipervolume.

    Algorithm
    ---------
    Após filtrar e ordenar a frente por f1 decrescente:
        HV = Σ_i (f1[i] − f1[i+1]) * (f2[i] − ref_f2)
    onde f1[k] = ref_f1 no último passo fictício.
    """
    ref = np.asarray(reference, dtype=np.float64)
    if ref.shape != (2,):
        raise ValueError("reference deve ter shape (2,)")

    pts = _to_array(front)
    # Remove soluções que não dominam o ponto de referência
    valid = (pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    # Filtra não dominadas e ordena por f1 decrescente
    pf = pareto_front(pts)

    hv = 0.0
    for i in range(len(pf)):
        f1_i = pf[i, 0]
        f2_i = pf[i, 1]
        f1_next = ref[0] if i + 1 == len(pf) else pf[i + 1, 0]
        hv += (f1_i - f1_next) * (f2_i - ref[1])

    return float(hv)


def hypervolume_contribution(
    point: tuple[float, float] | np.ndarray,
    front: list | np.ndarray,
    reference: tuple[float, float] | np.ndarray,
) -> float:
    """Contribuição incremental ao hipervolume ao adicionar `point` à `front`.

    Retorna HV(front ∪ {point}) − HV(front).
    Valor ≥ 0; zero se `point` for dominado pela frente atual.
    """
    ref = np.asarray(reference, dtype=np.float64)
    pts = _to_array(front)

    hv_before = hypervolume(pts, ref)

    p = np.asarray(point, dtype=np.float64).reshape(1, 2)
    pts_augmented = np.vstack([pts, p])
    hv_after = hypervolume(pts_augmented, ref)

    return max(0.0, hv_after - hv_before)


# ---------------------------------------------------------------------------
# 2. C-metric (Set Coverage)
# ---------------------------------------------------------------------------

def set_coverage(
    front_a: list | np.ndarray,
    front_b: list | np.ndarray,
) -> float:
    """C(A, B) — fração das soluções de B fracamente dominadas por A.

    Uma solução b ∈ B é fracamente dominada por A se ∃ a ∈ A tal que
    a[0] ≥ b[0] e a[1] ≥ b[1].

    Parameters
    ----------
    front_a, front_b : array-like shape (N, 2)
        Duas frentes de Pareto a comparar.

    Returns
    -------
    float ∈ [0, 1]
        Proporção de soluções de B cobertas por A.
    """
    A = _to_array(front_a)
    B = _to_array(front_b)

    if len(B) == 0:
        return 1.0

    covered = 0
    for b in B:
        # b é coberta se algum a ∈ A a domina fracamente
        if np.any((A[:, 0] >= b[0]) & (A[:, 1] >= b[1])):
            covered += 1

    return covered / len(B)


# ---------------------------------------------------------------------------
# 3. Epsilon Indicator Aditivo
# ---------------------------------------------------------------------------

def epsilon_indicator(
    front_a: list | np.ndarray,
    front_b: list | np.ndarray,
) -> float:
    """I_ε+(A, B) — indicador epsilon aditivo de A em relação a B.

    Retorna o menor ε ≥ 0 tal que para todo b ∈ B existe a ∈ A com:
        a[k] + ε ≥ b[k]  para k = 0, 1  (maximização)

    Interpretação: ε próximo de 0 indica que A aproxima B bem.
    Se A = B (frente ótima conhecida), então I_ε+(A, B) = 0 é o ideal.
    Para comparar dois algoritmos, o que produz menor I_ε+ em relação à
    referência é melhor.

    Parameters
    ----------
    front_a : array-like shape (N, 2) — frente produzida pelo algoritmo
    front_b : array-like shape (M, 2) — frente de referência

    Returns
    -------
    float
        Valor do indicador ε+ (menor é melhor).
    """
    A = _to_array(front_a)
    B = _to_array(front_b)

    eps_per_b = []
    for b in B:
        # Para cada b, encontra o menor ε tal que ∃ a : a + ε ≥ b
        # ε = max_k (b[k] − a[k]) para cada a; depois min sobre a ∈ A
        diffs = np.max(b - A, axis=1)   # shape (|A|,)
        eps_per_b.append(float(diffs.min()))

    return float(max(eps_per_b))


# ---------------------------------------------------------------------------
# Função de conveniência: calcula todos os indicadores de uma vez
# ---------------------------------------------------------------------------

def all_indicators(
    front_approx: list | np.ndarray,
    front_ref: list | np.ndarray,
    reference_point: tuple[float, float],
) -> dict[str, float]:
    """Calcula HV, C(approx, ref), C(ref, approx) e I_ε+.

    Parameters
    ----------
    front_approx : frente produzida pelo algoritmo em teste
    front_ref    : frente de referência (e.g. melhor frente conhecida)
    reference_point : ponto de referência para o cálculo do HV

    Returns
    -------
    dict com chaves: 'hv', 'coverage_ab', 'coverage_ba', 'epsilon'
    """
    hv_approx = hypervolume(front_approx, reference_point)
    hv_ref = hypervolume(front_ref, reference_point)
    cov_ab = set_coverage(front_approx, front_ref)
    cov_ba = set_coverage(front_ref, front_approx)
    eps = epsilon_indicator(front_approx, front_ref)

    return {
        "hv_approx": hv_approx,
        "hv_ref": hv_ref,
        "coverage_approx_over_ref": cov_ab,
        "coverage_ref_over_approx": cov_ba,
        "epsilon": eps,
    }


# ---------------------------------------------------------------------------
# Teste básico
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Frente A: melhor
    A = np.array([[5.0, 3.0], [4.0, 4.0], [3.0, 5.0]])
    # Frente B: ligeiramente pior
    B = np.array([[4.5, 2.5], [3.5, 3.5], [2.5, 4.5]])
    ref = np.array([0.0, 0.0])

    print("=== Hypervolume ===")
    print(f"  HV(A) = {hypervolume(A, ref):.4f}")
    print(f"  HV(B) = {hypervolume(B, ref):.4f}")

    print("\n=== Set Coverage ===")
    print(f"  C(A, B) = {set_coverage(A, B):.4f}  (esperado: 1.0 — A cobre tudo em B)")
    print(f"  C(B, A) = {set_coverage(B, A):.4f}  (esperado: 0.0 — B não cobre nada em A)")

    print("\n=== Epsilon Indicator ===")
    print(f"  I_ε+(A, B) = {epsilon_indicator(A, B):.4f}  (A ≈ referência — deve ser pequeno)")
    print(f"  I_ε+(B, A) = {epsilon_indicator(B, A):.4f}  (B pior que A)")

    print("\n=== Contribuição ao HV ===")
    existing = np.array([[5.0, 3.0], [3.0, 5.0]])
    new_pt = np.array([4.0, 4.0])
    contrib = hypervolume_contribution(new_pt, existing, ref)
    print(f"  Contribuição de (4,4) à frente {{(5,3),(3,5)}}: {contrib:.4f}")

    print("\n=== Pareto Front ===")
    mixed = np.array([[5.0, 3.0], [4.0, 4.0], [3.0, 5.0], [2.0, 2.0], [4.5, 3.5]])
    pf = pareto_front(mixed)
    print(f"  Frente não dominada de {len(mixed)} pontos → {len(pf)} pontos:")
    for p in pf:
        print(f"    {p}")
