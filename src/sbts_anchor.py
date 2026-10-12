"""Busca tabu baseada em solucoes (estilo SBTS, Lu, Martinez-Gavara, Hao & Lai 2023, ESWA 223:119856) para o MaxMin do CDP/GDP.

Candidata a substituir `anchor.maxmin_anchor` como motor da ancoragem (CLAUDE.md §2l/§2r: a ancora atual fica em ~0,70 do melhor valor
publicado no MDG-b n=500). Reimplementacao nossa em NumPy (nao e o codigo dos autores), a partir da descricao do artigo:
  - solucao inicial gulosa: ponto mais distante do conjunto (semente aleatoria) ate cumprir a capacidade B;
  - vizinhanca = Add(v) U Drop(u) U CSwap(u, v), com u em X = elementos criticos (distancia ao resto == f(M)) e
    v em Y = elementos fora de M com a maior e a segunda maior distancia ao conjunto;
  - a cada iteracao vai-se para o MELHOR vizinho nao-tabu (empates ao acaso); tabu = solucao ja visitada (aqui: hash de Zobrist de 64 bits
    em um conjunto exato, em vez das tres tabelas hash aproximadas do artigo); se todos os vizinhos sao tabu, toma-se o melhor (aspiracao).
No GDP, Add e CSwap respeitam o orcamento de custo (custo(M) <= K).
O valor de f(M \\ u) e recalculado exatamente (o artigo usa uma regra rapida por contadores); isso custa O(m^2) por elemento critico.
"""
from __future__ import annotations

import numpy as np

_NEG = -np.inf


def _has_cost(inst) -> bool:
    return hasattr(inst, "cost") and hasattr(inst, "max_budget") and np.isfinite(getattr(inst, "max_budget", np.inf))


def _greedy_init(inst, rng, Dinf):
    n, cap, B = inst.n, inst.capacity, inst.min_capacity
    cost = inst.cost if _has_cost(inst) else None
    budget = inst.max_budget if cost is not None else np.inf
    v0 = int(rng.integers(n))
    M = [v0]
    inM = np.zeros(n, dtype=bool)
    inM[v0] = True
    md = Dinf[v0].copy()
    cap_s = float(cap[v0])
    cost_s = float(cost[v0]) if cost is not None else 0.0
    while cap_s < B - 1e-9 or len(M) < 2:
        ok = ~inM
        if cost is not None:
            ok &= (cost_s + cost) <= budget + 1e-9
        if not ok.any():
            return None
        v = np.where(ok, md, _NEG)
        best = v.max()
        cands = np.flatnonzero(v >= best - 1e-12)
        c = int(cands[rng.integers(len(cands))])
        M.append(c)
        inM[c] = True
        cap_s += float(cap[c])
        cost_s += float(cost[c]) if cost is not None else 0.0
        md = np.minimum(md, Dinf[c])
    return M


def _f_without(Dinf, M, u_pos):
    """f(M \\ u): minimo das distancias entre os pares restantes (O(m^2))."""
    rest = np.delete(M, u_pos)
    if rest.size < 2:
        return _NEG
    return float(Dinf[np.ix_(rest, rest)].min())


def sbts_anchor(inst, rng: np.random.Generator, iters: int = 20000, start=None, restarts: int = 1,
                stall: int = 0, ks: int = 8, max_crit: int = 4):
    """Busca SBTS-like. `iters` = iteracoes por reinicio; `stall` > 0 reinicia (solucao gulosa nova) apos `stall` iteracoes sem melhorar
    o melhor do reinicio. Retorna (solucao ordenada, (MaxSum, MaxMin)) viavel ou None se nao se aplica."""
    if not (hasattr(inst, "min_capacity") and hasattr(inst, "capacity")):
        return None
    n, cap, B = inst.n, inst.capacity.astype(float), float(inst.min_capacity)
    Dinf = inst.D.astype(float).copy()
    np.fill_diagonal(Dinf, np.inf)
    cost = inst.cost.astype(float) if _has_cost(inst) else None
    budget = float(inst.max_budget) if cost is not None else np.inf
    zob = rng.integers(1, 2**62, size=n, dtype=np.int64)
    eps = 1e-12

    best_f, best_M = _NEG, None
    for r in range(max(1, restarts)):
        if r == 0 and start is not None:
            M = [int(x) for x in start]
        else:
            M = _greedy_init(inst, rng, Dinf)
            if M is None:
                continue
        inM = np.zeros(n, dtype=bool)
        inM[M] = True
        h = 0
        for x in M:
            h ^= int(zob[x])
        visited = {h}
        cap_s = float(cap[M].sum())
        cost_s = float(cost[M].sum()) if cost is not None else 0.0
        run_best, since = _NEG, 0
        for it in range(iters):
            Ma = np.array(M, dtype=np.intp)
            md = Dinf[:, Ma].min(axis=1)                  # de(v) para v fora, di(v) para v dentro
            f = float(md[Ma].min())
            if f > run_best + eps:
                run_best, since = f, 0
                if f > best_f + eps and cap_s >= B - 1e-9 and (cost is None or cost_s <= budget + 1e-9):
                    best_f, best_M = f, list(M)
            else:
                since += 1
                if stall and since >= stall:
                    break
            # ---- Add(v)
            f_add = np.where(inM, _NEG, np.minimum(f, md))
            if cost is not None:
                f_add = np.where((cost_s + cost) <= budget + 1e-9, f_add, _NEG)
            # ---- Drop(u): so os que mantem a capacidade
            drop_ok = inM & ((cap_s - cap) >= B - 1e-9)
            if len(M) <= 2:
                drop_ok[:] = False
            f_drop = np.full(n, _NEG)
            crit = inM & (md <= f + eps)                    # elementos criticos (X)
            noncrit = drop_ok & ~crit
            f_drop[noncrit] = f                             # tirar um nao-critico nao muda f
            pos_of = {int(x): k for k, x in enumerate(M)}
            f_rest = {}
            for u in np.flatnonzero(crit):
                fr = _f_without(Dinf, Ma, pos_of[int(u)])
                f_rest[int(u)] = fr
                if drop_ok[u]:
                    f_drop[u] = fr
            # ---- CSwap(u in X, v in Y): u critico; v entre os `ks` MELHORES fora de M (maior distancia ao resto) QUE MANTEM
            # capacidade e orcamento viaveis (no CDP sem folga e no GDP o filtro de viabilidade e o que torna a troca util)
            sw_f, sw_u, sw_v = [], [], []
            out_idx = np.flatnonzero(~inM)
            if out_idx.size and len(f_rest):
                crit_u = [u for u, fr in f_rest.items() if np.isfinite(fr)]
                if len(crit_u) > max_crit:
                    crit_u = list(rng.choice(crit_u, size=max_crit, replace=False))
                for u in crit_u:
                    ok = (cap_s - cap[u] + cap[out_idx]) >= B - 1e-9
                    if cost is not None:
                        ok &= (cost_s - cost[u] + cost[out_idx]) <= budget + 1e-9
                    cand = out_idx[ok]
                    if not cand.size:
                        continue
                    rest = Ma[Ma != u]
                    de = Dinf[np.ix_(cand, rest)].min(axis=1)
                    if cand.size > ks:
                        key = de + 1e-9 * rng.random(de.size) * max(1.0, float(np.abs(de).max()))
                        top = np.argpartition(-key, ks - 1)[:ks]
                    else:
                        top = np.arange(cand.size)
                    fv = np.minimum(f_rest[u], de[top])
                    sw_f.extend(fv.tolist()); sw_v.extend(cand[top].tolist()); sw_u.extend([u] * top.size)
            # ---- escolhe o melhor vizinho nao-tabu (empates ao acaso); se todos tabu, o melhor (aspiracao)
            k_sw = len(sw_f)
            fvals = np.concatenate([f_add, f_drop, np.array(sw_f, dtype=float)]) if k_sw else np.concatenate([f_add, f_drop])
            if not np.isfinite(fvals.max()):
                break
            order = np.lexsort((rng.random(fvals.size), -fvals))
            sw_u_a = np.array(sw_u, dtype=np.intp); sw_v_a = np.array(sw_v, dtype=np.intp)
            chosen = None
            first = None
            for idx in order:
                if fvals[idx] == _NEG:
                    break
                if idx < n:
                    typ, u, v = 0, -1, int(idx)
                elif idx < 2 * n:
                    typ, u, v = 1, int(idx - n), -1
                else:
                    typ, u, v = 2, int(sw_u_a[idx - 2 * n]), int(sw_v_a[idx - 2 * n])
                hh = h ^ (int(zob[v]) if typ != 1 else 0) ^ (int(zob[u]) if typ != 0 else 0)
                if first is None:
                    first = (typ, u, v, hh)
                if hh not in visited:
                    chosen = (typ, u, v, hh)
                    break
            if chosen is None:
                chosen = first
                if chosen is None:
                    break
            typ, u, v, hh = chosen
            if typ in (1, 2):
                M.remove(u); inM[u] = False
                cap_s -= cap[u]; cost_s -= (cost[u] if cost is not None else 0.0)
            if typ in (0, 2):
                M.append(v); inM[v] = True
                cap_s += cap[v]; cost_s += (cost[v] if cost is not None else 0.0)
            h = hh
            visited.add(h)
    if best_M is None:
        return None
    sol = np.array(sorted(best_M), dtype=np.intp)
    if not inst.is_feasible(sol):
        return None
    f = inst.evaluate(sol)
    return sol, (float(f[0]), float(f[1]))
