"""
bo_trajectory.py -- Tabu Search e VNS bi-objetivo no estilo de Colmenar, Marti & Duarte (2018)
("Heuristics for the bi-objective diversity problem", Expert Syst. Appl. 108:193-205), para MDP, CDP e GDP.

Por que existe (E11): no estudo deles o melhor metodo para o BODP foi uma busca tabu, e todos os metodos foram ajustados com irace.
Os baselines do projeto (nsga2.py, grasp_mo.py, vns.py) nao sao esses; este modulo acrescenta o TS e um VNS no mesmo esquema, para que
o MOEA/D seja comparado tambem com o melhor competidor da literatura do problema bi-objetivo (e com versoes AJUSTADAS dos baselines).

Esquema (Duarte et al. 2015, usado por Colmenar et al.): a APROXIMACAO DA FRENTE (arquivo ND) e a "solucao incumbente". Uma iteracao
explora a vizinhanca da solucao corrente S com uma busca local que alimenta o arquivo; se o arquivo nao mudou, nao houve melhora:
S passa a ser uma solucao nao-tabu sorteada do arquivo e o contador de iteracoes sem melhora sobe.

  * Construtivo (GRASP/RCL): 'altbws' alterna o objetivo (MaxMin <-> MaxSum) ENTRE construcoes; 'altins' alterna a cada elemento
    inserido; 'rnd' sorteia. alpha no sentido deles: g >= gmin + alpha*(gmax-gmin) (alpha=1 -> guloso, alpha=0 -> aleatorio).
  * Busca local 'pr' (LSPR): S1 = hill-climb de MaxMin a partir de S, S2 = hill-climb de MaxSum a partir de S, depois path relinking
    S1 -> S2 (passo aleatorio); todas as solucoes percorridas alimentam o arquivo. 'alt' (LSAlt) alterna o objetivo a cada chamada;
    'dom' (LSDom) aceita so trocas que dominam a corrente. Primeira melhora, troca 1-por-1 (avaliada em lote por elemento removido).
  * TS: lista tabu = hashes (Zobrist, exatos) de solucoes percorridas/escolhidas, ativos por `tenure` iteracoes; vizinhos tabu sao
    pulados. VNS: shake troca k elementos aleatorios; k volta a 1 quando o arquivo muda.

Desvios (documentados no manuscrito):
  * Criterio de parada: com `time_limit_s` o laco roda ate o limite; a estagnacao (`max_no_improve` iteracoes no TS, `restart_after`
    ciclos completos de k no VNS) dispara uma NOVA CONSTRUCAO em vez de encerrar (o original termina ao estagnar; aqui o orcamento e tempo).
  * "Melhora" = o arquivo recebeu alguma insercao (o original compara |ND|, que ignora substituicoes).
  * CDP/GDP (tamanho livre): trocas que violam capacidade/orcamento sao descartadas (mascara vetorizada); a busca local de MaxMin
    tambem aceita REMOVER um elemento quando a capacidade ainda e satisfeita e o MaxMin melhora estritamente, e no fim poda elementos
    redundantes (remover nunca reduz o MaxMin) inserindo ambos os pontos; o path relinking repara intermediarios inviaveis (inst.repair).
  * Nao ha movimento de insercao (add) no MaxSum: no CDP o MaxSum e maximo no conjunto inteiro, o que nao e o regime de interesse.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vns import _update_archive


@dataclass
class BOTrajConfig:
    method: Literal["tabu", "vns"] = "tabu"
    alpha: float = 0.94                                   # RCL: 1 = guloso, 0 = aleatorio (convencao de Colmenar et al.)
    constructive: Literal["altbws", "altins", "rnd"] = "altins"
    local_search: Literal["pr", "alt", "dom"] = "pr"
    tenure: int = 64                                      # TS: iteracoes que um hash permanece tabu
    max_no_improve: int = 70                              # TS: iteracoes sem melhora ate nova construcao
    kmax: int = 29                                        # VNS: tamanho maximo do shake
    # Valores RELATIVOS (como em Colmenar et al. 2018, que ajustam T e kmax em PORCENTAGEM): se definidos, tem precedencia sobre
    # tenure/kmax absolutos. tenure_frac: fracao do numero de elementos n (leitura nossa: o artigo diz apenas que o tamanho da lista tabu
    # foi configurado como porcentagem); kmax_frac: fracao do tamanho da solucao corrente.
    tenure_frac: float | None = None
    kmax_frac: float | None = None
    restart_after: int = 7                                # VNS: ciclos completos de k sem melhora ate nova construcao
    max_iter: int = 10**9                                 # teto de seguranca (iteracoes externas)
    # CDP/GDP: a busca local de MaxSum aceita INSERIR elemento que nao reduz o MaxMin (e cabe no orcamento). PADRAO False: com True o
    # tamanho da solucao cresce sem limite no CDP (MaxSum e monotono no tamanho) e o MaxSum chega a ~6x o da referencia (sonda de 7 out);
    # o estudo principal e o E11 comparam todos os metodos no regime de tamanho dirigido pela capacidade (ver manuscrito, Limitations).
    add_moves: bool = False
    time_limit_s: float | None = None
    seed: int | None = None


@dataclass
class BOTrajResult:
    pareto_front: np.ndarray
    n_iter_executed: int
    config: BOTrajConfig


class BOTrajectory:
    """TS / VNS bi-objetivo com a aproximacao da frente como solucao incumbente."""

    def __init__(self, instance, config: BOTrajConfig | None = None) -> None:
        self.inst = instance
        self.cfg = config or BOTrajConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.D = instance.D
        self.n = instance.n
        self.fixed = hasattr(instance, "m")                 # MDP: tamanho fixo, sem restricoes
        self.m = getattr(instance, "m", None)
        if not self.fixed:
            self.cap = np.asarray(instance.capacity, dtype=float)
            self.B = float(instance.min_capacity)
            self.cost = getattr(instance, "cost", None)
            self.K = float(instance.max_budget) if self.cost is not None else None
        self.zob = np.random.default_rng(987654321).integers(1, 2**62, size=self.n, dtype=np.uint64)
        self._t0 = 0.0
        self._alt_obj = 0
        self._cons_obj = 0
        # estado da busca
        self.arch_s: list[np.ndarray] = []
        self.arch_f: list[tuple] = []
        self.n_ins = 0
        self.tabu: dict[int, int] = {}
        self.it = 0

    # ------------------------------------------------------------------ utilidades
    def _timeout(self) -> bool:
        tl = self.cfg.time_limit_s
        return tl is not None and (_time.perf_counter() - self._t0) >= tl

    def _hash(self, S: np.ndarray) -> int:
        return int(np.bitwise_xor.reduce(self.zob[S]))

    def _tabu_hashes(self) -> np.ndarray:
        act = [h for h, e in self.tabu.items() if e > self.it]
        return np.array(act, dtype=np.uint64) if act else np.empty(0, dtype=np.uint64)

    def _add_tabu(self, S: np.ndarray) -> None:
        ten = round(self.cfg.tenure_frac * self.n) if self.cfg.tenure_frac else self.cfg.tenure
        self.tabu[self._hash(S)] = self.it + max(1, int(ten))

    def _record(self, S: np.ndarray, f: tuple) -> bool:
        ins = _update_archive(self.arch_s, self.arch_f, S, f)
        if ins:
            self.n_ins += 1
        return ins

    def _sums(self, S: np.ndarray) -> tuple[float, float]:
        capS = float(self.cap[S].sum()) if not self.fixed else 0.0
        costS = float(self.cost[S].sum()) if (not self.fixed and self.cost is not None) else 0.0
        return capS, costS

    # ------------------------------------------------------------------ construtivo
    def _construct(self) -> np.ndarray:
        cfg, rng, D, n = self.cfg, self.rng, self.D, self.n
        inS = np.zeros(n, dtype=bool)
        S: list[int] = []
        contrib = np.zeros(n)
        minrow = np.full(n, np.inf)
        curmin = np.inf
        if cfg.constructive == "rnd":
            obj = -1
        elif cfg.constructive == "altbws":
            obj = self._cons_obj
            self._cons_obj ^= 1
        else:
            obj = int(rng.integers(2))
        capS = costS = 0.0

        def done() -> bool:
            if self.fixed:
                return len(S) >= self.m
            return len(S) >= 2 and capS >= self.B - 1e-9

        while not done():
            cl = np.flatnonzero(~inS)
            if not self.fixed and self.cost is not None and S:
                ok = (costS + self.cost[cl]) <= self.K + 1e-9
                if ok.any():
                    cl = cl[ok]
            if cl.size == 0:
                break
            if obj == -1 or not S:
                j = int(rng.choice(cl))
            else:
                g = contrib[cl] if obj == 1 else np.minimum(curmin, minrow[cl])
                gmin, gmax = float(g.min()), float(g.max())
                thr = gmin + cfg.alpha * (gmax - gmin)
                rcl = cl[g >= thr - 1e-12]
                j = int(rng.choice(rcl))
            if S:
                curmin = min(curmin, float(minrow[j]))
            S.append(j)
            inS[j] = True
            contrib += D[j]
            minrow = np.minimum(minrow, D[j])
            if not self.fixed:
                capS += float(self.cap[j])
                if self.cost is not None:
                    costS += float(self.cost[j])
            if cfg.constructive == "altins":
                obj ^= 1
        sol = np.sort(np.array(S, dtype=np.intp))
        if not self.fixed and not self.inst.is_feasible(sol):
            sol = np.asarray(self.inst.repair(sol), dtype=np.intp)
        return sol

    # ------------------------------------------------------------------ busca local
    def _swap_mask(self, S, out_idx, cands, capS, costS):
        if self.fixed:
            return None
        o = int(S[out_idx])
        ok = (capS - self.cap[o] + self.cap[cands]) >= self.B - 1e-9
        if self.cost is not None:
            ok &= (costS - self.cost[o] + self.cost[cands]) <= self.K + 1e-9
        return ok

    def _ls(self, S: np.ndarray, f: tuple, k: int, dom: bool = False):
        """Hill-climb de primeira melhora. k=0: MaxSum; k=1: MaxMin (lexicografico com o outro objetivo). dom: so aceita dominancia."""
        inst, rng = self.inst, self.rng
        S = S.copy()
        self._record(S, f)
        improved = True
        while improved and not self._timeout():
            improved = False
            inS = np.zeros(self.n, dtype=bool)
            inS[S] = True
            cands = np.flatnonzero(~inS)
            capS, costS = self._sums(S)
            h0 = self._hash(S)
            tab = self._tabu_hashes()
            for oi in rng.permutation(len(S)):
                if self._timeout():
                    break
                o = int(S[oi])
                ms, mm = inst.evaluate_swaps_batch(S, f, int(oi), cands)
                ok = np.ones(cands.size, dtype=bool)
                mk = self._swap_mask(S, int(oi), cands, capS, costS)
                if mk is not None:
                    ok &= mk
                if tab.size:
                    hs = np.uint64(h0) ^ self.zob[o] ^ self.zob[cands]
                    ok &= ~np.isin(hs, tab)
                if not ok.any():
                    continue
                if dom:
                    imp = ok & (ms >= f[0] - 1e-9) & (mm >= f[1] - 1e-9) & ((ms > f[0] + 1e-9) | (mm > f[1] + 1e-9))
                    p = ms if k == 0 else mm
                    q = mm if k == 0 else ms
                else:
                    p, q = (ms, mm) if k == 0 else (mm, ms)
                    fp, fq = (f[0], f[1]) if k == 0 else (f[1], f[0])
                    imp = ok & ((p > fp + 1e-9) | ((np.abs(p - fp) <= 1e-9) & (q > fq + 1e-9)))
                if not imp.any():
                    continue
                idx = np.flatnonzero(imp)
                best = idx[np.lexsort((q[idx], p[idx]))[-1]]
                S[oi] = int(cands[best])
                S.sort()
                f = inst.evaluate(S)
                self._record(S, f)
                self._add_tabu(S)
                improved = True
                break                       # a vizinhanca mudou: recomeca a varredura
            if (not improved and k == 0 and not self.fixed and not dom and self.cfg.add_moves and f[1] > 1e-9
                    and (getattr(inst, "max_size", None) is None or len(S) < inst.max_size)):
                # insercao DOMINANTE (so CDP/GDP): sobe o MaxSum sem reduzir o MaxMin; respeita o orcamento. Evita o crescimento sem fim
                # (no CDP o MaxSum e maximo no conjunto inteiro): so entram elementos a distancia >= MaxMin de todos os membros.
                inS2 = np.zeros(self.n, dtype=bool)
                inS2[S] = True
                cands2 = np.flatnonzero(~inS2)
                capS2, costS2 = self._sums(S)
                if self.cost is not None:
                    ok2 = (costS2 + self.cost[cands2]) <= self.K + 1e-9
                    cands2 = cands2[ok2]
                if cands2.size:
                    Dc = self.D[np.ix_(cands2, S)]
                    ms2 = f[0] + Dc.sum(axis=1)
                    mm2 = np.minimum(f[1], Dc.min(axis=1))
                    ok3 = mm2 >= f[1] - 1e-9
                    tab2 = self._tabu_hashes()
                    if tab2.size:
                        ok3 &= ~np.isin(np.uint64(self._hash(S)) ^ self.zob[cands2], tab2)
                    if ok3.any():
                        j = int(cands2[np.flatnonzero(ok3)[np.argmax(ms2[ok3])]])
                        S = np.sort(np.append(S, j)).astype(np.intp)
                        f = inst.evaluate(S)
                        self._record(S, f)
                        self._add_tabu(S)
                        improved = True
            if not improved and k == 1 and not self.fixed and not dom:
                # tentativa de REMOVER (tamanho livre): estritamente melhora o MaxMin e mantem capacidade
                if len(S) >= 3:
                    capS, costS = self._sums(S)
                    for oi in rng.permutation(len(S)):
                        o = int(S[oi])
                        if capS - self.cap[o] < self.B - 1e-9:
                            continue
                        rest = np.delete(S, oi)
                        f2 = inst.evaluate(rest)
                        if f2[1] > f[1] + 1e-9:
                            S, f = rest, f2
                            self._record(S, f)
                            improved = True
                            break
        if not self.fixed and k == 1 and len(S) >= 3:
            S, f = self._prune(S, f)
        return S, f

    def _prune(self, S: np.ndarray, f: tuple):
        """Remove elementos redundantes (capacidade ainda satisfeita): o MaxMin nunca cai; registra o ponto resultante."""
        keep = [int(x) for x in S]
        capS = float(self.cap[S].sum())
        changed = False
        for oi in self.rng.permutation(len(S)):
            o = int(S[oi])
            if len(keep) <= 2:
                break
            if capS - self.cap[o] >= self.B - 1e-9:
                keep.remove(o)
                capS -= float(self.cap[o])
                changed = True
        if changed:
            S2 = np.sort(np.array(keep, dtype=np.intp))
            f2 = self.inst.evaluate(S2)
            self._record(S2, f2)
            return S2, f2
        return S, f

    def _path_relink(self, S1: np.ndarray, S2: np.ndarray) -> None:
        rng, inst = self.rng, self.inst
        cur = set(int(x) for x in S1)
        tgt = set(int(x) for x in S2)
        steps = 0
        while True:
            only1 = list(cur - tgt)
            only2 = list(tgt - cur)
            if not only1 and not only2:
                break
            if only1 and only2:
                cur.remove(int(rng.choice(only1)))
                cur.add(int(rng.choice(only2)))
            elif only2:
                cur.add(int(rng.choice(only2)))
            else:
                if len(cur) <= 2:
                    break
                cur.remove(int(rng.choice(only1)))
            sol = np.sort(np.array(list(cur), dtype=np.intp))
            if not self.fixed and not inst.is_feasible(sol):
                sol = np.asarray(inst.repair(sol), dtype=np.intp)
            if sol.size >= 2:
                self._record(sol, inst.evaluate(sol))
            steps += 1
            if steps > 4 * self.n or self._timeout():
                break

    def _explore(self, S: np.ndarray, f: tuple) -> None:
        ls = self.cfg.local_search
        if ls == "pr":
            S1, _ = self._ls(S, f, 1)
            S2, _ = self._ls(S, f, 0)
            self._path_relink(S1, S2)
        elif ls == "alt":
            self._ls(S, f, self._alt_obj)
            self._alt_obj ^= 1
        else:
            self._ls(S, f, 0, dom=True)

    # ------------------------------------------------------------------ shake (VNS)
    def _shake(self, S: np.ndarray, k: int) -> np.ndarray:
        rng = self.rng
        if self.n - len(S) < 1:
            return S.copy()
        k = max(1, min(int(k), len(S) - 1, self.n - len(S)))
        out = rng.choice(len(S), size=k, replace=False)
        inS = np.zeros(self.n, dtype=bool)
        inS[S] = True
        ins = rng.choice(np.flatnonzero(~inS), size=k, replace=False)
        new = np.sort(np.concatenate([np.delete(S, out), ins]).astype(np.intp))
        if not self.fixed and not self.inst.is_feasible(new):
            new = np.asarray(self.inst.repair(new), dtype=np.intp)
        return new

    # ------------------------------------------------------------------ laco principal
    def run(self) -> BOTrajResult:
        cfg, rng, inst = self.cfg, self.rng, self.inst
        self._t0 = _time.perf_counter()
        S = self._construct()
        f = inst.evaluate(S)
        self._record(S, f)
        noimp = 0
        k = 1
        cycles = 0
        outer = 0
        while outer < cfg.max_iter and not self._timeout():
            outer += 1
            self.it += 1
            before = self.n_ins
            if cfg.method == "tabu":
                self._explore(S, f)
                if self.n_ins == before:
                    noimp += 1
                    act = {h for h, e in self.tabu.items() if e > self.it}
                    cand = [i for i in range(len(self.arch_s)) if self._hash(self.arch_s[i]) not in act]
                    pool = cand if cand else list(range(len(self.arch_s)))
                    i = int(rng.choice(pool))
                    S, f = self.arch_s[i].copy(), self.arch_f[i]
                    self._add_tabu(S)
                else:
                    noimp = 0
                if noimp >= cfg.max_no_improve:
                    S = self._construct()
                    f = inst.evaluate(S)
                    self._record(S, f)
                    self.tabu.clear()
                    noimp = 0
            else:   # VNS
                Sp = self._shake(S, k)
                fp = inst.evaluate(Sp)
                self._explore(Sp, fp)
                if self.n_ins != before:
                    i = int(rng.integers(len(self.arch_s)))
                    S, f = self.arch_s[i].copy(), self.arch_f[i]
                    k = 1
                    cycles = 0
                else:
                    k += 1
                    kmax_eff = max(1, int(round(cfg.kmax_frac * len(S)))) if cfg.kmax_frac else cfg.kmax
                    if k > kmax_eff:
                        k = 1
                        cycles += 1
                        if cycles >= cfg.restart_after:
                            S = self._construct()
                            f = inst.evaluate(S)
                            self._record(S, f)
                            cycles = 0
        pts = np.array(self.arch_f, dtype=float).reshape(-1, 2)
        return BOTrajResult(pareto_front=pts, n_iter_executed=outer, config=cfg)

    # ------------------------------------------------------------------ polimento de um arquivo existente (E12)
    def polish(self, sols, objs):
        """Intensifica um arquivo ja existente (p.ex. o do MOEA/D) ate `time_limit_s`: busca local de MaxMin a partir do melhor membro em
        MaxMin, de MaxSum a partir do melhor em MaxSum, path relinking entre os dois resultados e, no tempo restante, exploracao (busca local
        'pr') a partir dos membros do arquivo em rodizio. Devolve (solucoes, objetivos) do arquivo resultante."""
        self._t0 = _time.perf_counter()
        self.arch_s, self.arch_f = [], []
        for S, f in zip(sols, objs):
            _update_archive(self.arch_s, self.arch_f, np.asarray(S, dtype=np.intp).copy(), (float(f[0]), float(f[1])))
        if not self.arch_s:
            return list(sols), list(objs)
        k1 = int(np.argmax([f[1] for f in self.arch_f]))
        k0 = int(np.argmax([f[0] for f in self.arch_f]))
        S1, _ = self._ls(self.arch_s[k1], self.arch_f[k1], 1)
        S2, _ = self._ls(self.arch_s[k0], self.arch_f[k0], 0)
        self._path_relink(S1, S2)
        order = list(self.rng.permutation(len(self.arch_s)))
        i = 0
        while not self._timeout():
            if i >= len(order):
                order = list(self.rng.permutation(len(self.arch_s)))
                i = 0
            j = order[i]
            i += 1
            if j >= len(self.arch_s):
                continue
            self.it += 1
            self._explore(self.arch_s[j].copy(), self.arch_f[j])
        return list(self.arch_s), list(self.arch_f)
