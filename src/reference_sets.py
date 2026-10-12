"""Frente de referencia FIXA por instancia + indicadores IGD+ / eps+ / hv_norm contra ela.

Motivo (PLANO_PUBLICACAO.md, A3): o pipeline original (experiment.compute_reference_front /
build_ref_point) agrupa as frentes dos metodos COMPARADOS e usa o pior ponto entre eles como
nadir. Logo hv_norm de um metodo muda quando outro metodo e incluido ou retirado (risco de
inversao de ranks) e o nadir e ditado pelo pior metodo. Aqui a referencia e a uniao nao-dominada
de TODOS os runs validos ja produzidos para a instancia, e o ponto de referencia e fixo.

Fontes validas (mesmas restricoes do problema original): catalogo, regime normalizado e as
ablacoes de normalizacao / operador unico / leave-one-out. EXCLUIDO de proposito:
ablation_gdp_relaxed (orcamento relaxado: solucoes podem violar a restricao do GDP original) e
qualquer pasta _stale.

Uso (maximizacao em ambos os objetivos):
    python src/reference_sets.py build            # grava results/reference_fronts/{CDP,GDP}.json
    python src/reference_sets.py sensitivity GDP  # ranks por metrica e correlacao entre metricas
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import hypervolume, pareto_front  # noqa: E402

REF_DIR = ROOT / "results" / "reference_fronts"
MARGIN = 0.01  # igual ao build_ref_point original


def source_dirs(problem: str) -> list[Path]:
    if problem == "MDP":   # so o regime normalizado; results/runs (MDP) e obsoleto e results/capacitated e CDP/GDP
        d = ROOT / "results_norm" / "runs"
        return [d] if d.exists() else []
    r = ROOT / "results"
    cands = [
        r / "capacitated" / problem / "runs",
        r / "capacitated_norm" / problem / "runs",
        r / "ablation_normalized" / problem / "runs",
        r / "ablation_single_operator" / problem / "runs",
        r / "ablation_pool_loo" / problem / "runs",
        r / "phase2_aos" / problem / "runs",
        r / "e4_equal_time" / problem / "runs",
    ]
    return [d for d in cands if d.exists()]


def collect(problem: str) -> dict[str, list[np.ndarray]]:
    """instancia -> lista de frentes (arrays Nx2) de todos os runs validos."""
    out: dict[str, list[np.ndarray]] = {}
    for d in source_dirs(problem):
        for f in d.glob("*.json"):  # glob nao desce em _stale/
            try:
                j = json.load(open(f))
                pf = np.asarray(j["pareto_front"], dtype=float)
                out.setdefault(j["instance"], []).append(pf.reshape(-1, 2))
            except Exception:
                continue
    return out


def build(problem: str) -> dict:
    fronts = collect(problem)
    ref = {}
    for inst, fs in fronts.items():
        allp = np.vstack([f for f in fs if len(f)])
        rf = pareto_front(allp)
        lo = allp.min(axis=0)
        hi = allp.max(axis=0)
        ref_pt = lo - MARGIN * np.maximum(hi - lo, [1.0, 1e-9])
        ref[inst] = {"front": rf.tolist(), "ref_point": ref_pt.tolist(), "n_runs": len(fs)}
    REF_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"problem": problem, "sources": [str(d.relative_to(ROOT)) for d in source_dirs(problem)],
               "n_instances": len(ref), "instances": ref}
    (REF_DIR / f"{problem}.json").write_text(json.dumps(payload))
    return payload


def load(problem: str) -> dict:
    p = REF_DIR / f"{problem}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} nao existe; rode: python src/reference_sets.py build")
    return json.load(open(p))["instances"]


def igd_plus(approx: np.ndarray, ref_front: np.ndarray, ref_point: np.ndarray) -> float:
    """IGD+ (Ishibuchi et al. 2015) para MAXIMIZACAO, objetivos normalizados por
    [ref_point, ideal da frente de referencia]. Menor e melhor; 0 = cobre a referencia."""
    ideal = ref_front.max(axis=0)
    span = np.maximum(ideal - ref_point, 1e-12)
    Z = (ref_front - ref_point) / span
    A = (approx - ref_point) / span
    if len(A) == 0:
        return float("inf")
    d = np.maximum(Z[:, None, :] - A[None, :, :], 0.0)       # so conta onde A e pior que Z
    return float(np.sqrt((d ** 2).sum(axis=2)).min(axis=1).mean())


def eps_plus(approx: np.ndarray, ref_front: np.ndarray, ref_point: np.ndarray) -> float:
    """Epsilon aditivo (maximizacao), normalizado como no IGD+. Menor e melhor."""
    ideal = ref_front.max(axis=0)
    span = np.maximum(ideal - ref_point, 1e-12)
    Z = (ref_front - ref_point) / span
    A = (approx - ref_point) / span
    if len(A) == 0:
        return float("inf")
    return float(np.maximum((Z[:, None, :] - A[None, :, :]).max(axis=2).min(axis=1), 0.0).max())


def indicators(approx, instance_ref: dict) -> dict:
    """hv_norm / igd_plus / eps_plus de uma frente contra a referencia FIXA da instancia."""
    approx = np.asarray(approx, dtype=float).reshape(-1, 2)
    rf = np.asarray(instance_ref["front"], dtype=float)
    rp = np.asarray(instance_ref["ref_point"], dtype=float)
    hv_ref = hypervolume(rf, rp)
    return {
        "hv_norm": hypervolume(approx, rp) / hv_ref if hv_ref > 0 else 0.0,
        "igd_plus": igd_plus(approx, rf, rp),
        "eps_plus": eps_plus(approx, rf, rp),
    }


def _sensitivity(problem: str, max_n: int = 150) -> None:
    """Compara ranks do catalogo historico (12 modos) por 4 criterios:
    (a) hv_norm com referencia AGRUPADA dos 12 (o que o pipeline original faz),
    (b) hv_norm com referencia FIXA, (c) IGD+ fixa, (d) eps+ fixa."""
    from scipy.stats import kendalltau, rankdata

    from experiment import ALL_MODES, build_ref_point, compute_indicators, compute_reference_front

    ref = load(problem)
    base = ROOT / "results" / "capacitated" / problem / "runs"
    rows = {"pooled_hv": [], "fixed_hv": [], "igd_plus": [], "eps_plus": []}
    n_used = 0
    for inst, r in ref.items():
        fs = {}
        for m in ALL_MODES:
            p = base / f"{inst}_{m}_rep00.json"
            if not p.exists():
                break
            fs[m] = np.asarray(json.load(open(p))["pareto_front"], dtype=float)
        else:
            if int(inst.split("_n")[1].split("_")[0]) > max_n:
                continue
            n_used += 1
            allf = list(fs.values())
            rf0, rp0 = compute_reference_front(allf), build_ref_point(allf)
            pooled = [compute_indicators(fs[m], rf0, rp0)["hv_norm"] for m in ALL_MODES]
            ind = [indicators(fs[m], r) for m in ALL_MODES]
            rows["pooled_hv"].append(rankdata([-x for x in pooled]))
            rows["fixed_hv"].append(rankdata([-i["hv_norm"] for i in ind]))
            rows["igd_plus"].append(rankdata([i["igd_plus"] for i in ind]))
            rows["eps_plus"].append(rankdata([i["eps_plus"] for i in ind]))
    avg = {k: np.mean(v, axis=0) for k, v in rows.items()}
    print(f"{problem} (n<={max_n}) N={n_used} instancias, 12 modos do catalogo historico\n")
    print(f"{'modo':16s} " + " ".join(f"{k:>10s}" for k in avg))
    for j in np.argsort(avg["pooled_hv"]):
        print(f"{ALL_MODES[j]:16s} " + " ".join(f"{avg[k][j]:10.2f}" for k in avg))
    print("\nTau de Kendall entre rankings medios (1 = mesma ordem):")
    ks = list(avg)
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            print(f"  {ks[a]:10s} x {ks[b]:10s}  tau = {kendalltau(avg[ks[a]], avg[ks[b]])[0]:.3f}")
    flips = {k: float(np.mean([np.sum(np.sign(np.subtract.outer(r1, r1)) != np.sign(np.subtract.outer(r2, r2))) / 2
                               for r1, r2 in zip(rows["pooled_hv"], rows[k])])) for k in ("fixed_hv", "igd_plus", "eps_plus")}
    print("\nPares de metodos (de 66) cuja ordem muda por instancia, vs hv_norm agrupado, em media:")
    for k, v in flips.items():
        print(f"  {k:10s} {v:5.1f} / 66")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        for p in ("CDP", "GDP", "MDP"):
            if p == "MDP" and not source_dirs("MDP"):
                continue
            pl = build(p)
            print(p, pl["n_instances"], "instancias; fontes:", pl["sources"])
    elif len(sys.argv) >= 3 and sys.argv[1] == "sensitivity":
        _sensitivity(sys.argv[2])
    else:
        print(__doc__)
