"""Avalia `sbts_anchor` isolado nas instancias CDP n=500 (GKD-c, MDG-b; b02+b03) contra a literatura (anchors_lit.py).

    python test_sbts_anchor.py ITERS [RESTARTS] [WORKERS]
Saida: razoes medias por familia + results/analysis/sbts_anchor_cdp_n500_it<ITERS>_r<RESTARTS>.json
"""
import csv, json, statistics as st, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
RESTARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def work(row):
    from capacitated_problem import CDPInstance
    from sbts_anchor import sbts_anchor
    import zlib
    inst = CDPInstance(ROOT / "instances" / row["filepath"])
    t = time.perf_counter()
    out = sbts_anchor(inst, np.random.default_rng(zlib.crc32(row["name"].encode())), iters=ITERS, restarts=RESTARTS)
    return row["name"], (out[1][1] if out else None), time.perf_counter() - t


if __name__ == "__main__":
    import anchors_lit as L
    ref = L.all_references()
    rows = [r for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv"))
            if r["problem"] == "CDP" and int(r["n"]) == 500]
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(work, rows))
    out = {}
    for fam in ("GKD-c", "MDG-b"):
        rr = [(n, f, t) for n, f, t in res if n.startswith(fam) and f is not None]
        g = lambda key: st.mean(f / ref[n][key] for n, f, t in rr)
        print(f"{fam} N={len(rr)} it={ITERS}x{RESTARTS}: /best {g('best'):.3f}  /sh10 {g('sbts_short_avg'):.3f}  /lg300 {g('sbts_avg'):.3f}  /sota {g('sota_avg'):.3f}  "
              f"atinge best: {sum(f >= ref[n]['best'] - 1e-9 for n, f, t in rr)}/{len(rr)}  tempo mediano {st.median(t for n, f, t in rr):.1f}s")
        out[fam] = dict(ratio_best=g('best'), ratio_sh10=g('sbts_short_avg'), ratio_lg300=g('sbts_avg'), ratio_sota=g('sota_avg'),
                        values={n: f for n, f, t in rr})
    Path(ROOT / f"results/analysis/sbts_anchor_cdp_n500_it{ITERS}_r{RESTARTS}.json").write_text(json.dumps(out, indent=1))
