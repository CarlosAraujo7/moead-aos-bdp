import csv, statistics as st, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
def work(row):
    from capacitated_problem import CDPInstance
    from sbts_anchor import sbts_anchor
    inst = CDPInstance(ROOT / "instances" / row["filepath"])
    t = time.perf_counter()
    out = sbts_anchor(inst, np.random.default_rng(zlib.crc32(row["name"].encode())), iters=ITERS)
    return row["name"], out[1][1], time.perf_counter() - t
if __name__ == "__main__":
    import anchors_lit as L
    ref = L.all_references()
    rows = [r for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == "CDP" and int(r["n"]) <= 150]
    with ProcessPoolExecutor(3) as ex:
        res = list(ex.map(work, rows))
    for fam in ("GKD-b_", "SOM-a"):
        for n_ in ("n50", "n150"):
            rr = [(n, f, t) for n, f, t in res if n.startswith(fam) and f"_{n_}_" in n]
            if not rr: continue
            print(f"{fam}{n_} N={len(rr)} it={ITERS}: /best {st.mean(f/ref[n]['best'] for n,f,t in rr):.3f} atinge {sum(f>=ref[n]['best']-1e-9 for n,f,t in rr)}/{len(rr)} tempo mediano {st.median(t for n,f,t in rr):.2f}s")
