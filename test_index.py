"""End-to-end test of erg_index on a subset of the PubChem CSV. Run (erg_cpp env): python test_index.py [N_rows]"""
import sys, os, shutil, time, numpy as np, pandas as pd
import erg_v2 as E, erg_index as X

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    CSV, OUT = "comprehensive_mw2000_collection.csv", "_test_idx"
    shutil.rmtree(OUT, ignore_errors=True)
    t = time.time()
    X.build(CSV, OUT, "Charged_Smiles", "ID", shard_rows=100_000, limit=N)
    print(f"build {N} rows: {time.time()-t:.1f}s")
    ix = X.Index(OUT)
    # resume: second build must skip everything
    X.build(CSV, OUT, "Charged_Smiles", "ID", shard_rows=100_000, limit=N, log=lambda m: None)
    assert X.Index(OUT).n == ix.n

    # full expanded matrix for brute force
    raw = np.concatenate([np.fromfile(ix._f(s["name"], "raw.bin"), np.uint8).reshape(-1, X.NCOL) for s in ix.shards])
    fz = E.fuzzy_expand(raw)
    n2 = (fz ** 2).sum(1)
    assert np.allclose(n2, np.concatenate([ix.sqnorm(i) for i in range(len(ix.shards))]), rtol=1e-5)
    df = pd.read_csv(CSV, usecols=["Charged_Smiles", "ID"], nrows=N)

    rng = np.random.default_rng(0)
    qrows = [0, 12345, int(rng.integers(N)), int(rng.integers(N))]
    qsmi = [df.Charged_Smiles[i] for i in qrows]
    for metric in ("cosine", "tanimoto", "dot"):
        t = time.time()
        res = ix.search(qsmi, k=25, metric=metric, chunk=50_000, workers=4)
        for qi, (q, r) in enumerate(zip(qrows, res)):
            qv = fz[np.where((raw == raw[q]).all(1))[0][0]]       # expanded query (first identical row)
            dot = fz @ qv; qn = qv @ qv
            s = dot if metric == "dot" else dot / np.sqrt(n2 * qn + 1e-30) if metric == "cosine" else dot / (n2 + qn - dot + 1e-30)
            if metric != "dot": s = np.clip(s, 0, 1)
            bf = np.sort(s)[::-1][:25]
            assert np.allclose(r["similarity"], bf, atol=2e-4, rtol=1e-4), (metric, qi, r["similarity"][:5], bf[:5])
            if metric != "dot": assert r["similarity"][0] > 0.9999
        print(f"{metric:9s}: exact top-25 == brute force for {len(qrows)} queries ({time.time()-t:.2f}s)")

    # id / smiles lookup vs CSV (failed-parse rows are skipped, so compare through the SMILES text)
    res = ix.search(qsmi[:1], k=10, chunk=50_000)
    ids, smi = ix.fetch(res[0]["global_row"])
    lut = dict(zip(df.ID.astype(str), df.Charged_Smiles))
    assert all(lut[str(i)] == s for i, s in zip(ids, smi))
    print("fetch(): ids/smiles match CSV rows; top hit:", ids[0], "sim", float(res[0]["similarity"][0]))
    shutil.rmtree(OUT, ignore_errors=True)
    print("ALL OK")
