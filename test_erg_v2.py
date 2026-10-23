"""Sanity checks for erg_v2 and comparison with v1 on a ChEMBL sample. Run: python test_erg_v2.py"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from erg_v2 import erg_from_smiles, COLUMNS

# 1) Hand checks
v = dict(zip(COLUMNS, erg_from_smiles("OCCN")))          # ethanolamine: two flip-flop atoms
assert v['count_HA'] == 2 and v['count_HD'] == 2, v
assert v['HA_HD_d3'] >= 2.0                              # O(HA)-N(HD) and O(HD)-N(HA), 3 bonds
assert v['HD_HD_d3'] >= 1.0 and v['HA_HA_d3'] >= 1.0
v = dict(zip(COLUMNS, erg_from_smiles("c1ccccc1O")))     # phenol: AR_HA must now populate
assert v['HA_AR_d1'] >= 1.0 and v['HD_AR_d1'] >= 1.0
v = dict(zip(COLUMNS, erg_from_smiles("C[NH3+].CC(=O)[O-]")))
assert v['count_PLUS'] == 1 and v['count_MINUS'] == 1
v = dict(zip(COLUMNS, erg_from_smiles("[NH3+]CCCC(=O)[O-]")))
assert sum(v[f'PLUS_MINUS_d{d}'] for d in range(1, 16)) > 0   # was always 0 in v1
print("hand checks OK")

# 2) Which columns are live on a ChEMBL sample, v2 vs v1 parquet
pf = pq.ParquetFile("ALL_chembl_ErG.parquet")
t = next(pf.iter_batches(batch_size=2000)).to_pandas()
v2 = np.array([erg_from_smiles(s) for s in t.SMILES])
ok = ~np.isnan(v2[:, 0])
v1 = t[COLUMNS].to_numpy(dtype=np.float64)
dead1 = {c for c, s in zip(COLUMNS, np.nansum(v1, 0)) if s == 0}
dead2 = {c for c, s in zip(COLUMNS, np.nansum(v2, 0)) if s == 0}
print(f"{ok.sum()}/{len(t)} parsed; all-zero columns: v1={len(dead1)}  v2={len(dead2)}")
same = [c for c in COLUMNS if c not in dead1 and c in dead2]
print("live in v1 but dead in v2 (should be none):", same)
# columns v1 got right must be unchanged except where HA/HD semantics changed
unaffected = [i for i, c in enumerate(COLUMNS)
              if not any(k in c for k in ('HD', 'HA_'))
              and 'count_HA' != c and 'AR_HA' not in c and 'PLUS_MINUS' not in c]
diff = np.abs(v1[ok][:, unaffected] - v2[ok][:, unaffected]).max()
print("max |v1-v2| on HY/AR/PLUS/MINUS-only columns:", diff)

# 3) Fuzzy as a linear post-step ------------------------------------------------
from erg_v2 import raw_from_smiles, fuzzy_expand, fuzzy_sqnorm, search, NBINS, FUZZY
raws = np.array([raw_from_smiles(s) for s in t.SMILES])
assert raws.dtype == np.uint8 and raws.shape[1] == 321
print("raw max:", raws.max(), " p99.99:", np.percentile(raws, 99.99), " saturated:", (raws == 255).sum())

def v1_style_fuzzy(r):                      # explicit loop, exactly as v1 accumulated
    out = r.astype(np.float32).copy()
    out[6:] = 0
    for p in range(21):
        for d in range(1, NBINS + 1):
            n = r[6 + p * NBINS + d - 1]
            out[6 + p * NBINS + d - 1] += n
            if d > 1: out[6 + p * NBINS + d - 2] += FUZZY * n
            if d < NBINS: out[6 + p * NBINS + d] += FUZZY * n
    return out
fz = fuzzy_expand(raws)
ref = np.array([v1_style_fuzzy(r) for r in raws[:300]])
assert np.abs(fz[:300] - ref).max() < 1e-4, np.abs(fz[:300] - ref).max()
assert np.abs(fz - v2).max() < 1e-4        # identical to the old single-pass erg_vector output
print("fuzzy_expand == explicit v1 loop == previous erg_vector: OK")

# 4) Query-only smoothing gives the same top-k as brute force on expanded vectors
sq = fuzzy_sqnorm(raws)
for metric in ("dot", "cosine", "tanimoto"):
    for qi in (0, 17, 1234):
        idx, sim = search(raws[qi], raws, k=10, metric=metric, db_sqnorm=sq, chunk=700)
        q = fz[qi]; dot = fz @ q; n2 = (fz ** 2).sum(1); qn = q @ q
        if metric == "dot": s = dot
        elif metric == "cosine": s = dot / (np.sqrt(n2 * qn) + 1e-12)
        else: s = dot / (n2 + qn - dot + 1e-12)
        bf = np.argsort(-s)[:10]
        assert np.allclose(np.sort(s[idx]), np.sort(s[bf]), atol=1e-4), (metric, qi)
        assert np.allclose(np.sort(sim), np.sort(s[idx]), atol=1e-4)
        if metric != "dot": assert 0 <= sim.min() and sim.max() <= 1.0001
        if metric != "dot": assert idx[0] == qi or sim[0] > 0.999   # self hit ~ 1.0
print("search == brute force; cosine/tanimoto in [0,1], self-similarity 1: OK")
print("bytes/mol: raw uint8 =", raws.shape[1], " fuzzy float32 =", fz.shape[1] * 4)
