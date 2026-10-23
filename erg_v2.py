"""
ErG v2 (topological, 2D): 6 count columns + 21 pair types x 15 distance bins = 321 values.

Fixes vs. erg_calc_fragments_topo.py (v1):
  1. HD was never assigned: HA took every N/O/S first and the flip-flop SMARTS
     '[#7,#8][H]' needs explicit Hs. Now an N/O with >=1 H (implicit or explicit) is a
     flip-flop atom and yields TWO sites: one HA and one HD (same atom).
  2. Pair keys were built with an alphabetical test (f1 <= f2), so AR_HA, AR_HD and
     MINUS_PLUS never matched a column and were silently dropped. Pair types are now
     canonicalised by the fixed feature order HA, HD, AR, HY, PLUS, MINUS.
  3. No RDKit fr_* descriptors, no MW/LogP/HBD/HBA columns.

Unchanged from v1: HY definition, AR (5/6/7 fully aromatic rings), PLUS/MINUS = formal
charge, topological (bond-count) distance clamped to [1, 15], fuzzy +-1 bin spillover 0.3.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

try:                      # optional C++/OpenMP kernel (cpp/build.bat, conda env erg_cpp); identical output
    import erg_kernel as _K
except Exception:
    _K = None

FEATURES = ['HA', 'HD', 'AR', 'HY', 'PLUS', 'MINUS']
FIDX = {f: i for i, f in enumerate(FEATURES)}
PAIR_TYPES = [f"{a}_{b}" for i, a in enumerate(FEATURES) for b in FEATURES[i:]]  # 21
NBINS = 15
FUZZY = 0.3
COUNT_COLS = [f"count_{f}" for f in FEATURES]
DIST_COLS = [f"{p}_d{d}" for p in PAIR_TYPES for d in range(1, NBINS + 1)]
COLUMNS = COUNT_COLS + DIST_COLS  # 321, same names/order as v1

# 6x6 lookup: (feature_i, feature_j) -> pair-type index (symmetric)
_PT_LUT = np.zeros((6, 6), dtype=np.int64)
for _k, _p in enumerate(PAIR_TYPES):
    _a, _b = _p.split('_')
    _PT_LUT[FIDX[_a], FIDX[_b]] = _k
    _PT_LUT[FIDX[_b], FIDX[_a]] = _k


_AR, _HY, _PLUS, _MINUS, _HA, _HD = (FIDX[k] for k in ('AR', 'HY', 'PLUS', 'MINUS', 'HA', 'HD'))


_HA_Z = frozenset((7, 8, 9, 16, 17, 35))   # N O F S Cl Br
_BIG = 1e12                                 # sentinel for "same atom"


def identify_sites(mol):
    """Return list of (feature_index, [atom indices]). One atom may appear in two sites (HA+HD)."""
    atoms = list(mol.GetAtoms())
    n = len(atoms)
    Z = [a.GetAtomicNum() for a in atoms]
    sites = []
    # AR: fully aromatic 5/6/7-membered rings
    for r in mol.GetRingInfo().AtomRings():
        if len(r) in (5, 6, 7) and all(atoms[i].GetIsAromatic() for i in r):
            sites.append((_AR, list(r)))
    # HY: components (>=3) of neutral non-aromatic C whose neighbours are all C
    hetero = bytearray(n)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    for i, j in bonds:
        if Z[i] != 6 or Z[j] != 6:
            hetero[i] = hetero[j] = 1
    ok = [Z[i] == 6 and not hetero[i] and not atoms[i].GetIsAromatic() and atoms[i].GetFormalCharge() == 0
          for i in range(n)]
    if sum(ok) >= 3:
        adj = {}
        for i, j in bonds:
            if ok[i] and ok[j]:
                adj.setdefault(i, []).append(j)
                adj.setdefault(j, []).append(i)
        seen = set()
        for s in adj:
            if s in seen:
                continue
            comp, stack = [], [s]
            seen.add(s)
            while stack:
                c = stack.pop()
                comp.append(c)
                for k in adj[c]:
                    if k not in seen:
                        seen.add(k)
                        stack.append(k)
            if len(comp) >= 3:
                sites.append((_HY, comp))
    # atom-centred: PLUS / MINUS / HA (+ HD for flip-flop N/O carrying H)
    for i, a in enumerate(atoms):
        q = a.GetFormalCharge()
        if q > 0:
            sites.append((_PLUS, [i]))
        elif q < 0:
            sites.append((_MINUS, [i]))
        elif Z[i] in _HA_Z:
            sites.append((_HA, [i]))
            if (Z[i] == 7 or Z[i] == 8) and a.GetTotalNumHs() > 0:
                sites.append((_HD, [i]))
    return sites


_TRIU = {}


def _triu(S):
    if S not in _TRIU:
        _TRIU[S] = np.triu_indices(S, 1)
    return _TRIU[S]


def erg_raw(mol):
    """Raw (un-fuzzed) integer vector, uint8[321]: 6 counts + 21x15 distance histogram.
    Values saturate at 255. Returns None for a failed molecule."""
    if mol is None:
        return None
    sites = identify_sites(mol)
    S = len(sites)
    ftype = np.fromiter((f for f, _ in sites), dtype=np.int64, count=S)
    counts = np.bincount(ftype, minlength=6)
    hist = np.zeros(len(PAIR_TYPES) * NBINS, dtype=np.int64)
    if S > 1:
        dmb = np.array(rdmolops.GetDistanceMatrix(mol))
        n = dmb.shape[0]
        np.fill_diagonal(dmb, _BIG)                      # skip a == b
        P = np.zeros((S, n), dtype=bool)
        for k, (_, A) in enumerate(sites):
            P[k, A] = True
        M = np.where(P[:, :, None], dmb[None, :, :], _BIG).min(1)   # site -> atom, min over members
        D = np.where(P[None, :, :], M[:, None, :], _BIG).min(2)     # site -> site
        i, j = _triu(S)
        d = D[i, j]
        keep = d < 1e6          # drops the HA/HD pair of one atom and cross-fragment pairs (RDKit: 1e8)
        d = np.clip(d[keep], 1, NBINS).astype(np.int64)
        pt = _PT_LUT[ftype[i[keep]], ftype[j[keep]]]
        hist = np.bincount(pt * NBINS + d - 1, minlength=hist.size)
    return np.minimum(np.concatenate([counts, hist]), 255).astype(np.uint8)


def raw_from_smiles(smiles):
    try:
        return erg_raw(Chem.MolFromSmiles(smiles))
    except Exception:
        return None


# ---------------------------------------------------------------- fuzzy (linear) ----
# fuzzy[k] = raw[k] + 0.3*raw[k-1] + 0.3*raw[k+1] along the distance axis (no wrap, no
# spill below d=1 / above d=15), applied to the 21x15 block only; counts are untouched.
F = (np.eye(NBINS) + FUZZY * (np.eye(NBINS, k=1) + np.eye(NBINS, k=-1))).astype(np.float32)
F2 = (F @ F).astype(np.float32)   # F is symmetric: <Fq, Fd> = <F2 q, d>


def fuzzy_expand(raw):
    """raw uint8 [N,321] (or [321]) -> float32 fuzzy vectors, identical to v1/erg_vector()."""
    raw = np.atleast_2d(raw).astype(np.float32)
    out = raw.copy()
    out[:, 6:] = (raw[:, 6:].reshape(-1, len(PAIR_TYPES), NBINS) @ F).reshape(len(raw), -1)
    return out


def _smooth(q, M):
    q = np.atleast_2d(q).astype(np.float32)
    out = q.copy()
    out[:, 6:] = (q[:, 6:].reshape(-1, len(PAIR_TYPES), NBINS) @ M).reshape(len(q), -1)
    return out


def fuzzy_sqnorm(raw, chunk=1_000_000):
    """||fuzzy(d)||^2 for each raw row; compute once at index time, store as float32."""
    return np.concatenate([(fuzzy_expand(raw[i:i + chunk]) ** 2).sum(1)
                           for i in range(0, len(raw), chunk)]).astype(np.float32)


def search(query_raw, db_raw, k=100, metric="cosine", db_sqnorm=None, chunk=500_000):
    """Exact top-k over a raw uint8 database (ndarray or np.memmap) for ONE query.
    Only the query is smoothed (F^2 q); the database is never expanded to fuzzy form.
    Returns (indices, similarity) best-first; distance = 1 - similarity.
      'cosine'   : <fq,fd> / (|fq||fd|)               in [0,1]  (default)
      'tanimoto' : <fq,fd> / (|fq|^2+|fd|^2-<fq,fd>)  in [0,1]  (continuous Tanimoto)
      'dot'      : <fq,fd>, unbounded (raw)
    Vectors are non-negative, so both normalised metrics stay in [0,1]; an all-zero
    vector (no pharmacophore features) has similarity 0 to everything."""
    q2 = _smooth(query_raw, F2)[0]                      # F^2 q
    qf = fuzzy_expand(query_raw)[0]
    qn2 = float(qf @ qf)
    if db_sqnorm is None and metric != "dot":
        db_sqnorm = fuzzy_sqnorm(db_raw)
    best_i, best_s = np.empty(0, np.int64), np.empty(0, np.float32)
    for s in range(0, len(db_raw), chunk):
        blk = np.asarray(db_raw[s:s + chunk], dtype=np.float32)
        dot = blk @ q2
        if metric == "dot":
            sim = dot
        else:
            n2 = db_sqnorm[s:s + chunk]
            den = np.sqrt(n2 * qn2) if metric == "cosine" else (n2 + qn2 - dot)
            sim = np.where(den > 0, dot / np.where(den > 0, den, 1), 0.0)
            sim = np.clip(sim, 0.0, 1.0)
        best_i = np.concatenate([best_i, np.arange(s, s + len(blk))])
        best_s = np.concatenate([best_s, sim.astype(np.float32)])
        if len(best_i) > 4 * k:
            keep = np.argpartition(-best_s, k)[:k]
            best_i, best_s = best_i[keep], best_s[keep]
    o = np.argsort(-best_s)[:k]
    return best_i[o], best_s[o]


def erg_vector(mol):
    """Backward-compatible fuzzy float32[321] (v1-style values); None on failure."""
    r = erg_raw(mol)
    return None if r is None else fuzzy_expand(r)[0]


def erg_from_smiles(smiles):
    try:
        return erg_vector(Chem.MolFromSmiles(smiles))
    except Exception:
        return None


def _raw_chunk(smiles_chunk):
    out = np.zeros((len(smiles_chunk), len(COLUMNS)), dtype=np.uint8)
    ok = np.zeros(len(smiles_chunk), dtype=bool)
    for k, smi in enumerate(smiles_chunk):
        v = raw_from_smiles(smi) if isinstance(smi, str) and smi else None
        if v is not None:
            out[k], ok[k] = v, True
    return out, ok


def raw_many(smiles, jobs=1, chunk=5000, progress=True, backend="auto"):
    """uint8 [N,321] raw vectors + ok mask. backend 'auto' uses the C++ kernel when importable
    (jobs = OpenMP threads), otherwise python (jobs>1 -> process pool; order preserved)."""
    from tqdm import tqdm
    if backend == "cpp" and _K is None:
        raise RuntimeError("erg_kernel not importable; build it with cpp/build.bat and run in the erg_cpp env")
    if backend in ("auto", "cpp") and _K is not None:
        step = 200_000
        parts = [_K.raw_batch(smiles[i:i + step], jobs)
                 for i in tqdm(range(0, len(smiles), step), desc="ErG v2 (C++)", disable=not progress)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    chunks = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
    if jobs <= 1:
        res = map(_raw_chunk, chunks)
    else:
        import multiprocessing as mp
        res = mp.Pool(jobs).imap(_raw_chunk, chunks)
    parts = list(tqdm(res, total=len(chunks), desc="ErG v2", disable=not progress))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def main():
    import argparse, os
    import pandas as pd
    from tqdm import tqdm
    ap = argparse.ArgumentParser(description="ErG v2 topological fingerprint from a CSV/parquet with a SMILES column")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None,
                    help="default: <input>_erg2.parquet (fuzzy float32) or <input>_erg2 (--raw)")
    ap.add_argument("--raw", action="store_true",
                    help="store raw uint8 [N,321] as <out>.raw.npy + <out>.sqnorm.npy + <out>.ids.parquet "
                         "(search-ready; smoothing is applied at query time)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="worker processes")
    ap.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    args = ap.parse_args()
    df = pd.read_parquet(args.input) if args.input.endswith('.parquet') else pd.read_csv(args.input)
    raw, ok = raw_many(df['SMILES'].tolist(), jobs=args.jobs, backend=args.backend)
    base = args.output or os.path.splitext(args.input)[0] + "_erg2"
    if args.raw:
        np.save(base + ".raw.npy", raw[ok])
        np.save(base + ".sqnorm.npy", fuzzy_sqnorm(raw[ok]))
        df[ok].reset_index(drop=True).to_parquet(base + ".ids.parquet", index=False)
        path = base + ".raw.npy"
    else:
        out = fuzzy_expand(raw)
        out[~ok] = np.nan
        res = pd.concat([df.reset_index(drop=True), pd.DataFrame(out, columns=COLUMNS)], axis=1)
        path = base if base.endswith(('.parquet', '.csv')) else base + ".parquet"
        res.to_parquet(path, index=False) if path.endswith('.parquet') else res.to_csv(path, index=False)
    print(f"{int((~ok).sum())} failed of {len(df)}; wrote {path}")
    if (raw == 255).any():
        print("warning: some raw counts saturated at 255")


if __name__ == "__main__":
    main()
